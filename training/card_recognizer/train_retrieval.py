#!/usr/bin/env python3
"""Fine-tune a local pretrained card encoder with ArcFace and isolated retrieval eval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from training.card_recognizer.canonical import prepare_encoder_input, preprocessing_contract, preprocessing_contract_hash
from training.card_recognizer.foundation_retrieval import MODEL_ID, MODEL_REVISION, load_dinov2, sha256_file
from training.card_recognizer.retrieval_training import (
    ArcFace, CardEmbedding, TrainingImages, build_training_plan, identity_key, load_card_rgb,
    prepare_fixed_position_embeddings,
)


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path: Path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def embed(model, rows: list[dict], root: Path, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            tensors = []
            for row in rows[start:start + batch_size]:
                path = (root / row['image']).resolve(strict=True)
                if not path.is_relative_to(root.resolve()) or sha256_file(path) != row['sha256']:
                    raise ValueError(f'Evaluation input path/hash mismatch: {row["image"]}')
                tensors.append(prepare_encoder_input(load_card_rgb(path)))
            features = model(torch.from_numpy(np.stack(tensors)).to(device)).cpu().numpy()
            if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < .99):
                raise ValueError('Invalid encoder vectors')
            outputs.append(features)
            if len(rows) > 5000 and (start // batch_size) % 100 == 0:
                print(json.dumps({'stage': 'full_gallery_embedding', 'done': start + len(tensors), 'total': len(rows)}), flush=True)
    return np.concatenate(outputs).astype(np.float32)


def retrieval_metrics(query_vectors, gallery_vectors, queries, gallery):
    ids = [identity_key(r['identity']) for r in gallery]
    records = []
    for vector, query in zip(query_vectors, queries):
        scores = gallery_vectors @ vector
        distinct = {}
        for key, score in zip(ids, scores):
            distinct[key] = max(distinct.get(key, -math.inf), float(score))
        ranked = sorted(distinct, key=lambda k: (-distinct[k], k))
        expected = identity_key(query['identity'])
        if expected not in distinct:
            raise ValueError(f'Evaluation truth is not enrolled: {expected}')
        rank = ranked.index(expected) + 1
        wrong = next(k for k in ranked if k != expected)
        records.append({'image': query['image'], 'source_group': query['source_group'],
                        'expected_identity': expected, 'rank': rank,
                        'predicted_identity': ranked[0], 'true_similarity': distinct[expected],
                        'best_other_identity': wrong, 'best_other_similarity': distinct[wrong]})
    count = len(records)
    return {'query_count': count, 'gallery_references': len(gallery), 'gallery_identities': len(set(ids)),
            'recall_at': {str(k): sum(r['rank'] <= k for r in records) / count for k in (1, 5, 20)},
            'mean_true_minus_best_other': float(np.mean([r['true_similarity'] - r['best_other_similarity'] for r in records])),
            'records': records, 'unknown_rejection_calibrated': False,
            'scope': 'Rectified AI-reviewed video crops; repeated views and few videos; not end-to-end camera accuracy'}


def evaluate(model, gallery, queries, root, device, batch_size):
    gallery_vectors = embed(model, gallery, root, device, batch_size)
    query_vectors = embed(model, queries, root, device, batch_size)
    return retrieval_metrics(query_vectors, gallery_vectors, queries, gallery)


def train(config: dict, output: Path, resume: bool = False):
    seed = int(config['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    device = config['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested Apple Metal GPU unavailable; refusing silent CPU fallback')
    if device not in {'mps', 'cpu', 'cuda'}:
        raise ValueError('Explicit mps/cpu/cuda device required')
    root = (ROOT / config['dataset_root']).resolve(strict=True)
    manifest_path = root / 'recognition/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    plan = build_training_plan(manifest, config['max_training_classes'], seed,
                               tuple(config.get('excluded_training_identities', [])))
    if config['identities_per_batch'] > plan['training_class_count']:
        raise ValueError('Batch requires more identities than the selected training pool')
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if not resume:
        write_json(output / 'training_plan.json', plan)
        write_json(output / 'resolved_config.json', config)
    runtime = (ROOT / config['transformers_runtime']).resolve(strict=True)
    sys.path.insert(0, str(runtime))
    import transformers
    model_dir = (ROOT / config['pretrained_model_path']).resolve(strict=True)
    weights_hash = sha256_file(model_dir / 'model.safetensors')
    if weights_hash != config['pretrained_weights_sha256']:
        raise ValueError('Pretrained weights differ from the pinned experiment initialization')
    backbone = load_dinov2(model_dir)
    positional_contract = prepare_fixed_position_embeddings(backbone)
    encoder = CardEmbedding(backbone, config['embedding_dimension'])
    encoder.configure_backbone(config['trainable_last_blocks'])
    arcface = ArcFace(config['embedding_dimension'], len(plan['classes']), config['margin_radians'], config['logit_scale'])
    encoder.to(device); arcface.to(device)
    backbone_params = [p for p in encoder.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': encoder.projection.parameters(), 'lr': config['projection_lr']},
        {'params': arcface.parameters(), 'lr': config['proxy_lr']},
    ], weight_decay=config['weight_decay'])
    epochs, steps = config['epochs'], config['steps_per_epoch']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps, eta_min=1e-6)
    contract = {'schema_version': 1, 'backbone': MODEL_ID, 'revision': MODEL_REVISION,
                'initial_weights_sha256': weights_hash, 'embedding_dimension': config['embedding_dimension'],
                'pretrained_config_sha256': sha256_file(model_dir / 'config.json'),
                'fixed_positional_grid': positional_contract,
                'pooling': 'CLS -> learned linear projection -> L2', 'training_objective': 'ArcFace',
                'deployment': 'Discard ArcFace proxies; encode both gallery and query with this checkpoint',
                'preprocessing': preprocessing_contract(), 'preprocessing_sha256': preprocessing_contract_hash(),
                'dataset_manifest_sha256': sha256_file(manifest_path), 'training_plan_sha256': plan_hash,
                'device': device, 'torch_version': str(torch.__version__), 'transformers_version': transformers.__version__,
                'validation_scope': 'Fixed sampled training-reference distractors plus all held-out OP-base/alias references; not the full catalog',
                'full_catalog_test': 'Once, after best checkpoint selection; no test-driven tuning',
                'class_count': len(plan['classes']), 'training_real_queries': plan['training_real_query_count']}
    contract['training_source_sha256'] = {name: sha256_file(Path(__file__).parent / name) for name in
                                          ('train_retrieval.py', 'retrieval_training.py', 'canonical.py', 'foundation_retrieval.py')}
    write_json(output / 'encoder_contract.json', contract)
    start_epoch, global_step, best_score = 0, 0, (-1., -1., -math.inf)
    if resume:
        checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['contract']['training_plan_sha256'] != plan_hash or checkpoint['config'] != config:
            raise ValueError('Resume data/config changed')
        encoder.load_state_dict(checkpoint['encoder']); arcface.load_state_dict(checkpoint['arcface'])
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch, global_step, best_score = checkpoint['epoch'], checkpoint['global_step'], tuple(checkpoint['best_score'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
    data = TrainingImages(root, plan['classes'], seed)
    started = time.monotonic()
    parameter_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(json.dumps({'stage': 'training_started', 'device': device, 'trainable_encoder_parameters': parameter_count,
                      'classes': len(plan['classes']), 'real_queries': plan['training_real_query_count'],
                      'epochs': epochs, 'steps_per_epoch': steps, 'batch_images': config['identities_per_batch'] * 2}), flush=True)
    for epoch in range(start_epoch, epochs):
        encoder.configure_backbone(0 if epoch < config['head_warmup_epochs'] else config['trainable_last_blocks'])
        encoder.train(); arcface.train()
        epoch_losses = []
        for step in range(steps):
            x, labels = data.batch(global_step, config['identities_per_batch'])
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            # A large linear weight is a reliable update probe. LayerNorm scales
            # can round unchanged in float32 despite valid, tiny gradients.
            probe = max(backbone_params, key=lambda p: p.numel()) if epoch >= config['head_warmup_epochs'] and step == 0 else None
            before = probe.detach().clone() if probe is not None else None
            features = encoder(x)
            logits = arcface(features, labels)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                                 max_norm=5., error_if_nonfinite=True, foreach=False)
            optimizer.step(); scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if step == 0 or global_step % config['log_every_steps'] == 0:
                status = {'status': 'training', 'epoch': epoch + 1, 'epochs': epochs, 'global_step': global_step,
                          'total_steps': epochs * steps, 'mean_epoch_loss': float(np.mean(epoch_losses)),
                          'gradient_norm_before_clip': float(norm.detach().cpu()), 'device': device,
                          'elapsed_seconds': time.monotonic() - started,
                          'backbone_trainable': epoch >= config['head_warmup_epochs']}
                if probe is not None:
                    status['backbone_probe_max_weight_change'] = float((probe.detach() - before).abs().max().cpu())
                    if status['backbone_probe_max_weight_change'] == 0:
                        raise RuntimeError('Trainable backbone did not update')
                write_json(output / 'status.json', status); print(json.dumps(status), flush=True)
        write_json(output / 'status.json', {'status': 'validating', 'epoch': epoch + 1, 'global_step': global_step, 'device': device})
        validation = evaluate(encoder, plan['validation_gallery'], plan['validation_queries'], root, device, config['evaluation_batch_size'])
        score = (validation['recall_at']['1'], validation['recall_at']['5'], validation['mean_true_minus_best_other'])
        improved = score > best_score
        if improved:
            best_score = score
        checkpoint = {'schema_version': 1, 'encoder': encoder.state_dict(), 'arcface': arcface.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,
                      'global_step': global_step, 'best_score': best_score, 'contract': contract, 'config': config,
                      'torch_rng_state': torch.get_rng_state()}
        save_checkpoint(output / 'last.pt', checkpoint)
        if improved:
            save_checkpoint(output / 'best.pt', checkpoint)
        validation.update(epoch=epoch + 1, global_step=global_step, training_mean_loss=float(np.mean(epoch_losses)))
        write_json(output / f'validation-epoch-{epoch + 1:02d}.json', validation)
        print(json.dumps({'stage': 'epoch_complete', 'epoch': epoch + 1, 'mean_loss': validation['training_mean_loss'],
                          'validation_recall_at': validation['recall_at'], 'best_checkpoint_updated': improved}), flush=True)
    if not config.get('final_full_catalog_evaluation', True):
        write_json(output / 'status.json', {'status': 'completed_smoke_only', 'epochs': epochs,
                                          'global_step': global_step, 'device': device})
        return
    # Test pixels are first loaded here, after all model selection decisions.
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(best['encoder'])
    write_json(output / 'status.json', {'status': 'final_full_catalog_evaluation', 'selected_epoch': best['epoch'], 'device': device})
    gallery_vectors = embed(encoder, manifest['gallery'], root, device, config['evaluation_batch_size'])
    np.save(output / 'gallery_embeddings.npy', gallery_vectors)
    for split in ('val', 'test'):
        queries = manifest['splits'][split]['queries']
        vectors = embed(encoder, queries, root, device, config['evaluation_batch_size'])
        report = retrieval_metrics(vectors, gallery_vectors, queries, manifest['gallery'])
        report.update(selected_epoch=best['epoch'], checkpoint_sha256=sha256_file(output / 'best.pt'))
        write_json(output / f'full_catalog_{split}.json', report)
    write_json(output / 'gallery_index.json', {'encoder_contract': contract, 'checkpoint_sha256': sha256_file(output / 'best.pt'),
                                             'vectors_sha256': sha256_file(output / 'gallery_embeddings.npy'), 'references': manifest['gallery'],
                                             'thresholds_calibrated': False, 'deployment_parity_verified': False})
    write_json(output / 'status.json', {'status': 'completed', 'epochs': epochs, 'global_step': global_step,
                                      'selected_epoch': best['epoch'], 'device': device, 'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'stage': 'completed', 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs/card_recognizer'):
        parser.error('Output must be below repository runs/card_recognizer')
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error('Output must be new/empty; use --resume for a saved run')
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    if not args.resume:
        snapshot = output / 'source_snapshot'
        snapshot.mkdir()
        for name in ('train_retrieval.py', 'retrieval_training.py', 'canonical.py', 'foundation_retrieval.py'):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
    try:
        train(config, output, args.resume)
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                          'error': str(exc), 'type': type(exc).__name__})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
