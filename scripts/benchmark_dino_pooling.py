"""Bounded pooled-token ablation for existing frozen DINOv2-small retrieval inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_frozen_retrieval import (  # noqa: E402
    DEFAULT_MIXED_INDEX, DEFAULT_QUERY_MANIFEST, DEFAULT_SOURCE_MANIFEST,
    DEFAULT_SYNTHETIC_REPORT, load_inputs, rank_identities, recall_by_split,
)
from training.card_recognizer.foundation_retrieval import (  # noqa: E402
    MODEL_ID, MODEL_REVISION, prepare_card_path, sha256_file,
)

DEFAULT_MODEL_DIR = ROOT / 'runs/card_recognizer/dinov2_baseline/hf_cache/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056'
DEFAULT_OUTPUT = ROOT / 'runs/card_recognizer/dinov2_pooling'


def pool_hidden_states(last_hidden_state):
    """Return normalized CLS, patch-mean, and concatenated CLS+patch-mean features."""
    import torch
    import torch.nn.functional as F

    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 2 or last_hidden_state.shape[2] < 1:
        raise ValueError('last_hidden_state must be [batch, CLS+patch tokens, feature_dim]')
    hidden = last_hidden_state.float()
    cls_raw = hidden[:, 0, :]
    patch_mean_raw = hidden[:, 1:, :].mean(dim=1)
    if not torch.isfinite(cls_raw).all() or not torch.isfinite(patch_mean_raw).all():
        raise ValueError('pooled DINO features must be finite')
    cls = F.normalize(cls_raw, p=2, dim=1)
    patch = F.normalize(patch_mean_raw, p=2, dim=1)
    combined = F.normalize(torch.cat([cls, patch], dim=1), p=2, dim=1)
    if torch.any(torch.linalg.vector_norm(cls, dim=1) < .999) or torch.any(torch.linalg.vector_norm(patch, dim=1) < .999):
        raise ValueError('zero pooled descriptor')
    return cls, patch, combined


def encode_all(model, paths: list[Path], batch_size: int, threads: int):
    import torch

    if batch_size != 16 or threads != 4:
        raise ValueError('This bounded reproduction fixes batch size=16 and CPU threads=4')
    torch.set_num_threads(threads)
    cls_rows, patch_rows, combined_rows = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = np.stack([prepare_card_path(path) for path in paths[start:start + batch_size]])
            output = model(pixel_values=torch.from_numpy(batch))
            cls, patch, combined = pool_hidden_states(output.last_hidden_state)
            cls_rows.append(cls.cpu().numpy().astype(np.float32, copy=False))
            patch_rows.append(patch.cpu().numpy().astype(np.float32, copy=False))
            combined_rows.append(combined.cpu().numpy().astype(np.float32, copy=False))
    elapsed = time.perf_counter() - started
    return tuple(np.concatenate(rows, axis=0) for rows in (cls_rows, patch_rows, combined_rows)), elapsed


def _aggregate_hash(values: list[str]) -> str:
    return hashlib.sha256((''.join(f'{value}\n' for value in values)).encode()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output must be new or empty: {output}')
    model_dir: Path = args.model_dir.resolve()
    if not model_dir.is_dir() or not (model_dir / 'model.safetensors').is_file() or not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Pinned local DINOv2 model snapshot unavailable: {model_dir}')
    runtime = args.runtime.resolve()
    if not runtime.is_dir():
        raise FileNotFoundError(f'Isolated Transformers runtime unavailable: {runtime}')
    sys.path.insert(0, str(runtime))
    import torch
    import transformers
    from training.card_recognizer.foundation_retrieval import load_dinov2

    torch.set_num_threads(4)
    references_args = Namespace(mixed_index=args.mixed_index, synthetic_report=args.synthetic_report,
                                 query_manifest=args.query_manifest, source_manifest=args.source_manifest,
                                 synthetic_root=args.synthetic_root)
    mixed, baseline, query_manifest, references, queries = load_inputs(references_args)
    if len(references) != 1431 or len(queries) != 1321:
        raise ValueError('Requires exact 1431-reference / 1321-query full benchmark manifests')
    baseline_report_path = args.baseline_report.resolve()
    baseline_report = json.loads(baseline_report_path.read_text())
    if baseline_report['dataset']['gallery_manifest_sha256'] != sha256_file(args.mixed_index):
        raise ValueError('Baseline report gallery manifest differs from current gallery')
    if baseline_report['dataset']['query_manifest_sha256'] != sha256_file(args.query_manifest):
        raise ValueError('Baseline report query manifest differs from current query manifest')
    if not baseline_report['comparison'].get('query_set_matches_previous_full_report') or len(baseline_report['rankings']) != 1321:
        raise ValueError('Reference CLS report is not the exact full 1321-query single-view run')
    if baseline_report['model_contract']['model_revision'] != MODEL_REVISION:
        raise ValueError('Baseline report model revision differs')
    weight_hash = sha256_file(model_dir / 'model.safetensors')
    config_hash = sha256_file(model_dir / 'config.json')
    if weight_hash != baseline_report['model_contract']['weights_sha256'] or config_hash != baseline_report['model_contract']['config_sha256']:
        raise ValueError('Pinned local DINOv2 weights/config hashes differ from completed baseline')

    model = load_dinov2(model_dir)
    gallery_paths = [Path(row['canonical_path']) for row in references]
    query_paths = [row['path'] for row in queries]
    gallery_vectors, gallery_seconds = encode_all(model, gallery_paths, 16, 4)
    query_vectors, query_seconds = encode_all(model, query_paths, 16, 4)
    methods = ('cls_token_l2', 'patch_mean_l2', 'concat_cls_patch_mean_l2')
    identities = [ref['identity'] for ref in references]
    expected = [query['identity'] for query in queries]
    rankings = {}
    recall = {}
    split_recall = {}
    for method_index, method in enumerate(methods):
        r, records = rank_identities(query_vectors[method_index], gallery_vectors[method_index], identities, expected)
        rankings[method] = records
        recall[method] = r
        split_recall[method] = recall_by_split(records, queries)
    previous_cls = baseline_report['comparison']['dinov2_recall']
    if any(abs(recall['cls_token_l2'][str(k)] - previous_cls[str(k)]) > 1e-9 for k in (1, 5, 20)):
        raise ValueError('Single-forward CLS rerun does not exactly match completed CLS benchmark')
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / 'pooling_features.npz'
    np.savez_compressed(feature_path,
        gallery_cls=gallery_vectors[0], gallery_patch_mean=gallery_vectors[1], gallery_concat=gallery_vectors[2],
        queries_cls=query_vectors[0], queries_patch_mean=query_vectors[1], queries_concat=query_vectors[2])
    compact_rankings = [{
        'query': query['relative_path'], 'identity': query['identity'], 'split': query['split'],
        'query_sha256': sha256_file(query['path']),
        'methods': {method: {'rank': rankings[method][idx]['rank'],
                             'best_identity': rankings[method][idx]['best_identity'],
                             'best_similarity': rankings[method][idx]['best_similarity']}
                    for method in methods},
    } for idx, query in enumerate(queries)]
    ranking_path = output / 'query_rankings.json'
    ranking_path.write_text(json.dumps(compact_rankings, indent=2) + '\n')
    model_contract = dict(baseline_report['model_contract'])
    model_contract['pooling_ablation'] = {
        'cls_token_l2': 'last_hidden_state[:,0,:] L2-normalized',
        'patch_mean_l2': 'mean(last_hidden_state[:,1:,:], axis=tokens), then L2-normalized',
        'concat_cls_patch_mean_l2': 'concatenate independently L2-normalized CLS and patch mean, then L2-normalize',
        'single_forward_shared_features': True,
    }
    report = {
        'schema_version': 1, 'experiment': 'frozen_dinov2_small_pooling_ablation_no_card_training',
        'model_contract': model_contract,
        'dataset': {'gallery_count': len(references), 'distinct_gallery_identities': len(set(identities)),
                    'gallery_manifest': str(args.mixed_index.resolve()), 'gallery_manifest_sha256': sha256_file(args.mixed_index),
                    'gallery_reference_identity_order_sha256': _aggregate_hash(identities),
                    'gallery_reference_image_sha256_ordered': [ref['verified_canonical_sha256'] for ref in references],
                    'query_manifest': str(args.query_manifest.resolve()), 'query_manifest_sha256': sha256_file(args.query_manifest),
                    'query_count': len(queries), 'query_identity_order_sha256': _aggregate_hash(expected),
                    'query_source_file_hashes_sha256': _aggregate_hash([sha256_file(q['path']) for q in queries]),
                    'synthetic_source_manifest': baseline_report['dataset']['synthetic_source_manifest'],
                    'synthetic_source_manifest_sha256': baseline_report['dataset']['synthetic_source_manifest_sha256'],
                    'baseline_report_sha256': sha256_file(baseline_report_path),
                    'query_orientation': 'single stored orientation; no 180-degree rerun'},
        'method': {'card_training': 'none', 'device': 'CPU', 'threads': 4, 'batch_size': 16,
                   'input_size': 'shared canonical 224x224 tensor from tcg-full-card-v1',
                   'forward_passes': 'one per image, with CLS and patch mean pooled from same last_hidden_state',
                   'similarity': 'exact cosine', 'identity_aggregation': 'max score across templates per distinct namespaced identity',
                   'thresholds_calibrated': False, 'ground_truth_verified': False,
                   'metrics_scope': 'synthetic renderer crops, not independent real images or camera accuracy'},
        'pooling_methods': methods,
        'comparison': {'recall_at_1_5_20': recall, 'split_recall_at_1_5_20': split_recall,
                       'previous_completed_cls_recall': previous_cls,
                       'same_full_query_set_cls_match': True,
                       'best_patch_mean_vs_cls_recall_at_1': recall['patch_mean_l2']['1'] - recall['cls_token_l2']['1'],
                       'best_concat_vs_cls_recall_at_1': recall['concat_cls_patch_mean_l2']['1'] - recall['cls_token_l2']['1']},
        'timing_seconds': {'gallery_one_forward_encoding': gallery_seconds, 'query_one_forward_encoding': query_seconds},
        'artifacts': {'features': str(feature_path), 'features_sha256': sha256_file(feature_path),
                      'query_rankings': str(ranking_path), 'query_rankings_sha256': sha256_file(ranking_path),
                      'weights_path': str((model_dir / 'model.safetensors').resolve()), 'weights_sha256': weight_hash,
                      'config_sha256': config_hash, 'transformers_version': transformers.__version__,
                      'torch_version': torch.__version__},
        'training_promotion': 'not_promoted',
    }
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixed-index', type=Path, default=DEFAULT_MIXED_INDEX)
    p.add_argument('--synthetic-report', type=Path, default=DEFAULT_SYNTHETIC_REPORT)
    p.add_argument('--query-manifest', type=Path, default=DEFAULT_QUERY_MANIFEST)
    p.add_argument('--source-manifest', type=Path, default=DEFAULT_SOURCE_MANIFEST)
    p.add_argument('--synthetic-root', type=Path, default=ROOT / 'runs/card_detector/synthetic')
    p.add_argument('--baseline-report', type=Path, default=ROOT / 'runs/card_recognizer/dinov2_baseline/report.json')
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument('--runtime', type=Path, default=ROOT / 'runs/foundation_teacher/runtime')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    result = run(args)
    print(json.dumps({'report': str(args.output / 'report.json'), 'recall_at_1_5_20': result['comparison']['recall_at_1_5_20'],
                      'timing_seconds': result['timing_seconds'], 'output_size_bytes': sum(x.stat().st_size for x in args.output.glob('*') if x.is_file())}, indent=2))


if __name__ == '__main__':
    main()
