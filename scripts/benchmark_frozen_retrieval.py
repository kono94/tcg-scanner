#!/usr/bin/env python3
"""Bounded frozen DINOv2-small retrieval comparison on existing card fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.card_recognizer.foundation_retrieval import (
    MODEL_ID, MODEL_REVISION, contract_metadata, encode_paths, load_dinov2, sha256_file,
)


DEFAULT_MIXED_INDEX = ROOT / 'runs/card_recognizer/cosine-onepiece-pokemon/manifest.json'
DEFAULT_SYNTHETIC_REPORT = ROOT / 'runs/card_recognizer/synthetic_retrieval_report.json'
DEFAULT_QUERY_MANIFEST = ROOT / 'runs/card_detector/synthetic/recognizer_manifest.json'
DEFAULT_SOURCE_MANIFEST = ROOT / 'runs/card_detector/synthetic/source_manifest.json'
DEFAULT_OUTPUT = ROOT / 'runs/card_recognizer/dinov2_baseline'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed-index', type=Path, default=DEFAULT_MIXED_INDEX)
    parser.add_argument('--synthetic-report', type=Path, default=DEFAULT_SYNTHETIC_REPORT)
    parser.add_argument('--query-manifest', type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument('--source-manifest', type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument('--source-crops', type=Path, help='produce unverified top-20 shortlists for an existing detector crop manifest')
    parser.add_argument('--synthetic-root', type=Path, default=ROOT / 'runs/card_detector/synthetic')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--queries-per-split', type=int, default=64)
    parser.add_argument('--all-queries', action='store_true', help='evaluate all 1321 renderer crops instead of the default stratified subset')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def load_inputs(args: argparse.Namespace):
    mixed = json.loads(args.mixed_index.read_text())
    baseline = json.loads(args.synthetic_report.read_text())
    query_manifest = json.loads(args.query_manifest.read_text())
    references = mixed.get('references', [])
    if len(references) != 1431:
        raise ValueError(f'expected 1431 mixed-index references; got {len(references)}')
    # Every gallery input is the exact source canonical_path used by the existing mixed index.
    for ref in references:
        path = Path(ref['canonical_path'])
        if not path.is_file():
            raise FileNotFoundError(f'missing gallery canonical reference: {path}')
        actual_hash = sha256_file(path)
        expected_hash = ref.get('canonical_sha256')
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f'canonical reference hash mismatch: {path}')
        ref['verified_canonical_sha256'] = actual_hash
    if baseline.get('query_manifest_sha256') != sha256_file(args.query_manifest):
        raise ValueError('synthetic query manifest hash differs from the comparison report')
    labels = {row['query']: row for row in baseline.get('records', [])}
    queries = []
    for item in query_manifest.get('queries', []):
        report_row = labels.get(item['image'])
        if report_row is None:
            raise ValueError(f'query absent from prior baseline report: {item["image"]}')
        queries.append({
            'path': args.synthetic_root / item['image'],
            'relative_path': item['image'],
            'identity': report_row['expected_identity'],
            'split': item['split'],
            'source_sha256': item.get('source_sha256'),
            'baseline_rank': report_row.get('rank'),
            'baseline_best_identity': report_row.get('best_identity'),
            'baseline_best_similarity': report_row.get('best_similarity'),
        })
    if len(queries) != 1321 or len(queries) != len(baseline.get('records', [])):
        raise ValueError(f'expected exact shared 1321-query manifest/report, found {len(queries)} queries and {len(baseline.get("records", []))} report rows')
    for query in queries:
        if not query['path'].is_file():
            raise FileNotFoundError(f'missing synthetic query crop: {query["path"]}')
    return mixed, baseline, query_manifest, references, queries


def deterministic_split_sample(queries: list[dict], per_split: int) -> list[dict]:
    if per_split < 1:
        raise ValueError('queries-per-split must be positive')
    selected = []
    for split in ('train', 'val'):
        group = [query for query in queries if query['split'] == split]
        if len(group) < per_split:
            raise ValueError(f'not enough {split} queries: {len(group)} < {per_split}')
        indices = np.linspace(0, len(group) - 1, per_split, dtype=np.int64)
        selected.extend(group[int(index)] for index in indices)
    return selected


def rank_identities(query_vectors: np.ndarray, gallery_vectors: np.ndarray,
                    identities: list[str], query_identities: list[str],
                    ks: tuple[int, ...] = (1, 5, 20),
                    alternate_query_vectors: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    if query_vectors.ndim != 2 or gallery_vectors.ndim != 2 or query_vectors.shape[1] != gallery_vectors.shape[1]:
        raise ValueError('query and gallery embeddings must be compatible matrices')
    if len(identities) != len(gallery_vectors) or len(query_identities) != len(query_vectors):
        raise ValueError('identity lengths must match embedding rows')
    gallery_identity_order = sorted(set(identities))
    identity_rows = {identity: [i for i, value in enumerate(identities) if value == identity]
                     for identity in gallery_identity_order}
    scores = query_vectors @ gallery_vectors.T
    if alternate_query_vectors is not None:
        if alternate_query_vectors.shape != query_vectors.shape:
            raise ValueError('alternate query embeddings must match the primary query matrix')
        scores = np.maximum(scores, alternate_query_vectors @ gallery_vectors.T)
    records = []
    for row, expected in enumerate(query_identities):
        identity_scores = [(identity, float(np.max(scores[row, indices])))
                           for identity, indices in identity_rows.items()]
        identity_scores.sort(key=lambda pair: (-pair[1], pair[0]))
        rank = next((i + 1 for i, (identity, _) in enumerate(identity_scores) if identity == expected), None)
        records.append({'rank': rank, 'best_identity': identity_scores[0][0],
                        'best_similarity': identity_scores[0][1],
                        'top_identities': [{'identity': identity, 'similarity': score}
                                           for identity, score in identity_scores[:max(ks)]]})
    recall = {str(k): sum(record['rank'] is not None and record['rank'] <= k for record in records) / len(records)
              for k in ks} if records else {str(k): None for k in ks}
    return recall, records


def recall_by_split(records: list[dict], queries: list[dict], ks: tuple[int, ...] = (1, 5, 20)) -> dict:
    result = {}
    for split in ('train', 'val'):
        group = [record for record, query in zip(records, queries) if query['split'] == split]
        result[split] = {'count': len(group), **{
            str(k): sum(item['rank'] is not None and item['rank'] <= k for item in group) / len(group)
            for k in ks
        }}
    return result


def snapshot_model(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('huggingface_hub is unavailable in the isolated runtime') from exc
    cache = args.output / 'hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache / 'hub')
    return Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache / 'hub'),
        allow_patterns=['config.json', 'model.safetensors'], local_files_only=args.local_files_only,
        token=False,
    ))


def source_crop_shortlists(args: argparse.Namespace) -> int:
    report_path = args.output / 'report.json'
    embeddings_path = args.output / 'frozen_embeddings.npz'
    if not report_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError('run the synthetic benchmark first to create report.json and subset_embeddings.npz')
    output_path = args.output / 'source_crop_shortlists.json'
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite existing shortlist artifact: {output_path}')
    source_manifest = json.loads(args.source_crops.read_text())
    query_items = source_manifest.get('queries', [])[:16]
    if len(query_items) != 16:
        raise ValueError(f'expected at least 16 detector crops; found {len(query_items)}')
    crop_root = args.source_crops.parent
    paths = [crop_root / item['image'] for item in query_items]
    for item, path in zip(query_items, paths):
        if not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError(f'detector crop missing or hash mismatch: {path}')
    with np.load(embeddings_path) as archive:
        gallery_features = archive['gallery'].copy()
    mixed = json.loads(args.mixed_index.read_text())
    gallery_ids = [ref['identity'] for ref in mixed['references']]
    if len(gallery_ids) != len(gallery_features):
        raise ValueError('saved DINO gallery vectors do not align with mixed-index references')
    model_dir = snapshot_model(args)
    model = load_dinov2(model_dir)
    paths = [str(path) for path in paths]
    raw, raw_seconds = encode_paths(model, paths, args.batch_size, args.threads, unrectified_crop=True)
    turned, turned_seconds = encode_paths(model, paths, args.batch_size, args.threads,
                                          rotate_180=True, unrectified_crop=True)
    unknown = ['__unlabeled_detector_crop__'] * len(paths)
    _, rank0 = rank_identities(raw, gallery_features, gallery_ids, unknown)
    _, rank180 = rank_identities(turned, gallery_features, gallery_ids, unknown)
    _, rankmax = rank_identities(raw, gallery_features, gallery_ids, unknown,
                                 alternate_query_vectors=turned)
    records = []
    for item, a, b, c in zip(query_items, rank0, rank180, rankmax):
        records.append({
            'image': item['image'], 'crop_sha256': item['sha256'],
            'source_group': item.get('source_group'), 'source_frame_image': item.get('source_frame_image'),
            'source_frame_sha256': item.get('source_frame_sha256'),
            'source_frame': item.get('source_frame'), 'source_time_seconds': item.get('time_seconds'),
            'detector_model': item.get('detector_model'), 'detector_revision': item.get('detector_revision'),
            'detector_prompt': item.get('detector_prompt'), 'detection_score': item.get('detection_score'),
            'bbox_aabb_source_pixels_clipped': item.get('bbox_aabb_source_pixels_clipped'),
            'crop_width': item.get('crop_width'), 'crop_height': item.get('crop_height'),
            'aabb_clipped_to_source': item.get('aabb_clipped_to_source'),
            'crop_policy': item.get('crop_policy'),
            'teacher_correspondence_proposal_only': item.get('teacher_correspondence_proposal_only'),
            'physical_card_status': item.get('physical_card_status', 'unknown'),
            'identity_status': 'unresolved', 'ground_truth_verified': False,
            'training_promotion': 'not_promoted',
            'top20_0_degree': a['top_identities'], 'top20_180_degree': b['top_identities'],
            'top20_max_over_0_and_180': c['top_identities'],
        })
    model_contract = json.loads(report_path.read_text())['model_contract']
    output = {
        'schema_version': 1,
        'artifact_type': 'unverified_frozen_foundation_retrieval_shortlists',
        'source_crop_manifest': str(args.source_crops),
        'source_crop_manifest_sha256': sha256_file(args.source_crops),
        'detection_count_in_source_manifest': source_manifest.get('all_detection_count'),
        'queries_shortlisted': len(records),
        'model_contract': model_contract,
        'gallery_manifest_sha256': sha256_file(args.mixed_index),
        'preprocessing_note': 'Detector AABB crops are canonicalized from full crop bounds using shared canonicalize_card, then normalized with prepare_encoder_input; no perspective corners are inferred.',
        'method': {'card_training': 'none', 'device': 'CPU', 'top_k': 20,
                   'candidate_ranking': 'max cosine per distinct namespaced identity',
                   'orientation_views': '0-degree, 180-degree, and separate max-score union',
                   'identity_status': 'unresolved', 'ground_truth_verified': False,
                   'training_promotion': 'not_promoted'},
        'timing_seconds': {'source_crops_0_degree': raw_seconds, 'source_crops_180_degree': turned_seconds},
        'records': records,
    }
    output_path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'shortlists': str(output_path), 'queries': len(records), 'time_seconds': output['timing_seconds']}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.source_crops:
        return source_crop_shortlists(args)
    for existing in (args.output / 'report.json', args.output / 'frozen_embeddings.npz'):
        if existing.exists():
            raise FileExistsError(f'refusing to overwrite existing benchmark artifact: {existing}')
    args.output.mkdir(parents=True, exist_ok=True)
    mixed, baseline, query_manifest, references, all_queries = load_inputs(args)
    queries = all_queries if args.all_queries else deterministic_split_sample(all_queries, args.queries_per_split)

    before_bytes = sum(Path(ref['canonical_path']).stat().st_size for ref in references)
    before_bytes += sum(query['path'].stat().st_size for query in queries)
    subset_label = 'all' if args.all_queries else f'{args.queries_per_split}/split'
    print(f'gallery={len(references)}; selected_queries={len(queries)} ({subset_label}); input_bytes={before_bytes}')
    model_dir = snapshot_model(args)
    weight_path = model_dir / 'model.safetensors'
    config_path = model_dir / 'config.json'
    if not weight_path.is_file() or not config_path.is_file():
        raise FileNotFoundError('pinned DINOv2 snapshot lacks model.safetensors or config.json')
    weights_hash = sha256_file(weight_path)
    config_hash = sha256_file(config_path)

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError('PyTorch/Transformers runtime unavailable') from exc
    model = load_dinov2(model_dir)
    gallery_paths = [ref['canonical_path'] for ref in references]
    gallery_ids = [ref['identity'] for ref in references]
    gallery_features, gallery_seconds = encode_paths(model, gallery_paths, args.batch_size, args.threads)
    query_features, query_seconds = encode_paths(model, [q['path'] for q in queries], args.batch_size, args.threads)
    orientation_features, orientation_seconds = encode_paths(
        model, [q['path'] for q in queries], args.batch_size, args.threads, rotate_180=True)
    retrieval_started = time.perf_counter()
    # Query and gallery are independently encoded; exact cosine search is CPU matrix multiply.
    # Encode helper returns L2 vectors; no cross-query self-matching occurs because query crops are separate renderer images.
    recall, rankings = rank_identities(query_features, gallery_features, gallery_ids,
                                       [q['identity'] for q in queries])
    orientation_recall, orientation_rankings = rank_identities(
        query_features, gallery_features, gallery_ids, [q['identity'] for q in queries],
        alternate_query_vectors=orientation_features)
    retrieval_seconds = time.perf_counter() - retrieval_started
    for query, ranking in zip(queries, rankings):
        ranking.update({k: query[k] for k in ('relative_path', 'identity', 'split', 'source_sha256',
                                               'baseline_rank', 'baseline_best_identity', 'baseline_best_similarity')})
        ranking['query_sha256'] = sha256_file(query['path'])
    for query, ranking in zip(queries, orientation_rankings):
        ranking.update({k: query[k] for k in ('relative_path', 'identity', 'split', 'source_sha256')})
        ranking['query_sha256'] = sha256_file(query['path'])

    baseline_rows = {r['query']: r for r in baseline['records']}
    baseline_subset_records = [
        {'rank': baseline_rows[q['relative_path']].get('rank'),
         'best_identity': baseline_rows[q['relative_path']].get('best_identity'),
         'best_similarity': baseline_rows[q['relative_path']].get('best_similarity')}
        for q in queries
    ]
    baseline_recall = {str(k): sum(r['rank'] is not None and r['rank'] <= k for r in baseline_subset_records) / len(queries)
                       for k in (1, 5, 20)}
    baseline_split = {}
    for split in ('train', 'val'):
        group = [r for q, r in zip(queries, baseline_subset_records) if q['split'] == split]
        baseline_split[split] = {'count': len(group), **{
            str(k): sum(r['rank'] is not None and r['rank'] <= k for r in group) / len(group)
            for k in (1, 5, 20)
        }}

    feature_path = args.output / 'frozen_embeddings.npz'
    np.savez_compressed(feature_path, gallery=gallery_features, queries=query_features)
    output = {
        'schema_version': 1,
        'experiment': 'frozen_foundation_retrieval_no_card_training',
        'model_contract': contract_metadata(weights_hash, config_hash, transformers.__version__, torch.__version__),
        'dataset': {
            'gallery_count': len(references), 'distinct_gallery_identities': len(set(gallery_ids)),
            'gallery_manifest': str(args.mixed_index), 'gallery_manifest_sha256': sha256_file(args.mixed_index),
            'gallery_input_aggregate_sha256': hashlib.sha256(''.join(
                f"{ref['identity']}\0{ref.get('canonical_sha256','')}\n" for ref in references).encode()).hexdigest(),
            'gallery_references': [{'identity': ref['identity'], 'canonical_path': ref['canonical_path'],
                                    'canonical_sha256': ref.get('verified_canonical_sha256')}
                                   for ref in references],
            'query_manifest': str(args.query_manifest), 'query_manifest_sha256': sha256_file(args.query_manifest),
            'synthetic_source_manifest': str(args.source_manifest),
            'synthetic_source_manifest_sha256': sha256_file(args.source_manifest),
            'previous_report': str(args.synthetic_report), 'previous_report_sha256': sha256_file(args.synthetic_report),
            'full_renderer_query_count': len(all_queries),
            'query_manifest_hash_in_previous_report': baseline.get('query_manifest_sha256'),
            'queries_derived_from_gallery_source_images': baseline.get('queries_derived_from_gallery_source_images'),
            'selected_query_count': len(queries), 'selection': 'full manifest order' if args.all_queries else f'deterministic evenly-spaced {args.queries_per_split} per split (train/val)',
            'selected_queries': [{'path': q['relative_path'], 'identity': q['identity'], 'split': q['split'],
                                  'renderer_source_sha256': q['source_sha256'],
                                  'query_file_sha256': sha256_file(q['path'])}
                                 for q in queries],
        },
        'method': {'encoder': MODEL_ID, 'revision': MODEL_REVISION, 'card_training': 'none',
                   'device': 'CPU', 'batch_size': args.batch_size, 'threads': args.threads,
                   'similarity': 'exact cosine', 'candidate_ranking': 'max score per distinct namespaced identity',
                   'query_pooling': 'CLS token',
                   'orientation_policy': 'single stored canonical query orientation; same renderer crop list as baseline',
                   'thresholds_calibrated': False,
                   'metrics_scope': 'synthetic renderer crops; not real-camera or independent-image accuracy'},
        'timing_seconds': {'gallery_encoding': gallery_seconds, 'query_encoding': query_seconds,
                           'query_180_encoding': orientation_seconds,
                           'retrieval': retrieval_seconds, 'total_model_and_retrieval': gallery_seconds + query_seconds + orientation_seconds + retrieval_seconds},
        'comparison': {
            'same_query_set_previous_baseline_recall': baseline_recall,
            'same_query_set_previous_baseline_split_recall': baseline_split,
            'dinov2_recall': recall,
            'dinov2_split_recall': recall_by_split(rankings, queries),
            'dinov2_recall_max_over_0_and_180': orientation_recall,
            'dinov2_split_recall_max_over_0_and_180': recall_by_split(orientation_rankings, queries),
            'previous_full_1321_query_report_recall': baseline['recall'],
            'previous_full_1321_query_report_split_recall': baseline['split_recall'],
            'query_set_matches_previous_full_report': bool(args.all_queries),
            'query_count_matches_previous_full_report': len(queries) == len(all_queries),
        },
        'rankings': rankings,
        'artifacts': {'embeddings': str(feature_path), 'embeddings_sha256': sha256_file(feature_path),
                      'weights_path': str(weight_path), 'weights_sha256': weights_hash,
                      'config_sha256': config_hash},
    }
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'report': str(report_path), 'gallery_recall': recall,
                      'split_recall': output['comparison']['dinov2_subset_split_recall'],
                      'same_subset_baseline_recall': baseline_recall,
                      'timing_seconds': output['timing_seconds'], 'weights_sha256': weights_hash}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
