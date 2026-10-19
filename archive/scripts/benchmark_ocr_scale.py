#!/usr/bin/env python3
"""Bounded Tesseract whole-card OCR scale probe for existing crop queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from training.card_recognizer.ocr_identity import load_catalog, resolve_ocr_catalog, _parse_tsv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, default=Path('runs/foundation_teacher/crop_ocr_01/queries.json'))
    parser.add_argument('--ocr-report', type=Path, default=Path('runs/foundation_teacher/crop_ocr_01/ocr_proposals/proposals.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/card_recognizer/ocr_whole_scale_probe/report.json'))
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        parser.error('count must be in [1,12]')

    queries = json.loads(args.queries.read_text())['queries'][:args.count]
    old = json.loads(args.ocr_report.read_text())
    previous_by_sha = {r.get('crop_sha256'): r for r in old.get('records', [])}
    catalog = load_catalog(['onepiece:en=datasets/card_recognizer/cards'])
    executable = 'tesseract'
    records, failures = [], []
    started = time.monotonic()
    for query in queries:
        image_path = args.queries.parent / query['image']
        if not image_path.is_file() or sha256(image_path) != query['sha256']:
            failures.append({'image': query['image'], 'error': 'missing_or_hash_mismatch'})
            continue
        image = Image.open(image_path).convert('RGB')
        pixels = np.asarray(image)
        h, w = pixels.shape[:2]
        previous = previous_by_sha.get(query['sha256'])
        prior_regions = []
        if previous:
            for candidate in previous.get('ocr', []):
                if candidate.get('orientation_candidate_degrees') == 0:
                    prior_regions = candidate.get('regions', [])
                    break
        attempts = []
        for requested_scale in (2, 3):
            effective_scale = min(float(requested_scale), 2400 / max(w, h))
            out_w, out_h = max(1, round(w * effective_scale)), max(1, round(h * effective_scale))
            resized = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
            import io
            buffer = io.BytesIO()
            resized.save(buffer, format='PNG')
            result = subprocess.run([executable, 'stdin', 'stdout', '-l', 'eng', '--psm', '11', 'tsv'],
                                    input=buffer.getvalue(), capture_output=True, timeout=30, check=False)
            try:
                regions = _parse_tsv(result, 'eng', f'tesseract_psm11_scale_{requested_scale}',
                                     source_scale=effective_scale)
                decision = resolve_ocr_catalog(regions, catalog, min_confidence=80)
                attempts.append({'requested_scale': requested_scale, 'effective_scale': effective_scale,
                                 'resized_width': out_w, 'resized_height': out_h,
                                 'raw_regions': regions, 'raw_text': ' | '.join(r['text'] for r in regions),
                                 'catalog_resolution_confidence_gate_80': decision})
            except Exception as exc:
                attempts.append({'requested_scale': requested_scale, 'effective_scale': effective_scale,
                                 'resized_width': out_w, 'resized_height': out_h,
                                 'error': str(exc), 'stderr': result.stderr.decode(errors='replace')[:1000]})
        records.append({'query': query, 'image_sha256': sha256(image_path), 'source_size': [w, h],
                        'orientation_degrees': 0, 'baseline_psm11_regions': prior_regions,
                        'baseline_catalog_resolution_confidence_gate_80': resolve_ocr_catalog(prior_regions, catalog, 80)
                        if prior_regions else None,
                        'attempts': attempts})
    report = {'artifact_type': 'bounded_whole_card_tesseract_scale_probe', 'schema_version': 1,
              'queries_manifest': str(args.queries), 'queries_manifest_sha256': sha256(args.queries),
              'prior_ocr_report': str(args.ocr_report), 'prior_ocr_report_sha256': sha256(args.ocr_report),
              'backend': 'tesseract --psm 11 TSV', 'orientation_degrees': 0,
              'requested_scales': [2, 3], 'maximum_long_edge_pixels': 2400,
              'confidence_gate': 80, 'catalog_entries': len(catalog), 'records': records,
              'failures': failures, 'elapsed_seconds': round(time.monotonic() - started, 3),
              'ground_truth_verified': False, 'training_promotion': 'not_promoted'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    total_decisions = [r.get('baseline_catalog_resolution_confidence_gate_80') for r in records]
    successes = sum(bool(d and d['status'] == 'ocr_catalog_candidate') for d in total_decisions)
    scale_successes = {str(s): sum(any(a.get('requested_scale') == s and
                                       a.get('catalog_resolution_confidence_gate_80', {}).get('status') == 'ocr_catalog_candidate'
                                       for a in r['attempts']) for r in records) for s in (2, 3)}
    print(json.dumps({'records': len(records), 'failures': len(failures), 'baseline_exact_candidates': successes,
                      'scale_exact_candidates': scale_successes, 'elapsed_seconds': report['elapsed_seconds'],
                      'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
