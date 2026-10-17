#!/usr/bin/env python3
"""Offline OCR/catalog proposals on localized crops; unresolved identities stay unlabeled."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.ocr_identity import load_catalog, resolve_ocr_catalog, tesseract_regions
from training.card_recognizer.foundation_retrieval import sha256_file as file_sha256


def propose(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty')
    if args.limit < 1:
        raise ValueError('limit must be positive')
    catalog = load_catalog(args.catalog)
    if not catalog:
        raise ValueError('No catalog records loaded')
    manifest = json.loads(args.queries.read_text())
    root = args.queries.resolve().parent
    records = []
    for item in manifest['queries'][:args.limit]:
        path = (root / item['image']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Query crop path escapes its manifest directory')
        with Image.open(path) as opened:
            image = np.asarray(opened.convert('RGB'))
        # OBBs have a 180-degree artwork ambiguity; OCR both candidates before visual fallback.
        orientations = [image, np.rot90(image, 2).copy()]
        observations, decisions, errors = [], [], []
        for degrees, candidate in zip([0, 180], orientations):
            try:
                regions = tesseract_regions(candidate, args.ocr_language)
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc))
                regions = []
            decision = resolve_ocr_catalog(regions, catalog, args.min_ocr_confidence)
            observations.append({'orientation_candidate_degrees': degrees, 'regions': regions})
            decisions.append(decision)
        direct = [d for d in decisions if d['status'] == 'ocr_catalog_candidate']
        identities = {d['candidate_identities'][0] for d in direct}
        # Opposite orientations must not silently override valid contradictory code evidence.
        conflict = len(identities) > 1 or any(d['status'] == 'conflicting_code_evidence' for d in decisions)
        if direct and any(d['status'] == 'no_catalog_match' and d['observed_codes'] for d in decisions):
            conflict = True
        visual, corroboration = [], None
        if not conflict and len(identities) == 1:
            status = 'ocr_catalog_candidate'
            identity = next(iter(identities))
        else:
            identity = None
            status = 'conflicting_code_evidence' if conflict else 'unresolved'
        records.append({'query': item, 'crop_sha256': file_sha256(path), 'status': status,
                        'ocr_catalog_identity': identity, 'ocr': observations, 'ocr_decisions': decisions,
                        'visual_fallback_shortlist': visual, 'visual_corroboration': corroboration,
                        'ocr_errors': errors, 'ground_truth_verified': False, 'training_promotion': 'not_promoted'})
        print(f"{len(records)}: {status} {identity or ''}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {'artifact_type': 'ocr_first_card_label_proposals', 'queries_manifest': str(args.queries.resolve()),
              'queries_manifest_sha256': file_sha256(args.queries), 'catalog_identities': len(catalog),
              'ocr_backend': 'tesseract_whole_card_psm11', 'visual_fallback': False,
              'records': records, 'ground_truth_verified': False, 'training_promotion': 'not_promoted'}
    (output / 'proposals.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--catalog', action='append', required=True, help='game:language=/path/to/source/sidecars')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--ocr-language', default='eng')
    parser.add_argument('--min-ocr-confidence', type=float, default=80)
    args = parser.parse_args()
    try:
        propose(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
