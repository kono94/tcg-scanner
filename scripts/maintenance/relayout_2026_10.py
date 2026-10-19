#!/usr/bin/env python3
"""One-off 2026-10-19 repository relayout: versioned inputs leave runs/.

datasets/ holds source data, labels and review records; models/ holds pretrained and
deployed weights; runs/ becomes local scratch. Paths stored inside moved data files are rewritten with the same mapping
(model artifacts stay byte-identical). Run from the repository root:

    .env/bin/python scripts/maintenance/relayout_2026_10.py [--dry-run]

Safe to rerun: moved folders are skipped. It also works on a partial mirror such as the
GPU machine's, where missing sources are reported and skipped.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DVC = [sys.executable, '-m', 'dvc']

# (old, new, rewrite stored paths inside). Pointers: one per POINTERS entry.
CATALOG = {'pokemon_catalog_sync': 'pokemon-en-tcgdex', 'pokemon_gap_fill': 'pokemon-en-pokemontcgio',
           'pokemon_en_reference_sync': 'pokemon-en-targeted', 'pokemon_ja_reference_sync': 'pokemon-ja-official',
           'onepiece_catalog_sync': 'onepiece-en-bandai', 'onepiece_ja_catalog_sync': 'onepiece-ja-bandai'}
MOVES = [
    ('datasets/card_recognizer/cards', 'datasets/sources/catalog/onepiece-en-bandai-2024', False),
    *[(f'runs/card_recognizer/{old}', f'datasets/sources/catalog/{new}', True) for old, new in CATALOG.items()],
    ('runs/video_mining', 'datasets/sources/videos', True),
    ('runs/captures', 'datasets/sources/phone/captures', True),
    ('runs/feedback', 'datasets/sources/phone/feedback', True),
    ('runs/capture_intake', 'datasets/sources/phone/intake', True),
    ('runs/dataset_cleanup', 'datasets/review', True),
    ('runs/card_detector/real_corners', 'datasets/detection/real-corners', True),
    ('runs/card_detector/real_review', 'datasets/detection/real-detection-review', True),
    ('runs/card_detector/scenes-v4-val', 'datasets/detection/parity-scenes-v4', True),
    # Training set of the deployed detector; exists only on the GPU PC.
    ('runs/card_detector/corners-v4mix', 'datasets/detection/corners-v4mix', True),
    ('runs/prices/prices.json', 'datasets/prices/prices.json', False),
    ('runs/ref_thumbs', 'datasets/reference-thumbnails', False),
    ('runs/pretrained/dinov2-small', 'models/pretrained/dinov2-small', False),
    ('runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017', 'models/card-encoder/dinov2-arcface-gpu-20261017', False),
    ('runs/card_detector/pose/corners-s640-v4-ft', 'models/corner-detector/corners-s640-v4-ft', False),
    ('runs/browser_export/gpu-20261017', 'models/browser-export/gpu-20261017', False),
    ('runs/browser_export/pose-fixture', 'models/browser-export/corner-decoder-fixture', False),
]
POINTERS = ['datasets/sources/catalog', 'datasets/sources/videos', 'datasets/sources/phone', 'datasets/review',
            'datasets/detection/real-corners', 'datasets/detection/real-detection-review',
            'datasets/detection/parity-scenes-v4', 'datasets/detection/corners-v4mix', 'datasets/prices/prices.json', 'datasets/reference-thumbnails',
            'models/pretrained/dinov2-small', 'models/card-encoder/dinov2-arcface-gpu-20261017',
            'models/corner-detector/corners-s640-v4-ft', 'models/browser-export/gpu-20261017',
            'models/browser-export/corner-decoder-fixture']
TEXT = {'.json', '.jsonl', '.txt', '.yaml', '.yml', '.csv', '.md', '.py', '.log', '.tsv'}
PATTERN = re.compile('|'.join(re.escape(old) for old, _, _ in sorted(MOVES, key=lambda m: -len(m[0])))
                     + r'(?=[/"\'\s,)\]]|$)')
MAPPING = {old: new for old, new, _ in MOVES}


def rewrite(folder: Path, dry: bool) -> int:
    changed = 0
    for path in sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in TEXT):
        text = path.read_text(errors='surrogateescape')
        new = PATTERN.sub(lambda m: MAPPING[m.group(0)], text)
        if new != text:
            changed += 1
            if not dry:
                path.write_text(new, errors='surrogateescape')
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-dvc', action='store_true', help='move and rewrite only (machines without the DVC cache)')
    args = ap.parse_args()
    for old, new, rewrite_inside in MOVES:
        src, dst = ROOT / old, ROOT / new
        pointer = src.with_name(src.name + '.dvc')
        if pointer.exists() and not args.dry_run and not args.no_dvc:
            subprocess.run([*DVC, 'remove', str(pointer.relative_to(ROOT))], cwd=ROOT, check=True)
        if dst.exists():
            print(f'already moved  {new}')
        elif not src.exists():
            print(f'missing        {old}')
            continue
        else:
            print(f'move           {old} -> {new}')
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        if rewrite_inside:
            n = rewrite(dst if dst.exists() else src, args.dry_run)
            print(f'  rewrote paths in {n} files')
    if not args.dry_run and not args.no_dvc:
        present = [p for p in POINTERS if (ROOT / p).exists()]
        subprocess.run([*DVC, 'add', *present], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
