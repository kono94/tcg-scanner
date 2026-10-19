#!/usr/bin/env python3
"""Small catalog reference images for a deployed scanner (served at /refs/, like the dev server).

The dev server serves full-size references from datasets/curated. A deployment gets JPEG
thumbnails at the same relative paths instead, only for the references the bundle's catalog
uses. Incremental: existing thumbnails are kept, so reruns only add new references.
Display only; never used for recognition or the gallery.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def thumb(job: tuple[Path, Path, int, int]) -> str:
    src, dst, width, quality = job
    if dst.exists():
        return 'kept'
    if not src.exists():
        return 'missing'
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert('RGB')
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
        tmp = dst.with_suffix('.tmp')
        im.save(tmp, 'JPEG', quality=quality, optimize=True, progressive=True)
        tmp.rename(dst)
    return 'written'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'web/public/bundle/catalog.json')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/reference-thumbnails')
    parser.add_argument('--width', type=int, default=240, help='candidate tiles are 92 css px; 240 covers 2-3x screens')
    parser.add_argument('--quality', type=int, default=78)
    args = parser.parse_args()
    refs = sorted({c['ref'] for c in json.loads(args.catalog.read_text())['cards'] if c.get('ref')})
    jobs = [(args.dataset / r, args.output / r, args.width, args.quality) for r in refs]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor() as pool:
        for result in pool.map(thumb, jobs, chunksize=64):
            counts[result] = counts.get(result, 0) + 1
    size = sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())
    print(json.dumps({'references': len(refs), **counts, 'output': str(args.output), 'megabytes': round(size / 1e6, 1)}))
    if counts.get('missing'):
        raise SystemExit('Some references are missing: run `dvc pull datasets/curated`')


if __name__ == '__main__':
    main()
