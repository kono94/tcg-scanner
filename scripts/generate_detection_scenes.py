#!/usr/bin/env python3
"""Generate synthetic detection scenes (images + true-corner JSON) from existing card images.

`scripts/build_corner_dataset.py` converts the JSON annotations to YOLO-pose labels.
Card faces: curated gallery references (all games) plus real train-split query crops.
Train and val scenes use disjoint card identities; real test-split crops are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_detector.scene_renderer import render  # noqa: E402

_TEXTURES: dict[str, list[str]] = {}


def card_pools(dataset: Path, val_fraction: float, seed: int) -> dict[str, list[str]]:
    manifest = json.loads((dataset / 'recognition/manifest.json').read_text())
    pools: dict[str, list[str]] = {'train': [], 'val': []}
    for row in manifest['gallery']:
        ident = row['identity']
        family = f"{ident['game']}:{ident['card_id'].split('_p')[0]}"
        h = int(hashlib.sha256(f'{seed}:{family}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        pools['val' if h < val_fraction else 'train'].append(str(dataset / row['image']))
    # Real photographed faces add sleeve glare and print texture; only the matching split.
    for split in ('train', 'val'):
        pools[split] += [str(dataset / q['image']) for q in manifest['splits'][split]['queries']]
    return pools


def work(task):
    index, split, seed, out, w, h, upright, version = task
    pixels, meta = render(seed, _TEXTURES[split], w, h, upright=upright, version=version)
    name = f'scene_{index:06d}'
    Image.fromarray(pixels).save(Path(out) / 'images' / split / f'{name}.jpg', quality=92)
    meta.update(split=split, image=f'images/{split}/{name}.jpg')
    (Path(out) / 'annotations' / split / f'{name}.json').write_text(json.dumps(meta))
    return split, meta['kind'], len(meta['instances']), len(meta['unlabeled_cards'])


def init(pools):
    _TEXTURES.update(pools)
    cv2.setNumThreads(1)


def preview(out: Path, split: str, n: int = 16):
    files = sorted((out / 'annotations' / split).glob('*.json'))[:n]
    tiles = []
    for f in files:
        meta = json.loads(f.read_text())
        im = Image.open(out / meta['image']).convert('RGB')
        d = ImageDraw.Draw(im)
        for inst in meta['instances']:
            q = [tuple(p) for p in inst['true_corners']]
            d.line(q + [q[0]], fill='#3cff8a', width=3)
        for inst in meta['unlabeled_cards']:
            q = [tuple(p) for p in inst['true_corners']]
            d.line(q + [q[0]], fill='#ff5050', width=2)
        im.thumbnail((480, 480))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 260, 16), fill='black')
        d.text((3, 2), f"{meta['kind']}/{meta.get('layout')} {len(meta['instances'])} lab {len(meta['unlabeled_cards'])} unl", fill='white')
        tiles.append(im)
    cols = 4
    sheet = Image.new('RGB', (cols * 480, math.ceil(len(tiles) / cols) * 480), '#111')
    for i, t in enumerate(tiles):
        sheet.paste(t, (i % cols * 480, i // cols * 480))
    sheet.save(out / f'preview_{split}.jpg', quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, default=6000)
    parser.add_argument('--val', type=int, default=600)
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 768], help='landscape size; a share of scenes is portrait')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--val-fraction', type=float, default=.08)
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--renderer-version', type=int, default=3, choices=[3, 4],
                        help='4 adds close-ups, card backs, sleeves/toploaders, settled cards and pocket seams')
    parser.add_argument('--legacy-rotations', action='store_true', help='renderer v2: random rotations incl. sideways/upside-down')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty')
    for kind in ('images', 'annotations'):
        for split in ('train', 'val'):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    pools = card_pools(args.dataset, args.val_fraction, args.seed)
    rng = np.random.default_rng(args.seed)
    tasks = []
    for i in range(args.val + args.train):
        split = 'val' if i < args.val else 'train'
        w, h = args.size
        if rng.random() < .35:  # phone held upright
            w, h = h, w
        tasks.append((i, split, args.seed * 100003 + i, str(out), w, h, not args.legacy_rotations, args.renderer_version))
    stats: dict = {}
    with Pool(args.workers, initializer=init, initargs=(pools,), maxtasksperchild=200) as pool:
        for n, (split, kind, labeled, unlabeled) in enumerate(pool.imap_unordered(work, tasks, chunksize=8)):
            s = stats.setdefault(split, {'scenes': 0, 'labeled': 0, 'unlabeled': 0, 'kinds': {}})
            s['scenes'] += 1; s['labeled'] += labeled; s['unlabeled'] += unlabeled
            s['kinds'][kind] = s['kinds'].get(kind, 0) + 1
            if n % 500 == 0:
                print(f'{n}/{len(tasks)}', flush=True)
    renderer = 'v2' if args.legacy_rotations else f'v{args.renderer_version}'
    summary = {'renderer': f'scene_renderer {renderer}', 'label_origin': f'synthetic_renderer_{renderer}', 'real_camera_validation': False,
               'card_sources': {k: len(v) for k, v in pools.items()}, 'identity_split': 'by game + base card number family',
               'seed': args.seed, 'stats': stats}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for split in ('train', 'val'):
        preview(out, split)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
