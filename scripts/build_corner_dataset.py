#!/usr/bin/env python3
"""YOLO-pose dataset of card corners: box + 4 keypoints (TL, TR, BR, BL in portrait image order).

Sources:
  * renderer-v2 synthetic scenes (true corners from the renderer), subsampled;
  * optional AI-reviewed real frames (`--real reviewed.json`), where cards that were not
    accepted are painted over so they are not learned as background.
Corner order is geometric (upright portrait reading), not artwork orientation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def portrait(quad) -> np.ndarray:
    """Same rule as web/src/vision/obb.ts portraitQuad."""
    p = [np.asarray(c, np.float64) for c in quad]
    if (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]) < 0:
        p = [p[0], p[3], p[2], p[1]]
    length = lambda a, b: float(np.hypot(*(a - b)))
    e01, e12 = length(p[0], p[1]) + length(p[2], p[3]), length(p[1], p[2]) + length(p[3], p[0])
    card_shaped = max(e01, e12) / min(e01, e12) >= 1.2
    best, tilt = None, np.inf
    for k in range(4):
        q = [p[(k + i) % 4] for i in range(4)]
        if card_shaped and length(q[0], q[1]) > length(q[1], q[2]):
            continue
        t = abs(np.arctan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
        if t < tilt:
            best, tilt = q, t
    return np.float64(best)


def tilt_deg(quad: np.ndarray) -> float:
    """Rotation of the card's long axis away from vertical, 0-90 degrees (upside down counts as upright)."""
    q = portrait(quad)
    v = (q[3] + q[2]) / 2 - (q[0] + q[1]) / 2
    a = abs(np.degrees(np.arctan2(v[0], v[1]))) % 180
    return float(min(a, 180 - a))


def label_line(quad: np.ndarray, w: int, h: int) -> str:
    q = portrait(quad)
    x0, y0 = np.clip(q.min(0), 0, [w - 1, h - 1])
    x1, y1 = np.clip(q.max(0), 0, [w - 1, h - 1])
    box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    kpts = []
    for x, y in q:
        kpts += [x / w, y / h, 2]
    return '0 ' + ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in box + kpts)


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, nargs='+', default=[ROOT / 'runs/card_detector/scenes-v2'],
                        help='one or more scene sets; train/val sizes apply per set')
    parser.add_argument('--max-tilt', type=float, default=90.0,
                        help='skip scenes with any labeled card turned more than this from upright (sideways cards make corner names ambiguous)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--val', type=int, default=150)
    parser.add_argument('--real', type=Path, help='reviewed real-frame corners JSON')
    parser.add_argument('--real-repeat', type=int, default=8, help='oversample real frames this many times per epoch')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty')
    rng = random.Random(args.seed)
    counts = {}
    skipped = 0
    for set_index, scenes in enumerate(args.scenes):
      for split, n in (('train', args.train), ('val', args.val)):
        if n <= 0:
            continue
        files = sorted((scenes / 'annotations' / split).glob('*.json'))
        rng.shuffle(files)
        taken = 0
        for f in files:
            if taken >= n:
                break
            meta = json.loads(f.read_text())
            if any(tilt_deg(np.asarray(i['true_corners'])) > args.max_tilt for i in meta['instances']):
                skipped += 1
                continue
            taken += 1
            img = scenes / meta['image']
            name = f's{set_index}_{img.name}'
            link(img, out / 'images' / split / name)
            lines = [label_line(np.asarray(i['true_corners']), meta['width'], meta['height']) for i in meta['instances']]
            (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
            (out / 'labels' / split / (Path(name).stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
            counts[split] = counts.get(split, 0) + 1
    real = 0
    if args.real:
        review = json.loads(args.real.read_text())
        for fr in review['frames']:
            if fr.get('split') == 'test':
                raise ValueError('Test-split frames must not be used for training')
            img = cv2.imread(str(ROOT / fr['frame']))
            h, w = img.shape[:2]
            # Paint over detected-but-rejected cards so they are not learned as background.
            for q in fr.get('mask', []):
                cv2.fillPoly(img, [np.int32(q)], tuple(int(v) for v in img.reshape(-1, 3).mean(0)))
            split = 'val' if fr.get('split') == 'val' else 'train'
            reps = 1 if split == 'val' else args.real_repeat
            for r in range(reps):
                name = f"real_{fr['id']}_{r}.jpg"
                (out / 'images' / split).mkdir(parents=True, exist_ok=True)
                (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / 'images' / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = [label_line(np.asarray(q), w, h) for q in fr['corners']]
                (out / 'labels' / split / name.replace('.jpg', '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
            real += 1
    (out / 'dataset.yaml').write_text(
        f'path: {json.dumps(str(out))}\ntrain: images/train\nval: images/val\nkpt_shape: [4, 3]\n'
        'flip_idx: [1, 0, 3, 2]\nnames:\n  0: card\n')
    summary = {'synthetic': counts, 'skipped_sideways_scenes': skipped, 'scene_sets': [str(x) for x in args.scenes], 'real_frames': real, 'corner_order': 'TL,TR,BR,BL upright portrait (geometric)',
               'label_origin': 'renderer_v2 true corners' + (' + AI-reviewed real corners' if real else '')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
