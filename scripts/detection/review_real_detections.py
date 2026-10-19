#!/usr/bin/env python3
"""Detector proposals on retained real video frames, with numbered overlays for visual review.

  propose  run a detector on every retained 4K frame; write proposals + review overlays
  score    apply a review file (accepted/rejected box ids, missed-card counts per frame)
           and report AI-reviewed precision/recall by source split

The review is an AI visual judgement of whether each box tightly covers one complete
physical card face. It is not independent corner ground truth. Frames stay grouped by
source video; recognition test-split sources are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEST_SOURCES = {'4QxpRG4E6d0', 'marinefordtcg_op12_two_boxes'}
VAL_SOURCES = {'evans-collection-community-binder-submissions-2026', 'pokemon-gen5-binder-pokeduell', 'zjoawf6wEgc'}


def source_of(frame: Path) -> str:
    for parent in [frame.parent.parent, frame.parent]:
        m = parent / 'manifest.json'
        if m.is_file():
            try:
                return json.loads(m.read_text())['source']['source_group']
            except (KeyError, json.JSONDecodeError):
                pass
    return frame.parent.parent.name


def frames(per_folder: int = 0) -> list[Path]:
    files = sorted(ROOT.glob('datasets/sources/videos/**/frames/*.jpg')) + sorted(ROOT.glob('datasets/sources/videos/**/frames/*.png'))
    if not per_folder:
        return files
    by: dict[Path, list[Path]] = {}
    for f in files:
        by.setdefault(f.parent, []).append(f)
    # Evenly spaced fixed subset per frame folder, stable across runs.
    picked = []
    for group in by.values():
        idx = np.linspace(0, len(group) - 1, min(per_folder, len(group))).round().astype(int)
        picked += [group[i] for i in sorted(set(idx))]
    return sorted(picked)


def propose(args):
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    out = args.output
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    records = []
    selected = [f for f in frames(args.per_folder) if not args.splits or
                ('test' if source_of(f) in TEST_SOURCES else 'val' if source_of(f) in VAL_SOURCES else 'train') in args.splits]
    for i, f in enumerate(selected):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=.7, verbose=False, max_det=300)[0]
        boxes = r.keypoints.xy.numpy().tolist() if len(r.boxes) else []  # predicted card corners
        conf = r.boxes.conf.numpy().round(3).tolist() if len(r.boxes) else []
        name = f'f{i:03d}'
        src = source_of(f)
        records.append({'id': name, 'frame': str(f.relative_to(ROOT)), 'source': src, 'width': w, 'height': h,
                        'split': 'test' if src in TEST_SOURCES else 'val' if src in VAL_SOURCES else 'other',
                        'boxes': [{'n': k + 1, 'corners': np.round(b, 1).tolist(), 'conf': c} for k, (b, c) in enumerate(zip(boxes, conf))]})
        scale = 1600 / w
        view = cv2.resize(img, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)
        for k, b in enumerate(boxes):
            q = (np.array(b) * scale).astype(np.int32)
            color = [(60, 255, 120), (255, 200, 40), (40, 160, 255), (255, 80, 255)][k % 4]
            cv2.polylines(view, [q], True, color, 2, cv2.LINE_AA)
            for corner in q:
                cv2.circle(view, tuple(int(v) for v in corner), 4, color, -1, cv2.LINE_AA)
            c = q.mean(0).astype(int)
            cv2.putText(view, str(k + 1), (c[0] - 9, c[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, str(k + 1), (c[0] - 9, c[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
        cv2.putText(view, f'{name} {src} {len(boxes)} boxes', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / 'overlays' / f'{name}.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / 'proposals.json').write_text(json.dumps({'weights': str(args.weights), 'imgsz': args.imgsz, 'conf': args.conf,
                                                    'frames': records}, indent=1))
    print(f'{len(records)} frames, {sum(len(r["boxes"]) for r in records)} boxes -> {out}')


def score(args):
    """Precision = accepted / non-neutral boxes; recall = accepted / counted complete foreground cards."""
    proposals = json.loads((args.output / 'proposals.json').read_text())
    review = json.loads(args.review.read_text())
    counts = json.loads(args.counts.read_text())['frames']
    totals: dict = {}
    for r in proposals['frames']:
        rv = review['frames'].get(r['id'], {})
        count = counts[r['frame']]['complete_cards']
        n = len(r['boxes'])
        rejected, neutral = set(rv.get('reject', [])), set(rv.get('neutral', []))
        if rv.get('reject_all'):
            rejected = {b['n'] for b in r['boxes']} - neutral
        accepted = n - len(rejected) - len(neutral)
        t = totals.setdefault(r['split'], {'frames': 0, 'boxes': 0, 'accepted': 0, 'cards': 0})
        t['frames'] += 1; t['boxes'] += n - len(neutral); t['accepted'] += accepted; t['cards'] += count
    overall = {k: sum(t[k] for t in totals.values()) for k in ('frames', 'boxes', 'accepted', 'cards')}
    for t in [*totals.values(), overall]:
        t['precision'] = round(t['accepted'] / t['boxes'], 3) if t['boxes'] else None
        t['recall'] = round(min(t['accepted'], t['cards']) / t['cards'], 3) if t['cards'] else None
    report = {'weights': proposals['weights'], 'imgsz': proposals['imgsz'], 'review': str(args.review),
              'label_basis': 'AI visual review of numbered detector boxes and AI card counts on retained real 4K frames; not corner ground truth',
              'by_split': totals, 'overall': overall}
    print(json.dumps(report, indent=2))
    (args.output / 'score.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('propose')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--per-folder', type=int, default=2, help='fixed evenly spaced frames per folder (0 = all)')
    p.add_argument('--splits', nargs='*', choices=['train', 'val', 'test'], help='only frames from these source splits')
    s = sub.add_parser('score')
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--review', type=Path, required=True)
    s.add_argument('--counts', type=Path, default=ROOT / 'datasets/detection/real-detection-review/frame_counts.json')
    args = parser.parse_args()
    propose(args) if args.cmd == 'propose' else score(args)


if __name__ == '__main__':
    sys.exit(main())
