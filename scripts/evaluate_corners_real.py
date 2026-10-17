#!/usr/bin/env python3
"""Corner accuracy of card-corner (YOLO-pose) detectors on reviewed real frames.

Reproduces the browser app's detector path: square letterbox (pad 114) of the whole frame,
confidence cut-off, the web `plausibleCard` shape filter (web/src/vision/pose.ts) before greedy
box NMS, and `portraitQuad` corner order. Labels come from scripts/real_corner_labels.py apply.

Per detector it reports, over labeled complete cards:
  recall / precision     IoU(quad) >= 0.5; detections on ignore regions are neither TP nor FP
  corner error           for precise labels only: corner distance / card diagonal (mean of both)
  good crops             precise cards detected with every corner within 2% / 3% / 5% of the diagonal
Results are split by scene group (binder pages vs hand-held close-ups) and source video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_corner_dataset import portrait  # noqa: E402
from real_corner_labels import diag, quad_iou  # noqa: E402


def plausible_card(q: np.ndarray, box: np.ndarray) -> bool:
    """Port of web/src/vision/pose.ts plausibleCard."""
    sign = 0
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if z == 0 or (sign and np.sign(z) != sign):
            return False
        sign = np.sign(z)
    ln = lambda a, b: float(np.hypot(*(a - b)))
    w = (ln(q[0], q[1]) + ln(q[3], q[2])) / 2
    h = (ln(q[0], q[3]) + ln(q[1], q[2])) / 2
    ratio = max(w, h) / max(min(w, h), 1e-9)
    if ratio > 2.0:
        return False
    area = abs(cv2.contourArea(np.float32(q)))
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    return box_area > 0 and area > 0.45 * box_area


def box_iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect(model, img: np.ndarray, size: int, conf: float, iou: float, shape_filter: bool = True):
    """App-equivalent detection: square letterbox, conf, shape filter, then greedy NMS."""
    h, w = img.shape[:2]
    s = size / max(w, h)
    nw, nh = round(w * s), round(h * s)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    r = model.predict(canvas, imgsz=size, conf=conf, iou=0.99, verbose=False, max_det=1000)[0]
    if not len(r.boxes):
        return []
    boxes = (r.boxes.xyxy.numpy() - [px, py, px, py]) / s
    kps = (r.keypoints.xy.numpy() - [px, py]) / s
    scores = r.boxes.conf.numpy()
    cand = [(float(c), b, k) for c, b, k in zip(scores, boxes, kps)]
    cand.sort(key=lambda t: -t[0])
    if shape_filter:
        cand = [t for t in cand if plausible_card(t[2], t[1])]
    kept = []
    for t in cand:
        if all(box_iou(k[1], t[1]) <= iou for k in kept):
            kept.append(t)
    return [(c, portrait(k)) for c, _, k in kept]


def corner_error(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, bool]:
    """Per-corner error / GT diagonal; flags a 180-degree corner-order flip."""
    d = diag(gt)
    e = np.linalg.norm(pred - gt, axis=1) / d
    flipped = np.linalg.norm(np.roll(pred, 2, 0) - gt, axis=1) / d
    return e, bool(flipped.max() < e.max() and flipped.max() < 0.1)


def group_of(frame: dict) -> str:
    return 'binder' if len(frame['cards']) >= 3 else 'closeup'


def evaluate(model, labels, size, conf, iou, shape_filter=True):
    rows = []
    for fr in labels['frames']:
        img = cv2.imread(str(ROOT / fr['frame']))
        dets = detect(model, img, size, conf, iou, shape_filter)
        gts = [np.float64(c['corners']) for c in fr['cards']]
        precise = [c['precise'] for c in fr['cards']]
        ign = [np.float64(q) for q in fr['ignore']]
        used = set()
        tp = fp = 0
        matched = {}
        for score, q in dets:
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if i not in used:
                    v = quad_iou(q, g)
                    if v > best:
                        best, bi = v, i
            if best >= 0.5:
                used.add(bi)
                matched[bi] = q
                tp += 1
            elif any(quad_iou(q, g) >= 0.3 for g in ign) or any(quad_iou(q, g) >= 0.3 for g in gts):
                pass  # on an ignore region, or a duplicate/partial of a labeled card already matched
            else:
                fp += 1
        errs = []
        for i, g in enumerate(gts):
            if precise[i]:
                if i in matched:
                    e, flip = corner_error(matched[i], g)
                    errs.append({'max': float(e.max()), 'mean': float(e.mean()), 'flip': flip})
                else:
                    errs.append(None)
        rows.append({'id': fr['id'], 'source': fr['source'], 'group': group_of(fr), 'cards': len(gts),
                     'tp': tp, 'fp': fp, 'precise': errs})
    return rows


def summarize(rows):
    tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows); n = sum(r['cards'] for r in rows)
    pe = [e for r in rows for e in r['precise']]
    det = [e for e in pe if e is not None]
    out = {'frames': len(rows), 'cards': n, 'tp': tp, 'fp': fp,
           'recall': round(tp / n, 3) if n else None, 'precision': round(tp / (tp + fp), 3) if tp + fp else None,
           'precise_cards': len(pe), 'precise_detected': len(det)}
    if det:
        mx = np.array([e['max'] for e in det]); mn = np.array([e['mean'] for e in det])
        out.update({'corner_err_mean_pct': round(100 * float(mn.mean()), 2), 'corner_err_median_pct': round(100 * float(np.median(mn)), 2),
                    'max_corner_err_median_pct': round(100 * float(np.median(mx)), 2),
                    'flips': sum(e['flip'] for e in det)})
        for t in (0.02, 0.03, 0.05):
            out[f'good_crops_{int(t * 100)}pct'] = round(float((mx <= t).sum()) / len(pe), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', type=Path, default=ROOT / 'runs/card_detector/real_corners/heldout_labels_v1.json')
    parser.add_argument('--detectors', nargs='+', required=True, help='name=weights.pt[@imgsz]')
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--no-shape-filter', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from ultralytics import YOLO
    labels = json.loads(args.labels.read_text())
    report = {'labels': str(args.labels), 'label_origin': labels['label_origin'], 'label_counts': labels['counts'],
              'conf': args.conf, 'iou': args.iou, 'shape_filter': not args.no_shape_filter, 'detectors': {}}
    for spec in args.detectors:
        name, rest = spec.split('=', 1)
        weights, _, size = rest.partition('@')
        model = YOLO(weights)
        rows = evaluate(model, labels, int(size or 640), args.conf, args.iou, not args.no_shape_filter)
        res = {'weights': weights, 'imgsz': int(size or 640), 'overall': summarize(rows),
               'by_group': {g: summarize([r for r in rows if r['group'] == g]) for g in sorted({r['group'] for r in rows})},
               'by_source': {s: summarize([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})}}
        report['detectors'][name] = res
        print(name, json.dumps(res['overall']), '| binder', json.dumps(res['by_group'].get('binder')),
              '| closeup recall', res['by_group'].get('closeup', {}).get('recall'), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
