#!/usr/bin/env python3
"""Detector box geometry vs true card corners on renderer-v2 validation scenes.

Per matched card (polygon IoU >= 0.5):
  angle_error   long-axis angle of the predicted box vs the true card, degrees (mod 180)
  corner_error  max corner distance after portrait ordering, as a fraction of card height
  rect_floor    the same corner error for the best rectangle (min-area rect of the true quad):
                the floor any OBB can reach on that card; the rest is detector error
Synthetic scenes only: this measures geometry on renders, not real-camera accuracy.
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
from evaluate_synthetic_pipeline import polygon_iou, portrait  # noqa: E402


def long_axis_deg(q: np.ndarray) -> float:
    q = portrait(q)
    v = (q[3] + q[2]) / 2 - (q[0] + q[1]) / 2  # top-centre -> bottom-centre
    return float(np.degrees(np.arctan2(v[1], v[0])))


def corner_error(a: np.ndarray, b: np.ndarray) -> float:
    a, b = portrait(a), portrait(b)
    h = np.linalg.norm(b[3] - b[0])
    # Best of the two 180-degree readings: orientation is a separate question.
    return float(min(np.linalg.norm(a - b, axis=1).max(), np.linalg.norm(np.roll(a, 2, 0) - b, axis=1).max()) / h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, nargs='+', required=True)
    parser.add_argument('--scenes', type=Path, default=ROOT / 'runs/card_detector/scenes-v2')
    parser.add_argument('--list', type=Path, help='image list (e.g. val_small.txt); default all val')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from ultralytics import YOLO
    images = [Path(l.strip()) for l in args.list.read_text().splitlines() if l.strip()] if args.list else sorted((args.scenes / 'images/val').glob('*.jpg'))
    report = {}
    for weights in args.weights:
        model = YOLO(str(weights))
        rows, n_true, n_pred = [], 0, 0
        for img in images:
            meta = json.loads((args.scenes / 'annotations/val' / (img.stem + '.json')).read_text())
            r = model.predict(str(img), imgsz=args.imgsz, conf=.25, iou=.7, verbose=False, device=args.device)[0]
            if model.task == 'pose':
                boxes = r.keypoints.xy.numpy() if len(r.boxes) else np.zeros((0, 4, 2))
            else:
                boxes = r.obb.xyxyxyxy.numpy() if len(r.obb) else np.zeros((0, 4, 2))
            n_pred += len(boxes)
            used = set()
            for inst in meta['instances']:
                n_true += 1
                t = np.float32(inst['true_corners'])
                ious = [polygon_iou(t, b) for b in boxes]
                k = int(np.argmax(ious)) if ious else -1
                if k < 0 or ious[k] < .5 or k in used:
                    continue
                used.add(k)
                rect = cv2.boxPoints(cv2.minAreaRect(t))
                d = abs(long_axis_deg(boxes[k]) - long_axis_deg(t)) % 180
                rows.append({'iou': ious[k], 'angle_error': min(d, 180 - d),
                             'corner_error': corner_error(boxes[k], t), 'rect_floor': corner_error(rect, t)})
        a = np.array([[r['iou'], r['angle_error'], r['corner_error'], r['rect_floor']] for r in rows]) if rows else np.zeros((0, 4))
        pct = lambda col, q: round(float(np.percentile(a[:, col], q)), 4) if len(a) else None
        report[str(weights)] = {
            'images': len(images), 'true_cards': n_true, 'predicted_boxes': n_pred, 'matched_iou50': len(rows),
            'recall_iou50': round(len(rows) / max(n_true, 1), 3), 'precision_iou50': round(len(rows) / max(n_pred, 1), 3),
            'mean_iou': pct(0, 50) and round(float(a[:, 0].mean()), 3),
            'angle_error_deg': {'median': pct(1, 50), 'p90': pct(1, 90)},
            'corner_error_of_height': {'median': pct(2, 50), 'p90': pct(2, 90)},
            'rectangle_floor_of_height': {'median': pct(3, 50), 'p90': pct(3, 90)},
        }
        print(json.dumps({str(weights.name): report[str(weights)]}), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
