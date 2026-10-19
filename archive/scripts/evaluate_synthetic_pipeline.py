#!/usr/bin/env python3
"""Detector-to-retrieval evaluation on synthetic validation scenes (true corners known).

Compares recognition from detected corner/OBB crops with recognition from true-corner crops, so
detector-induced loss is measured separately from the encoder. Synthetic scenes composite
gallery reference images; this is not a camera benchmark and overstates real accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from export_browser_models import load_encoder  # noqa: E402
from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input  # noqa: E402
from training.card_recognizer.retrieval_training import identity_key  # noqa: E402


def portrait(corners: np.ndarray) -> np.ndarray:
    """Same geometric ordering as web/src/vision/obb.ts portraitQuad."""
    p = [tuple(c) for c in corners]
    edge = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    if edge(p[0], p[1]) > edge(p[1], p[2]):
        p = p[1:] + p[:1]
    a, b, c, d = (p[0], p[1], p[2], p[3]) if (p[0][1] + p[1][1]) <= (p[2][1] + p[3][1]) else (p[2], p[3], p[0], p[1])
    if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
        a, b, c, d = b, a, d, c
    return np.float32([a, b, c, d])


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017')
    parser.add_argument('--detector', type=Path, default=ROOT / 'runs/card_detector/obb/corners-s640-v4-ft/weights/best.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--synthetic', type=Path, default=ROOT / 'runs/card_detector/synthetic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO
    torch.set_num_threads(4)
    run = args.run.resolve(strict=True)
    config = json.loads((run / 'resolved_config.json').read_text())
    index = json.loads((run / 'gallery_index.json').read_text())
    keys = [identity_key(r['identity']) for r in index['references']]
    gallery = np.load(run / 'gallery_embeddings.npy').astype(np.float32)
    encoder = load_encoder(run, config)
    detector = YOLO(str(args.detector))

    def rank(crop: np.ndarray, truth: str) -> int:
        with torch.inference_mode():
            v = encoder(torch.from_numpy(prepare_encoder_input(crop)[None])).numpy()[0]
        order = np.argsort(-(gallery @ v))
        ids = [keys[i] for i in order[:50]]
        return ids.index(truth) + 1 if truth in ids else 51

    rows, scenes = [], sorted((args.synthetic / 'annotations/val').glob('*.json'))
    detections_total = false_positives = 0
    for path in scenes:
        scene = json.loads(path.read_text())
        bgr = cv2.imread(str(args.synthetic / scene['image']))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = detector.predict(bgr, imgsz=args.imgsz, conf=0.25, iou=0.7, verbose=False)[0]
        boxes = (result.keypoints.xy.numpy() if len(result.boxes) else np.zeros((0, 4, 2))) if detector.task == 'pose' else (result.obb.xyxyxyxy.numpy() if len(result.obb) else np.zeros((0, 4, 2)))
        detections_total += len(boxes)
        used = set()
        for inst in scene['instances']:
            truth = f"onepiece:en:{inst['card_id']}"
            true_q = portrait(np.float32(inst['true_corners']))
            ious = [polygon_iou(true_q, portrait(b)) for b in boxes]
            best = int(np.argmax(ious)) if ious else -1
            matched = best >= 0 and ious[best] >= 0.5 and best not in used
            if matched:
                used.add(best)
            row = {'scene': path.stem, 'identity': truth, 'enrolled': truth in keys,
                   'true_corner_rank': rank(canonicalize_card(rgb, true_q), truth) if truth in keys else None,
                   'detected': matched, 'iou': ious[best] if ious else 0.0,
                   'detected_rank': rank(canonicalize_card(rgb, portrait(boxes[best])), truth) if matched and truth in keys else None}
            rows.append(row)
        false_positives += len(boxes) - len(used)
    enrolled = [r for r in rows if r['enrolled']]
    detected = [r for r in enrolled if r['detected']]
    report = {
        'scope': 'Synthetic validation scenes composited from gallery references; not real-camera accuracy',
        'scenes': len(scenes), 'instances': len(rows), 'enrolled_instances': len(enrolled),
        'detector': {'matched_iou50': sum(r['detected'] for r in rows), 'detections': detections_total,
                     'unmatched_detections': false_positives, 'mean_iou_matched': float(np.mean([r['iou'] for r in rows if r['detected']] or [0]))},
        'recall_at_1_true_corners': float(np.mean([r['true_corner_rank'] == 1 for r in enrolled])),
        'recall_at_1_detected_crops_given_detected': float(np.mean([r['detected_rank'] == 1 for r in detected])) if detected else None,
        'end_to_end_recall_at_1': float(np.mean([r['detected'] and r['detected_rank'] == 1 for r in enrolled])),
        'records': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
