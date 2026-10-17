#!/usr/bin/env python3
"""Compare detectors end to end on real frames from held-out (test/val) source videos.

For each detector: detect, straighten each card from its own corners (pose keypoints or the
rotated box), embed with an ONNX encoder, search the gallery, and count crops the pilot match
policy accepts (similarity >= 0.45 and margin >= 0.05 to the next different artwork). This is a
proxy for usable crops, not labeled accuracy: frames have no per-card identity labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from build_corner_dataset import portrait  # noqa: E402
from review_real_detections import TEST_SOURCES, VAL_SOURCES, frames, source_of  # noqa: E402
from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input  # noqa: E402


def family(identity: str) -> str:
    game, _lang, card = identity.split(':', 2)
    return f'onepiece:{card}' if game == 'onepiece' else identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detectors', nargs='+', required=True, help='name=weights.pt[@imgsz]')
    parser.add_argument('--export', type=Path, required=True, help='browser export dir with encoder.onnx, gallery.f16, gallery.json')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from ultralytics import YOLO
    sess = ort.InferenceSession(str(args.export / 'encoder.onnx'), providers=['CPUExecutionProvider'])
    meta = json.loads((args.export / 'gallery.json').read_text())
    G = np.fromfile(args.export / 'gallery.f16', dtype='<f2').astype(np.float32).reshape(-1, meta['dimension'])
    ids = meta['identities']
    held = [f for f in frames() if source_of(f) in TEST_SOURCES | VAL_SOURCES]
    report = {'frames': len(held), 'sources': sorted({source_of(f) for f in held}), 'detectors': {}}
    for spec in args.detectors:
        name, rest = spec.split('=', 1)
        weights, _, size = rest.partition('@')
        model = YOLO(weights)
        imgsz = int(size or 640)
        boxes_total = accepted = 0
        scores = []
        for f in held:
            img = cv2.imread(str(f))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = model.predict(img, imgsz=imgsz, conf=args.conf, iou=0.7, verbose=False)[0]
            quads = (r.keypoints.xy.numpy() if model.task == 'pose' else r.obb.xyxyxyxy.numpy()) if (len(r.boxes) if model.task == 'pose' else len(r.obb)) else []
            for q in quads:
                boxes_total += 1
                try:
                    crop = canonicalize_card(rgb, np.float32(portrait(q)))
                except ValueError:
                    continue
                v = sess.run(None, {'pixels': prepare_encoder_input(crop)[None]})[0][0]
                s = G @ v
                order = np.argsort(-s)[:20]
                top = ids[order[0]]
                comp = next((s[i] for i in order if family(ids[i]) != family(top)), 0.0)
                scores.append(float(s[order[0]]))
                accepted += bool(s[order[0]] >= 0.45 and s[order[0]] - comp >= 0.05)
        report['detectors'][name] = {'weights': weights, 'imgsz': imgsz, 'boxes': boxes_total, 'accepted': accepted,
                                     'median_top_score': float(np.median(scores)) if scores else None}
        print(json.dumps({name: report['detectors'][name]}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
