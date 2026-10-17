#!/usr/bin/env python3
"""Real-frame card-corner labels: teacher proposals, edge refinement, review sheets, labels.

  propose  run one or more teacher detectors (high input size, low confidence), fuse overlapping
           corner sets, add optional rough manual quads, snap every quad to straight image edges,
           and render numbered rectified review tiles
  apply    turn a review file (accept / ignore / reject per proposal id) into a labels file

Label origin is "teacher proposal + edge-line refinement + AI visual review"; it is not
independent human ground truth. Rejected proposals are dropped; `ignore` proposals become
ignore regions (a partially hidden or uncertain card that must neither count as a miss nor as a
false detection). Every complete card a frame shows must be accepted or ignored before the
frame is used for recall.

Corner order everywhere: TL, TR, BR, BL of the upright portrait reading (build_corner_dataset.portrait).
Coordinates are source-frame pixels.
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

TILE_W, TILE_H = 216, 300


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return float(inter / union) if union > 0 else 0.0


def diag(q: np.ndarray) -> float:
    return float((np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2)


def refine(gray_dx: np.ndarray, gray_dy: np.ndarray, quad: np.ndarray, search: float = 0.045) -> tuple[np.ndarray, list[float]]:
    """Snap each side to the strongest straight edge near it; corners = intersections of the side lines.

    The physical card corner is rounded, so the label is the intersection of the straight edges
    (the same convention as the renderer's texture quad). Sides whose edge evidence is weak keep
    their original line. Returns the refined quad and per-side inlier fractions.
    """
    h, w = gray_dx.shape
    q = np.float64(quad)
    c = q.mean(0)
    r = max(3.0, search * diag(q))
    lines, support = [], []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (a + b) / 2 - c) < 0:
            n = -n
        offsets = np.arange(-r, r + 0.5, 0.5)
        pts = []
        for t in np.linspace(0.1, 0.9, 48):
            p = a + (b - a) * t
            xs, ys = p[0] + offsets * n[0], p[1] + offsets * n[1]
            ok = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
            if ok.sum() < len(offsets) * 0.6:
                continue
            xi, yi = np.clip(xs, 1, w - 2), np.clip(ys, 1, h - 2)
            g = np.abs(cv2.remap(gray_dx, xi.astype(np.float32)[None], yi.astype(np.float32)[None], cv2.INTER_LINEAR)[0] * n[0] +
                       cv2.remap(gray_dy, xi.astype(np.float32)[None], yi.astype(np.float32)[None], cv2.INTER_LINEAR)[0] * n[1])
            g[~ok] = 0
            k = int(np.argmax(g))
            if g[k] <= 0:
                continue
            pts.append(p + offsets[k] * n)
        best, best_in = None, 0
        pts = np.array(pts)
        if len(pts) >= 10:
            rng = np.random.default_rng(i)
            for _ in range(200):
                i0, i1 = rng.choice(len(pts), 2, replace=False)
                u = pts[i1] - pts[i0]
                if np.linalg.norm(u) < 1e-6:
                    continue
                nn = np.array([-u[1], u[0]]) / np.linalg.norm(u)
                dist = np.abs((pts - pts[i0]) @ nn)
                inl = dist < max(1.0, 0.003 * diag(q))
                if inl.sum() > best_in:
                    best_in, best = inl.sum(), inl
        frac = best_in / 48
        if best is not None and frac >= 0.45:
            sel = pts[best]
            mean = sel.mean(0)
            _, _, vt = np.linalg.svd(sel - mean)
            lines.append((mean, vt[0]))
        else:
            lines.append((a, d))
        support.append(round(float(frac), 2))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        m = np.array([d1, -d2]).T
        if abs(np.linalg.det(m)) < 1e-6:
            out.append(q[i])
            continue
        t = np.linalg.solve(m, p2 - p1)
        out.append(p1 + t[0] * d1)
    out = np.array(out)
    # Guard: a refinement that moves a corner far (wrong edge) is discarded.
    if np.max(np.linalg.norm(out - q, axis=1)) > 2.5 * r:
        return q, support
    return out, support


def gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


def tile(img: np.ndarray, quad: np.ndarray, label: str, color=(0, 255, 0)) -> np.ndarray:
    """Overview of the rectified card plus four zoomed corner patches.

    The card is rectified to 500x700 from the label quad, so a correct label puts each physical
    corner (the intersection of the straight card edges) on the red cross of its patch. One patch
    pixel is 0.2% of the card width; the red/magenta rings mark 1% / 3% of the card diagonal.
    """
    cw, ch, pad = 500, 700, 70
    dst = np.float32([[pad, pad], [pad + cw, pad], [pad + cw, pad + ch], [pad, pad + ch]])
    m = cv2.getPerspectiveTransform(np.float32(quad), dst)
    big = cv2.warpPerspective(img, m, (cw + 2 * pad, ch + 2 * pad), flags=cv2.INTER_LINEAR)
    ov = cv2.resize(big, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    s = TILE_W / (cw + 2 * pad)
    cv2.polylines(ov, [np.int32(dst * s)], True, color, 1, cv2.LINE_AA)
    half, ps = 40, TILE_H // 2
    patches = []
    for p in dst:
        x, y = int(p[0]), int(p[1])
        patch = cv2.resize(big[y - half:y + half, x - half:x + half], (ps, ps), interpolation=cv2.INTER_NEAREST)
        f = ps / (2 * half)
        c = (ps // 2, ps // 2)
        cv2.line(patch, (0, c[1]), (ps, c[1]), (0, 255, 0), 1)
        cv2.line(patch, (c[0], 0), (c[0], ps), (0, 255, 0), 1)
        cv2.circle(patch, c, int(0.01 * np.hypot(cw, ch) * f), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(patch, c, int(0.03 * np.hypot(cw, ch) * f), (255, 0, 255), 1, cv2.LINE_AA)
        patches.append(patch)
    grid = np.vstack([np.hstack(patches[:2]), np.hstack([patches[3], patches[2]])])
    t = np.hstack([ov, grid])
    cv2.rectangle(t, (0, 0), (TILE_W, 18), (0, 0, 0), -1)
    cv2.putText(t, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def sheets(tiles: list[np.ndarray], out: Path, prefix: str, cols: int = 3, rows: int = 4):
    per = cols * rows
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        tw = tiles[0].shape[1]
        canvas = np.zeros((rows * TILE_H, cols * tw, 3), np.uint8)
        for k, t in enumerate(chunk):
            canvas[k // cols * TILE_H:(k // cols + 1) * TILE_H, k % cols * tw:(k % cols + 1) * tw] = t
        used = (len(chunk) + cols - 1) // cols
        cv2.imwrite(str(out / f'{prefix}_{s // per:02d}.jpg'), canvas[:used * TILE_H], [cv2.IMWRITE_JPEG_QUALITY, 88])


def predict(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, max_det=100)[0]
    if not len(r.boxes):
        return []
    return [(portrait(q), float(c)) for q, c in zip(r.keypoints.xy.numpy(), r.boxes.conf.numpy())]


def propose(args):
    from ultralytics import YOLO
    sys.path.insert(0, str(ROOT / 'scripts'))
    from review_real_detections import frames, source_of, TEST_SOURCES, VAL_SOURCES
    teachers = []
    for spec in args.teachers:
        w, _, size = spec.partition('@')
        teachers.append((YOLO(w), int(size or 1280)))
    manual = json.loads(args.manual.read_text()) if args.manual else {}
    out = args.output
    (out / 'sheets').mkdir(parents=True, exist_ok=True)
    (out / 'frames').mkdir(exist_ok=True)
    want = set(args.splits)
    records, tiles = [], []
    selected = [f for f in frames() if ('test' if source_of(f) in TEST_SOURCES else 'val' if source_of(f) in VAL_SOURCES else 'train') in want]
    if args.limit:
        selected = selected[:args.limit]
    pid = 0
    for fi, f in enumerate(selected):
        rel = str(f.relative_to(ROOT))
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        dets = []
        for model, size in teachers:
            dets += predict(model, img, size, args.conf)
        # Tiny boxes inside artwork are teacher noise; real cards here are at least min_diag of the frame.
        dets = [d for d in dets if diag(d[0]) >= args.min_diag * float(np.hypot(w, h))]
        # Fuse: cluster by IoU, confidence-weighted average of the corners.
        dets.sort(key=lambda d: -d[1])
        clusters = []
        for q, c in dets:
            for cl in clusters:
                if quad_iou(cl[0][0], q) > 0.6:
                    cl.append((q, c))
                    break
            else:
                clusters.append([(q, c)])
        cands = []
        for cl in clusters:
            ws = np.array([c for _, c in cl])
            q = (np.array([q for q, _ in cl]) * ws[:, None, None]).sum(0) / ws.sum()
            if len(cl) == 1 and ws.max() < args.single_conf:
                continue  # one weak teacher vote: almost always texture inside artwork
            cands.append({'quad': q, 'conf': float(ws.max()), 'votes': len(cl), 'origin': 'teacher'})
        for mq in manual.get(rel, []):
            cands.append({'quad': portrait(np.float64(mq)), 'conf': None, 'votes': 0, 'origin': 'manual_rough'})
        dx, dy = gradients(img)
        view = cv2.resize(img, (1600, int(h * 1600 / w)))
        sc = 1600 / w
        props = []
        for cnd in cands:
            q0 = cnd['quad']
            q1, sup = refine(dx, dy, q0)
            if args.refine_twice:
                q1, sup = refine(dx, dy, q1, search=0.015)
            q1 = portrait(q1)
            pid += 1
            name = f'p{pid:04d}'
            shift = float(np.max(np.linalg.norm(q1 - portrait(q0), axis=1)) / diag(q1))
            props.append({'id': name, 'corners': np.round(q1, 2).tolist(), 'teacher_corners': np.round(q0, 2).tolist(),
                          'conf': cnd['conf'], 'votes': cnd['votes'], 'origin': cnd['origin'], 'edge_support': sup,
                          'refine_shift': round(shift, 4)})
            tiles.append(tile(img, q1, f'{name} f{fi:02d} s{shift:.3f} e{min(sup):.2f}'))
            qq = np.int32(q1 * sc)
            cv2.polylines(view, [qq], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(view, name, tuple(int(v) for v in qq.mean(0) - [30, 0]), 0, .6, (0, 0, 0), 4)
            cv2.putText(view, name, tuple(int(v) for v in qq.mean(0) - [30, 0]), 0, .6, (0, 255, 255), 2)
        cv2.putText(view, f'f{fi:02d} {source_of(f)}', (10, 30), 0, .9, (0, 255, 255), 2)
        cv2.imwrite(str(out / 'frames' / f'f{fi:02d}.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        records.append({'id': f'f{fi:02d}', 'frame': rel, 'source': source_of(f), 'width': w, 'height': h,
                        'split': 'test' if source_of(f) in TEST_SOURCES else 'val' if source_of(f) in VAL_SOURCES else 'train',
                        'proposals': props})
        print(f'f{fi:02d} {len(props)} proposals', flush=True)
    sheets(tiles, out / 'sheets', 'sheet')
    write_props(out, args.teachers, args.conf, records)


def write_props(out, teachers, conf, records):
    (out / 'proposals.json').write_text(json.dumps({'teachers': teachers, 'conf': conf, 'frames': records}, indent=1))


def apply(args):
    """review.json: {"frames": {"f00": {"accept": [...], "coarse": [...], "ignore": [...]}}, "extra_coarse": {...}}.

    accept  corners verified on the zoomed corner patches (corner error and detection metrics)
    coarse  a complete card whose label is only box-accurate (detection metrics only)
    ignore  partially hidden/cut or ambiguous card: neither a miss nor a false detection
    Every other proposal is rejected. Frames listed in the review are exhaustively reviewed:
    every complete card is accepted, coarse, or added as an extra coarse quad.
    """
    props = json.loads((args.proposals).read_text())
    review = json.loads(args.review.read_text())
    out = []
    for fr in props['frames']:
        rv = review['frames'].get(fr['id'])
        if rv is None:
            continue
        by = {p['id']: p for p in fr['proposals']}
        cards = [{'corners': by[i]['corners'], 'precise': True, 'proposal': i} for i in rv.get('accept', [])]
        cards += [{'corners': by[i]['corners'], 'precise': False, 'proposal': i} for i in rv.get('coarse', [])]
        cards += [{'corners': q, 'precise': False, 'proposal': 'manual'} for q in review.get('extra_coarse', {}).get(fr['id'], [])]
        # Corners read by eye from gridded full-resolution corner zooms, then checked on the patches.
        cards += [{'corners': portrait(np.float64(q)).tolist(), 'precise': True, 'proposal': 'manual'}
                  for q in review.get('extra_precise', {}).get(fr['id'], [])]
        ignore = [by[i]['corners'] for i in rv.get('ignore', [])]
        out.append({'id': fr['id'], 'frame': fr['frame'], 'source': fr['source'], 'split': fr['split'],
                    'width': fr['width'], 'height': fr['height'], 'cards': cards, 'ignore': ignore,
                    # build_corner_dataset --real compatibility: precise corners are trained, the rest masked.
                    'corners': [c['corners'] for c in cards if c['precise']],
                    'mask': ignore + [c['corners'] for c in cards if not c['precise']]})
    meta = {'schema': 'real_corner_labels/1', 'corner_order': 'TL,TR,BR,BL upright portrait (geometric)',
            'coordinates': 'source frame pixels',
            'label_origin': review.get('label_origin', 'teacher proposals + edge-line refinement + AI visual review (not human ground truth)'),
            'reviewer': review.get('reviewer'), 'date': review.get('date'), 'teachers': props['teachers'],
            'counts': {'frames': len(out), 'cards': sum(len(f['cards']) for f in out),
                       'precise': sum(len(f['corners']) for f in out), 'ignore': sum(len(f['ignore']) for f in out)},
            'frames': out}
    args.output.write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta['counts']))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('propose')
    p.add_argument('--teachers', nargs='+', required=True, help='weights.pt[@imgsz]')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['test', 'val'], choices=['train', 'val', 'test'])
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--manual', type=Path, help='JSON {frame_rel_path: [rough quads]} for cards all teachers miss')
    p.add_argument('--min-diag', type=float, default=0.06, help='drop teacher quads smaller than this share of the frame diagonal')
    p.add_argument('--single-conf', type=float, default=0.5, help='minimum confidence for a proposal only one teacher made')
    p.add_argument('--refine-twice', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    a = sub.add_parser('apply')
    a.add_argument('--proposals', type=Path, required=True)
    a.add_argument('--review', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    propose(args) if args.cmd == 'propose' else apply(args)


if __name__ == '__main__':
    main()
