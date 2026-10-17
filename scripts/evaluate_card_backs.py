#!/usr/bin/env python3
"""Evaluate card-back gallery rows: are backs recognized, and do real fronts stay unaffected?

Runs the browser bundle's search in Python (fp32 ONNX encoder, tcg-full-card-v1 preprocessing,
gallery.f16 rows of the bundle languages plus the card-back rows from build_browser_bundle.py):

1. Real card fronts: every curated real query crop (train/val/test). Reports how many would
   become a back (top-1 back, and back decision at the thresholds), the distribution of their best
   back similarity, and whether adding back rows changes any top-1.
2. All card reference rows in the gallery: their best similarity to a back row (extra negatives;
   digital references, not camera crops).
3. Back positives: optional real back crops (``--real-backs``, held out of the gallery) and
   SYNTHETIC photo-like variants (perspective, blur, exposure/colour, glare, sleeve sheen, JPEG,
   corner noise) of every back reference and real back crop. Variants of a gallery reference are
   scored leave-one-reference-out when the game has another reference; otherwise they are marked
   as not independent.

Scores are cosine similarities, not probabilities. Thresholds picked from these numbers are
uncalibrated pilot values: the positives are few, partly synthetic, and not phone captures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from build_browser_bundle import embed_back_references  # noqa: E402
from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input  # noqa: E402

CARD_W, CARD_H = 630, 880


def family(identity: str) -> str:
    game, _lang, card_id = identity.split(':', 2) if identity.count(':') >= 2 else (identity, '', identity)
    return f'{game}:{card_id}' if game in ('onepiece', 'pokemon') else identity


def encode(sess, rgb: np.ndarray, corners=None) -> np.ndarray:
    v = sess.run(None, {'pixels': prepare_encoder_input(canonicalize_card(rgb, corners))[None]})[0][0].astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def synthetic_view(card: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """A photo-like view of a flat card image: returns (scene RGB, noisy corner estimate, parameters)."""
    card = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    W, H = 1400, 1400
    # Background: binder-like dark/grey/coloured plane with noise.
    base = rng.integers(20, 200, size=3)
    scene = np.clip(base + rng.normal(0, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    scale = rng.uniform(0.45, 0.85)
    cw, ch = CARD_W * scale, CARD_H * scale
    cx, cy = W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-150, 150)
    angle = np.deg2rad(rng.uniform(-25, 25) + (180 if rng.random() < 0.1 else 0))
    rect = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = rect @ rot.T + [cx, cy]
    dst += rng.normal(0, 0.04 * cw, dst.shape)  # perspective
    src = np.float32([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]])
    M = cv2.getPerspectiveTransform(src, np.float32(dst))
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((CARD_H, CARD_W), 255, np.uint8), M, (W, H))
    scene = np.where(mask[..., None] > 127, warped, scene).astype(np.float32)
    params = {'scale': round(scale, 3), 'angle_deg': round(float(np.rad2deg(angle)), 1)}
    # Exposure / white balance.
    gain = rng.uniform(0.6, 1.3)
    tint = rng.uniform(0.85, 1.15, 3)
    scene = scene * gain * tint
    params.update(gain=round(gain, 2))
    yy, xx = np.mgrid[0:H, 0:W]
    if rng.random() < 0.6:  # specular glare blob
        gx, gy = dst.mean(0) + rng.uniform(-0.4, 0.4, 2) * [cw, ch]
        sx, sy = rng.uniform(0.08, 0.3) * cw, rng.uniform(0.05, 0.2) * ch
        strength = rng.uniform(80, 230)
        scene += strength * np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))[..., None]
        params.update(glare=round(strength))
    if rng.random() < 0.6:  # sleeve / pocket sheen: low-contrast veil plus a diagonal band
        veil = rng.uniform(0.05, 0.25)
        scene = scene * (1 - veil) + 235 * veil
        t = (xx * np.cos(0.7) + yy * np.sin(0.7)) / W
        band = np.exp(-((t - rng.uniform(0.3, 1.2)) / 0.05) ** 2) * rng.uniform(20, 90)
        scene += band[..., None]
        params.update(sleeve=round(veil, 2))
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    k = int(rng.choice([0, 0, 3, 5]))
    if k:
        scene = cv2.GaussianBlur(scene, (k, k), 0)
        params.update(blur=k)
    q = int(rng.integers(45, 90))
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(scene, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    scene = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    params.update(jpeg=q)
    # Detector-like corner error (about 1.5% of the card width).
    corners = np.float32(dst + rng.normal(0, 0.015 * cw, dst.shape))
    if angle > np.pi / 2:  # upside-down card: the detector's portrait order starts at the visual top
        corners = np.roll(corners, 2, axis=0)
    return scene, corners, params


def summary(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {'n': 0}
    return {'n': int(a.size), 'min': round(float(a.min()), 4), 'p05': round(float(np.percentile(a, 5)), 4),
            'median': round(float(np.median(a)), 4), 'p95': round(float(np.percentile(a, 95)), 4),
            'max': round(float(a.max()), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', type=Path, default=ROOT / 'runs/browser_export/gpu-20261017')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--backs', type=Path, help='card-back manifest (default <dataset>/recognition/backs/manifest.json)')
    parser.add_argument('--real-backs', nargs='*', default=[],
                        help='held-out real back crops: <game>=<dir of full-card crops> (not gallery references)')
    parser.add_argument('--languages', default='en,de')
    parser.add_argument('--variants', type=int, default=24, help='synthetic variants per back image')
    parser.add_argument('--min-score', type=float, default=0.65)
    parser.add_argument('--min-margin', type=float, default=0.1)
    parser.add_argument('--samples', type=Path, help='write a contact sheet of a few synthetic variants here')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    backs_manifest = args.backs or args.dataset / 'recognition/backs/manifest.json'
    meta = json.loads((args.export / 'gallery.json').read_text())
    d = meta['dimension']
    languages = set(args.languages.split(','))
    keep = [i for i, ident in enumerate(meta['identities']) if ident.split(':')[1] in languages]
    G_cards = np.fromfile(args.export / 'gallery.f16', dtype='<f2').reshape(-1, d)[keep].astype(np.float32)
    card_ids = [meta['identities'][i] for i in keep]
    back_rows, back_ids, _, info = embed_back_references(backs_manifest, args.dataset, args.export / 'encoder.onnx', d)
    # Same storage precision as the bundle.
    B = back_rows.astype('<f2').astype(np.float32)
    back_manifest = json.loads(backs_manifest.read_text())['references']
    sess = ort.InferenceSession(str(args.export / 'encoder.onnx'), providers=['CPUExecutionProvider'])

    def decide(v: np.ndarray, back_mask: np.ndarray | None = None) -> dict:
        Bm = B if back_mask is None else B[back_mask]
        ids_b = back_ids if back_mask is None else [b for b, m in zip(back_ids, back_mask) if m]
        sc, sb = G_cards @ v, Bm @ v
        ci, bi = int(np.argmax(sc)), int(np.argmax(sb)) if len(sb) else -1
        best_back = float(sb[bi]) if len(sb) else -1.0
        top_is_back = best_back > float(sc[ci])
        margin = best_back - float(sc[ci])
        return {'card_top': card_ids[ci], 'card_score': float(sc[ci]), 'back_top': ids_b[bi] if len(sb) else None,
                'back_score': best_back, 'top_is_back': top_is_back,
                'back_decision': top_is_back and best_back >= args.min_score and margin >= args.min_margin}

    # 1. Real fronts.
    fronts = []
    for split in ('train', 'val', 'test'):
        for png in sorted((args.dataset / 'recognition' / split).rglob('*.png')):
            game, lang, card_id = png.relative_to(args.dataset / 'recognition' / split).parts[:3]
            v = encode(sess, np.asarray(Image.open(png).convert('RGB')))
            r = decide(v)
            r.update(split=split, identity=f'{game}:{lang}:{card_id}', path=str(png.relative_to(args.dataset)))
            fronts.append(r)
    front_report = {
        'queries': len(fronts), 'by_split': {s: sum(f['split'] == s for f in fronts) for s in ('train', 'val', 'test')},
        'top1_is_back': sum(f['top_is_back'] for f in fronts),
        'back_decisions': sum(f['back_decision'] for f in fronts),
        'best_back_score': summary([f['back_score'] for f in fronts]),
        'best_back_score_by_game': {g: summary([f['back_score'] for f in fronts if f['identity'].startswith(g)]) for g in ('onepiece', 'pokemon')},
        'top1_changed_by_back_rows': sum(f['top_is_back'] for f in fronts),
        'test_top1_changed': sum(f['top_is_back'] for f in fronts if f['split'] == 'test'),
        'test_queries': sum(f['split'] == 'test' for f in fronts),
        'top1_exact_correct_cards_only': sum(f['card_top'] == f['identity'] for f in fronts),
        'top1_family_correct_cards_only': sum(family(f['card_top']) == family(f['identity']) for f in fronts),
        'closest_to_back': sorted(({k: f[k] for k in ('path', 'identity', 'back_top', 'back_score', 'card_score')} for f in fronts),
                                  key=lambda f: -f['back_score'])[:5],
    }
    # 2. Every card reference row vs. back rows.
    ref_back = (G_cards @ B.T).max(1)
    ref_report = {'references': len(card_ids), 'best_back_score': summary(ref_back),
                  'above_min_score': int((ref_back >= args.min_score).sum()),
                  'closest': [{'identity': card_ids[i], 'back_score': round(float(ref_back[i]), 4)} for i in np.argsort(-ref_back)[:5]]}

    # 3. Back positives.
    rng = np.random.default_rng(20261017)
    positives = []
    sheet = []
    images = [(r['game'], args.dataset / r['image'], i, 'gallery-reference') for i, r in enumerate(back_manifest)]
    for spec in args.real_backs:
        game, folder = spec.split('=', 1)
        for f in sorted(Path(folder).glob('*.png')):
            images.append((game, f, None, 'real-heldout'))
    for game, path, ref_index, role in images:
        rgb = np.asarray(Image.open(path).convert('RGB'))
        mask = np.ones(len(back_ids), bool)
        independent = True
        if ref_index is not None:
            same_game = [j for j, r in enumerate(back_manifest) if r['game'] == game and j != ref_index]
            if same_game:
                mask[ref_index] = False
            else:
                independent = False  # only reference of its game: its own row stays in the gallery
        if role == 'real-heldout':
            r = decide(encode(sess, rgb), mask)
            positives.append({'game': game, 'source': str(path), 'role': role, 'synthetic': False, 'independent': True, **r})
        for k in range(args.variants):
            scene, corners, params = synthetic_view(rgb, rng)
            try:
                v = encode(sess, scene, corners)
            except ValueError:
                continue
            r = decide(v, mask)
            positives.append({'game': game, 'source': str(path), 'role': role + '+synthetic-view', 'synthetic': True,
                              'independent': independent, 'params': params, **r})
            if len(sheet) < 12 and k < 2:
                sheet.append(canonicalize_card(scene, corners))

    def rate(rows):
        n = len(rows)
        return {'n': n, 'back_decision': sum(r['back_decision'] for r in rows),
                'top1_back': sum(r['top_is_back'] for r in rows),
                'right_game': sum(r['back_decision'] and r['back_top'] == f"back:{r['game']}" for r in rows),
                'back_score': summary([r['back_score'] for r in rows]),
                'margin_to_best_card': summary([r['back_score'] - r['card_score'] for r in rows])}
    groups = {}
    for r in positives:
        key = f"{r['game']}|{r['role']}|{'independent' if r['independent'] else 'own-reference-in-gallery'}"
        groups.setdefault(key, []).append(r)
    report = {
        'thresholds': {'min_score': args.min_score, 'min_margin': args.min_margin, 'calibrated': False},
        'gallery': {'card_rows': len(card_ids), 'languages': sorted(languages), 'back_rows': len(back_ids), 'backs': info},
        'real_fronts': front_report,
        'card_reference_rows': ref_report,
        'back_positives': {k: rate(v) for k, v in sorted(groups.items())},
        'misses': [{k: r.get(k) for k in ('game', 'source', 'role', 'params', 'back_top', 'back_score', 'card_top', 'card_score')}
                   for r in positives if not r['back_decision']][:20],
    }
    if args.samples and sheet:
        Image.fromarray(np.hstack(sheet[:12])).save(args.samples)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
