#!/usr/bin/env python3
"""Acquire card-back reference images and write the curated back-reference manifest.

Backs are enrolled as special gallery rows (identity ``back:<game>``) so the normal similarity
search can recognize a card shown face down and keep it out of matches, counts and values.

Every source is pinned by URL and SHA-256; originals are kept byte-for-byte next to the derived
full-card references. Geometry is either the verified full-image bounds of a flat scan, or the
outline of an opaque card in an official illustration with transparency (corners fitted to its
alpha edges). Nothing here is synthetic; photo-like variants for evaluation are generated in
``scripts/release/evaluate_card_backs.py`` and labelled synthetic there.

Image rights: the images are copyrighted card designs (The Pokémon Company / Bandai) used here as
local recognition references only. They are not redistributed by this repository (datasets/ is
DVC-tracked, not committed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.recognition.canonical import PREPROCESSING_VERSION, canonicalize_card  # noqa: E402

CARD_ASPECT = 88 / 63  # height / width of a standard 63 x 88 mm card
USER_AGENT = 'tcg-scanner-research/0.1'

WIKI_FAIR_USE = ('Non-free image hosted on English Wikipedia under a fair-use rationale; copyright of the card design '
                 'belongs to its publisher. Used locally as a recognition reference only; not redistributed.')
IA_TERMS = ('Internet Archive community upload (collection opensource_image); no license stated by the uploader. '
            'Card design © The Pokémon Company. Used locally as recognition reference/evaluation only; not redistributed.')
SOURCES = [
    {
        'game': 'pokemon', 'label': 'Pokémon TCG back (international; English/German prints share it)',
        'url': 'https://upload.wikimedia.org/wikipedia/en/3/36/Pokemon_Trading_Card_Game_cardback.png',
        'page': 'https://en.wikipedia.org/wiki/File:Pokemon_Trading_Card_Game_cardback.png',
        'sha256': '66eaaa1c4835c185916d10a4a0774c79af175df228ed2acb0ce500fab1aa6488',
        'role': 'flat-scan', 'geometry': 'full-image',
        'license': WIKI_FAIR_USE,
        'attribution': 'The Pokémon Company; Wikipedia states the immediate source as Internet Archive item '
                       '150000cardlot_batch14 (a scan of a physical card).',
    },
    {
        'game': 'pokemon', 'label': 'Pokémon TCG backs: flatbed scan of 16 physical (vintage-era) cards',
        'url': 'https://archive.org/download/150000cardlot_batch14/1/-1-2.jpg',
        'page': 'https://archive.org/details/150000cardlot_batch14',
        'sha256': '75c1e2a523c5c88d29147486724728e3e65db31a4cf52b1e1aea6a5c0bf8c9d7',
        'role': 'real-scan', 'geometry': 'scan-contours',
        # Two physical cards become gallery references; the rest of this scan is held out for evaluation.
        'gallery_crops': [0, 5],
        'heldout_note': 'same scan session as two gallery references (different physical cards)',
        'license': IA_TERMS,
        'attribution': 'The Collectosaurus Project, Internet Archive item 150000cardlot_batch14 (scan set 1, back side)',
    },
    {
        'game': 'onepiece', 'label': 'One Piece Card Game back (blue main-deck back)',
        'url': 'https://upload.wikimedia.org/wikipedia/en/3/32/One_Piece_Card_Game_back.webp',
        'page': 'https://en.wikipedia.org/wiki/File:One_Piece_Card_Game_back.webp',
        'sha256': 'f5ac820b3efc790029d62186828d6b0924e0c119c9fc0792acb771a983650d1b',
        'role': 'flat-digital', 'geometry': 'full-image',
        'license': WIKI_FAIR_USE,
        'attribution': 'Bandai; Wikipedia credits https://boardgamegeek.com/image/6974116/one-piece-card-game',
    },
    {
        'game': 'onepiece', 'label': 'One Piece Card Game back, tilted illustration on the official site footer',
        'url': 'https://en.onepiece-cardgame.com/renewal/images/common/pc/footer_illust_card01.png',
        'page': 'https://en.onepiece-cardgame.com/rules/',
        'sha256': '2724f2d3a758d6b6c563f773d6b761423fa1d15f2b29280455e8d3e28f1d53e8',
        'role': 'official-illustration', 'geometry': 'alpha-outline', 'component': 'largest-blue',
        'license': 'Official Bandai website asset (© Eiichiro Oda/Shueisha, Toei Animation; Bandai). No reuse license '
                   'stated; used locally as a recognition reference only; not redistributed.',
        'attribution': 'Bandai, ONE PIECE CARD GAME official English website',
    },
    {
        'game': 'onepiece', 'label': 'One Piece Card Game back, second official footer illustration (mobile layout)',
        'url': 'https://en.onepiece-cardgame.com/renewal/images/common/sp/footer_illust_card.webp',
        'page': 'https://en.onepiece-cardgame.com/rules/',
        'sha256': 'a7b6f3589912f25f4998c46841eaff4ee0b4f08617b1501d86ca08e5d826f1b9',
        'role': 'official-illustration', 'geometry': 'alpha-outline', 'component': 'largest-blue',
        'license': 'Official Bandai website asset (© Eiichiro Oda/Shueisha, Toei Animation; Bandai). No reuse license '
                   'stated; used locally as a recognition reference only; not redistributed.',
        'attribution': 'Bandai, ONE PIECE CARD GAME official English website',
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flatten_alpha(rgba: np.ndarray, background: int = 128) -> np.ndarray:
    """RGB uint8 from RGBA over a neutral background (only the antialiased card edge is affected)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + background * (1 - a)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _intersect(l1, l2) -> np.ndarray:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], dtype=np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


def order_portrait(quad: np.ndarray) -> np.ndarray:
    """TL,TR,BR,BL for an upright portrait card: start at the top-most corner pair's left point,
    with the short edge on top (illustrations here are tilted < 45 degrees)."""
    q = np.asarray(quad, dtype=np.float64)
    centre = q.mean(0)
    angles = np.arctan2(q[:, 1] - centre[1], q[:, 0] - centre[0])
    q = q[np.argsort(angles)]  # clockwise in image coordinates (y down), starting near -pi
    # Choose the start so that edge 0 (TL->TR) is a short edge and lies on the upper side.
    best = None
    for s in range(4):
        r = np.roll(q, -s, axis=0)
        top, side = np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[2] - r[1])
        score = (top < side, -(r[0, 1] + r[1, 1]))
        if best is None or score > best[0]:
            best = (score, r)
    return best[1]


def alpha_card_quad(rgba: np.ndarray, component: str | int = 'largest-blue') -> np.ndarray:
    """Physical-edge corners of an opaque card in an RGBA illustration: straight-line fits to the
    alpha contour between rounded corners, intersected. Returns TL,TR,BR,BL float32."""
    mask = (rgba[..., 3] > 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = [i for i in range(1, n) if stats[i, 4] >= 2000]
    if component == 'largest-blue':
        def blueness(i):
            px = rgba[labels == i][:, :3].astype(np.float32)
            return (px[:, 2] - px[:, 0]).mean()
        blue = [i for i in comps if blueness(i) > 20]
        if not blue:
            raise ValueError('no blue card component')
        idx = max(blue, key=lambda i: stats[i, 4])
    else:
        idx = comps[int(component)]
    m = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    hull = cv2.convexHull(c.astype(np.float32))
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float64)
    if len(approx) != 4:
        raise ValueError(f'expected a quadrilateral outline, got {len(approx)} vertices')
    lines = []
    for k in range(4):
        p, q = approx[k], approx[(k + 1) % 4]
        d = q - p
        length = np.linalg.norm(d)
        u = d / length
        rel = c - p
        t = rel @ u
        dist = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        # Middle 70% of each side: excludes rounded corners.
        keep = (t > 0.15 * length) & (t < 0.85 * length) & (dist < 6)
        if keep.sum() < 10:
            raise ValueError('too few edge points for a line fit')
        vx, vy, x0, y0 = cv2.fitLine(c[keep].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((vx, vy, x0, y0))
    corners = np.array([_intersect(lines[k - 1], lines[k]) for k in range(4)])
    return order_portrait(corners).astype(np.float32)


def scan_card_quads(bgr: np.ndarray) -> list[np.ndarray]:
    """Upright cards on the black background of a flatbed scan: contour + minimum-area rectangle.

    A flatbed scan has no perspective, so the rectangle is the physical card edge (rounded corners
    excepted). Returns TL,TR,BR,BL quads in reading order; cards touching neighbours are skipped.
    """
    mask = ((bgr[..., 0] > 70) | (bgr.max(2) > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * bgr.shape[0] * bgr.shape[1]:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if not 1.3 < max(w, h) / min(w, h) < 1.5 or area / (w * h) < 0.97:
            continue  # not a single clean card outline
        quads.append(order_portrait(cv2.boxPoints(rect)).astype(np.float32))
    rows = sorted(quads, key=lambda q: q[:, 1].mean())
    # Reading order: group by row (centres within half a card height), then left to right.
    out, row = [], []
    for q in rows:
        if row and abs(q[:, 1].mean() - np.mean([r[:, 1].mean() for r in row])) > 0.5 * np.ptp(q[:, 1]):
            out += sorted(row, key=lambda r: r[:, 0].mean())
            row = []
        row.append(q)
    return out + sorted(row, key=lambda r: r[:, 0].mean())


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    return urllib.request.urlopen(req, timeout=120).read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--cache', type=Path, help='directory with already downloaded originals (matched by SHA-256)')
    args = parser.parse_args()
    out_root = args.dataset / 'recognition/backs'
    cached = {}
    if args.cache:
        for f in args.cache.rglob('*'):
            if f.is_file():
                cached[sha256_bytes(f.read_bytes())] = f
    references, heldout = [], []

    def save_png(rgb: np.ndarray, folder: Path, prefix: str) -> tuple[Path, str]:
        folder.mkdir(parents=True, exist_ok=True)
        buf = folder / 'tmp.png'
        Image.fromarray(rgb).save(buf, optimize=True)
        sha = sha256_bytes(buf.read_bytes())
        target = folder / f'{prefix}-{sha[:12]}.png'
        buf.replace(target)
        return target, sha

    for src in SOURCES:
        data = cached[src['sha256']].read_bytes() if src['sha256'] in cached else fetch(src['url'])
        if sha256_bytes(data) != src['sha256']:
            raise ValueError(f"{src['url']} changed: expected {src['sha256']}, got {sha256_bytes(data)}")
        game_dir = out_root / src['game']
        (game_dir / 'originals').mkdir(parents=True, exist_ok=True)
        original = game_dir / 'originals' / f"{src['sha256'][:12]}-{src['url'].rsplit('/', 1)[1].lstrip('-')}"
        original.write_bytes(data)
        image = Image.open(original)
        source = {'url': src['url'], 'page': src['page'], 'sha256': src['sha256'],
                  'original': str(original.relative_to(args.dataset)), 'bytes': len(data),
                  'license': src['license'], 'attribution': src['attribution']}
        crops: list[tuple[np.ndarray, dict]] = []
        if src['geometry'] == 'full-image':
            # Verified flat full-card bounds: stored as is (converted to an RGB PNG).
            crops.append((np.asarray(image.convert('RGB')), {'method': 'full-image'}))
        elif src['geometry'] == 'alpha-outline':
            rgba = np.asarray(image.convert('RGBA'))
            quad = alpha_card_quad(rgba, src.get('component', 'largest-blue'))
            width = int(round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))))
            crops.append((canonicalize_card(flatten_alpha(rgba), quad, width, int(round(width * CARD_ASPECT))), {
                'method': 'alpha-outline', 'corners_tl_tr_br_bl': quad.round(2).tolist(), 'coordinate_space': 'original image pixels',
                'warp': f'{PREPROCESSING_VERSION} canonicalize_card to {width}x{round(width * CARD_ASPECT)}', 'alpha_background': 128}))
        else:  # scan-contours: several physical cards on one flatbed scan
            rgb = np.asarray(image.convert('RGB'))
            for k, quad in enumerate(scan_card_quads(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))):
                width = int(round(np.linalg.norm(quad[1] - quad[0])))
                crops.append((canonicalize_card(rgb, quad, width, int(round(width * CARD_ASPECT))), {
                    'method': 'scan-contours', 'index_in_reading_order': k, 'corners_tl_tr_br_bl': quad.round(1).tolist(),
                    'coordinate_space': 'original image pixels', 'warp': f'{PREPROCESSING_VERSION} canonicalize_card'}))
        gallery_crops = src.get('gallery_crops')
        for k, (crop, geometry) in enumerate(crops):
            enrol = gallery_crops is None or k in gallery_crops
            folder = game_dir if enrol else game_dir / 'heldout'
            target, sha = save_png(crop, folder, 'reference' if enrol else 'query')
            row = {'identity': f"back:{src['game']}", 'kind': 'back', 'game': src['game'], 'label': src['label'],
                   'image': str(target.relative_to(args.dataset)), 'sha256': sha,
                   'width': int(crop.shape[1]), 'height': int(crop.shape[0]),
                   'role': src['role'], 'synthetic': False, 'geometry': geometry, 'source': source}
            if enrol:
                references.append(row)
            else:
                row['split_note'] = src.get('heldout_note', '')
                heldout.append(row)
        print(json.dumps({'source': src['url'], 'crops': len(crops), 'gallery': sum(1 for k in range(len(crops)) if gallery_crops is None or k in gallery_crops)}))
    manifest = {
        'schema_version': 1, 'artifact_type': 'card_back_references',
        'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'generator': 'scripts/catalog/acquire_card_backs.py',
        'gallery_adapter': 'known full-card bounds of each reference image (no detector), then the shared encoder contract',
        'identities': {
            'back:pokemon': 'Pokémon TCG international back (English, German and other non-Asian prints). '
                            'The Japanese/Asian back differs and has no reference yet.',
            'back:onepiece': 'One Piece Card Game blue main-deck back. DON!! card backs differ and have no reference.',
        },
        'notes': 'No real camera views of backs were found in the local video mining frames; references are flat '
                 'scans/digital images and official illustrations only. Synthetic photo-like variants for evaluation '
                 'are produced on the fly by scripts/release/evaluate_card_backs.py and are not stored here.',
        'references': references,
        # Real back crops NOT enrolled in the gallery, for evaluation only.
        'heldout_queries': heldout,
    }
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'manifest': str(out_root / 'manifest.json'), 'references': len(references), 'heldout_queries': len(heldout)}))


if __name__ == '__main__':
    main()
