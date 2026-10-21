#!/usr/bin/env python3
"""Enroll the synced Yu-Gi-Oh! catalog as gallery references for the browser bundle.

Known-gallery-bounds adapter: each YGOPRODeck artwork image is a verified full card, so it goes
straight to the shared canonicalizer and the fp32 export encoder (no detector, no augmentation).

Writes
- `datasets/curated/recognition/gallery/yugioh/en/<artwork_id>/reference-<sha12>.<ext>` (hard
  links to the source images) with the usual sidecar,
- `datasets/curated/recognition/enrolled/yugioh.json`: one row per artwork (identity
  `yugioh:en:<artwork_id>`, family, name, passcode) and each family's TCG printings
  (`yugioh:en:<set code>:<rarity code>`, with the YGOPRODeck set price in USD),
- `<export>/enrolled/yugioh.f16` + `yugioh.json`: L2-normalized embeddings in row order, with the
  encoder and enrollment hashes the bundle builder checks.

Artworks whose image is a card back (provider placeholders for unreleased art) are left out: their
similarity to the `back:yugioh` reference is at least BACK_LIKE; they are listed in the enrollment.

A family is one card (passcode) with all its artworks, merged with any other card whose image
bytes are identical. YGOPRODeck does not say which alternate artwork a set printing uses, so every
printing of a card belongs to the whole family and the artwork-to-printing mapping stays open.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.recognition.backbone import sha256_file  # noqa: E402

SOURCE = ROOT / 'datasets/sources/catalog/yugioh-en-ygoprodeck'
CURATED = ROOT / 'datasets/curated'
EXPORT = ROOT / 'models/browser-export/gpu-20261017'
BACKS = CURATED / 'recognition/backs/manifest.json'
BACK_LIKE = 0.8


def printing_id(card: dict) -> str | None:
    code = (card.get('set_code') or '').strip()
    rarity = re.sub(r'[^A-Za-z0-9+]', '', card.get('rarity_code') or '') or re.sub(r'[^A-Za-z0-9]', '', card.get('rarity') or '')
    if not code or not rarity or ':' in code:
        return None
    return f'yugioh:en:{code}:{rarity}'


def price(value) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return round(amount, 2) if amount > 0 else None


def families(records: list[dict]) -> dict[int, str]:
    """Family key per artwork: its passcode, merged across byte-identical images."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_sha: dict[str, str] = {}
    for r in records:
        card = f"yugioh:card:{r['card_id']}"
        find(card)
        other = by_sha.setdefault(r['image_sha256'], card)
        if other != card:
            parent[find(card)] = find(other)
    return {r['artwork_id']: find(f"yugioh:card:{r['card_id']}") for r in records}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256_file(dst) == sha256_file(src):
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def embed(paths: list[Path], encoder: Path, reuse: Path | None, source_manifest_sha: str) -> np.ndarray:
    if reuse:
        index = json.loads((reuse / 'gallery_index.json').read_text())
        if index['encoder_sha256'] != sha256_file(encoder) or index['catalog_manifest_sha256'] != source_manifest_sha:
            raise ValueError(f'{reuse} was embedded from another encoder or catalog manifest')
        return np.load(reuse / 'gallery.npy'), [r['artwork_id'] for r in index['rows']]
    import onnxruntime as ort
    from PIL import Image
    from training.recognition.canonical import canonicalize_card, prepare_encoder_input
    sess = ort.InferenceSession(str(encoder), providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    rows = []
    for i, path in enumerate(paths, 1):
        rgb = np.asarray(Image.open(path).convert('RGB'))
        rows.append(sess.run(None, {name: prepare_encoder_input(canonicalize_card(rgb))[None]})[0][0])
        if i % 1000 == 0:
            print(f'{i}/{len(paths)}', flush=True)
    return np.stack(rows), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--export', type=Path, default=EXPORT)
    ap.add_argument('--reuse-embeddings', type=Path, help='output of zero_shot_retrieval.py embed for the same catalog manifest and encoder')
    args = ap.parse_args()
    source_manifest = args.source / 'manifest.json'
    manifest = json.loads(source_manifest.read_text())
    records = sorted((r for r in manifest['cards'] if r.get('status') == 'downloaded'), key=lambda r: r['artwork_id'])
    family = families(records)

    rows, printings, images = [], {}, []
    for r in records:
        src = args.source / r['image_path']
        if sha256_file(src) != r['image_sha256']:
            raise ValueError(f'{src} differs from its manifest hash')
        rel = Path('recognition/gallery/yugioh/en') / str(r['artwork_id']) / f"reference-{r['image_sha256'][:12]}{src.suffix}"
        link(src, CURATED / rel)
        (CURATED / rel).with_name((CURATED / rel).name + '.json').write_text(json.dumps({
            'artifact_type': 'recognition_gallery_reference',
            'identity': {'card_id': str(r['artwork_id']), 'game': 'yugioh', 'language': 'en'},
            'source_path': str(src.relative_to(ROOT)),
            'source_record': {'provider': 'YGOPRODeck', 'name': r['name'], 'passcode': r['card_id'], 'artwork_id': r['artwork_id'],
                              'alternate_artwork': r['alternate_artwork'], 'source_url': r['source_image_url'],
                              'image_role': 'provider digital reference', 'sha256': r['image_sha256']},
            'source_sha256': r['image_sha256'],
        }, indent=2, ensure_ascii=False) + '\n')
        fam = family[r['artwork_id']]
        rows.append({'identity': f"yugioh:en:{r['artwork_id']}", 'family': fam, 'name': r['name'], 'passcode': f"{r['card_id']:08d}",
                     'frame_type': r['frame_type'], 'image': str(rel), 'sha256': r['image_sha256']})
        images.append(src)
        for p in r['tcg_printings']:
            pid = printing_id(p)
            if pid and pid not in printings:
                printings[pid] = {'id': pid, 'family': fam, 'name': r['name'], 'set_code': p['set_code'], 'set_name': p['set_name'],
                                  'rarity': p['rarity'], 'price_usd': price(p.get('price_usd'))}

    vectors, order = embed(images, args.export / 'encoder.onnx', args.reuse_embeddings, sha256_file(source_manifest))
    if order is not None and order != [r['artwork_id'] for r in records]:
        lookup = {a: i for i, a in enumerate(order)}
        vectors = vectors[[lookup[r['artwork_id']] for r in records]]
    vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.shape != (len(rows), 384) or not np.isfinite(vectors).all() or (norms < 1e-6).any():
        raise ValueError('invalid embeddings')
    vectors = vectors / norms
    backs = [r for r in json.loads(BACKS.read_text())['references'] if r['identity'] == 'back:yugioh']
    if not backs:
        raise ValueError(f'{BACKS} has no back:yugioh reference (scripts/catalog/acquire_card_backs.py)')
    back_vectors, _ = embed([CURATED / b['image'] for b in backs], args.export / 'encoder.onnx', None, '')
    back_vectors = back_vectors / np.linalg.norm(back_vectors, axis=1, keepdims=True)
    back_sim = (vectors @ back_vectors.T).max(axis=1)
    keep = back_sim < BACK_LIKE
    excluded = [{**rows[i], 'back_similarity': round(float(back_sim[i]), 3)} for i in np.where(~keep)[0]]
    rows = [r for r, k in zip(rows, keep) if k]
    vectors = vectors[keep]
    families_kept = {r['family'] for r in rows}
    printings = {k: p for k, p in printings.items() if p['family'] in families_kept}

    enrolled = CURATED / 'recognition/enrolled/yugioh.json'
    enrolled.parent.mkdir(parents=True, exist_ok=True)
    fetched = datetime.fromtimestamp((args.source / 'cardinfo.json').stat().st_mtime, timezone.utc).date().isoformat()
    enrolled.write_text(json.dumps({
        'schema_version': 1, 'artifact_type': 'enrolled_gallery', 'game': 'yugioh', 'language': 'en',
        'source_manifest': str(source_manifest.relative_to(ROOT)), 'source_manifest_sha256': sha256_file(source_manifest),
        'gallery_adapter': 'known full-card bounds of each reference image (no detector), then the shared encoder contract',
        'family_policy': 'one card (passcode) with all its artworks, merged across byte-identical images; printings belong to the whole family',
        'price_source': 'YGOPRODeck set price (USD); zero or missing prices are unknown', 'price_date': fetched,
        'excluded_back_like': excluded, 'back_like_threshold': BACK_LIKE,
        'artworks': rows, 'printings': sorted(printings.values(), key=lambda p: p['id']),
    }, ensure_ascii=False, separators=(',', ':')) + '\n')

    out = args.export / 'enrolled'
    out.mkdir(parents=True, exist_ok=True)
    vectors.astype('<f2').tofile(out / 'yugioh.f16')
    (out / 'yugioh.json').write_text(json.dumps({
        'schema_version': 1, 'game': 'yugioh', 'rows': len(rows), 'dimension': 384, 'dtype': 'float16-le',
        'identities': [r['identity'] for r in rows],
        'encoder_sha256': sha256_file(args.export / 'encoder.onnx'),
        'enrollment': str(enrolled.relative_to(ROOT)), 'enrollment_sha256': sha256_file(enrolled),
        'vectors_sha256': sha256_file(out / 'yugioh.f16'),
    }, indent=1) + '\n')
    print(json.dumps({'artworks': len(rows), 'excluded_back_like': [e['name'] for e in excluded], 'printings': len(printings),
                      'priced_printings': sum(p['price_usd'] is not None for p in printings.values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
