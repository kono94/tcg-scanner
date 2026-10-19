#!/usr/bin/env python3
"""Collect acquired physical-card references without guessing IDs from filenames.

This inventories available bytes; it does not compute embeddings, repair images,
claim complete catalog coverage, or assign labels to camera queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(repo: Path) -> dict:
    rows, excluded = [], []
    old_root = repo / 'datasets/sources/catalog/onepiece-en-bandai-2024'
    for image in sorted(old_root.rglob('*.jpg')):
        metadata = image.with_suffix('.json')
        if not metadata.is_file():
            raise ValueError(f'Missing legacy reference metadata: {image}')
        record = json.loads(metadata.read_text())
        if record['id'] != image.stem:
            raise ValueError(f'Legacy identity disagrees with metadata: {image}')
        rows.append(dict(path=image.relative_to(repo).as_posix(), sha256=digest(image),
                         game='onepiece', language='en', card_id=record['id'], name=record.get('name'),
                         printing='unknown', finish='unknown', source_url=record.get('image_url'),
                         provider='Bandai legacy corpus', image_role='digital_reference',
                         watermark_policy='original retained; SAMPLE may be present'))
    for subdir, filename, field in [('pokemon-en-tcgdex', 'catalog.json', None),
                                    ('onepiece-en-bandai', 'manifest.json', 'cards')]:
        root = repo / 'datasets/sources/catalog' / subdir
        payload = json.loads((root / filename).read_text())
        records = payload[field] if field else payload
        for record in records:
            if subdir.startswith('pokemon'):
                if record.get('domain') != 'pokemon_physical_tcg' or record.get('language') != 'en':
                    continue
                game, card_id = 'pokemon', record['catalog_id']
            else:
                game, card_id = 'onepiece', record['card_id']
                if record['status'] == 'present_local':
                    continue
            rel = record.get('image_path')
            if not rel:
                excluded.append(dict(game=game, card_id=card_id, reason='no_acquired_image'))
                continue
            image = root / rel
            if not image.resolve().is_relative_to(root.resolve()) or not image.is_file():
                raise ValueError(f'Invalid acquired reference path: {image}')
            sha = digest(image)
            if sha != record['image_sha256']:
                raise ValueError(f'Acquired reference checksum mismatch: {image}')
            rows.append(dict(path=image.relative_to(repo).as_posix(), sha256=sha, game=game,
                             language='en', card_id=card_id, name=record.get('name'),
                             printing='unknown', finish='unknown', provider='TCGdex' if game == 'pokemon' else 'Bandai',
                             source_url=record.get('image_url', record.get('source_image_url')),
                             source_catalog_url=record.get('source_catalog_url'),
                             image_role='digital_reference',
                             watermark_policy='original retained; not AI restored'))
    targeted = repo / 'datasets/sources/catalog/pokemon-en-pokemontcgio/targeted_downloads.json'
    if targeted.is_file():
        gap_root = targeted.parent
        for record in json.loads(targeted.read_text())['cards']:
            if record.get('status') != 'downloaded':
                continue
            image = gap_root / record['image_path']
            if not image.resolve().is_relative_to(gap_root.resolve()) or digest(image) != record['image_sha256']:
                raise ValueError(f'Invalid gap-fill reference: {image}')
            rows.append(dict(path=image.relative_to(repo).as_posix(), sha256=record['image_sha256'],
                             game='pokemon', language='en', card_id=record['tcgdex_id'], name=record['name'],
                             provider='pokemontcg.io', source_url=record['source_image_url'],
                             provider_card_id=record['source_card_id'], printing='unknown', finish='unknown',
                             image_role='digital_reference', mapping_evidence='explicit set map + unique number and name'))
        filled = {r['card_id'] for r in rows if r['game'] == 'pokemon'}
        excluded = [r for r in excluded if r['game'] != 'pokemon' or r['card_id'] not in filled]
    unique = {}
    for row in rows:
        key = (row['game'], row['language'], row['card_id'], row['sha256'])
        if key in unique:
            unique[key].setdefault('aliases', []).append(row['path'])
        else:
            unique[key] = row
    return dict(schema_version=1, artifact_type='available_physical_reference_manifest',
                gallery_complete=False, references=list(unique.values()),
                counts_by_game=dict(Counter(r['game'] for r in unique.values())),
                unavailable_physical_references=excluded,
                note='Physical English records only; source catalog gaps remain. No query labels or embeddings created.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = collect(args.repo_root.resolve())
    if args.output.exists():
        raise ValueError('Output already exists; create a new manifest version')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(counts_by_game=payload['counts_by_game'],
                         unavailable=len(payload['unavailable_physical_references']), output=str(args.output))))


if __name__ == '__main__':
    main()
