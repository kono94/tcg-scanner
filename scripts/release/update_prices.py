#!/usr/bin/env python3
"""Fetch current prices for every gallery identity, keyed by our exact namespaced IDs.

Sources (both expose the same card numbers our gallery uses):
  * One Piece: dotgg (TCGplayer USD + Cardmarket EUR), ids like OP01-120, OP01-120_p1, _r1.
    English gallery ids match dotgg 'en' rows; Japanese ids only match dotgg Japanese rows
    ('jp'/'jpe'), never English rows with the same number.
  * Pokemon: TCGdex card records (/v2/{lang}/cards/{id}), whose pricing block carries Cardmarket
    (EUR) and TCGplayer (USD). Our Pokemon ids are TCGdex ids.

Output: prices.json with one block per market (USD, EUR). A missing price stays absent (unknown,
never zero). Each quote records its basis (field and finish) so ranges stay explainable. The day's
ECB USD->EUR reference rate is recorded under `fx`; build_browser_bundle.py uses it to fill a
market's gaps from the other market, marking those quotes as converted.
Raw responses are cached under runs/prices/cache so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DOTGG = 'https://api.dotgg.gg/cgfw/getcards?game=onepiece&mode=indexed'
TCGDEX = 'https://api.tcgdex.net/v2'
FX = 'https://api.frankfurter.dev/v1/latest?from=USD&to=EUR'
UA = {'User-Agent': 'tcg-scanner-price-snapshot (local research)'}
TCGPLAYER_ORDER = ['normal', 'holofoil', 'reverseHolofoil', '1stEditionHolofoil', '1stEditionNormal', 'unlimitedHolofoil']


def get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except (URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'Failed after {retries} attempts: {url}')


def positive(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 2) if f > 0 else None


def onepiece(identities: list[str]) -> tuple[dict, dict, dict]:
    data = get_json(DOTGG)
    idx = {n: i for i, n in enumerate(data['names'])}
    rows = {}
    for r in data['data']:
        lang = 'en' if r[idx['language']] == 'en' else 'ja'
        rows[(lang, r[idx['id']])] = r
    usd, eur, stats = {}, {}, {'gallery': 0, 'matched_rows': 0}
    for ident in identities:
        game, lang, card = ident.split(':', 2)
        if game != 'onepiece':
            continue
        stats['gallery'] += 1
        r = rows.get((lang, card))
        if r is None:
            continue
        stats['matched_rows'] += 1
        for field, out, cur in (('price', usd, 'USD'), ('foilPrice', usd, 'USD'), ('cmPrice', eur, 'EUR'), ('cmFoilPrice', eur, 'EUR')):
            v = positive(r[idx[field]])
            if v is not None and ident not in out:  # non-foil first, then foil
                out[ident] = {'amount': v, 'basis': f'dotgg:{field}'}
    return usd, eur, stats


def pokemon_one(ident: str, cache: Path):
    _game, lang, card = ident.split(':', 2)
    f = cache / lang / f'{card}.json'
    if f.exists():
        return ident, json.loads(f.read_text())
    d = get_json(f'{TCGDEX}/{lang}/cards/{card}')
    rec = {'pricing': (d or {}).get('pricing'), 'found': d is not None, 'fetched': date.today().isoformat()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    return ident, rec


def pokemon(identities: list[str], cache: Path, workers: int) -> tuple[dict, dict, dict]:
    ids = [i for i in identities if i.startswith('pokemon:')]
    usd, eur = {}, {}
    stats = {'gallery': len(ids), 'records_found': 0, 'with_pricing': 0}
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(pokemon_one, i, cache) for i in ids]):
            ident, rec = fut.result()
            done += 1
            if done % 1000 == 0:
                print(f'pokemon {done}/{len(ids)}', flush=True)
            stats['records_found'] += rec['found']
            p = rec.get('pricing') or {}
            if p:
                stats['with_pricing'] += 1
            tp = p.get('tcgplayer') or {}
            for variant in TCGPLAYER_ORDER + [k for k in tp if k not in TCGPLAYER_ORDER]:
                v = tp.get(variant)
                if isinstance(v, dict) and positive(v.get('marketPrice')) is not None:
                    usd[ident] = {'amount': positive(v['marketPrice']), 'basis': f'tcgplayer:{variant}:marketPrice'}
                    break
            cm = p.get('cardmarket') or {}
            for field in ('avg30', 'trend', 'avg30-holo', 'trend-holo'):
                if positive(cm.get(field)) is not None:
                    eur[ident] = {'amount': positive(cm[field]), 'basis': f'cardmarket:{field}'}
                    break
    return usd, eur, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, default=ROOT / 'models/browser-export/gpu-20261017/gallery.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/prices/prices.json')
    parser.add_argument('--cache', type=Path, default=ROOT / f'runs/prices/cache/tcgdex-{date.today().isoformat()}')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    identities = json.loads(args.gallery.read_text())['identities']
    op_usd, op_eur, op_stats = onepiece(identities)
    print('onepiece', op_stats, 'usd', len(op_usd), 'eur', len(op_eur), flush=True)
    pk_usd, pk_eur, pk_stats = pokemon(identities, args.cache, args.workers)
    print('pokemon', pk_stats, 'usd', len(pk_usd), 'eur', len(pk_eur), flush=True)
    today = date.today().isoformat()
    fx = get_json(FX)
    out = {
        'schema_version': 2, 'snapshot_date': today,
        'fx': {'USD_EUR': fx['rates']['EUR'], 'date': fx['date'], 'source': 'ECB reference rate via frankfurter.dev'},
        'condition': 'not modelled (market aggregate)',
        'markets': {
            'USD': {'currency': 'USD', 'source': 'TCGplayer market price (via dotgg for One Piece, TCGdex for Pokémon)',
                    'quotes': {**op_usd, **pk_usd}},
            'EUR': {'currency': 'EUR', 'source': 'Cardmarket (via dotgg for One Piece, TCGdex 30-day average for Pokémon)',
                    'quotes': {**op_eur, **pk_eur}},
        },
        'coverage': {'gallery_identities': len(identities), 'onepiece': op_stats, 'pokemon': pk_stats,
                     'usd_priced': len(op_usd) + len(pk_usd), 'eur_priced': len(op_eur) + len(pk_eur)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(',', ':')))
    print(json.dumps(out['coverage'], indent=2))


if __name__ == '__main__':
    main()
