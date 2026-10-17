#!/usr/bin/env python3
"""Fill TCGdex Pokémon gallery gaps from pokemontcg.io, using auditable exact matches only.

Reads the TCGdex sync catalog, selects physical records whose image is unavailable or
still pending, and looks each one up in an explicitly mapped pokemontcg.io set. A
reference is downloaded only when the normalized collector number and normalized card
name both match exactly one pokemontcg.io card. Everything else stays unresolved with a
recorded reason. The TCGdex catalog/gallery is never modified; output is staged under
its own directory with per-image provenance and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_pokemon_gallery import append_jsonl, atomic_bytes, atomic_json  # noqa: E402

API = "https://api.pokemontcg.io/v2/cards"
IMAGE_HOSTS = ("images.pokemontcg.io", "images.scrydex.com")  # hosts the API itself returns
USER_AGENT = "tcg-scanner-pokemon-gap-fill/1.0"
TIMEOUT = (10, 60)
ATTEMPTS = 3
SET_ATTEMPTS = 6

# TCGdex set id -> pokemontcg.io set id. Explicit and reviewable; unmapped sets stay unresolved.
SET_MAP = {
    "swsh4.5sv": "swsh45sv", "sm7.5": "sm75", "sm3.5": "sm35", "swsh12.5gg": "swsh12pt5gg",
    "smp": "smp", "svp": "svp", "swshp": "swshp", "xyp": "xyp", "bwp": "bwp", "hgssp": "hsp",
    "ecard2": "ecard2", "ecard3": "ecard3", "30th-c": "me55c", "30th": "me55",
    "swsh9tg": "swsh9tg", "swsh10tg": "swsh10tg", "swsh11tg": "swsh11tg", "swsh12tg": "swsh12tg",
    "cel25": "cel25", "cel25cc": "cel25c", "sve": "sve", "bog": "bp", "exu": "ex10",
    "2011bw": "mcd11", "2012bw": "mcd12", "2014xy": "mcd14", "2015xy": "mcd15", "2016xy": "mcd16",
    "2017sm": "mcd17", "2018sm": "mcd18", "2019sm": "mcd19", "2021swsh": "mcd21", "2022swsh": "mcd22",
    "tk-ex-latia": "tk1a", "tk-ex-latio": "tk1b", "tk-ex-p": "tk2a", "tk-ex-m": "tk2b",
    "sm6": "sm6", "sm2": "sm2", "pl2": "pl2", "ex14": "ex14", "xy8": "xy8", "dc1": "dc1",
    "pop6": "pop6", "sv03.5": "sv3pt5",
}
# Classic Collection reprints: pokemontcg.io keeps the original printed numbers, so these sets
# fall back to a name that is unique within the mapped set. Recorded as matched_by_unique_name.
NAME_ONLY_FALLBACK_SETS = {"cel25cc", "30th-c"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_number(value: str) -> str:
    """Uppercase, decode %3F, and strip leading zeros from the numeric run: 'SV001' -> 'SV1'."""
    text = str(value).replace("%3F", "?").strip().upper()
    match = re.fullmatch(r"([A-Z]*)0*(\d+)([A-Z]*)", text)
    return f"{match.group(1)}{int(match.group(2))}{match.group(3)}" if match else text


def norm_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9?!]", "", text)
    return re.sub(r"^basic(?=[a-z]+energy$)", "", text)  # "Basic Grass Energy" == "Grass Energy"


def match_card(record: dict[str, Any], candidates: list[dict[str, Any]],
               name_only_fallback: bool = False) -> tuple[str, dict[str, Any] | None]:
    number = norm_number(record["local_id"])
    by_number = [c for c in candidates if norm_number(c["number"]) == number]
    named = [c for c in by_number if norm_name(c["name"]) == norm_name(record["name"])]
    if len(named) == 1:
        return "matched", named[0]
    if len(named) > 1:
        return "ambiguous", None
    if name_only_fallback:
        named = [c for c in candidates if norm_name(c["name"]) == norm_name(record["name"])]
        if len(named) == 1:
            return "matched_by_unique_name", named[0]
        return ("ambiguous" if named else "no_name_match"), None
    return ("name_mismatch" if by_number else "no_number_match"), None


def get_session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def fetch_set_cards(session, set_id: str, cache_dir: Path) -> list[dict[str, Any]]:
    cache = cache_dir / f"{set_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())["data"]
    cards, page = [], 1
    while True:
        for attempt in range(SET_ATTEMPTS):
            try:
                resp = session.get(API, params={"q": f"set.id:{set_id}", "pageSize": 250, "page": page,
                                                "select": "id,name,number,images,set"}, timeout=TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception:
                if attempt == SET_ATTEMPTS - 1:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        cards.extend(payload.get("data", []))
        if page * payload.get("pageSize", 250) >= payload.get("totalCount", 0):
            atomic_json(cache, {"set_id": set_id, "fetched_at": now_utc(), "data": cards})
            return cards
        page += 1


def download(session, urls: list[tuple[str, str]], dest: Path) -> dict[str, Any]:
    """Try (variant, url) pairs in order; a 404 moves on to the next variant."""
    last = None
    for variant, url in urls:
        if not any(url.startswith(f"https://{host}/") for host in IMAGE_HOSTS):
            raise ValueError(f"unexpected image host: {url}")
        try:
            return {**download_one(session, url, dest), "source_image_url": url, "image_variant": variant}
        except FileNotFoundError as exc:
            last = exc
    raise RuntimeError(f"all image variants missing: {last}")


def download_one(session, url: str, dest: Path) -> dict[str, Any]:
    last = None
    for attempt in range(ATTEMPTS):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 {url}")
            resp.raise_for_status()
            body = resp.content
            from io import BytesIO
            with Image.open(BytesIO(body)) as image:
                fmt, size = image.format, image.size
                image.verify()
            atomic_bytes(dest, body)
            return {"image_sha256": hashlib.sha256(body).hexdigest(), "image_bytes": len(body),
                    "image_format": fmt, "image_width": size[0], "image_height": size[1]}
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, bounded retry
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", type=Path, default=Path("runs/card_recognizer/pokemon_catalog_sync/catalog.json"))
    ap.add_argument("--output", type=Path, default=Path("runs/card_recognizer/pokemon_gap_fill"))
    ap.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    ap.add_argument("--dry-run", action="store_true", help="Match only; download nothing")
    args = ap.parse_args()

    records = json.loads(args.catalog.read_text())
    gaps = [r for r in records if r.get("domain") == "pokemon_physical_tcg" and r.get("image_status") != "downloaded"]
    manifest_path = args.output / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {row["catalog_identity"]: row for row in json.loads(manifest_path.read_text())["cards"]}
    session = get_session()

    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gaps:
        by_set[record["set_id"]].append(record)
    set_cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    jobs = []
    for tcgdex_set, members in sorted(by_set.items()):
        target = SET_MAP.get(tcgdex_set)
        if target and target not in set_cache:
            try:
                set_cache[target] = fetch_set_cards(session, target, args.output / "source_sets")
            except Exception as exc:  # noqa: BLE001
                print(f"set {target}: lookup failed: {exc}", file=sys.stderr)
                set_cache[target] = None
        for record in members:
            row = {"catalog_identity": record["catalog_identity"], "tcgdex_id": record["provider_id"],
                   "tcgdex_set_id": tcgdex_set, "local_id": record["local_id"], "name": record["name"],
                   "tcgdex_image_status": record["image_status"], "source_provider": "pokemontcg.io",
                   "source_set_id": target, "image_role": "provider digital reference; not an independent capture",
                   "physical_finish": "unresolved", "physical_edition": "unresolved"}
            if not target:
                row["status"] = "no_source_set_mapping"
            elif set_cache[target] is None:
                row["status"] = "source_set_lookup_failed"
            else:
                outcome, card = match_card(record, set_cache[target], tcgdex_set in NAME_ONLY_FALLBACK_SETS)
                row["status"] = row["match_method"] = outcome
                if card:
                    variants = [(v, card["images"][v]) for v in ("large", "small") if card["images"].get(v)]
                    url = variants[0][1]
                    suffix = Path(url).suffix or ".png"
                    row.update({"source_card_id": card["id"], "source_name": card["name"], "source_number": card["number"],
                                "source_image_url": url, "_variants": variants,
                                "image_path": str(Path("cards") / tcgdex_set / f"{record['provider_id']}{suffix}")})
                    old = previous.get(record["catalog_identity"], {})
                    dest = args.output / row["image_path"]
                    if old.get("status") == "downloaded" and old.get("source_card_id") == card["id"] and dest.exists() \
                            and hashlib.sha256(dest.read_bytes()).hexdigest() == old.get("image_sha256"):
                        row.update({k: old[k] for k in old if k.startswith("image_")}
                                   | {"status": "downloaded", "source_image_url": old["source_image_url"]})
                    elif not args.dry_run:
                        jobs.append(row)
            rows.append(row)

    progress = args.output / "progress.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, session, r["_variants"], args.output / r["image_path"]): r for r in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                row.update(future.result())
                row["status"] = "downloaded"
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "download_failed", "last_error": str(exc)})
            append_jsonl(progress, {"catalog_identity": row["catalog_identity"], "status": row["status"]})
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} downloaded", flush=True)

    for row in rows:
        row.pop("_variants", None)
    counts = Counter(r["status"] for r in rows)
    atomic_json(manifest_path, {
        "schema_version": 1, "generated_at": now_utc(),
        "purpose": "fill TCGdex English physical-card image gaps; TCGdex catalog identity remains authoritative",
        "source_catalog": str(args.catalog), "source_provider": "pokemontcg.io", "source_api": API,
        "match_policy": "explicit set map + normalized collector number + normalized card name; exactly one candidate",
        "name_only_fallback_sets": sorted(NAME_ONLY_FALLBACK_SETS),
        "image_variant_policy": "large (hi-res) preferred; small used only when large returns 404, recorded per row",
        "set_map": SET_MAP, "gap_record_count": len(gaps), "status_counts": dict(counts),
        "license_note": "Public accessibility does not establish additional reuse rights.",
        "canonical_gallery_mutation": False, "cards": rows,
    })
    print(json.dumps(dict(counts), indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
