#!/usr/bin/env python3
"""Resumable, source-discovered Japanese One Piece reference sync from Bandai.

The official card-list HTML is the catalog: set selectors and each modal's
actual image URL are parsed from the page. No card filenames are guessed.
Downloaded assets are staged separately from the training corpus.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

BASE = "https://www.onepiece-cardgame.com"
LIST = f"{BASE}/cardlist/"
UA = "tcg-scanner-official-onepiece-ja-reference-sync/1.0 (public catalog audit)"
DEFAULT_OUT = Path("runs/card_recognizer/onepiece_ja_catalog_sync")
MAX_DEFAULT = 1024 * 1024 * 1024
MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request(url: str) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": UA}, timeout=(10, 40))
    response.raise_for_status()
    return response


def parse_series(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    select = soup.select_one("select#series")
    if not select:
        raise RuntimeError("Bandai card list did not contain select#series")
    result = []
    for option in select.find_all("option"):
        value = option.get("value", "").strip()
        label = re.sub(r"\s+", " ", unescape(option.get_text(" ", strip=True))).strip()
        if not value or label in {"ALL", "Recording"}:
            continue
        result.append({"series_value": value, "series_label": label})
    if not result:
        raise RuntimeError("No official series options found")
    return result


def parse_cards(html: str, series: dict, source_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []
    seen = set()
    for block in soup.select("dl.modalCol[id]"):
        card_id = block.get("id", "").strip()
        # Match exact physical card print IDs and optional official art suffix.
        if not re.fullmatch(r"(?:OP|ST|EB)\d{2}-\d{2,3}(?:_p\d+)?|P-\d{2,3}(?:_p\d+)?", card_id):
            continue
        image = block.select_one("img.lazy[data-src]")
        if not image:
            continue
        image_url = urljoin(BASE, image.get("data-src", "").replace("..", ""))
        if not image_url.startswith(BASE + "/images/cardlist/card/"):
            continue
        name_node = block.select_one(".cardName")
        name = re.sub(r"\s+", " ", name_node.get_text(" ", strip=True)) if name_node else image.get("alt")
        if card_id in seen:
            continue
        seen.add(card_id)
        m = re.match(r"([A-Z]+\d*)-", card_id)
        records.append({
            "catalog_identity": f"onepiece:ja:{card_id}", "language": "ja",
            "card_id": card_id, "set_id": m.group(1) if m else None,
            "name": name, "series_value": series["series_value"],
            "series_label": series["series_label"], "source_catalog_url": source_url,
            "source_image_url": image_url, "status": "discovered",
        })
    return records


def local_ids(root: Path) -> set[str]:
    ids = set()
    if root.exists():
        for path in root.rglob("*.jpg"):
            ids.add(path.stem)
        for path in root.rglob("*.png"):
            ids.add(path.stem)
    return ids


def image_info(data: bytes) -> tuple[str, int, int]:
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            fmt, size = im.format, im.size
            im.verify()
        if not fmt or size[0] < 100 or size[1] < 100:
            raise ValueError("invalid/tiny image")
        return fmt.lower(), size[0], size[1]
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image response: {exc}") from exc


def safe_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--local-cards", type=Path, default=Path("runs/card_recognizer/onepiece_ja_local_gallery"), help="Optional existing Japanese gallery; must not point at the English corpus")
    ap.add_argument("--sets", nargs="*", help="Set prefixes such as OP12 OP01; defaults to all official series")
    ap.add_argument("--max-download-bytes", type=int, default=MAX_DEFAULT)
    ap.add_argument("--max-downloads", type=int, default=0, help="Optional cap for a small first/resume pass")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--catalog-only", action="store_true")
    args = ap.parse_args()
    if not (1 <= args.workers <= 3):
        ap.error("--workers must be 1..3")
    if args.max_download_bytes < 0:
        ap.error("--max-download-bytes must be nonnegative")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    free_bytes = __import__("shutil").disk_usage(out).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"disk reserve requires {MIN_FREE_BYTES} free bytes; found {free_bytes}")
    cards_dir = out / "cards"
    cards_dir.mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    root_resp = request(LIST)
    root_html = root_resp.text
    series_list = parse_series(root_html)
    if args.sets:
        wanted = {s.upper() for s in args.sets}
        series_list = [s for s in series_list if any(w in s["series_label"].upper() or w.replace("OP", "OP-").replace("ST", "ST-").replace("EB", "EB-") in s["series_label"].upper() for w in wanted)]
        if not series_list:
            raise RuntimeError(f"No official series labels matched --sets {sorted(wanted)}")

    records, failures = [], []
    for series in series_list:
        url = f"{LIST}?series={series['series_value']}"
        try:
            response = request(url)
            records.extend(parse_cards(response.text, series, response.url))
        except Exception as exc:
            failures.append({"series": series, "url": url, "error": str(exc)})
        time.sleep(0.08)
    # Official listings may overlap; retain the first exact catalog mapping.
    unique = {}
    for r in records:
        if r["card_id"] in unique and unique[r["card_id"]]["source_image_url"] != r["source_image_url"]:
            failures.append({"card_id": r["card_id"], "error": "conflicting official image URL across series listings", "urls": [unique[r["card_id"]]["source_image_url"], r["source_image_url"]]})
            continue
        unique[r["card_id"]] = r
    records = list(unique.values())
    existing = local_ids(args.local_cards)
    previous = {r.get("card_id"): r for r in old.get("cards", []) if r.get("card_id")}
    for r in records:
        prior = previous.get(r["card_id"], {})
        rel = prior.get("image_path")
        staged = out / rel if rel else None
        if r["card_id"] in existing:
            r["status"] = "present_local"
            r["local_reference_exists"] = True
        elif staged and staged.is_file():
            data = staged.read_bytes()
            try:
                fmt, w, h = image_info(data)
                r.update(status="downloaded", image_path=str(staged.relative_to(out)), image_sha256=sha256(data), image_width=w, image_height=h, image_format=fmt, image_bytes=len(data))
            except Exception as exc:
                r.update(status="invalid_staged", error=str(exc))
        else:
            r["status"] = "pending"
    pending = [r for r in records if r["status"] == "pending"]
    if args.max_downloads:
        pending = pending[:args.max_downloads]
    downloaded_bytes = 0
    if not args.catalog_only:
        def fetch(r):
            resp = request(r["source_image_url"])
            body = resp.content
            fmt, w, h = image_info(body)
            return r, body, fmt, w, h
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, r): r for r in pending}
            for future in concurrent.futures.as_completed(futures):
                r = futures[future]
                try:
                    item, body, fmt, w, h = future.result()
                    if downloaded_bytes + len(body) > args.max_download_bytes:
                        item.update(status="deferred_budget", image_bytes=len(body), error="per-run download byte cap reached")
                        continue
                    ext = "png" if fmt == "png" else fmt
                    dest = cards_dir / f"{item['card_id']}.{ext}"
                    if dest.exists() and sha256(dest.read_bytes()) != sha256(body):
                        dest = cards_dir / f"{item['card_id']}__sha{sha256(body)[:8]}.{ext}"
                    safe_write(dest, body)
                    downloaded_bytes += len(body)
                    item.update(status="downloaded", image_path=str(dest.relative_to(out)), image_sha256=sha256(body), image_width=w, image_height=h, image_format=fmt, image_bytes=len(body), fetched_at=now(), final_image_url=future.result()[0]["source_image_url"])
                except Exception as exc:
                    r.update(status="unavailable", error=str(exc), checked_at=now())
    # Keep unresolved counts explicit; a missing local reference is not a broken official asset.
    status_counts = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    manifest = {
        "schema_version": 1, "generated_at": now(), "provider": "Bandai official Japanese One Piece card list",
        "source_catalog_url": root_resp.url, "catalog_discovery": "series selector options and dl.modalCol records parsed from official HTML; source image URLs taken from each modal img[data-src]",
        "language": "ja", "image_role": "official Japanese digital reference; not an independent capture", "license_note": "Public accessibility does not establish additional reuse rights.",
        "disk_free_bytes_at_start": free_bytes, "minimum_disk_reserve_bytes": MIN_FREE_BYTES,
        "series_options_seen": len(series_list), "discovered_unique_printings": len(records), "selected_sets": args.sets or "all official series selector options",
        "existing_local_id_count": len(existing), "downloaded_this_run_bytes": downloaded_bytes,
        "per_run_download_byte_cap": args.max_download_bytes, "status_counts": status_counts,
        "series_failures": failures, "cards": sorted(records, key=lambda x: x["card_id"]),
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(json.dumps({k: manifest[k] for k in ("series_options_seen", "discovered_unique_printings", "existing_local_id_count", "downloaded_this_run_bytes", "status_counts")}, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
