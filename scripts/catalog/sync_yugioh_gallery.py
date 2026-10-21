#!/usr/bin/env python3
"""Resumable English Yu-Gi-Oh! catalog + artwork reference sync from YGOPRODeck.

One record per artwork (YGOPRODeck card image id), covering old and new cards. The
card's TCG set printings (set code x rarity) are kept with it as plausible versions:
the API does not say which alternate artwork belongs to which set code, so that
mapping stays unresolved rather than guessed. Images are downloaded and stored
locally (YGOPRODeck asks not to hotlink) at a bounded request rate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
IMAGE_PREFIX = "https://images.ygoprodeck.com/images/cards/"
UA = "tcg-scanner-yugioh-reference-sync/1.0 (public catalog audit)"
DEFAULT_OUT = Path("datasets/sources/catalog/yugioh-en-ygoprodeck")
MAX_DEFAULT = 4 * 1024 * 1024 * 1024
# YGOPRODeck's documented limit is 20 requests/second; stay well below it.
MIN_INTERVAL = 0.1
# Tournament deck-profile placeholders, not physical cards.
PLACEHOLDER_NAMES = {"???"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RateLimiter:
    def __init__(self, interval: float):
        self.interval, self.next, self.lock = interval, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            delay = self.next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next = max(self.next, time.monotonic()) + self.interval


def request(url: str, limiter: RateLimiter, attempts: int = 3) -> requests.Response:
    for attempt in range(attempts):
        limiter.wait()
        try:
            response = requests.get(url, headers={"User-Agent": UA}, timeout=(10, 60))
            if response.status_code == 429 and attempt + 1 < attempts:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def normalize(payload: dict) -> list[dict]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("YGOPRODeck cardinfo did not return a data array")
    records, seen = [], set()
    for card in data:
        if card.get("name") in PLACEHOLDER_NAMES:
            continue
        card_id = int(card["id"])
        printings = [{
            "set_name": s.get("set_name"), "set_code": s.get("set_code"),
            "rarity": s.get("set_rarity"), "rarity_code": s.get("set_rarity_code"),
            "price_usd": s.get("set_price"),
        } for s in card.get("card_sets") or []]
        images = card.get("card_images") or []
        for index, image in enumerate(images):
            art_id = int(image["id"])
            if art_id in seen:
                raise ValueError(f"duplicate artwork id {art_id}")
            seen.add(art_id)
            url = image.get("image_url") or ""
            records.append({
                # Artwork-level identity; the passcode (card_id) names the underlying card.
                "catalog_identity": f"yugioh:en:{art_id}", "language": "en",
                "artwork_id": art_id, "card_id": card_id, "artwork_index": index,
                "alternate_artwork": art_id != card_id, "artworks_of_card": len(images),
                "name": card.get("name"), "type": card.get("type"), "frame_type": card.get("frameType"),
                "tcg_printings": printings, "tcg_printed": bool(printings),
                "printing_artwork_mapping": "single artwork" if len(images) == 1 else "unresolved across alternate artworks",
                "source_image_url": url if url.startswith(IMAGE_PREFIX) else None,
                "status": "pending" if url.startswith(IMAGE_PREFIX) else "unavailable",
            })
    return records


def image_info(data: bytes) -> tuple[str, int, int]:
    try:
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


def write_manifest(path: Path, manifest: dict):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-download-bytes", type=int, default=MAX_DEFAULT)
    ap.add_argument("--max-downloads", type=int, default=0, help="Optional cap for a small first/resume pass")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--catalog-only", action="store_true")
    args = ap.parse_args()
    if not (1 <= args.workers <= 6):
        ap.error("--workers must be 1..6")
    out = args.output
    cards_dir = out / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    limiter = RateLimiter(MIN_INTERVAL)

    response = request(API, limiter)
    raw = response.content
    safe_write(out / "cardinfo.json", raw)
    records = normalize(json.loads(raw))

    previous = {r["artwork_id"]: r for r in old.get("cards", [])}
    for r in records:
        rel = previous.get(r["artwork_id"], {}).get("image_path")
        staged = out / rel if rel else None
        if staged and staged.is_file():
            data = staged.read_bytes()
            try:
                fmt, w, h = image_info(data)
                r.update(status="downloaded", image_path=rel, image_sha256=sha256(data), image_width=w,
                         image_height=h, image_format=fmt, image_bytes=len(data))
            except ValueError as exc:
                r.update(status="pending", error=f"invalid staged file: {exc}")
    pending = [r for r in records if r["status"] == "pending"]
    if args.max_downloads:
        pending = pending[:args.max_downloads]

    downloaded_bytes = 0
    if not args.catalog_only and pending:
        def fetch(r):
            body = request(r["source_image_url"], limiter).content
            return body, *image_info(body)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, r): r for r in pending}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                r = futures[future]
                try:
                    body, fmt, w, h = future.result()
                    if downloaded_bytes + len(body) > args.max_download_bytes:
                        r.update(status="deferred_budget", error="per-run download byte cap reached")
                        continue
                    ext = "jpg" if fmt == "jpeg" else fmt
                    dest = cards_dir / f"{r['artwork_id']}.{ext}"
                    safe_write(dest, body)
                    downloaded_bytes += len(body)
                    r.update(status="downloaded", image_path=str(dest.relative_to(out)), image_sha256=sha256(body),
                             image_width=w, image_height=h, image_format=fmt, image_bytes=len(body), fetched_at=now())
                    r.pop("error", None)
                except Exception as exc:
                    r.update(status="unavailable", error=str(exc), checked_at=now())
                if done % 500 == 0:
                    print(f"{done}/{len(pending)} images, {downloaded_bytes / 1e6:.0f} MB", flush=True)

    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    manifest = {
        "schema_version": 1, "generated_at": now(), "provider": "YGOPRODeck API v7 (English TCG/OCG card database)",
        "source_catalog_url": API, "cardinfo_sha256": sha256(raw), "language": "en",
        "identity": "one record per artwork (YGOPRODeck image id); card_id is the passcode of the underlying card",
        "image_role": "provider digital reference; not an independent capture",
        "license_note": "Card images are Konami's; YGOPRODeck asks that images be downloaded, not hotlinked. Public accessibility does not establish additional reuse rights.",
        "excluded": "deck-profile placeholder cards ('???')",
        "cards_seen": len({r["card_id"] for r in records}), "artworks": len(records),
        "tcg_printings": sum(len(r["tcg_printings"]) for r in records if r["artwork_index"] == 0),
        "artworks_without_tcg_printing": sum(not r["tcg_printed"] for r in records),
        "downloaded_this_run_bytes": downloaded_bytes, "per_run_download_byte_cap": args.max_download_bytes,
        "status_counts": status_counts, "cards": sorted(records, key=lambda x: x["artwork_id"]),
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("cards_seen", "artworks", "tcg_printings", "artworks_without_tcg_printing", "downloaded_this_run_bytes", "status_counts")}, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
