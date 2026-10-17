#!/usr/bin/env python3
"""Stage a bounded set of TCGdex Pokémon card images with source provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError


API_BASE = "https://api.tcgdex.net/v2"
DOC_CARD = "https://tcgdex.dev/rest/card"
DOC_ASSETS = "https://tcgdex.dev/assets"
USER_AGENT = "tcg-scanner-reference-acquisition/1.0"
TIMEOUT_SECONDS = 20
MAX_LIMIT = 200
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_slug(value: str, label: str) -> str:
    value = value.strip()
    if not value or not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label}: expected letters, digits, underscore, or hyphen")
    return value


def image_url_for(card: dict[str, Any]) -> str:
    """Build the documented high-resolution PNG URL from TCGdex card metadata."""
    raw = card.get("image")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Card {card.get('id', '?')} has no image URL")
    parsed = urlparse(raw)
    if parsed.scheme != "https" or parsed.netloc != "assets.tcgdex.net":
        raise ValueError(f"Unexpected TCGdex image URL: {raw}")
    return raw.rstrip("/") + "/high.png"


def printed_number_fields(local_id: object, official_total: object) -> dict[str, object]:
    """Preserve TCGdex localId verbatim; derive a numerator only when unambiguous."""
    printed = str(local_id) if local_id is not None else None
    numerator = int(printed) if printed and printed.isdigit() else None
    total = official_total if isinstance(official_total, int) and not isinstance(official_total, bool) else None
    return {"printed_number": printed, "printed_numerator": numerator, "official_total": total}


def catalog_record(card: dict[str, Any], detail: dict[str, Any], language: str,
                   set_id: str, source_image_url: str, image_sha256: str) -> dict[str, Any]:
    card_id = card.get("id")
    local_id = detail.get("localId", card.get("localId"))
    if not isinstance(card_id, str) or not card_id:
        raise ValueError("Card record is missing its catalog id")
    if not isinstance(local_id, (str, int)) or str(local_id) == "":
        raise ValueError(f"Card {card_id} is missing localId")
    card_set = detail.get("set") if isinstance(detail.get("set"), dict) else {}
    count = card_set.get("cardCount") if isinstance(card_set.get("cardCount"), dict) else {}
    variants = detail.get("variants") if isinstance(detail.get("variants"), dict) else {}
    return {
        "game": "pokemon",
        "catalog_identity": f"pokemon:{language}:{set_id}:{local_id}",
        "catalog_id": card_id,
        "language": language,
        "set_id": set_id,
        "set_name": card_set.get("name"),
        "local_number": str(local_id),
        **printed_number_fields(local_id, count.get("official")),
        "name": detail.get("name", card.get("name")),
        "source": "TCGdex REST API and assets",
        "source_catalog_url": f"{API_BASE}/{language}/cards/{card_id}",
        "source_image_url": source_image_url,
        "image_sha256": image_sha256,
        "physical_finish": "unresolved",
        "physical_edition": "unresolved",
        "available_catalog_variants": variants,
        "identity_note": "One digital catalog image may represent multiple physical finishes or editions; no exact physical copy is asserted.",
        "license_or_reuse_basis": "not established by this acquisition manifest",
        "training_promotion": "not_promoted",
    }


def validate_image_bytes(data: bytes) -> None:
    if not data:
        raise ValueError("Image response is empty")
    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG image bytes, received {image.format or 'unknown'}")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Response is not a valid PNG image") from exc


def require_requests():
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("This command requires the project's requests dependency") from exc
    return requests


def request_json(url: str, requests_module: Any) -> dict[str, Any]:
    response = requests_module.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return payload


def request_bytes(url: str, requests_module: Any) -> bytes:
    response = requests_module.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.content
    validate_image_bytes(data)
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_output_path(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")


def acquire(set_id: str = "swsh3", language: str = "en", limit: int = 32,
            output: Path = Path("runs/card_recognizer/pokemon_references"),
            requests_module: Any | None = None) -> dict[str, Any]:
    set_id = validate_slug(set_id, "set id")
    language = validate_slug(language, "language")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    output = Path(output)
    check_output_path(output)
    requests_module = requests_module or require_requests()

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        set_url = f"{API_BASE}/{language}/sets/{set_id}"
        set_data = request_json(set_url, requests_module)
        cards = set_data.get("cards")
        if not isinstance(cards, list):
            raise ValueError("Set response is missing its cards list")
        selected = cards[:limit]
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for brief in selected:
            if not isinstance(brief, dict) or not isinstance(brief.get("id"), str):
                raise ValueError("Set contains an invalid card entry")
            card_id = brief["id"]
            validate_slug(card_id, "catalog id")
            if card_id in seen_ids:
                raise ValueError(f"Duplicate catalog id in set response: {card_id}")
            seen_ids.add(card_id)
            detail_url = f"{API_BASE}/{language}/cards/{card_id}"
            detail = request_json(detail_url, requests_module)
            if detail.get("id") != card_id:
                raise ValueError(f"Card detail id mismatch for {card_id}")
            if not isinstance(detail.get("set"), dict) or detail["set"].get("id") != set_id:
                raise ValueError(f"Card {card_id} does not belong to requested set {set_id}")
            image_url = image_url_for(detail)
            image_data = request_bytes(image_url, requests_module)
            digest = sha256_bytes(image_data)
            card_dir = stage / "cards" / set_id
            card_dir.mkdir(parents=True, exist_ok=True)
            image_path = card_dir / f"{card_id}.png"
            image_path.write_bytes(image_data)
            record = catalog_record(brief, detail, language, set_id, image_url, digest)
            record["image_path"] = image_path.relative_to(stage).as_posix()
            sidecar_path = image_path.with_suffix(".json")
            sidecar_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "game": "pokemon",
            "source": "TCGdex",
            "set_id": set_id,
            "language": language,
            "set_url": set_url,
            "documentation": {"card_api": DOC_CARD, "asset_format": DOC_ASSETS},
            "requested_limit": limit,
            "saved_count": len(records),
            "physical_finish_and_edition": "unresolved for every catalog image",
            "training_promotion": "not_promoted",
            "records": records,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            output.rmdir()
        stage.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_id", default="swsh3", help="TCGdex set id (default: swsh3)")
    parser.add_argument("--language", default="en", help="TCGdex language code (default: en)")
    parser.add_argument("--limit", type=int, default=32, help=f"Maximum cards to acquire (1-{MAX_LIMIT}, default: 32)")
    parser.add_argument("--output", type=Path, default=Path("runs/card_recognizer/pokemon_references"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = acquire(args.set_id, args.language, args.limit, args.output)
    except Exception as exc:
        raise SystemExit(f"Pokemon reference acquisition failed: {exc}") from exc
    print(f"Saved {manifest['saved_count']} TCGdex Pokémon references to {args.output}")
    print("Physical finish/edition remain unresolved; data is staged and not promoted to training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
