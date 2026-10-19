#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CARDS_DIR = Path("datasets/card_recognizer/cards")
DEFAULT_RAW_DIR = Path("datasets/card_recognizer/real_raw")


@dataclass(frozen=True)
class SourceRow:
    card_id: str
    image_url: str | None
    image_path: Path | None
    source: str | None
    license: str | None
    notes: str | None


@dataclass(frozen=True)
class ImportedImage:
    card_id: str
    raw_image_path: Path
    training_image_path: Path | None
    sha256: str
    source_url: str | None
    source_path: str | None
    source: str | None
    license: str | None
    notes: str | None


def normalize_set_code(value: str) -> str:
    text = value.strip().upper().replace("-", "")
    if re.fullmatch(r"\d{1,2}", text):
        return f"OP{int(text):02d}"
    if re.fullmatch(r"OP\d{1,2}", text):
        return f"OP{int(text[2:]):02d}"
    if re.fullmatch(r"ST\d{1,2}", text):
        return f"ST{int(text[2:]):02d}"
    if re.fullmatch(r"EB\d{1,2}", text):
        return f"EB{int(text[2:]):02d}"
    return text


def normalize_card_id(value: str) -> str:
    return value.strip().upper().replace("_P", "_p")


def card_id_from_training_image(path: Path) -> str:
    return path.stem.split("__", 1)[0]


def load_known_card_ids(cards_dir: Path, set_code: str) -> set[str]:
    set_dir = cards_dir / set_code
    if not set_dir.exists():
        raise FileNotFoundError(f"Set directory does not exist: {set_dir}")

    card_ids: set[str] = set()
    for path in set_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            card_ids.add(normalize_card_id(path.stem))
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            parts = path.relative_to(set_dir).parts
            if len(parts) >= 2:
                card_ids.add(normalize_card_id(parts[0]))
            else:
                card_ids.add(normalize_card_id(card_id_from_training_image(path)))
    return card_ids


def read_source_manifest(path: Path) -> list[SourceRow]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl_source_manifest(path)
    return read_csv_source_manifest(path)


def read_csv_source_manifest(path: Path) -> list[SourceRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            source_row_from_mapping(row, base_dir=path.parent, row_number=index)
            for index, row in enumerate(reader, 2)
        ]


def read_jsonl_source_manifest(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rows.append(source_row_from_mapping(json.loads(line), base_dir=path.parent, row_number=index))
    return rows


def source_row_from_mapping(mapping: dict[str, object], *, base_dir: Path, row_number: int) -> SourceRow:
    card_id = str(mapping.get("card_id") or mapping.get("id") or "").strip()
    image_url = optional_str(mapping.get("image_url") or mapping.get("url"))
    image_path_value = optional_str(mapping.get("image_path") or mapping.get("path") or mapping.get("file"))
    image_path = resolve_manifest_path(image_path_value, base_dir) if image_path_value else None
    if not card_id:
        raise ValueError(f"Missing card_id in source manifest row {row_number}")
    if not image_url and image_path is None:
        raise ValueError(f"Missing image_url or image_path in source manifest row {row_number}")
    return SourceRow(
        card_id=normalize_card_id(card_id),
        image_url=image_url,
        image_path=image_path,
        source=optional_str(mapping.get("source")),
        license=optional_str(mapping.get("license")),
        notes=optional_str(mapping.get("notes")),
    )


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def require_requests():
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Downloading URLs requires requests. Install the project dependencies with "
            "`pip install -r requirements.txt`, or provide local image_path values instead."
        ) from exc
    return requests


def download_image(url: str, *, timeout: int, user_agent: str) -> Path:
    requests = require_requests()
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if content_type and not content_type.lower().startswith("image/"):
        raise ValueError(f"URL did not return an image content-type: {content_type}")

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".img"

    temp = tempfile.NamedTemporaryFile(prefix="real-card-", suffix=suffix, delete=False)
    try:
        temp.write(response.content)
        temp.close()
        return Path(temp.name)
    except Exception:
        temp.close()
        Path(temp.name).unlink(missing_ok=True)
        raise


def copy_local_image(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Source image does not exist: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported source image suffix for {path}. Supported: {sorted(IMAGE_SUFFIXES)}")
    temp = tempfile.NamedTemporaryFile(prefix="real-card-", suffix=path.suffix.lower(), delete=False)
    temp.close()
    shutil.copyfile(path, temp.name)
    return Path(temp.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_to_jpeg(source_path: Path, output_path: Path) -> None:
    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, "JPEG", quality=95, optimize=True)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Source is not a readable image: {source_path}") from exc


def existing_hashes(raw_set_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for sidecar in raw_set_dir.rglob("*.json"):
        try:
            with sidecar.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        sha = payload.get("sha256")
        if isinstance(sha, str):
            hashes.add(sha)
    return hashes


def write_sidecar(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def append_manifest(path: Path, images: Iterable[ImportedImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for image in images:
            handle.write(
                json.dumps(
                    {
                        "card_id": image.card_id,
                        "raw_image_path": str(image.raw_image_path),
                        "training_image_path": str(image.training_image_path) if image.training_image_path else None,
                        "sha256": image.sha256,
                        "source_url": image.source_url,
                        "source_path": image.source_path,
                        "source": image.source,
                        "license": image.license,
                        "notes": image.notes,
                    },
                    ensure_ascii=True,
                )
            )
            handle.write("\n")


def import_source_rows(
    rows: list[SourceRow],
    *,
    set_code: str,
    cards_dir: Path,
    raw_dir: Path,
    promote_to_training: bool,
    timeout: int,
    user_agent: str,
) -> tuple[list[ImportedImage], int]:
    known_card_ids = load_known_card_ids(cards_dir, set_code)
    raw_set_dir = raw_dir / set_code
    seen_hashes = existing_hashes(raw_set_dir)

    imported: list[ImportedImage] = []
    skipped = 0
    for index, row in enumerate(rows, 1):
        if row.card_id not in known_card_ids:
            raise ValueError(
                f"Unknown card_id {row.card_id} for {set_code}. "
                f"Add the official card first or fix source manifest row {index}."
            )

        temp_path: Path | None = None
        try:
            if row.image_url:
                temp_path = download_image(row.image_url, timeout=timeout, user_agent=user_agent)
            elif row.image_path:
                temp_path = copy_local_image(row.image_path)
            else:
                raise ValueError(f"Missing image source for {row.card_id}")
            sha = sha256_file(temp_path)
            if sha in seen_hashes:
                skipped += 1
                continue

            short_sha = sha[:12]
            raw_image_path = raw_set_dir / row.card_id / f"{row.card_id}__real-{short_sha}.jpg"
            normalize_to_jpeg(temp_path, raw_image_path)
            sidecar_path = raw_image_path.with_suffix(".json")
            now = datetime.now(timezone.utc).isoformat()
            write_sidecar(
                sidecar_path,
                {
                    "card_id": row.card_id,
                    "set_code": set_code,
                    "sha256": sha,
                    "source_url": row.image_url,
                    "source_path": str(row.image_path) if row.image_path else None,
                    "source": row.source,
                    "license": row.license,
                    "notes": row.notes,
                    "imported_at": now,
                    "intended_use": "card recognizer dataset; use only when you have rights or permission",
                },
            )

            training_image_path: Path | None = None
            if promote_to_training:
                training_image_path = cards_dir / set_code / f"{row.card_id}__real-{short_sha}.jpg"
                shutil.copyfile(raw_image_path, training_image_path)

            seen_hashes.add(sha)
            imported.append(
                ImportedImage(
                    card_id=row.card_id,
                    raw_image_path=raw_image_path,
                    training_image_path=training_image_path,
                    sha256=sha,
                    source_url=row.image_url,
                    source_path=str(row.image_path) if row.image_path else None,
                    source=row.source,
                    license=row.license,
                    notes=row.notes,
                )
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    append_manifest(raw_set_dir / "manifest.jsonl", imported)
    return imported, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import authorized real-life One Piece card photos into a raw dataset area, "
            "and optionally stage them as extra recognizer training samples."
        )
    )
    parser.add_argument(
        "--set",
        required=True,
        dest="set_code",
        help="One Piece set code or number, for example OP03, OP-03, or 3.",
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        type=Path,
        help=(
            "CSV or JSONL with card_id plus image_url or image_path columns. "
            "Optional columns: source, license, notes."
        ),
    )
    parser.add_argument("--cards-dir", type=Path, default=DEFAULT_CARDS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--promote-to-training",
        action="store_true",
        help="Also copy imported images to cards-dir/<SET>/CARDID__real-<hash>.jpg.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--user-agent",
        default="tcg-scanner-real-card-image-importer/1.0",
        help="User-Agent used for authorized image URL downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_code = normalize_set_code(args.set_code)
    rows = read_source_manifest(args.source_manifest)
    imported, skipped = import_source_rows(
        rows,
        set_code=set_code,
        cards_dir=args.cards_dir,
        raw_dir=args.raw_dir,
        promote_to_training=args.promote_to_training,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )

    print(f"Set: {set_code}")
    print(f"Read source rows: {len(rows)}")
    print(f"Imported images: {len(imported)}")
    print(f"Skipped duplicates: {skipped}")
    print(f"Raw output: {args.raw_dir / set_code}")
    if args.promote_to_training:
        print(f"Training samples staged in: {args.cards_dir / set_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
