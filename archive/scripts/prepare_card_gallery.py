#!/usr/bin/env python3
"""Prepare canonical reference crops; this does not create embeddings or a search index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.card_recognizer.canonical import (  # noqa: E402
    PREPROCESSING_VERSION, canonicalize_card, preprocessing_contract,
    preprocessing_contract_hash,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _card_id(path: Path, cards_dir: Path) -> str:
    # Training's CARDID__sample convention means another reference for CARDID.
    relative = path.relative_to(cards_dir)
    identity = relative.parts[-2] if len(relative.parts) >= 3 else path.stem
    return identity.split("__", 1)[0]


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source(path: Path) -> None:
    with path.open("rb") as source:
        if source.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"unresolved Git LFS pointer: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"corrupt or unsupported image: {path}") from exc


def _decode_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Exception as exc:
        raise ValueError(f"failed to decode image: {path}") from exc


def prepare_gallery(cards_dir: Path, output: Path, limit: int | None = None) -> dict:
    cards_dir, output = cards_dir.resolve(), output.resolve()
    if not cards_dir.is_dir():
        raise ValueError(f"cards directory does not exist: {cards_dir}")
    if output == cards_dir or cards_dir in output.parents or output in cards_dir.parents:
        raise ValueError("output and input directories must not contain one another")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing nonempty output directory: {output}")
    sources = sorted(p for p in cards_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        sources = sources[:limit]
    if not sources:
        raise ValueError(f"no supported image files found under {cards_dir}")

    for source in sources:
        _verify_source(source)

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        card_id = _card_id(source, cards_dir)
        rel = source.relative_to(cards_dir)
        # Include the source suffix so same-stem JPG/PNG references cannot collide.
        destination = output / "crops" / rel.parent / f"{source.name}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb = _decode_rgb(source)
        crop = canonicalize_card(rgb)
        if not cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write canonical crop: {destination}")
        records.append({
            "card_id": card_id,
            "source_path": rel.as_posix(),
            "source_sha256": _source_digest(source),
            "canonical_crop": destination.relative_to(output).as_posix(),
            "canonical_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_contract_sha256": preprocessing_contract_hash(),
            "source_role": "digital_reference_assumed_full_face_and_upright",
        })
    manifest = {
        "artifact_type": "canonical_reference_crops_only",
        "contains_embeddings": False,
        "contains_search_index": False,
        "note": "Canonical reference images only; full-face bounds and upright orientation are assumed from digital source images and have not been independently verified.",
        "preprocessing": preprocessing_contract(),
        "preprocessing_contract_sha256": preprocessing_contract_hash(),
        "references": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-dir", type=Path, default=Path("datasets/card_recognizer/cards"))
    parser.add_argument("--output", type=Path, default=Path("runs/card_recognizer/gallery_prepared"))
    parser.add_argument("--limit", type=int, help="prepare only the first N sorted references (smoke checks)")
    args = parser.parse_args()
    manifest = prepare_gallery(args.cards_dir, args.output, args.limit)
    print(f"Prepared {len(manifest['references'])} canonical reference crops in {args.output}")


if __name__ == "__main__":
    main()
