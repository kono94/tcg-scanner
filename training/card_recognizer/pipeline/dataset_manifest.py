from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm

from .paths import PROJECT_ROOT, resolve_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SAMPLE_SEPARATOR = "__"


@dataclass(frozen=True)
class CardRecord:
    image_path: str
    label: str
    card_id: str
    set_code: str
    metadata_path: str | None = None


def infer_card_id(image_path: Path) -> str:
    return image_path.stem.split(SAMPLE_SEPARATOR, 1)[0]


def label_for_card_id(card_id: str) -> str:
    return f"{card_id}.jpg"


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_training_records(cards_dir: str | Path) -> list[CardRecord]:
    cards_root = resolve_path(cards_dir)
    if not cards_root.exists():
        raise FileNotFoundError(f"Card image directory does not exist: {cards_root}")

    records: list[CardRecord] = []
    image_paths = sorted(path for path in cards_root.rglob("*") if path.is_file())
    for image_path in tqdm(image_paths, desc=f"Discovering train cards in {cards_root.name}", unit="file"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parts = image_path.relative_to(cards_root).parts
        if len(relative_parts) < 2:
            continue
        set_code = relative_parts[0]
        if len(relative_parts) >= 3:
            card_id = relative_parts[1]
        else:
            card_id = infer_card_id(image_path)
        metadata_candidates = [
            cards_root / set_code / f"{card_id}.json",
            image_path.parent / f"{card_id}.json",
            image_path.parent / "metadata.json",
            image_path.with_suffix(".json"),
        ]
        metadata_path = next((path for path in metadata_candidates if path.exists()), None)
        records.append(
            CardRecord(
                image_path=project_relative(image_path),
                label=label_for_card_id(card_id),
                card_id=card_id,
                set_code=set_code,
                metadata_path=project_relative(metadata_path) if metadata_path else None,
            )
        )
    if not records:
        raise ValueError(f"No card images found under {cards_root}")
    return sorted(records, key=lambda record: (record.set_code, record.label, record.image_path))


def _label_lookup(records: list[CardRecord]) -> dict[str, CardRecord]:
    lookup: dict[str, CardRecord] = {}
    for record in records:
        lookup[record.label] = record
        lookup[record.card_id] = record
        lookup[Path(record.label).stem] = record
    return lookup


def _resolve_eval_record(image_path: Path, eval_root: Path, lookup: dict[str, CardRecord]) -> CardRecord:
    candidates: list[str] = []
    parent = image_path.parent
    while parent != eval_root and eval_root in parent.parents:
        candidates.append(parent.name)
        parent = parent.parent
    candidates.extend([image_path.name, image_path.stem])
    if SAMPLE_SEPARATOR in image_path.stem:
        candidates.append(image_path.stem.split(SAMPLE_SEPARATOR, 1)[0])

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
        stem = Path(candidate).stem
        if stem in lookup:
            return lookup[stem]

    expected = "datasets/card_recognizer/manual_eval/OP01-001/IMG_0001.jpg"
    raise ValueError(
        f"Could not infer card label for manual eval image {image_path}. "
        f"Put phone images in a folder named after the card id or label, for example {expected}."
    )


def discover_manual_eval_records(eval_dir: str | Path, train_records: list[CardRecord], *, required: bool) -> list[CardRecord]:
    eval_root = resolve_path(eval_dir)
    if not eval_root.exists():
        if required:
            raise FileNotFoundError(
                f"Manual evaluation directory does not exist: {eval_root}. "
                "Create it and add phone images under folders named by card id, for example OP01-001/IMG_0001.jpg."
            )
        return []

    lookup = _label_lookup(train_records)
    records: list[CardRecord] = []
    image_paths = sorted(path for path in eval_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in tqdm(image_paths, desc=f"Discovering manual eval images in {eval_root.name}", unit="file"):
        source = _resolve_eval_record(image_path, eval_root, lookup)
        records.append(
            CardRecord(
                image_path=project_relative(image_path),
                label=source.label,
                card_id=source.card_id,
                set_code=source.set_code,
                metadata_path=source.metadata_path,
            )
        )

    if required and not records:
        raise ValueError(
            f"No manual evaluation images found under {eval_root}. "
            "Add phone images under folders named by card id, for example OP01-001/IMG_0001.jpg."
        )
    return sorted(records, key=lambda record: (record.set_code, record.label, record.image_path))


def build_dataset_manifest(
    *,
    cards_dir: str | Path,
    eval_dir: str | Path,
    eval_required: bool,
) -> dict:
    train_records = discover_training_records(cards_dir)
    eval_records = discover_manual_eval_records(eval_dir, train_records, required=eval_required)
    labels = sorted({record.label for record in train_records})
    return {
        "version": 1,
        "description": (
            "Classifier dataset manifest. Train uses every scraped card sample from cards_dir. "
            "Validation uses manually captured phone-image samples from eval_dir."
        ),
        "cards_dir": project_relative(resolve_path(cards_dir)),
        "eval_dir": project_relative(resolve_path(eval_dir)),
        "labels": labels,
        "datasets": {
            "train": [asdict(record) for record in train_records],
            "eval": [asdict(record) for record in eval_records],
        },
    }


def load_dataset_manifest(path: str | Path) -> dict:
    manifest_path = resolve_path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataset_manifest(payload: dict, output_path: str | Path, csv_output_path: str | Path | None = None) -> None:
    json_path = resolve_path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    if csv_output_path is None:
        csv_output_path = json_path.with_suffix(".csv")
    csv_path = resolve_path(csv_output_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "image_path", "label", "card_id", "set_code", "metadata_path"],
        )
        writer.writeheader()
        for dataset_name, records in payload["datasets"].items():
            for record in tqdm(records, desc=f"Writing dataset CSV {dataset_name}", unit="row"):
                writer.writerow({"dataset": dataset_name, **record})
