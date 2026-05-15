#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm


def build_index(cards_dir: Path) -> list[dict[str, str | None]]:
    cards: list[dict[str, str | None]] = []
    json_paths = sorted(cards_dir.glob("*/*.json"))
    print(f"Building card index from {len(json_paths)} metadata files in {cards_dir}...")
    for json_path in tqdm(json_paths, desc="Reading card metadata", unit="file"):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        card_id = data.get("id") or json_path.stem
        name = data.get("name") or "Unknown"
        image_path = json_path.with_suffix(".jpg")
        cards.append(
            {
                "id": card_id,
                "name": name,
                "imageLabel": image_path.name,
                "displayPrice": data.get("display_price"),
                "priceSource": data.get("price_source"),
            }
        )

    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact iOS card metadata index.")
    parser.add_argument("--cards-dir", default="datasets/card_recognizer/cards", type=Path)
    parser.add_argument(
        "--output",
        default="tcg-scanner-app/tcg-scanner-app/Resources/card_index.json",
        type=Path,
    )
    args = parser.parse_args()

    cards = build_index(args.cards_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing app card index to {args.output}...")
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(cards, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"Wrote {len(cards)} cards to {args.output}")


if __name__ == "__main__":
    main()
