from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


PRICE_API_URL = "https://api.dotgg.gg/cgfw/getcards?game=onepiece&mode=indexed"
DEFAULT_CARD_INDEX_PATH = Path("tcg-scanner-app/tcg-scanner-app/Resources/card_index.json")


def fetch_onepiece_price_rows() -> tuple[list[str], list[list[object]]]:
    request = Request(PRICE_API_URL, headers={"User-Agent": "tcg-scanner-price-snapshot"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)

    return data["names"], data["data"]


def optional_price(value: object) -> float | None:
    if value in (None, ""):
        return None
    price = float(value)
    return price if price > 0 else None


def selected_price(row: list[object], indexes: dict[str, int]) -> tuple[float | None, str | None]:
    for field_name in ("foilPrice", "price", "cmFoilPrice", "cmPrice"):
        price = optional_price(row[indexes[field_name]])
        if price is not None:
            return price, f"dotgg:{field_name}"

    return None, None


def display_price(price: float | None) -> str | None:
    if price is None:
        return None
    return f"${price:.2f}"


def update_card_index_prices(card_index_path: Path = DEFAULT_CARD_INDEX_PATH) -> None:
    names, rows = fetch_onepiece_price_rows()
    indexes = {name: index for index, name in enumerate(names)}

    prices_by_id = {}
    for row in rows:
        card_id = row[indexes["id"]]
        price, source = selected_price(row, indexes)
        prices_by_id[card_id] = {
            "displayPrice": display_price(price),
            "priceSource": source,
        }

    with card_index_path.open("r", encoding="utf-8") as handle:
        cards = json.load(handle)

    updated_count = 0
    priced_count = 0
    for card in cards:
        price_snapshot = prices_by_id.get(card["id"])
        if price_snapshot is None:
            card["displayPrice"] = None
            card["priceSource"] = None
            continue

        card["displayPrice"] = price_snapshot["displayPrice"]
        card["priceSource"] = price_snapshot["priceSource"]
        updated_count += 1
        if card["displayPrice"] is not None:
            priced_count += 1

    with card_index_path.open("w", encoding="utf-8") as handle:
        json.dump(cards, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"Updated {updated_count} card index rows; {priced_count} have prices.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh One Piece TCG card prices.")
    parser.add_argument(
        "--card-index",
        type=Path,
        default=DEFAULT_CARD_INDEX_PATH,
        help="Also write bundled app price snapshots to this card_index.json file.",
    )
    args = parser.parse_args()

    update_card_index_prices(args.card_index)

if __name__ == '__main__':
    main()
