from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/fetch_pokemon_references.py"
SPEC = importlib.util.spec_from_file_location("fetch_pokemon_references", MODULE_PATH)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 32), (30, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeRequests:
    def __init__(self):
        self.calls = []

    def get(self, url, headers, timeout):
        self.calls.append((url, timeout))
        if url.endswith("/sets/swsh3"):
            return FakeResponse({"id": "swsh3", "cards": [{"id": "swsh3-007", "name": "Wimpod", "localId": "007"}]})
        if url.endswith("/cards/swsh3-007"):
            return FakeResponse({
                "id": "swsh3-007", "name": "Wimpod", "localId": "007",
                "image": "https://assets.tcgdex.net/en/swsh/swsh3/007",
                "set": {"id": "swsh3", "name": "Darkness Ablaze", "cardCount": {"official": 189}},
                "variants": {"normal": True, "reverse": True, "holo": False, "firstEdition": False},
            })
        if url.endswith("/007/high.png"):
            return FakeResponse(content=png_bytes())
        raise AssertionError(f"Unexpected request URL: {url}")


class PokemonReferenceImportTests(unittest.TestCase):
    def test_preserves_zero_padded_and_alphanumeric_printed_numbers(self):
        self.assertEqual(fetcher.printed_number_fields("007", 189), {
            "printed_number": "007", "printed_numerator": 7, "official_total": 189
        })
        self.assertEqual(fetcher.printed_number_fields("TG01", 189), {
            "printed_number": "TG01", "printed_numerator": None, "official_total": 189
        })

    def test_maps_catalog_variant_evidence_without_claiming_physical_finish(self):
        record = fetcher.catalog_record(
            {"id": "swsh3-007", "localId": "007", "name": "Wimpod"},
            {"id": "swsh3-007", "localId": "007", "set": {"id": "swsh3", "name": "Darkness Ablaze", "cardCount": {"official": 189}},
             "variants": {"normal": True, "reverse": True}},
            "en", "swsh3", "https://assets.tcgdex.net/en/swsh/swsh3/007/high.png", "abc123")
        self.assertEqual(record["catalog_identity"], "pokemon:en:swsh3:007")
        self.assertEqual(record["physical_finish"], "unresolved")
        self.assertEqual(record["available_catalog_variants"], {"normal": True, "reverse": True})
        self.assertEqual(record["training_promotion"], "not_promoted")

    def test_acquires_bounded_image_and_writes_hash_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "staged"
            fake = FakeRequests()
            manifest = fetcher.acquire("swsh3", "en", 1, output, fake)
            self.assertEqual(manifest["saved_count"], 1)
            self.assertEqual([call[0] for call in fake.calls], [
                "https://api.tcgdex.net/v2/en/sets/swsh3",
                "https://api.tcgdex.net/v2/en/cards/swsh3-007",
                "https://assets.tcgdex.net/en/swsh/swsh3/007/high.png",
            ])
            image_path = output / "cards/swsh3/swsh3-007.png"
            sidecar = json.loads(image_path.with_suffix(".json").read_text())
            self.assertEqual(sidecar["image_sha256"], fetcher.sha256_bytes(png_bytes()))
            self.assertTrue((output / "manifest.json").is_file())

    def test_refuses_nonempty_output_and_path_traversal_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "occupied"
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            fake = FakeRequests()
            with self.assertRaisesRegex(ValueError, "new or empty"):
                fetcher.acquire("swsh3", "en", 1, output, fake)
            with self.assertRaisesRegex(ValueError, "Invalid set id"):
                fetcher.acquire("../swsh3", "en", 1, Path(tmp) / "fresh", fake)
            self.assertEqual(fake.calls, [])
            self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_rejects_bad_image_response_and_cleans_staging_directory(self):
        class BadImageRequests(FakeRequests):
            def get(self, url, headers, timeout):
                if url.endswith("/007/high.png"):
                    self.calls.append((url, timeout))
                    return FakeResponse(content=b"not an image")
                return super().get(url, headers, timeout)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "staged"
            with self.assertRaisesRegex(ValueError, "valid PNG"):
                fetcher.acquire("swsh3", "en", 1, output, BadImageRequests())
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
