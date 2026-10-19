from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/catalog/sync_pokemon_gallery.py"
SPEC = importlib.util.spec_from_file_location("sync_pokemon_gallery", SCRIPT)
syncer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(syncer)


def webp_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 60), (80, 120, 160)).save(buffer, format="WEBP")
    return buffer.getvalue()


def brief_cards():
    return [
        {"id": "setA-001", "localId": "001", "name": "Alpha 1", "image": "https://assets.tcgdex.net/en/setA/setA/001"},
        {"id": "setA-002", "localId": "002", "name": "Alpha 2", "image": "https://assets.tcgdex.net/en/setA/setA/002"},
        {"id": "setB-01a", "localId": "01a", "name": "Beta 1", "image": "https://assets.tcgdex.net/en/setB/setB/01a"},
        {"id": "setC-7", "localId": "7", "name": "Gamma 7", "image": "https://assets.tcgdex.net/en/setC/setC/7"},
        {"id": "odd-card", "localId": "promotional", "name": "Odd", "image": None},
    ]


class FakeResponse:
    def __init__(self, payload=None, data=b""):
        self._payload = payload
        self.data = data
        self.content = json.dumps(payload).encode() if payload is not None else data
        self.headers = {"content-type": "image/webp", "content-length": str(len(data))}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start:start + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, cards=None, image=None):
        self.cards = brief_cards() if cards is None else cards
        self.image = webp_bytes() if image is None else image
        self.calls = []

    def get(self, url, headers, timeout, stream=False):
        self.calls.append((url, timeout, stream))
        if url.endswith("/v2/en/cards"):
            return FakeResponse(payload=self.cards)
        if url.endswith("/v2/en/series/tcgp"):
            return FakeResponse(payload={"id": "tcgp", "sets": [{"id": "setC"}]})
        if url.startswith("https://assets.tcgdex.net/"):
            return FakeResponse(data=self.image)
        raise AssertionError(f"Unexpected URL (per-card detail calls are not expected): {url}")


class PokemonGallerySyncTests(unittest.TestCase):
    def test_correct_provider_namespace_and_conservative_set_derivation(self):
        records = syncer.normalize_catalog(brief_cards())
        self.assertEqual(records[0]["catalog_identity"], "tcgdex:en:setA-001")
        self.assertEqual(records[0]["set_id"], "setA")
        self.assertEqual(records[0]["local_id"], "001")
        self.assertEqual(records[2]["set_id"], "setB")
        self.assertEqual(records[-1]["set_id"], None)
        self.assertEqual(records[-1]["image_status"], "unavailable")
        self.assertEqual(records[0]["physical_finish"], "unresolved")

    def test_dotted_set_ids_are_safe_but_traversal_is_rejected(self):
        self.assertEqual(syncer.parse_set_id("sv03.5-001", "001"), "sv03.5")
        self.assertEqual(syncer.parse_set_id("me02.5-001", "001"), "me02.5")
        self.assertEqual(syncer.parse_set_id("../escape-001", "001"), None)
        self.assertEqual(syncer.parse_set_id("..-001", "001"), None)
        with self.assertRaises(ValueError):
            syncer.safe_segment("../escape", "set id")

    def test_fair_round_robin_spreads_selection_across_sets(self):
        records = syncer.normalize_catalog(brief_cards())
        ordered = syncer.fair_round_robin(records)
        self.assertEqual([r["set_id"] for r in ordered[:4]], ["setA", "setB", "setC", "setA"])

    def test_bounded_sync_resumes_and_writes_per_id_sidecars_without_detail_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gallery"
            fake = FakeSession()
            first = syncer.sync(max_images=2, max_download_mib=1, min_free_gib=0, output=out, session=fake)
            self.assertEqual(first["catalog_count"], 5)
            self.assertEqual(first["downloaded_count"], 2)
            self.assertFalse(first["gallery_images_complete"])
            first_set_ids = {Path(r["image_path"]).parts[1] for r in json.loads((out / "catalog.json").read_text()) if r["image_status"] == "downloaded"}
            self.assertEqual(len(first_set_ids), 2)
            self.assertTrue((out / "cards/setA/setA-001.json").is_file())
            self.assertTrue(all(not "/cards/" in url for url, _, _ in fake.calls))

            second = syncer.sync(max_images=4, max_download_mib=1, min_free_gib=0, output=out, session=fake)
            self.assertEqual(second["downloaded_count"], 3)
            self.assertEqual(second["downloaded_this_run"], 1)
            self.assertTrue(second["gallery_images_complete"])
            self.assertEqual(second["excluded_digital_download_count"], 1)
            self.assertEqual(second["excluded_unknown_domain_count"], 1)
            state = json.loads((out / "sync_state.json").read_text())
            self.assertEqual(state["downloaded_bytes_total"], len(webp_bytes()) * 3)

    def test_hard_byte_budget_does_not_write_an_oversized_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gallery"
            result = syncer.sync(max_images=4, max_download_mib=0, min_free_gib=0, output=out, session=FakeSession())
            self.assertEqual(result["downloaded_count"], 0)
            self.assertFalse(result["gallery_images_complete"])
            self.assertEqual(result["downloaded_bytes_total"], 0)
            self.assertFalse(any((out / "cards").rglob("*.webp")))

    def test_full_catalog_snapshot_is_not_rewritten_for_every_image(self):
        cards = [{"id": f"setA-{i:03d}", "localId": f"{i:03d}", "name": f"Card {i}",
                  "image": f"https://assets.tcgdex.net/en/setA/setA/{i:03d}"} for i in range(1, 102)]
        fake = FakeSession(cards=cards)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gallery"
            real_atomic_json = syncer.atomic_json
            catalog_writes = []

            def count_catalog_writes(path, value):
                if Path(path).name == "catalog.json":
                    catalog_writes.append(path)
                return real_atomic_json(path, value)

            with mock.patch.object(syncer, "atomic_json", side_effect=count_catalog_writes):
                result = syncer.sync(max_images=101, max_download_mib=1, min_free_gib=0,
                                     output=out, session=fake)
            self.assertEqual(result["downloaded_count"], 101)
            self.assertLessEqual(len(catalog_writes), 3)
            self.assertEqual(len((out / "progress.jsonl").read_text().splitlines()), 101)
            state = json.loads((out / "sync_state.json").read_text())
            self.assertEqual(state["catalog_source_sha256"], result["catalog_source_sha256"])
            self.assertEqual(state["pocket_series_source_sha256"], result["pocket_series_source_sha256"])

    def test_rejects_unsafe_ids_and_nonempty_unrecognized_output(self):
        with self.assertRaisesRegex(ValueError, "Unsafe catalog id"):
            syncer.normalize_catalog([{"id": "../bad", "localId": "1"}])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gallery"
            out.mkdir()
            (out / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                syncer.sync(output=out, session=FakeSession())
            self.assertEqual((out / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
