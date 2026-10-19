from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from training.card_recognizer.canonical import preprocessing_contract_hash

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_card_gallery", ROOT / "archive/scripts/prepare_card_gallery.py")
gallery_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery_script)


class PrepareGalleryTests(unittest.TestCase):
    def test_gallery_maps_variants_and_refuses_unsafe_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "cards"
            (source_dir / "OP01").mkdir(parents=True)
            nested = source_dir / "OP01" / "OP01-002"
            nested.mkdir()
            rgb = np.full((20, 15, 3), (10, 40, 200), dtype=np.uint8)
            cv2.imwrite(str(source_dir / "OP01" / "OP01-001_p1.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / "OP01" / "OP01-001__sample.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(nested / "front.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(nested / "front.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            output = root / "out"
            manifest = gallery_script.prepare_gallery(source_dir, output)
            self.assertEqual(
                [r["card_id"] for r in manifest["references"]],
                ["OP01-001", "OP01-001_p1", "OP01-002", "OP01-002"],
            )
            self.assertTrue(all(r["preprocessing_contract_sha256"] == preprocessing_contract_hash() for r in manifest["references"]))
            self.assertFalse(manifest["contains_embeddings"])
            self.assertTrue((output / "crops/OP01/OP01-001_p1.jpg.png").is_file())
            self.assertTrue((output / "crops/OP01/OP01-002/front.jpg.png").is_file())
            self.assertTrue((output / "crops/OP01/OP01-002/front.png.png").is_file())
            saved = json.loads((output / "manifest.json").read_text())
            self.assertEqual(saved["artifact_type"], "canonical_reference_crops_only")
            with self.assertRaises(ValueError):
                gallery_script.prepare_gallery(source_dir, output)

    def test_gallery_rejects_lfs_pointer_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "cards"
            source_dir.mkdir()
            (source_dir / "OP01-001.jpg").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n")
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                gallery_script.prepare_gallery(source_dir, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
