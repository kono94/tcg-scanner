from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from training.card_recognizer.canonical import (
    canonicalize_card, prepare_encoder_input, preprocessing_contract_hash,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_card_gallery", ROOT / "scripts/prepare_card_gallery.py")
gallery_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery_script)


class CanonicalPreprocessingTests(unittest.TestCase):
    def test_identity_bounds_and_semantic_corner_warp(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:10, :15] = (255, 0, 0)
        image[:10, 15:] = (0, 255, 0)
        image[10:, 15:] = (0, 0, 255)
        image[10:, :15] = (255, 255, 0)
        identity = canonicalize_card(image, width=30, height=20)
        np.testing.assert_array_equal(identity, image)
        # Source card is rotated in the image; semantic TL,TR,BR,BL still map to upright output.
        rotated_order = np.array([[29, 0], [29, 19], [0, 19], [0, 0]], dtype=np.float32)
        result = canonicalize_card(image, rotated_order, width=30, height=20)
        self.assertEqual(tuple(result[0, 0]), (0, 255, 0))
        self.assertEqual(tuple(result[0, -1]), (0, 0, 255))

    def test_encoder_padding_normalization_and_contract_hash(self):
        image = np.full((4, 2, 3), 128, dtype=np.uint8)
        tensor = prepare_encoder_input(image, size=8)
        self.assertEqual(tensor.shape, (3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[:, :, :2], np.zeros((3, 8, 2), dtype=np.float32))
        np.testing.assert_array_equal(tensor[:, :, 6:], np.zeros((3, 8, 2), dtype=np.float32))
        expected = (128 / 255 - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        expected_region = np.broadcast_to(expected[:, None, None], (3, 4, 4))
        np.testing.assert_allclose(tensor[:, 2:6, 2:6], expected_region, atol=1e-6)
        self.assertEqual(len(preprocessing_contract_hash()), 64)

    def test_rejects_invalid_images_corners_and_normalization(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            canonicalize_card(image, np.zeros((4, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            canonicalize_card(image.astype(np.float32))
        with self.assertRaises(ValueError):
            canonicalize_card(np.zeros((1, 5, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            prepare_encoder_input(image, std=(1, 0, 1))

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
