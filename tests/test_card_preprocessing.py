from __future__ import annotations

import unittest

import numpy as np

from training.recognition.canonical import (
    canonicalize_card, prepare_encoder_input, preprocessing_contract_hash,
)

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


if __name__ == "__main__":
    unittest.main()
