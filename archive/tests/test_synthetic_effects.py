import unittest

import numpy as np

from training.card_detector.synthetic_effects import apply_camera_effects, make_background


class SyntheticEffectsTests(unittest.TestCase):
    def test_backgrounds_are_seeded_and_have_requested_shape(self):
        for style in ("binder", "desk", "plain"):
            first = make_background(160, 240, np.random.default_rng(12), style, [(20, 30, 80, 120)])
            again = make_background(160, 240, np.random.default_rng(12), style, [(20, 30, 80, 120)])
            self.assertEqual(first.shape, (240, 160, 3))
            self.assertEqual(first.dtype, np.uint8)
            np.testing.assert_array_equal(first, again)

    def test_binder_pockets_are_drawn_and_bounds_are_clipped(self):
        base = make_background(160, 200, np.random.default_rng(3), "binder")
        page = make_background(160, 200, np.random.default_rng(3), "binder", [(-5, 20, 90, 140)])
        self.assertGreater(np.count_nonzero(base != page), 0)
        self.assertEqual(page.shape, base.shape)

    def test_effects_are_deterministic_preserve_geometry_and_do_not_mutate_input(self):
        image = np.full((96, 128, 3), 140, dtype=np.uint8)
        image[20:75, 35:85] = (180, 100, 60)
        before = image.copy()
        a, pa = apply_camera_effects(image, np.random.default_rng(91), 1.0)
        b, pb = apply_camera_effects(image, np.random.default_rng(91), 1.0)
        self.assertEqual(a.shape, image.shape)
        self.assertEqual(a.dtype, np.uint8)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(pa, pb)
        np.testing.assert_array_equal(image, before)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(np.mean(np.abs(a.astype(float) - image.astype(float))), 0)
        self.assertIsInstance(pa["jpeg_quality"], int)

    def test_zero_strength_retains_size_and_reports_no_added_noise_or_glare(self):
        image = np.full((24, 31, 3), 90, dtype=np.uint8)
        out, params = apply_camera_effects(image, np.random.default_rng(0), 0)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(params["noise_sigma"], 0)
        self.assertEqual(params["glare_alpha"], 0)
        self.assertIsNone(params["jpeg_quality"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            make_background(0, 10, np.random.default_rng(1), "plain")
        with self.assertRaises(ValueError):
            make_background(10, 10, np.random.default_rng(1), "unknown")
        with self.assertRaises(ValueError):
            apply_camera_effects(np.zeros((10, 10), dtype=np.uint8), np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
