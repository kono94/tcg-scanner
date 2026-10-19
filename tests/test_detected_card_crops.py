import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'video' / 'extract_detected_cards.py'
SPEC = importlib.util.spec_from_file_location('extract_detected_cards', MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def rotated_rectangle(center, width, height, degrees):
    angle = np.deg2rad(degrees)
    x_axis = np.array([np.cos(angle), np.sin(angle)]) * width / 2
    y_axis = np.array([-np.sin(angle), np.cos(angle)]) * height / 2
    cx, cy = center
    return np.array([np.array([cx, cy]) - x_axis - y_axis,
                     np.array([cx, cy]) + x_axis - y_axis,
                     np.array([cx, cy]) + x_axis + y_axis,
                     np.array([cx, cy]) - x_axis + y_axis])


class DetectedCardCropGeometryTests(unittest.TestCase):
    def test_portrait_corner_order_is_cyclic_for_unrotated_card(self):
        quad = np.array([[100, 50], [172, 50], [172, 150], [100, 150]], dtype=float)
        corners, card_width, card_height = module.validate_source_quad(quad, 400, 300, 50)
        np.testing.assert_allclose(corners, quad, atol=1e-5)
        self.assertAlmostEqual(card_width, 72)
        self.assertAlmostEqual(card_height, 100)

    def test_rotation_aligned_order_handles_landscape_quad_and_shuffled_vertices(self):
        # This geometry represents a portrait card turned 90 degrees in source pixels.
        quad = rotated_rectangle((240, 180), width=72, height=300, degrees=90)
        shuffled = quad[[2, 0, 3, 1]]
        corners, card_width, card_height = module.validate_source_quad(shuffled, 500, 400, 60)
        self.assertGreater(card_height, card_width)
        self.assertAlmostEqual(card_width, 72, delta=.1)
        self.assertAlmostEqual(card_height, 300, delta=.1)
        self.assertGreater(module._signed_area(corners), 0)

    def test_rejects_outside_source_bounds(self):
        quad = np.array([[-1, 10], [80, 10], [80, 130], [-1, 130]], dtype=float)
        with self.assertRaisesRegex(ValueError, 'outside source-frame bounds'):
            module.validate_source_quad(quad, 200, 200, 20)

    def test_rejects_card_below_source_short_edge_minimum(self):
        quad = np.array([[10, 10], [80, 10], [80, 100], [10, 100]], dtype=float)
        with self.assertRaisesRegex(ValueError, 'below 80px minimum'):
            module.validate_source_quad(quad, 200, 200, 80)

    def test_rejects_degenerate_or_nonfinite_corner_sets(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            module.portrait_aligned_corners(np.array([[0, 0], [1, 0], [1, np.nan], [0, 1]]))
        with self.assertRaisesRegex(ValueError, 'area is too small'):
            module.portrait_aligned_corners(np.array([[0, 0], [10, 0], [20, 0], [30, 0]]))

    def test_rejects_nonconvex_corner_set(self):
        quad = np.array([[0, 0], [100, 0], [50, 50], [0, 100]], dtype=float)
        with self.assertRaisesRegex(ValueError, 'convex quadrilateral'):
            module.portrait_aligned_corners(quad)

    def test_crop_dimensions_cap_at_1024_and_do_not_upscale(self):
        self.assertEqual(module.crop_dimensions(240, 1000), (240, 1000, 'source_scale_no_upscale'))
        self.assertEqual(module.crop_dimensions(500, 2000), (256, 1024, 'downsampled_to_1024_cap'))


if __name__ == '__main__':
    unittest.main()
