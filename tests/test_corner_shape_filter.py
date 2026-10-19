import unittest

import numpy as np

from scripts.detection.evaluate_corners_real import plausible_card


class CornerShapeFilterTests(unittest.TestCase):
    def test_perspective_foreshortened_card_is_not_rejected_for_being_square(self):
        for height in (63, 66, 88):
            quad = np.array([[0, 0], [63, 0], [63, height], [0, height]], dtype=float)
            self.assertTrue(plausible_card(quad, np.array([0, 0, 63, height])))

    def test_crossed_and_excessively_stretched_quads_still_reject(self):
        box = np.array([0, 0, 63, 88])
        for quad in ([[0, 0], [63, 88], [63, 0], [0, 88]],
                     [[0, 0], [20, 0], [20, 88], [0, 88]]):
            self.assertFalse(plausible_card(np.array(quad, dtype=float), box))

    def test_small_quad_cannot_fill_an_unrelated_large_detection_box(self):
        quad = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=float)
        self.assertFalse(plausible_card(quad, np.array([0, 0, 200, 200])))
