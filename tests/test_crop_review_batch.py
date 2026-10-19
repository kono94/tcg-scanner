import unittest

import cv2
import numpy as np

from scripts.curation.propose_crop_review_batch import (
    map_crop_quad_to_source,
    match_features,
    rectification_eligibility,
)


class CropReviewBatchTests(unittest.TestCase):
    def test_crop_quad_maps_with_independent_source_offsets_and_axis_scales(self):
        crop_quad = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=np.float32)

        mapped = map_crop_quad_to_source(crop_quad, (200, 300), (100, 200, 500, 800))

        np.testing.assert_allclose(
            mapped,
            [[100, 200], [500, 200], [500, 800], [100, 800]],
        )

    def test_preview_can_pass_spatial_guards_when_strict_card_shape_gate_fails(self):
        # A broad, non-card-shaped homography is deliberately rejected by the
        # strict matcher aspect heuristic, while still being useful as a review
        # preview if its support and source-frame geometry are sound.
        ref_points = [(float(x), float(y)) for y in (0, 33, 66, 99) for x in (0, 33, 66, 99)]
        query_points = [(2 * x + 5, .6 * y + 20) for x, y in ref_points]
        ref_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in ref_points]
        query_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in query_points]
        descriptors = np.ones((len(ref_points), 8), dtype=np.float32)

        class ExactMatcher:
            def knnMatch(self, _ref, _query, k=2):
                return [[cv2.DMatch(i, i, .1), cv2.DMatch(i, i, 1.0)]
                        for i in range(len(ref_points))]

        strict = match_features(
            (ref_keypoints, descriptors, (1, 1), 100, 100),
            (query_keypoints, descriptors.copy(), (1, 1), 220, 100),
            ExactMatcher(),
        )

        self.assertFalse(strict['accepted'])
        self.assertIn('implausible_card_quad', strict['rejection_reasons'])
        quad_source = np.asarray(strict['quad_query_native_xy'], dtype=np.float32) + [10, 10]
        preview_ok, reasons = rectification_eligibility(
            strict, quad_source, (300, 300, 3))

        self.assertTrue(preview_ok, reasons)

    def test_malformed_and_zero_descriptors_return_rejections_without_cv_errors(self):
        keypoints = [cv2.KeyPoint(float(i), float(i), 1) for i in range(8)]
        cases = [
            ('one-dimensional', np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32)),
            ('zero', np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)),
            ('dimension-mismatch', np.ones((8, 8), dtype=np.float32), np.ones((8, 7), dtype=np.float32)),
            ('non-finite', np.full((8, 8), np.nan, dtype=np.float32), np.ones((8, 8), dtype=np.float32)),
        ]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        for name, ref_desc, query_desc in cases:
            with self.subTest(name=name):
                result = match_features(
                    (keypoints, ref_desc, (1, 1), 100, 100),
                    (keypoints, query_desc, (1, 1), 100, 100),
                    matcher,
                )
                self.assertFalse(result['accepted'])
                self.assertIn('invalid_or_zero_descriptors', result['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
