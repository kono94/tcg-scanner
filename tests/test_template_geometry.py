import unittest

import cv2
import numpy as np

from scripts.curation.verify_card_geometry import source_resolution_crop, verify_homography


def textured_card(seed=4):
    rng = np.random.default_rng(seed)
    image = np.full((620, 440, 3), 245, np.uint8)
    for y in range(28, 590, 36):
        for x in range(24, 420, 34):
            color = tuple(int(v) for v in rng.integers(25, 225, 3))
            cv2.rectangle(image, (x, y), (x + 18, y + 18), color, -1)
    for i in range(14):
        p = tuple(int(v) for v in rng.integers([15, 15], [425, 605]))
        cv2.circle(image, p, int(rng.integers(3, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return image


class TemplateGeometryTests(unittest.TestCase):
    def test_ocr_crop_preserves_source_detail_and_rejects_outside_pixels(self):
        image = textured_card()
        quad = np.float32([[0, 0], [439, 0], [439, 619], [0, 619]])
        crop = source_resolution_crop(image, quad)
        self.assertEqual(crop.shape, (619, 439, 3))
        self.assertEqual(source_resolution_crop(image, quad, max_edge=300).shape, (300, 212, 3))
        quad[0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'outside source'):
            source_resolution_crop(image, quad)

    def test_known_perspective_transform_produces_card_quad(self):
        ref = textured_card()
        h, w = 700, 600
        corners = np.float32([[75, 40], [500, 72], [470, 655], [54, 620]])
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [439, 0], [439, 619], [0, 619]]), corners)
        query = cv2.warpPerspective(ref, H, (w, h), borderValue=(255, 255, 255))
        out = verify_homography(ref, query)
        self.assertTrue(out['accepted'], out['rejection_reasons'])
        np.testing.assert_allclose(np.asarray(out['quad_xy']), corners, atol=6)
        self.assertGreater(out['reference_coverage'], .12)
        self.assertGreater(out['query_coverage'], .12)

    def test_rectification_recovers_card_aspect_without_upsampling_source_edges(self):
        image = np.zeros((500, 600, 3), np.uint8)
        quad = np.float32([[10, 10], [500, 40], [470, 420], [30, 400]])
        crop = source_resolution_crop(image, quad, aspect_ratio=224/312)
        height, width = crop.shape[:2]
        self.assertAlmostEqual(width/height, 224/312, delta=.004)
        edges = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
        self.assertLessEqual(width, min(edges[0], edges[2]))
        self.assertLessEqual(height, min(edges[1], edges[3]))
        with self.assertRaises(ValueError):
            source_resolution_crop(image, quad, aspect_ratio=float('nan'))

    def test_artwork_subpatch_match_is_rejected_for_low_spatial_coverage(self):
        ref = textured_card(9)
        # Only a central art window is copied into a query crop; it cannot justify card edges.
        query = np.full_like(ref, 255)
        query[180:440, 115:325] = ref[180:440, 115:325]
        out = verify_homography(ref, query)
        self.assertFalse(out['accepted'])
        self.assertTrue(any('coverage' in r or 'spatially_spread' in r for r in out['rejection_reasons']))

    def test_featureless_degenerate_inputs_reject_cleanly(self):
        ref = np.full((620, 440, 3), 128, np.uint8)
        query = ref.copy()
        out = verify_homography(ref, query)
        self.assertFalse(out['accepted'])
        self.assertIn('insufficient_descriptors', out['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
