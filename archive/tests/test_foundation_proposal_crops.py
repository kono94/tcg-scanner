import unittest

from scripts.crop_foundation_proposals import clipped_aabb, iou, recognition_geometry


class FoundationProposalCropTests(unittest.TestCase):
    def test_source_resolution_aabb_is_floor_ceil_clipped_without_rotation(self):
        clipped, rounded, was_clipped = clipped_aabb([10.2, 20.4, 100.1, 180.2], 200, 300)
        self.assertEqual(clipped, [10, 20, 101, 181])
        self.assertEqual(rounded, [10, 20, 100, 180])
        self.assertFalse(was_clipped)

    def test_out_of_frame_edges_are_clipped_and_recorded(self):
        clipped, _, was_clipped = clipped_aabb([-4.2, 10, 50, 310], 200, 300)
        self.assertEqual(clipped, [0, 10, 50, 300])
        self.assertTrue(was_clipped)

    def test_empty_or_nonfinite_bounds_reject(self):
        with self.assertRaises(ValueError):
            clipped_aabb([30, 10, 20, 40], 200, 300)
        with self.assertRaises(ValueError):
            clipped_aabb([float('nan'), 10, 20, 40], 200, 300)

    def test_teacher_correspondence_iou_is_exact_for_same_box(self):
        self.assertEqual(iou([1, 2, 11, 22], [1, 2, 11, 22]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_recognition_geometry_rejects_tiny_and_page_shapes_without_claiming_identity(self):
        self.assertTrue(recognition_geometry(400, 560, 3840, 2160)['recognition_geometry_eligible'])
        self.assertFalse(recognition_geometry(80, 120, 3840, 2160)['recognition_geometry_eligible'])
        self.assertFalse(recognition_geometry(3840, 2160, 3840, 2160)['recognition_geometry_eligible'])
        self.assertFalse(recognition_geometry(400, 560, 3840, 2160)['complete_physical_card_verified'])


if __name__ == '__main__':
    unittest.main()
