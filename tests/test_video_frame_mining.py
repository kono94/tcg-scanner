import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


spec = importlib.util.spec_from_file_location('mine_video_frames', Path(__file__).resolve().parents[1] / 'scripts/video/mine_video_frames.py')
miner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(miner)


class VideoMiningTests(unittest.TestCase):
    def fixture(self, root):
        video = root / 'source.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 8, (160, 120))
        self.assertTrue(writer.isOpened())
        rng = np.random.default_rng(42)
        scenes = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(2)]
        for frame in scenes:
            for _ in range(16):
                writer.write(frame)
        writer.release()
        return argparse.Namespace(video=video, output=root / 'output', source_url='https://example.test/video',
                                  source_group='same-match', reuse_basis='test-fixture', contains_overlays='no',
                                  sample_seconds=.5, window_seconds=1, start=0, end=None, max_samples=100,
                                  max_frames=20, min_sharpness=10, dedup_difference=.005, min_source_height=0)

    def test_selects_diverse_frames_and_retains_source_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            original_hash = miner.sha256_file(args.video)
            report = miner.mine(args)
            self.assertEqual(report['sampled_frames'], 8)
            self.assertEqual(report['saved_frames'], 2)
            self.assertEqual(report['rejected_duplicate_candidates'], 2)
            self.assertEqual(report['source']['sha256'], original_hash)
            self.assertFalse(report['real_camera_validation'])
            for record in report['records']:
                self.assertEqual(record['source_group'], 'same-match')
                self.assertEqual(record['time_seconds'], record['frame_index'] / 8)
                self.assertEqual(record['identity_label_status'], 'unlabeled')
                self.assertEqual(record['geometry_label_status'], 'unlabeled')
                self.assertTrue((args.output / record['image']).is_file())
            self.assertEqual(miner.sha256_file(args.video), original_hash)
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                miner.mine(args)

    def test_invalid_time_range_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            args.start, args.end = 2, 1
            with self.assertRaisesRegex(ValueError, 'end'):
                miner.mine(args)
            self.assertFalse(args.output.exists())

    def test_low_resolution_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            args.min_source_height = 1440
            with self.assertRaisesRegex(ValueError, 'below required 1440'):
                miner.mine(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
