import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import scripts.collect_video_frames as collector
from training.card_recognizer.foundation_retrieval import sha256_file as file_sha256


class VideoFrameCollectionTests(unittest.TestCase):
    def test_hashes_dimensions_and_exact_duplicates_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            Image.new('RGB', (3840, 2160)).save(root / 'frame.jpg')
            record = {'image': 'frame.jpg', 'width': 3840, 'height': 2160,
                      'sha256': file_sha256(root / 'frame.jpg'), 'frame_index': 60, 'time_seconds': 1.0}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'source': {'source_group': 'video'}, 'records': [record, record]}))
            with patch.object(collector, 'ROOT', root):
                result = collector.collect([manifest], root / 'selected.json')
                self.assertEqual(result['selected_frames'], 1)
                self.assertEqual(len(result['exact_duplicate_frames_removed']), 1)
                self.assertFalse(result['ground_truth_verified'])
                record['sha256'] = 'wrong'
                manifest.write_text(json.dumps({'source': {'source_group': 'video'}, 'records': [record]}))
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    collector.collect([manifest], root / 'rejected.json')

    def test_unknown_source_group_cannot_enter_capture_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            Image.new('RGB', (3840, 2160)).save(root / 'frame.jpg')
            record = {'image': 'frame.jpg', 'width': 3840, 'height': 2160,
                      'sha256': file_sha256(root / 'frame.jpg')}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'records': [record]}))
            with patch.object(collector, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Source group missing'):
                    collector.collect([manifest], root / 'selected.json')
