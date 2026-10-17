import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.benchmark_frozen_retrieval import deterministic_split_sample, rank_identities
from training.card_recognizer.foundation_retrieval import prepare_card_path, prepare_detected_crop_path


class FoundationRetrievalTests(unittest.TestCase):
    def test_canonical_card_uses_shared_fit_pad_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'card.png'
            Image.fromarray(np.full((312, 224, 3), 127, dtype=np.uint8), 'RGB').save(path)
            tensor = prepare_card_path(path)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())

    def test_prepare_rejects_unwarped_or_wrong_size_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.png'
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), 'RGB').save(path)
            with self.assertRaisesRegex(ValueError, '224x312'):
                prepare_card_path(path)

    def test_detected_crop_uses_shared_full_bounds_canonicalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detector-crop.png'
            Image.fromarray(np.full((676, 466, 3), 127, dtype=np.uint8), 'RGB').save(path)
            tensor = prepare_detected_crop_path(path)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertTrue(np.isfinite(tensor).all())

    def test_stratified_sampler_is_deterministic_and_even_by_split(self):
        queries = [{'split': split, 'index': i} for split in ('train', 'val') for i in range(10)]
        first = deterministic_split_sample(queries, 4)
        second = deterministic_split_sample(queries, 4)
        self.assertEqual(first, second)
        self.assertEqual([sum(x['split'] == split for x in first) for split in ('train', 'val')], [4, 4])

    def test_retrieval_ranks_distinct_identities_and_reports_recall(self):
        gallery = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
        queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
        recall, records = rank_identities(queries, gallery,
                                          ['card-a', 'card-a', 'card-b'],
                                          ['card-a', 'card-b'])
        self.assertEqual(recall, {'1': 1.0, '5': 1.0, '20': 1.0})
        self.assertEqual([record['rank'] for record in records], [1, 1])
        self.assertEqual(records[0]['top_identities'][0]['identity'], 'card-a')


if __name__ == '__main__':
    unittest.main()
