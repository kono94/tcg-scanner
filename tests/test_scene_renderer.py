import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.card_detector.scene_renderer import render


class SceneRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        cls.textures = []
        for i in range(4):
            path = Path(cls.tmp.name) / f'card{i}.png'
            Image.fromarray(rng.integers(0, 255, (88, 63, 3), dtype=np.uint8)).resize((126, 176)).save(path)
            cls.textures.append(str(path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_are_complete_cards_inside_the_frame(self):
        kinds = set()
        for seed in range(24):
            pixels, meta = render(seed, self.textures, 512, 384)
            kinds.add(meta['kind'])
            self.assertEqual(pixels.shape, (384, 512, 3))
            for inst in meta['instances']:
                quad = np.asarray(inst['true_corners'])
                self.assertTrue(np.all(quad >= 0) and np.all(quad[:, 0] <= 511) and np.all(quad[:, 1] <= 383))
                self.assertGreaterEqual(inst['visible_fraction'], .6)
            for card in meta['unlabeled_cards']:
                self.assertIn(card['reason'], {'outside_frame', 'occluded', 'too_small'})
            if meta['kind'] == 'negative':
                self.assertEqual(meta['instances'], [])
        self.assertGreaterEqual(len(kinds), 3)

    def test_deterministic_per_seed(self):
        a, ma = render(5, self.textures, 320, 240)
        b, mb = render(5, self.textures, 320, 240)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ma['instances'], mb['instances'])


if __name__ == '__main__':
    unittest.main()
