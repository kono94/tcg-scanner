import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_synthetic_cards', ROOT / 'scripts/generate_synthetic_cards.py')
synthetic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = synthetic
spec.loader.exec_module(synthetic)


class SyntheticCardsTests(unittest.TestCase):
    def make_references(self, root):
        folder = root / 'cards' / 'SET'
        folder.mkdir(parents=True)
        for i in range(6):
            image = np.zeros((88, 63, 3), dtype=np.uint8)
            image[:] = (i * 30, 80 + i * 10, 200 - i * 20)
            image[10:30, 12:25] = (255, 230, 20)
            Image.fromarray(image).save(folder / f'SET-{i:03d}.jpg')
        return synthetic.discover_references(root / 'cards')

    def test_identity_and_duplicate_hash_groups_do_not_cross_splits(self):
        refs = [synthetic.Reference(Path('a'), 'a', 'shared'),
                synthetic.Reference(Path('b'), 'b', 'shared'),
                synthetic.Reference(Path('b2'), 'b', 'second'),
                synthetic.Reference(Path('c'), 'c', 'third')]
        pools = synthetic.partition_references(refs, 42, .2)
        self.assertTrue(pools['train'] and pools['val'])
        self.assertFalse({r.card_id for r in pools['train']} & {r.card_id for r in pools['val']})
        self.assertFalse({r.sha256 for r in pools['train']} & {r.sha256 for r in pools['val']})
        self.assertEqual(pools, synthetic.partition_references(refs, 42, .2))

    def test_variable_counts_and_exact_quads_with_enclosing_rectangles(self):
        with tempfile.TemporaryDirectory() as directory:
            refs = self.make_references(Path(directory))
            for count in [0, 1, 7, 12]:
                image, meta = synthetic.render_scene(refs, 53 + count, 480, 360, count)
                self.assertEqual(image.shape, (360, 480, 3))
                self.assertEqual(len(meta['instances']), count)
                for item in meta['instances']:
                    quad = np.asarray(item['true_corners'], np.float32)
                    obb = np.asarray(item['obb_corners'], np.float32)
                    self.assertTrue(cv2.isContourConvex(quad))
                    self.assertGreaterEqual(abs(cv2.contourArea(obb)) + .01, abs(cv2.contourArea(quad)))
                    edges = np.roll(obb, -1, axis=0) - obb
                    self.assertAlmostEqual(float(np.dot(edges[0], edges[1])), 0, delta=.02)
                    for point in quad:
                        self.assertGreaterEqual(cv2.pointPolygonTest(obb, tuple(float(x) for x in point), True), -.001)
                    self.assertTrue(np.all(obb >= 0))
                    self.assertTrue(np.all(obb < [480, 360]))

    def test_scene_pixels_and_metadata_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            refs = self.make_references(Path(directory))
            first, first_meta = synthetic.render_scene(refs, 7, 480, 360, 9, 'binder')
            again, again_meta = synthetic.render_scene(refs, 7, 480, 360, 9, 'binder')
            np.testing.assert_array_equal(first, again)
            self.assertEqual(first_meta, again_meta)

    def test_dataset_roundtrip_and_safe_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_references(root)
            args = argparse.Namespace(cards_dir=root / 'cards', output=root / 'generated', scenes=8,
                                      width=320, height=256, max_cards=12, val_fraction=.25, seed=91,
                                      video_seconds=0, fps=10)
            summary = synthetic.generate(args)
            self.assertEqual(summary['scenes'], 8)
            self.assertIn('0', summary['counts'])
            total = 0
            for annotation in args.output.glob('annotations/*/*.json'):
                data = json.loads(annotation.read_text())
                label = args.output / 'labels' / data['split'] / (annotation.stem + '.txt')
                lines = label.read_text().splitlines()
                self.assertEqual(len(lines), data['count'])
                for line in lines:
                    values = line.split()
                    self.assertEqual(len(values), 9)
                    self.assertEqual(values[0], '0')
                    self.assertTrue(all(0 <= float(x) <= 1 for x in values[1:]))
                total += data['count']
            self.assertEqual(total, summary['instances'])
            crops = json.loads((args.output / 'recognizer_manifest.json').read_text())
            self.assertEqual(len(crops['queries']), total)
            self.assertTrue(crops['preprocessing_contract_sha256'])
            before = hashlib.sha256((args.output / 'summary.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                synthetic.generate(args)
            self.assertEqual(before, hashlib.sha256((args.output / 'summary.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
