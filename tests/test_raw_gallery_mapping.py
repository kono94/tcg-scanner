import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.build_raw_dino_gallery import catalog_fronts
from training.card_recognizer.retrieval import file_sha256


class RawGalleryMappingTests(unittest.TestCase):
    def test_variant_language_and_source_checksum_survive_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cards'
            root.mkdir()
            path = root / 'OP01-001_p1.jpg'
            Image.new('RGB', (20, 28)).save(path)
            metadata = {'id': 'OP01-001_p1', 'image_sha256': file_sha256(path)}
            (root / 'OP01-001_p1.json').write_text(json.dumps(metadata))
            refs = catalog_fronts([f'onepiece:en={root}'], Path(directory) / 'output')
            self.assertEqual(refs[0]['identity'], 'onepiece:en:OP01-001_p1')
            self.assertEqual(refs[0]['source_sha256'], metadata['image_sha256'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                catalog_fronts([f'onepiece:en={root}'], Path(directory) / 'output')

    def test_digital_and_namespace_mismatch_cannot_be_silently_enrolled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cards'
            root.mkdir()
            path = root / 'test.json'
            path.write_text(json.dumps({'id': 'A1-1', 'domain': 'pokemon_pocket_digital'}))
            with self.assertRaisesRegex(ValueError, 'No downloaded'):
                catalog_fronts([f'pokemon:en={root}'], Path(directory) / 'output')
            path.write_text(json.dumps({'id': 'test', 'language': 'ja'}))
            with self.assertRaisesRegex(ValueError, 'namespace mismatch'):
                catalog_fronts([f'pokemon:en={root}'], Path(directory) / 'output')

    def test_output_cannot_contain_source_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'separately'):
                catalog_fronts([f'onepiece:en={root}'], root / 'output')

    def test_exact_filename_aliases_do_not_duplicate_identity_but_conflicts_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cards'
            root.mkdir()
            for name in ('set.5-001', 'set%2E5-001'):
                Image.new('RGB', (20, 28)).save(root / f'{name}.png')
                (root / f'{name}.json').write_text(json.dumps({'id': 'set.5-001'}))
            refs = catalog_fronts([f'pokemon:en={root}'], Path(directory) / 'output')
            self.assertEqual(len(refs), 1)
            self.assertEqual(len(refs[0]['exact_source_aliases']), 1)
            Image.new('RGB', (20, 28), 'red').save(root / 'set.5-001.png')
            with self.assertRaisesRegex(ValueError, 'conflicting image bytes'):
                catalog_fronts([f'pokemon:en={root}'], Path(directory) / 'output')
