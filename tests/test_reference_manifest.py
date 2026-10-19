import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from scripts.curation.build_reference_manifest import collect


class ReferenceManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        legacy = self.root / 'datasets/sources/catalog/onepiece-en-bandai-2024/OP01'
        legacy.mkdir(parents=True)
        image = legacy / 'OP01-001_p1.jpg'
        Image.new('RGB', (20, 28), 'red').save(image)
        image.with_suffix('.json').write_text(json.dumps({'id': image.stem, 'name': 'Luffy'}))
        poke = self.root / 'datasets/sources/catalog/pokemon-en-tcgdex'
        poke.mkdir(parents=True)
        image = poke / 'card.webp'
        Image.new('RGB', (20, 28), 'blue').save(image)
        self.record = dict(domain='pokemon_physical_tcg', language='en', catalog_id='sv03.5-200',
                           name='Blastoise ex', image_path='card.webp',
                           image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        self.catalog = poke / 'catalog.json'
        self.catalog.write_text(json.dumps([self.record]))
        op = self.root / 'datasets/sources/catalog/onepiece-en-bandai'
        op.mkdir(parents=True)
        (op / 'manifest.json').write_text(json.dumps({'cards': []}))

    def tearDown(self):
        self.temp.cleanup()

    def test_physical_namespace_variant_and_provider_period_survive(self):
        digital = dict(self.record, domain='pokemon_pocket_digital', catalog_id='A1-001')
        unavailable = dict(self.record, catalog_id='missing', image_path=None)
        self.catalog.write_text(json.dumps([self.record, digital, unavailable]))
        result = collect(self.root)
        self.assertEqual(result['counts_by_game'], {'onepiece': 1, 'pokemon': 1})
        self.assertEqual([r['card_id'] for r in result['references']], ['OP01-001_p1', 'sv03.5-200'])
        self.assertEqual(result['unavailable_physical_references'][0]['card_id'], 'missing')
        self.assertFalse(result['gallery_complete'])

    def test_acquired_image_bytes_must_match_provenance(self):
        self.catalog.write_text(json.dumps([dict(self.record, image_sha256='0'*64)]))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            collect(self.root)

    def test_same_identity_and_bytes_do_not_duplicate_gallery_templates(self):
        self.catalog.write_text(json.dumps([self.record, self.record]))
        result = collect(self.root)
        self.assertEqual(result['counts_by_game']['pokemon'], 1)
        self.assertEqual(result['references'][-1]['aliases'], ['datasets/sources/catalog/pokemon-en-tcgdex/card.webp'])


if __name__ == '__main__':
    unittest.main()
