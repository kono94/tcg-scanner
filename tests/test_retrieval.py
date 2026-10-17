import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.card_recognizer.retrieval import CosineGallery, normalized, file_sha256


class RetrievalTests(unittest.TestCase):
    def test_templates_do_not_occupy_distinct_identity_shortlist(self):
        gallery = CosineGallery(np.array([[1, 0], [.99, .01], [0, 1]], np.float32),
                                [{'identity': 'onepiece:en:A'}, {'identity': 'onepiece:en:A'}, {'identity': 'pokemon:en:B'}], 'same')
        result = gallery.search(np.array([[1, 0]], np.float32), 'same', 2)[0]
        self.assertEqual([r['identity'] for r in result], ['onepiece:en:A', 'pokemon:en:B'])
        self.assertAlmostEqual(result[0]['similarity'], 1)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            gallery.search(np.array([[1, 0]], np.float32), 'different')

    def test_invalid_descriptors_are_rejected(self):
        for values in [[[0, 0]], [[float('nan'), 1]], [1, 2]]:
            with self.assertRaises(ValueError):
                normalized(np.asarray(values))

    def test_loader_checks_contract_and_descriptor_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / 'vectors.npy', np.array([[1, 0]], np.float32))
            contract = {'dimension': 2}
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            manifest = {'encoder_contract': contract, 'encoder_contract_sha256': digest,
                        'vectors_sha256': file_sha256(root / 'vectors.npy'), 'references': [{'identity': 'onepiece:en:A'}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(CosineGallery.load(root).vectors.shape, (1, 2))
            manifest['encoder_contract']['dimension'] = 3
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'contract checksum'):
                CosineGallery.load(root)
            manifest['encoder_contract_sha256'] = hashlib.sha256(json.dumps(manifest['encoder_contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'dimension'):
                CosineGallery.load(root)


if __name__ == '__main__':
    unittest.main()
