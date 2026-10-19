from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'curation'))
import curate_datasets as curation  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DatasetCurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        self.input_root = self.repo / 'inputs'
        self.input_root.mkdir()
        (self.input_root / 'gallery').mkdir()
        (self.input_root / 'scenes').mkdir()
        (self.input_root / 'crops').mkdir()
        self.reference = self.input_root / 'gallery' / 'ref.jpg'
        self.source = self.input_root / 'scenes' / 'scene.jpg'
        self.crop = self.input_root / 'crops' / 'crop.jpg'
        Image.new('RGB', (32, 48), (200, 20, 30)).save(self.reference)
        Image.new('RGB', (96, 72), (40, 50, 60)).save(self.source)
        Image.new('RGB', (48, 64), (80, 90, 100)).save(self.crop)
        self.references_manifest = self.input_root / 'references.json'
        self.review_manifest = self.input_root / 'review.json'
        self.output = self.repo / 'datasets' / 'curated'
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_inputs(self, *, split='train', source_group='video-a', crop_path='crops/crop.jpg',
                      crop_sha=None, reference_sha=None, card_id='OP01-001', status='verified_reference_match',
                      single=True, rejected=None):
        self.references_manifest.write_text(json.dumps({
            'schema_version': 1,
            'references': [{
                'path': 'gallery/ref.jpg', 'sha256': digest(self.reference), 'game': 'one_piece',
                'language': 'en', 'card_id': 'OP01-001', 'printing': 'unknown', 'finish': 'unknown',
            }],
        }))
        row = {
            'crop_path': crop_path, 'crop_sha256': crop_sha or digest(self.crop),
            'source_path': 'scenes/scene.jpg', 'source_sha256': digest(self.source),
            'source_group': source_group, 'split': split, 'status': status,
            'single_complete_card': single,
            'identity': {'game': 'one_piece', 'language': 'en', 'card_id': card_id},
            'reference_path': 'gallery/ref.jpg', 'reference_sha256': reference_sha or digest(self.reference),
            'annotation_provenance': 'root-sol catalog and crop review',
            'annotation_notes': 'single complete face; exact printing unresolved',
            'printing': 'unknown', 'finish': 'unknown',
        }
        self.review_manifest.write_text(json.dumps({
            'schema_version': 1, 'reviewer': 'root-sol', 'review_provenance': 'cross-checked against reference manifest',
            'queries': [row], 'rejected': rejected or [],
        }))

    def _curate(self, **kwargs):
        return curation.curate_dataset(
            repo_root=self.repo, input_root=self.input_root,
            references_manifest=self.references_manifest, review_manifest=self.review_manifest,
            output=self.output, **kwargs,
        )

    def test_copies_namespaced_identity_and_preserves_unknown_fields(self):
        result = self._curate()
        gallery = self.output / 'recognition/gallery/one_piece/en/OP01-001' / f'reference-{digest(self.reference)[:12]}.jpg'
        query = self.output / 'recognition/train/one_piece/en/OP01-001' / f'query-{digest(self.crop)[:16]}.jpg'
        self.assertTrue(gallery.is_file())
        self.assertTrue(query.is_file())
        self.assertTrue(self.reference.is_file())  # curation copies, never moves
        sidecar = json.loads(query.with_suffix('.jpg.json').read_text())
        self.assertEqual(sidecar['source_record']['printing'], 'unknown')
        self.assertEqual(sidecar['source_record']['finish'], 'unknown')
        self.assertEqual(result['recognition_split_counts']['train'], 1)

    def test_rejects_source_group_crossing_splits(self):
        self._write_inputs(split='train')
        payload = json.loads(self.review_manifest.read_text())
        second = dict(payload['queries'][0], crop_path='crops/crop.jpg', split='test')
        payload['queries'].append(second)
        self.review_manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(curation.CurationError, 'crosses'):
            self._curate()
        self.assertFalse(self.output.exists())

    def test_rejects_gallery_bytes_in_evaluation(self):
        Image.new('RGB', (32, 48), (200, 20, 30)).save(self.crop)
        self._write_inputs(split='test', crop_sha=digest(self.crop))
        with self.assertRaisesRegex(curation.CurationError, 'duplicates gallery'):
            self._curate()

    def test_rejects_reference_hash_mismatch_and_bad_crop_hash(self):
        self._write_inputs(reference_sha='0' * 64)
        with self.assertRaisesRegex(curation.CurationError, 'reference_sha256 mismatch'):
            self._curate()
        self._write_inputs(crop_sha='0' * 64)
        with self.assertRaisesRegex(curation.CurationError, 'crop_sha256 mismatch'):
            self._curate()

    def test_rejects_unsafe_identity_and_non_single_card(self):
        self._write_inputs(card_id='../OP01-001')
        with self.assertRaisesRegex(curation.CurationError, 'safe, nonempty path component'):
            self._curate()
        self._write_inputs(single=False)
        with self.assertRaisesRegex(curation.CurationError, 'single_complete_card=true'):
            self._curate()

    def test_period_in_provider_set_id_is_preserved(self):
        self._write_inputs(card_id='sv03.5-200')
        refs = json.loads(self.references_manifest.read_text())
        refs['references'][0]['card_id'] = 'sv03.5-200'
        self.references_manifest.write_text(json.dumps(refs))
        self._curate()
        self.assertTrue((self.output / 'recognition/train/one_piece/en/sv03.5-200').is_dir())

    def test_rejected_deletion_is_limited_to_allowlisted_ignored_crop_root(self):
        allowed = self.repo / 'datasets/sources/videos/run-a/crops'
        allowed.mkdir(parents=True)
        crop = allowed / 'rejected.jpg'
        Image.new('RGB', (20, 30), (1, 2, 3)).save(crop)
        rejected = [{'status': 'rejected', 'crop_path': 'rejected.jpg', 'crop_sha256': digest(crop)}]
        with self.assertRaisesRegex(curation.CurationError, 'allowlisted'):
            curation._validate_rejected_crop_root(self.repo, self.input_root / 'gallery')
        self._curate()
        deleted = curation._remove_rejected(rejected, self.repo, allowed, self.output)
        self.assertEqual(deleted, 1)
        self.assertFalse(crop.exists())


if __name__ == '__main__':
    unittest.main()
