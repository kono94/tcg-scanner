import unittest
from unittest.mock import patch

import numpy as np
import subprocess

from training.recognition.ocr_identity import _parse_tsv, refine_id_regions, resolve_ocr_catalog


def record(token):
    return {'expected_printed_tokens': [{'game': 'onepiece', 'kind': 'printed_number', 'token': token}]}


class OCRCatalogTests(unittest.TestCase):
    def test_unique_code_can_resolve_without_any_encoder(self):
        result = resolve_ocr_catalog([{'text': 'OP03-094', 'confidence': 95}], {'onepiece:en:OP03-094': record('OP03-094')})
        self.assertEqual(result['status'], 'ocr_catalog_candidate')
        self.assertFalse(result['ground_truth_verified'])

    def test_shared_number_does_not_prove_variant(self):
        catalog = {'onepiece:en:OP01-001': record('OP01-001'), 'onepiece:en:OP01-001_p1': record('OP01-001')}
        result = resolve_ocr_catalog([{'text': 'OP01-001', 'confidence': 95}], catalog)
        self.assertEqual(result['status'], 'ambiguous_catalog_code')
        self.assertEqual(len(result['candidate_identities']), 2)

    def test_conflicting_or_unknown_codes_are_not_fuzzy_matched(self):
        catalog = {'onepiece:en:OP03-094': record('OP03-094')}
        unknown = resolve_ocr_catalog([{'text': 'OP03-095', 'confidence': 95}], catalog)
        self.assertEqual(unknown['status'], 'no_catalog_match')
        conflict = resolve_ocr_catalog([{'text': 'OP03-094 OP03-095', 'confidence': 95}], catalog)
        self.assertEqual(conflict['status'], 'conflicting_code_evidence')

    def test_low_confidence_text_is_not_promoted(self):
        result = resolve_ocr_catalog([{'text': 'OP03-094', 'confidence': 50}], {'onepiece:en:OP03-094': record('OP03-094')})
        self.assertEqual(result['status'], 'unreadable_or_no_id')

    def test_tsv_boxes_map_upscaled_crop_back_to_original_pixels(self):
        tsv = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
               '5\t1\t1\t1\t1\t1\t4\t6\t20\t10\t87.5\tOP03-094\n')
        result = subprocess.CompletedProcess(['tesseract'], 0, tsv.encode(), b'')
        parsed = _parse_tsv(result, 'eng', 'test', source_origin=(100, 200), source_scale=2)
        self.assertEqual(parsed[0]['bbox_xyxy'], [102, 203, 112, 208])
        self.assertEqual(parsed[0]['words'][0]['confidence'], 87.5)

    def test_refinement_preserves_raw_text_and_records_source_crop_and_scale(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = [{'text': 'opos-oo4mm @', 'confidence': 30,
                    'bbox_xyxy': [20, 20, 40, 30], 'words': []}]
        tsv = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
               '5\t1\t1\t1\t1\t1\t20\t10\t80\t20\t92.0\tOP03-094\n')
        result = subprocess.CompletedProcess(['tesseract'], 0, tsv.encode(), b'')
        with patch('training.recognition.ocr_identity.shutil.which', return_value='/usr/bin/tesseract'), \
             patch('training.recognition.ocr_identity.subprocess.run', return_value=result) as run:
            retries = refine_id_regions(image, regions, scales=(2,), padding_fraction=0.25)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(retries[0]['trigger_text'], 'opos-oo4mm @')
        self.assertEqual(retries[0]['source_crop_bbox_xyxy'], [15, 18, 45, 32])
        self.assertEqual(retries[0]['scale'], 2)
        self.assertEqual(retries[0]['raw_text'], 'OP03-094')
        # Enlarged OCR's [20,10,100,30] maps through crop origin and 2x scale.
        self.assertEqual(retries[0]['raw_regions'][0]['bbox_xyxy'], [25, 23, 65, 33])
        self.assertEqual(retries[0]['confidence'], 92.0)

    def test_refinement_budget_caps_distinct_id_like_regions_at_twelve(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        regions = [{'text': f'OP03-{i:03d}', 'confidence': 50,
                    'bbox_xyxy': [i * 10, 20, i * 10 + 8, 30], 'words': []} for i in range(1, 20)]
        empty_tsv = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n')
        result = subprocess.CompletedProcess(['tesseract'], 0, empty_tsv.encode(), b'')
        with patch('training.recognition.ocr_identity.shutil.which', return_value='/usr/bin/tesseract'), \
             patch('training.recognition.ocr_identity.subprocess.run', return_value=result) as run:
            retries = refine_id_regions(image, regions)
        self.assertEqual(len(retries), 12 * 2)
        self.assertEqual(run.call_count, 12 * 2)

    def test_exact_refined_code_can_match_without_changing_original_observation(self):
        region = {'text': 'opos-oo4mm @', 'confidence': 31,
                  'bbox_xyxy': [1, 2, 3, 4], 'refinements': [
                      {'raw_regions': [{'text': 'OP03-094', 'confidence': 92,
                                        'bbox_xyxy': [1, 2, 3, 4]}]}]}
        result = resolve_ocr_catalog([region], {'onepiece:en:OP03-094': record('OP03-094')}, min_confidence=80)
        self.assertEqual(result['status'], 'ocr_catalog_candidate')
        self.assertEqual(region['text'], 'opos-oo4mm @')
        self.assertFalse(result['ground_truth_verified'])


if __name__ == '__main__':
    unittest.main()
