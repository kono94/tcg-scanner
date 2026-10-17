"""Whole-card text regions and direct catalog lookup, before optional visual retrieval."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

from training.card_recognizer.identity_evidence import extract_printed_codes


def _parse_tsv(result: subprocess.CompletedProcess, language: str, backend: str,
               source_origin: tuple[int, int] = (0, 0), source_scale: float = 1.0) -> list[dict]:
    """Parse Tesseract TSV and map boxes back into the original card crop."""
    if result.returncode:
        raise RuntimeError(f'Tesseract failed: {result.stderr.decode(errors="replace")[:400]}')
    lines = {}
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8', errors='replace')), delimiter='\t', quoting=csv.QUOTE_NONE):
        text = (row.get('text') or '').strip()
        if row.get('level') != '5' or not text:
            continue
        confidence = float(row['conf'])
        if confidence < 0:
            continue
        x0, y0 = int(row['left']), int(row['top'])
        x1, y1 = x0 + int(row['width']), y0 + int(row['height'])
        ox, oy = source_origin
        box = [round(ox + x0 / source_scale), round(oy + y0 / source_scale),
               round(ox + x1 / source_scale), round(oy + y1 / source_scale)]
        key = tuple(row[x] for x in ['page_num', 'block_num', 'par_num', 'line_num'])
        word = {'text': text, 'confidence': confidence, 'bbox_xyxy': box}
        lines.setdefault(key, []).append(word)
    regions = []
    for words in lines.values():
        words.sort(key=lambda w: w['bbox_xyxy'][0])
        regions.append({'text': ' '.join(w['text'] for w in words),
                        'confidence': min(w['confidence'] for w in words),
                        'bbox_xyxy': [min(w['bbox_xyxy'][0] for w in words), min(w['bbox_xyxy'][1] for w in words),
                                      max(w['bbox_xyxy'][2] for w in words), max(w['bbox_xyxy'][3] for w in words)],
                        'coordinate_space': 'orientation_candidate_crop_pixels_top_left',
                        'backend': backend, 'language_model': language, 'words': words})
    return regions


_ID_LIKE = re.compile(r'(?i)(?=[A-Z0-9/-]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+)+|(?<!\d)\d{8}(?!\d)')


def _candidate_regions(regions: list[dict], max_regions: int) -> list[dict]:
    """Pick distinct OCR boxes containing ID-shaped observed text, never guessed codes."""
    if max_regions == 0:
        return []
    candidates, seen = [], set()
    for region in regions:
        for item in [region, *region.get('words', [])]:
            text, box = str(item.get('text', '')).strip(), item.get('bbox_xyxy')
            if not _ID_LIKE.search(text) or not isinstance(box, list) or len(box) != 4:
                continue
            key = tuple(int(v) for v in box)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({'text': text, 'bbox_xyxy': list(key), 'confidence': item.get('confidence')})
            if len(candidates) >= max_regions:
                return candidates
    return candidates


def refine_id_regions(image: np.ndarray, regions: list[dict], language: str = 'eng',
                      max_regions: int = 12, scales: tuple[int, ...] = (2, 4),
                      padding_fraction: float = 0.30) -> list[dict]:
    """Retry observed ID-like OCR boxes at larger scales; source boxes stay in original pixels.

    No text correction is applied. Each retry retains the exact source crop box and raw OCR.
    """
    if not isinstance(max_regions, int) or not 0 <= max_regions <= 12:
        raise ValueError('max_regions must be an integer in [0,12]')
    if not scales or any(not isinstance(s, int) or s < 2 or s > 4 for s in scales):
        raise ValueError('scales must contain integers from 2 through 4')
    if not np.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1:
        raise ValueError('padding_fraction must be finite and in [0,1]')
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError('OCR expects RGB uint8 pixels')
    executable = shutil.which('tesseract')
    if executable is None:
        raise RuntimeError('Tesseract is unavailable; no OCR evidence can be asserted')
    height, width = pixels.shape[:2]
    refinements = []
    for candidate in _candidate_regions(regions, max_regions):
        x0, y0, x1, y1 = candidate['bbox_xyxy']
        pad_x = max(2, round((x1 - x0) * padding_fraction))
        pad_y = max(2, round((y1 - y0) * padding_fraction))
        crop_box = [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]
        cx0, cy0, cx1, cy1 = crop_box
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        crop = pixels[cy0:cy1, cx0:cx1]
        for scale in scales:
            enlarged = np.asarray(Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.Resampling.LANCZOS))
            buffer = io.BytesIO()
            Image.fromarray(enlarged).save(buffer, format='PNG')
            result = subprocess.run([executable, 'stdin', 'stdout', '-l', language, '--psm', '7', 'tsv'],
                                    input=buffer.getvalue(), capture_output=True, timeout=30, check=False)
            parsed = _parse_tsv(result, language, 'tesseract_psm7_refinement', (cx0, cy0), float(scale))
            refinements.append({'trigger_text': candidate['text'], 'trigger_bbox_xyxy': candidate['bbox_xyxy'],
                                'source_crop_bbox_xyxy': crop_box, 'scale': scale,
                                'backend': 'tesseract_psm7_refinement', 'raw_regions': parsed,
                                'raw_text': ' | '.join(r['text'] for r in parsed),
                                'confidence': min((r['confidence'] for r in parsed), default=0.0),
                                'coordinate_space': 'orientation_candidate_crop_pixels_top_left'})
    return refinements


def tesseract_regions(image: np.ndarray, language: str = 'eng', refine: bool = True,
                      max_refinement_regions: int = 12) -> list[dict]:
    """OCR the whole RGB card and optionally retry observed ID-like boxes (not fixed ID zones)."""
    executable = shutil.which('tesseract')
    if executable is None:
        raise RuntimeError('Tesseract is unavailable; no OCR evidence can be asserted')
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError('OCR expects RGB uint8 pixels')
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    result = subprocess.run([executable, 'stdin', 'stdout', '-l', language, '--psm', '11', 'tsv'],
                            input=buffer.getvalue(), capture_output=True, timeout=30, check=False)
    regions = _parse_tsv(result, language, 'tesseract_psm11')
    if refine:
        refinements = refine_id_regions(pixels, regions, language, max_refinement_regions)
        # Preserve every original observation; retries are additive evidence only.
        for region in regions:
            region['refinements'] = [r for r in refinements if r['trigger_bbox_xyxy'] == region['bbox_xyxy'] or
                                     any(w['bbox_xyxy'] == r['trigger_bbox_xyxy'] for w in region.get('words', []))]
    return regions


def load_catalog(specifications: list[str]) -> dict[str, dict]:
    """Load known One Piece/TCGdex sidecars; explicit game:language=/source directory."""
    catalog = {}
    for specification in specifications:
        namespace, separator, location = specification.partition('=')
        if not separator or namespace.count(':') != 1:
            raise ValueError('Catalog input must be game:language=/path/to/catalog')
        game, language = namespace.split(':')
        for path in sorted(Path(location).rglob('*.json')):
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                continue
            card_id = data.get('catalog_id') or data.get('provider_id') or data.get('id')
            if not isinstance(card_id, str):
                continue
            if data.get('game', game) != game or data.get('language', language) != language:
                continue
            identity = f'{game}:{language}:{card_id}'
            tokens = data.get('expected_printed_tokens', [])
            if game == 'onepiece':
                base = card_id.split('_p', 1)[0]
                tokens = [t for t in extract_printed_codes(base) if t['game'] == game]
            elif game == 'pokemon' and data.get('official_total'):
                local = str(data.get('local_number') or data.get('local_id') or data.get('printed_number') or '')
                total = str(data['official_total'])
                tokens = [{'game': game, 'kind': 'collector_number', 'token': f'{local}/{total}'.upper()}]
                if local.isdigit():
                    # Numeric padding aliases represent the same observed numeric value, never another catalog ID.
                    for numerator in [str(int(local)), str(int(local)).zfill(3)]:
                        token = {'game': game, 'kind': 'collector_number', 'token': f'{numerator}/{total}'}
                        if token not in tokens:
                            tokens.append(token)
            catalog[identity] = {'identity': identity, 'game': game, 'language': language, 'card_id': card_id,
                                 'name': data.get('name'), 'expected_printed_tokens': tokens,
                                 'printing_variant': card_id.split('_p', 1)[1] if '_p' in card_id else '',
                                 'identity_scope': 'catalog_reference', 'physical_finish': 'unresolved',
                                 'source_sidecar': str(path)}
    return catalog


def resolve_ocr_catalog(regions: list[dict], catalog: dict[str, dict], min_confidence: float = 80) -> dict:
    """Use exact observed tokens. A direct candidate is not a verified physical-print label."""
    if not 0 <= min_confidence <= 100:
        raise ValueError('OCR confidence gate must be in [0,100]')
    lookup = {}
    for identity, record in catalog.items():
        for token in record.get('expected_printed_tokens', []):
            key = (token['game'], token['kind'], token['token'].upper())
            lookup.setdefault(key, set()).add(identity)
    observations, matched_sets = [], []
    expanded_regions = list(regions)
    for region in regions:
        expanded_regions.extend(region.get('words', []))
        for retry in region.get('refinements', []):
            expanded_regions.extend(retry.get('raw_regions', []))
            for raw_region in retry.get('raw_regions', []):
                expanded_regions.extend(raw_region.get('words', []))
    for region in expanded_regions:
        if not np.isfinite(region['confidence']) or region['confidence'] < min_confidence:
            continue
        for token in extract_printed_codes(region['text']):
            key = (token['game'], token['kind'], token['token'])
            identities = lookup.get(key, set())
            observations.append({**token, 'confidence': region['confidence'], 'bbox_xyxy': region.get('bbox_xyxy'),
                                 'catalog_matches': sorted(identities)})
            if identities:
                matched_sets.append(identities)
    unmatched = [o for o in observations if not o['catalog_matches']]
    if not observations:
        status, candidates = 'unreadable_or_no_id', set()
    elif not matched_sets:
        status, candidates = 'no_catalog_match', set()
    else:
        candidates = set.intersection(*matched_sets)
        if not candidates or unmatched:
            status = 'conflicting_code_evidence'
        elif len(candidates) == 1:
            status = 'ocr_catalog_candidate'
        else:
            status = 'ambiguous_catalog_code'
    return {'status': status, 'candidate_identities': sorted(candidates), 'observed_codes': observations,
            'ground_truth_verified': False, 'confidence_gate_calibrated': False,
            'language_and_physical_printing_verified': False, 'min_ocr_confidence': min_confidence}
