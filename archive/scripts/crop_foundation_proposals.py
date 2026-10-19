"""Crop Grounding DINO AABB proposals at source resolution; no corner/identity claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clipped_aabb(box, width: int, height: int) -> tuple[list[int], list[int], bool]:
    if len(box) != 4 or not np.isfinite(np.asarray(box, dtype=np.float64)).all():
        raise ValueError('AABB must contain four finite coordinates')
    x1, y1, x2, y2 = map(float, box)
    clipped = [max(0, min(width, int(np.floor(x1)))), max(0, min(height, int(np.floor(y1)))),
               max(0, min(width, int(np.ceil(x2)))), max(0, min(height, int(np.ceil(y2))))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError('AABB has no in-bounds area after clipping')
    was_clipped = x1 < 0 or y1 < 0 or x2 > width or y2 > height
    return clipped, [int(round(v)) for v in (x1, y1, x2, y2)], was_clipped


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(1e-9, aa + bb - inter))


def recognition_geometry(crop_width: int, crop_height: int, source_width: int, source_height: int) -> dict:
    """Provisional input screening, never a claim that a proposal depicts a complete card."""
    short, long = min(crop_width, crop_height), max(crop_width, crop_height)
    aspect = short / max(1, long)
    fraction = crop_width * crop_height / max(1, source_width * source_height)
    reasons = []
    if short < 200:
        reasons.append('source_short_edge_below_200_pixels')
    if not .45 <= aspect <= .95:
        reasons.append('aabb_aspect_outside_provisional_card_range')
    if fraction > .85:
        reasons.append('proposal_covers_almost_entire_scene')
    return {'source_short_edge_pixels': short, 'aabb_short_long_aspect': aspect,
            'source_area_fraction': fraction, 'recognition_geometry_eligible': not reasons,
            'geometry_screen_reasons': reasons, 'quality_thresholds_calibrated': False,
            'complete_physical_card_verified': False}


def run(detector_manifest: Path, output: Path, eligible_only: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output must be new or empty: {output}')
    data = json.loads(detector_manifest.read_text())
    if data.get('artifact_type') != 'zero_shot_localization_proposals':
        raise ValueError('Expected a zero_shot_localization_proposals manifest')
    if not data.get('model') or not data.get('revision') or not data.get('weight_sha256'):
        raise ValueError('Detector manifest is missing model provenance')
    output.mkdir(parents=True, exist_ok=True)
    crop_dir = output / 'crops'
    crop_dir.mkdir(exist_ok=True)
    all_queries = []
    rejected_proposals = []
    proposal_count = 0
    selection_buckets = {'onepiece_physical_correspondence': [], 'onepiece_overlay_correspondence': [],
                         'pokemon_physical_correspondence': [], 'other_proposals': []}
    for frame_result in data['results']:
        src = (ROOT / frame_result['source_image']).resolve()
        if ROOT.resolve() not in src.parents:
            raise ValueError('Detector source image escapes repository root')
        if sha256(src) != frame_result['source_sha256']:
            raise ValueError(f'Source frame hash mismatch: {src}')
        with Image.open(src) as im:
            rgb = np.asarray(im.convert('RGB'))
        height, width = rgb.shape[:2]
        if (width, height) != (frame_result['source_width'], frame_result['source_height']):
            raise ValueError(f'Source frame dimensions changed: {src}')
        if not frame_result['detections_by_prompt']:
            continue
        prompt_result = frame_result['detections_by_prompt'][0]
        prompt = prompt_result['prompt']
        teacher_phys = frame_result.get('teacher_physical_proposals', [])
        teacher_overlays = frame_result.get('teacher_excluded_overlays', [])
        teacher_path = (ROOT / frame_result['teacher_annotations']).resolve()
        if ROOT.resolve() not in teacher_path.parents:
            raise ValueError('Teacher annotation path escapes repository root')
        if sha256(teacher_path) != frame_result['teacher_sha256']:
            raise ValueError('Teacher annotation checksum mismatch')
        teacher_data = json.loads(teacher_path.read_text())
        teacher_frame = next((x for x in teacher_data['instances'] if x.get('image') == frame_result['source_image']), None)
        if teacher_frame is None:
            raise ValueError('No matching frame in teacher annotation provenance')
        source_group = teacher_frame.get('source_group')
        source_frame = teacher_frame.get('frame_index')
        if source_frame is None:
            source_frame = int(src.stem.split('_')[-1]) if src.stem.split('_')[-1].isdigit() else -1
        timestamp = teacher_frame.get('excerpt_local_timestamp_seconds')
        for det_index, det in enumerate(prompt_result['detections']):
            proposal_count += 1
            raw = det['bbox_xyxy_source_pixels']
            clipped, rounded_raw, was_clipped = clipped_aabb(raw, width, height)
            x1, y1, x2, y2 = clipped
            geometry = recognition_geometry(x2 - x1, y2 - y1, width, height)
            if eligible_only and not geometry['recognition_geometry_eligible']:
                rejected_proposals.append({'source_frame_image': str(src.relative_to(ROOT)),
                    'source_frame_sha256': frame_result['source_sha256'], 'detector_index': det_index,
                    'bbox_aabb_source_pixels': clipped, 'detection_score': float(det['score']), **geometry})
                continue
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError('Clipping produced an empty crop')
            source_key = hashlib.sha256(str(src.relative_to(ROOT)).encode()).hexdigest()[:12]
            stem = f"{source_key}_{src.stem}_p0_det_{det_index:03d}"
            crop_rel = Path('crops') / f'{stem}.png'
            Image.fromarray(crop).save(output / crop_rel, compress_level=1)
            physical_scores = [(iou(clipped, t['bbox_xyxy']), t) for t in teacher_phys]
            overlay_scores = [(iou(clipped, t['bbox_xyxy']), t) for t in teacher_overlays]
            best_physical = max(physical_scores, default=(0.0, None), key=lambda x: x[0])
            best_overlay = max(overlay_scores, default=(0.0, None), key=lambda x: x[0])
            if best_overlay[0] >= best_physical[0] and best_overlay[0] >= .5:
                correspondence = {'status': 'teacher_overlay_bbox_correspondence_proposal',
                                  'annotation_instance_id': best_overlay[1]['id'], 'iou': best_overlay[0]}
                bucket = 'onepiece_overlay_correspondence'
            elif best_physical[0] >= .5:
                correspondence = {'status': 'teacher_physical_bbox_correspondence_proposal',
                                  'annotation_instance_id': best_physical[1]['id'], 'iou': best_physical[0]}
                bucket = 'onepiece_physical_correspondence' if 'onepiece' in frame_result['source_image'] else 'pokemon_physical_correspondence'
            else:
                correspondence = {'status': 'no_teacher_bbox_correspondence', 'annotation_instance_id': None,
                                  'iou': max(best_physical[0], best_overlay[0])}
                bucket = 'other_proposals'
            query = {'image': crop_rel.as_posix(), 'sha256': sha256(output / crop_rel),
                     'source_group': source_group, 'source_frame': source_frame, 'time_seconds': timestamp,
                     'source_frame_image': str(src.relative_to(ROOT)), 'source_frame_sha256': frame_result['source_sha256'],
                     'source_width': width, 'source_height': height,
                     'detector_model': data['model'], 'detector_revision': data['revision'],
                     'detector_weights_sha256': data['weight_sha256'], 'detector_prompt': prompt,
                     'detection_score': float(det['score']), 'detector_label': det['text_label'],
                     'bbox_aabb_source_pixels_raw_float': [float(v) for v in raw],
                     'bbox_aabb_source_pixels_raw_rounded': rounded_raw,
                     'bbox_aabb_source_pixels_clipped': clipped, 'aabb_clipped_to_source': was_clipped,
                     'crop_width': int(crop.shape[1]), 'crop_height': int(crop.shape[0]),
                     **geometry,
                     'crop_policy': 'axis_aligned_source_pixel_bounds_only_no_perspective_corners',
                     'artwork_upright': 'unknown; OCR pipeline evaluates 0 and 180 degree candidates',
                     'identity_status': 'unresolved', 'physical_card_status': 'unknown',
                     'teacher_correspondence_proposal_only': correspondence,
                     'ground_truth_verified': False, 'training_promotion': 'not_promoted'}
            all_queries.append(query)
            selection_buckets[bucket].append(query)
    # OCR is capped downstream at 16 total. Put physical/overlay-correspondence examples from both games first,
    # then retain all remaining detections in the full manifest unfiltered.
    pokemon_other = [q for q in selection_buckets['other_proposals'] if 'luna-pokemon' in q['source_frame_image']]
    pokemon_other.sort(key=lambda q: -q['detection_score'])
    ordered = (selection_buckets['onepiece_physical_correspondence'][:11] +
               selection_buckets['onepiece_overlay_correspondence'][:1] +
               selection_buckets['pokemon_physical_correspondence'][:1] +
               pokemon_other[:3])
    selected_ids = {id(q) for q in ordered}
    queries_for_ocr = ordered + [q for q in all_queries if id(q) not in selected_ids]
    manifest = {'schema_version': 1, 'artifact_type': 'foundation_detector_source_aabb_crops',
                'detector_manifest': str(detector_manifest.resolve()),
                'detector_manifest_sha256': sha256(detector_manifest),
                'detector_model': data['model'], 'detector_revision': data['revision'],
                'detector_weights_sha256': data['weight_sha256'],
                'crop_policy': 'clipped axis-aligned source-pixel bounds; no synthesized perspective corners or upscaling',
                'all_detection_count': proposal_count, 'crop_count': len(all_queries),
                'geometry_filter_enabled': eligible_only, 'geometry_rejected_proposals': rejected_proposals,
                'ocr_priority_order': 'teacher correspondence proposals first, then other detections; no identity labels',
                'recognition_geometry_eligible_count': sum(q['recognition_geometry_eligible'] for q in all_queries),
                'queries': queries_for_ocr, 'ground_truth_verified': False, 'training_promotion': 'not_promoted'}
    (output / 'queries.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--eligible-only', action='store_true', help='Save only provisional geometry-eligible recognition crops; record every rejected proposal')
    args = p.parse_args()
    result = run(args.detector_manifest, args.output, args.eligible_only)
    print(json.dumps({'all_detections': result['all_detection_count'], 'saved_crops': result['crop_count'],
                      'geometry_eligible': result['recognition_geometry_eligible_count'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
