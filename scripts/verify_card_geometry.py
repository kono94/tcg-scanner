"""Bounded feature-geometry teacher prototype; results are proposals, never ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.canonical import canonicalize_card  # noqa: E402
from training.card_recognizer.retrieval import CosineGallery, FrozenCardEncoder, file_sha256  # noqa: E402


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / max(1.0, width * height))


def verify_homography(reference_rgb: np.ndarray, query_rgb: np.ndarray) -> dict:
    """Find reference-to-query card geometry and reject localized artwork-only matches."""
    reasons: list[str] = []
    detector = cv2.SIFT_create(nfeatures=1800) if hasattr(cv2, 'SIFT_create') else cv2.ORB_create(nfeatures=2500)
    norm = cv2.NORM_L2 if hasattr(cv2, 'SIFT_create') else cv2.NORM_HAMMING
    kp_ref, des_ref = detector.detectAndCompute(_gray(reference_rgb), None)
    kp_q, des_q = detector.detectAndCompute(_gray(query_rgb), None)
    base = {'feature_count_reference': len(kp_ref), 'feature_count_query': len(kp_q), 'good_match_count': 0,
            'inlier_count': 0, 'inlier_ratio': 0.0, 'reference_coverage': 0.0, 'query_coverage': 0.0,
            'reprojection_error_px': None, 'quad_xy': None, 'accepted': False, 'rejection_reasons': reasons}
    if des_ref is None or des_q is None or len(des_ref) < 4 or len(des_q) < 4:
        reasons.append('insufficient_descriptors')
        return base
    pairs = cv2.BFMatcher(norm).knnMatch(des_ref, des_q, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
    base['good_match_count'] = len(good)
    if len(good) < 8:
        reasons.append('too_few_ratio_test_matches')
        return base
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
    if H is None or mask is None:
        reasons.append('homography_not_found')
        return base
    inliers = mask.ravel().astype(bool)
    a, b = src.reshape(-1, 2)[inliers], dst.reshape(-1, 2)[inliers]
    n = int(inliers.sum())
    base.update(inlier_count=n, inlier_ratio=float(n / len(good)),
                reference_coverage=_coverage(a, reference_rgb.shape[1], reference_rgb.shape[0]),
                query_coverage=_coverage(b, query_rgb.shape[1], query_rgb.shape[0]))
    projected = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    base['reprojection_error_px'] = float(np.sqrt(np.mean(np.sum((projected - b) ** 2, axis=1))))
    rh, rw = reference_rgb.shape[:2]
    qh, qw = query_rgb.shape[:2]
    ref_corners = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(ref_corners, H).reshape(4, 2)
    base['quad_xy'] = quad.tolist()
    contour = quad.astype(np.float32)
    convex = cv2.isContourConvex(contour.reshape(-1, 1, 2))
    area = abs(float(cv2.contourArea(contour)))
    bounds = bool(np.all(quad[:, 0] >= -0.03 * qw) and np.all(quad[:, 0] <= 1.03 * qw) and
                  np.all(quad[:, 1] >= -0.03 * qh) and np.all(quad[:, 1] <= 1.03 * qh))
    widths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in (0, 2)]
    heights = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in (1, 3)]
    ratio = (sum(widths) / 2) / max(1e-6, sum(heights) / 2)
    plausible = 0.48 <= ratio <= 0.82 and 0.18 <= area / (qw * qh) <= 1.10
    if n < 12: reasons.append('too_few_ransac_inliers')
    if base['inlier_ratio'] < 0.45: reasons.append('low_inlier_ratio')
    if base['reference_coverage'] < 0.12: reasons.append('reference_inliers_not_spatially_spread')
    if base['query_coverage'] < 0.12: reasons.append('query_inliers_not_spatially_spread')
    if base['reprojection_error_px'] > 4.0: reasons.append('high_reprojection_error')
    if not convex or area < 0.15 * qw * qh: reasons.append('degenerate_or_small_mapped_quad')
    if not bounds: reasons.append('mapped_quad_out_of_teacher_crop')
    if not plausible: reasons.append('implausible_card_quad')
    base['accepted'] = not reasons
    return base


def _pil_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def source_resolution_crop(rgb: np.ndarray, quad: np.ndarray, max_edge: int = 1024,
                           aspect_ratio: float | None = None) -> np.ndarray:
    """Rectify in-bounds source geometry without collapsing printed text to encoder size."""
    quad = np.asarray(quad, np.float32)
    h, w = rgb.shape[:2]
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError('Source quad must contain four finite points')
    if np.any(quad < 0) or np.any(quad[:, 0] > w - 1) or np.any(quad[:, 1] > h - 1):
        raise ValueError('Mapped quad extends outside source pixels')
    if max_edge < 2:
        raise ValueError('max_edge must be at least two')
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    width, height = float(min(edges[0], edges[2])), float(min(edges[1], edges[3]))
    if aspect_ratio is not None:
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError('aspect_ratio must be finite and positive')
        # Recover the card-plane shape while staying below both source-edge budgets.
        height = min(height, width / aspect_ratio)
        width = height * aspect_ratio
    scale = min(1.0, max_edge / max(width, height))
    return canonicalize_card(rgb, quad, width=max(2, int(width * scale)), height=max(2, int(height * scale)))


def run(teacher_path: Path, gallery_path: Path, weights: Path, ref_root: Path, output: Path, limit: int = 11) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(teacher_path.read_text())
    gallery = CosineGallery.load(gallery_path)
    encoder = FrozenCardEncoder(weights)
    records = []
    instances = [(frame, i) for frame in annotations['instances'] for i in frame['instances']
                 if i.get('physical_card') is True and i.get('overlay_graphic') is not True]
    for frame_record, instance in instances[:limit]:
        frame_path = (ROOT / frame_record['image']).resolve()
        if not frame_path.is_relative_to(ROOT):
            raise ValueError('Source frame must be inside the repository')
        source_hash = file_sha256(frame_path)
        if source_hash != frame_record.get('image_sha256'):
            raise ValueError('Teacher source-frame checksum mismatch')
        rgb = _pil_rgb(frame_path)
        x1, y1, x2, y2 = map(int, instance['bbox_xyxy'])
        h, w = rgb.shape[:2]
        record = {'proposal_id': instance['id'], 'source_frame': str(frame_path.relative_to(ROOT)),
                  'source_frame_sha256': source_hash, 'teacher_bbox_xyxy': [x1, y1, x2, y2],
                  'teacher_annotation_sha256': file_sha256(teacher_path), 'teacher_geometry_status': 'approximate_bbox_not_ground_truth',
                  'training_promotion': 'not_promoted', 'candidate_results': []}
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            record['rejection_reasons'] = ['teacher_bbox_out_of_source_bounds']; records.append(record); continue
        crop = rgb[y1:y2, x1:x2]
        ranked = gallery.search(encoder.encode([crop]), encoder.contract_hash, k=20)[0]
        for rank, candidate in enumerate(ranked, 1):
            ident = candidate['identity']
            game, lang, card_id = ident.split(':', 2)
            if game != 'onepiece': continue
            rel = candidate.get('source_path')
            ref_path = (ref_root / rel).resolve() if rel else None
            if not ref_path or not ref_path.is_file() or ref_root.resolve() not in ref_path.parents:
                continue
            ref = _pil_rgb(ref_path)
            result = verify_homography(ref, crop)
            item = {'rank': rank, 'identity': ident, 'similarity': candidate['similarity'],
                    'reference_path': str(ref_path.relative_to(ROOT)), 'reference_sha256': file_sha256(ref_path), **result}
            if result['accepted']:
                quad = np.asarray(result['quad_xy'], np.float32) + np.float32([x1, y1])
                item['quad_source_xy'] = quad.tolist()
                try:
                    ocr_crop = source_resolution_crop(rgb, quad)
                except ValueError:
                    item['accepted'] = False
                    item['rejection_reasons'].append('mapped_quad_out_of_source_bounds_or_invalid')
                    record['candidate_results'].append(item)
                    continue
                key = hashlib.sha256(json.dumps([source_hash, instance['id'], ident], separators=(',', ':')).encode()).hexdigest()[:24]
                item['canonical_crop'] = f'canonical/{key}.png'
                item['ocr_crop'] = f'ocr/{key}.png'
                canonical = canonicalize_card(rgb, corners=quad)
                (output / 'canonical').mkdir(exist_ok=True)
                (output / 'ocr').mkdir(exist_ok=True)
                Image.fromarray(canonical).save(output / item['canonical_crop'])
                Image.fromarray(ocr_crop).save(output / item['ocr_crop'])
                item['canonical_sha256'] = file_sha256(output / item['canonical_crop'])
                item['ocr_sha256'] = file_sha256(output / item['ocr_crop'])
                item['ocr_dimensions'] = [ocr_crop.shape[1], ocr_crop.shape[0]]
            record['candidate_results'].append(item)
        record['proposal_status'] = 'geometry_proposal_unverified' if any(x['accepted'] for x in record['candidate_results']) else 'unresolved_quarantined'
        records.append(record)
    result = {'schema_version': 1, 'artifact_type': 'feature_geometry_teacher_proposals', 'ground_truth_verified': False,
              'training_promotion': 'not_promoted', 'candidate_limit_per_crop': 20,
              'ocr_crop_policy': 'in-bounds source quad; minimum opposing edge lengths; no intentional upscale; longest edge capped at 1024',
              'teacher_annotation': str(teacher_path), 'teacher_annotation_sha256': file_sha256(teacher_path),
              'gallery': str(gallery_path), 'encoder_contract_sha256': encoder.contract_hash,
              'weights_sha256': encoder.contract['weights_sha256'], 'records': records}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-annotations', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.limit <= 11: parser.error('--limit must be between 1 and 11')
    result = run(args.teacher_annotations, args.gallery, args.weights, args.reference_root, args.output, args.limit)
    print(json.dumps({'records': len(result['records']), 'accepted': sum(x.get('proposal_status') == 'geometry_proposal_unverified' for x in result['records']), 'output': str(args.output)}))


if __name__ == '__main__': main()
