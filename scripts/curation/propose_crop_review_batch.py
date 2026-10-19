#!/usr/bin/env python3
"""Cached, proposal-only local feature matching for observed card crops.

Observed names and optional catalog IDs only narrow the candidate references.
Every match is unverified and must be checked visually by a reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def repo_file(repo: Path, value: str) -> Path:
    path = (repo / value).resolve(strict=True)
    if not path.is_relative_to(repo.resolve()) or not path.is_file():
        raise ValueError(f'Expected repository file: {value}')
    return path


def observation_fields(obs: dict) -> tuple[str, str, str | None, str | None]:
    name = obs.get('observed_name') or obs.get('observation', {}).get('name_read')
    game = obs.get('game') or obs.get('observed_game')
    language = obs.get('language') or obs.get('observation', {}).get('language')
    if language is None:
        language = obs.get('language_observed')
        if language == 'English':
            language = 'en'
    ids = obs.get('candidate_card_ids')
    if ids is None:
        ids = obs.get('candidate_base_ids')
    return str(name or ''), str(game or ''), language, set(map(str, ids)) if ids else None


def image_path_field(obs: dict) -> str:
    return (obs.get('crop_path') or obs.get('image') or obs.get('source_crop_path')
            or obs.get('crop_image') or '')


def scaled_image_features(detector, image: np.ndarray, max_dim: int, max_features: int):
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1:
        work = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = image
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if keypoints and len(keypoints) > max_features:
        order = sorted(range(len(keypoints)), key=lambda i: keypoints[i].response, reverse=True)[:max_features]
        keypoints = [keypoints[i] for i in order]
        descriptors = descriptors[order]
    return keypoints, descriptors, (w / work.shape[1], h / work.shape[0]), work.shape[1], work.shape[0]


def coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / max(1, width * height))


def match_features(ref_pack, query_pack, matcher, ratio: float = .72) -> dict:
    kr, dr, rs, rw, rh = ref_pack
    kq, dq, qs, qw, qh = query_pack
    result = {'feature_count_reference': len(kr), 'feature_count_query': len(kq),
              'good_match_count': 0, 'inlier_count': 0, 'inlier_ratio': 0.0,
              'reference_coverage': 0.0, 'query_coverage': 0.0,
              'reprojection_error_px': None, 'quad_working_xy': None,
              'quad_query_native_xy': None, 'accepted': False, 'rejection_reasons': []}
    if dr is None or dq is None:
        result['rejection_reasons'].append('insufficient_descriptors')
        return result
    if (not isinstance(dr, np.ndarray) or not isinstance(dq, np.ndarray)
            or dr.ndim != 2 or dq.ndim != 2 or dr.shape[1] == 0
            or dr.shape[1] != dq.shape[1] or dr.dtype != dq.dtype
            or not np.issubdtype(dr.dtype, np.number)
            or not np.isfinite(dr).all() or not np.isfinite(dq).all()
            or not np.any(dr) or not np.any(dq)):
        result['rejection_reasons'].append('invalid_or_zero_descriptors')
        return result
    if len(dr) < 4 or len(dq) < 4:
        result['rejection_reasons'].append('insufficient_descriptors')
        return result
    try:
        pairs = matcher.knnMatch(dr, dq, k=2)
    except cv2.error:
        result['rejection_reasons'].append('descriptor_matching_failed')
        return result
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < ratio * b.distance]
    result['good_match_count'] = len(good)
    if len(good) < 8:
        result['rejection_reasons'].append('too_few_ratio_test_matches')
        return result
    src = np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    if H is None or mask is None:
        result['rejection_reasons'].append('homography_not_found')
        return result
    inliers = mask.ravel().astype(bool)
    a, b = src.reshape(-1, 2)[inliers], dst.reshape(-1, 2)[inliers]
    n = len(a)
    result.update(inlier_count=n, inlier_ratio=float(n / len(good)),
                  reference_coverage=coverage(a, rw, rh), query_coverage=coverage(b, qw, qh))
    projected = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    result['reprojection_error_px'] = float(np.sqrt(np.mean(np.sum((projected - b) ** 2, axis=1))))
    ref_corners = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(ref_corners, H).reshape(4, 2)
    native_quad = quad * np.asarray(qs, np.float32)
    result['quad_working_xy'] = quad.tolist()
    result['quad_query_native_xy'] = native_quad.tolist()
    result['query_working_size'] = [qw, qh]
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    convex = cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2))
    bounds = bool(np.all(quad[:, 0] >= -.03 * qw) and np.all(quad[:, 0] <= 1.03 * qw)
                  and np.all(quad[:, 1] >= -.03 * qh) and np.all(quad[:, 1] <= 1.03 * qh))
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    height = (np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[3] - quad[0])) / 2
    aspect = width / max(1e-6, height)
    plausible = .48 <= aspect <= .82 and .18 <= area / max(1, qw * qh) <= 1.10
    reasons = result['rejection_reasons']
    if n < 12: reasons.append('too_few_ransac_inliers')
    if result['inlier_ratio'] < .45: reasons.append('low_inlier_ratio')
    if result['reference_coverage'] < .12: reasons.append('reference_inliers_not_spatially_spread')
    if result['query_coverage'] < .12: reasons.append('query_inliers_not_spatially_spread')
    if result['reprojection_error_px'] > 4: reasons.append('high_reprojection_error')
    if not convex or area < .15 * qw * qh: reasons.append('degenerate_or_small_mapped_quad')
    if not bounds: reasons.append('mapped_quad_out_of_query_crop')
    if not plausible: reasons.append('implausible_card_quad')
    result['accepted'] = not reasons
    return result


def candidate_score(metrics: dict) -> tuple:
    # Spatial support comes first. Strict aspect/bounds heuristics remain visible
    # but cannot bury a strong perspective match behind a weaker generic template.
    err = metrics.get('reprojection_error_px')
    supported = (metrics['inlier_count'] >= 12 and metrics['inlier_ratio'] >= .45
                 and metrics['reference_coverage'] >= .12 and metrics['query_coverage'] >= .12
                 and err is not None and err <= 4)
    support = metrics['inlier_count'] * min(1.0, metrics['reference_coverage'] / .30) * min(1.0, metrics['query_coverage'] / .30)
    return (int(supported), support, metrics['inlier_count'], metrics['inlier_ratio'],
            metrics['reference_coverage'], metrics['query_coverage'], int(metrics['accepted']),
            -(err if err is not None else 1e9))


def rectification_eligibility(metrics: dict, quad_source: np.ndarray | None,
                              source_shape: tuple[int, ...] | None) -> tuple[bool, list[str]]:
    """Proposal-only guards for a preview warp; card aspect is intentionally ignored."""
    reasons = []
    if metrics['inlier_count'] < 12: reasons.append('too_few_ransac_inliers')
    if metrics['inlier_ratio'] < .45: reasons.append('low_inlier_ratio')
    if metrics['reference_coverage'] < .12: reasons.append('reference_inliers_not_spatially_spread')
    if metrics['query_coverage'] < .12: reasons.append('query_inliers_not_spatially_spread')
    err = metrics.get('reprojection_error_px')
    if err is None or err > 4: reasons.append('high_or_missing_reprojection_error')
    quad = metrics.get('quad_working_xy')
    if quad is None:
        reasons.append('mapped_quad_missing')
    else:
        q = np.asarray(quad, np.float32)
        area = abs(float(cv2.contourArea(q)))
        qw, qh = metrics['query_working_size']
        if not cv2.isContourConvex(q.reshape(-1, 1, 2)) or area < .15 * qw * qh:
            reasons.append('degenerate_or_small_mapped_quad')
    if quad_source is None or source_shape is None:
        reasons.append('source_frame_geometry_missing')
    else:
        h, w = source_shape[:2]
        if (not np.isfinite(quad_source).all() or np.any(quad_source[:, 0] < 0)
                or np.any(quad_source[:, 0] > w - 1) or np.any(quad_source[:, 1] < 0)
                or np.any(quad_source[:, 1] > h - 1)):
            reasons.append('mapped_quad_outside_source_frame')
    return not reasons, reasons


def map_crop_quad_to_source(quad_crop: np.ndarray, crop_size: tuple[int, int],
                            source_bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Map crop-native x/y points into source pixels using the crop's actual dimensions."""
    crop_w, crop_h = crop_size
    x1, y1, x2, y2 = source_bbox_xyxy
    if crop_w <= 0 or crop_h <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError('Crop size and source bbox must have positive dimensions')
    quad = np.asarray(quad_crop, np.float32)
    if quad.ndim != 2 or quad.shape[1] != 2 or not np.isfinite(quad).all():
        raise ValueError('Expected a finite Nx2 crop quad')
    sx, sy = (x2 - x1) / crop_w, (y2 - y1) / crop_h
    return quad * np.float32([sx, sy]) + np.float32([x1, y1])


def build_contact_sheets(repo: Path, output: Path, rows: list[dict], per_sheet: int = 12) -> None:
    for start in range(0, len(rows), per_sheet):
        subset = rows[start:start + per_sheet]
        sheet = Image.new('RGB', (1440, math.ceil(len(subset) / 3) * 420), (248, 248, 248))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(subset):
            x, y = (i % 3) * 480, (i // 3) * 420
            crop = Image.open(repo / row['crop_path']).convert('RGB')
            crop.thumbnail((148, 300)); sheet.paste(crop, (x + 4, y + 4))
            candidates = row['candidates']
            if candidates:
                ref = Image.open(repo / candidates[0]['reference']['path']).convert('RGB')
                ref.thumbnail((148, 300)); sheet.paste(ref, (x + 160, y + 4))
                rectified = candidates[0].get('rectified_path')
                if rectified:
                    rect = Image.open(repo / rectified).convert('RGB')
                    rect.thumbnail((148, 300)); sheet.paste(rect, (x + 318, y + 4))
                review_id = row.get('rootreview_index', row.get('review_index', row['index']))
                lines = [f"#{review_id} {row['observed_name']}"]
                for rank, candidate in enumerate(candidates, 1):
                    m = candidate['geometry']
                    lines.append(f"{rank}. {candidate['reference']['card_id']} inl={m['inlier_count']} cov={m['reference_coverage']:.2f}/{m['query_coverage']:.2f} {'geom' if m['accepted'] else 'weak'}")
                lines.append('PROPOSED ONLY · review artwork')
                draw.multiline_text((x + 4, y + 310), '\n'.join(lines), fill='black', spacing=2)
            else:
                review_id = row.get('rootreview_index', row.get('review_index', row['index']))
                draw.multiline_text((x + 4, y + 310), f"#{review_id} {row['observed_name']}\nNO SHORTLIST MATCH\nPROPOSED ONLY", fill='black')
        sheet.save(output / f'paired-contact-{start // per_sheet + 1:03d}.jpg', quality=92)


def build_reference_grid(repo: Path, references: list[dict], name: str, output: Path,
                         columns: int = 8, per_page: int = 40) -> dict:
    """Render every exact-name reference as a review grid; captions are hypotheses only."""
    matches = [r for r in references if str(r.get('game', '')).casefold() == 'onepiece'
               and str(r.get('name', '')).casefold() == name.casefold()]
    matches.sort(key=lambda r: (str(r.get('language', '')), str(r.get('card_id', '')), str(r.get('path', ''))))
    if not matches:
        raise ValueError(f'No exact One Piece references found for name: {name}')
    output.mkdir(parents=True, exist_ok=True)
    slug = ''.join(c.lower() if c.isalnum() else '-' for c in name).strip('-')
    pages = []
    tile_w, tile_h, image_h = 200, 240, 180
    for start in range(0, len(matches), per_page):
        subset = matches[start:start + per_page]
        sheet = Image.new('RGB', (columns * tile_w, math.ceil(len(subset) / columns) * tile_h), (250, 250, 250))
        draw = ImageDraw.Draw(sheet)
        for j, ref in enumerate(subset):
            path = repo_file(repo, ref['path'])
            if sha(path) != ref['sha256']:
                raise ValueError(f'Reference hash mismatch: {ref["path"]}')
            image = Image.open(path).convert('RGB')
            image.thumbnail((145, image_h))
            col, row = j % columns, j // columns
            x, y = col * tile_w, row * tile_h
            sheet.paste(image, (x + (tile_w - image.width) // 2, y + 2))
            caption = f"{ref.get('card_id', '?')}\n{ref.get('language', '?')} · {ref.get('set_id', '')}"
            draw.multiline_text((x + 4, y + image_h + 8), caption, fill='black', spacing=2)
        page_path = output / f'{slug}-{start // per_page + 1:02d}.jpg'
        sheet.save(page_path, quality=94)
        pages.append({'path': page_path.relative_to(repo).as_posix(), 'first_reference': start,
                      'reference_count': len(subset)})
    result = {'name_filter': name, 'game_filter': 'onepiece', 'status': 'REFERENCE_GRID_FOR_VISUAL_REVIEW',
              'identity_asserted': False, 'reference_count': len(matches), 'pages': pages,
              'references': [{'card_id': r.get('card_id'), 'language': r.get('language'),
                              'path': r.get('path'), 'sha256': r.get('sha256')} for r in matches]}
    (output / f'{slug}-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def propose(repo: Path, observations: list[dict], references: list[dict], output: Path,
            max_dim: int = 700, max_features: int = 1200) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rect_dir = output / 'rectified'; rect_dir.mkdir(exist_ok=True)
    detector = cv2.SIFT_create(nfeatures=max_features) if hasattr(cv2, 'SIFT_create') else cv2.ORB_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2 if hasattr(cv2, 'SIFT_create') else cv2.NORM_HAMMING)
    cv2.setNumThreads(4)

    # Build strict observed-name/game/language/optional-ID shortlists first.
    by_key: dict[tuple, list[dict]] = {}
    for ref in references:
        by_key.setdefault((str(ref.get('game', '')).casefold(), str(ref.get('name', '')).casefold(),
                           str(ref.get('language', '')).casefold()), []).append(ref)
    needed = set()
    query_rows = []
    for i, obs in enumerate(observations):
        name, game, language, ids = observation_fields(obs)
        crop_rel = image_path_field(obs)
        if not name or not game or not crop_rel:
            raise ValueError(f'Observation {i} needs observed_name, game, and crop image path')
        lang = str(language or '').casefold()
        candidates = by_key.get((game.casefold(), name.casefold(), lang), []) if lang else [
            r for (g, n, _), values in by_key.items() if g == game.casefold() and n == name.casefold() for r in values]
        if ids is not None:
            candidates = [r for r in candidates if any(
                str(r.get('card_id')) == card_id or str(r.get('card_id', '')).startswith(card_id + '_')
                for card_id in ids)]
        needed.update(str(r['path']) for r in candidates)
        query_rows.append((i, obs, name, game, language, crop_rel, candidates))

    feature_cache: dict[str, tuple] = {}
    hash_cache: dict[str, str] = {}
    path_meta: dict[str, tuple[Path, np.ndarray]] = {}
    for rel in sorted(needed):
        path = repo_file(repo, rel)
        actual = sha(path)
        expected = next(r['sha256'] for r in references if r['path'] == rel)
        if actual != expected:
            raise ValueError(f'Reference hash mismatch: {rel}')
        image = rgb(path)
        feature_cache[rel] = scaled_image_features(detector, image, max_dim, max_features)
        hash_cache[rel] = actual
        path_meta[rel] = (path, image)

    from scripts.curation.verify_card_geometry import source_resolution_crop
    rows = []
    for i, obs, name, game, language, crop_rel, refs in query_rows:
        crop_path = repo_file(repo, crop_rel)
        actual_crop_hash = sha(crop_path)
        expected_crop_hash = obs.get('crop_sha256') or obs.get('sha256') or obs.get('source_crop_sha256')
        if expected_crop_hash and actual_crop_hash != expected_crop_hash:
            raise ValueError(f'Query hash mismatch: {crop_rel}')
        query = rgb(crop_path)
        qpack = scaled_image_features(detector, query, max_dim, max_features)
        ranked = []
        for ref in refs:
            metrics = match_features(feature_cache[ref['path']], qpack, matcher)
            ranked.append((candidate_score(metrics), ref, metrics))
        ranked.sort(key=lambda x: x[0], reverse=True)
        row = dict(obs)
        row.update(index=i, status='PROPOSED_ONLY', proposal_only=True,
                   observed_name=name, game=game, language=language,
                   crop_path=crop_rel, crop_sha256=actual_crop_hash, candidates=[])
        source_rel = obs.get('source_frame_image') or obs.get('source_frame')
        source_bbox = (obs.get('bbox') or obs.get('bbox_aabb_source_pixels_clipped')
                       or obs.get('bbox_aabb_source_pixels_raw_rounded'))
        source_path = None
        source_image = None
        source_hash = None
        if source_rel and source_bbox:
            source_path = repo_file(repo, source_rel)
            source_hash = sha(source_path)
            expected = obs.get('source_frame_sha256')
            if expected and source_hash != expected:
                raise ValueError(f'Source frame hash mismatch: {source_rel}')
            source_image = rgb(source_path)
            x1, y1, x2, y2 = map(float, source_bbox)
            row['source_frame_image'] = source_rel
            row['source_frame_sha256'] = source_hash
            row['source_bbox_xyxy'] = [x1, y1, x2, y2]
        for rank, (_, ref, metrics) in enumerate(ranked[:3], 1):
            candidate = {'rank': rank,
                         'reference': {k: ref.get(k) for k in ('game', 'language', 'card_id', 'name', 'set_id', 'path', 'sha256', 'provider', 'printing') if k in ref},
                         'geometry': metrics}
            quad_crop = np.asarray(metrics['quad_query_native_xy'], np.float32) if metrics['quad_query_native_xy'] else None
            quad_source = None
            if quad_crop is not None and source_image is not None:
                # Detection crop pixels and rounded source bbox can differ by one pixel;
                # map each axis by its actual native dimensions before offsetting.
                crop_h, crop_w = query.shape[:2]
                quad_source = map_crop_quad_to_source(
                    quad_crop, (crop_w, crop_h), tuple(map(float, source_bbox)))
            eligible, eligibility_reasons = rectification_eligibility(
                metrics, quad_source, source_image.shape if source_image is not None else None)
            candidate['rectification_proposal_eligible'] = eligible
            candidate['rectification_proposal_rejection_reasons'] = eligibility_reasons
            candidate['proposal_status'] = 'RECTIFICATION_PREVIEW_PROPOSED_ONLY' if eligible else 'NO_RECTIFICATION_PREVIEW'
            if eligible and quad_crop is not None and source_image is not None and quad_source is not None:
                try:
                    rectified = source_resolution_crop(source_image, quad_source, aspect_ratio=224 / 312)
                except ValueError as exc:
                    candidate['rectification_proposal_eligible'] = False
                    candidate['proposal_status'] = 'NO_RECTIFICATION_PREVIEW'
                    candidate['rectification_error'] = str(exc)
                else:
                    safe_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(ref['card_id']))
                    dest = rect_dir / f'{i:04d}-{rank}-{safe_id}.png'
                    Image.fromarray(rectified).save(dest)
                    candidate.update(rectified_path=dest.relative_to(repo).as_posix(),
                                     rectified_sha256=sha(dest), quad_source_frame_xy=quad_source.tolist())
            elif quad_crop is not None and source_image is None:
                candidate['quad_query_native_xy'] = quad_crop.tolist()
                candidate['rectification_error'] = 'source_frame_image and source bbox are required for native-frame rectification'
            row['candidates'].append(candidate)
        rows.append(row)
        print(f'{i + 1}/{len(observations)} {name}: {len(refs)} reference templates', flush=True)

    build_contact_sheets(repo, output, rows)
    result = {'schema_version': 1, 'artifact_type': 'cached_feature_crop_review_proposals',
              'status': 'PROPOSED_ONLY', 'identity_verified': False, 'labels_approved': False,
              'training_promotion': 'not_promoted', 'feature_method': 'SIFT' if hasattr(cv2, 'SIFT_create') else 'ORB',
              'feature_cache_policy': {'max_working_dimension': max_dim, 'max_features': max_features,
                                       'features_computed_once_per_unique_reference_and_query': True,
                                       'ratio_test': .72, 'ransac_threshold_working_pixels': 2.5},
              'rectification_proposal_policy': 'preview only if spatial homography support passes (12 inliers, ratio .45, coverage .12 each, reprojection <=4, convex quad area >=15% crop) and mapped quad lies within the full source frame; aspect plausibility does not gate preview; strict geometry accepted/rejection reasons remain recorded',
              'shortlist_policy': 'exact observed name and game; exact observed language when supplied; optional candidate_card_ids only narrows templates',
              'proposals': rows}
    (output / 'proposals.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=Path('datasets/review/all_references_v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-working-dimension', type=int, default=700)
    parser.add_argument('--max-features', type=int, default=1200)
    parser.add_argument('--reference-grid-name', action='append', default=[],
                        help='Also render an 8-column review grid for every exact One Piece reference name (repeatable).')
    parser.add_argument('--reference-grid-dir', type=Path,
                        help='Output directory for requested reference grids.')
    args = parser.parse_args()
    observations = json.loads(args.observations.read_text())
    references = json.loads(args.references.read_text())
    obs_rows = observations.get('observations', observations.get('queries', observations.get('proposals')))
    ref_rows = references.get('references')
    if not isinstance(obs_rows, list) or not isinstance(ref_rows, list):
        parser.error('Expected observations/queries and references arrays')
    if args.reference_grid_name and not args.reference_grid_dir:
        parser.error('--reference-grid-dir is required with --reference-grid-name')
    result = propose(ROOT, obs_rows, ref_rows, args.output.resolve(), args.max_working_dimension, args.max_features)
    grids = [build_reference_grid(ROOT, ref_rows, name, args.reference_grid_dir.resolve())
             for name in args.reference_grid_name]
    print(json.dumps({'proposal_count': len(result['proposals']), 'output': str(args.output.resolve()),
                      'contact_sheets': math.ceil(len(result['proposals']) / 12),
                      'reference_grids': [{'name': g['name_filter'], 'references': g['reference_count'], 'pages': g['pages']} for g in grids]}))


if __name__ == '__main__':
    main()
