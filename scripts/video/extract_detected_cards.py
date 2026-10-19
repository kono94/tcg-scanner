#!/usr/bin/env python3
"""Extract bounded, unlabeled per-card crops from mined frames using a local YOLO-pose corner model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.recognition.canonical import canonicalize_card  # noqa: E402

MAX_CROP_EDGE = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def portrait_aligned_corners(vertices: np.ndarray) -> np.ndarray:
    """Return a cyclic convex quad ordered TL,TR,BR,BL with its long axis vertical.

    Corner geometry determines portrait alignment only. It cannot determine artwork-up;
    callers must treat upright orientation as unknown and handle 0/180 separately.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Card corners must be four finite 2-D points')
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    cyclic = points[order]
    area = _signed_area(cyclic)
    if abs(area) < 1.0:
        raise ValueError('Card quad area is too small')
    # Image-coordinate clockwise order gives positive signed area.
    if area < 0:
        cyclic = cyclic[::-1]
    edges = np.roll(cyclic, -1, axis=0) - cyclic
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1.0):
        raise ValueError('Card quad contains a degenerate edge')
    crosses = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(crosses > 1e-3) or np.all(crosses < -1e-3)):
        raise ValueError('Card corners must form a convex quadrilateral')
    # The two short edges are the card's width edges. For nearly square shapes the
    # inferred portrait axis is not meaningful, so reject instead of guessing.
    short_pair = [i for i in range(4) if lengths[i] <= min(lengths) * 1.15]
    if len(short_pair) != 2 or (short_pair[1] - short_pair[0]) % 2 != 0:
        raise ValueError('Card quad does not have a clear pair of opposite short edges')
    if max(lengths) / min(lengths) < 1.15:
        raise ValueError('Card quad aspect ratio is too ambiguous for portrait alignment')
    candidates = []
    for i in short_pair:
        midpoint_y = float((cyclic[i, 1] + cyclic[(i + 1) % 4, 1]) / 2)
        midpoint_x = float((cyclic[i, 0] + cyclic[(i + 1) % 4, 0]) / 2)
        candidates.append((midpoint_y, midpoint_x, i))
    start = min(candidates)[2]
    ordered = np.roll(cyclic, -start, axis=0)
    if ordered[1, 0] < ordered[0, 0]:
        # Reverse while keeping the selected top edge as the first edge.
        ordered = np.array([ordered[1], ordered[0], ordered[3], ordered[2]])
    if _signed_area(ordered) <= 0:
        raise ValueError('Unable to establish stable cyclic corner order')
    return ordered.astype(np.float32)


def validate_source_quad(vertices: np.ndarray, width: int, height: int,
                         min_card_short_edge: float) -> tuple[np.ndarray, float, float]:
    """Validate a predicted quad and return ordered corners and width/height edge estimates."""
    if width < 2 or height < 2:
        raise ValueError('Source frame dimensions must be at least 2 pixels')
    corners = portrait_aligned_corners(vertices)
    if np.any(corners[:, 0] < 0) or np.any(corners[:, 1] < 0) or \
            np.any(corners[:, 0] > width - 1) or np.any(corners[:, 1] > height - 1):
        raise ValueError('Card quad extends outside source-frame bounds')
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    card_width = float((lengths[0] + lengths[2]) / 2)
    card_height = float((lengths[1] + lengths[3]) / 2)
    if card_width < min_card_short_edge:
        raise ValueError(f'Card short edge {card_width:.1f}px is below {min_card_short_edge}px minimum')
    if card_height < card_width:
        raise ValueError('Portrait ordering failed: long edge is not vertical')
    return corners, card_width, card_height


def crop_dimensions(card_width: float, card_height: float, max_edge: int = MAX_CROP_EDGE) -> tuple[int, int, str]:
    """Choose source-scale dimensions capped at max_edge, without intentional upscaling."""
    if not all(math.isfinite(v) and v > 0 for v in (card_width, card_height)) or max_edge < 2:
        raise ValueError('Crop dimensions and edge cap must be positive and finite')
    scale = min(1.0, max_edge / max(card_width, card_height))
    width = max(2, int(round(card_width * scale)))
    height = max(2, int(round(card_height * scale)))
    mode = 'downsampled_to_1024_cap' if scale < 1 else 'source_scale_no_upscale'
    return width, height, mode


def _manifest_frames(manifest_path: Path, max_frames: int) -> tuple[list[tuple[Path, dict]], dict, Path]:
    root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest.get('records')
    if manifest.get('artifact_type') != 'unlabeled_video_frame_candidates' or not isinstance(rows, list):
        raise ValueError('Expected mine_video_frames manifest with records')
    source = manifest.get('source') or {}
    if not isinstance(source, dict):
        raise ValueError('Invalid source metadata in frames manifest')
    records = []
    for item in rows[:max_frames]:
        if not isinstance(item, dict):
            raise ValueError('Frame manifest records must be objects')
        rel = Path(item.get('image', ''))
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid or missing frame path: {rel}')
        digest = sha256_file(path)
        if item.get('sha256') and digest != item['sha256']:
            raise ValueError(f'Frame checksum mismatch: {rel}')
        group = item.get('source_group') or source.get('source_group')
        if not isinstance(group, str) or not group:
            raise ValueError(f'Missing source_group for {rel}')
        frame_index = item.get('frame_index')
        seconds = item.get('time_seconds')
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or \
                not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid source frame index/time for {rel}')
        records.append((path, {**item, 'sha256': digest, 'source_group': group,
                               'source_frame': frame_index, 'time_seconds': float(seconds)}))
    if not records:
        raise ValueError('Frame manifest contains no records within --max-frames')
    return records, source, root


def extract(args: argparse.Namespace) -> dict:
    values = (args.confidence, args.min_card_short_edge)
    if args.imgsz < 1 or args.max_frames < 1 or args.threads < 1 or \
            not all(math.isfinite(v) for v in values) or not 0 <= args.confidence <= 1 or \
            args.min_card_short_edge < 1:
        raise ValueError('Invalid imgsz/frame limit/confidence/minimum card edge')
    manifest_path = args.frames_manifest.resolve()
    detector_path, output = args.detector.resolve(), args.output.resolve()
    if not manifest_path.is_file():
        raise ValueError(f'Frames manifest does not exist: {manifest_path}')
    if not detector_path.is_file() or detector_path.suffix.lower() != '.pt':
        raise ValueError('--detector must name an existing local .pt weight file; downloads are disabled')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    frames, source, frame_root = _manifest_frames(manifest_path, args.max_frames)
    if output == frame_root or output in frame_root.parents or frame_root in output.parents:
        raise ValueError('Output and frame-manifest directories must not contain one another')
    if output == detector_path or output in detector_path.parents or detector_path in output.parents:
        raise ValueError('Output must not contain or overwrite the detector weights')

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    model = YOLO(str(detector_path))
    if getattr(model, 'task', None) != 'pose':
        raise ValueError(f'Detector task must be pose (card corners), got {getattr(model, "task", None)!r}')
    weight_hash = sha256_file(detector_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'queries').mkdir(exist_ok=True)
    queries, rejected = [], []
    for frame_path, info in frames:
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            rejected.append({'source_frame': info['source_frame'], 'reason': 'decode_failed'})
            continue
        height, width = bgr.shape[:2]
        prediction = model.predict(source=bgr, imgsz=args.imgsz, conf=args.confidence, verbose=False)
        if len(prediction) != 1:
            rejected.append({'source_frame': info['source_frame'], 'reason': 'unexpected_prediction_count'})
            continue
        result = prediction[0]
        if result.keypoints is None or len(result.boxes) == 0:
            continue
        polys = result.keypoints.xy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        for det_index, (poly, score) in enumerate(zip(polys, confidences)):
            try:
                corners, edge_w, edge_h = validate_source_quad(
                    poly, width, height, args.min_card_short_edge)
            except ValueError as exc:
                rejected.append({'source_frame': info['source_frame'], 'detection_index': det_index,
                                 'reason': str(exc)})
                continue
            crop_w, crop_h, resize_mode = crop_dimensions(edge_w, edge_h)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                crop = canonicalize_card(rgb, corners, width=crop_w, height=crop_h)
            except ValueError as exc:
                rejected.append({'source_frame': info['source_frame'], 'detection_index': det_index,
                                 'reason': f'canonical_crop_rejected: {exc}'})
                continue
            relative = Path('queries') / f"frame_{info['source_frame']:09d}_det_{det_index:04d}.png"
            destination = output / relative
            if not cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                raise OSError(f'Failed to write crop: {relative}')
            queries.append({
                'image': relative.as_posix(),
                'sha256': sha256_file(destination),
                'source_group': info['source_group'],
                'source_frame': info['source_frame'],
                'source_frame_image': str(frame_path.relative_to(frame_root).as_posix()),
                'source_frame_sha256': info['sha256'],
                'source_video_sha256': source.get('sha256'),
                'time_seconds': info['time_seconds'],
                'geometry_label_status': 'predicted_pose_corners',
                'identity_label_status': 'unlabeled',
                'prediction_confidence': float(score),
                'detector_weights_sha256': weight_hash,
                'corners_oriented_source_pixels_tl_tr_br_bl': corners.tolist(),
                'source_frame_width': width,
                'source_frame_height': height,
                'effective_source_pixels': {'short_edge': edge_w, 'long_edge': edge_h},
                'crop_pixels': {'width': crop_w, 'height': crop_h},
                'crop_resize': resize_mode,
                'warp_interpolation': 'opencv_INTER_LINEAR',
                'artwork_upright': 'unknown; corner geometry cannot distinguish 0 from 180 degrees',
                'automatic_training_promotion': False,
            })
    report = {
        'schema_version': 1,
        'artifact_type': 'unlabeled_detected_card_crop_candidates',
        'source_frames_manifest': str(manifest_path),
        'source_frames_manifest_sha256': sha256_file(manifest_path),
        'source_group': source.get('source_group'),
        'source_video_sha256': source.get('sha256'),
        'detector_weights_sha256': weight_hash,
        'detector_task': 'pose',
        'settings': {'imgsz': args.imgsz, 'confidence': args.confidence,
                     'max_frames': args.max_frames, 'min_card_short_edge': args.min_card_short_edge,
                     'max_crop_edge': MAX_CROP_EDGE},
        'queries': queries,
        'rejected_detections': rejected,
        'real_camera_validation': False,
        'training_data': False,
    }
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-manifest', required=True, type=Path)
    parser.add_argument('--detector', required=True, type=Path, help='Existing local YOLO-pose corner .pt weights; no download')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--min-card-short-edge', type=float, default=120)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    try:
        report = extract(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps({'frames_manifest': report['source_frames_manifest'],
                      'queries': len(report['queries']),
                      'rejected_detections': len(report['rejected_detections']),
                      'manifest': str(args.output.resolve() / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
