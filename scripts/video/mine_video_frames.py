#!/usr/bin/env python3
"""Mine diverse candidate frames from a local video; does not create card labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def thumbnail(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA)


def difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255)


def mine(args: argparse.Namespace) -> dict:
    video, output = args.video.resolve(), args.output.resolve()
    values = [args.sample_seconds, args.window_seconds, args.start, args.min_sharpness, args.dedup_difference]
    if not all(math.isfinite(x) for x in values) or args.sample_seconds <= 0 or args.window_seconds <= 0 or args.start < 0:
        raise ValueError('Sampling/window intervals must be positive and start nonnegative; settings must be finite')
    if args.max_frames < 1 or args.max_samples < 1 or args.min_sharpness < 0 or not 0 <= args.dedup_difference <= 1 or args.min_source_height < 0:
        raise ValueError('Invalid frame/sample limits, sharpness or deduplication difference')
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        raise ValueError('end must be finite and later than start')
    if not video.is_file() or output == video or output in video.parents:
        raise ValueError('Input must be a local video file outside the output directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot open video with a valid frame rate')
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if min(source_width, source_height) < args.min_source_height:
            raise ValueError(f'Source short edge {min(source_width, source_height)} is below required {args.min_source_height}; select a higher-resolution original')
        source_hash = sha256_file(video)
        group = args.source_group or source_hash
        output.mkdir(parents=True, exist_ok=True)
        (output / 'frames').mkdir(exist_ok=True)
        records, accepted_thumbs = [], []
        previous, candidate, window = None, None, None
        sampled = rejected_duplicates = rejected_quality = 0

        def save_candidate() -> None:
            nonlocal rejected_duplicates
            if candidate is None or len(records) >= args.max_frames:
                return
            frame, small, metrics = candidate
            # Whole-frame heuristic only: this is not a card identity/legibility check.
            if any(difference(small, other) <= args.dedup_difference for other in accepted_thumbs):
                rejected_duplicates += 1
                return
            path = Path('frames') / f"frame_{metrics['frame_index']:09d}.jpg"
            if not cv2.imwrite(str(output / path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'Cannot save {path}')
            records.append({**metrics, 'image': path.as_posix(), 'width': frame.shape[1], 'height': frame.shape[0],
                            'sha256': sha256_file(output / path), 'source_group': group,
                            'identity_label_status': 'unlabeled', 'geometry_label_status': 'unlabeled',
                            'contains_burned_in_overlays': args.contains_overlays})
            accepted_thumbs.append(small)

        # Decode sequentially rather than repeatedly seeking compressed video keyframes.
        index = 0
        next_sample = args.start
        while sampled < args.max_samples and len(records) < args.max_frames:
            timestamp = index / fps
            if args.end is not None and timestamp >= args.end:
                break
            if not capture.grab():
                break
            frame_index = index
            index += 1
            if timestamp + 1e-9 < next_sample:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            next_sample = timestamp + args.sample_seconds
            sampled += 1
            small = thumbnail(frame)
            motion = difference(small, previous) if previous is not None else None
            previous = small
            sharpness = float(cv2.Laplacian(small, cv2.CV_32F).var())
            key = int((timestamp - args.start) / args.window_seconds)
            if key != window:
                save_candidate()
                candidate, window = None, key
            if sharpness < args.min_sharpness:
                rejected_quality += 1
                continue
            # A high-contrast graphic can score well; subsequent card filtering is required.
            score = math.log1p(sharpness) / (1 + 50 * (motion if motion is not None else 1))
            metrics = {'frame_index': frame_index, 'time_seconds': timestamp,
                       'sharpness_thumbnail_laplacian_variance': sharpness,
                       'motion_thumbnail_mean_absolute_difference': motion, 'selection_score': score}
            if candidate is None or score > candidate[2]['selection_score']:
                candidate = (frame, small, metrics)
        save_candidate()
        report = {'schema_version': 1, 'artifact_type': 'unlabeled_video_frame_candidates',
                  'source': {'path': str(video), 'sha256': source_hash, 'source_url': args.source_url,
                             'source_group': group, 'reuse_basis': args.reuse_basis, 'reported_fps': fps,
                             'width': source_width, 'height': source_height},
                  'settings': {k: v for k, v in vars(args).items() if k not in {'video', 'output'}},
                  'sampled_frames': sampled, 'saved_frames': len(records),
                  'rejected_low_sharpness_samples': rejected_quality, 'rejected_duplicate_candidates': rejected_duplicates,
                  'timestamp_policy': 'frame_index / reported_fps; constant-frame-rate inputs only',
                  'real_camera_validation': False, 'records': records}
        (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-url', help='Provenance only; this tool does not download URLs')
    parser.add_argument('--source-group', help='Keep all videos from the same match/session in one data split')
    parser.add_argument('--reuse-basis', default='unspecified')
    parser.add_argument('--contains-overlays', choices=['unknown', 'yes', 'no'], default='unknown')
    parser.add_argument('--sample-seconds', type=float, default=.5)
    parser.add_argument('--window-seconds', type=float, default=2)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--max-samples', type=int, default=3600)
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--min-sharpness', type=float, default=10)
    parser.add_argument('--min-source-short-edge', '--min-source-height', dest='min_source_height', type=int, default=1440,
                        help='Actual decoded short edge (handles portrait too); lower only for explicit local test fixtures')
    parser.add_argument('--dedup-difference', type=float, default=.005)
    args = parser.parse_args()
    try:
        report = mine(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps({k: report[k] for k in ['sampled_frames', 'saved_frames', 'rejected_low_sharpness_samples', 'rejected_duplicate_candidates']}, indent=2))


if __name__ == '__main__':
    main()
