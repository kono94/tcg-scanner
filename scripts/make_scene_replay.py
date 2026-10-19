#!/usr/bin/env python3
"""Encode synthetic validation scenes as a camera-mock replay video with per-frame true corners.

Input is a `scripts/generate_detection_scenes.py` output. Each selected scene shakes, then holds
still, so the browser's stability check can arm a capture. Synthetic pixels with renderer-known
corners: a pipeline/latency test input, not a camera benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# Card-bearing scene kinds in replay order; a binder page with at least nine cards comes first.
KINDS = [('binder', 9), ('loose', 1), ('held', 1), ('binder', 2)]


def select_scenes(scenes: Path, width: int, height: int) -> list[dict]:
    metas = [json.loads(p.read_text()) for p in sorted((scenes / 'annotations/val').glob('*.json'))]
    metas = [m for m in metas if (m['width'], m['height']) == (width, height)]
    chosen: list[dict] = []
    for kind, minimum in KINDS:
        match = next((m for m in metas if m['kind'] == kind and len(m['instances']) >= minimum and m not in chosen), None)
        if match is not None:
            chosen.append(match)
    if not chosen:
        raise SystemExit(f'No card-bearing {width}x{height} validation scenes in {scenes}')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, default=ROOT / 'runs/card_detector/scenes-v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/card_detector/replay-v3')
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 768])
    parser.add_argument('--seconds-per-scene', type=float, default=4.0)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    width, height = args.size
    selection = select_scenes(args.scenes, width, height)
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'synthetic_replay.mp4'
    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', str(video)]
    per_scene = round(args.seconds_per_scene * args.fps)
    frames = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for segment, meta in enumerate(selection):
                source = np.asarray(Image.open(args.scenes / meta['image']).convert('RGB'))
                for k in range(per_scene):
                    local = k / per_scene
                    stable = local >= .35
                    amplitude = 0 if stable else 1 - local / .35
                    dx = amplitude * math.sin(local * 40) * width * .018
                    dy = amplitude * math.cos(local * 35) * height * .016
                    transform = np.float64([[1, 0, dx], [0, 1, dy]])
                    image = cv2.warpAffine(source, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                    process.stdin.write(np.ascontiguousarray(image).tobytes())
                    frames.append({'frame_index': len(frames), 'time_seconds': len(frames) / args.fps, 'scene_index': segment,
                                   'expected_stable_phase': stable,
                                   'true_corners': [(np.asarray(i['true_corners']) + [dx, dy]).round(2).tolist() for i in meta['instances']]})
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise SystemExit('ffmpeg failed to encode the replay')
    manifest = {'video': video.name, 'fps': args.fps, 'width': width, 'height': height,
                'label_origin': 'synthetic_renderer', 'real_camera_validation': False,
                'source_scenes': [{'image': str((args.scenes / m['image']).relative_to(ROOT)) if (args.scenes / m['image']).is_relative_to(ROOT) else m['image'],
                                   'kind': m['kind'], 'cards': len(m['instances'])} for m in selection],
                'frames': frames}
    (args.output / 'synthetic_replay.json').write_text(json.dumps(manifest) + '\n')
    print(json.dumps({'video': str(video), 'frames': len(frames), 'scenes': manifest['source_scenes']}, indent=2))


if __name__ == '__main__':
    main()
