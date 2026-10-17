#!/usr/bin/env python3
"""Validate selected-frame manifests and collect unlabeled localization inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.foundation_retrieval import sha256_file as file_sha256


def collect(manifests: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValueError('Output manifest must be new')
    frames, duplicates, seen = [], [], {}
    for manifest_path in manifests:
        manifest_path = manifest_path.resolve()
        data = json.loads(manifest_path.read_text())
        source = data.get('source', {})
        records = data.get('records', data.get('frames', []))
        manifest_hash = file_sha256(manifest_path)
        for record in records:
            relative = record.get('image') or record.get('path')
            if not isinstance(relative, str):
                raise ValueError(f'Frame path missing: {manifest_path}')
            candidates = [(manifest_path.parent / relative).resolve(), (ROOT / relative).resolve()]
            matches = {p for p in candidates if p.is_relative_to(ROOT) and p.is_file()}
            if len(matches) != 1:
                raise ValueError(f'Frame path missing or ambiguous: {relative}')
            image = matches.pop()
            digest = file_sha256(image)
            if digest != record.get('sha256'):
                raise ValueError(f'Frame checksum mismatch: {image}')
            with Image.open(image) as opened:
                width, height = opened.size
            if min(width, height) < 1440:
                raise ValueError(f'Frame is below minimum source-pixel resolution: {image}')
            if [width, height] != [record.get('width'), record.get('height')]:
                raise ValueError(f'Frame dimensions disagree with manifest: {image}')
            group = record.get('source_group') or source.get('source_group')
            if not group:
                raise ValueError(f'Source group missing: {image}')
            frame = {'image': str(image.relative_to(ROOT)), 'image_sha256': digest,
                'source_group': group, 'source_url': source.get('source_url') or source.get('url'),
                'frame_index': record.get('frame_index'),
                'excerpt_local_timestamp_seconds': record.get('time_seconds'),
                'timestamp_policy': data.get('timestamp_policy', 'unknown; do not infer exact source time'),
                'source_manifest': str(manifest_path), 'source_manifest_sha256': manifest_hash,
                'width': width, 'height': height, 'instances': []}
            if digest in seen:
                duplicates.append({'duplicate': frame, 'retained_image': seen[digest]})
                continue
            seen[digest] = frame['image']
            frames.append(frame)
    if not frames:
        raise ValueError('No eligible selected frames')
    result = {'schema_version': 1, 'annotation_type': 'unlabeled_selected_video_frames',
        'ground_truth_verified': False, 'training_promotion': 'not_promoted',
        'selected_frames': len(frames), 'source_groups': sorted({f['source_group'] for f in frames}),
        'exact_duplicate_frames_removed': duplicates,
        'near_duplicate_audit': 'not established by exact hashes', 'instances': frames}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.manifest, args.output)
    print(json.dumps({'frames': result['selected_frames'], 'source_groups': result['source_groups']}))
