#!/usr/bin/env python3
"""Propose local reference matches from observed names; never approve labels.

Input observations carry crop/source paths, source bbox and an observed name.
Names only narrow the search. Shared layout/text can produce false feature
matches; a visual reference check remains necessary before curation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_card_geometry import source_resolution_crop, verify_homography


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def inside_file(repo, rel):
    path = (repo / rel).resolve(strict=True)
    if not path.is_relative_to(repo) or not path.is_file():
        raise ValueError(f'Input must be a repository file: {rel}')
    return path


def propose(repo, observations, references, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rectified').mkdir(exist_ok=True)
    by_name = {}
    for r in references:
        by_name.setdefault((r['game'], r.get('name', '').casefold()), []).append(r)
    rows = []
    cv2.setNumThreads(4)
    for index, obs in enumerate(observations):
        query_path = inside_file(repo, obs['crop_path'])
        if sha(query_path) != obs['crop_sha256']:
            raise ValueError('Query hash mismatch')
        query = rgb(query_path)
        candidates = by_name.get((obs['game'], obs['observed_name'].casefold()), [])
        ranked = []
        for r in candidates:
            reference_path = inside_file(repo, r['path'])
            if sha(reference_path) != r['sha256']:
                raise ValueError('Reference hash mismatch')
            metrics = verify_homography(rgb(reference_path), query)
            # Spatial support outranks a pile of degenerate common-text matches.
            score = (metrics['accepted'], metrics['inlier_count'] * min(1, metrics['reference_coverage'] / .35)
                     if (metrics['reprojection_error_px'] or 100) < 4 else 0)
            ranked.append((score, r, metrics))
        ranked.sort(key=lambda x: x[0], reverse=True)
        row = dict(obs, index=index, status='PROPOSED_ONLY', candidates=[])
        for rank, (_, ref, metrics) in enumerate(ranked[:3]):
            candidate = dict(reference=ref, geometry=metrics)
            quad = metrics['quad_xy']
            tolerable = set(metrics['rejection_reasons']) <= {'mapped_quad_out_of_teacher_crop'}
            if quad is not None and tolerable:
                source = inside_file(repo, obs['source_frame_image'])
                if sha(source) != obs['source_frame_sha256']:
                    raise ValueError('Source frame hash mismatch')
                source_quad = np.asarray(quad, np.float32) + np.asarray(obs['bbox'][:2], np.float32)
                try:
                    warped = source_resolution_crop(rgb(source), source_quad, aspect_ratio=224/312)
                except ValueError:
                    pass
                else:
                    dest = output / 'rectified' / f'{index:03d}-{rank}-{ref["card_id"]}.png'
                    Image.fromarray(warped).save(dest)
                    candidate.update(rectified_path=dest.relative_to(repo).as_posix(),
                                     rectified_sha256=sha(dest), quad_source_xy=source_quad.tolist())
            row['candidates'].append(candidate)
        rows.append(row)
        print(f'{index + 1}/{len(observations)} {obs["observed_name"]}: {len(candidates)} references', flush=True)
    for start in range(0, len(rows), 12):
        subset = rows[start:start + 12]
        sheet = Image.new('RGB', (1440, ((len(subset)+3)//4)*350), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(subset):
            x, y = (n % 4)*360, (n // 4)*350
            candidate = row['candidates'][0] if row['candidates'] else None
            path = candidate.get('rectified_path', row['crop_path']) if candidate else row['crop_path']
            im = Image.open(repo / path).convert('RGB'); im.thumbnail((174, 300)); sheet.paste(im, (x, y))
            if candidate:
                im = Image.open(repo / candidate['reference']['path']).convert('RGB')
                im.thumbnail((174, 300)); sheet.paste(im, (x+180, y))
            label = candidate['reference']['card_id'] if candidate else 'NO REFERENCE'
            draw.multiline_text((x, y+304), f'{row["index"]}: {row["observed_name"]}\n{label}\nPROPOSED; visual review required', fill='black')
        sheet.save(output / f'paired-{start//12+1:02d}.jpg', quality=95)
    result = dict(schema_version=1, status='PROPOSED_ONLY', proposals=rows)
    (output / 'proposals.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    propose(ROOT, json.loads(args.observations.read_text())['observations'],
            json.loads(args.references.read_text())['references'], args.output.resolve())


if __name__ == '__main__':
    main()
