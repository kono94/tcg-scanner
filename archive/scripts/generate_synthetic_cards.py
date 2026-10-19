#!/usr/bin/env python3
"""Seeded card compositing with exact identities/quads and derived YOLO OBB labels.

The pixels are synthetic bootstrap data. Their known geometry is ground truth for
these renders only, never evidence of accuracy on a real phone camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_detector.synthetic_effects import apply_camera_effects, make_background


@dataclass(frozen=True)
class Reference:
    path: Path
    card_id: str
    sha256: str


def discover_references(cards_dir: Path) -> list[Reference]:
    records = []
    for path in sorted(cards_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        parts = path.relative_to(cards_dir).parts
        if len(parts) < 2:
            continue
        card_id = parts[1] if len(parts) >= 3 else path.stem.split('__', 1)[0]
        data = path.read_bytes()
        if data.startswith(b'version https://git-lfs'):
            raise ValueError(f'Unhydrated Git LFS image: {path}')
        with Image.open(path) as image:
            image.verify()
        records.append(Reference(path.resolve(), card_id, hashlib.sha256(data).hexdigest()))
    if len(records) < 2:
        raise ValueError('Need at least two distinct reference identities for disjoint train/val sources')
    return records


def partition_references(records: list[Reference], seed: int, val_fraction: float) -> dict[str, list[Reference]]:
    """Keep all views of an identity AND duplicate source hashes in one split."""
    parents = {r.card_id: r.card_id for r in records}

    def group(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    by_hash = {}
    for record in records:
        if record.sha256 in by_hash:
            a, b = group(record.card_id), group(by_hash[record.sha256])
            parents[max(a, b)] = min(a, b)
        by_hash[record.sha256] = record.card_id
    groups = {}
    for record in records:
        groups.setdefault(group(record.card_id), []).append(record)
    if len(groups) < 2:
        raise ValueError('Cannot create independent source splits from a single identity/hash group')
    keys = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    n_val = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val])
    out = {'train': [], 'val': []}
    for key in keys:
        out['val' if key in val_keys else 'train'].extend(groups[key])
    return out


def quad_to_obb(quad: np.ndarray) -> np.ndarray:
    """Minimum-area enclosing rotated RECTANGLE, distinct from the true quad."""
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or abs(cv2.contourArea(quad)) < 1:
        raise ValueError('Expected a finite nondegenerate 4x2 card quadrilateral')
    return cv2.boxPoints(cv2.minAreaRect(quad)).astype(np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[None], transform)[0]


def render_card(scene: np.ndarray, reference: Reference, quad: np.ndarray) -> None:
    with Image.open(reference.path) as image:
        card = np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize((240, 336))).copy()
    source = np.float32([[0, 0], [239, 0], [239, 335], [0, 335]])
    transform = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    h, w = scene.shape[:2]
    # Rounded cutout, with labels at intersections of the straight physical edges.
    alpha_image = Image.new('L', (240, 336))
    ImageDraw.Draw(alpha_image).rounded_rectangle((0, 0, 239, 335), radius=7, fill=255)
    alpha = cv2.warpPerspective(np.asarray(alpha_image), transform, (w, h), flags=cv2.INTER_LINEAR) / 255.0
    pixels = cv2.warpPerspective(card, transform, (w, h), flags=cv2.INTER_LINEAR)
    offset = np.float32([[1, 0, 3], [0, 1, 5]])
    shadow = cv2.warpAffine(alpha.astype(np.float32), offset, (w, h))
    shadow = cv2.GaussianBlur(shadow, (0, 0), 3)
    base = scene.astype(np.float32) * (1 - .23 * shadow[..., None])
    scene[:] = np.clip(pixels * alpha[..., None] + base * (1 - alpha[..., None]), 0, 255).astype(np.uint8)


def render_scene(
    references: list[Reference], seed: int, width: int, height: int,
    count: int, style: str | None = None,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    style = style or str(rng.choice(['binder', 'desk', 'plain'], p=[.55, .35, .10]))
    if style == 'binder':
        columns = int(rng.choice([2, 3, 4]))
        rows = max(2, math.ceil((count + int(rng.integers(0, columns + 1))) / columns))
    else:
        columns = max(1, math.ceil(math.sqrt(max(1, count) * width / height * .72)))
        rows = max(1, math.ceil(max(1, count) / columns))
    margin_x, margin_y = width * .10, height * .10
    cell_w, cell_h = (width - 2 * margin_x) / columns, (height - 2 * margin_y) / rows
    pockets = []
    for row in range(rows):
        for col in range(columns):
            x, y = margin_x + col * cell_w, margin_y + row * cell_h
            pocket_h = min(cell_h * .95, cell_w * .95 / .7143)
            pocket_w = pocket_h * .7143
            cx, cy = x + cell_w / 2, y + cell_h / 2
            pockets.append((int(cx - pocket_w / 2), int(cy - pocket_h / 2),
                            int(cx + pocket_w / 2), int(cy + pocket_h / 2)))
    scene = make_background(width, height, rng, style, pockets if style == 'binder' else ())
    slots = rng.choice(len(pockets), size=count, replace=False).tolist()
    instances = []
    duplicate_probability = .16
    for index, slot in enumerate(slots):
        reference = (references[int(rng.integers(len(references)))] if not instances or rng.random() > duplicate_probability
                     else next(r for r in references if str(r.path) == instances[-1]['source_path']))
        x0, y0, x1, y1 = pockets[slot]
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float64)
        scale = rng.uniform(.72, .91) if style == 'binder' else rng.uniform(.52, .70)
        card_h = min(y1 - y0, (x1 - x0) / .7143) * scale
        card_w = card_h * .7143
        local = np.array([[-card_w/2, -card_h/2], [card_w/2, -card_h/2],
                          [card_w/2, card_h/2], [-card_w/2, card_h/2]])
        angle = float(rng.uniform(-5, 5) if style == 'binder' else rng.uniform(-40, 40))
        if rng.random() < .25:
            angle += 180
        radians = math.radians(angle)
        rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
        # Perspective variation is known from the renderer, not guessed by a VLM.
        distortion = float(rng.uniform(0, .055))
        quad = (local + rng.uniform(-1, 1, (4, 2)) * [card_w * distortion, card_h * distortion]) @ rotation.T
        quad += center + rng.uniform(-.035, .035, 2) * [cell_w, cell_h]
        if np.any(quad < 1) or np.any(quad[:, 0] > width - 2) or np.any(quad[:, 1] > height - 2):
            # Reduce extent around the intended center, preserving corner semantics.
            quad = center + (quad - center) * .8
        render_card(scene, reference, quad)
        instances.append({
            'instance_id': f'card-{index:03d}', 'card_id': reference.card_id,
            'source_path': str(reference.path), 'source_sha256': reference.sha256,
            'true_corners': quad.tolist(), 'in_plane_rotation_degrees': angle,
            'identity_label_source': 'composited_reference',
        })
    # Apply one mild camera homography to the full scene and every physical corner.
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    inset = min(width, height) * .025
    target = source * .94 + [width * .03, height * .03]
    target += rng.uniform(-inset, inset, (4, 2))
    global_transform = cv2.getPerspectiveTransform(source, target.astype(np.float32))
    border = tuple(int(x) for x in rng.integers(55, 120, 3))
    scene = cv2.warpPerspective(scene, global_transform, (width, height), borderValue=border)
    for instance in instances:
        quad = transform_points(np.asarray(instance['true_corners']), global_transform)
        obb = quad_to_obb(quad)
        if np.any(obb < 0) or np.any(obb[:, 0] >= width) or np.any(obb[:, 1] >= height):
            raise ValueError('Generated OBB outside image; increase image margin')
        instance['true_corners'] = quad.tolist()
        instance['obb_corners'] = obb.tolist()
    scene, effects = apply_camera_effects(scene, rng)
    metadata = {
        'seed': seed, 'width': width, 'height': height, 'style': style,
        'rows': rows, 'columns': columns, 'count': len(instances),
        'coordinate_space': 'oriented_source_pixels_top_left',
        'corner_order': 'source_artwork_TL_TR_BR_BL_mapped_into_scene',
        'label_origin': 'synthetic_renderer', 'real_camera_validation': False,
        'scene_homography': global_transform.tolist(), 'effects': effects,
        'instances': instances,
    }
    return scene, metadata


def write_obb_labels(path: Path, metadata: dict) -> None:
    lines = []
    size = np.array([metadata['width'], metadata['height']], dtype=np.float64)
    for instance in metadata['instances']:
        coords = np.asarray(instance['obb_corners']) / size
        if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords > 1):
            raise ValueError('Invalid normalized OBB label')
        lines.append('0 ' + ' '.join(f'{x:.8f}' for x in coords.ravel()))
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def save_preview(records: list[dict], root: Path) -> None:
    columns, thumb_w, thumb_h = 4, 280, 245
    sheet = Image.new('RGB', (columns * thumb_w, math.ceil(len(records) / columns) * thumb_h), '#202622')
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        with Image.open(root / record['image']) as opened:
            image = opened.convert('RGB')
        painter = ImageDraw.Draw(image)
        for instance in record['metadata']['instances']:
            quad = [tuple(x) for x in instance['true_corners']]
            obb = [tuple(x) for x in instance['obb_corners']]
            painter.line(obb + [obb[0]], fill='#ffb84f', width=2)
            painter.line(quad + [quad[0]], fill='#6fffbb', width=2)
        image.thumbnail((thumb_w - 12, thumb_h - 32))
        x, y = i % columns * thumb_w, i // columns * thumb_h
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 4))
        draw.text((x + 8, y + thumb_h - 24), f"{record['split']} | {record['metadata']['style']} | {record['metadata']['count']} cards", fill='white')
    sheet.save(root / 'preview.jpg', quality=92)


def generate_replay(records: list[dict], root: Path, seconds: float, fps: int) -> dict:
    """Synthetic moving/steady phases, clean MP4 plus exact per-frame labels."""
    selection = [r for r in records if r['split'] == 'val'][:4]
    frame_count = max(len(selection), round(seconds * fps))
    width, height = selection[0]['metadata']['width'], selection[0]['metadata']['height']
    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', str(root / 'synthetic_replay.mp4')]
    source_images = [np.asarray(Image.open(root / r['image']).convert('RGB')) for r in selection]
    frames = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for i in range(frame_count):
                segment = min(len(selection) - 1, i * len(selection) // frame_count)
                segment_start = math.ceil(segment * frame_count / len(selection))
                segment_length = math.ceil((segment + 1) * frame_count / len(selection)) - segment_start
                local = (i - segment_start) / max(segment_length, 1)
                stable = local >= .35
                amplitude = (1 - local / .35) if not stable else 0
                dx = amplitude * math.sin(local * 40) * width * .018
                dy = amplitude * math.cos(local * 35) * height * .016
                transform = np.float64([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
                image = cv2.warpPerspective(source_images[segment], transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                process.stdin.write(image.tobytes())
                instances = [{
                    'instance_id': inst['instance_id'], 'card_id': inst['card_id'],
                    'true_corners': transform_points(np.asarray(inst['true_corners']), transform).tolist(),
                } for inst in selection[segment]['metadata']['instances']]
                frames.append({'frame_index': i, 'time_seconds': i / fps, 'scene_index': segment,
                               'expected_stable_phase': stable, 'instances': instances})
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed to encode the synthetic replay')
    manifest = {'video': 'synthetic_replay.mp4', 'fps': fps, 'width': width, 'height': height,
                'label_origin': 'synthetic_renderer', 'real_camera_validation': False,
                'source_scenes': [r['image'] for r in selection], 'frames': frames}
    write_json(root / 'synthetic_replay.json', manifest)
    return {'video': 'synthetic_replay.mp4', 'annotations': 'synthetic_replay.json', 'frame_count': frame_count}


def generate(args: argparse.Namespace) -> dict:
    if args.scenes < 4 or not 0 <= args.max_cards <= 100 or min(args.width, args.height) < 256:
        raise ValueError('Need >=4 scenes, max-cards in [0,100], and dimensions >=256')
    if args.width % 2 or args.height % 2 or not .05 <= args.val_fraction <= .5:
        raise ValueError('Dimensions must be even; val-fraction must be in [.05,.5]')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output: {root}')
    source = args.cards_dir.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('Output and source directories must not contain one another')
    references = discover_references(source)
    pools = partition_references(references, args.seed, args.val_fraction)
    root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        for kind in ['images', 'labels', 'annotations', 'recognizer_crops']:
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    n_val = max(2, round(args.scenes * args.val_fraction))
    records, crop_records = [], []
    # Import the SAME canonicalizer used by gallery inserts; no stochastic crop transforms.
    from training.card_recognizer.canonical import canonicalize_card, preprocessing_contract, preprocessing_contract_hash
    for index in range(args.scenes):
        split = 'val' if index < n_val else 'train'
        ordinal = index if split == 'val' else index - n_val
        seed = args.seed + index * 1009
        rng = np.random.default_rng(seed)
        count = int(rng.integers(args.max_cards + 1))
        if ordinal % 8 == 0:
            count = 0
        elif ordinal % 8 == 1:
            count = min(1, args.max_cards)
        elif ordinal % 8 == 2:
            count = min(9, args.max_cards)
        elif ordinal % 8 == 3:
            count = args.max_cards
        scene, metadata = render_scene(pools[split], seed, args.width, args.height, count)
        name = f'scene_{index:06d}'
        image_path = Path('images') / split / f'{name}.jpg'
        Image.fromarray(scene).save(root / image_path, quality=94)
        # Crop from the persisted JPEG, so query pixels match the actual dataset artifact.
        persisted_scene = np.asarray(Image.open(root / image_path).convert('RGB'))
        metadata.update({'split': split, 'image': image_path.as_posix(),
                         'preprocessing_contract_sha256': preprocessing_contract_hash()})
        write_obb_labels(root / 'labels' / split / f'{name}.txt', metadata)
        write_json(root / 'annotations' / split / f'{name}.json', metadata)
        for instance in metadata['instances']:
            crop = canonicalize_card(persisted_scene, np.asarray(instance['true_corners']))
            crop_path = Path('recognizer_crops') / split / f"{name}__{instance['instance_id']}.png"
            Image.fromarray(crop).save(root / crop_path)
            crop_records.append({'image': crop_path.as_posix(), 'card_id': instance['card_id'],
                                 'scene': image_path.as_posix(), 'split': split,
                                 'source_sha256': instance['source_sha256'], 'geometry': 'renderer_true_corners'})
        records.append({'split': split, 'image': image_path.as_posix(), 'metadata': metadata})
        if index % 25 == 0 or index + 1 == args.scenes:
            print(f'Generated {index + 1}/{args.scenes} scenes', flush=True)
    # JSON string is valid YAML quoting, including paths containing spaces.
    (root / 'dataset.yaml').write_text(f'path: {json.dumps(str(root))}\ntrain: images/train\nval: images/val\nnames:\n  0: card\n', encoding='utf-8')
    write_json(root / 'recognizer_manifest.json', {'label_origin': 'synthetic_renderer',
               'preprocessing': preprocessing_contract(), 'preprocessing_contract_sha256': preprocessing_contract_hash(),
               'queries': crop_records})
    sources = {split: [{'path': str(r.path), 'card_id': r.card_id, 'sha256': r.sha256} for r in pool]
               for split, pool in pools.items()}
    write_json(root / 'source_manifest.json', {'seed': args.seed, 'split_policy': 'disjoint_identity_and_source_hash', 'sources': sources})
    preview = records[:4] + records[n_val:n_val + 8]
    save_preview(preview, root)
    summary = {'version': 1, 'seed': args.seed, 'scenes': len(records), 'instances': len(crop_records),
               'counts': {str(n): sum(r['metadata']['count'] == n for r in records) for n in sorted({r['metadata']['count'] for r in records})},
               'splits': {s: sum(r['split'] == s for r in records) for s in pools},
               'geometry': 'true_perspective_quads_and_derived_minimum_area_rectangles',
               'real_camera_validation': False, 'dataset': 'dataset.yaml', 'preview': 'preview.jpg'}
    if args.video_seconds > 0:
        summary['replay'] = generate_replay(records, root, args.video_seconds, args.fps)
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards-dir', type=Path, default=ROOT / 'datasets/card_recognizer/cards')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/card_detector/synthetic')
    parser.add_argument('--scenes', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--max-cards', type=int, default=14)
    parser.add_argument('--val-fraction', type=float, default=.2)
    parser.add_argument('--video-seconds', type=float, default=0, help='Optional clean synthetic replay with exact per-frame annotations')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1 or not math.isfinite(args.video_seconds) or args.video_seconds < 0:
        parser.error('fps must be positive and video-seconds finite/nonnegative')
    try:
        generate(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
