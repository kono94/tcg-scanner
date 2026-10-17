"""Small CPU-only Grounding DINO zero-shot localization probe (not card recognition)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
PROMPTS = ['trading card.', 'a physical card from a trading card game.']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(annotations: list[Path], output: Path, runtime: Path, input_short_edge: int = 800,
        box_threshold: float = 0.20, text_threshold: float = 0.20, max_frames: int = 2,
        model_cache: Path | None = None, single_prompt: bool = False) -> dict:
    if not 1 <= max_frames <= 32:
        raise ValueError('max_frames must be 1..32')
    if not 224 <= input_short_edge <= 1333 or not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError('Invalid input size or detection thresholds')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('HF_HOME', str((output / 'hf_home').resolve()))
    sys.path.insert(0, str(runtime.resolve()))
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(4)
    model_cache = model_cache.resolve() if model_cache else output / 'model_cache'
    model_dir = snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=str(model_cache), token=False,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
    )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, token=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).to('cpu').eval()
    model.requires_grad_(False)

    jobs = []
    for ann_path in annotations:
        data = json.loads(ann_path.read_text())
        for frame in data['instances']:
            physical = [x for x in frame['instances'] if x.get('physical_card') is True and x.get('overlay_graphic') is not True]
            overlays = [x for x in frame['instances'] if x.get('overlay_graphic') is True or x.get('physical_card') is False]
            jobs.append((ann_path, frame, physical, overlays))
    jobs = jobs[:max_frames]
    results = []
    for ann_path, frame, physical, overlays in jobs:
        image_path = (ROOT / frame['image']).resolve()
        if ROOT.resolve() not in image_path.parents:
            raise ValueError('Frame path escapes repository root')
        if sha256(image_path) != frame.get('image_sha256'):
            raise ValueError('Source frame checksum differs from input annotation manifest')
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        result = {'source_image': str(image_path.relative_to(ROOT)), 'source_sha256': sha256(image_path),
                  'source_width': width, 'source_height': height, 'teacher_annotations': str(ann_path),
                  'teacher_sha256': sha256(ann_path), 'teacher_physical_proposals': physical,
                  'teacher_excluded_overlays': overlays, 'detections_by_prompt': []}
        selected_prompts = PROMPTS[:1] if single_prompt else PROMPTS
        for prompt in selected_prompts:
            t0 = time.perf_counter()
            inputs = processor(images=image, text=prompt, return_tensors='pt', size={'shortest_edge': input_short_edge, 'longest_edge': 1333})
            with torch.inference_mode():
                outputs = model(**inputs)
                post = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[(height, width)],
                )[0]
            elapsed = time.perf_counter() - t0
            boxes = post['boxes'].cpu().numpy()
            scores = post['scores'].cpu().numpy()
            labels = post['labels']
            found = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [float(v) for v in box]
                found.append({'bbox_xyxy_source_pixels': [x1, y1, x2, y2], 'score': float(score), 'text_label': str(label)})
            def iou(a, b):
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                intersection = ix * iy
                area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                return intersection / max(1e-9, area_a + area_b - intersection)
            pairs = sorted(((iou(t['bbox_xyxy'], d['bbox_xyxy_source_pixels']), ti, di)
                            for ti, t in enumerate(physical) for di, d in enumerate(found)), reverse=True)
            used_t, used_d, matched_50 = set(), set(), 0
            for score, ti, di in pairs:
                if score < 0.5 or ti in used_t or di in used_d:
                    continue
                used_t.add(ti); used_d.add(di); matched_50 += 1
            overlay_iou = max((iou(o['bbox_xyxy'], d['bbox_xyxy_source_pixels'])
                               for o in overlays for d in found), default=0.0)
            result['detections_by_prompt'].append({'prompt': prompt, 'box_threshold': box_threshold,
                'text_threshold': text_threshold, 'preprocess_resize': {'shortest_edge': input_short_edge, 'longest_edge': 1333},
                'elapsed_seconds_cpu_preprocess_forward_postprocess': elapsed, 'detections': found,
                'teacher_bbox_comparison_not_ground_truth': {
                    'one_to_one_physical_teacher_proposals_iou_at_least_0_5': matched_50,
                    'physical_teacher_proposal_count': len(physical),
                    'max_excluded_overlay_bbox_iou': overlay_iou,
                }})
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        for instance in physical:
            box = instance['bbox_xyxy']
            draw.rectangle(tuple(box), outline=(30, 220, 70), width=6)
            draw.text((box[0], max(0, box[1] - 20)), f"teacher:{instance['id']}", fill=(30, 220, 70))
        for instance in overlays:
            box = instance['bbox_xyxy']
            draw.rectangle(tuple(box), outline=(240, 40, 40), width=6)
            draw.text((box[0], max(0, box[1] - 20)), f"excluded:{instance['id']}", fill=(240, 40, 40))
        colors = [(255, 190, 0), (20, 210, 240)]
        for prompt_result, color in zip(result['detections_by_prompt'], colors):
            for index, det in enumerate(prompt_result['detections']):
                box = [round(v) for v in det['bbox_xyxy_source_pixels']]
                draw.rectangle(tuple(box), outline=color, width=4)
                draw.text((box[0], box[1]), f"{prompt_result['prompt'][:12]} {det['score']:.2f}", fill=color)
        source_key = hashlib.sha256(str(image_path.relative_to(ROOT)).encode()).hexdigest()[:12]
        preview = output / f"{source_key}_{image_path.stem}_grounding_dino.jpg"
        vis.save(preview, quality=90)
        result['preview'] = str(preview.relative_to(output))
        results.append(result)
    weights_path = Path(model_dir) / 'model.safetensors'
    manifest = {'schema_version': 1, 'artifact_type': 'zero_shot_localization_proposals',
        'model': MODEL_ID, 'revision': MODEL_REVISION, 'weight_sha256': sha256(weights_path),
        'transformers_version': __import__('transformers').__version__, 'torch_version': torch.__version__,
        'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'input_short_edge': input_short_edge,
        'box_threshold': box_threshold, 'text_threshold': text_threshold, 'prompts': PROMPTS[:1] if single_prompt else PROMPTS,
        'model_cache': str(model_cache),
        'ground_truth_verified': False, 'identity_claims': False, 'corner_claims': False,
        'training_promotion': 'not_promoted', 'results': results}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--input-short-edge', type=int, default=800)
    p.add_argument('--box-threshold', type=float, default=.20)
    p.add_argument('--text-threshold', type=float, default=.20)
    p.add_argument('--max-frames', type=int, default=2)
    p.add_argument('--model-cache', type=Path, help='Reuse the public-weight cache across output directories')
    p.add_argument('--single-prompt', action='store_true', help='Use only the first prompt for bounded frame audits')
    args = p.parse_args()
    manifest = run(args.annotations, args.output, args.runtime, args.input_short_edge,
                   args.box_threshold, args.text_threshold, args.max_frames, args.model_cache, args.single_prompt)
    print(json.dumps({'frames': len(manifest['results']), 'output': str(args.output), 'weight_sha256': manifest['weight_sha256']}))


if __name__ == '__main__':
    main()
