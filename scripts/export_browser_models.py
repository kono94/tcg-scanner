#!/usr/bin/env python3
"""Export the retrieval encoder, pose/OBB detector and gallery with runtime parity fixtures.

Parity compares real outputs: PyTorch CPU vs onnxruntime on actual crops/scenes, and the
stored training-run gallery vectors vs re-encoded references. Fixtures use validation
queries and synthetic validation scenes only; test pixels are not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input, preprocessing_contract_hash
from training.card_recognizer.foundation_retrieval import load_dinov2, sha256_file
from training.card_recognizer.retrieval_training import CardEmbedding, identity_key, prepare_fixed_position_embeddings

OPSET = 17


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load_encoder(run: Path, config: dict) -> CardEmbedding:
    runtime = ROOT / config['transformers_runtime']
    if runtime.is_dir():
        sys.path.insert(0, str(runtime.resolve()))
    model_dir = (ROOT / config['pretrained_model_path']).resolve(strict=True)
    backbone = load_dinov2(model_dir)
    prepare_fixed_position_embeddings(backbone)
    encoder = CardEmbedding(backbone, config['embedding_dimension'])
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    return encoder.eval()


def raw_card(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def export_encoder(encoder, path: Path):
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(encoder, dummy, str(path), opset_version=OPSET, input_names=['pixels'],
                      output_names=['embedding'], dynamic_axes={'pixels': {0: 'batch'}, 'embedding': {0: 'batch'}},
                      do_constant_folding=True)


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Ultralytics-style centered letterbox with gray 114 padding; returns NCHW float32 in [0,1]."""
    import cv2
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    rw, rh = round(w * scale), round(h * scale)
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    x, y = (size - rw) // 2, (size - rh) // 2
    canvas[y:y + rh, x:x + rw] = resized
    tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), {'scale': scale, 'pad_x': x, 'pad_y': y, 'source_size': [w, h]}


def ort_session(path: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def latency_ms(fn, warmup: int = 3, runs: int = 15) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)), 'runs': runs, 'warmup': warmup}


def topk(vector: np.ndarray, gallery: np.ndarray, keys: list[str], k: int = 5) -> list[dict]:
    scores = gallery @ vector
    order = np.argsort(-scores)[:k]
    return [{'identity': keys[i], 'index': int(i), 'score': float(scores[i])} for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017')
    parser.add_argument('--detector', type=Path, default=ROOT / 'runs/card_detector/obb/corners-s640-v4-ft/weights/best.pt')
    parser.add_argument('--synthetic', type=Path, default=ROOT / 'runs/card_detector/synthetic')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=int, default=12, help='validation queries written as golden fixtures')
    parser.add_argument('--gallery-checks', type=int, default=32, help='references re-encoded to check stored vectors')
    parser.add_argument('--detector-size', type=int, default=640, help='square detector input; must match how the weights were trained')
    args = parser.parse_args()
    DETECTOR_SIZE = args.detector_size
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty')
    (out / 'fixtures').mkdir(parents=True)
    torch.set_num_threads(4)
    run = args.run.resolve(strict=True)
    config = json.loads((run / 'resolved_config.json').read_text())
    index = json.loads((run / 'gallery_index.json').read_text())
    if index['checkpoint_sha256'] != sha256_file(run / 'best.pt'):
        raise ValueError('Checkpoint differs from the gallery index')
    if index['vectors_sha256'] != sha256_file(run / 'gallery_embeddings.npy'):
        raise ValueError('Gallery vectors differ from the gallery index')
    if index['encoder_contract']['preprocessing_sha256'] != preprocessing_contract_hash():
        raise ValueError('Preprocessing contract changed since the gallery was built')
    gallery = np.load(run / 'gallery_embeddings.npy').astype(np.float32)
    references = index['references']
    keys = [identity_key(r['identity']) for r in references]
    report: dict = {'schema_version': 1, 'source_run': str(run.relative_to(ROOT)),
                    'checkpoint_sha256': index['checkpoint_sha256'], 'opset': OPSET,
                    'onnxruntime_execution_provider': 'CPUExecutionProvider',
                    'scope': 'Runtime parity and desktop CPU latency only; not recognition accuracy or phone performance'}

    # Encoder export and parity on real validation crops.
    encoder = load_encoder(run, config)
    encoder_path = out / 'encoder.onnx'
    export_encoder(encoder, encoder_path)
    session = ort_session(encoder_path)
    manifest = json.loads((args.dataset / 'recognition/manifest.json').read_text())
    queries = manifest['splits']['val']['queries'][:args.fixtures]
    fixtures, max_diff, min_cos = [], 0.0, 1.0
    for n, row in enumerate(queries):
        path = (args.dataset / row['image']).resolve(strict=True)
        if sha256_file(path) != row['sha256']:
            raise ValueError(f'Query hash mismatch: {row["image"]}')
        source = raw_card(path)
        canonical = canonicalize_card(source, width=224, height=312)
        tensor = prepare_encoder_input(canonical)
        with torch.inference_mode():
            reference = encoder(torch.from_numpy(tensor[None])).numpy()[0]
        onnx_vector = session.run(None, {'pixels': tensor[None]})[0][0]
        max_diff = max(max_diff, float(np.abs(reference - onnx_vector).max()))
        min_cos = min(min_cos, float(reference @ onnx_vector))
        name = f'query-{n:02d}'
        Image.fromarray(source).save(out / 'fixtures' / f'{name}-source.png')
        Image.fromarray(canonical).save(out / 'fixtures' / f'{name}-canonical.png')
        source.tofile(out / 'fixtures' / f'{name}-source.rgb')
        canonical.tofile(out / 'fixtures' / f'{name}-canonical.rgb')
        tensor.astype('<f4').tofile(out / 'fixtures' / f'{name}-input.f32')
        h, w = source.shape[:2]
        fixtures.append({'name': name, 'expected_identity': identity_key(row['identity']),
                         'source': f'{name}-source.png', 'canonical': f'{name}-canonical.png', 'input': f'{name}-input.f32',
                         'source_rgb': f'{name}-source.rgb', 'source_size': [w, h], 'canonical_rgb': f'{name}-canonical.rgb',
                         'corners': [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                         'embedding_torch_cpu': [round(float(v), 7) for v in reference],
                         'topk_torch_cpu': topk(reference, gallery, keys)})
    report['encoder'] = {'path': 'encoder.onnx', 'sha256': sha256_file(encoder_path), 'bytes': encoder_path.stat().st_size,
                         'parity_queries': len(fixtures), 'max_abs_diff_vs_torch_cpu': max_diff,
                         'min_cosine_vs_torch_cpu': min_cos,
                         'latency_batch1': latency_ms(lambda: session.run(None, {'pixels': np.zeros((1, 3, 224, 224), np.float32)}))}

    # Stored gallery vectors were encoded on MPS during training; check they agree with this runtime.
    rng = np.random.default_rng(0)
    check_rows = rng.choice(len(references), size=min(args.gallery_checks, len(references)), replace=False)
    gallery_cos = []
    for i in check_rows:
        row = references[int(i)]
        path = (args.dataset / row['image']).resolve(strict=True)
        tensor = prepare_encoder_input(canonicalize_card(raw_card(path), width=224, height=312))
        vector = session.run(None, {'pixels': tensor[None]})[0][0]
        gallery_cos.append(float(vector @ gallery[int(i)]))
    report['gallery_vectors_vs_onnx'] = {'checked': len(gallery_cos), 'min_cosine': min(gallery_cos),
                                         'mean_cosine': float(np.mean(gallery_cos))}

    # Gallery as little-endian float16 rows plus an identity table.
    half = gallery.astype('<f2')
    half.tofile(out / 'gallery.f16')
    roundtrip = half.astype(np.float32)
    report['gallery'] = {'path': 'gallery.f16', 'rows': int(gallery.shape[0]), 'dimension': int(gallery.shape[1]),
                         'dtype': 'float16-le', 'bytes': (out / 'gallery.f16').stat().st_size,
                         'sha256': sha256_file(out / 'gallery.f16'),
                         'fp16_top1_agreement': float(np.mean([
                             topk(np.asarray(f['embedding_torch_cpu'], np.float32), roundtrip, keys, 1)[0]['identity'] ==
                             f['topk_torch_cpu'][0]['identity'] for f in fixtures]))}
    write_json(out / 'gallery.json', {'schema_version': 1, 'vectors': 'gallery.f16', 'rows': int(gallery.shape[0]),
                                      'dimension': int(gallery.shape[1]), 'dtype': 'float16-le',
                                      'vectors_sha256': report['gallery']['sha256'],
                                      'encoder_sha256': report['encoder']['sha256'],
                                      'checkpoint_sha256': index['checkpoint_sha256'],
                                      'preprocessing': index['encoder_contract']['preprocessing'],
                                      'preprocessing_sha256': preprocessing_contract_hash(),
                                      'normalization': 'L2 rows; cosine = dot product',
                                      'thresholds_calibrated': False, 'identities': keys})

    # Detector export and raw-output parity on synthetic validation scenes.
    from ultralytics import YOLO
    yolo = YOLO(str(args.detector))
    if yolo.task not in {'pose', 'obb'}:
        raise ValueError(f'Unsupported detector task: {yolo.task}')
    exported = Path(yolo.export(format='onnx', imgsz=DETECTOR_SIZE, opset=OPSET, simplify=False, dynamic=False))
    detector_path = out / 'detector.onnx'
    exported.replace(detector_path)
    detector = ort_session(detector_path)
    torch_detector = YOLO(str(args.detector)).model.float().eval()
    scenes = sorted((args.synthetic / 'annotations/val').glob('*.json'))
    # Include empty, single-card and >9-card scenes without assuming a fixed scene count.
    selected = []
    for minimum, maximum in [(0, 0), (1, 1), (9, 9), (10, float('inf'))]:
        candidate = next((s for s in scenes if minimum <= len(json.loads(s.read_text())['instances']) <= maximum), None)
        if candidate is not None:
            selected.append(candidate)
    scenes = selected
    if not scenes:
        raise ValueError('No synthetic validation scenes available for detector parity')
    det_diff, det_fixtures = 0.0, []
    for n, scene_path in enumerate(scenes):
        scene = json.loads(scene_path.read_text())
        rgb = raw_card(args.synthetic / scene['image'])
        tensor, transform = letterbox(rgb, DETECTOR_SIZE)
        with torch.inference_mode():
            reference = torch_detector(torch.from_numpy(tensor))
            reference = (reference[0] if isinstance(reference, (list, tuple)) else reference).numpy()
        output = detector.run(None, {detector.get_inputs()[0].name: tensor})[0]
        det_diff = max(det_diff, float(np.abs(reference - output).max()))
        # Decode this exact square-letterbox output with the matching Ultralytics decoder.
        from ultralytics.utils import ops
        pred = ops.non_max_suppression(torch.from_numpy(output.copy()), 0.25, 0.7, nc=1, rotated=yolo.task == 'obb')[0]
        if yolo.task == 'pose':
            corners = ops.scale_coords((DETECTOR_SIZE, DETECTOR_SIZE), pred[:, 6:].view(len(pred), 4, 3).clone(), rgb.shape)
            decoded = {'ultralytics_corners': corners[..., :2].numpy().round(3).tolist()}
        else:
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes((DETECTOR_SIZE, DETECTOR_SIZE), rboxes[:, :4], rgb.shape, xywh=True)
            decoded = {'ultralytics_obb_xywhr': rboxes.numpy().round(3).tolist()}
        name = f'scene-{n:02d}'
        Image.fromarray(rgb).save(out / 'fixtures' / f'{name}.png')
        tensor.astype('<f4').tofile(out / 'fixtures' / f'{name}-input.f32')
        output.astype('<f4').tofile(out / 'fixtures' / f'{name}-output.f32')
        det_fixtures.append({'name': name, 'image': f'{name}.png', 'input_size': DETECTOR_SIZE, 'input': f'{name}-input.f32', 'output': f'{name}-output.f32',
                             'output_shape': list(output.shape), 'letterbox': transform,
                             'true_corners': [i['true_corners'] for i in scene['instances']],
                             'kind': yolo.task, **decoded,
                             'ultralytics_conf': pred[:, 4].numpy().round(5).tolist(),
                             'decode': {'conf': 0.25, 'iou': 0.7, 'nms': 'greedy IoU' if yolo.task == 'pose' else 'fast probiou'},
                             'source_scene': scene['image'], 'label_origin': scene.get('label_origin', 'renderer-known corners')})
    if det_diff > 0.05:
        raise ValueError(f'Detector export parity failed: max absolute difference {det_diff}')
    report['detector'] = {'path': 'detector.onnx', 'sha256': sha256_file(detector_path), 'bytes': detector_path.stat().st_size,
                          'input_size': DETECTOR_SIZE, 'letterbox_pad_value': 114, 'output_shape': det_fixtures[0]['output_shape'],
                          'parity_scenes': len(det_fixtures), 'max_abs_diff_vs_torch_cpu': det_diff,
                          'kind': yolo.task, 'source_weights_sha256': sha256_file(args.detector),
                          'weights_origin': str(args.detector.relative_to(ROOT)) if args.detector.is_absolute() else str(args.detector),
                          'latency': latency_ms(lambda: detector.run(None, {detector.get_inputs()[0].name:
                                                                            np.zeros((1, 3, DETECTOR_SIZE, DETECTOR_SIZE), np.float32)}))}
    # A tilted-quad warp exercises true perspective, not just full-bounds scaling.
    first = fixtures[0]
    source = raw_card(out / 'fixtures' / first['source'])
    h, w = source.shape[:2]
    quad = np.float32([[0.07 * w, 0.04 * h], [0.95 * w, 0.09 * h], [0.9 * w, 0.97 * h], [0.03 * w, 0.92 * h]])
    canonicalize_card(source, quad, 224, 312).tofile(out / 'fixtures' / 'warp-perspective.rgb')
    write_json(out / 'fixtures/warp.json', {'source_rgb': first['source_rgb'], 'source_size': [w, h],
                                            'corners': quad.tolist(), 'expected_rgb': 'warp-perspective.rgb'})
    write_json(out / 'fixtures/encoder.json', {'gallery_rows': int(gallery.shape[0]), 'fixtures': fixtures})
    write_json(out / 'fixtures/detector.json', {'fixtures': det_fixtures})
    write_json(out / 'export_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in ('encoder', 'gallery', 'gallery_vectors_vs_onnx', 'detector')}, indent=2))


if __name__ == '__main__':
    main()
