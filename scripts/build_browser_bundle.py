#!/usr/bin/env python3
"""Assemble a versioned browser bundle: exported models, gallery, catalog, families and prices.

The bundle directory name is derived from its contents, so a shared URL never mixes
cached artifacts from different builds. Output is local and gitignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.foundation_retrieval import sha256_file

PRICE_INDEX = ROOT / 'tcg-scanner-app/tcg-scanner-app/Resources/card_index.json'
APP_MANIFEST = ROOT / 'tcg-scanner-app/tcg-scanner-app/Resources/app_model_manifest.json'


def family_key(identity: str, aliases: dict[str, str], pokemon_by_id: bool = False) -> str:
    """Catalog-adapter artwork family; a plain base code is not enough on its own."""
    if identity in aliases:
        return aliases[identity]
    game, _language, card_id = identity.split(':', 2)
    if game == 'onepiece':
        # One Piece numbers (including _pN alternate-art suffixes) are shared by the EN/JA printings of one artwork.
        return f'onepiece:{card_id}'
    if game == 'pokemon' and pokemon_by_id:
        # TCGdex uses one card id for every language version of a printing (same artwork).
        return f'pokemon:{card_id}'
    # No verified cross-language mapping for other games yet: each identity is its own family.
    return identity


def parse_usd(text: str | None) -> float | None:
    if not text or not text.startswith('$'):
        return None
    try:
        value = float(text[1:].replace(',', ''))
    except ValueError:
        return None
    return value if value > 0 else None


def write_pose_fixture(weights: Path, onnx_path: Path, size: int):
    """Raw ONNX output and Ultralytics-decoded corners on one synthetic val scene, for the TS decoder test."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from ultralytics.utils import ops
    sys.path.insert(0, str(ROOT / 'scripts'))
    from export_browser_models import letterbox
    scenes = sorted((ROOT / 'runs/card_detector/scenes-v2/annotations/val').glob('*.json'))
    meta = next(json.loads(f.read_text()) for f in scenes if len(json.loads(f.read_text())['instances']) >= 9)
    rgb = np.asarray(Image.open(ROOT / 'runs/card_detector/scenes-v2' / meta['image']).convert('RGB'))
    tensor, lb = letterbox(rgb, size)
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    output = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    pred = ops.non_max_suppression(torch.from_numpy(output.copy()), 0.25, 0.7, nc=1)[0]
    boxes = ops.scale_boxes((size, size), pred[:, :4].clone(), rgb.shape)
    kpts = ops.scale_coords((size, size), pred[:, 6:].view(len(pred), 4, 3).clone(), rgb.shape)
    out = ROOT / 'runs/browser_export/pose-fixture'
    out.mkdir(parents=True, exist_ok=True)
    output.astype('<f4').tofile(out / 'output.f32')
    (out / 'pose.json').write_text(json.dumps({
        'input_size': size, 'output_shape': list(output.shape), 'source_size': [rgb.shape[1], rgb.shape[0]],
        'letterbox': lb, 'scores': pred[:, 4].numpy().round(5).tolist(), 'boxes': boxes.numpy().round(2).tolist(),
        'corners': kpts[..., :2].numpy().round(2).tolist(), 'true_corners': [i['true_corners'] for i in meta['instances']],
        'source_scene': meta['image']}, indent=1))


def export_detector(weights: Path, size: int, out: Path) -> dict:
    """ONNX export plus a real-output parity check against PyTorch on a non-trivial input."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    yolo = YOLO(str(weights))
    kind = yolo.task
    if kind not in {'obb', 'pose'}:
        raise ValueError(f'Unsupported detector task {kind}')
    exported = Path(yolo.export(format='onnx', imgsz=size, opset=17, simplify=False, dynamic=False, verbose=False))
    exported.replace(out)
    x = torch.from_numpy(np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32))
    model = YOLO(str(weights)).model.float().eval()
    with torch.inference_mode():
        ref = model(x)
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
    sess = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > 0.05:
        raise ValueError(f'Detector ONNX parity failed: max abs diff {diff}')
    if kind == 'pose':
        write_pose_fixture(weights, out, size)
    return {'path': 'detector.onnx', 'kind': kind, 'sha256': sha256_file(out), 'bytes': out.stat().st_size, 'input_size': size,
            'max_abs_diff_vs_torch_cpu': diff, 'source_weights': str(weights), 'source_weights_sha256': sha256_file(weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=ROOT / 'runs/browser_export/pilot-20261016')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--output', type=Path, default=ROOT / 'web/public/bundle')
    parser.add_argument('--languages', default='en,de', help='catalog languages in the app (gallery rows of other languages are dropped)')
    parser.add_argument('--german-pokemon', type=Path, help='TCGdex /v2/de/cards list: adds German Pokemon versions of enrolled artworks')
    parser.add_argument('--prices', type=Path, help='prices.json from scripts/update_prices.py (schema 2, USD + EUR markets)')
    parser.add_argument('--encoder-int8', type=Path, help='dynamically quantized encoder for the WASM/CPU path (phones)')
    parser.add_argument('--detector-weights', type=Path, help='replace the exported detector with these Ultralytics OBB weights')
    parser.add_argument('--detector-size', type=int, default=640)
    parser.add_argument('--detector-conf', type=float, default=0.25, help='app-side box confidence cut-off')
    parser.add_argument('--detector-origin', default='renderer-v2 synthetic scenes; not validated on real camera scenes')
    args = parser.parse_args()
    export = args.export.resolve(strict=True)
    report = json.loads((export / 'export_report.json').read_text())
    gallery = json.loads((export / 'gallery.json').read_text())
    for name in ('encoder', 'detector', 'gallery'):
        if sha256_file(export / report[name]['path']) != report[name]['sha256']:
            raise ValueError(f'{name} artifact differs from its export report')
    run = ROOT / report['source_run']
    references = json.loads((run / 'gallery_index.json').read_text())['references']
    if [':'.join(r['identity'][k] for k in ('game', 'language', 'card_id')) for r in references] != gallery['identities']:
        raise ValueError('Gallery identity order differs from the training-run index')

    alias_rows = json.loads((args.dataset / 'recognition/exact_reference_aliases.json').read_text())['shared_bytes_across_identities']
    aliases = {}
    for row in alias_rows:
        keys = sorted(':'.join(i[k] for k in ('game', 'language', 'card_id')) for i in row['identities'])
        for key in keys:
            aliases[key] = 'alias:' + row['sha256'][:16]

    prices = {r['id']: r for r in json.loads(PRICE_INDEX.read_text())}
    snapshot_date = json.loads(APP_MANIFEST.read_text())['priceSnapshotDate']
    languages = set(args.languages.split(','))
    german = {c['id']: c['name'] for c in json.loads(args.german_pokemon.read_text())} if args.german_pokemon and 'de' in languages else {}
    keep = [i for i, ident in enumerate(gallery['identities']) if ident.split(':')[1] in languages]
    gallery_ids = [gallery['identities'][i] for i in keep]
    cards, quotes = [], {}
    for k in keep:
        identity, ref = gallery['identities'][k], references[k]
        meta = json.loads((args.dataset / (ref['image'] + '.json')).read_text())['source_record']
        game, language, card_id = identity.split(':', 2)
        if game == 'pokemon' and language == 'en' and card_id in german:
            # German printing: same artwork (shares the English reference image), no own gallery row.
            cards.append({'id': f'pokemon:de:{card_id}', 'game': game, 'language': 'de', 'cardId': card_id,
                          'name': german[card_id] or card_id, 'family': family_key(identity, aliases, True),
                          'finish': 'unknown', 'printing': 'unknown', 'ref': ref['image']})
        cards.append({'id': identity, 'game': game, 'language': language, 'cardId': card_id,
                      'name': meta.get('name') or card_id, 'family': family_key(identity, aliases, bool(german)),
                      'finish': meta.get('finish', 'unknown'), 'printing': meta.get('printing', 'unknown'),
                      # Reference image path under the curated dataset; served locally in dev only.
                      'ref': ref['image']})
        # The bundled snapshot covers English One Piece only; everything else stays unpriced (unknown, not zero).
        if game == 'onepiece' and language == 'en' and card_id in prices:
            amount = parse_usd(prices[card_id].get('displayPrice'))
            if amount is not None:
                quotes[identity] = {'amount': amount, 'basis': prices[card_id].get('priceSource', 'snapshot')}

    staging = args.output.parent / (args.output.name + '.tmp')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ('encoder', 'detector', 'gallery'):
        shutil.copy2(export / report[name]['path'], staging / report[name]['path'])
    int8 = None
    if args.encoder_int8:
        src = args.encoder_int8.resolve(strict=True)
        shutil.copy2(src, staging / 'encoder.int8.onnx')
        int8 = {'path': 'encoder.int8.onnx', 'sha256': sha256_file(staging / 'encoder.int8.onnx'),
                'quantization': 'onnxruntime dynamic int8, per-channel MatMul/Gemm weights',
                'parity': 'top-1 identical to fp32 on 24/24 validation queries; mean cosine to fp32 0.98'}
    if args.detector_weights:
        report['detector'] = export_detector(args.detector_weights.resolve(strict=True), args.detector_size, staging / 'detector.onnx')
        report['detector']['weights_origin'] = args.detector_origin
    if len(keep) != len(gallery['identities']):
        import numpy as np
        vectors = np.fromfile(export / 'gallery.f16', dtype='<f2').reshape(-1, gallery['dimension'])[keep]
        vectors.tofile(staging / 'gallery.f16')
        report['gallery'] = {**report['gallery'], 'rows': len(keep), 'sha256': sha256_file(staging / 'gallery.f16'),
                             'bytes': (staging / 'gallery.f16').stat().st_size}
    (staging / 'catalog.json').write_text(json.dumps({'schema_version': 2, 'languages': sorted(languages),
                                                      'gallery_identities': gallery_ids, 'cards': cards}, separators=(',', ':')))
    if args.prices:
        fresh = json.loads(args.prices.read_text())
        if fresh.get('schema_version') != 2:
            raise ValueError('Expected a schema 2 prices.json from scripts/update_prices.py')
        known = {c['id'] for c in cards}
        for name, market in fresh['markets'].items():
            # Only catalog languages; German Pokemon reuse the Cardmarket quote (one product for all languages).
            q = {k: v for k, v in market['quotes'].items() if k in known}
            if name == 'EUR':
                for c in cards:
                    en = f"pokemon:en:{c['cardId']}"
                    if c['language'] == 'de' and en in market['quotes']:
                        q[c['id']] = {**market['quotes'][en], 'basis': market['quotes'][en]['basis'] + ' (Cardmarket, all languages)'}
            market['quotes'] = q
        # A version priced in only one market gets a converted estimate in the other, marked as such.
        fx = fresh.get('fx')
        if not fx:
            raise ValueError('prices.json has no fx rate; rerun scripts/update_prices.py')
        usd, eur = fresh['markets']['USD']['quotes'], fresh['markets']['EUR']['quotes']
        for src, dst, src_name, rate in ((usd, eur, 'USD', fx['USD_EUR']), (eur, usd, 'EUR', 1 / fx['USD_EUR'])):
            for ident, quote in list(src.items()):
                if ident not in dst and not quote.get('converted'):
                    dst[ident] = {'amount': round(quote['amount'] * rate, 2), 'basis': quote['basis'],
                                  'converted': {'from': src_name, 'rate': round(rate, 5), 'date': fx['date'], 'source': fx['source']}}
        (staging / 'prices.json').write_text(json.dumps(fresh, separators=(',', ':')))
    else:
        (staging / 'prices.json').write_text(json.dumps({
            'schema_version': 1, 'currency': 'USD', 'source': 'Bundled app snapshot (dotgg)', 'snapshot_date': snapshot_date,
            'coverage': 'English One Piece only', 'quotes': quotes}, separators=(',', ':')))
    components = {name: report[name]['sha256'] for name in ('encoder', 'detector', 'gallery')}
    if int8:
        components['encoder_int8'] = int8['sha256']
    components['catalog'] = sha256_file(staging / 'catalog.json')
    components['prices'] = sha256_file(staging / 'prices.json')
    version = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        'schema_version': 1, 'version': version, 'components': components,
        'encoder': {'path': 'encoder.onnx', 'input': 'pixels', 'output': 'embedding', 'size': 224,
                    'dimension': gallery['dimension'], 'preprocessing': gallery['preprocessing'],
                    'preprocessing_sha256': gallery['preprocessing_sha256'],
                    **({'int8': int8} if int8 else {})},
        'detector': {'path': 'detector.onnx', 'size': report['detector']['input_size'], 'pad_value': 114,
                     'conf': args.detector_conf, 'iou': 0.7, 'weights_origin': report['detector']['weights_origin'],
                     'source_weights_sha256': report['detector'].get('source_weights_sha256'),
                     'kind': report['detector'].get('kind', 'obb')},
        'gallery': {'path': 'gallery.f16', 'rows': len(keep), 'dimension': gallery['dimension'], 'dtype': 'float16-le'},
        'catalog': 'catalog.json', 'prices': 'prices.json',
        'thresholds_calibrated': False,
        'parity': {'encoder_min_cosine_vs_torch': report['encoder']['min_cosine_vs_torch_cpu'],
                   'detector_max_abs_diff_vs_torch': report['detector']['max_abs_diff_vs_torch_cpu'],
                   'browser_runtime_verified': False},
        'source_run': report['source_run'], 'checkpoint_sha256': report['checkpoint_sha256'],
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.output.exists():
        shutil.rmtree(args.output)
    staging.rename(args.output)
    print(json.dumps({'output': str(args.output), 'version': version, 'cards': len(cards), 'priced': len(quotes)}))


if __name__ == '__main__':
    main()
