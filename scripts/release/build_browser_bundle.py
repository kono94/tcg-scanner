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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.recognition.backbone import sha256_file


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


def write_pose_fixture(weights: Path, onnx_path: Path, size: int):
    """Raw ONNX output and Ultralytics-decoded corners on one synthetic val scene, for the TS decoder test."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from ultralytics.utils import ops
    from scripts.release.export_browser_models import letterbox
    scenes = sorted((ROOT / 'datasets/detection/parity-scenes-v4/annotations/val').glob('*.json'))
    meta = next(json.loads(f.read_text()) for f in scenes if len(json.loads(f.read_text())['instances']) >= 9)
    rgb = np.asarray(Image.open(ROOT / 'datasets/detection/parity-scenes-v4' / meta['image']).convert('RGB'))
    tensor, lb = letterbox(rgb, size)
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    output = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    pred = ops.non_max_suppression(torch.from_numpy(output.copy()), 0.25, 0.7, nc=1)[0]
    boxes = ops.scale_boxes((size, size), pred[:, :4].clone(), rgb.shape)
    kpts = ops.scale_coords((size, size), pred[:, 6:].view(len(pred), 4, 3).clone(), rgb.shape)
    out = ROOT / 'models/browser-export/corner-decoder-fixture'
    out.mkdir(parents=True, exist_ok=True)
    output.astype('<f4').tofile(out / 'output.f32')
    (out / 'pose.json').write_text(json.dumps({
        'input_size': size, 'output_shape': list(output.shape), 'source_size': [rgb.shape[1], rgb.shape[0]],
        'letterbox': lb, 'scores': pred[:, 4].numpy().round(5).tolist(), 'boxes': boxes.numpy().round(2).tolist(),
        'corners': kpts[..., :2].numpy().round(2).tolist(), 'true_corners': [i['true_corners'] for i in meta['instances']],
        'source_scene': meta['image']}, indent=1))


BACK_NAMES = {'pokemon': 'Pokémon card back', 'onepiece': 'One Piece card back'}


def embed_back_references(manifest_path: Path, dataset: Path, encoder: Path, dimension: int):
    """Gallery rows for card-back references (identity ``back:<game>``).

    Known-gallery-bounds adapter: each reference is a verified full-card image, so it goes straight to
    the shared canonicalizer and encoder input (tcg-full-card-v1) without a detector, and through the
    same fp32 ONNX encoder as the card gallery. Rows are L2-normalized; invalid vectors are rejected.
    Returns (float32 rows, row identities, catalog entries, manifest info).
    """
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from training.recognition.canonical import PREPROCESSING_VERSION, canonicalize_card, prepare_encoder_input
    data = json.loads(manifest_path.read_text())
    if data.get('artifact_type') != 'card_back_references':
        raise ValueError(f'{manifest_path} is not a card-back reference manifest')
    sess = ort.InferenceSession(str(encoder), providers=['CPUExecutionProvider'])
    rows, identities, cards = [], [], {}
    for ref in data['references']:
        if ref.get('synthetic'):
            raise ValueError(f"synthetic image {ref['image']} must not be enrolled as a gallery reference")
        path = dataset / ref['image']
        if sha256_file(path) != ref['sha256']:
            raise ValueError(f'{path} differs from its manifest hash')
        rgb = np.asarray(Image.open(path).convert('RGB'))
        v = sess.run(None, {sess.get_inputs()[0].name: prepare_encoder_input(canonicalize_card(rgb))[None]})[0][0].astype(np.float64)
        norm = float(np.linalg.norm(v))
        if v.shape != (dimension,) or not np.isfinite(v).all() or norm < 1e-6:
            raise ValueError(f'invalid back embedding for {path}')
        rows.append((v / norm).astype(np.float32))
        identities.append(ref['identity'])
        game = ref['game']
        cards.setdefault(ref['identity'], {
            'id': ref['identity'], 'game': game, 'language': 'any', 'cardId': 'back', 'name': BACK_NAMES.get(game, f'{game} card back'),
            # Own family: a back never shares an artwork family with a real card.
            'family': ref['identity'], 'finish': 'n/a', 'printing': 'n/a', 'kind': 'back', 'ref': ref['image']})
    info = {'rows': len(rows), 'identities': sorted(cards), 'references_manifest': str(manifest_path),
            'references_manifest_sha256': sha256_file(manifest_path), 'encoder': encoder.name,
            'preprocessing': PREPROCESSING_VERSION, 'adapter': 'known full-card bounds (no detector)',
            'note': 'Back rows are appended after the card rows; backs are never priced, matched or counted as cards.'}
    return np.stack(rows), identities, list(cards.values()), info


def load_enrolled(path: Path, encoder_sha256: str, dimension: int):
    """Gallery rows, catalog entries and USD quotes of an enrolled game (scripts/catalog/enroll_yugioh_gallery.py).

    Each artwork is a gallery row (`kind: artwork`, the printing left open); its family's printings are
    catalog-only versions carrying the prices. Embeddings must come from this bundle's fp32 encoder.
    """
    import numpy as np
    meta = json.loads(path.read_text())
    if meta['encoder_sha256'] != encoder_sha256:
        raise ValueError(f'{path} was embedded with another encoder; rerun the enrollment')
    vectors_path = path.with_suffix('.f16')
    if sha256_file(vectors_path) != meta['vectors_sha256']:
        raise ValueError(f'{vectors_path} differs from {path.name}')
    enrollment = ROOT / meta['enrollment']
    if sha256_file(enrollment) != meta['enrollment_sha256']:
        raise ValueError(f'{enrollment} changed since the embeddings were made')
    data = json.loads(enrollment.read_text())
    if [a['identity'] for a in data['artworks']] != meta['identities']:
        raise ValueError(f'{enrollment} and {path.name} list different rows')
    rows = np.fromfile(vectors_path, dtype='<f2').reshape(-1, dimension)
    game, language = data['game'], data['language']
    cards = [{'id': a['identity'], 'game': game, 'language': language, 'cardId': a['passcode'], 'name': a['name'],
              'family': a['family'], 'finish': 'unknown', 'printing': 'unknown', 'kind': 'artwork', 'ref': a['image']}
             for a in data['artworks']]
    first_ref = {}
    for a in data['artworks']:
        first_ref.setdefault(a['family'], a['image'])
    quotes = {}
    for p in data['printings']:
        cards.append({'id': p['id'], 'game': game, 'language': language, 'cardId': p['set_code'], 'name': p['name'],
                      'family': p['family'], 'finish': p['rarity'], 'printing': p['set_name'], 'ref': first_ref[p['family']]})
        if p['price_usd'] is not None:
            quotes[p['id']] = {'amount': p['price_usd'], 'basis': f"ygoprodeck:set_price {data['price_date']}"}
    info = {'game': game, 'rows': len(rows), 'artworks': len(data['artworks']), 'printings': len(data['printings']),
            'priced_printings': len(quotes), 'enrollment_sha256': meta['enrollment_sha256'], 'price_source': data['price_source']}
    return rows, list(meta['identities']), cards, quotes, info


def convert_detector_fp16(src: Path, dst: Path, evaluated: Path, fixtures: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    from scripts.release.compare_model_precisions import to_fp16
    to_fp16(src, dst)
    new, ref = (ort.InferenceSession(str(p), providers=['CPUExecutionProvider']) for p in (dst, evaluated))
    scenes = sorted(fixtures.glob('scene-*-input.f32'))
    if not scenes:
        raise ValueError(f'no detector parity scenes in {fixtures}')
    for scene in scenes:
        x = np.fromfile(scene, dtype=np.float32).reshape(1, 3, *new.get_inputs()[0].shape[2:])
        diff = float(np.abs(new.run(None, {'images': x})[0] - ref.run(None, {'images': x})[0]).max())
        if diff > 1e-3:
            raise ValueError(f'fp16 detector differs from the evaluated one on {scene.name} (max abs diff {diff})')


def export_detector(weights: Path, size: int, out: Path) -> dict:
    """ONNX export plus a real-output parity check against PyTorch on a non-trivial input."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    yolo = YOLO(str(weights))
    kind = yolo.task
    if kind != 'pose':
        raise ValueError(f'Detector must be a YOLO-pose corner model, got task {kind}')
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
    write_pose_fixture(weights, out, size)
    return {'path': 'detector.onnx', 'kind': kind, 'sha256': sha256_file(out), 'bytes': out.stat().st_size, 'input_size': size,
            'max_abs_diff_vs_torch_cpu': diff, 'source_weights': str(weights), 'source_weights_sha256': sha256_file(weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=ROOT / 'models/browser-export/gpu-20261017')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/curated')
    parser.add_argument('--output', type=Path, default=ROOT / 'web/public/bundle')
    parser.add_argument('--languages', default='en,de', help='catalog languages in the app (gallery rows of other languages are dropped)')
    parser.add_argument('--german-pokemon', type=Path, default=ROOT / 'datasets/sources/catalog/pokemon-de-tcgdex/cards.json',
                        help='saved TCGdex /v2/de/cards list: adds German Pokemon versions of enrolled artworks')
    parser.add_argument('--prices', type=Path, default=ROOT / 'datasets/prices/prices.json',
                        help='prices.json from scripts/release/update_prices.py (schema 2, USD + EUR markets)')
    parser.add_argument('--encoder-int8', type=Path, help='dynamically quantized encoder for the WASM path')
    parser.add_argument('--fp16', action='store_true',
                        help='ship the fp16 encoder/detector from <export>/precision_report.json (scripts/release/compare_model_precisions.py)')
    parser.add_argument('--detector-weights', type=Path, help='replace the exported detector with these Ultralytics pose weights')
    parser.add_argument('--detector-size', type=int, default=640)
    parser.add_argument('--detector-conf', type=float, default=0.6, help='app-side box confidence cut-off')
    parser.add_argument('--detector-origin', default='corners-s640-v4-ft; synthetic pretraining plus AI-reviewed real-corner fine tuning; not device-verified')
    parser.add_argument('--backs', type=Path, help='card-back reference manifest (default: <dataset>/recognition/backs/manifest.json when present)')
    parser.add_argument('--no-backs', action='store_true', help='build without card-back gallery rows')
    parser.add_argument('--logos', type=Path, default=ROOT / 'datasets/sources/catalog/game-logos/manifest.json',
                        help='game logos for the game picker (scripts/catalog/acquire_game_logos.py); skipped when absent')
    parser.add_argument('--enrolled', type=Path, action='append', default=[],
                        help='enrolled gallery (e.g. <export>/enrolled/yugioh.json from scripts/catalog/enroll_yugioh_gallery.py); repeatable')
    args = parser.parse_args()
    export = args.export.resolve(strict=True)
    prices_path = args.prices.resolve(strict=True)
    report = json.loads((export / 'export_report.json').read_text())
    gallery = json.loads((export / 'gallery.json').read_text())
    for name in ('encoder', 'detector', 'gallery'):
        if sha256_file(export / report[name]['path']) != report[name]['sha256']:
            raise ValueError(f'{name} artifact differs from its export report')
    run = ROOT / report['source_run']
    if sha256_file(run / 'best.pt') != report['checkpoint_sha256']:
        raise ValueError(f'{run} is not the checkpoint this export came from')
    references = json.loads((run / 'gallery_index.json').read_text())['references']
    if [':'.join(r['identity'][k] for k in ('game', 'language', 'card_id')) for r in references] != gallery['identities']:
        raise ValueError('Gallery identity order differs from the training-run index')

    alias_rows = json.loads((args.dataset / 'recognition/exact_reference_aliases.json').read_text())['shared_bytes_across_identities']
    aliases = {}
    for row in alias_rows:
        keys = sorted(':'.join(i[k] for k in ('game', 'language', 'card_id')) for i in row['identities'])
        for key in keys:
            aliases[key] = 'alias:' + row['sha256'][:16]

    languages = set(args.languages.split(','))
    german = {c['id']: c['name'] for c in json.loads(args.german_pokemon.resolve(strict=True).read_text())} if 'de' in languages else {}
    keep = [i for i, ident in enumerate(gallery['identities']) if ident.split(':')[1] in languages]
    gallery_ids = [gallery['identities'][i] for i in keep]
    cards = []
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
                'parity': 'top-1 identical to fp32 on 296/298 real queries; mean cosine to fp32 0.991 (precision_report.json)'}
    if args.detector_weights:
        report['detector'] = export_detector(args.detector_weights.resolve(strict=True), args.detector_size, staging / 'detector.onnx')
        report['detector']['weights_origin'] = args.detector_origin
    fp16 = {}
    if args.fp16:
        precision = json.loads((export / 'precision_report.json').read_text())
        for name, file in (('encoder', 'encoder.fp16.onnx'), ('detector', 'detector.fp16.onnx')):
            if sha256_file(export / file) != precision['models'][file]['sha256']:
                raise ValueError(f'{file} differs from precision_report.json')
            if precision['fp16_sources'][file] == report[name]['sha256']:
                shutil.copy2(export / file, staging / file)
            elif name == 'detector' and args.detector_weights:
                # Re-exported detector (same weights, different bytes): convert it here and require the
                # evaluated fp16 detector's outputs, so the precision report still describes it.
                convert_detector_fp16(staging / 'detector.onnx', staging / file, export / file, export / 'fixtures')
            else:
                raise ValueError(f'{file} was converted from a different {name}; rerun compare_model_precisions.py')
            fp16[name] = {'path': file, 'sha256': sha256_file(staging / file),
                          'conversion': precision['conversion'] if name == 'encoder' else 'onnxruntime.transformers.float16, keep_io_types'}
        e, d = precision['encoder'], precision['detector']['labeled_heldout']
        fp16['encoder']['parity'] = (f"top-1 same as fp32 on {e['fp16']['top1_same_as_fp32']}/{e['queries']} real queries; "
                                     f"min cosine to fp32 {e['fp16']['cosine_to_fp32']['min']}")
        fp16['detector']['parity'] = (f"{d['fp16']['found_within_5pct']}/{d['fp16']['cards']} held-out real cards found "
                                      f"(fp32 {d['fp32']['found_within_5pct']})")
    enrolled_rows, enrolled_quotes, enrolled_info = [], {}, []
    for path in args.enrolled:
        rows, extra_ids, extra_cards, quotes, info = load_enrolled(path.resolve(strict=True), report['encoder']['sha256'], gallery['dimension'])
        enrolled_rows.append(rows)
        gallery_ids += extra_ids
        cards += extra_cards
        enrolled_quotes.update(quotes)
        enrolled_info.append(info)
    backs_manifest = None if args.no_backs else (args.backs or args.dataset / 'recognition/backs/manifest.json')
    backs = None
    if backs_manifest is not None and (args.backs or backs_manifest.exists()):
        back_rows, back_ids, back_cards, backs = embed_back_references(backs_manifest.resolve(strict=True), args.dataset,
                                                                       export / report['encoder']['path'], gallery['dimension'])
        gallery_ids += back_ids
        cards += back_cards
    if len(keep) != len(gallery['identities']) or backs or enrolled_rows:
        import numpy as np
        vectors = np.concatenate([np.fromfile(export / 'gallery.f16', dtype='<f2').reshape(-1, gallery['dimension'])[keep], *enrolled_rows])
        if backs:
            vectors = np.concatenate([vectors, back_rows.astype('<f2')])
        vectors.tofile(staging / 'gallery.f16')
        report['gallery'] = {**report['gallery'], 'rows': len(vectors), 'sha256': sha256_file(staging / 'gallery.f16'),
                             'bytes': (staging / 'gallery.f16').stat().st_size}
    logos = {}
    if args.logos.exists():
        (staging / 'logos').mkdir()
        for logo in json.loads(args.logos.read_text())['logos']:
            src = args.logos.parent / logo['file']
            if sha256_file(src) != logo['sha256']:
                raise ValueError(f'{src} differs from its logo manifest')
            shutil.copy2(src, staging / 'logos' / logo['file'])
            logos[logo['game']] = {'path': f"logos/{logo['file']}", 'sha256': logo['sha256'],
                                   **({'monochromeDark': True} if logo.get('monochrome_dark') else {})}
    (staging / 'catalog.json').write_text(json.dumps({'schema_version': 2, 'languages': sorted(languages),
                                                      'gallery_identities': gallery_ids, 'cards': cards}, separators=(',', ':')))
    fresh = json.loads(prices_path.read_text())
    if fresh.get('schema_version') != 2:
        raise ValueError('Expected a schema 2 prices.json from scripts/release/update_prices.py')
    known = {c['id'] for c in cards}
    for name, market in fresh['markets'].items():
        # Only catalog languages; German Pokemon reuse the Cardmarket quote (one product for all languages).
        q = {k: v for k, v in market['quotes'].items() if k in known}
        if name == 'EUR':
            for c in cards:
                en = f"pokemon:en:{c['cardId']}"
                if c['language'] == 'de' and en in market['quotes']:
                    q[c['id']] = {**market['quotes'][en], 'basis': market['quotes'][en]['basis'] + ' (Cardmarket, all languages)'}
        if name == 'USD':
            q.update(enrolled_quotes)
        market['quotes'] = q
    if enrolled_quotes:
        fresh['markets']['USD']['source'] += '; YGOPRODeck set prices for Yu-Gi-Oh!'
    # A version priced in only one market gets a converted estimate in the other, marked as such.
    fx = fresh.get('fx')
    if not fx:
        raise ValueError('prices.json has no fx rate; rerun scripts/release/update_prices.py')
    usd, eur = fresh['markets']['USD']['quotes'], fresh['markets']['EUR']['quotes']
    for src, dst, src_name, rate in ((usd, eur, 'USD', fx['USD_EUR']), (eur, usd, 'EUR', 1 / fx['USD_EUR'])):
        for ident, quote in list(src.items()):
            if ident not in dst and not quote.get('converted'):
                dst[ident] = {'amount': round(quote['amount'] * rate, 2), 'basis': quote['basis'],
                              'converted': {'from': src_name, 'rate': round(rate, 5), 'date': fx['date'], 'source': fx['source']}}
    (staging / 'prices.json').write_text(json.dumps(fresh, separators=(',', ':')))
    components = {name: report[name]['sha256'] for name in ('encoder', 'detector', 'gallery')}
    if int8:
        components['encoder_int8'] = int8['sha256']
    for name, variant in fp16.items():
        components[f'{name}_fp16'] = variant['sha256']
    if logos:
        components['logos'] = hashlib.sha256(json.dumps(logos, sort_keys=True).encode()).hexdigest()
    components['catalog'] = sha256_file(staging / 'catalog.json')
    components['prices'] = sha256_file(staging / 'prices.json')
    version = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        'schema_version': 1, 'version': version, 'components': components,
        'encoder': {'path': 'encoder.onnx', 'input': 'pixels', 'output': 'embedding', 'size': 224,
                    'dimension': gallery['dimension'], 'preprocessing': gallery['preprocessing'],
                    'preprocessing_sha256': gallery['preprocessing_sha256'],
                    **({'fp16': fp16['encoder']} if fp16 else {}), **({'int8': int8} if int8 else {})},
        'detector': {'path': 'detector.onnx', 'size': report['detector']['input_size'], 'pad_value': 114,
                     'conf': args.detector_conf, 'iou': 0.7, 'weights_origin': report['detector']['weights_origin'],
                     'source_weights_sha256': report['detector'].get('source_weights_sha256'),
                     'kind': report['detector'].get('kind', 'pose'), **({'fp16': fp16['detector']} if fp16 else {})},
        'gallery': {'path': 'gallery.f16', 'rows': len(gallery_ids), 'dimension': gallery['dimension'], 'dtype': 'float16-le',
                    'card_rows': len(keep), 'enrolled_rows': sum(i['rows'] for i in enrolled_info),
                    'back_rows': backs['rows'] if backs else 0},
        'games': sorted({c['game'] for c in cards}),
        **({'logos': logos} if logos else {}),
        **({'enrolled': enrolled_info} if enrolled_info else {}),
        **({'backs': backs} if backs else {}),
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
    print(json.dumps({'output': str(args.output), 'version': version, 'cards': len(cards), 'priced': len({*usd, *eur}),
                      'gallery_rows': len(gallery_ids), 'back_rows': backs['rows'] if backs else 0}))


if __name__ == '__main__':
    main()
