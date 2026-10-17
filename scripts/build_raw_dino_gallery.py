#!/usr/bin/env python3
"""Stream catalog fronts into a frozen DINO cosine index without duplicating crop files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input
from training.card_recognizer.foundation_retrieval import contract_metadata, load_dinov2
from training.card_recognizer.retrieval import CosineGallery, file_sha256, normalized


def catalog_fronts(specifications: list[str], output: Path) -> list[dict]:
    output = output.resolve()
    references = []
    identities = {}
    for specification in specifications:
        namespace, separator, location = specification.partition('=')
        if not separator or not re.fullmatch(r'[a-z][a-z0-9_]*:[a-z][a-z0-9-]*', namespace):
            raise ValueError('Catalog specification must be game:language=/path/to/cards')
        game, language = namespace.split(':')
        root = Path(location).resolve()
        if not root.is_dir() or root == output or root in output.parents or output in root.parents:
            raise ValueError('Catalog must exist separately from output')
        for sidecar in sorted(root.rglob('*.json')):
            data = json.loads(sidecar.read_text())
            if not isinstance(data, dict):
                continue
            card_id = data.get('catalog_id') or data.get('provider_id') or data.get('id')
            if not isinstance(card_id, str):
                continue
            if data.get('game', game) != game or data.get('language', language) != language:
                raise ValueError(f'Catalog namespace mismatch: {sidecar}')
            if data.get('domain') not in (None, 'pokemon_physical_tcg'):
                continue
            if data.get('image_status', 'downloaded') != 'downloaded':
                continue
            paths = [sidecar.with_suffix(suffix) for suffix in ('.webp', '.jpg', '.jpeg', '.png')
                     if sidecar.with_suffix(suffix).is_file()]
            if len(paths) != 1 or not paths[0].resolve().is_relative_to(root):
                raise ValueError(f'Expected one in-root front image for {sidecar}')
            identity = f'{namespace}:{card_id}'
            path = paths[0].resolve()
            digest = file_sha256(path)
            if data.get('image_sha256') and data['image_sha256'] != digest:
                raise ValueError(f'Source image checksum mismatch: {path}')
            if identity in identities:
                prior = identities[identity]
                if prior['source_sha256'] != digest:
                    raise ValueError(f'Duplicate catalog identity has conflicting image bytes: {identity}')
                prior.setdefault('exact_source_aliases', []).append({'source_path': str(path),
                    'source_sidecar': str(sidecar), 'sidecar_sha256': file_sha256(sidecar)})
                continue
            references.append({'identity': identity, 'game': game, 'language': language,
                'card_id': card_id, 'name': data.get('name'), 'source_path': str(path),
                'source_sha256': digest, 'source_sidecar': str(sidecar),
                'sidecar_sha256': file_sha256(sidecar), 'source_url': data.get('image_url'),
                'physical_finish': data.get('physical_finish', 'unresolved'),
                'reference_role': 'catalog_digital_front_not_independent_query'})
            identities[identity] = references[-1]
    if not references:
        raise ValueError('No downloaded catalog fronts found')
    return references


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty')
    if args.batch < 1 or args.threads < 1:
        raise ValueError('Batch and threads must be positive')
    references = catalog_fronts(args.catalog, output)
    weight_hash = file_sha256(args.model_dir / 'model.safetensors')
    config_hash = file_sha256(args.model_dir / 'config.json')
    if (weight_hash != 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1' or
            config_hash != '1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'):
        raise ValueError('Expected the pinned official DINOv2-small model/config bytes')
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    import transformers
    torch.set_num_threads(args.threads)
    model = load_dinov2(args.model_dir).requires_grad_(False)
    contract = contract_metadata(weight_hash, config_hash,
                                 transformers.__version__, torch.__version__)
    digest = contract.pop('contract_sha256')
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(output / 'vectors.npy', mode='w+', dtype=np.float32,
                                       shape=(len(references), 384))
    with torch.inference_mode():
        for start in range(0, len(references), args.batch):
            tensors = []
            for ref in references[start:start + args.batch]:
                with Image.open(ref['source_path']) as opened:
                    rgb = np.asarray(opened.convert('RGB'))
                tensors.append(prepare_encoder_input(canonicalize_card(rgb)))
            encoded = model(pixel_values=torch.from_numpy(np.stack(tensors))).last_hidden_state[:, 0, :]
            matrix[start:start + len(tensors)] = normalized(encoded.cpu().numpy())
            if start == 0 or (start // args.batch) % 25 == 0:
                print(f'Encoded {start + len(tensors)}/{len(references)} catalog fronts', flush=True)
    matrix.flush()
    CosineGallery(matrix, references, digest)
    del matrix
    manifest = {'artifact_type': 'frozen_card_cosine_gallery', 'encoder_contract': contract,
        'encoder_contract_sha256': digest, 'vectors_sha256': file_sha256(output / 'vectors.npy'),
        'references': references, 'identities': len(references),
        'source_policy': 'downloaded catalog sidecars and verified full-front image bounds; streamed canonicalization',
        'canonical_crop_files_written': False, 'real_camera_validation': False,
        'acceptance_thresholds_calibrated': False, 'card_training': 'none',
        'catalog_completion_claim': 'snapshot of downloaded fronts, not every provider record or physical printing'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='append', required=True)
    parser.add_argument('--model-dir', type=Path, required=True, help='Local official pinned DINOv2-small snapshot')
    parser.add_argument('--runtime', type=Path, default=ROOT / 'runs/foundation_teacher/runtime')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    try:
        manifest = build(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Indexed {manifest['identities']} catalog identities", flush=True)


if __name__ == '__main__':
    main()
