#!/usr/bin/env python3
"""Create a namespaced, multi-game exact-cosine index from prepared reference crops."""
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
from training.card_recognizer.canonical import preprocessing_contract_hash
from training.card_recognizer.retrieval import FrozenCardEncoder, CosineGallery, file_sha256


def build(args: argparse.Namespace) -> dict:
    if args.batch < 1 or args.threads < 1:
        raise ValueError('batch and threads must be positive')
    if not re.fullmatch('[a-z][a-z0-9-]*', args.language):
        raise ValueError('Invalid language namespace')
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty')
    pending, manifests = [], []
    for spec in args.gallery:
        game, separator, location = spec.partition('=')
        if not separator or not re.fullmatch('[a-z][a-z0-9_]*', game):
            raise ValueError('--gallery must be game=/path/to/prepared/gallery')
        root = Path(location).resolve()
        if root == output or root in output.parents or output in root.parents:
            raise ValueError('Input and output directories must not contain each other')
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['preprocessing_contract_sha256'] != preprocessing_contract_hash():
            raise ValueError('Prepared images use an incompatible preprocessing contract')
        manifests.append({'game': game, 'path': str(manifest_path), 'sha256': file_sha256(manifest_path)})
        for reference in manifest['references']:
            path = (root / reference['canonical_crop']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Invalid canonical reference path')
            if file_sha256(path) != reference['canonical_sha256']:
                raise ValueError(f'Canonical reference checksum mismatch: {path}')
            pending.append((path, {**reference, 'game': game, 'language': args.language,
                                   'identity': f"{game}:{args.language}:{reference['card_id']}",
                                   'canonical_path': str(path)}))
    if not pending:
        raise ValueError('No references to encode')
    import torch
    torch.set_num_threads(args.threads)
    encoder = FrozenCardEncoder(args.weights, args.device)
    vectors = []
    for start in range(0, len(pending), args.batch):
        images = []
        for path, _ in pending[start:start + args.batch]:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert('RGB')))
        vectors.append(encoder.encode(images))
        print(f'Encoded {min(start + args.batch, len(pending))}/{len(pending)} references', flush=True)
    matrix = np.concatenate(vectors)
    references = [r for _, r in pending]
    CosineGallery(matrix, references, encoder.contract_hash)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'vectors.npy', matrix, allow_pickle=False)
    manifest = {'artifact_type': 'frozen_card_cosine_gallery', 'encoder_contract': encoder.contract,
                'encoder_contract_sha256': encoder.contract_hash, 'vectors_sha256': file_sha256(output / 'vectors.npy'),
                'source_manifests': manifests, 'references': references,
                'identities': len({r['identity'] for r in references}),
                'real_camera_validation': False, 'acceptance_thresholds_calibrated': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', action='append', required=True, help='Repeatable game=/path/to/prepared/gallery')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--language', default='en', help='Language namespace for these prepared reference inputs')
    args = parser.parse_args()
    try:
        manifest = build(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Indexed {len(manifest['references'])} references / {manifest['identities']} namespaced identities")


if __name__ == '__main__':
    main()
