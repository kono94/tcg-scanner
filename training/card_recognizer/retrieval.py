"""Frozen descriptor and exact cosine search baseline, shared across card games."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from training.card_recognizer.canonical import (
    canonicalize_card, prepare_encoder_input, preprocessing_contract_hash,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Descriptors must be a finite 2-D matrix')
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError('Zero descriptors cannot be normalized')
    return vectors / lengths


class FrozenCardEncoder:
    """ImageNet MobileNet features; an uncalibrated card-retrieval baseline, not a trained TCG encoder."""

    def __init__(self, weights: Path, device: str = 'cpu'):
        import torch
        import torchvision
        from torchvision.models import mobilenet_v3_large

        self.torch = torch
        self.device = device
        model = mobilenet_v3_large(weights=None)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        # Global pooled backbone features, before the ImageNet classifier/projection.
        model.classifier = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.model.requires_grad_(False)
        self.contract = {
            'encoder': 'torchvision_mobilenet_v3_large_imagenet1k_v2_pooled_backbone',
            'weights_sha256': file_sha256(weights), 'dimension': 960,
            'preprocessing_contract_sha256': preprocessing_contract_hash(),
            'normalization': 'L2', 'dtype': 'float32',
            'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
            'card_training': 'none', 'card_threshold_calibration': 'none',
        }
        payload = json.dumps(self.contract, sort_keys=True, separators=(',', ':')).encode()
        self.contract_hash = hashlib.sha256(payload).hexdigest()

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 960), dtype=np.float32)
        tensors = np.stack([prepare_encoder_input(canonicalize_card(image)) for image in images])
        with self.torch.inference_mode():
            result = self.model(self.torch.from_numpy(tensors).to(self.device)).cpu().numpy()
        return normalized(result)


class CosineGallery:
    def __init__(self, vectors: np.ndarray, references: list[dict], encoder_contract_hash: str):
        self.vectors = normalized(vectors)
        if len(references) != len(self.vectors) or not references:
            raise ValueError('Nonempty references and descriptor rows must match')
        if any(not isinstance(r.get('identity'), str) or ':' not in r['identity'] for r in references):
            raise ValueError('References need namespaced game:identity keys')
        self.references = references
        self.encoder_contract_hash = encoder_contract_hash

    def search(self, query: np.ndarray, encoder_contract_hash: str, k: int = 5) -> list[list[dict]]:
        if encoder_contract_hash != self.encoder_contract_hash:
            raise ValueError('Query/gallery encoder-preprocessing contract mismatch')
        if k < 1:
            raise ValueError('k must be positive')
        query = normalized(query)
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError('Descriptor dimensions do not match')
        scores = query @ self.vectors.T
        results = []
        for row in scores:
            distinct, seen = [], set()
            for index in np.argsort(-row, kind='stable'):
                reference = self.references[index]
                identity = reference['identity']
                if identity in seen:
                    continue
                seen.add(identity)
                distinct.append({**reference, 'similarity': float(row[index])})
                if len(distinct) == k:
                    break
            results.append(distinct)
        return results

    @classmethod
    def load(cls, root: Path) -> 'CosineGallery':
        manifest = json.loads((root / 'manifest.json').read_text())
        contract = manifest['encoder_contract']
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != manifest['encoder_contract_sha256']:
            raise ValueError('Gallery encoder contract checksum mismatch')
        path = root / 'vectors.npy'
        if file_sha256(path) != manifest['vectors_sha256']:
            raise ValueError('Gallery descriptor artifact checksum mismatch')
        vectors = np.load(path, allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[1] != contract['dimension']:
            raise ValueError('Gallery descriptor dimension does not match encoder contract')
        return cls(vectors, manifest['references'], manifest['encoder_contract_sha256'])
