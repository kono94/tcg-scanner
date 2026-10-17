"""Frozen DINOv2-small descriptors under the shared card preprocessing contract."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from training.card_recognizer.canonical import canonicalize_card, prepare_encoder_input, preprocessing_contract_hash


MODEL_ID = 'facebook/dinov2-small'
MODEL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
POOLING = 'last_hidden_state CLS token, L2-normalized'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_card_path(path: str | Path, rotate_180: bool = False) -> np.ndarray:
    """Load an already-canonical RGB full-card crop and apply shared normalized 224 fit/pad."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        if rotate_180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        rgb = np.asarray(image, dtype=np.uint8)
    if rgb.shape[:2] != (312, 224):
        raise ValueError(f'canonical full-card crop must be 224x312; got {rgb.shape[1]}x{rgb.shape[0]} from {path}')
    # Both gallery and synthetic query files are already warped full-card crops under
    # tcg-full-card-v1; re-warping them as an identity transform would alter pixels.
    return prepare_encoder_input(rgb, size=224)


def prepare_detected_crop_path(path: str | Path, rotate_180: bool = False) -> np.ndarray:
    """Apply full-bounds canonicalization then shared fit/pad to an unrectified detector crop."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        if rotate_180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        rgb = np.asarray(image, dtype=np.uint8)
    canonical = canonicalize_card(rgb, corners=None, width=224, height=312)
    return prepare_encoder_input(canonical, size=224)


def load_dinov2(model_path: str | Path):
    """Load only the official built-in Transformers class and local pinned files."""
    try:
        from transformers import Dinov2Model
    except ImportError as exc:
        raise RuntimeError('Transformers Dinov2Model is unavailable in the selected runtime') from exc
    model = Dinov2Model.from_pretrained(
        str(model_path), local_files_only=True, use_safetensors=True,
        trust_remote_code=False,
    )
    model.eval()
    return model


def encode_paths(model, paths: Iterable[str | Path], batch_size: int = 16,
                 num_threads: int = 4, rotate_180: bool = False,
                 unrectified_crop: bool = False) -> tuple[np.ndarray, float]:
    """Encode card files in bounded CPU batches, returning CLS features and seconds."""
    import time
    import torch
    import torch.nn.functional as functional

    if batch_size < 1 or num_threads < 1:
        raise ValueError('batch_size and num_threads must be positive')
    torch.set_num_threads(num_threads)
    inputs = [Path(path) for path in paths]
    if not inputs:
        return np.empty((0, 384), dtype=np.float32), 0.0
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            prepare = prepare_detected_crop_path if unrectified_crop else prepare_card_path
            batch = np.stack([prepare(path, rotate_180=rotate_180)
                              for path in inputs[start:start + batch_size]])
            pixel_values = torch.from_numpy(batch)
            # Transformers 4.46 Dinov2Embeddings interpolates its positional grid
            # internally for this shared 224x224 card tensor size.
            result = model(pixel_values=pixel_values)
            cls = functional.normalize(result.last_hidden_state[:, 0, :].float(), p=2, dim=1)
            outputs.append(cls.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False), time.perf_counter() - started


def contract_metadata(weights_sha256: str, config_sha256: str, transformers_version: str,
                      torch_version: str) -> dict:
    """Describe the complete frozen feature contract; thresholds remain uncalibrated."""
    import json
    from training.card_recognizer.canonical import preprocessing_contract

    contract = {
        'model_id': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'weights_sha256': weights_sha256,
        'config_sha256': config_sha256,
        'dimension': 384,
        'pooling': POOLING,
        'pooling_sha256': hashlib.sha256(POOLING.encode()).hexdigest(),
        'normalization': 'L2 float32',
        'card_training': 'none',
        'preprocessing': preprocessing_contract(),
        'preprocessing_contract_sha256': preprocessing_contract_hash(),
        'transformers_version': transformers_version,
        'torch_version': torch_version,
    }
    raw = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
    contract['contract_sha256'] = hashlib.sha256(raw).hexdigest()
    return contract
