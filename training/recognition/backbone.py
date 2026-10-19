"""Pinned DINOv2 backbone loading and artifact hashing for retrieval training/export."""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ID = 'facebook/dinov2-small'
MODEL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
# config.json + model.safetensors of MODEL_REVISION (DVC: models/pretrained/dinov2-small.dvc).
PRETRAINED_DIR = Path(__file__).resolve().parents[2] / 'models/pretrained/dinov2-small'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dinov2(model_path: str | Path = PRETRAINED_DIR, weights_sha256: str | None = None):
    """Load only the official built-in Transformers class and local pinned files."""
    from transformers import Dinov2Model
    if weights_sha256 and sha256_file(Path(model_path) / 'model.safetensors') != weights_sha256:
        raise ValueError(f'Pretrained weights in {model_path} differ from the pinned initialization')
    model = Dinov2Model.from_pretrained(
        str(model_path), local_files_only=True, use_safetensors=True,
        trust_remote_code=False,
    )
    model.eval()
    return model
