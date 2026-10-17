"""Pinned DINOv2 backbone loading and artifact hashing for retrieval training/export."""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ID = 'facebook/dinov2-small'
MODEL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



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
