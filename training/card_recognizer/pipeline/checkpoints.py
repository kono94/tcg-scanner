from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from .lightning_module import RecognizerLightningModule
from .paths import resolve_path


def normalize_model_state_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        normalized[key.replace("embedding_layer.", "feature_layer.")] = value
    return normalized


def load_lightning_or_state_dict(checkpoint_path: str | Path, cfg: dict[str, Any], num_classes: int, map_location="cpu"):
    path = resolve_path(checkpoint_path)
    raw = torch.load(path, map_location=map_location, weights_only=False)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.setdefault("model", {})["pretrained"] = False
    module = RecognizerLightningModule(eval_cfg, num_classes)

    if isinstance(raw, dict) and "state_dict" in raw:
        module.load_state_dict(normalize_model_state_keys(raw["state_dict"]))
    elif isinstance(raw, dict):
        state = normalize_model_state_keys(raw)
        if any(key.startswith("model.") for key in state):
            module.load_state_dict(state, strict=False)
        else:
            module.model.load_state_dict(state)
    else:
        raise TypeError(f"Unsupported checkpoint format: {path}")

    module.eval()
    return module
