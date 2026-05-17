from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, resolve_path


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "mobilenetv3_classifier",
        "seed": 42,
        "output_dir": "runs/card_recognizer/experiments",
    },
    "dataset": {
        "cards_dir": "datasets/card_recognizer/cards",
        "manual_eval_dir": "datasets/card_recognizer/manual_eval",
        "eval_required": True,
    },
    "model": {
        "feature_dim": 256,
        "pretrained": True,
    },
    "data": {
        "batch_size": 32,
        "num_workers": 0,
        "pin_memory": True,
    },
    "augmentation": {
        "resize": 256,
        "crop_size": 224,
        "random_resized_crop_scale": [0.9, 1.0],
        "color_jitter": {
            "brightness": 0.35,
            "contrast": 0.35,
            "saturation": 0.2,
            "hue": 0.03,
        },
        "random_affine": {
            "degrees": 5,
            "translate": [0.04, 0.04],
            "scale": [0.97, 1.04],
        },
        "random_perspective": {
            "distortion_scale": 0.12,
            "p": 0.3,
        },
        "random_erasing": {
            "p": 0.25,
            "scale": [0.01, 0.06],
        },
    },
    "optimizer": {
        "name": "adamw",
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "backbone_lr_multiplier": 0.1,
    },
    "scheduler": {
        "name": "cosine",
        "warmup_epochs": 1,
        "min_lr": 0.000001,
    },
    "trainer": {
        "max_epochs": 20,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "32-true",
        "deterministic": True,
        "log_every_n_steps": 10,
        "check_val_every_n_epoch": 1,
    },
    "tracking": {
        "csv": True,
        "tensorboard": False,
    },
    "evaluation": {
        "app_target_precision": 0.95,
        "app_min_coverage": 0.5,
        "softmax_thresholds": [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95],
        "margin_thresholds": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    cfg = deep_update(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(config_path)
    return cfg


def save_resolved_config(cfg: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def project_path(cfg: dict[str, Any], *keys: str) -> Path:
    value: Any = cfg
    for key in keys:
        value = value[key]
    return resolve_path(value, PROJECT_ROOT)
