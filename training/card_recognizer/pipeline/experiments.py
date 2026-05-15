from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config, write_json
from .dataset_manifest import build_dataset_manifest, write_dataset_manifest
from .paths import PROJECT_ROOT, resolve_path


DEFAULT_EXPERIMENTS_ROOT = "runs/card_recognizer/experiments"


def make_experiment_id() -> str:
    return str(int(time.time()))


def experiments_root(cfg: dict[str, Any] | None = None, root: str | Path | None = None) -> Path:
    if root is not None:
        return resolve_path(root)
    if cfg is not None:
        return resolve_path(cfg["experiment"]["output_dir"])
    return resolve_path(DEFAULT_EXPERIMENTS_ROOT)


def experiment_dir(experiment_id: str, cfg: dict[str, Any] | None = None, root: str | Path | None = None) -> Path:
    return experiments_root(cfg, root) / experiment_id


def resolved_config_path(run_dir: Path) -> Path:
    return run_dir / "resolved_config.yaml"


def metadata_path(run_dir: Path) -> Path:
    return run_dir / "experiment.json"


def load_experiment_config(experiment_id: str, root: str | Path | None = None) -> dict[str, Any]:
    run_dir = experiment_dir(experiment_id, root=root)
    return load_config(resolved_config_path(run_dir))


def load_experiment_metadata(run_dir: Path) -> dict[str, Any]:
    path = metadata_path(run_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_experiment_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    existing = load_experiment_metadata(run_dir)
    existing.update(payload)
    write_json(existing, metadata_path(run_dir))


def checkpoint_for_experiment(run_dir: Path, preferred: str = "best") -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    metadata = load_experiment_metadata(run_dir)

    if preferred == "best" and metadata.get("best_checkpoint"):
        candidate = resolve_path(metadata["best_checkpoint"])
        if candidate.exists():
            return candidate

    if preferred == "last":
        candidate = checkpoints_dir / "last.ckpt"
        if candidate.exists():
            return candidate

    ckpts = sorted(checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return ckpts[0]


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def generate_dataset_artifacts(run_dir: Path, cfg: dict[str, Any]) -> dict[str, int]:
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg["dataset"]
    print("Building experiment dataset manifest...")
    payload = build_dataset_manifest(
        cards_dir=dataset_cfg["cards_dir"],
        eval_dir=dataset_cfg["manual_eval_dir"],
        eval_required=bool(dataset_cfg.get("eval_required", True)),
    )

    manifest_path = dataset_dir / "manifest.json"
    labels_path = dataset_dir / "names.txt"
    print(f"Writing experiment dataset manifest files to {dataset_dir}...")
    write_dataset_manifest(payload, manifest_path, dataset_dir / "manifest.csv")
    labels_path.write_text("\n".join(payload["labels"]) + "\n", encoding="utf-8")

    cfg["dataset"]["manifest_path"] = project_relative(manifest_path)
    cfg["dataset"]["labels_path"] = project_relative(labels_path)
    return {name: len(records) for name, records in payload["datasets"].items()}


def create_experiment(run_dir: Path, experiment_id: str, cfg: dict[str, Any]) -> None:
    print(f"Creating experiment {experiment_id} at {run_dir}...")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        for child in ["artifacts", "checkpoints", "dataset", "logs", "reports"]:
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        dataset_counts = generate_dataset_artifacts(run_dir, cfg)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    write_experiment_metadata(
        run_dir,
        {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "created",
            "dataset_counts": dataset_counts,
            "layout": {
                "artifacts": "artifacts",
                "checkpoints": "checkpoints",
                "dataset": "dataset",
                "logs": "logs",
                "reports": "reports",
                "resolved_config": "resolved_config.yaml",
            },
        },
    )
