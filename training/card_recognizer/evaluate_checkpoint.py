#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pipeline.checkpoints import load_lightning_or_state_dict
from pipeline.config import write_json
from pipeline.data import CardRecognizerDataModule
from pipeline.evaluation import (
    app_threshold_report_payload,
    classification_report_payload,
    collect_outputs,
    save_confusion_matrix,
)
from pipeline.experiments import (
    DEFAULT_EXPERIMENTS_ROOT,
    checkpoint_for_experiment,
    experiment_dir,
    load_experiment_config,
    write_experiment_metadata,
)
from pipeline.paths import resolve_path


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a card recognizer checkpoint.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID under the experiments root.")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--checkpoint-kind", default="best", choices=["best", "last", "latest"])
    parser.add_argument(
        "--dataset",
        default="val",
        choices=["train", "val"],
        help="val points to the manual phone-image folder.",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--target-precision", default=None, type=float)
    parser.add_argument("--min-coverage", default=None, type=float)
    args = parser.parse_args()

    run_dir = experiment_dir(args.experiment_id, root=args.experiments_root)
    cfg = load_experiment_config(args.experiment_id, args.experiments_root)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else checkpoint_for_experiment(run_dir, args.checkpoint_kind)
    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir / "reports" / "eval"
    print(f"Evaluating experiment {args.experiment_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    print("Preparing datamodule...")
    datamodule = CardRecognizerDataModule(cfg)
    datamodule.setup()
    print(f"Known classes: {datamodule.num_classes}")
    print("Loading model...")
    model = load_lightning_or_state_dict(checkpoint_path, cfg, datamodule.num_classes)
    device = choose_device()
    print(f"Using device: {device}")

    requested_dataset = args.dataset
    dataset_name = requested_dataset
    outputs = collect_outputs(model, datamodule.dataset_dataloader(dataset_name), device, desc=f"Evaluating {dataset_name}")
    eval_cfg = cfg.get("evaluation", {})
    target_precision = float(args.target_precision if args.target_precision is not None else eval_cfg.get("app_target_precision", 0.95))
    min_coverage = float(args.min_coverage if args.min_coverage is not None else eval_cfg.get("app_min_coverage", 0.5))
    report = {
        "experiment_id": args.experiment_id,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "classification": classification_report_payload(outputs, datamodule.idx_to_label),
        "app_thresholds": app_threshold_report_payload(
            outputs,
            confidence_thresholds=eval_cfg.get("softmax_thresholds", [0.0, 0.03, 0.05, 0.1, 0.5]),
            margin_thresholds=eval_cfg.get("margin_thresholds", [0.0, 0.01, 0.03, 0.05, 0.1]),
            target_precision=target_precision,
            min_coverage=min_coverage,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, output_dir / f"{dataset_name}_classification_report.json")
    save_confusion_matrix(outputs, datamodule.labels, output_dir / f"{dataset_name}_confusion_matrix")
    write_experiment_metadata(
        run_dir,
        {"last_evaluated_dataset": dataset_name, "last_eval_report": str(output_dir / f"{dataset_name}_classification_report.json")},
    )
    print(f"Wrote report to {output_dir / f'{dataset_name}_classification_report.json'}")


if __name__ == "__main__":
    main()
