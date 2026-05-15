#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import write_json
from pipeline.experiments import DEFAULT_EXPERIMENTS_ROOT, experiment_dir, write_experiment_metadata
from pipeline.paths import resolve_path


def load_json(path: Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_at(report: dict, key: str):
    value = report
    for part in key.split("."):
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new recognizer evaluation reports.")
    parser.add_argument("--old-experiment-id", required=True)
    parser.add_argument("--new-experiment-id", required=True)
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT, type=Path)
    parser.add_argument("--dataset", default="val", choices=["train", "val"], help="val points to the manual phone-image folder.")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        default=["classification.top1", "classification.top5", "classification.ece", "classification.nll"],
    )
    args = parser.parse_args()

    old_run_dir = experiment_dir(args.old_experiment_id, root=args.experiments_root)
    new_run_dir = experiment_dir(args.new_experiment_id, root=args.experiments_root)
    requested_dataset = args.dataset
    dataset_name = requested_dataset
    old_report_path = old_run_dir / "reports" / "eval" / f"{dataset_name}_classification_report.json"
    new_report_path = new_run_dir / "reports" / "eval" / f"{dataset_name}_classification_report.json"
    old = load_json(old_report_path)
    new = load_json(new_report_path)
    rows = []
    for key in args.metric:
        try:
            old_value = metric_at(old, key)
            new_value = metric_at(new, key)
        except KeyError:
            continue
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            delta = new_value - old_value
        else:
            delta = None
        rows.append({"metric": key, "old": old_value, "new": new_value, "delta": delta})

    report = {
        "old_experiment_id": args.old_experiment_id,
        "new_experiment_id": args.new_experiment_id,
        "dataset": dataset_name,
        "old_report": str(old_report_path),
        "new_report": str(new_report_path),
        "metrics": rows,
    }
    if args.output:
        output_path = resolve_path(args.output)
    else:
        output_path = new_run_dir / "reports" / f"regression_vs_{args.old_experiment_id}.json"
    write_json(report, output_path)
    write_experiment_metadata(new_run_dir, {"last_regression_report": str(output_path)})
    print(f"Wrote regression report: {output_path}")


if __name__ == "__main__":
    main()
