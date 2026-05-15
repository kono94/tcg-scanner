#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - user-facing dependency guard
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "training/card_recognizer/configs/mobilenetv3_classifier.yaml"
EXPERIMENTS_ROOT = REPO_ROOT / "runs/card_recognizer/experiments"
APP_COREML_MODEL = REPO_ROOT / "tcg-scanner-app/tcg-scanner-app/Models/card_recognizer.mlpackage"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def ask_bool(prompt: str, default: bool = True) -> bool:
    label = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{label}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def choose(prompt: str, options: list[tuple[str, str]]) -> str:
    print(f"\n{prompt}")
    for idx, (_, label) in enumerate(options, start=1):
        print(f"  {idx}. {label}")
    while True:
        value = ask("Choose", "1")
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        print("Pick one of the listed numbers.")


def require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for config editing. Run: pip install -r requirements.txt")


def load_yaml(path: Path) -> dict[str, Any]:
    require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def experiment_id_default() -> str:
    return str(int(time.time()))


def experiment_dir(experiment_id: str) -> Path:
    return EXPERIMENTS_ROOT / experiment_id


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_experiments() -> list[str]:
    if not EXPERIMENTS_ROOT.exists():
        return []
    return sorted([path.name for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()], reverse=True)


def pick_experiment(prompt: str = "Experiment ID") -> str:
    experiments = list_experiments()
    if experiments:
        print("\nRecent experiments:")
        for experiment_id in experiments[:8]:
            describe_experiment(experiment_id, compact=True)
    default = experiments[0] if experiments else ""
    return ask(prompt, default)


def command_string(cmd: list[str]) -> str:
    return " ".join(cmd)


def run(cmd: list[str], *, confirm: bool = True) -> bool:
    print(f"\n$ {command_string(cmd)}")
    if confirm and not ask_bool("Run this command?", True):
        print("Skipped.")
        return False
    print("\nStarting command. Output below is from the subprocess.")
    print("Press Ctrl-C only if you want to abort the running command.")
    start = time.monotonic()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start
        print(f"\nCommand interrupted after {elapsed:.1f}s.")
        input("Press Enter to return to the main menu...")
        return False

    elapsed = time.monotonic() - start
    if result.returncode == 0:
        print(f"\nCommand finished successfully in {elapsed:.1f}s.")
    else:
        print(f"\nCommand failed with exit code {result.returncode} after {elapsed:.1f}s.")
    print("Returning to the main menu.")
    input("Press Enter to continue...")
    return result.returncode == 0


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def deep_update(base: dict[str, Any], path: list[str], value: Any) -> None:
    target = base
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def make_training_config_interactively(base_config: Path, experiment_id: str) -> Path:
    cfg = load_yaml(base_config)
    dataset = cfg.setdefault("dataset", {})
    trainer = cfg.setdefault("trainer", {})
    data = cfg.setdefault("data", {})
    optimizer = cfg.setdefault("optimizer", {})

    print("\nTraining configuration. Press Enter to keep each default.")
    dataset["cards_dir"] = ask("Cards directory", str(dataset.get("cards_dir", "datasets/card_recognizer/cards")))
    dataset["manual_eval_dir"] = ask(
        "Manual phone-image validation folder",
        str(dataset.get("manual_eval_dir", "datasets/card_recognizer/manual_eval")),
    )
    dataset["eval_required"] = True

    trainer["max_epochs"] = int(ask("Max epochs", str(trainer.get("max_epochs", 20))))
    data["batch_size"] = int(ask("Batch size", str(data.get("batch_size", 32))))
    data["num_workers"] = int(ask("DataLoader workers", str(data.get("num_workers", 0))))
    optimizer["lr"] = float(ask("Learning rate", str(optimizer.get("lr", 0.0003))))

    tmp_dir = Path(tempfile.gettempdir()) / "tcg-scanner-cli"
    config_path = tmp_dir / f"{experiment_id}.yaml"
    write_yaml(config_path, cfg)
    print(f"Prepared temporary training config: {config_path}")
    return config_path


def train_recognizer(full_pipeline: bool = False) -> str | None:
    base_config = Path(ask("Base YAML config", rel(DEFAULT_CONFIG)))
    if not base_config.is_absolute():
        base_config = REPO_ROOT / base_config
    experiment_id = ask("Experiment ID", experiment_id_default())
    config_path = make_training_config_interactively(base_config, experiment_id)
    ok = run(
        python_cmd(
            "training/card_recognizer/train_lightning.py",
            "--config",
            str(config_path),
            "--experiment-id",
            experiment_id,
        )
    )
    if not ok:
        return None

    if full_pipeline:
        post_training_flow(experiment_id)
    return experiment_id


def post_training_flow(experiment_id: str) -> None:
    if ask_bool("Evaluate classifier report now?", True):
        evaluate_classifier(experiment_id)
    if ask_bool("Export CoreML recognizer now?", False):
        export_coreml(experiment_id)


def evaluate_classifier(experiment_id: str | None = None) -> None:
    experiment_id = experiment_id or pick_experiment()
    dataset = ask("Dataset", "val")
    if run(python_cmd("training/card_recognizer/evaluate_checkpoint.py", "--experiment-id", experiment_id, "--dataset", dataset)):
        summarize_classification_report(experiment_id, dataset)


def export_coreml(experiment_id: str | None = None) -> None:
    experiment_id = experiment_id or pick_experiment()
    install = ask_bool("Install into the iOS app bundle and stamp manifest?", False)
    cmd = python_cmd("scripts/export_recognizer_coreml.py", "--experiment-id", experiment_id)
    if install:
        cmd.append("--install-to-app")
    run(cmd)


def benchmark_latency() -> None:
    experiment_id = pick_experiment()
    cmd = python_cmd("training/card_recognizer/benchmark_inference.py", "--experiment-id", experiment_id)
    if ask_bool("Also benchmark current app CoreML model on this host?", True):
        cmd.extend(["--coreml-model", rel(APP_COREML_MODEL)])
    if run(cmd):
        path = experiment_dir(experiment_id) / "reports/latency_report.json"
        if path.exists():
            report = load_json(path)
            print("\nLatency report:")
            for row in report.get("results", []):
                if "error" in row:
                    print(f"  {row.get('backend')}: {row['error']}")
                else:
                    print(f"  {row.get('backend')}: {row.get('mean_ms', 0):.2f} ms")


def compare_experiments() -> None:
    old_id = pick_experiment("Baseline experiment ID")
    new_id = pick_experiment("Candidate experiment ID")
    dataset = ask("Dataset", "val")
    if run(
        python_cmd(
            "training/card_recognizer/regression_report.py",
            "--old-experiment-id",
            old_id,
            "--new-experiment-id",
            new_id,
            "--dataset",
            dataset,
        )
    ):
        report_path = experiment_dir(new_id) / "reports" / f"regression_vs_{old_id}.json"
        if report_path.exists():
            report = load_json(report_path)
            print("\nRegression metrics:")
            for row in report.get("metrics", []):
                delta = row.get("delta")
                delta_text = "-" if delta is None else f"{delta:+.4f}"
                print(f"  {row['metric']}: old={row['old']} new={row['new']} delta={delta_text}")


def add_new_one_piece_set() -> None:
    print("\nAdd New One Piece Set")
    print("Place card images and metadata here:")
    set_code = ask("New set code", "OPXX").upper()
    set_dir = REPO_ROOT / "cards" / set_code
    print(f"  {rel(set_dir)}/")
    print("Expected files look like:")
    print(f"  {set_code}-001.jpg")
    print(f"  {set_code}-001.json")
    if not set_dir.exists() and ask_bool("Create this directory now?", True):
        set_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {rel(set_dir)}")

    images = list(set_dir.glob("*.jpg")) + list(set_dir.glob("*.jpeg")) + list(set_dir.glob("*.png")) + list(set_dir.glob("*.webp"))
    jsons = list(set_dir.glob("*.json"))
    print(f"\nCurrent files in {rel(set_dir)}: {len(images)} images, {len(jsons)} json metadata files.")

    print("\nClassifier-first policy:")
    print("  New card sets are supported by retraining the full classifier on the complete corpus.")
    print("  Validation uses your manually captured phone images, not automatic holdout data.")
    print("  Add scraped card images and metadata under datasets/card_recognizer/cards/<set-code>/")
    print("  Add new phone images under datasets/card_recognizer/manual_eval/<card-id>/ before training.")
    print("  After training, compare the new experiment against the previous best experiment on that same validation set.")
    if not ask_bool("Train a new full classifier experiment now?", True):
        return

    baseline = pick_experiment("Baseline experiment ID to compare against after retraining")
    new_experiment_id = train_recognizer(full_pipeline=False)
    if new_experiment_id is None:
        return
    if ask_bool("Evaluate the new classifier now?", True):
        evaluate_classifier(new_experiment_id)
    if baseline and ask_bool("Compare the new classifier against the baseline?", True):
        run(
            python_cmd(
                "training/card_recognizer/regression_report.py",
                "--old-experiment-id",
                baseline,
                "--new-experiment-id",
                new_experiment_id,
                "--dataset",
                "val",
            )
        )


def refresh_app_resources() -> None:
    if ask_bool("Refresh bundled price snapshot?", True):
        run(python_cmd("scripts/scrape_prices.py"))
    if ask_bool("Rebuild compact iOS card metadata index?", True):
        run(python_cmd("scripts/build_card_index.py"))


def ios_workflow() -> None:
    action = choose(
        "iOS workflow",
        [
            ("build", "Build app for generic iOS device"),
            ("test", "Run simulator unit tests"),
        ],
    )
    if action == "build":
        run(
            [
                "xcodebuild",
                "-project",
                "tcg-scanner-app/tcg-scanner-app.xcodeproj",
                "-scheme",
                "tcg-scanner-app",
                "-destination",
                "generic/platform=iOS",
                "build",
            ]
        )
    else:
        destination = ask("Simulator destination", "platform=iOS Simulator,name=iPhone 15,OS=17.4")
        run(
            [
                "xcodebuild",
                "-project",
                "tcg-scanner-app/tcg-scanner-app.xcodeproj",
                "-scheme",
                "tcg-scanner-app",
                "-destination",
                destination,
                "test",
            ]
        )


def summarize_classification_report(experiment_id: str, dataset: str = "val") -> None:
    path = experiment_dir(experiment_id) / "reports/eval" / f"{dataset}_classification_report.json"
    if not path.exists():
        print(f"No classification report found at {rel(path)}")
        return
    report = load_json(path)
    metrics = report.get("classification", {})
    print("\nClassifier report summary:")
    for key in ["count", "top1", "top5", "nll", "brier", "ece"]:
        if key in metrics:
            value = metrics[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    per_set = metrics.get("per_set", {})
    if per_set:
        print("  per-set top1:")
        for set_code, row in sorted(per_set.items())[:12]:
            print(f"    {set_code}: {row.get('top1', 0):.4f} ({row.get('count', 0)} cards)")


def describe_experiment(experiment_id: str, *, compact: bool = False) -> None:
    run_dir = experiment_dir(experiment_id)
    metadata_path = run_dir / "experiment.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    status = metadata.get("status", "unknown")
    score = metadata.get("best_score")
    score_text = f" best={score:.4f}" if isinstance(score, float) else ""
    print(f"  {experiment_id}: {status}{score_text} ({rel(run_dir)})")
    if compact:
        return
    if metadata:
        print(json.dumps(metadata, indent=2))
    for path in [
        run_dir / "reports/eval/val_classification_report.json",
        run_dir / "reports/latency_report.json",
    ]:
        if path.exists():
            print(f"  report: {rel(path)}")


def inspect_experiments() -> None:
    experiments = list_experiments()
    if not experiments:
        print("No experiments found yet.")
        return
    for experiment_id in experiments[:20]:
        describe_experiment(experiment_id, compact=True)
    if ask_bool("Open details for one experiment?", True):
        describe_experiment(pick_experiment())


def main_menu() -> None:
    print("\nTCG Scanner CLI")
    print("A guided wrapper for training, evaluation, model export, data updates, and iOS checks.")
    while True:
        action = choose(
            "Top-level workflows",
            [
                ("full_train", "Train One Piece classifier, then optionally evaluate and export"),
                ("train", "Train One Piece classifier only"),
                ("add_set", "Add/evaluate a new One Piece set"),
                ("eval", "Evaluate classifier report for an experiment"),
                ("export", "Export/install recognizer CoreML package"),
                ("compare", "Compare two experiments"),
                ("latency", "Run recognizer latency benchmark"),
                ("resources", "Refresh app card metadata and price resources"),
                ("ios", "Build or test iOS app"),
                ("inspect", "List/inspect experiments"),
                ("quit", "Quit"),
            ],
        )
        if action == "quit":
            return
        try:
            if action == "full_train":
                train_recognizer(full_pipeline=True)
            elif action == "train":
                train_recognizer(full_pipeline=False)
            elif action == "add_set":
                add_new_one_piece_set()
            elif action == "eval":
                evaluate_classifier()
            elif action == "export":
                export_coreml()
            elif action == "compare":
                compare_experiments()
            elif action == "latency":
                benchmark_latency()
            elif action == "resources":
                refresh_app_resources()
            elif action == "ios":
                ios_workflow()
            elif action == "inspect":
                inspect_experiments()
        except KeyboardInterrupt:
            print("\nInterrupted workflow.")
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main_menu()
