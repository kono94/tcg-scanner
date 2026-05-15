#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from pipeline.checkpoints import load_lightning_or_state_dict
from pipeline.config import write_json
from pipeline.data import CardRecognizerDataModule
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


def benchmark_torch(model, device: torch.device, iterations: int, warmup: int) -> dict:
    print(f"Benchmarking PyTorch on {device}...")
    model.to(device)
    model.eval()
    image = torch.rand(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc="Torch warmup", unit="iter"):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in tqdm(range(iterations), desc="Torch benchmark", unit="iter"):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "backend": f"torch:{device.type}",
        "iterations": iterations,
        "mean_ms": elapsed * 1000.0 / iterations,
    }


def benchmark_coreml(model_path: Path, iterations: int, warmup: int) -> dict:
    import coremltools as ct
    from PIL import Image

    print(f"Benchmarking CoreML host inference: {model_path}")
    model = ct.models.MLModel(model_path)
    image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    for _ in tqdm(range(warmup), desc="CoreML warmup", unit="iter"):
        model.predict({"image": image})
    start = time.perf_counter()
    for _ in tqdm(range(iterations), desc="CoreML benchmark", unit="iter"):
        model.predict({"image": image})
    elapsed = time.perf_counter() - start
    return {
        "backend": "coreml:host",
        "model_path": str(model_path),
        "iterations": iterations,
        "mean_ms": elapsed * 1000.0 / iterations,
        "note": "Host CoreML timing is not a substitute for physical iPhone latency.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark recognizer inference latency.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID under the experiments root.")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--checkpoint-kind", default="best", choices=["best", "last", "latest"])
    parser.add_argument("--coreml-model", default=None, type=Path)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    run_dir = experiment_dir(args.experiment_id, root=args.experiments_root)
    cfg = load_experiment_config(args.experiment_id, args.experiments_root)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else checkpoint_for_experiment(run_dir, args.checkpoint_kind)
    output_path = resolve_path(args.output) if args.output else run_dir / "reports" / "latency_report.json"
    print(f"Running latency benchmark for experiment {args.experiment_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    print("Preparing datamodule...")
    datamodule = CardRecognizerDataModule(cfg)
    datamodule.setup()
    print(f"Known classes: {datamodule.num_classes}")
    print("Loading model...")
    model = load_lightning_or_state_dict(checkpoint_path, cfg, datamodule.num_classes)
    results = [benchmark_torch(model, choose_device(), args.iterations, args.warmup)]

    if args.coreml_model is not None:
        try:
            results.append(benchmark_coreml(resolve_path(args.coreml_model), args.iterations, args.warmup))
        except Exception as exc:
            results.append({"backend": "coreml:host", "error": str(exc)})

    report = {"experiment_id": args.experiment_id, "checkpoint": str(checkpoint_path), "results": results}
    write_json(report, output_path)
    write_experiment_metadata(run_dir, {"latency_report": str(output_path)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
