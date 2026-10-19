#!/usr/bin/env python3
"""Train the Ultralytics YOLO-pose card-corner detector (box + 4 corner keypoints)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO-pose dataset.yaml from scripts/detection/build_corner_dataset.py")
    parser.add_argument("--model", default="yolov8s-pose.pt", help="Pretrained YOLOv8 pose weights, or a checkpoint to fine-tune")
    parser.add_argument("--from-scratch", action="store_true", help="Use the model YAML without downloading pretrained weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Ultralytics device, or auto (CUDA, then MPS, then CPU)")
    parser.add_argument("--project", type=Path, default=Path("runs/detection/pose"))
    parser.add_argument("--name", default="synthetic-baseline")
    parser.add_argument("--seed", type=int, default=42)
    # Augmentation on top of the renderer's own variation (cards appear at any rotation and mirrored layouts).
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0, help="Share of the training images to use (smoke runs)")
    # Fine-tuning controls.
    parser.add_argument("--time", type=float, default=None, help="Hard training budget in hours; overrides epochs")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else REPO_ROOT / args.data
    project_path = args.project if args.project.is_absolute() else REPO_ROOT / args.project
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data_path}")
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        raise SystemExit("epochs, imgsz, and batch must be positive; workers cannot be negative")

    # Keep Ultralytics and Matplotlib settings local to this checkout.
    config_dir = REPO_ROOT / "runs/detection/config"
    matplotlib_dir = REPO_ROOT / "runs/detection/matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    import torch
    import yaml
    from ultralytics import YOLO

    with data_path.open(encoding="utf-8") as stream:
        dataset = yaml.safe_load(stream)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("names"), (dict, list)):
        raise SystemExit(f"Invalid dataset YAML (expected names): {data_path}")
    if args.from_scratch:
        model_source = "yolov8s-pose.yaml"
    else:
        model_source = args.model
    model = YOLO(model_source)
    # Box + card-corner keypoints; the dataset declares kpt_shape [4, 3].
    if model.task != "pose":
        raise SystemExit(f"Model task must be 'pose', got {model.task!r}: {model_source}")

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(project_path),
        name=args.name,
        exist_ok=False,
        seed=args.seed,
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        scale=args.scale,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        fraction=args.fraction,
        time=args.time,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        plots=True,
    )


if __name__ == "__main__":
    main()
