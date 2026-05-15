#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pipeline.config import load_config, save_resolved_config
from pipeline.data import CardRecognizerDataModule
from pipeline.experiments import create_experiment, experiment_dir, make_experiment_id, write_experiment_metadata
from pipeline.lightning_module import RecognizerLightningModule


def build_loggers(cfg: dict, run_dir: Path):
    loggers = []
    if cfg["tracking"].get("csv", True):
        loggers.append(CSVLogger(save_dir=str(run_dir / "logs"), name="csv", version=""))
    if cfg["tracking"].get("tensorboard", False):
        loggers.append(TensorBoardLogger(save_dir=str(run_dir / "logs"), name="tensorboard", version=""))
    return loggers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the card recognizer with PyTorch Lightning.")
    parser.add_argument("--config", default="training/card_recognizer/configs/mobilenetv3_classifier.yaml", type=Path)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Immutable experiment ID. Defaults to the current Unix timestamp in seconds.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_id = args.experiment_id or make_experiment_id()
    cfg["experiment"]["id"] = experiment_id

    seed = int(cfg["experiment"]["seed"])
    print(f"Training experiment {experiment_id}")
    print(f"Config: {args.config}")
    print(f"Seed: {seed}")
    L.seed_everything(seed, workers=True)

    run_dir = experiment_dir(experiment_id, cfg)
    create_experiment(run_dir, experiment_id, cfg)
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")
    print(f"Resolved config written to {run_dir / 'resolved_config.yaml'}")

    print("Preparing datamodule...")
    datamodule = CardRecognizerDataModule(cfg)
    datamodule.setup()
    print(f"Known classes: {datamodule.num_classes}")
    print(f"Train batches: {len(datamodule.train_dataloader())}")
    validation_batches = len(datamodule.val_dataloader())
    print(f"Validation batches: {validation_batches}")
    if validation_batches == 0:
        raise ValueError(
            "No manual validation images were found. Add phone images under "
            f"{cfg['dataset']['manual_eval_dir']}/<card-id>/ before training."
        )
    print("Initializing model...")
    model = RecognizerLightningModule(cfg, datamodule.num_classes)

    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="{epoch:03d}-{val_top1:.4f}",
        monitor="val_top1",
        mode="max",
        save_last=True,
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        logger=build_loggers(cfg, run_dir),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        default_root_dir=str(run_dir),
        **cfg["trainer"],
    )
    print("Starting training. Lightning progress bars should appear below.")
    trainer.fit(model, datamodule=datamodule)
    print("Training finished. Running best-checkpoint evaluation pass...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    write_experiment_metadata(
        run_dir,
        {
            "status": "trained",
            "best_checkpoint": checkpoint.best_model_path,
            "best_score": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        },
    )
    print(f"Experiment ID: {experiment_id}")
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
