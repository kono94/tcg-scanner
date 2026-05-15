from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .paths import add_project_root_to_path

add_project_root_to_path()
from model import CardModel  # noqa: E402


class RecognizerLightningModule(L.LightningModule):
    def __init__(self, cfg: dict[str, Any], num_classes: int):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "num_classes": num_classes})
        self.cfg = cfg
        self.num_classes = num_classes
        model_cfg = cfg.get("model", {})
        self.freeze_backbone_epochs = int(model_cfg.get("freeze_backbone_epochs", 0))
        self.model = CardModel(
            num_labels=num_classes,
            feature_dim=model_cfg.get("feature_dim", 256),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
        self._backbone_trainable: bool | None = None
        self.set_backbone_trainable(self.freeze_backbone_epochs <= 0)
        top5 = min(5, num_classes)
        self.train_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top5, average="micro")
        self.test_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.test_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top5, average="micro")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(images)

    def set_backbone_trainable(self, trainable: bool) -> None:
        if self._backbone_trainable == trainable:
            return
        for param in self.model.base_model.parameters():
            param.requires_grad = trainable
        self._backbone_trainable = trainable

    def on_train_epoch_start(self) -> None:
        if self.freeze_backbone_epochs <= 0:
            self.set_backbone_trainable(True)
            return

        trainable = self.current_epoch >= self.freeze_backbone_epochs
        previous = self._backbone_trainable
        self.set_backbone_trainable(trainable)
        if previous != trainable:
            state = "unfrozen" if trainable else "frozen"
            self.print(f"Backbone {state} at epoch {self.current_epoch + 1}.")

    def shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        labels = batch["label"]
        features, logits = self(batch["image"])
        batch_size = int(labels.shape[0])
        loss = F.cross_entropy(logits, labels)
        self.log(f"{stage}/loss", loss, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)

        if stage == "train":
            self.train_top1(logits, labels)
            self.log("train/top1", self.train_top1, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        elif stage == "val":
            self.val_top1(logits, labels)
            self.val_top5(logits, labels)
            self.log("val/top1", self.val_top1, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/top5", self.val_top5, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val_top1", self.val_top1, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        elif stage == "test":
            self.test_top1(logits, labels)
            self.test_top5(logits, labels)
            self.log("test/top1", self.test_top1, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("test/top5", self.test_top5, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        self.log(f"{stage}/feature_norm", features.norm(dim=1).mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self.shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        base_lr = float(opt_cfg["lr"])
        backbone_lr = base_lr * float(opt_cfg.get("backbone_lr_multiplier", 1.0))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        params = [
            {"params": self.model.base_model.parameters(), "lr": backbone_lr},
            {"params": self.model.feature_layer.parameters(), "lr": base_lr},
            {"params": self.model.classification_layer.parameters(), "lr": base_lr},
        ]
        name = opt_cfg.get("name", "adamw").lower()
        if name == "adam":
            optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        elif name == "sgd":
            optimizer = torch.optim.SGD(params, lr=base_lr, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

        sched_cfg = self.cfg.get("scheduler", {})
        if sched_cfg.get("name", "none").lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.cfg["trainer"]["max_epochs"])),
                eta_min=float(sched_cfg.get("min_lr", 0.0)),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
