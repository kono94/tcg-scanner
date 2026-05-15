from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset_manifest import load_dataset_manifest
from .paths import PROJECT_ROOT, resolve_path


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CardImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], label_to_idx: dict[str, int], transform=None):
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = resolve_path(record["image_path"], PROJECT_ROOT)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            view_seed = record.get("view_seed")
            if view_seed is None:
                image = self.transform(image)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(view_seed))
                    image = self.transform(image)
        label_index = self.label_to_idx.get(record["label"], -1)
        return {
            "image": image,
            "label": torch.tensor(label_index, dtype=torch.long),
            "label_name": record["label"],
            "card_id": record["card_id"],
            "set_code": record["set_code"],
            "image_path": str(image_path),
        }


def build_train_transform(cfg: dict[str, Any]):
    aug = cfg["augmentation"]
    ops: list[Any] = [
        transforms.Resize(aug["resize"]),
        transforms.RandomResizedCrop(
            aug["crop_size"],
            scale=tuple(aug["random_resized_crop_scale"]),
        ),
    ]
    jitter = aug.get("color_jitter", {})
    if jitter:
        ops.append(transforms.ColorJitter(**jitter))
    affine = aug.get("random_affine", {})
    if affine:
        ops.append(
            transforms.RandomAffine(
                degrees=affine.get("degrees", 0),
                translate=tuple(affine.get("translate", [0.0, 0.0])),
                scale=tuple(affine.get("scale", [1.0, 1.0])),
            )
        )
    perspective = aug.get("random_perspective", {})
    if perspective and perspective.get("p", 0) > 0:
        ops.append(
            transforms.RandomPerspective(
                distortion_scale=perspective.get("distortion_scale", 0.1),
                p=perspective.get("p", 0.0),
            )
        )
    ops.append(transforms.ToTensor())
    erasing = aug.get("random_erasing", {})
    if erasing and erasing.get("p", 0) > 0:
        ops.append(transforms.RandomErasing(p=erasing["p"], scale=tuple(erasing.get("scale", [0.02, 0.1]))))
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def build_eval_transform(cfg: dict[str, Any]):
    aug = cfg["augmentation"]
    return transforms.Compose(
        [
            transforms.Resize(aug["resize"]),
            transforms.CenterCrop(aug["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CardRecognizerDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.manifest = load_dataset_manifest(cfg["dataset"]["manifest_path"])
        self.labels = list(self.manifest["labels"])
        self.label_to_idx = {label: index for index, label in enumerate(self.labels)}
        self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}
        self.batch_size = int(cfg["data"]["batch_size"])
        self.num_workers = int(cfg["data"]["num_workers"])
        self.pin_memory = bool(cfg["data"].get("pin_memory", True))
        self.train_dataset: CardImageDataset | None = None
        self.eval_dataset: CardImageDataset | None = None

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def setup(self, stage: str | None = None) -> None:
        train_transform = build_train_transform(self.cfg)
        eval_transform = build_eval_transform(self.cfg)
        datasets = self.manifest["datasets"]
        self.train_dataset = CardImageDataset(datasets.get("train", []), self.label_to_idx, train_transform)
        self.eval_dataset = CardImageDataset(datasets.get("eval", []), self.label_to_idx, eval_transform)

    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.eval_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.eval_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def dataset_dataloader(self, name: str) -> DataLoader:
        if name in {"val", "test", "eval"}:
            return self._loader(self.eval_dataset, shuffle=False)
        dataset = {
            "train": self.train_dataset,
        }[name]
        return self._loader(dataset, shuffle=False)
