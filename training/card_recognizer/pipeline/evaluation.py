from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, log_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import write_json


@torch.inference_mode()
def collect_outputs(model, dataloader: DataLoader, device: torch.device, desc: str = "Collecting model outputs") -> dict[str, Any]:
    logits_parts: list[torch.Tensor] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    model.to(device)
    model.eval()
    for batch in tqdm(dataloader, desc=desc, unit="batch"):
        images = batch["image"].to(device)
        _, logits = model(images)
        logits_parts.append(logits.cpu())
        labels.extend([int(x) for x in batch["label"]])
        for i in range(len(batch["label"])):
            records.append(
                {
                    "label": batch["label_name"][i],
                    "card_id": batch["card_id"][i],
                    "set_code": batch["set_code"][i],
                    "image_path": batch["image_path"][i],
                }
            )
    if logits_parts:
        logits = torch.cat(logits_parts).numpy()
    else:
        logits = np.empty((0, 0), dtype=np.float32)
    return {"logits": logits, "labels": np.array(labels), "records": records}


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return float("nan")
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([label in row for label, row in zip(labels, topk)]))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    if len(labels) == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(np.mean(confidences[mask])) - float(np.mean(correct[mask])))
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def per_set_accuracy(probs: np.ndarray, labels: np.ndarray, records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    predictions = probs.argmax(axis=1) if len(labels) else np.array([])
    for set_code in sorted({record["set_code"] for record in records}):
        indices = [idx for idx, record in enumerate(records) if record["set_code"] == set_code]
        if not indices:
            continue
        result[set_code] = {
            "count": len(indices),
            "top1": float(np.mean(predictions[indices] == labels[indices])),
        }
    return result


def threshold_sweep_payload(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, float | int | None]]:
    if len(labels) == 0:
        return []

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels
    rows: list[dict[str, float | int | None]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        accepted = confidences >= threshold
        accepted_count = int(np.sum(accepted))
        precision = float(np.mean(correct[accepted])) if accepted_count else None
        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "rejected": int(len(labels) - accepted_count),
                "coverage": float(np.mean(accepted)),
                "precision": precision,
            }
        )
    return rows


def margin_sweep_payload(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, float | int | None]]:
    if len(labels) == 0:
        return []

    if probs.shape[1] < 2:
        margins = np.ones(len(labels), dtype=np.float32)
    else:
        top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
        margins = top2.max(axis=1) - top2.min(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels
    rows: list[dict[str, float | int | None]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        accepted = margins >= threshold
        accepted_count = int(np.sum(accepted))
        precision = float(np.mean(correct[accepted])) if accepted_count else None
        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "rejected": int(len(labels) - accepted_count),
                "coverage": float(np.mean(accepted)),
                "precision": precision,
            }
        )
    return rows


def select_threshold(
    rows: list[dict[str, float | int | None]],
    target_precision: float,
    min_coverage: float,
) -> dict[str, float | int | None] | None:
    viable = [
        row
        for row in rows
        if row["precision"] is not None
        and float(row["precision"]) >= target_precision
        and float(row["coverage"]) >= min_coverage
    ]
    if viable:
        return max(viable, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))

    return None


def best_threshold_row(rows: list[dict[str, float | int | None]]) -> dict[str, float | int | None] | None:
    eligible = [row for row in rows if row["precision"] is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["precision"]), float(row["coverage"])))


def classification_report_payload(outputs: dict[str, Any], idx_to_label: dict[int, str]) -> dict[str, Any]:
    logits = outputs["logits"]
    labels = outputs["labels"]
    if logits.size == 0:
        return {"count": 0}
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    nll = float(log_loss(labels, probs, labels=list(range(probs.shape[1]))))
    return {
        "count": int(len(labels)),
        "top1": topk_accuracy(probs, labels, 1),
        "top5": topk_accuracy(probs, labels, 5),
        "nll": nll,
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "per_set": per_set_accuracy(probs, labels, outputs["records"]),
        "labels": [idx_to_label[index] for index in range(len(idx_to_label))],
    }


def app_threshold_report_payload(
    outputs: dict[str, Any],
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
    target_precision: float,
    min_coverage: float,
) -> dict[str, Any]:
    logits = outputs["logits"]
    labels = outputs["labels"]
    if logits.size == 0:
        return {
            "target_precision": target_precision,
            "min_coverage": min_coverage,
            "confidence": [],
            "margin": [],
            "recommended_min_confidence": None,
            "recommended_min_margin": None,
        }

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    confidence = threshold_sweep_payload(probs, labels, confidence_thresholds)
    margin = margin_sweep_payload(probs, labels, margin_thresholds)
    selected_confidence = select_threshold(confidence, target_precision, min_coverage)
    selected_margin = select_threshold(margin, target_precision, min_coverage)
    return {
        "target_precision": target_precision,
        "min_coverage": min_coverage,
        "confidence": confidence,
        "margin": margin,
        "recommended_min_confidence": selected_confidence["threshold"] if selected_confidence else None,
        "recommended_min_margin": selected_margin["threshold"] if selected_margin else None,
        "selected_confidence_row": selected_confidence,
        "selected_margin_row": selected_margin,
        "best_confidence_row": best_threshold_row(confidence),
        "best_margin_row": best_threshold_row(margin),
    }


def save_confusion_matrix(outputs: dict[str, Any], labels: list[str], output_prefix: Path, max_labels_for_png: int = 80) -> None:
    logits = outputs["logits"]
    y_true = outputs["labels"]
    if logits.size == 0:
        return
    y_pred = logits.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    print(f"Writing confusion matrix CSV: {output_prefix.with_suffix('.csv')}")
    np.savetxt(output_prefix.with_suffix(".csv"), matrix, fmt="%d", delimiter=",")

    if len(labels) > max_labels_for_png:
        print(f"Skipping confusion matrix PNG because {len(labels)} labels exceeds {max_labels_for_png}.")
        return
    print(f"Writing confusion matrix PNG: {output_prefix.with_suffix('.png')}")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)


def save_report(report: dict[str, Any], output_path: Path) -> None:
    write_json(report, output_path)
