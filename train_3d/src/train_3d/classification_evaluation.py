"""Additive evaluation records for deterministic CPU/GPU shard merging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from train_3d.classification_metrics import confusion_matrix, metrics_from_confusion


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def node_count_bin(value: int) -> str:
    value = int(value)
    for edge in (10, 30, 100, 300, 1000, 3000, 10000):
        if value < edge:
            return f"lt_{edge}"
    return "ge_10000"


def empty_additive(num_classes: int) -> dict:
    return {
        "confusion": [[0] * int(num_classes) for _ in range(int(num_classes))],
        "loss_sum": 0.0,
        "event_count": 0,
        "node_count": 0,
    }


def add_observations(target: dict, labels, predictions, losses, node_counts,
                     num_classes: int) -> None:
    matrix = confusion_matrix(labels, predictions, num_classes).cpu()
    target_matrix = torch.as_tensor(target["confusion"], dtype=torch.long)
    target["confusion"] = (target_matrix + matrix).tolist()
    target["loss_sum"] += float(torch.as_tensor(losses).sum())
    target["event_count"] += int(torch.as_tensor(labels).numel())
    target["node_count"] += int(torch.as_tensor(node_counts).sum())


def merge_additive(records: list[dict], num_classes: int) -> dict:
    merged = empty_additive(num_classes)
    for record in records:
        matrix = torch.as_tensor(record["confusion"], dtype=torch.long)
        if tuple(matrix.shape) != (num_classes, num_classes):
            raise ValueError("Shard confusion matrix has the wrong shape")
        merged["confusion"] = (
            torch.as_tensor(merged["confusion"], dtype=torch.long) + matrix
        ).tolist()
        merged["loss_sum"] += float(record["loss_sum"])
        merged["event_count"] += int(record["event_count"])
        merged["node_count"] += int(record.get("node_count", 0))
    return merged


def finalize_additive(record: dict, class_names) -> dict:
    matrix = torch.as_tensor(record["confusion"], dtype=torch.long)
    metrics = metrics_from_confusion(matrix)
    count = int(record["event_count"])
    row_sum = matrix.sum(dim=1, keepdim=True)
    normalized = matrix.double() / row_sum.clamp_min(1)
    result = dict(record)
    result.update({
        "loss": float(record["loss_sum"]) / count if count else None,
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "confusion_row_normalized": normalized.tolist(),
        "per_class": [
            {
                "class_name": str(name),
                "precision": float(metrics["precision"][i]),
                "recall": float(metrics["recall"][i]),
                "f1": float(metrics["f1"][i]),
                "support": int(metrics["support"][i]),
            }
            for i, name in enumerate(class_names)
        ],
    })
    return result


def merge_grouped(grouped_records: list[dict], num_classes: int) -> dict:
    keys = sorted({key for groups in grouped_records for key in groups})
    return {
        key: merge_additive(
            [groups[key] for groups in grouped_records if key in groups], num_classes
        )
        for key in keys
    }


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
