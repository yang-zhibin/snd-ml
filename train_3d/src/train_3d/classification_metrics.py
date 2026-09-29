"""Additive multiclass statistics suitable for DDP and CPU-shard merging."""

from __future__ import annotations

import torch


def confusion_matrix(labels, predictions, num_classes: int):
    labels = labels.long().reshape(-1)
    predictions = predictions.long().reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if labels.numel() and (
        labels.min() < 0 or predictions.min() < 0
        or labels.max() >= num_classes or predictions.max() >= num_classes
    ):
        raise ValueError("labels and predictions must lie inside the class range")
    flat = labels * int(num_classes) + predictions
    return torch.bincount(flat, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(confusion):
    matrix = confusion.to(torch.float64)
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("confusion must be a square matrix")
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    present = support > 0
    total = support.sum()
    accuracy = true_positive.sum() / total.clamp_min(1)
    macro_f1 = f1[present].mean() if present.any() else matrix.new_tensor(0.0)
    balanced_accuracy = (
        recall[present].mean() if present.any() else matrix.new_tensor(0.0)
    )
    weighted_f1 = (f1 * support).sum() / total.clamp_min(1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }
