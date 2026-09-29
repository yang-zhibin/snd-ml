"""Common seven-class Lightning module for sparse ResNet and GravNet."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from train_3d.classification_metrics import confusion_matrix, metrics_from_confusion
from train_3d.classification_evaluation import node_count_bin
from train_3d.models.classifier_factory import build_classifier


def classification_loss_and_accuracy(logits, labels, num_classes: int):
    if logits.ndim != 2 or logits.size(1) != int(num_classes):
        raise ValueError(
            f"Expected logits with shape [batch,{num_classes}], got {tuple(logits.shape)}"
        )
    labels = labels.long()
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("Labels must have one entry per logit row")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"Labels must be in [0,{num_classes - 1}]")
    loss = nn.functional.cross_entropy(logits, labels)
    accuracy = logits.argmax(dim=1).eq(labels).float().mean()
    return loss, accuracy


class ClassificationModule(pl.LightningModule):
    def __init__(self, model_config: dict, optim_config: dict,
                 class_names: list[str] | tuple[str, ...],
                 evaluation_config: dict | None = None,
                 objective_config: dict | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.model_family = str(model_config["family"])
        self.model = build_classifier(model_config, self.num_classes)
        self.lr = float(optim_config["lr"])
        self.weight_decay = float(optim_config.get("weight_decay", 0.0))
        self.objective_config = dict(objective_config or {})
        if self.objective_config.get("name", "cross_entropy") != "cross_entropy":
            raise ValueError("Only the cross_entropy objective is supported")
        if self.objective_config.get("class_weights") is not None:
            raise ValueError("The initial classifier requires unweighted cross-entropy")
        self.evaluation_config = dict(evaluation_config or {})
        self.prediction_table_max_rows = int(
            self.evaluation_config.get("prediction_table_max_rows", 2000)
        )
        self.validation_rows = []
        self.register_buffer(
            "train_confusion", torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "val_confusion", torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            persistent=False,
        )

    def forward(self, batch):
        if self.model_family == "sparse_resnet":
            return self.model(batch["coords"].int(), batch["features"].float())
        if self.model_family in {"pyg_gravnet", "native_scifi_gravnet"}:
            return self.model(batch["x"].float(), batch["batch"].long())
        if self.model_family == "two_view_sparse_resnet":
            return self.model(
                batch["vertical_coords"].int(), batch["vertical_features"].float(),
                batch["horizontal_coords"].int(),
                batch["horizontal_features"].float(),
                batch["vertical_present"].bool(),
                batch["horizontal_present"].bool(),
            )
        if self.model_family == "clustered_pyg_gravnet":
            return self.model(
                batch["x"].float(), batch["cluster_ptr"].long(),
                batch["cluster_event"].long(), int(batch["label"].numel()),
            )
        raise RuntimeError(f"Unknown model family {self.model_family!r}")

    def on_train_epoch_start(self):
        self.train_confusion.zero_()
        self.train_population = {"domain": {}, "density_bin": {}}

    def on_validation_epoch_start(self):
        self.val_confusion.zero_()
        self.validation_rows = []
        self.validation_grouped_confusion = {
            "domain": {}, "density_bin": {}, "node_count_bin": {}
        }

    def _update_validation_groups(self, batch, labels, predictions):
        for index in range(len(labels)):
            keys = {
                "domain": str(batch["domain"][index]),
                "density_bin": str(batch["density_bin"][index] or "none"),
                "node_count_bin": node_count_bin(int(batch["n_nodes"][index])),
            }
            event_matrix = confusion_matrix(
                labels[index:index + 1], predictions[index:index + 1], self.num_classes
            ).cpu()
            for group, key in keys.items():
                current = self.validation_grouped_confusion[group].setdefault(
                    key, torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
                )
                current.add_(event_matrix)

    def _nonfinite_error(self, batch, batch_idx, loss):
        identities = list(zip(
            batch.get("dataset_file_uid", []),
            batch.get("source_entry", torch.empty(0, dtype=torch.long)).tolist(),
        ))
        nodes = batch.get("n_nodes", torch.empty(0, dtype=torch.long)).tolist()
        values = batch.get("features", batch.get("x"))
        feature_range = None
        if values is not None and values.numel():
            feature_range = [
                float(values.detach().min().cpu()),
                float(values.detach().max().cpu()),
            ]
        return RuntimeError(
            f"Non-finite loss={float(loss.detach().cpu())} family={self.model_family} "
            f"epoch={self.current_epoch} batch={batch_idx} "
            f"events={identities[:10]} nodes={nodes[:10]} "
            f"feature_range={feature_range}"
        )

    def _step(self, batch, batch_idx: int, prefix: str):
        logits = self(batch)
        loss, accuracy = classification_loss_and_accuracy(
            logits, batch["label"], self.num_classes
        )
        if not torch.isfinite(loss):
            raise self._nonfinite_error(batch, batch_idx, loss)
        predictions = logits.argmax(dim=1)
        target_confusion = self.train_confusion if prefix == "train" else self.val_confusion
        batch_confusion = confusion_matrix(
            batch["label"], predictions, self.num_classes
        )
        target_confusion.add_(batch_confusion)
        if prefix == "val" and len(self.validation_rows) < self.prediction_table_max_rows:
            probabilities = torch.softmax(logits.detach(), dim=1).cpu()
            labels = batch["label"].detach().cpu()
            predictions_cpu = predictions.detach().cpu()
            remaining = self.prediction_table_max_rows - len(self.validation_rows)
            for index in range(min(len(labels), remaining)):
                self.validation_rows.append({
                    "dataset_file_uid": batch["dataset_file_uid"][index],
                    "source_entry": int(batch["source_entry"][index]),
                    "run_id": int(batch["run_id"][index]),
                    "event_id": int(batch["event_id"][index]),
                    "event_index": int(batch["event_index"][index]),
                    "domain": batch["domain"][index],
                    "density_bin": batch["density_bin"][index],
                    "n_nodes": int(batch["n_nodes"][index]),
                    "true_class": self.class_names[int(labels[index])],
                    "predicted_class": self.class_names[int(predictions_cpu[index])],
                    "probabilities": probabilities[index].tolist(),
                })
        if prefix == "val":
            self._update_validation_groups(
                batch, batch["label"].detach().cpu(), predictions.detach().cpu()
            )
        batch_size = int(batch["label"].size(0))
        self.log(
            f"{prefix}/loss", loss, on_step=prefix == "train", on_epoch=True,
            prog_bar=True, batch_size=batch_size, sync_dist=True,
        )
        self.log(
            f"{prefix}/accuracy_batch", accuracy, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_size, sync_dist=True,
        )
        if prefix == "train":
            # These deliberately describe the current batch. Epoch-wide metrics
            # are computed from the accumulated confusion matrix below. Keeping
            # distinct names makes the W&B step and epoch plots unambiguous.
            batch_metrics = metrics_from_confusion(batch_confusion)
            self.log(
                "train/accuracy_step", accuracy, on_step=True, on_epoch=False,
                batch_size=batch_size, sync_dist=True,
            )
            for name in ("balanced_accuracy", "macro_f1", "weighted_f1"):
                self.log(
                    f"train/{name}_step", batch_metrics[name].float(),
                    on_step=True, on_epoch=False, batch_size=batch_size,
                    sync_dist=True,
                )
            node_count = int(batch["n_nodes"].sum())
            self.log("batch/events", float(batch_size), on_step=True, on_epoch=False)
            self.log("batch/nodes", float(node_count), on_step=True, on_epoch=False)
            if self.model_family in {
                "pyg_gravnet", "native_scifi_gravnet", "clustered_pyg_gravnet"
            }:
                k = int(self.hparams.model_config.get("k", 0))
                self.log("batch/edges_estimate", float(k * node_count),
                         on_step=True, on_epoch=False)
            if self.model_family == "clustered_pyg_gravnet":
                cluster_sizes = batch["cluster_ptr"][1:] - batch["cluster_ptr"][:-1]
                self.log("batch/clusters", float(len(cluster_sizes)),
                         on_step=True, on_epoch=False)
                self.log("batch/max_cluster_nodes", float(cluster_sizes.max()),
                         on_step=True, on_epoch=False)
                self.log("batch/mean_cluster_nodes", cluster_sizes.float().mean(),
                         on_step=True, on_epoch=False)
                self.log("batch/max_clusters_per_event", float(batch["n_clusters"].max()),
                         on_step=True, on_epoch=False)
            for domain in batch["domain"]:
                self.train_population["domain"][domain] = (
                    self.train_population["domain"].get(domain, 0) + 1
                )
            for density_bin in batch["density_bin"]:
                key = density_bin or "none"
                self.train_population["density_bin"][key] = (
                    self.train_population["density_bin"].get(key, 0) + 1
                )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def _global_confusion(self, matrix):
        result = matrix.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result

    def _log_epoch_metrics(self, prefix: str, matrix):
        global_confusion = self._global_confusion(matrix)
        metrics = metrics_from_confusion(global_confusion)
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            self.log(f"{prefix}/{name}", metrics[name].float(), prog_bar=name == "macro_f1")
        for index, class_name in enumerate(self.class_names):
            self.log(f"{prefix}/support/{class_name}", metrics["support"][index].float())
            for name in ("precision", "recall", "f1"):
                self.log(f"{prefix}/{name}/{class_name}", metrics[name][index].float())
        return global_confusion, metrics

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.train_confusion)
        populations = [self.train_population]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            populations = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(populations, self.train_population)
        for group in ("domain", "density_bin"):
            keys = sorted({key for item in populations for key in item[group]})
            for key in keys:
                count = sum(item[group].get(key, 0) for item in populations)
                self.log(f"train/population/{group}/{key}", float(count))

    def on_validation_epoch_end(self):
        confusion, metrics = self._log_epoch_metrics("val", self.val_confusion)
        validation_rows = self.validation_rows
        grouped_sets = [self.validation_grouped_confusion]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, self.validation_rows)
            validation_rows = [row for rank_rows in gathered for row in rank_rows]
            validation_rows.sort(
                key=lambda row: (row["dataset_file_uid"], row["source_entry"])
            )
            validation_rows = validation_rows[:self.prediction_table_max_rows]
            grouped_sets = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                grouped_sets, self.validation_grouped_confusion
            )
        if not self.trainer.is_global_zero:
            return
        confusion = confusion.detach().cpu()
        grouped_report = {}
        for group in ("domain", "density_bin", "node_count_bin"):
            keys = sorted({key for item in grouped_sets for key in item[group]})
            grouped_report[group] = {}
            for key in keys:
                group_confusion = sum(
                    (item[group][key] for item in grouped_sets if key in item[group]),
                    torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
                )
                group_metrics = metrics_from_confusion(group_confusion)
                grouped_report[group][key] = {
                    "event_count": int(group_confusion.sum()),
                    "accuracy": float(group_metrics["accuracy"]),
                    "balanced_accuracy": float(group_metrics["balanced_accuracy"]),
                    "macro_f1": float(group_metrics["macro_f1"]),
                    "weighted_f1": float(group_metrics["weighted_f1"]),
                    "confusion": group_confusion.tolist(),
                }
        report = {
            "epoch": int(self.current_epoch),
            "class_names": list(self.class_names),
            "confusion": confusion.tolist(),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "per_class": [
                {
                    "class_name": name,
                    "precision": float(metrics["precision"][index]),
                    "recall": float(metrics["recall"][index]),
                    "f1": float(metrics["f1"][index]),
                    "support": int(metrics["support"][index]),
                }
                for index, name in enumerate(self.class_names)
            ],
            "bounded_predictions": validation_rows,
            "grouped": grouped_report,
        }
        output = Path(self.trainer.default_root_dir) / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        (output / f"epoch_{int(self.current_epoch):03d}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        interval = max(1, int(self.evaluation_config.get(
            "diagnostic_table_every_n_epochs", 1
        )))
        if (int(self.current_epoch) + 1) % interval == 0:
            self._log_wandb_tables(report)

    def _log_wandb_tables(self, report):
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or experiment is False:
            return
        try:
            import wandb
        except ImportError:
            return
        per_class = wandb.Table(
            columns=["class", "precision", "recall", "f1", "support"],
            data=[
                [row["class_name"], row["precision"], row["recall"], row["f1"], row["support"]]
                for row in report["per_class"]
            ],
        )
        prediction_columns = [
            "dataset_file_uid", "source_entry", "run_id", "event_id",
            "event_index", "domain", "density_bin", "n_nodes", "true_class",
            "predicted_class", *[f"prob_{name}" for name in self.class_names],
        ]
        predictions = wandb.Table(
            columns=prediction_columns,
            data=[
                [
                    row["dataset_file_uid"], row["source_entry"], row["run_id"],
                    row["event_id"], row["event_index"], row["domain"],
                    row["density_bin"], row["n_nodes"], row["true_class"],
                    row["predicted_class"], *row["probabilities"],
                ]
                for row in report["bounded_predictions"]
            ],
        )
        confusion = wandb.Table(
            columns=["true_class", *self.class_names],
            data=[
                [name, *row]
                for name, row in zip(self.class_names, report["confusion"])
            ],
        )
        matrix = torch.as_tensor(report["confusion"], dtype=torch.float64)
        normalized_values = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1)
        normalized = wandb.Table(
            columns=["true_class", *self.class_names],
            data=[
                [name, *row]
                for name, row in zip(self.class_names, normalized_values.tolist())
            ],
        )
        grouped = wandb.Table(
            columns=["group", "value", "events", "accuracy", "balanced_accuracy",
                     "macro_f1", "weighted_f1"],
            data=[
                [group, key, values["event_count"], values["accuracy"],
                 values["balanced_accuracy"], values["macro_f1"],
                 values["weighted_f1"]]
                for group, entries in report["grouped"].items()
                for key, values in entries.items()
            ],
        )
        experiment.log({
            "val/per_class_table": per_class,
            "val/predictions_bounded": predictions,
            "val/confusion_absolute": confusion,
            "val/confusion_row_normalized": normalized,
            "val/grouped_metrics": grouped,
        })

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# Compatibility name used by the first partial classifier entrypoint.
Hits3DClassifier = ClassificationModule
