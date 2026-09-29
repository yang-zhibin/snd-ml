"""Shared fold-aware SciFi DataModule for GravNet and sparse ResNet."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import pandas as pd
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from train_3d.data.classification_collate import (
    ClusteredGravNetClassificationCollator,
    GravNetClassificationCollator,
    NativeGravNetClassificationCollator,
    SparseClassificationCollator,
    SparseRepresentation,
    TwoViewSparseClassificationCollator,
)
from train_3d.data.gravnet_partition import GravNetPartitionConfig
from train_3d.data.data_driven_fold_dataset import (
    DeterministicStratifiedValidationEvents,
    DeterministicTrainingReferenceSampler,
    HierarchicalTrainingSampler,
    NaturalFoldEvents,
    RankSlicedReferences,
)
from train_3d.data.node_budget_batch_sampler import (
    EventSizeIndex,
    NodeBudgetBatchSampler,
)
from train_3d.data.scifi_hit_event_dataset import (
    SciFiHitEventDataset,
    SciFiNormalization,
)
from train_3d.data.native_scifi import NativeSciFiNormalization


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, config: dict, cv_fold: int):
        super().__init__()
        self.config = config
        self.cv_fold = int(cv_fold)
        self.train_reference_sampler = None
        self.train_batch_sampler = None
        self.validation_batch_sampler = None
        self.dataset = None

    def setup(self, stage=None):
        data = self.config["data"]
        representation = self.config["representation"]
        model_family = self.config["model"]["family"]
        views = pd.read_csv(data["fold_views"], keep_default_na=False)
        present_classes = set(views["class_name"].astype(str))
        expected_classes = set(self.config["class_names"])
        unexpected = present_classes - expected_classes
        if unexpected:
            raise ValueError(
                f"Fold views contain classes outside the training contract: {sorted(unexpected)}"
            )
        missing = expected_classes - present_classes
        if missing:
            raise ValueError(f"Fold views lack configured classes: {sorted(missing)}")
        metadata_root = data.get("scifi_metadata_root")
        native_family = model_family in {
            "native_scifi_gravnet", "two_view_sparse_resnet"
        }
        if metadata_root:
            metadata_root = Path(str(metadata_root))
            normalization_name = (
                "native_normalization.json" if native_family else "normalization.json"
            )
            normalization_path = (
                metadata_root / f"fold_{self.cv_fold}" / normalization_name
            )
        else:
            path_key = (
                "native_normalization_path" if native_family else "normalization_path"
            )
            normalization_path = str(data[path_key]).format(
                cv_fold=self.cv_fold
            )
        normalization = (
            NativeSciFiNormalization.load(normalization_path, self.cv_fold)
            if native_family else
            SciFiNormalization.load(normalization_path, expected_fold=self.cv_fold)
        )
        self.dataset = SciFiHitEventDataset(
            references=None,
            tree_name=data.get("tree_name", "hit3D"),
            max_open_files=int(data.get("max_open_files", 8)),
            reader_backend=data.get("reader_backend", "pyroot"),
            class_names=self.config["class_names"],
            excluded_classes=self.config["excluded_classes"],
            include_station=model_family == "clustered_pyg_gravnet",
            input_contract="native" if native_family else "candidate",
        )
        hierarchy = HierarchicalTrainingSampler(
            views,
            cv_fold=self.cv_fold,
            seed=int(self.config["runtime"]["seed"]),
            class_probabilities=data.get("class_probabilities", "uniform"),
            domain_probabilities=data.get("domain_probabilities"),
            muondis_density_probabilities=data.get(
                "muondis_density_probabilities", "uniform"
            ),
        )
        if model_family == "pyg_gravnet":
            collator = GravNetClassificationCollator(normalization)
            budget_representation = "gravnet"
        elif model_family == "clustered_pyg_gravnet":
            partition = representation.get("partition", {})
            if partition.get("algorithm") != "station_xy_recursive_bisection":
                raise ValueError("Unsupported clustered GravNet partition algorithm")
            if partition.get("halo") is not None:
                raise ValueError("Clustered GravNet v1 does not support halos")
            collator = ClusteredGravNetClassificationCollator(
                normalization,
                GravNetPartitionConfig(
                    max_cluster_nodes=int(partition["max_cluster_nodes"]),
                    cluster_by_station=bool(partition.get("cluster_by_station", True)),
                    partition_axes=tuple(partition.get("partition_axes", (0, 1))),
                ),
            )
            budget_representation = "gravnet"
        elif model_family == "sparse_resnet":
            coordinate_min = representation.get("coordinate_min")
            coordinate_max = representation.get("coordinate_max")
            if coordinate_min is None or coordinate_max is None:
                summary_path = (
                    metadata_root / f"fold_{self.cv_fold}" / "summary.json"
                    if metadata_root else
                    str(data["representation_summary"]).format(cv_fold=self.cv_fold)
                )
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                if int(summary["cv_fold"]) != self.cv_fold:
                    raise ValueError("Representation summary has the wrong fold")
                coordinate_min = summary["coordinate_min"]
                coordinate_max = summary["coordinate_max"]
                if tuple(map(float, summary["voxel_size"])) != tuple(
                    map(float, representation["voxel_size"])
                ):
                    raise ValueError("Configured voxel size differs from the size manifest")
            sparse = SparseRepresentation(
                voxel_size=tuple(representation["voxel_size"]),
                coordinate_min=tuple(coordinate_min),
                coordinate_max=tuple(coordinate_max),
            )
            collator = SparseClassificationCollator(normalization, sparse)
            budget_representation = "sparse"
        elif model_family == "native_scifi_gravnet":
            collator = NativeGravNetClassificationCollator(normalization)
            budget_representation = "native"
        elif model_family == "two_view_sparse_resnet":
            collator = TwoViewSparseClassificationCollator(normalization)
            budget_representation = "two_view_sparse"
        else:
            raise ValueError(f"Unknown model family {model_family!r}")
        if metadata_root:
            local_cache = os.environ.get(
                "SND_ML_METADATA_CACHE", data.get("metadata_local_cache") or ""
            )
            if local_cache == "auto":
                local_cache = os.environ.get("TMPDIR", tempfile.gettempdir())
            size_index = EventSizeIndex.from_binary(
                metadata_root, self.cv_fold, budget_representation,
                local_cache_dir=local_cache or None,
            )
        else:
            size_manifest_path = str(data["event_size_manifest"]).format(
                cv_fold=self.cv_fold
            )
            size_index = EventSizeIndex.from_csv(size_manifest_path)
        self.train_reference_sampler = DeterministicTrainingReferenceSampler(
            hierarchy, epoch_size=int(data["epoch_size"]), epoch=0,
            accept=lambda reference: size_index.size(
                reference, budget_representation
            ) > 0,
        )
        validation_sampling = data.get("validation_sampling", "natural")
        if validation_sampling == "natural":
            validation_population = NaturalFoldEvents(
                views, self.cv_fold, "validation"
            )
        elif validation_sampling == "stratified_fixed":
            validation_population = DeterministicStratifiedValidationEvents(
                views, self.cv_fold, self.config["class_names"],
                total_size=int(data["validation_size"]),
                seed=int(data.get("validation_seed", self.config["runtime"]["seed"])),
                accept=lambda reference: size_index.size(
                    reference, budget_representation
                ) > 0,
            )
        else:  # The config validator should make this unreachable.
            raise ValueError(f"Unsupported validation_sampling {validation_sampling!r}")
        validation_references = RankSlicedReferences(validation_population)
        max_events = int(data["max_events_per_batch"])
        max_nodes = int(data["max_nodes_per_batch"])
        self.train_batch_sampler = NodeBudgetBatchSampler(
            self.train_reference_sampler, size_index, budget_representation,
            max_events, max_nodes,
        )
        self.validation_batch_sampler = NodeBudgetBatchSampler(
            validation_references, size_index, budget_representation,
            max_events, max_nodes,
        )
        self.collator = collator

    def set_epoch(self, epoch: int) -> None:
        if self.train_reference_sampler is None:
            raise RuntimeError("Call setup before set_epoch")
        self.train_reference_sampler.set_epoch(epoch)

    def _loader(self, batch_sampler):
        data = self.config["data"]
        workers = int(data.get("num_workers", 0))
        kwargs = {
            "dataset": self.dataset,
            "batch_sampler": batch_sampler,
            "num_workers": workers,
            "pin_memory": bool(data.get("pin_memory", True)),
            "collate_fn": self.collator,
        }
        if workers:
            kwargs["persistent_workers"] = bool(data.get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._loader(self.train_batch_sampler)

    def val_dataloader(self):
        return self._loader(self.validation_batch_sampler)


class AdvanceSampledEpoch(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.set_epoch(int(trainer.current_epoch))


class ThroughputMonitor(pl.Callback):
    """Measure loader wait, step time, throughput, and CUDA peak allocation."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._last_end = time.perf_counter()
        if pl_module.device.type == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self._data_seconds = now - self._last_end
        self._step_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.device.type == "cuda":
            import torch
            torch.cuda.synchronize(pl_module.device)
        now = time.perf_counter()
        step_seconds = max(now - self._step_start, 1e-12)
        events = int(batch["label"].numel())
        nodes = int(batch["n_nodes"].sum())
        clusters = int(batch.get("n_clusters", batch["n_nodes"].new_zeros(1)).sum())
        pl_module.log("timing/data_s", self._data_seconds, on_step=True, on_epoch=False)
        pl_module.log("timing/step_s", step_seconds, on_step=True, on_epoch=False)
        pl_module.log("throughput/events_s", events / step_seconds,
                      on_step=True, on_epoch=False)
        pl_module.log("throughput/nodes_s", nodes / step_seconds,
                      on_step=True, on_epoch=False)
        if clusters:
            pl_module.log("throughput/clusters_s", clusters / step_seconds,
                          on_step=True, on_epoch=False)
        self._last_end = now

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            import torch
            peak = torch.cuda.max_memory_allocated(pl_module.device) / (1024 ** 2)
            pl_module.log("memory/gpu_peak_mb", peak, on_step=False, on_epoch=True)
            reserved = torch.cuda.max_memory_reserved(pl_module.device) / (1024 ** 2)
            pl_module.log("memory/gpu_peak_reserved_mb", reserved,
                          on_step=False, on_epoch=True)
