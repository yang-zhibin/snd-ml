"""ROOT-backed PyTorch datasets for the data-driven 3D-hit classifier."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from train_3d.data.voxelize import build_sparse_voxels
from train_3d.data.classification_contract import (
    EXCLUDED_CLASS_NAMES,
    TRAINING_CLASS_NAMES,
    make_training_labels,
    training_label,
)
from train_3d.data.data_driven_fold_dataset import (
    HierarchicalTrainingSampler,
    NaturalValidationEvents,
    SampledTrainingEvents,
)


HIT_BRANCHES = {
    "x": "hit_x",
    "y": "hit_y",
    "z": "hit_z",
    "qdc": "hit_qdc",
    "station": "hit_station",
    "detType": "hit_detType",
}
IDENTITY_BRANCHES = ("runId", "eventId", "eventIndex", "pdgCode", "energy")


class RootHit3DEventReader:
    """Read individual Hit3D entries with a small worker-local PyROOT cache."""

    def __init__(self, tree_name: str = "hit3D", max_open_files: int = 8):
        if int(max_open_files) < 1:
            raise ValueError("max_open_files must be positive")
        self.tree_name = str(tree_name)
        self.max_open_files = int(max_open_files)
        self._files = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        return state

    def close(self):
        for root_file, _ in self._files.values():
            root_file.Close()
        self._files.clear()

    def _tree(self, path: str):
        import ROOT

        path = str(path)
        cached = self._files.pop(path, None)
        if cached is None:
            root_file = ROOT.TFile.Open(path, "READ")
            if not root_file or root_file.IsZombie():
                raise OSError(f"Cannot open ROOT file {path}")
            tree = root_file.Get(self.tree_name)
            if not tree:
                root_file.Close()
                raise KeyError(f"Tree {self.tree_name!r} is missing from {path}")
            cached = (root_file, tree)
        self._files[path] = cached
        while len(self._files) > self.max_open_files:
            _, (old_file, _) = self._files.popitem(last=False)
            old_file.Close()
        return cached[1]

    def read(self, reference) -> dict:
        tree = self._tree(reference.hit3d_path)
        entry = int(reference.source_entry)
        n_entries = int(tree.GetEntries())
        if entry < 0 or entry >= n_entries:
            raise IndexError(
                f"Entry {entry} is outside {reference.hit3d_path} "
                f"with {n_entries} entries"
            )
        available = {branch.GetName() for branch in tree.GetListOfBranches()}
        missing = set(HIT_BRANCHES.values()) - available
        if missing:
            raise KeyError(f"Hit3D file lacks branches: {sorted(missing)}")
        if tree.GetEntry(entry) < 0:
            raise OSError(f"Failed to read {reference.hit3d_path}:{entry}")
        dtypes = {
            "x": np.float32, "y": np.float32, "z": np.float32,
            "qdc": np.float32, "station": np.int16, "detType": np.int16,
        }
        hits = {
            key: np.fromiter(getattr(tree, branch), dtype=dtypes[key])
            for key, branch in HIT_BRANCHES.items()
        }
        lengths = {len(value) for value in hits.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Hit-vector length mismatch in {reference.hit3d_path}:{entry}"
            )
        event = {"all_3dHits": hits}
        for branch in IDENTITY_BRANCHES:
            if branch in available:
                value = getattr(tree, branch)
                event[branch] = value.item() if hasattr(value, "item") else value
        return event


class RootBackedClassificationDataset(Dataset):
    """Convert sampled/natural event references directly into sparse voxels."""

    def __init__(self, references, ranges, voxel_size, feature_channels,
                 tree_name="hit3D", max_open_files=8,
                 clamp_out_of_bounds=False,
                 class_names=TRAINING_CLASS_NAMES,
                 excluded_classes=EXCLUDED_CLASS_NAMES):
        self.references = references
        self.ranges = ranges
        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.feature_channels = list(feature_channels)
        self.clamp_out_of_bounds = bool(clamp_out_of_bounds)
        self.excluded_classes = tuple(str(name) for name in excluded_classes)
        self.training_labels = make_training_labels(class_names, self.excluded_classes)
        self.reader = RootHit3DEventReader(tree_name, max_open_files)

    def __len__(self):
        return len(self.references)

    def set_epoch(self, epoch: int):
        setter = getattr(self.references, "set_epoch", None)
        if setter is None:
            raise TypeError("This dataset does not use sampled training references")
        setter(epoch)

    def __getitem__(self, index):
        reference = self.references[index]
        event = self.reader.read(reference)
        coords, features = build_sparse_voxels(
            event["all_3dHits"], ranges=self.ranges,
            voxel_size=self.voxel_size, feature_channels=self.feature_channels,
            clamp_out_of_bounds=self.clamp_out_of_bounds,
        )
        label = training_label(
            reference.class_name, self.training_labels, self.excluded_classes
        )
        return {
            "coords": torch.from_numpy(coords),
            "features": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.long),
            "source_class_id": torch.tensor(reference.class_id, dtype=torch.long),
            "class_name": reference.class_name,
            "domain": reference.domain,
            "density_bin": reference.density_bin,
            "dataset_file_uid": reference.dataset_file_uid,
            "source_entry": reference.source_entry,
            "run_id": int(event.get("runId", -1)),
            "event_id": int(event.get("eventId", -1)),
            "event_index": int(event.get("eventIndex", -1)),
            "pdg_code": int(event.get("pdgCode", -999)),
            "energy": float(event.get("energy", np.nan)),
        }


def build_root_fold_datasets(
    fold_views, cv_fold, epoch_size, ranges, voxel_size, feature_channels,
    seed=0, class_probabilities=None, domain_probabilities=None,
    muondis_density_probabilities="uniform",
    tree_name="hit3D", max_open_files=8, clamp_out_of_bounds=False,
    class_names=TRAINING_CLASS_NAMES, excluded_classes=EXCLUDED_CLASS_NAMES,
):
    """Build one sampled training dataset and one natural validation dataset."""
    sampler = HierarchicalTrainingSampler(
        fold_views, cv_fold=cv_fold, seed=seed,
        class_probabilities=class_probabilities,
        domain_probabilities=domain_probabilities,
        muondis_density_probabilities=muondis_density_probabilities,
    )
    training_references = SampledTrainingEvents(sampler, epoch_size=epoch_size)
    validation_references = NaturalValidationEvents(fold_views, cv_fold=cv_fold)
    common = {
        "ranges": ranges,
        "voxel_size": voxel_size,
        "feature_channels": feature_channels,
        "tree_name": tree_name,
        "max_open_files": max_open_files,
        "clamp_out_of_bounds": clamp_out_of_bounds,
        "class_names": class_names,
        "excluded_classes": excluded_classes,
    }
    return {
        "train": RootBackedClassificationDataset(training_references, **common),
        "validation": RootBackedClassificationDataset(validation_references, **common),
    }


def classification_sparse_collate_fn(batch):
    kept = [
        sample for sample in batch
        if sample["coords"].numel() and sample["features"].numel()
    ]
    if not kept:
        raise RuntimeError("Sparse classification batch contains no active voxels")
    coordinates = []
    features = []
    for batch_index, sample in enumerate(kept):
        batch_column = torch.full(
            (sample["coords"].size(0), 1), batch_index, dtype=torch.int32
        )
        coordinates.append(torch.cat([
            batch_column, sample["coords"].to(torch.int32)
        ], dim=1))
        features.append(sample["features"].to(torch.float32))
    return {
        "coords": torch.cat(coordinates, dim=0),
        "features": torch.cat(features, dim=0),
        "label": torch.stack([sample["label"] for sample in kept]),
        "source_class_id": torch.stack([
            sample["source_class_id"] for sample in kept
        ]),
        "class_name": [sample["class_name"] for sample in kept],
        "domain": [sample["domain"] for sample in kept],
        "density_bin": [sample["density_bin"] for sample in kept],
        "dataset_file_uid": [sample["dataset_file_uid"] for sample in kept],
        "source_entry": torch.tensor(
            [sample["source_entry"] for sample in kept], dtype=torch.long
        ),
        "run_id": torch.tensor([sample["run_id"] for sample in kept], dtype=torch.long),
        "event_id": torch.tensor([sample["event_id"] for sample in kept], dtype=torch.long),
        "event_index": torch.tensor(
            [sample["event_index"] for sample in kept], dtype=torch.long
        ),
        "pdg_code": torch.tensor([sample["pdg_code"] for sample in kept], dtype=torch.long),
        "energy": torch.tensor([sample["energy"] for sample in kept], dtype=torch.float32),
    }
