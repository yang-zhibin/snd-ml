from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from train_3d.data.dataset import Hits3DDataset, load_npz_events
from train_3d.data.voxelize import infer_coordinate_ranges


def sparse_collate_fn(batch):
    kept_samples = []
    for sample in batch:
        if sample["coords"].numel() == 0 or sample["features"].numel() == 0:
            continue
        kept_samples.append(sample)

    if not kept_samples:
        raise RuntimeError(
            "Sparse batch contains no active voxels. "
            "Check preprocessing, coordinate ranges, and voxel size."
        )

    coords_list = []
    features_list = []
    targets = []
    event_ids = []
    run_ids = []

    for batch_index, sample in enumerate(kept_samples):
        coords = sample["coords"].to(dtype=torch.int32)
        features = sample["features"].to(dtype=torch.float32)

        batch_column = torch.full((coords.size(0), 1), batch_index, dtype=torch.int32)
        coords_with_batch = torch.cat([batch_column, coords], dim=1)

        coords_list.append(coords_with_batch)
        features_list.append(features)
        targets.append(sample["y"].to(dtype=torch.float32))
        event_ids.append(sample["event_id"])
        run_ids.append(sample["run_id"])

    if coords_list:
        coords = torch.cat(coords_list, dim=0)
        features = torch.cat(features_list, dim=0)
    else:
        coords = torch.zeros((0, 4), dtype=torch.int32)
        features = torch.zeros((0, 0), dtype=torch.float32)

    return {
        "coords": coords,
        "features": features,
        "y": torch.stack(targets, dim=0),
        "energy": torch.stack([sample["energy"].to(dtype=torch.float32) for sample in kept_samples], dim=0),
        "event_id": torch.tensor(event_ids, dtype=torch.long),
        "run_id": torch.tensor(run_ids, dtype=torch.long),
    }


class Hits3DDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, data_config, voxel_config):
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.data_config = data_config
        self.voxel_config = voxel_config
        self.train_dataset = None
        self.val_dataset = None
        self.coordinate_ranges = None

    def setup(self, stage=None):
        train_events, _ = load_npz_events(self.train_path)
        val_events, _ = load_npz_events(self.val_path)

        self.coordinate_ranges = self._resolve_coordinate_ranges(train_events)

        common_kwargs = {
            "ranges": self.coordinate_ranges,
            "voxel_size": self.voxel_config["voxel_size"],
            "feature_channels": self.voxel_config["feature_channels"],
            "drop_missing_energy": self.data_config.get("drop_missing_energy", True),
            "clamp_out_of_bounds": self.voxel_config.get("clamp_out_of_bounds", False),
            "target_transform": self.data_config.get("target_transform", "none"),
        }

        self.train_dataset = Hits3DDataset(train_events, **common_kwargs)
        self.val_dataset = Hits3DDataset(val_events, **common_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            pin_memory=True,
            collate_fn=sparse_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            pin_memory=True,
            collate_fn=sparse_collate_fn,
        )

    def _resolve_coordinate_ranges(self, train_events):
        ranges = {}
        inferred = infer_coordinate_ranges(train_events)

        for key in ("x_range", "y_range", "z_range"):
            value = self.voxel_config.get(key)
            if value is None:
                ranges[key] = inferred[key]
            else:
                ranges[key] = tuple(float(v) for v in value)

        return ranges
