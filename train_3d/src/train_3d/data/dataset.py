from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from train_3d.data.voxelize import build_sparse_voxels


def load_npz_events(npz_path):
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    events = list(data["events"])
    metadata = data["metadata"].item() if "metadata" in data else {}
    return events, metadata


class Hits3DDataset(Dataset):
    def __init__(
        self,
        events,
        ranges,
        voxel_size,
        feature_channels,
        drop_missing_energy=True,
        clamp_out_of_bounds=False,
        target_transform="none",
    ):
        self.events = []
        self.ranges = ranges
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.feature_channels = list(feature_channels)
        self.clamp_out_of_bounds = bool(clamp_out_of_bounds)
        self.target_transform = str(target_transform)

        for event in events:
            energy = event.get("energy")
            if drop_missing_energy and energy is None:
                continue
            try:
                energy = float(energy)
            except (TypeError, ValueError):
                if drop_missing_energy:
                    continue
                energy = np.nan
            if drop_missing_energy and not np.isfinite(energy):
                continue
            hits = event.get("all_3dHits", {})
            x = np.asarray(hits.get("x", []), dtype=np.float32)
            if x.size == 0:
                continue
            self.events.append(event)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[index]
        coords, features = build_sparse_voxels(
            event.get("all_3dHits", {}),
            ranges=self.ranges,
            voxel_size=self.voxel_size,
            feature_channels=self.feature_channels,
            clamp_out_of_bounds=self.clamp_out_of_bounds,
        )

        energy = float(event["energy"])
        if self.target_transform == "log1p":
            if energy < 0.0:
                raise ValueError(f"Energy must be non-negative for log1p transform, got {energy}")
            target = np.log1p(energy)
        elif self.target_transform == "none":
            target = energy
        else:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")

        sample = {
            "coords": torch.from_numpy(coords),
            "features": torch.from_numpy(features),
            "y": torch.tensor(target, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "event_id": int(event.get("eventId", -1)),
            "run_id": int(event.get("runId", -1)),
        }
        return sample
