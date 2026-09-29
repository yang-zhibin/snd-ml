"""Model-specific batching built from one shared SciFi event contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from train_3d.data.scifi_hit_event_dataset import SciFiNormalization
from train_3d.data.gravnet_partition import (
    GravNetPartitionConfig,
    partition_event,
)
from train_3d.data.native_scifi import (
    HORIZONTAL,
    VERTICAL,
    NativeSciFiNormalization,
    build_sparse_view,
    validate_native_scifi_hits,
)


MODEL_INPUT_KEYS = frozenset({"position", "qdc", "detector_type"})


def _validate_sample(sample: dict) -> None:
    position = np.asarray(sample["position"])
    qdc = np.asarray(sample["qdc"])
    detector_type = np.asarray(sample["detector_type"])
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"position must have shape [N,3], got {position.shape}")
    if qdc.ndim != 1 or detector_type.ndim != 1:
        raise ValueError("qdc and detector_type must have shape [N]")
    if not (len(position) == len(qdc) == len(detector_type)):
        raise ValueError("Filtered SciFi arrays are not aligned")
    if np.any(detector_type != 1):
        raise ValueError("A non-SciFi detector row reached model collation")


def _metadata(batch: list[dict]) -> dict:
    result = {
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "source_class_id": torch.tensor(
            [item["source_class_id"] for item in batch], dtype=torch.long
        ),
        "class_name": [item["class_name"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "density_bin": [item["density_bin"] for item in batch],
        "dataset_file_uid": [item["dataset_file_uid"] for item in batch],
        "source_entry": torch.tensor(
            [item["source_entry"] for item in batch], dtype=torch.long
        ),
        "run_id": torch.tensor([item["run_id"] for item in batch], dtype=torch.long),
        "event_id": torch.tensor([item["event_id"] for item in batch], dtype=torch.long),
        "event_index": torch.tensor(
            [item["event_index"] for item in batch], dtype=torch.long
        ),
        "pdg_code": torch.tensor([item["pdg_code"] for item in batch], dtype=torch.long),
        "energy": torch.tensor([item["energy"] for item in batch], dtype=torch.float32),
        "n_candidates": torch.tensor(
            [item["n_candidates"] for item in batch], dtype=torch.long
        ),
    }
    for key in (
        "n_native_hits", "n_vertical_hits", "n_horizontal_hits",
        "n_vertical_sites", "n_horizontal_sites",
    ):
        if all(key in item for item in batch):
            result[key] = torch.tensor(
                [item[key] for item in batch], dtype=torch.long
            )
    return result


def _require_nonempty(batch: list[dict]) -> None:
    empty = [
        f"{item['dataset_file_uid']}:{item['source_entry']}"
        for item in batch if not len(item["qdc"])
    ]
    if empty:
        raise ValueError(
            "Events without selected SciFi candidates reached a batch: "
            + ", ".join(empty[:10])
        )


@dataclass(frozen=True)
class SparseRepresentation:
    voxel_size: tuple[float, float, float]
    coordinate_min: tuple[float, float, float]
    coordinate_max: tuple[float, float, float]

    def __post_init__(self):
        if len(self.voxel_size) != 3 or len(self.coordinate_min) != 3 or len(self.coordinate_max) != 3:
            raise ValueError("Sparse representation requires three coordinate axes")
        if np.any(np.asarray(self.voxel_size, dtype=float) <= 0):
            raise ValueError("voxel_size entries must be positive")
        if np.any(np.asarray(self.coordinate_max) <= np.asarray(self.coordinate_min)):
            raise ValueError("coordinate_max must exceed coordinate_min")


def build_scifi_sparse_voxels(sample: dict, normalization: SciFiNormalization,
                               representation: SparseRepresentation):
    """Quantize candidates and reduce collisions without adding extra inputs."""
    _validate_sample(sample)
    position = np.asarray(sample["position"], dtype=np.float32)
    qdc = np.asarray(sample["qdc"], dtype=np.float32)
    if not len(position):
        return np.empty((0, 3), np.int32), np.empty((0, 2), np.float32)
    lower = np.asarray(representation.coordinate_min, dtype=np.float32)
    size = np.asarray(representation.voxel_size, dtype=np.float32)
    coords = np.floor((position - lower) / size).astype(np.int32)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    qdc_sum = np.zeros(len(unique_coords), dtype=np.float32)
    np.add.at(qdc_sum, inverse, qdc)
    qdc_feature = normalization.transform_qdc(qdc_sum)
    detector_feature = np.ones(len(unique_coords), dtype=np.float32)
    features = np.column_stack([qdc_feature, detector_feature]).astype(
        np.float32, copy=False
    )
    return unique_coords.astype(np.int32, copy=False), features


class SparseClassificationCollator:
    def __init__(self, normalization: SciFiNormalization,
                 representation: SparseRepresentation):
        self.normalization = normalization
        self.representation = representation

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        _require_nonempty(batch)
        coordinates = []
        features = []
        voxel_counts = []
        for batch_index, sample in enumerate(batch):
            coords, values = build_scifi_sparse_voxels(
                sample, self.normalization, self.representation
            )
            if not len(coords):
                raise ValueError("SciFi event produced no active sparse voxels")
            batch_column = np.full((len(coords), 1), batch_index, dtype=np.int32)
            coordinates.append(np.concatenate([batch_column, coords], axis=1))
            features.append(values)
            voxel_counts.append(len(coords))
        result = _metadata(batch)
        result.update({
            "coords": torch.from_numpy(np.concatenate(coordinates, axis=0)),
            "features": torch.from_numpy(np.concatenate(features, axis=0)),
            "n_nodes": torch.tensor(voxel_counts, dtype=torch.long),
            "model_family": "sparse_resnet",
        })
        return result


class GravNetClassificationCollator:
    def __init__(self, normalization: SciFiNormalization):
        self.normalization = normalization

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        _require_nonempty(batch)
        positions = []
        node_features = []
        batch_index = []
        node_counts = []
        for event_index, sample in enumerate(batch):
            _validate_sample(sample)
            position = self.normalization.normalize_position(sample["position"])
            qdc = self.normalization.transform_qdc(sample["qdc"])[:, None]
            detector_type = np.asarray(sample["detector_type"], dtype=np.float32)[:, None]
            positions.append(position)
            # XYZ is intentionally included in x: GravNet learns coordinates from x.
            node_features.append(np.concatenate([position, qdc, detector_type], axis=1))
            batch_index.append(np.full(len(position), event_index, dtype=np.int64))
            node_counts.append(len(position))
        result = _metadata(batch)
        result.update({
            "x": torch.from_numpy(np.concatenate(node_features, axis=0)).float(),
            "pos": torch.from_numpy(np.concatenate(positions, axis=0)).float(),
            "batch": torch.from_numpy(np.concatenate(batch_index, axis=0)).long(),
            "n_nodes": torch.tensor(node_counts, dtype=torch.long),
            "model_family": "pyg_gravnet",
        })
        return result


class ClusteredGravNetClassificationCollator:
    """Reorder full events into deterministic, bounded spatial clusters."""

    def __init__(self, normalization: SciFiNormalization,
                 partition: GravNetPartitionConfig):
        self.normalization = normalization
        self.partition = partition

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        _require_nonempty(batch)
        positions = []
        node_features = []
        cluster_ptr = [0]
        cluster_event = []
        cluster_station = []
        node_counts = []
        cluster_counts = []
        for event_index, sample in enumerate(batch):
            _validate_sample(sample)
            if "station" not in sample:
                raise KeyError("Clustered GravNet sample lacks station")
            raw_position = np.asarray(sample["position"], dtype=np.float32)
            station = np.asarray(sample["station"], dtype=np.int16)
            clusters = partition_event(raw_position, station, self.partition)
            if not clusters:
                raise ValueError("SciFi event produced no GravNet clusters")
            normalized = self.normalization.normalize_position(raw_position)
            qdc = self.normalization.transform_qdc(sample["qdc"])[:, None]
            detector = np.asarray(sample["detector_type"], dtype=np.float32)[:, None]
            features = np.concatenate([normalized, qdc, detector], axis=1)
            for indices in clusters:
                positions.append(normalized[indices])
                node_features.append(features[indices])
                cluster_ptr.append(cluster_ptr[-1] + len(indices))
                cluster_event.append(event_index)
                values = np.unique(station[indices])
                cluster_station.append(int(values[0]) if len(values) == 1 else -1)
            node_counts.append(len(raw_position))
            cluster_counts.append(len(clusters))
        result = _metadata(batch)
        result.update({
            "x": torch.from_numpy(np.concatenate(node_features, axis=0)).float(),
            "pos": torch.from_numpy(np.concatenate(positions, axis=0)).float(),
            "cluster_ptr": torch.tensor(cluster_ptr, dtype=torch.long),
            "cluster_event": torch.tensor(cluster_event, dtype=torch.long),
            "cluster_station": torch.tensor(cluster_station, dtype=torch.int16),
            "n_nodes": torch.tensor(node_counts, dtype=torch.long),
            "n_clusters": torch.tensor(cluster_counts, dtype=torch.long),
            "model_family": "clustered_pyg_gravnet",
        })
        return result


class NativeGravNetClassificationCollator:
    """Collate one graph node for each detector-native SciFi hit."""

    def __init__(self, normalization: NativeSciFiNormalization):
        self.normalization = normalization

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        _require_nonempty(batch)
        features = []
        batch_index = []
        counts = []
        for event_index, sample in enumerate(batch):
            event = validate_native_scifi_hits(sample)
            if not len(event["qdc"]):
                raise ValueError("Native GravNet cannot process an empty event")
            features.append(self.normalization.node_features(event))
            batch_index.append(np.full(len(event["qdc"]), event_index, np.int64))
            counts.append(len(event["qdc"]))
        result = _metadata(batch)
        result.update({
            "x": torch.from_numpy(np.concatenate(features)).float(),
            "batch": torch.from_numpy(np.concatenate(batch_index)).long(),
            "n_nodes": torch.tensor(counts, dtype=torch.long),
            "model_family": "native_scifi_gravnet",
        })
        return result


class TwoViewSparseClassificationCollator:
    """Build two exact station/channel sparse detector views."""

    def __init__(self, normalization: NativeSciFiNormalization):
        self.normalization = normalization

    @staticmethod
    def _append_view(coordinates, features, event_index, target_coords,
                     target_features):
        if not len(coordinates):
            return
        batch_column = np.full((len(coordinates), 1), event_index, np.int32)
        target_coords.append(np.concatenate([batch_column, coordinates], axis=1))
        target_features.append(features)

    @staticmethod
    def _concat_or_empty(values, shape, dtype):
        return np.concatenate(values, axis=0) if values else np.empty(shape, dtype)

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        _require_nonempty(batch)
        vertical_coords, vertical_features = [], []
        horizontal_coords, horizontal_features = [], []
        vertical_present, horizontal_present, counts = [], [], []
        vertical_sites, horizontal_sites = [], []
        for event_index, sample in enumerate(batch):
            event = validate_native_scifi_hits(sample)
            if not len(event["qdc"]):
                raise ValueError("Two-view sparse ResNet cannot process an empty event")
            v_coords, v_features = build_sparse_view(
                event, VERTICAL, self.normalization
            )
            h_coords, h_features = build_sparse_view(
                event, HORIZONTAL, self.normalization
            )
            self._append_view(
                v_coords, v_features, event_index, vertical_coords, vertical_features
            )
            self._append_view(
                h_coords, h_features, event_index, horizontal_coords,
                horizontal_features,
            )
            vertical_present.append(bool(len(v_coords)))
            horizontal_present.append(bool(len(h_coords)))
            vertical_sites.append(len(v_coords))
            horizontal_sites.append(len(h_coords))
            counts.append(len(event["qdc"]))
        result = _metadata(batch)
        result.update({
            "vertical_coords": torch.from_numpy(self._concat_or_empty(
                vertical_coords, (0, 3), np.int32
            )),
            "vertical_features": torch.from_numpy(self._concat_or_empty(
                vertical_features, (0, 2), np.float32
            )),
            "horizontal_coords": torch.from_numpy(self._concat_or_empty(
                horizontal_coords, (0, 3), np.int32
            )),
            "horizontal_features": torch.from_numpy(self._concat_or_empty(
                horizontal_features, (0, 2), np.float32
            )),
            "vertical_present": torch.tensor(vertical_present, dtype=torch.bool),
            "horizontal_present": torch.tensor(horizontal_present, dtype=torch.bool),
            "n_nodes": torch.tensor(counts, dtype=torch.long),
            "n_vertical_sites": torch.tensor(vertical_sites, dtype=torch.long),
            "n_horizontal_sites": torch.tensor(horizontal_sites, dtype=torch.long),
            "model_family": "two_view_sparse_resnet",
        })
        return result
