"""Streaming training-only normalization and SciFi size diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_3d.data.scifi_hit_event_dataset import SciFiNormalization
from train_3d.data.native_scifi import (
    NativeSciFiNormalization,
    measured_coordinate,
)


class RunningMoments:
    def __init__(self, dimensions: int):
        self.dimensions = int(dimensions)
        self.count = 0
        self.mean = np.zeros(self.dimensions, dtype=np.float64)
        self.m2 = np.zeros(self.dimensions, dtype=np.float64)
        self.minimum = np.full(self.dimensions, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimensions, -np.inf, dtype=np.float64)

    def update(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dimensions)
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError("Cannot accumulate non-finite statistics")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta ** 2 * self.count * batch_count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def merge(self, other: "RunningMoments") -> None:
        if self.dimensions != other.dimensions:
            raise ValueError("Cannot merge moments with different dimensions")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / total
        self.m2 += other.m2 + delta ** 2 * self.count * other.count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)

    def to_dict(self) -> dict:
        """Return a lossless JSON-compatible accumulator payload."""
        return {
            "dimensions": self.dimensions,
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "minimum": self.minimum.tolist() if self.count else None,
            "maximum": self.maximum.tolist() if self.count else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunningMoments":
        result = cls(int(payload["dimensions"]))
        result.count = int(payload["count"])
        for name in ("mean", "m2"):
            value = np.asarray(payload[name], dtype=np.float64)
            if value.shape != (result.dimensions,):
                raise ValueError(f"Invalid {name} shape in running moments")
            setattr(result, name, value)
        if result.count < 0:
            raise ValueError("Running-moment count cannot be negative")
        for name, empty_value in (("minimum", np.inf), ("maximum", -np.inf)):
            payload_value = payload[name]
            value = (
                np.full(result.dimensions, empty_value, dtype=np.float64)
                if payload_value is None else np.asarray(payload_value, dtype=np.float64)
            )
            if value.shape != (result.dimensions,):
                raise ValueError(f"Invalid {name} shape in running moments")
            setattr(result, name, value)
        return result

    @property
    def scale(self):
        if self.count < 2:
            return np.ones(self.dimensions, dtype=np.float64)
        scale = np.sqrt(self.m2 / self.count)
        return np.where(scale > 0, scale, 1.0)


@dataclass
class SciFiStatistics:
    dataset_version: str
    cv_fold: int

    def __post_init__(self):
        self.position = RunningMoments(3)
        self.qdc_log = RunningMoments(1)
        self.events = 0
        self.empty_events = 0

    def update(self, event: dict) -> None:
        position = np.asarray(event["position"], dtype=np.float32)
        qdc = np.asarray(event["qdc"], dtype=np.float32)
        self.events += 1
        if not len(qdc):
            self.empty_events += 1
            return
        if np.any(qdc < 0):
            raise ValueError("QDC statistics received a negative value")
        self.position.update(position)
        self.qdc_log.update(np.log1p(qdc)[:, None])

    def merge(self, other: "SciFiStatistics") -> None:
        if self.dataset_version != other.dataset_version:
            raise ValueError("Cannot merge different dataset versions")
        self.position.merge(other.position)
        self.qdc_log.merge(other.qdc_log)
        self.events += other.events
        self.empty_events += other.empty_events

    def to_dict(self) -> dict:
        return {
            "dataset_version": self.dataset_version,
            "cv_fold": int(self.cv_fold),
            "events": int(self.events),
            "empty_events": int(self.empty_events),
            "position": self.position.to_dict(),
            "qdc_log": self.qdc_log.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SciFiStatistics":
        result = cls(str(payload["dataset_version"]), int(payload["cv_fold"]))
        result.events = int(payload["events"])
        result.empty_events = int(payload["empty_events"])
        result.position = RunningMoments.from_dict(payload["position"])
        result.qdc_log = RunningMoments.from_dict(payload["qdc_log"])
        if result.events < 0 or result.empty_events < 0:
            raise ValueError("SciFi event counts cannot be negative")
        if result.empty_events > result.events:
            raise ValueError("Empty-event count exceeds event count")
        return result

    def normalization(self) -> SciFiNormalization:
        if not self.position.count or not self.qdc_log.count:
            raise ValueError("Cannot build normalization without SciFi candidates")
        return SciFiNormalization(
            version="scifi_xyz_qdc_dettype_v1",
            dataset_version=self.dataset_version,
            cv_fold=int(self.cv_fold),
            position_mean=tuple(float(value) for value in self.position.mean),
            position_scale=tuple(float(value) for value in self.position.scale),
            qdc_transform="log1p",
            qdc_mean=float(self.qdc_log.mean[0]),
            qdc_scale=float(self.qdc_log.scale[0]),
            training_events=int(self.events),
            training_candidates=int(self.position.count),
        )

    def coordinate_bounds(self, padding) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not self.position.count:
            raise ValueError("Cannot build coordinate bounds without candidates")
        padding = np.broadcast_to(np.asarray(padding, dtype=float), (3,))
        if np.any(padding < 0):
            raise ValueError("Coordinate padding cannot be negative")
        lower = self.position.minimum - padding
        upper = self.position.maximum + padding
        return tuple(map(float, lower)), tuple(map(float, upper))


def count_sparse_voxels(position, coordinate_min, coordinate_max, voxel_size) -> int:
    position = np.asarray(position, dtype=np.float32)
    if not len(position):
        return 0
    lower = np.asarray(coordinate_min, dtype=np.float32)
    if not np.isfinite(position).all():
        raise ValueError("Sparse coordinate input contains non-finite values")
    coordinates = np.floor(
        (position - lower) / np.asarray(voxel_size, dtype=np.float32)
    ).astype(np.int32)
    return int(len(np.unique(coordinates, axis=0)))


@dataclass
class NativeSciFiStatistics:
    """Streaming training-only moments for detector-native SciFi hits."""

    dataset_version: str
    cv_fold: int

    def __post_init__(self):
        self.u = RunningMoments(1)
        self.z = RunningMoments(1)
        self.qdc_log = RunningMoments(1)
        self.channel = RunningMoments(1)
        self.station = RunningMoments(1)
        self.events = 0
        self.empty_events = 0

    def update(self, event: dict) -> None:
        self.events += 1
        if not len(event["qdc"]):
            self.empty_events += 1
            return
        self.u.update(measured_coordinate(event)[:, None])
        self.z.update(np.asarray(event["z"], dtype=np.float32)[:, None])
        self.qdc_log.update(np.log1p(event["qdc"])[:, None])
        self.channel.update(np.asarray(event["channel"])[:, None])
        self.station.update(np.asarray(event["station"])[:, None])

    def merge(self, other: "NativeSciFiStatistics") -> None:
        if self.dataset_version != other.dataset_version:
            raise ValueError("Cannot merge different dataset versions")
        for name in ("u", "z", "qdc_log", "channel", "station"):
            getattr(self, name).merge(getattr(other, name))
        self.events += other.events
        self.empty_events += other.empty_events

    def normalization(self) -> NativeSciFiNormalization:
        if not self.u.count:
            raise ValueError("Cannot build normalization without native SciFi hits")
        # Fixed detector edges preserve one global meaning across folds while
        # still recording the observed range in metadata summaries.
        return NativeSciFiNormalization(
            version="scifi_native_hit_normalization_v1",
            dataset_version=self.dataset_version,
            cv_fold=int(self.cv_fold),
            u_mean=float(self.u.mean[0]), u_scale=float(self.u.scale[0]),
            z_mean=float(self.z.mean[0]), z_scale=float(self.z.scale[0]),
            qdc_transform="log1p",
            qdc_mean=float(self.qdc_log.mean[0]),
            qdc_scale=float(self.qdc_log.scale[0]),
            channel_min=-0.5, channel_max=1535.5,
            station_min=0.5, station_max=5.5,
            training_events=int(self.events),
            training_native_hits=int(self.u.count),
        )
