"""Deterministic detector-space partitions for memory-bounded GravNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GravNetPartitionConfig:
    max_cluster_nodes: int
    cluster_by_station: bool = True
    partition_axes: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        if int(self.max_cluster_nodes) < 1:
            raise ValueError("max_cluster_nodes must be positive")
        axes = tuple(int(axis) for axis in self.partition_axes)
        if not axes or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("partition_axes must contain XYZ axis indices")
        if len(set(axes)) != len(axes):
            raise ValueError("partition_axes must be unique")
        object.__setattr__(self, "partition_axes", axes)


def _partition_group(position: np.ndarray, indices: np.ndarray,
                     config: GravNetPartitionConfig) -> list[np.ndarray]:
    pending = [np.asarray(indices, dtype=np.int64)]
    leaves = []
    while pending:
        current = pending.pop()
        if len(current) <= config.max_cluster_nodes:
            leaves.append(current)
            continue
        spans = np.ptp(position[current][:, config.partition_axes], axis=0)
        axis = config.partition_axes[int(np.argmax(spans))]
        order = np.argsort(position[current, axis], kind="stable")
        ordered = current[order]
        middle = len(ordered) // 2
        # Push right first so the deterministic traversal emits left first.
        pending.append(ordered[middle:])
        pending.append(ordered[:middle])
    return leaves


def partition_event(position, station, config: GravNetPartitionConfig) -> list[np.ndarray]:
    """Return a deterministic, exhaustive list of non-empty core-node indices."""
    position = np.asarray(position, dtype=np.float32)
    station = np.asarray(station)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"position must have shape [N,3], got {position.shape}")
    if station.ndim != 1 or len(station) != len(position):
        raise ValueError("station must have shape [N] aligned with position")
    if not np.isfinite(position).all():
        raise ValueError("position contains non-finite values")
    if not len(position):
        return []
    groups = []
    if config.cluster_by_station:
        for value in np.unique(station):
            groups.append(np.flatnonzero(station == value).astype(np.int64, copy=False))
    else:
        groups.append(np.arange(len(position), dtype=np.int64))
    clusters = []
    for group in groups:
        clusters.extend(_partition_group(position, group, config))
    return clusters
