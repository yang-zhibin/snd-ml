"""Resumable per-file event-size products for SciFi metadata v2."""

from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile

import numpy as np


SCHEMA_VERSION = 1


def load_event_size_group_plan(
    path,
    *,
    metadata_plan_sha256: str | None = None,
    statistics_complete_sha256: str | None = None,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported event-size group-plan schema")
    if (
        metadata_plan_sha256 is not None
        and payload.get("metadata_plan_sha256") != metadata_plan_sha256
    ):
        raise ValueError("Event-size group plan used a different metadata plan")
    if (
        statistics_complete_sha256 is not None
        and payload.get("statistics_complete_sha256")
        != statistics_complete_sha256
    ):
        raise ValueError("Event-size group plan used different statistics")
    groups = payload.get("groups", [])
    if len(groups) != int(payload.get("n_groups", -1)):
        raise ValueError("Event-size group count is inconsistent")
    if [int(group["group_index"]) for group in groups] != list(range(len(groups))):
        raise ValueError("Event-size group indices are not contiguous")
    return payload


def event_size_cache_path(cache_root, group_index: int, ordinal: int) -> Path:
    return (
        Path(cache_root)
        / f"group_{int(group_index):03d}"
        / f"file_{int(ordinal):05d}.npz"
    )


def write_event_size_file_atomic(
    path,
    *,
    record: dict,
    group_index: int,
    metadata_plan_sha256: str,
    statistics_complete_sha256: str,
    source_entries,
    n_candidates,
    n_voxels,
) -> None:
    """Write one complete file product; the final pathname is the marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
                ordinal=np.asarray(int(record["ordinal"]), dtype=np.int64),
                group_index=np.asarray(int(group_index), dtype=np.int64),
                dataset_file_uid=np.asarray(str(record["dataset_file_uid"])),
                metadata_plan_sha256=np.asarray(str(metadata_plan_sha256)),
                statistics_complete_sha256=np.asarray(
                    str(statistics_complete_sha256)
                ),
                source_entries=np.asarray(source_entries, dtype=np.int64),
                n_candidates=np.asarray(n_candidates, dtype=np.uint32),
                n_voxels=np.asarray(n_voxels, dtype=np.uint32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_event_size_file(
    path,
    *,
    record: dict,
    group_index: int,
    n_folds: int,
    metadata_plan_sha256: str,
    statistics_complete_sha256: str,
) -> dict[str, np.ndarray]:
    """Load and fully validate one atomic per-file event-size product."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema_version", "ordinal", "group_index", "dataset_file_uid",
            "metadata_plan_sha256", "statistics_complete_sha256",
            "source_entries", "n_candidates", "n_voxels",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Event-size cache {path} lacks {sorted(missing)}")
        scalar_checks = {
            "schema_version": SCHEMA_VERSION,
            "ordinal": int(record["ordinal"]),
            "group_index": int(group_index),
            "dataset_file_uid": str(record["dataset_file_uid"]),
            "metadata_plan_sha256": str(metadata_plan_sha256),
            "statistics_complete_sha256": str(statistics_complete_sha256),
        }
        for name, expected in scalar_checks.items():
            value = payload[name]
            if value.shape != () or value.item() != expected:
                raise ValueError(
                    f"Event-size cache {path} has wrong {name}: "
                    f"{value.item() if value.shape == () else value.shape!r}"
                )
        result = {
            "source_entries": np.asarray(payload["source_entries"]),
            "n_candidates": np.asarray(payload["n_candidates"]),
            "n_voxels": np.asarray(payload["n_voxels"]),
        }
    count = int(record["n_events"])
    if result["source_entries"].dtype != np.int64 or result["source_entries"].shape != (count,):
        raise ValueError(f"Event-size cache {path} has invalid source_entries")
    if result["n_candidates"].dtype != np.uint32 or result["n_candidates"].shape != (count,):
        raise ValueError(f"Event-size cache {path} has invalid n_candidates")
    expected_voxel_shape = (int(n_folds), count)
    if result["n_voxels"].dtype != np.uint32 or result["n_voxels"].shape != expected_voxel_shape:
        raise ValueError(f"Event-size cache {path} has invalid n_voxels")
    if len(result["source_entries"]) and np.any(
        result["source_entries"][1:] <= result["source_entries"][:-1]
    ):
        raise ValueError(f"Event-size cache {path} has unsorted source entries")
    candidates = result["n_candidates"][None, :]
    if np.any(result["n_voxels"] > candidates):
        raise ValueError(f"Event-size cache {path} has more voxels than candidates")
    if np.any((result["n_voxels"] == 0) != (candidates == 0)):
        raise ValueError(f"Event-size cache {path} has inconsistent empty events")
    return result


def group_by_ordinal(group_plan: dict) -> dict[int, int]:
    result = {}
    for group in group_plan["groups"]:
        group_index = int(group["group_index"])
        for ordinal in group["file_ordinals"]:
            ordinal = int(ordinal)
            if ordinal in result:
                raise ValueError(f"File ordinal {ordinal} occurs in multiple groups")
            result[ordinal] = group_index
    return result
