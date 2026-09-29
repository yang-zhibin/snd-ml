"""Compact, checksummed metadata for the data-driven SciFi classifiers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
METADATA_VERSION = "scifi_training_metadata_v2"


def file_sha256(path, chunk_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stage_binary_metadata(root, cv_fold: int, representation: str, cache_parent) -> Path:
    """Atomically stage only one model family's mmap inputs to local storage."""
    import fcntl

    root = Path(root)
    complete_path = root / "complete.json"
    complete_bytes = complete_path.read_bytes()
    identity = hashlib.sha256(complete_bytes).hexdigest()
    cache = Path(cache_parent) / f"snd_ml_scifi_metadata_{identity[:16]}"
    cache.mkdir(parents=True, exist_ok=True)
    size_name = {
        "gravnet": "n_candidates.npy",
        "sparse": f"n_voxels_fold_{int(cv_fold)}.npy",
        "native": "n_native_hits.npy",
        "two_view_sparse": "n_sparse_sites.npy",
    }.get(representation)
    if size_name is None:
        raise ValueError("Unsupported event-size representation")
    names = ("index.json", "source_entries.npy", size_name)
    complete = json.loads(complete_bytes)
    with (cache / ".stage.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker_path = cache / "stage_complete.json"
        valid = False
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                valid = marker.get("metadata_complete_sha256") == identity and all(
                    (cache / name).is_file()
                    and (cache / name).stat().st_size == int(
                        complete["products"][name]["bytes"]
                    ) for name in names
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                valid = False
        if not valid:
            for name in names:
                source = root / name
                temporary = cache / f".{name}.{os.getpid()}.tmp"
                shutil.copyfile(source, temporary)
                expected = complete["products"][name]["sha256"]
                if file_sha256(temporary) != expected:
                    temporary.unlink()
                    raise ValueError(f"Checksum mismatch while staging {name}")
                os.replace(temporary, cache / name)
            (cache / "complete.json").write_bytes(complete_bytes)
            write_json_atomic(marker_path, {
                "metadata_complete_sha256": identity,
                "source_root": str(root.resolve()),
                "files": list(names),
            })
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return cache


def write_json_atomic(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_npy_atomic(path, values) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_selected_entries(record: dict) -> np.ndarray:
    """Load and canonicalize selected source entries for one ROOT file."""
    count = int(record["n_events"])
    expression = str(record.get("event_filter", "true")).strip().lower()
    if expression in {"", "true", "1"}:
        return np.arange(count, dtype=np.int64)
    path = str(record.get("entry_index_path", "")).strip()
    if not path:
        raise ValueError(
            f"Filtered file {record['dataset_file_uid']} has no entry index"
        )
    with np.load(path, allow_pickle=False) as payload:
        entries = np.asarray(payload["entries"])
    if entries.dtype.kind not in "iu" or len(entries) != count:
        raise ValueError(f"Invalid entry index for {record['dataset_file_uid']}")
    entries = entries.astype(np.int64, copy=False)
    if len(entries) and (entries[0] < 0 or np.any(entries[1:] <= entries[:-1])):
        raise ValueError(
            f"Entry index must be non-negative, unique and sorted for "
            f"{record['dataset_file_uid']}"
        )
    return entries


def iter_scifi_events(record: dict, tree_name: str, step_size: int):
    """Yield selected events in ascending source-entry order using bounded reads."""
    import uproot
    from train_3d.data.scifi_hit_event_dataset import (
        RAW_HIT_BRANCHES, validate_and_select_scifi,
    )

    if int(step_size) <= 0:
        raise ValueError("step_size must be positive")
    entries = load_selected_entries(record)
    if not len(entries):
        return
    with uproot.open(record["hit3d_path"]) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"Tree {tree_name!r} is missing from {record['hit3d_path']}")
        tree = root_file[tree_name]
        missing = set(RAW_HIT_BRANCHES.values()) - set(tree.keys())
        if missing:
            raise KeyError(f"Hit3D file lacks branches: {sorted(missing)}")
        if int(entries[-1]) >= int(tree.num_entries):
            raise IndexError(
                f"Selected entry {entries[-1]} is outside {record['hit3d_path']}"
            )
        block_ids = np.unique(entries // int(step_size))
        cursor = 0
        branches = list(RAW_HIT_BRANCHES.values())
        for block in block_ids:
            start = int(block) * int(step_size)
            stop = min(start + int(step_size), int(tree.num_entries))
            right = int(np.searchsorted(entries, stop, side="left"))
            selected = entries[cursor:right]
            payload = tree.arrays(
                branches, entry_start=start, entry_stop=stop, library="ak", how=dict
            )
            for source_entry in selected:
                local = int(source_entry) - start
                raw = {
                    name: np.asarray(payload[branch][local])
                    for name, branch in RAW_HIT_BRANCHES.items()
                }
                yield int(source_entry), validate_and_select_scifi(raw)
            cursor = right
    if cursor != len(entries):
        raise RuntimeError(f"Did not read every selected entry for {record['dataset_file_uid']}")


def iter_native_scifi_events(record: dict, tree_name: str, step_size: int):
    """Yield strictly validated native SciFi events using bounded uproot reads."""
    import uproot
    from train_3d.data.native_scifi import (
        NATIVE_SCIFI_BRANCHES, validate_native_scifi_hits,
    )

    if int(step_size) <= 0:
        raise ValueError("step_size must be positive")
    entries = load_selected_entries(record)
    if not len(entries):
        return
    with uproot.open(record["hit3d_path"]) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"Tree {tree_name!r} is missing from {record['hit3d_path']}")
        tree = root_file[tree_name]
        branches = list(NATIVE_SCIFI_BRANCHES.values())
        missing = set(branches) - set(tree.keys())
        if missing:
            raise KeyError(f"Hit3D file lacks native branches: {sorted(missing)}")
        if int(entries[-1]) >= int(tree.num_entries):
            raise IndexError(
                f"Selected entry {entries[-1]} is outside {record['hit3d_path']}"
            )
        block_ids = np.unique(entries // int(step_size))
        cursor = 0
        for block in block_ids:
            start = int(block) * int(step_size)
            stop = min(start + int(step_size), int(tree.num_entries))
            right = int(np.searchsorted(entries, stop, side="left"))
            selected = entries[cursor:right]
            payload = tree.arrays(
                branches, entry_start=start, entry_stop=stop, library="ak", how=dict
            )
            for source_entry in selected:
                local = int(source_entry) - start
                raw = {
                    name: np.asarray(payload[branch][local])
                    for name, branch in NATIVE_SCIFI_BRANCHES.items()
                }
                yield int(source_entry), validate_native_scifi_hits(raw)
            cursor = right
    if cursor != len(entries):
        raise RuntimeError(f"Did not read every selected entry for {record['dataset_file_uid']}")


def unique_file_rows(fold_views: pd.DataFrame, n_folds: int) -> list[dict]:
    """Return every logical dataset file once, in legacy CV-zero order."""
    selected = fold_views.loc[
        pd.to_numeric(fold_views["cv_fold"]).astype(int).eq(0)
    ].copy()
    split_order = pd.Categorical(
        selected["split"].astype(str), categories=["train", "validation"], ordered=True
    )
    selected = selected.assign(_split_order=split_order).sort_values(
        ["_split_order"], kind="stable"
    )
    if selected["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("CV fold zero contains duplicate dataset_file_uid values")
    physical = set(pd.to_numeric(selected["fold"]).astype(int))
    if physical != set(range(int(n_folds))):
        raise ValueError("Fold views do not contain every configured physical fold")
    records = []
    offset = 0
    for ordinal, (_, row) in enumerate(selected.iterrows()):
        count = int(row["n_events"])
        if count < 0:
            raise ValueError("n_events cannot be negative")
        record = {
            "ordinal": ordinal,
            "dataset_file_uid": str(row["dataset_file_uid"]),
            "hit3d_path": str(row["hit3d_path"]),
            "physical_fold": int(row["fold"]),
            "class_name": str(row["class_name"]),
            "domain": str(row["domain"]),
            "n_events": count,
            "event_filter": str(row.get("event_filter", "true")),
            "entry_index_path": str(row.get("entry_index_path", "")),
            "global_offset": offset,
        }
        records.append(record)
        offset += count
    return records


def assign_shards(records: list[dict], n_shards: int) -> list[int]:
    """Deterministic largest-first file assignment with balanced event totals."""
    if int(n_shards) <= 0:
        raise ValueError("n_shards must be positive")
    loads = [0] * int(n_shards)
    assignment = [0] * len(records)
    for record in sorted(records, key=lambda item: (-item["n_events"], item["ordinal"])):
        shard = min(range(int(n_shards)), key=lambda index: (loads[index], index))
        assignment[int(record["ordinal"])] = shard
        loads[shard] += int(record["n_events"])
    return assignment


def load_plan(path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported SciFi metadata plan schema")
    if payload.get("metadata_version") != METADATA_VERSION:
        raise ValueError("Unexpected SciFi metadata version")
    records = payload.get("files", [])
    if [int(item["ordinal"]) for item in records] != list(range(len(records))):
        raise ValueError("Metadata plan file ordinals are not contiguous")
    expected = 0
    for record in records:
        if int(record["global_offset"]) != expected:
            raise ValueError("Metadata plan offsets are not contiguous")
        expected += int(record["n_events"])
    if expected != int(payload["total_events"]):
        raise ValueError("Metadata plan event total is inconsistent")
    return payload
