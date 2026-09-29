"""Reference batching constrained by event count and representation size."""

from __future__ import annotations

from dataclasses import dataclass

import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSize:
    n_candidates: int
    n_voxels: int
    n_native_hits: int | None = None
    n_sparse_sites: int | None = None


class EventSizeIndex:
    REQUIRED = {"dataset_file_uid", "source_entry", "n_candidates", "n_voxels"}

    def __init__(self, rows: pd.DataFrame):
        missing = self.REQUIRED - set(rows.columns)
        if missing:
            raise ValueError(f"Event-size manifest lacks columns: {sorted(missing)}")
        keys = rows[["dataset_file_uid", "source_entry"]].astype(
            {"dataset_file_uid": str, "source_entry": int}
        )
        if keys.duplicated().any():
            raise ValueError("Event-size manifest contains duplicate event identities")
        self._sizes = {}
        self._binary_representation = None
        for _, row in rows.iterrows():
            size = EventSize(
                int(row["n_candidates"]), int(row["n_voxels"]),
                int(row["n_native_hits"]) if "n_native_hits" in rows.columns else None,
                int(row["n_sparse_sites"]) if "n_sparse_sites" in rows.columns else None,
            )
            if size.n_candidates < 0 or size.n_voxels < 0:
                raise ValueError("Event sizes cannot be negative")
            self._sizes[(str(row["dataset_file_uid"]), int(row["source_entry"]))] = size

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path, keep_default_na=False))

    @classmethod
    def from_binary(cls, root, cv_fold: int, representation: str,
                    local_cache_dir=None):
        """Open the compact v2 index without materializing per-event objects."""
        if representation not in {"gravnet", "sparse", "native", "two_view_sparse"}:
            raise ValueError("Unsupported event-size representation")
        root = Path(root)
        if local_cache_dir:
            from train_3d.data.scifi_metadata_v2 import stage_binary_metadata
            root = stage_binary_metadata(
                root, cv_fold, representation, local_cache_dir
            )
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        if int(complete.get("schema_version", -1)) != 2:
            raise ValueError("Unsupported binary event-size schema")
        n_folds = int(index["n_folds"])
        if int(cv_fold) < 0 or int(cv_fold) >= n_folds:
            raise ValueError("cv_fold is outside the binary metadata index")
        result = cls.__new__(cls)
        result._sizes = None
        result._binary_representation = representation
        result._source_entries = np.load(
            root / "source_entries.npy", mmap_mode="r", allow_pickle=False
        )
        size_name = {
            "gravnet": "n_candidates.npy",
            "sparse": f"n_voxels_fold_{int(cv_fold)}.npy",
            "native": "n_native_hits.npy",
            "two_view_sparse": "n_sparse_sites.npy",
        }[representation]
        result._binary_sizes = np.load(
            root / size_name, mmap_mode="r", allow_pickle=False
        )
        total = int(index["total_events"])
        if result._source_entries.shape != (total,) or result._binary_sizes.shape != (total,):
            raise ValueError("Binary event-size arrays have inconsistent shapes")
        result._files = {}
        for item in index["files"]:
            uid = str(item["dataset_file_uid"])
            if uid in result._files:
                raise ValueError(f"Duplicate dataset_file_uid in binary index: {uid}")
            offset, count = int(item["offset"]), int(item["count"])
            if offset < 0 or count < 0 or offset + count > total:
                raise ValueError(f"Invalid binary index segment for {uid}")
            lookup = str(item.get("lookup", "sorted"))
            if lookup not in {"direct", "sorted"}:
                raise ValueError(f"Invalid lookup mode for {uid}: {lookup}")
            result._files[uid] = (offset, count, lookup)
        return result

    def _binary_ordinal(self, reference) -> int:
        uid = str(reference.dataset_file_uid)
        try:
            offset, count, lookup = self._files[uid]
        except KeyError as exc:
            raise KeyError(f"Event-size index lacks dataset file {uid}") from exc
        entry = int(reference.source_entry)
        if lookup == "direct":
            local = entry
        else:
            segment = self._source_entries[offset:offset + count]
            local = int(np.searchsorted(segment, entry))
        if (
            local < 0 or local >= count
            or int(self._source_entries[offset + local]) != entry
        ):
            raise KeyError(f"Event-size index lacks {uid}:{entry}")
        return offset + local

    def size(self, reference, representation: str) -> int:
        if self._binary_representation is not None:
            if representation != self._binary_representation:
                raise ValueError(
                    f"Binary index was opened for {self._binary_representation}, "
                    f"not {representation}"
                )
            return int(self._binary_sizes[self._binary_ordinal(reference)])
        key = (str(reference.dataset_file_uid), int(reference.source_entry))
        try:
            value = self._sizes[key]
        except KeyError as exc:
            raise KeyError(f"Event-size manifest lacks {key[0]}:{key[1]}") from exc
        if representation == "gravnet":
            return value.n_candidates
        if representation == "sparse":
            return value.n_voxels
        if representation == "native":
            if value.n_native_hits is None:
                raise ValueError("Event-size manifest lacks n_native_hits")
            return value.n_native_hits
        if representation == "two_view_sparse":
            if value.n_sparse_sites is None:
                raise ValueError("Event-size manifest lacks n_sparse_sites")
            return value.n_sparse_sites
        raise ValueError("Unsupported event-size representation")


class NodeBudgetBatchSampler:
    """Pack references without crossing either configured batch limit."""

    def __init__(self, references, size_index: EventSizeIndex,
                 representation: str, max_events: int, max_nodes: int):
        if int(max_events) <= 0 or int(max_nodes) <= 0:
            raise ValueError("Batch limits must be positive")
        self.references = references
        self.size_index = size_index
        self.representation = str(representation)
        self.max_events = int(max_events)
        self.max_nodes = int(max_nodes)

    def __iter__(self):
        batch = []
        nodes = 0
        for reference in self.references:
            size = self.size_index.size(reference, self.representation)
            # Empty selected-SciFi events cannot form either representation.
            # They are deterministically excluded from model batches and are
            # counted in the preprocessing summary.
            if size == 0:
                continue
            if batch and (len(batch) >= self.max_events or nodes + size > self.max_nodes):
                yield batch
                batch, nodes = [], 0
            batch.append(reference)
            nodes += size
            # An oversized event is yielded alone; it is never silently dropped.
            if size > self.max_nodes or len(batch) >= self.max_events:
                yield batch
                batch, nodes = [], 0
        if batch:
            yield batch

    def __len__(self):
        # Exact for a stable finite reference iterable; sampled iterables are
        # deterministic within an epoch, so materializing here is acceptable
        # only when a caller explicitly asks for length.
        count = 0
        batch_events = 0
        nodes = 0
        for reference in self.references:
            size = self.size_index.size(reference, self.representation)
            if size == 0:
                continue
            if batch_events and (
                batch_events >= self.max_events or nodes + size > self.max_nodes
            ):
                count += 1
                batch_events, nodes = 0, 0
            batch_events += 1
            nodes += size
            if size > self.max_nodes or batch_events >= self.max_events:
                count += 1
                batch_events, nodes = 0, 0
        return count + int(batch_events > 0)
