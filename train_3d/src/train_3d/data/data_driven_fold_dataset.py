"""Leakage-safe event references for the data-driven five-fold classifier.

Training draws with replacement using the hierarchy class -> domain -> event.
For muon-DIS the last step is density bin -> event, with populated density
bins chosen uniformly. Validation is never sampled and visits each event once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def distributed_rank_world():
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
    except ImportError:
        pass
    return 0, 1


@dataclass(frozen=True)
class EventReference:
    dataset_file_uid: str
    hit3d_path: str
    source_entry: int
    class_id: int
    class_name: str
    domain: str
    density_bin: str = ""


class _Pool:
    def __init__(self, candidates):
        self.candidates = candidates
        self.cumulative = np.cumsum([size for _, size, _ in candidates], dtype=np.int64)
        if not len(self.cumulative) or int(self.cumulative[-1]) <= 0:
            raise ValueError("Cannot construct an empty event pool")

    @property
    def size(self) -> int:
        return int(self.cumulative[-1])

    def locate(self, flat_index: int):
        selected = int(np.searchsorted(self.cumulative, flat_index, side="right"))
        previous = int(self.cumulative[selected - 1]) if selected else 0
        row, _, entries = self.candidates[selected]
        local = flat_index - previous
        source_entry = local if entries is None else int(entries[local])
        return row, int(source_entry)


def _probabilities(names, requested, natural_sizes):
    names = list(sorted(names))
    if requested is None:
        weights = np.asarray([natural_sizes[name] for name in names], dtype=float)
    elif requested == "uniform":
        weights = np.ones(len(names), dtype=float)
    else:
        extra = set(requested) - set(names)
        if extra:
            raise ValueError(f"Probabilities reference unavailable values: {sorted(extra)}")
        weights = np.asarray([float(requested.get(name, 0.0)) for name in names])
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Probabilities must be finite, non-negative, and non-zero")
    return names, weights / weights.sum()


def _entry_candidates(row: pd.Series):
    n_events = int(row["n_events"])
    expression = str(row.get("event_filter", "true")).strip().lower()
    if expression in {"", "true", "1"}:
        return n_events, None
    path = str(row.get("entry_index_path", "")).strip()
    if not path:
        raise ValueError(f"Filtered row {row['dataset_file_uid']} lacks entry_index_path")
    with np.load(Path(path), allow_pickle=False) as payload:
        entries = np.asarray(payload["entries"])
    if entries.dtype.kind not in "iu" or len(entries) != n_events:
        raise ValueError(f"Invalid entry index for {row['dataset_file_uid']}")
    return n_events, entries.astype(np.int64, copy=False)


def select_fold_rows(fold_views: pd.DataFrame, cv_fold: int, split: str) -> pd.DataFrame:
    selected = fold_views.loc[
        pd.to_numeric(fold_views["cv_fold"]).astype(int).eq(int(cv_fold))
        & fold_views["split"].astype(str).eq(split)
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows for cv_fold={cv_fold}, split={split}")
    if selected["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Fold view contains duplicate logical dataset rows")
    expected_validation = pd.to_numeric(selected["fold"]).astype(int).eq(int(cv_fold))
    if split == "validation" and not expected_validation.all():
        raise ValueError("Validation contains a non-held-out source group")
    if split == "train" and expected_validation.any():
        raise ValueError("Training contains a held-out source group")
    return selected.reset_index(drop=True)


class HierarchicalTrainingSampler:
    """Draw a reproducible training epoch without changing physics weights."""

    def __init__(self, fold_views: pd.DataFrame, cv_fold: int, seed: int = 0,
                 class_probabilities=None, domain_probabilities=None,
                 muondis_density_probabilities="uniform"):
        self.rows = select_fold_rows(fold_views, cv_fold, "train")
        self.cv_fold = int(cv_fold)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.pools = {}
        self.density_pools = {}
        for _, row in self.rows.iterrows():
            key = (str(row["class_name"]), str(row["domain"]))
            if key[0] == "muon_DIS":
                path = Path(str(row["density_index_path"]))
                with np.load(path, allow_pickle=False) as payload:
                    for bin_name in sorted(name for name in payload.files if name.startswith("bin_")):
                        entries = np.asarray(payload[bin_name])
                        if entries.dtype.kind not in "iu":
                            raise TypeError(f"Non-integral density index {path}:{bin_name}")
                        if len(entries):
                            self.density_pools.setdefault((*key, bin_name), []).append(
                                (row, len(entries), entries.astype(np.int64, copy=False))
                            )
            else:
                size, entries = _entry_candidates(row)
                if size:
                    self.pools.setdefault(key, []).append((row, size, entries))
        self.pools = {key: _Pool(value) for key, value in self.pools.items()}
        self.density_pools = {
            key: _Pool(value) for key, value in self.density_pools.items()
        }
        available_classes = {key[0] for key in self.pools} | {key[0] for key in self.density_pools}
        if not available_classes:
            raise ValueError("Training fold contains no usable events")
        class_sizes = {
            name: sum(pool.size for key, pool in self.pools.items() if key[0] == name)
            + sum(pool.size for key, pool in self.density_pools.items() if key[0] == name)
            for name in available_classes
        }
        self.class_choice = _probabilities(available_classes, class_probabilities, class_sizes)
        self.domain_choices = {}
        self.density_choices = {}
        domain_probabilities = domain_probabilities or {}
        for class_name in available_classes:
            domains = {key[1] for key in self.pools if key[0] == class_name} | {
                key[1] for key in self.density_pools if key[0] == class_name
            }
            sizes = {
                domain: sum(pool.size for key, pool in self.pools.items()
                            if key[:2] == (class_name, domain))
                + sum(pool.size for key, pool in self.density_pools.items()
                      if key[:2] == (class_name, domain))
                for domain in domains
            }
            self.domain_choices[class_name] = _probabilities(
                domains, domain_probabilities.get(class_name), sizes
            )
            if class_name == "muon_DIS":
                for domain in domains:
                    bins = {
                        key[2] for key in self.density_pools
                        if key[:2] == (class_name, domain)
                    }
                    bin_sizes = {
                        bin_name: self.density_pools[
                            (class_name, domain, bin_name)
                        ].size
                        for bin_name in bins
                    }
                    self.density_choices[(class_name, domain)] = _probabilities(
                        bins, muondis_density_probabilities, bin_sizes
                    )

    def sample_one(self, rng=None) -> EventReference:
        rng = self.rng if rng is None else rng
        classes, class_p = self.class_choice
        class_name = str(rng.choice(classes, p=class_p))
        domains, domain_p = self.domain_choices[class_name]
        domain = str(rng.choice(domains, p=domain_p))
        density_bin = ""
        if class_name == "muon_DIS":
            bins, bin_p = self.density_choices[(class_name, domain)]
            density_bin = str(rng.choice(bins, p=bin_p))
            pool = self.density_pools[(class_name, domain, density_bin)]
        else:
            pool = self.pools[(class_name, domain)]
        row, source_entry = pool.locate(int(rng.integers(pool.size)))
        return EventReference(
            dataset_file_uid=str(row["dataset_file_uid"]),
            hit3d_path=str(row["hit3d_path"]), source_entry=source_entry,
            class_id=int(row["class_id"]), class_name=class_name, domain=domain,
            density_bin=density_bin,
        )

    def sample(self, epoch_size: int) -> list[EventReference]:
        if epoch_size < 0:
            raise ValueError("epoch_size cannot be negative")
        return [self.sample_one() for _ in range(epoch_size)]

    def sample_epoch(self, epoch_size: int, epoch: int) -> list[EventReference]:
        """Return a fold/epoch-specific sample independent of call history."""
        if epoch_size < 0:
            raise ValueError("epoch_size cannot be negative")
        if int(epoch) < 0:
            raise ValueError("epoch cannot be negative")
        seed = np.random.SeedSequence([self.base_seed, self.cv_fold, int(epoch)])
        rng = np.random.default_rng(seed)
        return [self.sample_one(rng) for _ in range(epoch_size)]


class NaturalFoldEvents:
    """Iterate one fold-view split exactly once in catalog/event order."""

    def __init__(self, fold_views: pd.DataFrame, cv_fold: int,
                 split: str = "validation"):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.split = split
        self.rows = select_fold_rows(fold_views, cv_fold, split)
        if split == "validation" and not self.rows["sampling_strategy"].astype(str).eq("natural").all():
            raise ValueError("Validation sampling strategy must be natural")
        candidates = []
        self.density_indices = {}
        for _, row in self.rows.iterrows():
            size, entries = _entry_candidates(row)
            if size:
                candidates.append((row, size, entries))
            density_path = Path(str(row.get("density_index_path", "")))
            if str(row["class_name"]) == "muon_DIS" and density_path.is_file():
                with np.load(density_path, allow_pickle=False) as payload:
                    self.density_indices[str(row["dataset_file_uid"])] = [
                        (name, np.sort(np.asarray(payload[name], dtype=np.int64)))
                        for name in sorted(payload.files)
                        if name.startswith("bin_") or name == "invalid"
                    ]
        self.pool = _Pool(candidates)

    def _density_bin(self, row, source_entry):
        for name, entries in self.density_indices.get(
            str(row["dataset_file_uid"]), ()
        ):
            index = int(np.searchsorted(entries, source_entry))
            if index < len(entries) and int(entries[index]) == int(source_entry):
                return name
        return ""

    def __len__(self) -> int:
        return self.pool.size

    def __getitem__(self, index: int) -> EventReference:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row, source_entry = self.pool.locate(int(index))
        return EventReference(
            dataset_file_uid=str(row["dataset_file_uid"]),
            hit3d_path=str(row["hit3d_path"]), source_entry=source_entry,
            class_id=int(row["class_id"]), class_name=str(row["class_name"]),
            domain=str(row["domain"]),
            density_bin=self._density_bin(row, source_entry),
        )

    def __iter__(self) -> Iterator[EventReference]:
        for index in range(len(self)):
            yield self[index]


class NaturalValidationEvents(NaturalFoldEvents):
    """Backward-compatible natural held-out-fold event sequence."""

    def __init__(self, fold_views: pd.DataFrame, cv_fold: int):
        super().__init__(fold_views, cv_fold, split="validation")


class DeterministicStratifiedValidationEvents:
    """Fixed held-out validation subset with an exact per-class quota.

    Selection is performed before node-budget batching, so changing batch
    limits cannot change the validation population. Candidate events are drawn
    without replacement from each class and interleaved in configured class
    order. The optional ``accept`` predicate is used to exclude events that
    cannot form the requested model representation.
    """

    def __init__(self, fold_views: pd.DataFrame, cv_fold: int,
                 class_names, total_size: int, seed: int = 0, accept=None):
        class_names = tuple(str(name) for name in class_names)
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be a non-empty unique sequence")
        if int(total_size) <= 0:
            raise ValueError("Stratified validation size must be positive")

        self.rows = select_fold_rows(fold_views, cv_fold, "validation")
        if not self.rows["sampling_strategy"].astype(str).eq("natural").all():
            raise ValueError("Stratified validation requires natural source rows")
        available = set(self.rows["class_name"].astype(str))
        missing = set(class_names) - available
        if missing:
            raise ValueError(
                f"Validation fold lacks configured classes: {sorted(missing)}"
            )

        density_indices = {}
        pools = {}
        for _, row in self.rows.iterrows():
            size, entries = _entry_candidates(row)
            if size:
                pools.setdefault(str(row["class_name"]), []).append(
                    (row, size, entries)
                )
            density_path = Path(str(row.get("density_index_path", "")))
            if str(row["class_name"]) == "muon_DIS" and density_path.is_file():
                with np.load(density_path, allow_pickle=False) as payload:
                    density_indices[str(row["dataset_file_uid"])] = [
                        (name, np.sort(np.asarray(payload[name], dtype=np.int64)))
                        for name in sorted(payload.files)
                        if name.startswith("bin_") or name == "invalid"
                    ]
        pools = {name: _Pool(candidates) for name, candidates in pools.items()}

        def make_reference(pool, flat_index):
            row, source_entry = pool.locate(int(flat_index))
            density_bin = ""
            for name, entries in density_indices.get(
                str(row["dataset_file_uid"]), ()
            ):
                index = int(np.searchsorted(entries, source_entry))
                if index < len(entries) and int(entries[index]) == source_entry:
                    density_bin = name
                    break
            return EventReference(
                dataset_file_uid=str(row["dataset_file_uid"]),
                hit3d_path=str(row["hit3d_path"]),
                source_entry=int(source_entry), class_id=int(row["class_id"]),
                class_name=str(row["class_name"]), domain=str(row["domain"]),
                density_bin=density_bin,
            )

        base_quota, remainder = divmod(int(total_size), len(class_names))
        selected_by_class = []
        for class_index, class_name in enumerate(class_names):
            quota = base_quota + int(class_index < remainder)
            pool = pools.get(class_name)
            if pool is None or pool.size < quota:
                available_count = 0 if pool is None else pool.size
                raise ValueError(
                    f"Validation class {class_name!r} has {available_count} events, "
                    f"fewer than its requested quota {quota}"
                )
            # Independent per-class streams keep selections stable if catalog
            # sizes in another class change.
            rng = np.random.default_rng(np.random.SeedSequence([
                int(seed), int(cv_fold), int(class_index), 0x51A7,
            ]))
            probe_size = min(pool.size, max(quota * 2, quota + 64))
            # Avoid ``choice(..., replace=False)`` here: some NumPy versions
            # allocate an array proportional to the complete (potentially very
            # large) validation pool. The requested subset is small, so direct
            # rejection sampling keeps memory proportional to the subset.
            if probe_size > pool.size // 2:
                probe_indices = list(map(int, rng.permutation(pool.size)[:probe_size]))
                probed = set(probe_indices)
            else:
                probed = set()
                probe_indices = []
                while len(probe_indices) < probe_size:
                    flat_index = int(rng.integers(pool.size))
                    if flat_index not in probed:
                        probed.add(flat_index)
                        probe_indices.append(flat_index)
            accepted = []
            for flat_index in probe_indices:
                reference = make_reference(pool, flat_index)
                if accept is None or accept(reference):
                    accepted.append(reference)
                    if len(accepted) == quota:
                        break
            # Empty events are normally rare. A deterministic exhaustive
            # fallback guarantees exact quotas when the initial probe happens
            # to contain too many unusable events.
            if len(accepted) < quota:
                for flat_index in range(pool.size):
                    if flat_index in probed:
                        continue
                    reference = make_reference(pool, flat_index)
                    if accept is None or accept(reference):
                        accepted.append(reference)
                        if len(accepted) == quota:
                            break
            if len(accepted) < quota:
                raise ValueError(
                    f"Validation class {class_name!r} has only {len(accepted)} "
                    f"usable events, fewer than its requested quota {quota}"
                )
            selected_by_class.append(accepted)

        # Round-robin interleaving prevents a partial diagnostic iteration from
        # silently seeing only the first class, while the complete set remains
        # exactly stratified.
        self.references = []
        for offset in range(max(map(len, selected_by_class))):
            for selected in selected_by_class:
                if offset < len(selected):
                    self.references.append(selected[offset])
        if len(self.references) != int(total_size):
            raise RuntimeError("Stratified validation constructed the wrong size")

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> EventReference:
        return self.references[index]

    def __iter__(self) -> Iterator[EventReference]:
        return iter(self.references)


class DeterministicTrainingReferenceSampler:
    """Yield sampled references in the main process for the selected epoch.

    DataLoader treats yielded :class:`EventReference` instances as indices.
    Workers therefore receive the new epoch references even when persistent
    workers are enabled; no mutable sampled list lives inside worker datasets.
    """

    def __init__(self, sampler: HierarchicalTrainingSampler, epoch_size: int,
                 epoch: int = 0, rank: int | None = None,
                 world_size: int | None = None, accept=None):
        if int(epoch_size) <= 0:
            raise ValueError("epoch_size must be positive")
        self.sampler = sampler
        self.epoch_size = int(epoch_size)
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be supplied together")
        self.fixed_rank_world = None if rank is None else (int(rank), int(world_size))
        self.accept = accept
        if self.fixed_rank_world is not None:
            rank, world_size = self.fixed_rank_world
            if world_size <= 0 or rank < 0 or rank >= world_size:
                raise ValueError("Require 0 <= rank < world_size")
        self.set_epoch(epoch)

    def _rank_world(self):
        return self.fixed_rank_world or distributed_rank_world()

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)
        self._cached_key = None
        self._cached_references = None

    def __len__(self) -> int:
        _, world_size = self._rank_world()
        return (self.epoch_size + world_size - 1) // world_size

    def __iter__(self):
        rank, world_size = self._rank_world()
        cache_key = (self.epoch, rank, world_size)
        if self._cached_key == cache_key:
            yield from self._cached_references
            return
        local_size = len(self)
        # Pad deterministically to equal rank lengths, preventing DDP step
        # mismatches. At most world_size-1 references are repeated globally.
        global_size = local_size * world_size
        requested = global_size
        references = self.sampler.sample_epoch(requested, self.epoch)
        if self.accept is not None:
            accepted = [reference for reference in references if self.accept(reference)]
            while len(accepted) < global_size:
                requested *= 2
                if requested > max(global_size * 1024, 1_000_000):
                    raise RuntimeError("Could not sample enough non-empty model events")
                references = self.sampler.sample_epoch(requested, self.epoch)
                accepted = [reference for reference in references if self.accept(reference)]
            references = accepted[:global_size]
        local = references[rank:global_size:world_size]
        self._cached_key = cache_key
        self._cached_references = local
        yield from local


class RankSlicedReferences:
    """Disjoint rank-strided view used for exhaustive DDP validation."""

    def __init__(self, references, rank: int | None = None,
                 world_size: int | None = None):
        self.references = references
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be supplied together")
        self.fixed_rank_world = None if rank is None else (int(rank), int(world_size))

    def _rank_world(self):
        return self.fixed_rank_world or distributed_rank_world()

    def __len__(self):
        rank, world_size = self._rank_world()
        remaining = len(self.references) - rank
        return max(0, (remaining + world_size - 1) // world_size)

    def __iter__(self):
        rank, world_size = self._rank_world()
        for index in range(rank, len(self.references), world_size):
            yield self.references[index]


class SampledTrainingEvents:
    """Indexable references for one deterministic sampled training epoch."""

    def __init__(self, sampler: HierarchicalTrainingSampler, epoch_size: int,
                 epoch: int = 0):
        if int(epoch_size) <= 0:
            raise ValueError("epoch_size must be positive")
        self.sampler = sampler
        self.epoch_size = int(epoch_size)
        self.epoch = -1
        self.references = []
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)
        self.references = self.sampler.sample_epoch(self.epoch_size, self.epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> EventReference:
        return self.references[index]


def luminosity_exposure(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Sum file luminosity once per unique source, never once per class/event."""
    required = {*group_columns, "lumi_source_uid", "lumi_per_file"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Luminosity rows lack columns: {sorted(missing)}")
    values = rows[list(group_columns) + ["lumi_source_uid", "lumi_per_file"]].drop_duplicates()
    consistency = values.groupby("lumi_source_uid")["lumi_per_file"].nunique()
    if (consistency > 1).any():
        raise ValueError("One luminosity source has inconsistent luminosity values")
    values = values.drop_duplicates(list(group_columns) + ["lumi_source_uid"])
    return values.groupby(group_columns, dropna=False, as_index=False)["lumi_per_file"].sum()
