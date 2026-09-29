"""Deterministic training-only sampling of muon-DIS events by SciFi density.

Choose a populated density bin uniformly, then an event uniformly within that
bin. Sampling is with replacement so rare bins do not shorten an epoch.
Validation must bypass this sampler and retain its natural distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventReference:
    dataset_file_uid: str
    hit3d_path: str
    source_entry: int
    density_bin: str
    domain: str


class UniformDensityBinSampler:
    """Sample event references without loading the large ROOT hit payload."""

    def __init__(self, rows: pd.DataFrame, seed: int = 0,
                 domain_probabilities: dict[str, float] | None = None):
        required = {
            "dataset_file_uid", "hit3d_path", "density_index_path", "domain",
            "class_name", "split", "sampling_strategy",
        }
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Sampler rows lack columns: {sorted(missing)}")
        selected = rows.loc[
            rows["class_name"].eq("muon_DIS")
            & rows["split"].eq("train")
            & rows["sampling_strategy"].eq("uniform_populated_density_bins")
        ].copy()
        if selected.empty:
            raise ValueError("No training muon-DIS rows were supplied")
        if selected["dataset_file_uid"].astype(str).duplicated().any():
            raise ValueError("Sampler input contains duplicate dataset_file_uid values")

        self.rows = selected.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        bins_by_domain: dict[str, list[str]] = {}
        raw_pools: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
        for row_index, row in self.rows.iterrows():
            path = Path(str(row["density_index_path"]))
            with np.load(path, allow_pickle=False) as payload:
                for name in sorted(payload.files):
                    if not name.startswith("bin_"):
                        continue
                    values = np.asarray(payload[name])
                    if values.dtype.kind not in "iu":
                        raise TypeError(f"Density index {path}:{name} is not integral")
                    if values.size:
                        domain = str(row["domain"])
                        raw_pools.setdefault((domain, name), []).append((row_index, values))
                        bins_by_domain.setdefault(domain, []).append(name)
        self.bin_names = {
            domain: sorted(set(names)) for domain, names in bins_by_domain.items()
        }
        if not self.bin_names:
            raise ValueError("All configured density bins are empty")
        self.pools = {
            key: self._make_pool(candidates) for key, candidates in raw_pools.items()
        }
        global_candidates: dict[str, list[tuple[int, np.ndarray]]] = {}
        for (_, bin_name), candidates in raw_pools.items():
            global_candidates.setdefault(bin_name, []).extend(candidates)
        self.global_pools = {
            name: self._make_pool(candidates)
            for name, candidates in global_candidates.items()
        }
        self.domain_probabilities = self._normalise_domain_probabilities(domain_probabilities)

    @staticmethod
    def _make_pool(candidates: list[tuple[int, np.ndarray]]):
        cumulative = np.cumsum([len(values) for _, values in candidates], dtype=np.int64)
        return candidates, cumulative

    def _normalise_domain_probabilities(self, requested):
        available = sorted(self.bin_names)
        if requested is None:
            return None
        extra = set(requested) - set(available)
        if extra:
            raise ValueError(f"Requested domains have no indexed events: {sorted(extra)}")
        weights = np.asarray([float(requested.get(domain, 0.0)) for domain in available])
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Domain probabilities must be finite, non-negative, and non-zero")
        weights /= weights.sum()
        return available, weights

    def _choose_domain_and_bin(self) -> tuple[str | None, str]:
        if self.domain_probabilities is not None:
            domains, probabilities = self.domain_probabilities
            domain = str(self.rng.choice(domains, p=probabilities))
            return domain, str(self.rng.choice(self.bin_names[domain]))
        global_bins = sorted({name for names in self.bin_names.values() for name in names})
        return None, str(self.rng.choice(global_bins))

    def sample_one(self) -> EventReference:
        domain, bin_name = self._choose_domain_and_bin()
        candidates, cumulative = (
            self.global_pools[bin_name]
            if domain is None else self.pools[(domain, bin_name)]
        )
        flat_index = int(self.rng.integers(int(cumulative[-1])))
        selected = int(np.searchsorted(cumulative, flat_index, side="right"))
        previous = int(cumulative[selected - 1]) if selected else 0
        row_index, values = candidates[selected]
        source_entry = int(values[flat_index - previous])
        row = self.rows.iloc[row_index]
        return EventReference(
            dataset_file_uid=str(row["dataset_file_uid"]),
            hit3d_path=str(row["hit3d_path"]),
            source_entry=source_entry,
            density_bin=bin_name,
            domain=str(row["domain"]),
        )

    def sample(self, size: int) -> list[EventReference]:
        if size < 0:
            raise ValueError("Sample size cannot be negative")
        return [self.sample_one() for _ in range(size)]
