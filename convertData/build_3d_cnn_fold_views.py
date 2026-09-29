#!/usr/bin/env python3
"""Expand the natural dataset catalog into five train/validation views."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import load_yaml


def build_fold_views(catalog: pd.DataFrame, density_manifest: pd.DataFrame,
                     n_folds: int, entry_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    if "fold" not in catalog:
        raise ValueError("Folded catalog lacks the fold column")
    actual_folds = set(pd.to_numeric(catalog["fold"]).astype(int))
    expected_folds = set(range(n_folds))
    if actual_folds != expected_folds:
        raise ValueError(f"Expected folds {sorted(expected_folds)}, found {sorted(actual_folds)}")

    index_paths = {}
    if not density_manifest.empty:
        if density_manifest["dataset_file_uid"].astype(str).duplicated().any():
            raise ValueError("Density manifest contains duplicate dataset_file_uid values")
        index_paths = dict(zip(
            density_manifest["dataset_file_uid"].astype(str),
            density_manifest["index_path"].astype(str),
        ))
    entry_paths = {}
    if entry_manifest is not None and not entry_manifest.empty:
        if entry_manifest["dataset_file_uid"].astype(str).duplicated().any():
            raise ValueError("Entry-index manifest contains duplicate dataset_file_uid values")
        entry_paths = dict(zip(
            entry_manifest["dataset_file_uid"].astype(str),
            entry_manifest["entry_index_path"].astype(str),
        ))

    frames = []
    for held_out_fold in range(n_folds):
        view = catalog.copy()
        view.insert(0, "cv_fold", held_out_fold)
        view.insert(1, "split", "train")
        held_out = pd.to_numeric(view["fold"]).astype(int).eq(held_out_fold)
        view.loc[held_out, "split"] = "validation"
        view.insert(2, "sampling_strategy", "natural")
        uniform = view["split"].eq("train") & view["class_name"].eq("muon_DIS")
        view.loc[uniform, "sampling_strategy"] = "uniform_populated_density_bins"
        view.insert(3, "density_index_path", view["dataset_file_uid"].astype(str).map(
            index_paths
        ).fillna(""))
        view.insert(4, "entry_index_path", view["dataset_file_uid"].astype(str).map(
            entry_paths
        ).fillna(""))
        missing = view["class_name"].eq("muon_DIS") & view["density_index_path"].eq("")
        if missing.any():
            identifiers = view.loc[missing, "dataset_file_uid"].astype(str).head().tolist()
            raise ValueError(f"Muon-DIS rows lack density indices: {identifiers}")
        filters = view.get("event_filter", pd.Series("true", index=view.index)).astype(str)
        filtered = ~filters.str.strip().str.lower().isin(["", "true", "1"])
        missing_entries = filtered & view["entry_index_path"].eq("")
        if missing_entries.any():
            identifiers = view.loc[missing_entries, "dataset_file_uid"].astype(str).head().tolist()
            raise ValueError(f"Filtered rows lack physical entry indices: {identifiers}")
        frames.append(view)
    result = pd.concat(frames, ignore_index=True)
    if result.groupby("dataset_file_uid")["cv_fold"].nunique().min() != n_folds:
        raise RuntimeError("Not every dataset row occurs exactly once in every CV view")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folded-catalog", required=True)
    parser.add_argument("--density-index-manifest", required=True)
    parser.add_argument("--entry-index-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = build_fold_views(
        pd.read_csv(args.folded_catalog, keep_default_na=False),
        pd.read_csv(args.density_index_manifest, keep_default_na=False),
        int(config["cross_validation"]["n_folds"]),
        pd.read_csv(args.entry_index_manifest, keep_default_na=False)
        if args.entry_index_manifest else pd.DataFrame(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rows covering all cross-validation views")


if __name__ == "__main__":
    main()
