#!/usr/bin/env python3
"""Build a stable source-file catalog for the data-driven 3D-CNN workflow."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import (
    finite_float,
    json_compact,
    load_yaml,
    resolve_product_path,
    resolve_repo_path,
    stable_uid,
)


OUTPUT_COLUMNS = [
    "dataset_version", "dataset_key", "role", "domain", "metadata_csv",
    "metadata_row", "source_uid", "fold_group_uid", "lumi_source_uid",
    "data_type", "subfolder", "partition", "run", "production",
    "energy_bin", "n_event_metadata", "lumi_per_file", "feature_path",
    "hit3d_path", "feature_available", "hit3d_available", "input_available",
    "feature_entries", "hit3d_entries", "input_empty", "class_mode",
    "direct_class", "pdg_class_map",
]


def root_tree_entries(path: str, tree_name: str) -> int:
    """Read a ROOT tree entry count without loading event payloads."""
    import uproot

    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"Tree {tree_name!r} is missing from {path}")
        return int(root_file[tree_name].num_entries)


def inspect_product_pair(
    pair: tuple[str, str], feature_tree_name: str, hit3d_tree_name: str,
    inspect_entries: bool = True,
) -> tuple[bool, bool, bool, int, int, bool]:
    """Classify product existence and the valid zero-entry pair edge case."""
    feature_path, hit3d_path = pair
    feature_available = os.path.exists(feature_path)
    hit3d_available = os.path.exists(hit3d_path)
    input_available = feature_available and hit3d_available
    feature_entries = -1
    hit3d_entries = -1
    input_empty = False
    if input_available and inspect_entries:
        feature_entries = root_tree_entries(feature_path, feature_tree_name)
        hit3d_entries = root_tree_entries(hit3d_path, hit3d_tree_name)
        input_empty = feature_entries == 0 and hit3d_entries == 0
    return (
        feature_available,
        hit3d_available,
        input_available,
        feature_entries,
        hit3d_entries,
        input_empty,
    )


def required_columns_for_dataset(definition: dict) -> set[str]:
    required = {
        "data_type", "subfolder", "partition", "n_event", "lumi_per_file",
        "output_base_path", "feature_path", "hit3d_path",
    }
    if str(definition.get("fold_group", "source")) == "run":
        required.add(str(definition.get("run_column", "run")))
    return required


def build_catalog(config: dict, config_path: str | Path) -> pd.DataFrame:
    repo_root = Path(config_path).resolve().parents[4]
    rows = []
    datasets = config.get("source_datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("source_datasets must be a non-empty mapping")

    for dataset_key, definition in datasets.items():
        if not bool(definition.get("enabled", True)):
            continue
        metadata_path = resolve_repo_path(definition["metadata"], repo_root)
        metadata = pd.read_csv(metadata_path)
        missing = required_columns_for_dataset(definition) - set(metadata.columns)
        if missing:
            raise ValueError(
                f"{dataset_key} metadata lacks required columns: {sorted(missing)}"
            )
        for metadata_row, row in metadata.iterrows():
            feature_path = resolve_product_path(row, "feature_path")
            hit3d_path = resolve_product_path(row, "hit3d_path")
            source_uid = stable_uid(
                dataset_key,
                Path(metadata_path).resolve(),
                int(metadata_row),
                feature_path,
                hit3d_path,
            )
            run_column = str(definition.get("run_column", "run"))
            run_value = row.get(run_column, "")
            run = "" if pd.isna(run_value) else str(run_value)
            grouping = str(definition.get("fold_group", "source"))
            if grouping == "run":
                if not run:
                    raise ValueError(f"{dataset_key} row {metadata_row} has no run")
                fold_group_uid = f"data_run_{run}"
            elif grouping == "source":
                fold_group_uid = source_uid
            else:
                raise ValueError(f"Unsupported fold_group {grouping!r}")

            energy_value = row.get(str(definition.get("energy_column", "energy_range")), "")
            energy_bin = "" if pd.isna(energy_value) else str(energy_value)
            production_column = str(definition.get("production_column", "production_profile"))
            production_value = row.get(production_column, row["partition"])
            production = "" if pd.isna(production_value) else str(production_value)
            pdg_map = definition.get("pdg_class_map", {}) or {}
            rows.append({
                "dataset_version": config["dataset_version"],
                "dataset_key": str(dataset_key),
                "role": str(definition["role"]),
                "domain": str(definition["domain"]),
                "metadata_csv": metadata_path,
                "metadata_row": int(metadata_row),
                "source_uid": source_uid,
                "fold_group_uid": fold_group_uid,
                "lumi_source_uid": source_uid,
                "data_type": str(row["data_type"]),
                "subfolder": str(row["subfolder"]),
                "partition": str(row["partition"]),
                "run": run,
                "production": production,
                "energy_bin": energy_bin,
                "n_event_metadata": int(row["n_event"]),
                "lumi_per_file": finite_float(row["lumi_per_file"]),
                "feature_path": feature_path,
                "hit3d_path": hit3d_path,
                "feature_available": False,
                "hit3d_available": False,
                "input_available": False,
                "feature_entries": -1,
                "hit3d_entries": -1,
                "input_empty": False,
                "class_mode": str(definition["class_mode"]),
                "direct_class": str(definition.get("direct_class", "")),
                "pdg_class_map": json_compact({str(k): str(v) for k, v in pdg_map.items()}),
            })
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result.empty:
        raise ValueError("No source rows were configured")
    if result["source_uid"].duplicated().any():
        raise ValueError("Generated duplicate source_uid values")
    workers = int(config.get("availability_workers", 32))
    products = list(zip(
        result["feature_path"], result["hit3d_path"], result["role"],
    ))
    trees = config.get("trees", {})
    feature_tree_name = str(trees.get("feature", "sndData"))
    hit3d_tree_name = str(trees.get("hit3d", "hit3D"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        states = list(executor.map(
            lambda item: inspect_product_pair(
                (item[0], item[1]),
                feature_tree_name,
                hit3d_tree_name,
                inspect_entries=str(item[2]) in {"muon_split", "neutrino_split"},
            ),
            products,
        ))
    result[[
        "feature_available", "hit3d_available", "input_available",
        "feature_entries", "hit3d_entries", "input_empty",
    ]] = states
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    catalog = build_catalog(config, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    available = int(catalog["input_available"].sum())
    empty = int(catalog["input_empty"].sum())
    print(
        f"Wrote {len(catalog)} sources to {output}; {available} have feature "
        f"and Hit3D, including {empty} aligned empty pairs"
    )


if __name__ == "__main__":
    main()
