#!/usr/bin/env python3
"""Build compact per-bin entry indices for one muon-DIS Hit3D file."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_driven_3d_dataset import configured_density_bins, density_bin_index, load_yaml


def selected_row(catalog_path: str, dataset_file_uid: str) -> pd.Series:
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    rows = catalog.loc[catalog["dataset_file_uid"].astype(str) == dataset_file_uid]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {dataset_file_uid}, found {len(rows)}")
    row = rows.iloc[0]
    if str(row["class_name"]) != "muon_DIS":
        raise ValueError(f"{dataset_file_uid} is not a muon_DIS dataset file")
    return row


def build_indices(hit3d_path: str, tree_name: str, variable: str, bins: list[dict]):
    import ROOT
    root_file = ROOT.TFile.Open(str(hit3d_path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open {hit3d_path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise KeyError(f"Tree {tree_name!r} is missing from {hit3d_path}")
        if not tree.GetBranch(variable):
            raise KeyError(f"Branch {variable!r} is missing from {hit3d_path}")
        entries = [[] for _ in bins]
        invalid = []
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus(variable, 1)
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            value = getattr(tree, variable)
            index = density_bin_index(value, bins)
            (invalid if index < 0 else entries[index]).append(entry)
        if int(tree.GetEntries()) > np.iinfo(np.int32).max:
            raise OverflowError("One source file exceeds the int32 density-index range")
        return [np.asarray(values, dtype=np.int32) for values in entries], np.asarray(
            invalid, dtype=np.int32
        )
    finally:
        root_file.Close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-catalog", required=True)
    parser.add_argument("--dataset-file-uid", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-row", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    row = selected_row(args.dataset_catalog, args.dataset_file_uid)
    bins = configured_density_bins(config)
    variable = str(config["density_sampling"]["variable"])
    arrays, invalid = build_indices(row["hit3d_path"], row["tree_name"], variable, bins)
    total = sum(len(values) for values in arrays) + len(invalid)
    if total != int(row["n_events"]):
        raise RuntimeError(f"Density index closure failed: {total} != {row['n_events']}")
    payload = {f"bin_{item['index']}": arrays[item["index"]] for item in bins}
    payload["invalid"] = invalid
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    counts = {key: int(len(value)) for key, value in payload.items()}
    audit = {
        "dataset_file_uid": args.dataset_file_uid,
        "source_uid": row["source_uid"],
        "domain": row["domain"],
        "fold_group_uid": row["fold_group_uid"],
        "hit3d_path": row["hit3d_path"],
        "index_path": str(output),
        "n_events": total,
        "density_counts": json.dumps(counts, sort_keys=True),
        "density_bin_schema": json.dumps([
            {
                "key": f"bin_{item['index']}",
                "name": item["name"],
                "low": item["low"],
                "high": None if math.isinf(item["high"]) else item["high"],
            }
            for item in bins
        ], sort_keys=True),
    }
    output_row = Path(args.output_row)
    output_row.parent.mkdir(parents=True, exist_ok=True)
    with open(output_row, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit))
        writer.writeheader()
        writer.writerow(audit)
    print(f"Wrote {total} density indices for {args.dataset_file_uid}")


if __name__ == "__main__":
    main()
