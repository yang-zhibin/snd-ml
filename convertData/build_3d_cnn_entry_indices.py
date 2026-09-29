#!/usr/bin/env python3
"""Build physical ROOT-entry indices for one filtered logical dataset row."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def selected_row(catalog_path: str, dataset_file_uid: str) -> pd.Series:
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    rows = catalog.loc[catalog["dataset_file_uid"].astype(str).eq(dataset_file_uid)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {dataset_file_uid}, found {len(rows)}")
    return rows.iloc[0]


def build_indices(hit3d_path: str, tree_name: str, expression: str) -> np.ndarray:
    """Return original TTree entry numbers passing a ROOT selection expression."""
    import ROOT

    root_file = ROOT.TFile.Open(str(hit3d_path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open {hit3d_path}")
    try:
        tree = root_file.Get(str(tree_name))
        if not tree:
            raise KeyError(f"Tree {tree_name!r} is missing from {hit3d_path}")
        n_entries = int(tree.GetEntries())
        if n_entries > np.iinfo(np.int32).max:
            raise OverflowError("One source file exceeds the int32 entry-index range")
        formula = ROOT.TTreeFormula("ml_event_filter", str(expression), tree)
        selected = []
        for entry in range(n_entries):
            tree.LoadTree(entry)
            tree.GetEntry(entry)
            if bool(formula.EvalInstance()):
                selected.append(entry)
        return np.asarray(selected, dtype=np.int32)
    finally:
        root_file.Close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-catalog", required=True)
    parser.add_argument("--dataset-file-uid", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-row", required=True)
    args = parser.parse_args()

    row = selected_row(args.dataset_catalog, args.dataset_file_uid)
    expression = str(row["event_filter"]).strip()
    if expression.lower() in {"", "true", "1"}:
        raise ValueError("Direct/unfiltered rows do not need an entry-index file")
    entries = build_indices(row["hit3d_path"], row["tree_name"], expression)
    if len(entries) != int(row["n_events"]):
        raise RuntimeError(
            f"Entry-index closure failed for {args.dataset_file_uid}: "
            f"{len(entries)} != {row['n_events']}"
        )
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, entries=entries)
    audit = {
        "dataset_file_uid": args.dataset_file_uid,
        "hit3d_path": row["hit3d_path"],
        "event_filter": expression,
        "entry_index_path": str(output),
        "n_events": int(len(entries)),
    }
    output_row = Path(args.output_row)
    output_row.parent.mkdir(parents=True, exist_ok=True)
    with output_row.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit))
        writer.writeheader()
        writer.writerow(audit)


if __name__ == "__main__":
    main()
