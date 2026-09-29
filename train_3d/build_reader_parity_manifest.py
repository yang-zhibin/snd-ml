#!/usr/bin/env python3
"""Select a deterministic, bounded production sample for reader parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.data_driven_fold_dataset import _entry_candidates, select_fold_rows


def spaced(values, count):
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return []
    indices = np.linspace(0, len(values) - 1, min(int(count), len(values)), dtype=int)
    return [int(values[index]) for index in np.unique(indices)]


def reference(row, entry, category):
    return {
        "dataset_file_uid": str(row["dataset_file_uid"]),
        "hit3d_path": str(row["hit3d_path"]),
        "source_entry": int(entry),
        "class_id": int(row["class_id"]),
        "class_name": str(row["class_name"]),
        "domain": str(row["domain"]),
        "density_bin": category.removeprefix("density:"),
        "categories": [category],
    }


def add(selected, item):
    key = (item["dataset_file_uid"], item["source_entry"])
    if key in selected:
        selected[key]["categories"] = sorted(set(
            selected[key]["categories"] + item["categories"]
        ))
    else:
        selected[key] = item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-views", required=True)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.per_category < 1:
        raise ValueError("per-category must be positive")

    views = pd.read_csv(args.fold_views, keep_default_na=False)
    rows = select_fold_rows(views, args.cv_fold, "validation").sort_values(
        ["class_name", "domain", "dataset_file_uid"]
    )
    selected = {}
    for (class_name, domain), group in rows.groupby(
        ["class_name", "domain"], sort=True
    ):
        remaining = args.per_category
        for _, row in group.iterrows():
            size, entries = _entry_candidates(row)
            candidates = np.arange(size, dtype=np.int64) if entries is None else entries
            for entry in spaced(candidates, remaining):
                add(selected, reference(row, entry, f"class:{class_name}:{domain}"))
                remaining -= 1
                if remaining == 0:
                    break
            if remaining == 0:
                break

    muon_rows = rows.loc[rows["class_name"].astype(str).eq("muon_DIS")]
    for bin_name in [f"bin_{index}" for index in range(5)]:
        remaining = args.per_category
        for _, row in muon_rows.iterrows():
            index_path = Path(str(row.get("density_index_path", "")))
            if not index_path.is_file():
                continue
            with np.load(index_path, allow_pickle=False) as payload:
                entries = np.asarray(payload[bin_name], dtype=np.int64)
            for entry in spaced(entries, remaining):
                add(selected, reference(row, entry, f"density:{bin_name}"))
                remaining -= 1
                if remaining == 0:
                    break
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"Could not select {args.per_category} events for {bin_name}")

    payload = {
        "schema_version": 1,
        "cv_fold": int(args.cv_fold),
        "per_category": int(args.per_category),
        "fold_views": str(Path(args.fold_views).resolve()),
        "references": list(selected.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Selected {len(selected)} unique production references into {output}")


if __name__ == "__main__":
    main()
