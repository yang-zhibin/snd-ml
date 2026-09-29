#!/usr/bin/env python3
"""Validate completeness, provenance and invariants of SciFi metadata v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_hit_event_dataset import SciFiNormalization  # noqa: E402
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, load_selected_entries, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    root = Path(args.metadata_root)
    complete_path = root / "complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete["plan_sha256"] != file_sha256(args.plan):
        raise ValueError("Metadata completion marker used a different plan")
    for name, details in complete["products"].items():
        path = root / name
        if path.stat().st_size != int(details["bytes"]):
            raise ValueError(f"Byte-size mismatch for {name}")
        if file_sha256(path) != details["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    total = int(plan["total_events"])
    if int(index["total_events"]) != total or len(index["files"]) != len(plan["files"]):
        raise ValueError("Index dimensions differ from the plan")
    entries = np.load(root / "source_entries.npy", mmap_mode="r", allow_pickle=False)
    candidates = np.load(root / "n_candidates.npy", mmap_mode="r", allow_pickle=False)
    if entries.dtype != np.int64 or entries.shape != (total,):
        raise ValueError("source_entries.npy has the wrong dtype or shape")
    if candidates.dtype != np.uint32 or candidates.shape != (total,):
        raise ValueError("n_candidates.npy has the wrong dtype or shape")
    voxel_arrays = []
    for fold in range(int(plan["n_folds"])):
        values = np.load(
            root / f"n_voxels_fold_{fold}.npy", mmap_mode="r", allow_pickle=False
        )
        if values.dtype != np.uint32 or values.shape != (total,):
            raise ValueError(f"n_voxels_fold_{fold}.npy has the wrong dtype or shape")
        if np.any(values > candidates):
            raise ValueError(f"Fold {fold} has more voxels than candidates")
        if np.any((values == 0) != (candidates == 0)):
            raise ValueError(f"Fold {fold} empty-event flags differ from candidates")
        voxel_arrays.append(values)
        SciFiNormalization.load(
            root / f"fold_{fold}" / "normalization.json", expected_fold=fold
        )
        summary = json.loads(
            (root / f"fold_{fold}" / "summary.json").read_text(encoding="utf-8")
        )
        if int(summary["cv_fold"]) != fold:
            raise ValueError(f"Summary for fold {fold} has a wrong cv_fold")
    for record, indexed in zip(plan["files"], index["files"]):
        if indexed["dataset_file_uid"] != record["dataset_file_uid"]:
            raise ValueError("Index file order differs from the plan")
        start = int(record["global_offset"])
        stop = start + int(record["n_events"])
        expected = load_selected_entries(record)
        if not np.array_equal(entries[start:stop], expected):
            raise ValueError(f"Source-entry mismatch for {record['dataset_file_uid']}")
    report = {
        "schema_version": 1,
        "valid": True,
        "metadata_complete_sha256": file_sha256(complete_path),
        "plan_sha256": file_sha256(args.plan),
        "total_files": len(plan["files"]),
        "total_events": total,
        "empty_events": int(np.count_nonzero(candidates == 0)),
        "folds": int(plan["n_folds"]),
    }
    write_json_atomic(args.output, report)
    print(
        f"Validated {total:,} events in {len(plan['files']):,} files "
        f"across {plan['n_folds']} folds"
    )


if __name__ == "__main__":
    main()
