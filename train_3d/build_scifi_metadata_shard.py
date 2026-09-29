#!/usr/bin/env python3
"""Build one statistics or event-size shard for SciFi metadata v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, iter_scifi_events, load_plan, save_npy_atomic, write_json_atomic,
)
from train_3d.data.scifi_statistics import (  # noqa: E402
    SciFiStatistics, count_sparse_voxels,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("statistics", "event-sizes"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--statistics-root")
    parser.add_argument("--step-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args(argv)


def selected_records(plan, shard_index):
    if shard_index < 0 or shard_index >= int(plan["n_shards"]):
        raise ValueError("shard-index is outside the plan")
    return [
        record for record in plan["files"] if int(record["shard"]) == shard_index
    ]


def load_bounds(statistics_root, n_folds):
    result = {}
    for fold in range(n_folds):
        path = Path(statistics_root) / f"fold_{fold}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[fold] = (payload["coordinate_min"], payload["coordinate_max"])
    return result


def build_statistics(args, plan, records, output):
    by_fold = {
        fold: SciFiStatistics(plan["dataset_version"], fold)
        for fold in range(int(plan["n_folds"]))
    }
    processed = 0
    started = time.perf_counter()
    for record in records:
        target = by_fold[int(record["physical_fold"])]
        for _, event in iter_scifi_events(record, plan["tree_name"], args.step_size):
            target.update(event)
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-12)
                print(
                    f"statistics shard {args.shard_index}: {processed:,} events "
                    f"({processed / elapsed:.1f} events/s)", flush=True,
                )
    expected = sum(int(record["n_events"]) for record in records)
    if processed != expected:
        raise RuntimeError(f"Processed {processed} events but shard expects {expected}")
    payload = {
        "mode": "statistics",
        "schema_version": 2,
        "shard_index": args.shard_index,
        "plan_sha256": file_sha256(args.plan),
        "events": processed,
        "statistics_by_physical_fold": {
            str(fold): statistics.to_dict()
            for fold, statistics in by_fold.items()
        },
    }
    write_json_atomic(output / "statistics.json", payload)
    write_json_atomic(output / "complete.json", {
        "mode": "statistics", "shard_index": args.shard_index,
        "events": processed,
        "statistics_sha256": file_sha256(output / "statistics.json"),
        "plan_sha256": file_sha256(args.plan),
    })


def build_event_sizes(args, plan, records, output):
    if not args.statistics_root:
        raise ValueError("--statistics-root is required for event-sizes mode")
    n_folds = int(plan["n_folds"])
    bounds = load_bounds(args.statistics_root, n_folds)
    statistics_complete = Path(args.statistics_root) / "statistics_complete.json"
    if not statistics_complete.is_file():
        raise FileNotFoundError(statistics_complete)
    statistics_payload = json.loads(statistics_complete.read_text(encoding="utf-8"))
    voxel_size = tuple(float(value) for value in statistics_payload["voxel_size"])
    total = sum(int(record["n_events"]) for record in records)
    entries = np.empty(total, dtype=np.int64)
    candidates = np.empty(total, dtype=np.uint32)
    voxels = [np.empty(total, dtype=np.uint32) for _ in range(n_folds)]
    file_segments = []
    cursor = 0
    started = time.perf_counter()
    for record in records:
        local_start = cursor
        for source_entry, event in iter_scifi_events(
            record, plan["tree_name"], args.step_size
        ):
            n_candidates = len(event["qdc"])
            if n_candidates > np.iinfo(np.uint32).max:
                raise OverflowError("Candidate count exceeds uint32")
            entries[cursor] = source_entry
            candidates[cursor] = n_candidates
            for fold in range(n_folds):
                lower, upper = bounds[fold]
                count = count_sparse_voxels(
                    event["position"], lower, upper, voxel_size
                )
                voxels[fold][cursor] = count
            cursor += 1
            if args.progress_every and cursor % args.progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-12)
                print(
                    f"event-size shard {args.shard_index}: {cursor:,} events "
                    f"({cursor / elapsed:.1f} events/s)", flush=True,
                )
        count = int(record["n_events"])
        if cursor - local_start != count:
            raise RuntimeError(f"Wrong event count for {record['dataset_file_uid']}")
        file_segments.append({
            "ordinal": int(record["ordinal"]),
            "local_offset": local_start,
            "count": count,
        })
    if cursor != total:
        raise RuntimeError(f"Processed {cursor} events but shard expects {total}")
    save_npy_atomic(output / "source_entries.npy", entries)
    save_npy_atomic(output / "n_candidates.npy", candidates)
    for fold, values in enumerate(voxels):
        save_npy_atomic(output / f"n_voxels_fold_{fold}.npy", values)
    files = ["source_entries.npy", "n_candidates.npy"] + [
        f"n_voxels_fold_{fold}.npy" for fold in range(n_folds)
    ]
    write_json_atomic(output / "complete.json", {
        "mode": "event-sizes",
        "schema_version": 2,
        "shard_index": args.shard_index,
        "plan_sha256": file_sha256(args.plan),
        "statistics_sha256": file_sha256(statistics_complete),
        "events": total,
        "file_segments": file_segments,
        "files": {
            name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in files
        },
    })


def main(argv=None):
    args = parse_args(argv)
    if args.step_size <= 0:
        raise ValueError("step-size must be positive")
    plan = load_plan(args.plan)
    records = selected_records(plan, args.shard_index)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed shard {marker}")
    if args.mode == "statistics":
        build_statistics(args, plan, records, output)
    else:
        build_event_sizes(args, plan, records, output)
    print(f"Completed {args.mode} shard {args.shard_index} with {len(records)} files")


if __name__ == "__main__":
    main()
