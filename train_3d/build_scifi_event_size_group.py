#!/usr/bin/env python3
"""Build one resumable group of per-file SciFi event-size products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_event_size_groups import (  # noqa: E402
    event_size_cache_path, load_event_size_file, load_event_size_group_plan,
    write_event_size_file_atomic,
)
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, iter_scifi_events, load_plan, write_json_atomic,
)
from train_3d.data.scifi_statistics import count_sparse_voxels  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--group-plan", required=True)
    parser.add_argument("--group-index", required=True, type=int)
    parser.add_argument("--statistics-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-size", type=int, default=10000)
    return parser.parse_args(argv)


def load_bounds(statistics_root, n_folds):
    result = {}
    for fold in range(int(n_folds)):
        path = Path(statistics_root) / f"fold_{fold}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[fold] = (payload["coordinate_min"], payload["coordinate_max"])
    return result


def build_file(record, plan, bounds, voxel_size, step_size):
    count = int(record["n_events"])
    n_folds = int(plan["n_folds"])
    source_entries = np.empty(count, dtype=np.int64)
    candidates = np.empty(count, dtype=np.uint32)
    voxels = np.empty((n_folds, count), dtype=np.uint32)
    cursor = 0
    for source_entry, event in iter_scifi_events(
        record, plan["tree_name"], step_size
    ):
        n_candidates = len(event["qdc"])
        if n_candidates > np.iinfo(np.uint32).max:
            raise OverflowError("Candidate count exceeds uint32")
        source_entries[cursor] = source_entry
        candidates[cursor] = n_candidates
        for fold in range(n_folds):
            lower, upper = bounds[fold]
            voxels[fold, cursor] = count_sparse_voxels(
                event["position"], lower, upper, voxel_size
            )
        cursor += 1
    if cursor != count:
        raise RuntimeError(
            f"Processed {cursor} events but {record['dataset_file_uid']} expects {count}"
        )
    return source_entries, candidates, voxels


def main(argv=None):
    args = parse_args(argv)
    if int(args.step_size) <= 0:
        raise ValueError("step-size must be positive")
    plan = load_plan(args.plan)
    metadata_plan_sha256 = file_sha256(args.plan)
    statistics_path = Path(args.statistics_root) / "statistics_complete.json"
    statistics_complete_sha256 = file_sha256(statistics_path)
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    group_plan = load_event_size_group_plan(
        args.group_plan,
        metadata_plan_sha256=metadata_plan_sha256,
        statistics_complete_sha256=statistics_complete_sha256,
    )
    if int(group_plan["n_folds"]) != int(plan["n_folds"]):
        raise ValueError("Event-size group plan has a different fold count")
    if args.group_index < 0 or args.group_index >= int(group_plan["n_groups"]):
        raise ValueError("group-index is outside the event-size group plan")
    group = group_plan["groups"][args.group_index]
    records = [plan["files"][int(ordinal)] for ordinal in group["file_ordinals"]]
    bounds = load_bounds(args.statistics_root, plan["n_folds"])
    voxel_size = tuple(float(value) for value in statistics["voxel_size"])
    started = time.perf_counter()
    processed = 0
    reused = 0
    created = 0
    replaced = 0
    products = []
    for file_number, record in enumerate(records, start=1):
        path = event_size_cache_path(
            args.cache_root, args.group_index, record["ordinal"]
        )
        cache_status = "reused"
        try:
            load_event_size_file(
                path,
                record=record,
                group_index=args.group_index,
                n_folds=plan["n_folds"],
                metadata_plan_sha256=metadata_plan_sha256,
                statistics_complete_sha256=statistics_complete_sha256,
            )
            reused += 1
        except FileNotFoundError:
            cache_status = "created"
        except ValueError as error:
            cache_status = "replaced"
            print(
                f"Replacing incompatible event-size cache {path}: {error}",
                flush=True,
            )
        if cache_status != "reused":
            source_entries, candidates, voxels = build_file(
                record, plan, bounds, voxel_size, args.step_size
            )
            write_event_size_file_atomic(
                path,
                record=record,
                group_index=args.group_index,
                metadata_plan_sha256=metadata_plan_sha256,
                statistics_complete_sha256=statistics_complete_sha256,
                source_entries=source_entries,
                n_candidates=candidates,
                n_voxels=voxels,
            )
            if cache_status == "created":
                created += 1
            else:
                replaced += 1
        processed += int(record["n_events"])
        products.append({
            "ordinal": int(record["ordinal"]),
            "path": str(path.relative_to(Path(args.cache_root))),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
        elapsed = max(time.perf_counter() - started, 1e-12)
        print(
            f"event-size group {args.group_index}: {file_number}/{len(records)} files, "
            f"{processed:,} events ({processed / elapsed:.1f} events/s), "
            f"reused={reused}",
            flush=True,
        )
    if processed != int(group["events"]):
        raise RuntimeError("Event-size group event count differs from its plan")
    write_json_atomic(args.output, {
        "schema_version": 1,
        "group_index": int(args.group_index),
        "event_size_group_plan_sha256": file_sha256(args.group_plan),
        "metadata_plan_sha256": metadata_plan_sha256,
        "statistics_complete_sha256": statistics_complete_sha256,
        "files": len(records),
        "events": processed,
        "reused_files": reused,
        "created_files": created,
        "replaced_files": replaced,
        "products": products,
    })
    print(
        f"Completed event-size group {args.group_index} with {len(records)} files; "
        f"reused {reused}, created {created}, and replaced {replaced} cached "
        f"products"
    )


if __name__ == "__main__":
    main()
