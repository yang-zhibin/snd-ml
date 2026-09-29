#!/usr/bin/env python3
"""Migrate completed legacy event-size shards into the per-file cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_event_size_groups import (  # noqa: E402
    event_size_cache_path, group_by_ordinal, load_event_size_file,
    load_event_size_group_plan, write_event_size_file_atomic,
)
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--group-plan", required=True)
    parser.add_argument("--legacy-shards-root", required=True)
    parser.add_argument("--statistics-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    metadata_sha = file_sha256(args.plan)
    statistics_path = Path(args.statistics_root) / "statistics_complete.json"
    statistics_sha = file_sha256(statistics_path)
    group_plan = load_event_size_group_plan(
        args.group_plan,
        metadata_plan_sha256=metadata_sha,
        statistics_complete_sha256=statistics_sha,
    )
    ordinal_groups = group_by_ordinal(group_plan)
    migrated = 0
    replaced = 0
    reused = 0
    completed_shards = 0
    completed_shard_indices = []
    skipped_shards = []
    for shard in range(int(plan["n_shards"])):
        directory = Path(args.legacy_shards_root) / f"shard_{shard:03d}"
        marker_path = directory / "complete.json"
        if not marker_path.is_file():
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("plan_sha256") != metadata_sha:
            skipped_shards.append({
                "shard_index": shard,
                "reason": "different metadata plan",
            })
            print(
                f"Skipping legacy shard {shard}: different metadata plan",
                flush=True,
            )
            continue
        if marker.get("statistics_sha256") != statistics_sha:
            skipped_shards.append({
                "shard_index": shard,
                "reason": "different statistics",
            })
            print(
                f"Skipping legacy shard {shard}: different statistics",
                flush=True,
            )
            continue
        sources = {}
        for name, details in marker["files"].items():
            path = directory / name
            if details["sha256"] != file_sha256(path):
                raise ValueError(f"Legacy shard {shard} checksum mismatch for {name}")
            sources[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        for segment in marker["file_segments"]:
            ordinal = int(segment["ordinal"])
            record = plan["files"][ordinal]
            group_index = ordinal_groups[ordinal]
            path = event_size_cache_path(args.cache_root, group_index, ordinal)
            if path.is_file():
                try:
                    load_event_size_file(
                        path,
                        record=record,
                        group_index=group_index,
                        n_folds=plan["n_folds"],
                        metadata_plan_sha256=metadata_sha,
                        statistics_complete_sha256=statistics_sha,
                    )
                except ValueError as error:
                    print(
                        f"Replacing incompatible event-size cache {path}: {error}",
                        flush=True,
                    )
                    replaced += 1
                else:
                    reused += 1
                    continue
            start = int(segment["local_offset"])
            stop = start + int(segment["count"])
            n_voxels = np.stack([
                sources[f"n_voxels_fold_{fold}.npy"][start:stop]
                for fold in range(int(plan["n_folds"]))
            ])
            write_event_size_file_atomic(
                path,
                record=record,
                group_index=group_index,
                metadata_plan_sha256=metadata_sha,
                statistics_complete_sha256=statistics_sha,
                source_entries=sources["source_entries.npy"][start:stop],
                n_candidates=sources["n_candidates.npy"][start:stop],
                n_voxels=n_voxels,
            )
            migrated += 1
        completed_shards += 1
        completed_shard_indices.append(shard)
    write_json_atomic(args.output, {
        "schema_version": 1,
        "metadata_plan_sha256": metadata_sha,
        "statistics_complete_sha256": statistics_sha,
        "event_size_group_plan_sha256": file_sha256(args.group_plan),
        "completed_legacy_shards": completed_shard_indices,
        "skipped_legacy_shards": skipped_shards,
        "migrated_files": migrated,
        "replaced_files": replaced,
        "reused_files": reused,
    })
    print(
        f"Migrated {migrated:,} files from {completed_shards} completed legacy "
        f"shards; replaced {replaced:,} and reused {reused:,} existing cache "
        f"files; skipped {len(skipped_shards):,} incompatible legacy shards"
    )


if __name__ == "__main__":
    main()
