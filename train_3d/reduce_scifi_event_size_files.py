#!/usr/bin/env python3
"""Assemble resumable per-file event sizes into training mmap arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_event_size_groups import (  # noqa: E402
    event_size_cache_path, group_by_ordinal, load_event_size_file,
    load_event_size_group_plan,
)
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--group-plan", required=True)
    parser.add_argument("--groups-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--statistics-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def create_output(path, dtype, length):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    return temporary, np.lib.format.open_memmap(
        temporary, mode="w+", dtype=dtype, shape=(length,)
    )


def validate_group_markers(
    group_plan, group_plan_path, groups_root, cache_root, metadata_sha, statistics_sha
):
    expected_group_plan_sha = file_sha256(group_plan_path)
    for group in group_plan["groups"]:
        index = int(group["group_index"])
        marker_path = Path(groups_root) / f"group_{index:03d}" / "complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checks = {
            "group_index": index,
            "event_size_group_plan_sha256": expected_group_plan_sha,
            "metadata_plan_sha256": metadata_sha,
            "statistics_complete_sha256": statistics_sha,
            "files": len(group["file_ordinals"]),
            "events": int(group["events"]),
        }
        for name, expected in checks.items():
            if marker.get(name) != expected:
                raise ValueError(f"Event-size group {index} has wrong {name}")
        marker_ordinals = [int(item["ordinal"]) for item in marker["products"]]
        if marker_ordinals != [int(value) for value in group["file_ordinals"]]:
            raise ValueError(f"Event-size group {index} has wrong file ordinals")
        for product in marker["products"]:
            path = Path(cache_root) / product["path"]
            # The canonical cache location is validated during reduction.  The
            # marker checksum additionally catches post-completion mutation.
            if path.is_file() and (
                path.stat().st_size != int(product["bytes"])
                or file_sha256(path) != product["sha256"]
            ):
                raise ValueError(f"Event-size group {index} product changed")


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
    if int(group_plan["total_files"]) != len(plan["files"]):
        raise ValueError("Event-size group plan has a different file count")
    if int(group_plan["total_events"]) != int(plan["total_events"]):
        raise ValueError("Event-size group plan has a different event count")
    ordinal_groups = group_by_ordinal(group_plan)
    if set(ordinal_groups) != set(range(len(plan["files"]))):
        raise ValueError("Event-size groups do not cover every file ordinal")
    validate_group_markers(
        group_plan, args.group_plan, args.groups_root, args.cache_root,
        metadata_sha, statistics_sha
    )

    output = Path(args.output_root)
    final_marker = output / "complete.json"
    if final_marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed metadata {final_marker}")
    total = int(plan["total_events"])
    n_folds = int(plan["n_folds"])
    targets = {
        "source_entries.npy": create_output(
            output / "source_entries.npy", np.int64, total
        ),
        "n_candidates.npy": create_output(
            output / "n_candidates.npy", np.uint32, total
        ),
    }
    for fold in range(n_folds):
        name = f"n_voxels_fold_{fold}.npy"
        targets[name] = create_output(output / name, np.uint32, total)
    try:
        for file_number, record in enumerate(plan["files"], start=1):
            ordinal = int(record["ordinal"])
            group_index = ordinal_groups[ordinal]
            path = event_size_cache_path(args.cache_root, group_index, ordinal)
            arrays = load_event_size_file(
                path,
                record=record,
                group_index=group_index,
                n_folds=n_folds,
                metadata_plan_sha256=metadata_sha,
                statistics_complete_sha256=statistics_sha,
            )
            start = int(record["global_offset"])
            stop = start + int(record["n_events"])
            targets["source_entries.npy"][1][start:stop] = arrays["source_entries"]
            targets["n_candidates.npy"][1][start:stop] = arrays["n_candidates"]
            for fold in range(n_folds):
                targets[f"n_voxels_fold_{fold}.npy"][1][start:stop] = (
                    arrays["n_voxels"][fold]
                )
            if file_number % 500 == 0:
                print(f"Reduced {file_number:,}/{len(plan['files']):,} files", flush=True)
        for _, target in targets.values():
            target.flush()
        del target
        for name, (temporary, array) in targets.items():
            array._mmap.close()
            os.replace(temporary, output / name)
    except BaseException:
        for temporary, array in targets.values():
            try:
                array._mmap.close()
            except (AttributeError, ValueError):
                pass
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise

    index = {
        "schema_version": 2,
        "metadata_version": plan["metadata_version"],
        "dataset_version": plan["dataset_version"],
        "representation_version": plan["representation_version"],
        "n_folds": n_folds,
        "total_events": total,
        "files": [{
            "dataset_file_uid": record["dataset_file_uid"],
            "offset": int(record["global_offset"]),
            "count": int(record["n_events"]),
            "lookup": (
                "direct" if str(record.get("event_filter", "true")).strip().lower()
                in {"", "true", "1"} else "sorted"
            ),
        } for record in plan["files"]],
    }
    write_json_atomic(output / "index.json", index)
    product_names = ["index.json", "source_entries.npy", "n_candidates.npy"] + [
        f"n_voxels_fold_{fold}.npy" for fold in range(n_folds)
    ]
    products = {
        name: {
            "sha256": file_sha256(output / name),
            "bytes": (output / name).stat().st_size,
        } for name in product_names
    }
    write_json_atomic(final_marker, {
        "schema_version": 2,
        "metadata_version": plan["metadata_version"],
        "dataset_version": plan["dataset_version"],
        "representation_version": plan["representation_version"],
        "plan_sha256": metadata_sha,
        "statistics_complete_sha256": statistics_sha,
        "event_size_group_plan_sha256": file_sha256(args.group_plan),
        "n_folds": n_folds,
        "total_events": total,
        "products": products,
    })
    print(f"Assembled resumable metadata for {total:,} events")


if __name__ == "__main__":
    main()
