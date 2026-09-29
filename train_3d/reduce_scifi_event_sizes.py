#!/usr/bin/env python3
"""Assemble event-size shards into compact memory-mappable metadata arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shards-root", required=True)
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


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    output = Path(args.output_root)
    final_marker = output / "complete.json"
    if final_marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed metadata {final_marker}")
    statistics_marker = Path(args.statistics_root) / "statistics_complete.json"
    statistics = json.loads(statistics_marker.read_text(encoding="utf-8"))
    if statistics["plan_sha256"] != file_sha256(args.plan):
        raise ValueError("Reduced statistics used a different shard plan")
    n_folds = int(plan["n_folds"])
    total = int(plan["total_events"])
    targets = {
        "source_entries.npy": create_output(output / "source_entries.npy", np.int64, total),
        "n_candidates.npy": create_output(output / "n_candidates.npy", np.uint32, total),
    }
    for fold in range(n_folds):
        name = f"n_voxels_fold_{fold}.npy"
        targets[name] = create_output(output / name, np.uint32, total)
    seen = np.zeros(len(plan["files"]), dtype=np.bool_)
    try:
        for shard in range(int(plan["n_shards"])):
            directory = Path(args.shards_root) / f"shard_{shard:03d}"
            marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            if marker["plan_sha256"] != file_sha256(args.plan):
                raise ValueError(f"Event-size shard {shard} used a different plan")
            if marker["statistics_sha256"] != file_sha256(statistics_marker):
                raise ValueError(f"Event-size shard {shard} used different statistics")
            sources = {}
            for name, details in marker["files"].items():
                path = directory / name
                if details["sha256"] != file_sha256(path):
                    raise ValueError(f"Checksum mismatch for shard {shard} {name}")
                sources[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            for segment in marker["file_segments"]:
                ordinal = int(segment["ordinal"])
                if seen[ordinal]:
                    raise ValueError(f"File ordinal {ordinal} occurs in multiple shards")
                record = plan["files"][ordinal]
                count = int(segment["count"])
                if count != int(record["n_events"]):
                    raise ValueError(f"File count mismatch for ordinal {ordinal}")
                source_start = int(segment["local_offset"])
                source_stop = source_start + count
                target_start = int(record["global_offset"])
                target_stop = target_start + count
                for name, source in sources.items():
                    targets[name][1][target_start:target_stop] = source[source_start:source_stop]
                seen[ordinal] = True
        if not seen.all():
            raise ValueError("One or more planned files have no event-size segment")
        for _, target in targets.values():
            target.flush()
        del target
        for name, (temporary, array) in targets.items():
            array._mmap.close()
            os.replace(temporary, output / name)
    except BaseException:
        for temporary, _ in targets.values():
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
        "plan_sha256": file_sha256(args.plan),
        "statistics_complete_sha256": file_sha256(statistics_marker),
        "n_folds": n_folds,
        "total_events": total,
        "products": products,
    })
    print(f"Assembled memory-mapped metadata for {total:,} events")


if __name__ == "__main__":
    main()
