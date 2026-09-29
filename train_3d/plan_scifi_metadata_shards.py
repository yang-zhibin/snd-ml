#!/usr/bin/env python3
"""Create a deterministic file-level shard plan for SciFi metadata v2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    METADATA_VERSION, SCHEMA_VERSION, assign_shards, file_sha256,
    unique_file_rows, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-shards", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    views_path = Path(data["fold_views"]).resolve()
    views = pd.read_csv(views_path, keep_default_na=False)
    n_folds = int(data.get("n_folds", 5))
    records = unique_file_rows(views, n_folds)
    assignment = assign_shards(records, args.num_shards)
    loads = [0] * int(args.num_shards)
    for record, shard in zip(records, assignment):
        record["shard"] = int(shard)
        loads[shard] += int(record["n_events"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata_version": METADATA_VERSION,
        "dataset_version": str(config["dataset_version"]),
        "representation_version": str(config["representation"]["version"]),
        "tree_name": str(data.get("tree_name", "hit3D")),
        "n_folds": n_folds,
        "n_shards": int(args.num_shards),
        "total_files": len(records),
        "total_events": sum(int(record["n_events"]) for record in records),
        "fold_views": str(views_path),
        "fold_views_sha256": file_sha256(views_path),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "shard_event_counts": loads,
        "files": records,
    }
    write_json_atomic(args.output, payload)
    print(
        f"Planned {payload['total_events']:,} events from {len(records):,} files "
        f"in {args.num_shards} deterministic shards"
    )


if __name__ == "__main__":
    main()
