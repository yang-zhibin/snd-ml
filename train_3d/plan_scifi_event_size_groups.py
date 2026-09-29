#!/usr/bin/env python3
"""Plan independently balanced, resumable SciFi event-size groups."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_event_size_groups import SCHEMA_VERSION  # noqa: E402
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    assign_shards, file_sha256, load_plan, write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--statistics-complete", required=True)
    parser.add_argument("--num-groups", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if int(args.num_groups) <= 0:
        raise ValueError("num-groups must be positive")
    plan = load_plan(args.plan)
    records = plan["files"]
    assignment = assign_shards(records, int(args.num_groups))
    groups = [{"group_index": index, "file_ordinals": [], "events": 0}
              for index in range(int(args.num_groups))]
    for record, group_index in zip(records, assignment):
        group = groups[int(group_index)]
        group["file_ordinals"].append(int(record["ordinal"]))
        group["events"] += int(record["n_events"])
    for group in groups:
        group["file_ordinals"].sort()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata_plan_sha256": file_sha256(args.plan),
        "statistics_complete_sha256": file_sha256(args.statistics_complete),
        "n_groups": int(args.num_groups),
        "n_folds": int(plan["n_folds"]),
        "total_files": len(records),
        "total_events": int(plan["total_events"]),
        "groups": groups,
    }
    write_json_atomic(args.output, payload)
    nonempty = [group["events"] for group in groups if group["file_ordinals"]]
    print(
        f"Planned {len(records):,} files in {args.num_groups} event-size groups; "
        f"event range {min(nonempty, default=0):,}..{max(nonempty, default=0):,}"
    )


if __name__ == "__main__":
    main()
