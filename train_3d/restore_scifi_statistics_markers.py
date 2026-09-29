#!/usr/bin/env python3
"""Restore validated statistics completion markers without rescanning ROOT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shards-root", required=True)
    args = parser.parse_args()

    plan = load_plan(args.plan)
    plan_sha = file_sha256(args.plan)
    restored = 0
    for shard in range(int(plan["n_shards"])):
        directory = Path(args.shards_root) / f"shard_{shard:03d}"
        statistics_path = directory / "statistics.json"
        marker_path = directory / "complete.json"
        payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        expected_events = sum(
            int(record["n_events"])
            for record in plan["files"]
            if int(record["shard"]) == shard
        )
        checks = {
            "mode": "statistics",
            "schema_version": 2,
            "shard_index": shard,
            "plan_sha256": plan_sha,
            "events": expected_events,
        }
        for name, expected in checks.items():
            if payload.get(name) != expected:
                raise ValueError(
                    f"Statistics shard {shard} has wrong {name}: "
                    f"{payload.get(name)!r} != {expected!r}"
                )
        write_json_atomic(marker_path, {
            "mode": "statistics",
            "shard_index": shard,
            "events": expected_events,
            "statistics_sha256": file_sha256(statistics_path),
            "plan_sha256": plan_sha,
        })
        # Match the product timestamp so recovery does not falsely invalidate
        # an already completed downstream statistics reduction.
        timestamp = statistics_path.stat().st_mtime_ns
        os.utime(marker_path, ns=(timestamp, timestamp))
        restored += 1
    print(f"Restored and validated {restored} statistics completion markers")


if __name__ == "__main__":
    main()
