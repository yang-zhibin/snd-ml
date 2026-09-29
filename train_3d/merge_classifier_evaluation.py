#!/usr/bin/env python3
"""Merge additive classifier evaluation shards without averaging shard metrics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.classification_evaluation import (  # noqa: E402
    finalize_additive, merge_additive, merge_grouped, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output")
    return parser.parse_args(argv)


def merge_payloads(payloads):
    if not payloads:
        raise ValueError("No evaluation shards supplied")
    invariant_keys = ("class_names", "cv_fold", "num_shards", "checkpoint_sha256", "config_sha256")
    for key in invariant_keys:
        if len({json.dumps(payload[key], sort_keys=True) for payload in payloads}) != 1:
            raise ValueError(f"Evaluation shards disagree on {key}")
    expected = int(payloads[0]["num_shards"])
    indices = [int(payload["shard_index"]) for payload in payloads]
    if sorted(indices) != list(range(expected)) or len(indices) != len(set(indices)):
        raise ValueError("Evaluation shard set is incomplete or duplicated")
    ordered = sorted(payloads, key=lambda payload: int(payload["shard_index"]))
    for left, right in zip(ordered, ordered[1:]):
        if int(left["reference_stop"]) != int(right["reference_start"]):
            raise ValueError("Evaluation shard reference ranges are not contiguous")
    class_names = ordered[0]["class_names"]
    merged = merge_additive([item["additive"] for item in ordered], len(class_names))
    grouped = {}
    for group_name in ("domain", "density_bin", "node_count_bin"):
        combined = merge_grouped(
            [item.get("grouped", {}).get(group_name, {}) for item in ordered],
            len(class_names),
        )
        grouped[group_name] = {
            key: finalize_additive(record, class_names)
            for key, record in combined.items()
        }
    return {
        "schema_version": 1,
        "class_names": class_names,
        "cv_fold": ordered[0]["cv_fold"],
        "num_shards": expected,
        "checkpoint_sha256": ordered[0]["checkpoint_sha256"],
        "config_sha256": ordered[0]["config_sha256"],
        "reference_count": sum(int(item["reference_count"]) for item in ordered),
        "skipped_empty_events": sum(int(item.get("skipped_empty_events", 0)) for item in ordered),
        "global": finalize_additive(merged, class_names),
        "grouped": grouped,
        "wall_seconds_sum": sum(float(item.get("wall_seconds", 0.0)) for item in ordered),
    }


def main(argv=None):
    args = parse_args(argv)
    paths = [Path(path) for pattern in args.shards for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError("No shard JSON files matched")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    write_json(args.output, merge_payloads(payloads))
    if args.predictions_output:
        output = Path(args.predictions_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as destination:
            for payload in sorted(payloads, key=lambda item: int(item["shard_index"])):
                source = payload.get("predictions_path")
                if source:
                    with open(source, "rb") as stream:
                        shutil.copyfileobj(stream, destination)


if __name__ == "__main__":
    main()
