#!/usr/bin/env python3
"""Reduce deterministic statistics shards into five training-fold products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, load_plan, write_json_atomic,
)
from train_3d.data.scifi_statistics import SciFiStatistics  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    n_folds = int(plan["n_folds"])
    by_physical_fold = {
        fold: SciFiStatistics(plan["dataset_version"], fold)
        for fold in range(n_folds)
    }
    total = 0
    shard_hashes = []
    for shard in range(int(plan["n_shards"])):
        directory = Path(args.shards_root) / f"shard_{shard:03d}"
        marker_path = directory / "complete.json"
        payload_path = directory / "statistics.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker["plan_sha256"] != file_sha256(args.plan):
            raise ValueError(f"Statistics shard {shard} used a different plan")
        if marker["statistics_sha256"] != file_sha256(payload_path):
            raise ValueError(f"Statistics shard {shard} checksum mismatch")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if int(payload["shard_index"]) != shard:
            raise ValueError(f"Statistics shard index mismatch for {shard}")
        for fold in range(n_folds):
            by_physical_fold[fold].merge(SciFiStatistics.from_dict(
                payload["statistics_by_physical_fold"][str(fold)]
            ))
        total += int(payload["events"])
        shard_hashes.append(file_sha256(marker_path))
    if total != int(plan["total_events"]):
        raise ValueError("Reduced statistics event total differs from the plan")
    voxel_size = tuple(float(value) for value in config["representation"]["voxel_size"])
    output = Path(args.output_root)
    products = {}
    for held_out in range(n_folds):
        training = SciFiStatistics(plan["dataset_version"], held_out)
        for physical_fold in range(n_folds):
            if physical_fold != held_out:
                training.merge(by_physical_fold[physical_fold])
        lower, upper = training.coordinate_bounds(voxel_size)
        fold_dir = output / f"fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        normalization_path = fold_dir / "normalization.json"
        training.normalization().save(normalization_path)
        summary = {
            "metadata_schema_version": 2,
            "version": plan["representation_version"],
            "dataset_version": plan["dataset_version"],
            "cv_fold": held_out,
            "normalization_uses_training_only": True,
            "coordinate_min": lower,
            "coordinate_max": upper,
            "voxel_size": voxel_size,
            "scanned_unique_events": total,
            "training_events": int(training.events),
            "training_candidates": int(training.position.count),
            "training_empty_events": int(training.empty_events),
            "position_min": training.position.minimum.tolist(),
            "position_max": training.position.maximum.tolist(),
        }
        summary_path = fold_dir / "summary.json"
        write_json_atomic(summary_path, summary)
        products[str(held_out)] = {
            "normalization_sha256": file_sha256(normalization_path),
            "summary_sha256": file_sha256(summary_path),
        }
    write_json_atomic(output / "statistics_complete.json", {
        "schema_version": 2,
        "metadata_version": plan["metadata_version"],
        "dataset_version": plan["dataset_version"],
        "representation_version": plan["representation_version"],
        "plan_sha256": file_sha256(args.plan),
        "events": total,
        "n_folds": n_folds,
        "voxel_size": voxel_size,
        "shard_complete_sha256": shard_hashes,
        "fold_products": products,
    })
    print(f"Reduced {plan['n_shards']} statistics shards covering {total:,} events")


if __name__ == "__main__":
    main()
