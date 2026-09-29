#!/usr/bin/env python3
"""Scan Hit3D twice total and build all five folds' SciFi metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.data_driven_fold_dataset import NaturalFoldEvents
from train_3d.data.scifi_hit_event_dataset import make_scifi_reader
from train_3d.data.scifi_statistics import SciFiStatistics, count_sparse_voxels


FIELDS = [
    "dataset_file_uid", "source_entry", "physical_fold", "class_name", "domain",
    "n_candidates", "n_voxels",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def unique_references(views):
    """Use CV view zero to address every logical event exactly once."""
    for split in ("train", "validation"):
        yield from NaturalFoldEvents(views, 0, split)


def limited(iterable, maximum):
    for index, value in enumerate(iterable):
        if maximum is not None and index >= int(maximum):
            break
        yield value


def merged_training_statistics(by_physical_fold, held_out, dataset_version):
    result = SciFiStatistics(dataset_version, held_out)
    for physical_fold, statistics in sorted(by_physical_fold.items()):
        if physical_fold != held_out:
            result.merge(statistics)
    return result


def main():
    args = parse_args()
    config = load_yaml(args.config)
    data = config["data"]
    representation = config["representation"]
    views = pd.read_csv(data["fold_views"], keep_default_na=False)
    n_folds = int(data.get("n_folds", 5))
    if set(pd.to_numeric(views["fold"]).astype(int)) != set(range(n_folds)):
        raise ValueError("Fold views do not contain the configured physical folds")
    physical_fold_by_uid = (
        views[["dataset_file_uid", "fold"]]
        .drop_duplicates()
        .set_index("dataset_file_uid")["fold"].astype(int).to_dict()
    )
    reader = make_scifi_reader(
        backend=data.get("reader_backend", "pyroot"),
        tree_name=data.get("tree_name", "hit3D"),
        max_open_files=int(data.get("max_open_files", 8)),
    )

    # Pass one accumulates moments by physical fold. Leave-one-fold-out
    # training constants are produced by merging the other four accumulators.
    by_physical_fold = {
        fold: SciFiStatistics(config["dataset_version"], fold)
        for fold in range(n_folds)
    }
    scanned = 0
    pass_started = time.perf_counter()
    for reference in limited(unique_references(views), args.max_events):
        physical_fold = physical_fold_by_uid[reference.dataset_file_uid]
        by_physical_fold[physical_fold].update(reader.read(reference))
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            elapsed = time.perf_counter() - pass_started
            print(
                f"SciFi metadata pass 1: {scanned:,} events "
                f"({scanned / elapsed:.1f} events/s)", flush=True,
            )
    training_statistics = {
        fold: merged_training_statistics(
            by_physical_fold, fold, config["dataset_version"]
        )
        for fold in range(n_folds)
    }
    voxel_size = tuple(float(value) for value in representation["voxel_size"])
    bounds = {
        fold: training_statistics[fold].coordinate_bounds(voxel_size)
        for fold in range(n_folds)
    }
    output_root = Path(args.output_root)
    writers, streams = {}, {}
    for fold in range(n_folds):
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_statistics[fold].normalization().save(fold_dir / "normalization.json")
        stream = (fold_dir / "event_sizes.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        streams[fold], writers[fold] = stream, writer

    # Pass two reads each event once more and derives all five voxel counts in
    # memory. A per-fold two-pass workflow would perform ten ROOT scans.
    empty_counts = {fold: 0 for fold in range(n_folds)}
    rows_written = 0
    pass_started = time.perf_counter()
    try:
        for reference in limited(unique_references(views), args.max_events):
            event = reader.read(reference)
            physical_fold = physical_fold_by_uid[reference.dataset_file_uid]
            n_candidates = int(len(event["qdc"]))
            for fold in range(n_folds):
                lower, upper = bounds[fold]
                writers[fold].writerow({
                    "dataset_file_uid": reference.dataset_file_uid,
                    "source_entry": reference.source_entry,
                    "physical_fold": physical_fold,
                    "class_name": reference.class_name,
                    "domain": reference.domain,
                    "n_candidates": n_candidates,
                    "n_voxels": count_sparse_voxels(
                        event["position"], lower, upper, voxel_size
                    ),
                })
                empty_counts[fold] += int(n_candidates == 0)
            rows_written += 1
            if args.progress_every > 0 and rows_written % args.progress_every == 0:
                elapsed = time.perf_counter() - pass_started
                print(
                    f"SciFi metadata pass 2: {rows_written:,} events "
                    f"({rows_written / elapsed:.1f} events/s)", flush=True,
                )
    finally:
        for stream in streams.values():
            stream.close()
        reader.close()

    for fold in range(n_folds):
        statistics = training_statistics[fold]
        lower, upper = bounds[fold]
        summary = {
            "version": "scifi_xyz_qdc_dettype_v1",
            "dataset_version": config["dataset_version"],
            "cv_fold": fold,
            "normalization_uses_training_only": True,
            "debug_max_events": args.max_events,
            "coordinate_min": lower,
            "coordinate_max": upper,
            "voxel_size": voxel_size,
            "scanned_unique_events": scanned,
            "manifest_rows": rows_written,
            "empty_events": empty_counts[fold],
            "training_events": statistics.events,
            "training_candidates": statistics.position.count,
            "position_min": statistics.position.minimum.tolist(),
            "position_max": statistics.position.maximum.tolist(),
        }
        (output_root / f"fold_{fold}" / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"Scanned {scanned} unique events in two ROOT passes and built "
        f"metadata for {n_folds} held-out folds"
    )


if __name__ == "__main__":
    main()
