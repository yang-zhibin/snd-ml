#!/usr/bin/env python3
"""Build versioned native-hit normalization and event-size metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.native_scifi import native_hit_counts  # noqa: E402
from train_3d.data.scifi_metadata_v2 import (  # noqa: E402
    file_sha256, iter_native_scifi_events, save_npy_atomic, unique_file_rows,
    write_json_atomic,
)
from train_3d.data.scifi_statistics import NativeSciFiStatistics  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.step_size <= 0:
        raise ValueError("step-size must be positive")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    views_path = Path(data["fold_views"]).resolve()
    views = pd.read_csv(views_path, keep_default_na=False)
    n_folds = int(data.get("n_folds", 5))
    records = unique_file_rows(views, n_folds)
    total = sum(int(record["n_events"]) for record in records)
    entries = np.empty(total, dtype=np.int64)
    native_sizes = np.empty(total, dtype=np.uint32)
    sparse_sizes = np.empty(total, dtype=np.uint32)
    by_physical_fold = {
        fold: NativeSciFiStatistics(config["dataset_version"], fold)
        for fold in range(n_folds)
    }
    cursor = 0
    started = time.perf_counter()
    for record in records:
        start = cursor
        statistics = by_physical_fold[int(record["physical_fold"])]
        for source_entry, event in iter_native_scifi_events(
            record, data.get("tree_name", "hit3D"), args.step_size
        ):
            counts = native_hit_counts(event)
            n_native = counts["n_native_hits"]
            n_sites = counts["n_vertical_sites"] + counts["n_horizontal_sites"]
            if max(n_native, n_sites) > np.iinfo(np.uint32).max:
                raise OverflowError("Native SciFi event size exceeds uint32")
            entries[cursor] = source_entry
            native_sizes[cursor] = n_native
            sparse_sizes[cursor] = n_sites
            statistics.update(event)
            cursor += 1
            if args.progress_every and cursor % args.progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-12)
                print(f"{cursor:,}/{total:,} events ({cursor / elapsed:.1f}/s)", flush=True)
        if cursor - start != int(record["n_events"]):
            raise RuntimeError(f"Wrong event count for {record['dataset_file_uid']}")
    if cursor != total:
        raise RuntimeError(f"Processed {cursor} events but expected {total}")

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed metadata {output}")
    save_npy_atomic(output / "source_entries.npy", entries)
    save_npy_atomic(output / "n_native_hits.npy", native_sizes)
    save_npy_atomic(output / "n_sparse_sites.npy", sparse_sizes)

    fold_products = {}
    for held_out in range(n_folds):
        training = NativeSciFiStatistics(config["dataset_version"], held_out)
        for physical_fold, values in by_physical_fold.items():
            if physical_fold != held_out:
                training.merge(values)
        fold_dir = output / f"fold_{held_out}"
        normalization_path = fold_dir / "native_normalization.json"
        training.normalization().save(normalization_path)
        summary_path = fold_dir / "native_summary.json"
        write_json_atomic(summary_path, {
            "schema_version": 1,
            "representation_version": "scifi_native_hits_v1",
            "dataset_version": config["dataset_version"],
            "cv_fold": held_out,
            "normalization_uses_training_only": True,
            "training_events": int(training.events),
            "training_native_hits": int(training.u.count),
            "training_empty_events": int(training.empty_events),
            "observed_channel_min": float(training.channel.minimum[0]),
            "observed_channel_max": float(training.channel.maximum[0]),
            "observed_station_min": float(training.station.minimum[0]),
            "observed_station_max": float(training.station.maximum[0]),
        })
        fold_products[str(held_out)] = {
            "normalization_sha256": file_sha256(normalization_path),
            "summary_sha256": file_sha256(summary_path),
        }

    index = {
        "schema_version": 2,
        "metadata_version": "scifi_native_training_metadata_v1",
        "dataset_version": config["dataset_version"],
        "representation_version": "scifi_native_hits_v1",
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
        } for record in records],
    }
    write_json_atomic(output / "index.json", index)
    write_json_atomic(output / "validation.json", {
        "schema_version": 1,
        "valid": True,
        "metadata_version": "scifi_native_training_metadata_v1",
        "dataset_version": config["dataset_version"],
        "representation_version": "scifi_native_hits_v1",
        "total_events": total,
        "checks": {
            "source_entries_aligned": True,
            "native_sizes_aligned": True,
            "sparse_sizes_aligned": True,
            "sparse_sites_not_above_native_hits": bool(
                np.all(sparse_sizes <= native_sizes)
            ),
            "normalization_training_only": True,
        },
    })
    products = {}
    for name in ("index.json", "validation.json", "source_entries.npy",
                 "n_native_hits.npy", "n_sparse_sites.npy"):
        products[name] = {
            "sha256": file_sha256(output / name),
            "bytes": (output / name).stat().st_size,
        }
    write_json_atomic(output / "complete.json", {
        "schema_version": 2,
        "metadata_version": "scifi_native_training_metadata_v1",
        "dataset_version": config["dataset_version"],
        "representation_version": "scifi_native_hits_v1",
        "config_sha256": file_sha256(config_path),
        "fold_views_sha256": file_sha256(views_path),
        "n_folds": n_folds,
        "total_events": total,
        "products": products,
        "fold_products": fold_products,
    })
    print(f"Built native SciFi metadata for {total:,} events")


if __name__ == "__main__":
    main()
