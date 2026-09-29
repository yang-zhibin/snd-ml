#!/usr/bin/env python3
"""Benchmark bounded real-event batches from one classifier fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import yaml


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.classification_datamodule import ClassificationDataModule
from train_3d.data.classification_labels import validate_scifi_classification_config


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batches < 1 or args.warmup_batches < 0:
        raise ValueError("batches must be positive and warmup-batches non-negative")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_scifi_classification_config(config)
    config["data"] = dict(config["data"])
    config["data"]["cv_fold"] = int(args.cv_fold)
    config["data"]["num_workers"] = int(args.num_workers)
    config["data"]["persistent_workers"] = bool(args.num_workers)

    module = ClassificationDataModule(config, args.cv_fold)
    setup_start = time.perf_counter()
    module.setup("fit")
    setup_seconds = time.perf_counter() - setup_start
    loader = module.train_dataloader() if args.split == "train" else module.val_dataloader()
    iterator = iter(loader)

    for _ in range(args.warmup_batches):
        next(iterator)

    latencies = []
    events = 0
    nodes = 0
    started = time.perf_counter()
    completed = 0
    for _ in range(args.batches):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        latencies.append(time.perf_counter() - batch_start)
        events += int(batch["label"].numel())
        nodes += int(batch["n_nodes"].sum())
        completed += 1
    elapsed = time.perf_counter() - started
    if completed == 0:
        raise RuntimeError("Dataloader produced no measured batches")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "hostname": platform.node(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "model_version": config["model_version"],
        "model_family": config["model"]["family"],
        "cv_fold": int(args.cv_fold),
        "split": args.split,
        "num_workers": int(args.num_workers),
        "warmup_batches": int(args.warmup_batches),
        "requested_batches": int(args.batches),
        "completed_batches": completed,
        "events": events,
        "nodes": nodes,
        "setup_seconds": setup_seconds,
        "measurement_seconds": elapsed,
        "events_per_second": events / elapsed,
        "nodes_per_second": nodes / elapsed,
        "batch_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "maximum": max(latencies),
        },
        "environment": {
            "lcg_release": os.environ.get("SND_ML_LCG_RELEASE"),
            "lcg_platform": os.environ.get("SND_ML_LCG_PLATFORM"),
            "overlay_sha256": os.environ.get("SND_ML_OVERLAY_SHA256"),
            "source_revision": os.environ.get("SND_ML_SOURCE_REVISION"),
            "metadata_cache": os.environ.get("SND_ML_METADATA_CACHE"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
