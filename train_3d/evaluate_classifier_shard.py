#!/usr/bin/env python3
"""Evaluate one contiguous natural-validation shard with additive outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.classification_evaluation import (  # noqa: E402
    add_observations, empty_additive, file_sha256, node_count_bin, write_json,
)
from train_3d.classification_module import ClassificationModule  # noqa: E402
from train_3d.data.classification_datamodule import ClassificationDataModule  # noqa: E402
from train_3d.data.data_driven_fold_dataset import NaturalFoldEvents  # noqa: E402
from train_3d.data.node_budget_batch_sampler import NodeBudgetBatchSampler  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Training resolved_config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--training-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", help="Optional full JSONL prediction output")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args(argv)


def contiguous_shard(sequence, index: int, count: int):
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("Require 0 <= shard-index < num-shards")
    start = len(sequence) * index // count
    stop = len(sequence) * (index + 1) // count
    return [sequence[position] for position in range(start, stop)], start, stop


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main(argv=None):
    args = parse_args(argv)
    if args.training_manifest:
        manifest = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
        args.config = manifest["files"]["resolved_config"]["path"]
        args.checkpoint = manifest["best_checkpoint"]
        if file_sha256(args.config) != manifest["files"]["resolved_config"]["sha256"]:
            raise ValueError("Resolved config checksum differs from training manifest")
        if file_sha256(args.checkpoint) != manifest["best_checkpoint_sha256"]:
            raise ValueError("Best checkpoint checksum differs from training manifest")
    if not args.config or not args.checkpoint:
        raise ValueError("Provide --training-manifest or both --config and --checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    threads = max(1, int(args.num_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fold = int(config["data"]["cv_fold"])
    module = ClassificationModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device
    ).to(args.device).eval()
    if tuple(module.class_names) != tuple(config["class_names"]):
        raise ValueError("Checkpoint class order differs from resolved config")
    datamodule = ClassificationDataModule(config, fold)
    datamodule.setup("validate")
    views = pd.read_csv(config["data"]["fold_views"], keep_default_na=False)
    natural = NaturalFoldEvents(views, fold, "validation")
    references, start, stop = contiguous_shard(natural, args.shard_index, args.num_shards)
    batch_sampler = NodeBudgetBatchSampler(
        references,
        datamodule.validation_batch_sampler.size_index,
        datamodule.validation_batch_sampler.representation,
        datamodule.validation_batch_sampler.max_events,
        datamodule.validation_batch_sampler.max_nodes,
    )
    loader = datamodule._loader(batch_sampler)
    total = empty_additive(len(module.class_names))
    grouped = {"domain": {}, "density_bin": {}, "node_count_bin": {}}
    prediction_stream = None
    if args.predictions:
        prediction_path = Path(args.predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_stream = prediction_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader:
                device_batch = move_batch(batch, args.device)
                logits = module(device_batch)
                losses = torch.nn.functional.cross_entropy(
                    logits, device_batch["label"], reduction="none"
                )
                if not torch.isfinite(losses).all():
                    raise RuntimeError("Non-finite evaluation loss")
                predictions = logits.argmax(dim=1)
                probabilities = logits.softmax(dim=1)
                labels_cpu = device_batch["label"].cpu()
                predictions_cpu = predictions.cpu()
                losses_cpu = losses.cpu()
                nodes_cpu = batch["n_nodes"].cpu()
                add_observations(
                    total, labels_cpu, predictions_cpu, losses_cpu, nodes_cpu,
                    len(module.class_names),
                )
                for i in range(len(labels_cpu)):
                    keys = {
                        "domain": str(batch["domain"][i]),
                        "density_bin": str(batch["density_bin"][i] or "none"),
                        "node_count_bin": node_count_bin(int(nodes_cpu[i])),
                    }
                    for group_name, key in keys.items():
                        target = grouped[group_name].setdefault(
                            key, empty_additive(len(module.class_names))
                        )
                        add_observations(
                            target, labels_cpu[i:i + 1], predictions_cpu[i:i + 1],
                            losses_cpu[i:i + 1], nodes_cpu[i:i + 1],
                            len(module.class_names),
                        )
                    if prediction_stream is not None:
                        row = {
                            "dataset_file_uid": batch["dataset_file_uid"][i],
                            "source_entry": int(batch["source_entry"][i]),
                            "run_id": int(batch["run_id"][i]),
                            "event_id": int(batch["event_id"][i]),
                            "event_index": int(batch["event_index"][i]),
                            "domain": batch["domain"][i],
                            "density_bin": batch["density_bin"][i],
                            "n_nodes": int(nodes_cpu[i]),
                            "true_class": module.class_names[int(labels_cpu[i])],
                            "predicted_class": module.class_names[int(predictions_cpu[i])],
                            "probabilities": probabilities[i].cpu().tolist(),
                        }
                        prediction_stream.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        if prediction_stream is not None:
            prediction_stream.close()
    wall = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "class_names": list(module.class_names),
        "cv_fold": fold,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "reference_start": start,
        "reference_stop": stop,
        "reference_count": stop - start,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "additive": total,
        "grouped": grouped,
        "skipped_empty_events": (stop - start) - int(total["event_count"]),
        "wall_seconds": wall,
        "events_per_second": total["event_count"] / wall if wall else None,
        "nodes_per_second": total["node_count"] / wall if wall else None,
        "predictions_path": str(Path(args.predictions).resolve()) if args.predictions else None,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
