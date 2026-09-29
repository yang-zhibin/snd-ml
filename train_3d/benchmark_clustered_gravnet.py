#!/usr/bin/env python3
"""GPU memory staircase for the clustered GravNet production architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.models.clustered_pyg_gravnet_classifier import (  # noqa: E402
    ClusteredPyGGravNetClassifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--event-sizes", type=int, nargs="+",
                        default=[20_000, 100_000, 400_000, 1_800_000])
    parser.add_argument("--cluster-nodes", type=int, default=8192)
    parser.add_argument("--nodes-per-forward", type=int, default=8192)
    parser.add_argument("--k", type=int, default=16)
    return parser.parse_args()


def cluster_ptr(nodes: int, cluster_nodes: int):
    values = list(range(0, int(nodes), int(cluster_nodes)))
    if not values or values[-1] != int(nodes):
        values.append(int(nodes))
    return torch.tensor(values, dtype=torch.long)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the clustered GravNet benchmark")
    device = torch.device("cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": torch.cuda.get_device_name(device),
        "cluster_nodes": args.cluster_nodes,
        "nodes_per_forward": args.nodes_per_forward,
        "k": args.k,
        "results": [],
    }
    for nodes in args.event_sizes:
        torch.manual_seed(42)
        model = ClusteredPyGGravNetClassifier(
            in_channels=5, hidden_channels=64, num_classes=7, blocks=3,
            space_dimensions=4, propagate_dimensions=64, k=args.k,
            dropout=0.1, max_cluster_nodes=args.cluster_nodes,
            max_nodes_per_forward=args.nodes_per_forward,
            activation_checkpointing=True,
        ).to(device).train()
        x = torch.randn(int(nodes), 5, dtype=torch.float32, device=device)
        pointers = cluster_ptr(nodes, args.cluster_nodes).to(device)
        events = torch.zeros(len(pointers) - 1, dtype=torch.long, device=device)
        label = torch.tensor([0], dtype=torch.long, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            logits = model(x, pointers, events, 1)
            loss = torch.nn.functional.cross_entropy(logits, label)
            loss.backward()
            torch.cuda.synchronize(device)
        except Exception as exc:
            report["results"].append({
                "nodes": int(nodes), "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise
        elapsed = time.perf_counter() - started
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        result = {
            "nodes": int(nodes),
            "clusters": len(pointers) - 1,
            "status": "PASS" if torch.isfinite(loss) and gradients_finite else "FAIL",
            "loss": float(loss.detach().cpu()),
            "seconds": elapsed,
            "nodes_per_second": int(nodes) / elapsed,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
            "gradients_finite": gradients_finite,
        }
        report["results"].append(result)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "PASS":
            raise RuntimeError(f"Non-finite result at {nodes} nodes")
        del model, x, pointers, events, label, logits, loss
        torch.cuda.empty_cache()
    print("clustered GravNet GPU staircase: PASS")


if __name__ == "__main__":
    main()
