#!/usr/bin/env python3
"""One-step CUDA canary for native GravNet and two-view sparse ResNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from train_3d.models.pyg_gravnet_classifier import PyGGravNetClassifier
from train_3d.models.two_view_sparse_resnet_classifier import (
    TwoViewSparseResNetClassifier,
)


def timed_step(model, forward, labels):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = forward()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "loss": float(loss.detach().cpu()),
        "logits_shape": list(logits.shape),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "finite_gradients": all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ),
    }


def sparse_coordinates(events=2, sites_per_event=3000):
    rows = []
    for event in range(events):
        for index in range(sites_per_event):
            rows.append([event, index % 5, index % 1536])
    # Repeated station/channel pairs would be illegal sparse inputs. The first
    # 3,000 indices are unique because lcm(5,1536)=7680.
    return torch.tensor(rows, dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gravnet-nodes", type=int, default=20000)
    parser.add_argument("--sparse-sites-per-view-event", type=int, default=3000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native-model GPU canary")
    device = torch.device("cuda")
    events = 2
    gravnet = PyGGravNetClassifier(
        in_channels=7, hidden_channels=64, num_classes=7, blocks=3,
        space_dimensions=4, propagate_dimensions=64, k=16, dropout=0.1,
    ).to(device).train()
    counts = [args.gravnet_nodes // events] * events
    counts[-1] += args.gravnet_nodes - sum(counts)
    graph_x = torch.randn(args.gravnet_nodes, 7, device=device)
    graph_batch = torch.repeat_interleave(
        torch.arange(events, device=device), torch.tensor(counts, device=device)
    )
    labels = torch.tensor([0, 6], device=device)
    graph_result = timed_step(
        gravnet, lambda: gravnet(graph_x, graph_batch), labels
    )

    sparse = TwoViewSparseResNetClassifier(
        in_channels=2, num_classes=7, channels=(32, 64, 128, 256),
        dropout=0.1, dimension=2,
    ).to(device).train()
    coordinates = sparse_coordinates(events, args.sparse_sites_per_view_event)
    features_v = torch.randn(len(coordinates), 2, device=device)
    features_h = torch.randn(len(coordinates), 2, device=device)
    present = torch.ones(events, dtype=torch.bool, device=device)
    sparse_result = timed_step(
        sparse,
        lambda: sparse(
            coordinates, features_v, coordinates, features_h, present, present
        ),
        labels,
    )
    payload = {
        "status": "PASS" if (
            graph_result["finite_gradients"] and sparse_result["finite_gradients"]
        ) else "FAIL",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "native_gravnet": graph_result,
        "two_view_sparse_resnet": sparse_result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise RuntimeError("A native model produced non-finite gradients")


if __name__ == "__main__":
    main()
