#!/usr/bin/env python3
"""Audit approximate native-hit reconstruction from legacy hit3D candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.native_scifi import (  # noqa: E402
    native_hit_counts, reconstruct_native_from_candidates,
)


BRANCHES = {
    "station": "hit_station", "detector_type": "hit_detType",
    "x": "hit_x", "y": "hit_y", "z": "hit_z", "time_ns": "hit_time_ns",
    "v_channel": "hit_v_channel", "h_channel": "hit_h_channel",
    "v_qdc": "hit_v_qdc", "h_qdc": "hit_h_qdc",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--tree", default="hit3D")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    candidate_counts, native_counts, vertical, horizontal = [], [], [], []
    contradictory = 0
    for path in args.paths:
        with uproot.open(path) as root_file:
            tree = root_file[args.tree]
            missing = set(BRANCHES.values()) - set(tree.keys())
            if missing:
                raise KeyError(f"{path} lacks branches: {sorted(missing)}")
            stop = min(int(tree.num_entries), int(args.max_events) - len(native_counts))
            payload = tree.arrays(list(BRANCHES.values()), entry_stop=stop,
                                  library="ak", how=dict)
            for entry in range(stop):
                raw = {name: np.asarray(payload[branch][entry])
                       for name, branch in BRANCHES.items()}
                candidate_counts.append(int((raw["detector_type"] == 1).sum()))
                try:
                    event = reconstruct_native_from_candidates(raw)
                except ValueError:
                    contradictory += 1
                    continue
                counts = native_hit_counts(event)
                native_counts.append(counts["n_native_hits"])
                vertical.append(counts["n_vertical_hits"])
                horizontal.append(counts["n_horizontal_hits"])
        if len(native_counts) >= args.max_events:
            break
    candidates = np.asarray(candidate_counts, dtype=float)
    native = np.asarray(native_counts, dtype=float)
    report = {
        "schema": "legacy_native_reconstruction_audit_v1",
        "physics_qualified": False,
        "warning": "Separate fiber Z/time and detector ID are not recoverable.",
        "events_attempted": len(candidate_counts),
        "events_reconstructed": len(native_counts),
        "contradictory_events": contradictory,
        "candidate_count": {
            "median": float(np.median(candidates)) if len(candidates) else None,
            "max": int(candidates.max()) if len(candidates) else None,
        },
        "native_count": {
            "median": float(np.median(native)) if len(native) else None,
            "p95": float(np.percentile(native, 95)) if len(native) else None,
            "max": int(native.max()) if len(native) else None,
        },
        "candidate_to_native_ratio_median": (
            float(np.median(candidates[:len(native)] / np.maximum(native, 1)))
            if len(native) and len(candidates) == len(native) else None
        ),
        "events_missing_vertical": int(np.sum(np.asarray(vertical) == 0)),
        "events_missing_horizontal": int(np.sum(np.asarray(horizontal) == 0)),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
