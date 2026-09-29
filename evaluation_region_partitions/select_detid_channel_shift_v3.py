#!/usr/bin/env python3
"""Freeze one reviewed Run-8329 event and its exact SciFi track for v3."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import yaml

from scifi_edge_transform_sndsw import (
    find_track_entry,
    open_track_tree,
    select_track,
    track_input_path,
)


def atomic_csv(row: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--event-id", type=int, help="Override the configured event ID")
    parser.add_argument("--source-category", help="Override the configured source category")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    input_config = config["input"]
    target = (
        int(input_config["run_id"]),
        int(args.event_id if args.event_id is not None else input_config["event_id"]),
    )
    source_category = (
        args.source_category
        if args.source_category is not None else input_config.get("source_category")
    )
    with open(input_config["source_manifest"], newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.DictReader(stream)
            if (int(row["runId"]), int(row["eventId"])) == target
            and (
                not source_category or row.get("category") == source_category
            )
        ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source-manifest row for {target}, found {len(matches)}")
    row = matches[0]
    path = track_input_path(input_config["track_input_pattern"], row)
    source, tree = open_track_tree(path, input_config.get("track_tree", "sndData"))
    try:
        entry = find_track_entry(tree, *target)
        track_config = config["track"]
        selected, _ = select_track(
            tree,
            [int(value) for value in track_config["type_priority"]],
            int(track_config.get("min_ndf", 1)),
            int(track_config.get("min_scifi_planes_per_projection", 2)),
        )
    finally:
        source.Close()
    expected_type = input_config.get("expected_track_type")
    expected_index = input_config.get("expected_track_index")
    if expected_type is not None and int(selected["track_type"]) != int(expected_type):
        raise RuntimeError("Selected track type differs from the reviewed event")
    if expected_index is not None and int(selected["track_index"]) != int(expected_index):
        raise RuntimeError("Selected track index differs from the reviewed event")
    z_reference = 0.5 * (
        min(item["z_cm"] for item in selected["collapsed_scifi_points"])
        + max(item["z_cm"] for item in selected["collapsed_scifi_points"])
    )
    output = dict(row)
    output.update({
        "event_sequence": 1,
        "track_input_path": path,
        "track_tree": input_config.get("track_tree", "sndData"),
        "track_entry": entry,
        "track_index": selected["track_index"],
        "track_type": selected["track_type"],
        "track_flag": selected["track_flag"],
        "track_ndf": selected["track_ndf"],
        "track_chi2ndf": selected["track_chi2ndf"],
        "track_slope_xz": selected["projection_x"].slope,
        "track_slope_yz": selected["projection_y"].slope,
        "track_fit_max_residual_x_cm": selected["projection_x"].max_abs_residual_cm,
        "track_fit_max_residual_y_cm": selected["projection_y"].max_abs_residual_cm,
        "reference_z_cm": z_reference,
        "reference_x_cm": selected["projection_x"].at(z_reference),
        "reference_y_cm": selected["projection_y"].at(z_reference),
        "track_mom_x": selected.get("track_mom_x", ""),
        "track_mom_y": selected.get("track_mom_y", ""),
        "track_mom_z": selected.get("track_mom_z", ""),
    })
    for field in ("track_slope_xz", "track_slope_yz", "reference_z_cm",
                  "reference_x_cm", "reference_y_cm"):
        if not math.isfinite(float(output[field])):
            raise RuntimeError(f"Non-finite frozen track field {field}")
    atomic_csv(output, args.output)
    print(
        f"Selected Run/Event {target[0]}/{target[1]} track "
        f"type={selected['track_type']} index={selected['track_index']}"
    )


if __name__ == "__main__":
    main()
