#!/usr/bin/env python3
"""Select the densest Run-8329 veto-tagged events usable by the v3 transform."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import yaml

from detid_channel_shift import choose_transform_axis
from scifi_edge_transform_sndsw import (
    find_track_entry,
    open_track_tree,
    select_track,
    track_input_path,
)


def atomic_csv(rows: list[dict], path: str, first_fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(first_fields)
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({name: row.get(name, "") for name in fields} for row in rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def occupancy(row: dict) -> int:
    return sum(int(float(row.get(f"count_scifi{station}", 0))) for station in range(1, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-selected", required=True)
    parser.add_argument("--output-rejected", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    input_config = config["input"]
    sample_config = config["high_density_sample"]
    sample_size = int(sample_config["sample_size"])
    excluded = {int(value) for value in sample_config.get("exclude_event_ids", [])}
    minimum_occupancy = int(sample_config.get("min_feature_scifi_occupancy", 0))
    with open(input_config["source_manifest"], newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    unique = {}
    for row in source_rows:
        if row.get("population") != "failure" or int(float(row.get("count_veto", 0))) <= 0:
            continue
        if not row.get("category", "").startswith("run8329__failure__"):
            continue
        event_id = int(row["eventId"])
        if event_id in excluded:
            continue
        value = occupancy(row)
        if value < minimum_occupancy:
            continue
        key = (int(row["runId"]), event_id)
        candidate_key = (-value, row.get("category", ""), int(row["source_event_index"]))
        if key not in unique or candidate_key < unique[key][0]:
            unique[key] = (candidate_key, row)
    ranked = sorted(
        (row for _, row in unique.values()),
        key=lambda row: (-occupancy(row), int(row["runId"]), int(row["eventId"])),
    )
    if not ranked:
        raise RuntimeError("No high-density veto-tagged candidates were found")

    track_config = config["track"]
    transform_config = config["transform"]
    direction_config = transform_config["direction"]
    selected_rows, rejected_rows = [], []
    open_path = None
    source = tree = None
    try:
        for density_rank, row in enumerate(ranked, start=1):
            path = track_input_path(input_config["track_input_pattern"], row)
            try:
                if path != open_path:
                    if source:
                        source.Close()
                    source, tree = open_track_tree(path, input_config.get("track_tree", "sndData"))
                    open_path = path
                entry = find_track_entry(tree, int(row["runId"]), int(row["eventId"]))
                track, _ = select_track(
                    tree,
                    [int(value) for value in track_config["type_priority"]],
                    int(track_config.get("min_ndf", 1)),
                    int(track_config.get("min_scifi_planes_per_projection", 2)),
                )
                momentum_z = track.get("track_mom_z")
                if (
                    direction_config.get("use_track_pz", True)
                    and momentum_z is not None and math.isfinite(float(momentum_z))
                    and float(momentum_z) != 0
                ):
                    pz_sign = 1 if float(momentum_z) > 0 else -1
                    direction_source = "track_pz"
                else:
                    pz_sign = int(direction_config["assumed_pz_sign"])
                    direction_source = "assumed_positive_z" if pz_sign > 0 else "assumed_negative_z"
                decision = choose_transform_axis(
                    track["projection_x"].slope,
                    track["projection_y"].slope,
                    pz_sign,
                    direction_source,
                    float(transform_config.get("min_abs_slope", 0.0)),
                    float(transform_config.get("min_axis_dominance_ratio", 1.0)),
                )
                z_reference = 0.5 * (
                    min(item["z_cm"] for item in track["collapsed_scifi_points"])
                    + max(item["z_cm"] for item in track["collapsed_scifi_points"])
                )
                output = dict(row)
                output.update({
                    "density_rank": density_rank,
                    "feature_scifi_occupancy": occupancy(row),
                    "track_input_path": path,
                    "track_tree": input_config.get("track_tree", "sndData"),
                    "track_entry": entry,
                    "track_index": track["track_index"],
                    "track_type": track["track_type"],
                    "track_flag": track["track_flag"],
                    "track_ndf": track["track_ndf"],
                    "track_chi2ndf": track["track_chi2ndf"],
                    "track_slope_xz": track["projection_x"].slope,
                    "track_slope_yz": track["projection_y"].slope,
                    "track_fit_max_residual_x_cm": track["projection_x"].max_abs_residual_cm,
                    "track_fit_max_residual_y_cm": track["projection_y"].max_abs_residual_cm,
                    "reference_z_cm": z_reference,
                    "reference_x_cm": track["projection_x"].at(z_reference),
                    "reference_y_cm": track["projection_y"].at(z_reference),
                    "track_mom_x": track.get("track_mom_x", ""),
                    "track_mom_y": track.get("track_mom_y", ""),
                    "track_mom_z": track.get("track_mom_z", ""),
                    "selected_axis_preview": decision.selected_axis,
                    "entry_side_preview": decision.entry_side,
                    "axis_dominance_ratio_preview": decision.axis_dominance_ratio,
                })
                selected_rows.append(output)
                if len(selected_rows) >= sample_size:
                    break
            except Exception as error:
                rejected_rows.append({
                    **row,
                    "density_rank": density_rank,
                    "feature_scifi_occupancy": occupancy(row),
                    "rejection_reason": str(error),
                })
    finally:
        if source:
            source.Close()
    if len(selected_rows) != sample_size:
        raise RuntimeError(
            f"Selected {len(selected_rows)} usable events, requested {sample_size}"
        )
    for sequence, row in enumerate(selected_rows, start=1):
        row["event_sequence"] = sequence
    atomic_csv(
        selected_rows, args.output_selected,
        ["event_sequence", "density_rank", "feature_scifi_occupancy"],
    )
    atomic_csv(
        rejected_rows, args.output_rejected,
        ["density_rank", "feature_scifi_occupancy", "rejection_reason"],
    )
    print(
        f"Selected {len(selected_rows)} high-density events after scanning "
        f"{selected_rows[-1]['density_rank']} ranked candidates; rejected={len(rejected_rows)}"
    )


if __name__ == "__main__":
    main()
