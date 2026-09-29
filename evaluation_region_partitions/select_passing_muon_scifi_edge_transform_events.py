#!/usr/bin/env python3
"""Select a deterministic Run-8329 Muon-DIS edge-transform validation sample."""

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


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(rows: list[dict], path: str, preferred_fields: list[str] | None = None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred_fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def identity(row: dict) -> tuple[int, int]:
    return int(row["runId"]), int(row["eventId"])


def rank_sample(valid: list[dict], sample_size: int) -> list[dict]:
    if not valid:
        return []
    x_values = [float(row["reference_x_cm"]) for row in valid]
    y_values = [float(row["reference_y_cm"]) for row in valid]
    x_mid = 0.5 * (min(x_values) + max(x_values))
    y_mid = 0.5 * (min(y_values) + max(y_values))
    x_scale = max(max(x_values) - min(x_values), 1.0)
    y_scale = max(max(y_values) - min(y_values), 1.0)
    rankings = [
        ("low_x", lambda row: float(row["reference_x_cm"])),
        ("high_x", lambda row: -float(row["reference_x_cm"])),
        ("low_y", lambda row: float(row["reference_y_cm"])),
        ("high_y", lambda row: -float(row["reference_y_cm"])),
        ("central", lambda row: (
            ((float(row["reference_x_cm"]) - x_mid) / x_scale) ** 2
            + ((float(row["reference_y_cm"]) - y_mid) / y_scale) ** 2
        )),
        ("largest_abs_x_slope", lambda row: -abs(float(row["track_slope_xz"]))),
        ("largest_abs_y_slope", lambda row: -abs(float(row["track_slope_yz"]))),
        ("highest_scifi_occupancy", lambda row: -int(row["scifi_occupancy"])),
    ]
    chosen = {}
    for reason, score in rankings:
        row = min(valid, key=lambda value: (score(value), identity(value)))
        key = identity(row)
        chosen.setdefault(key, {**row, "selection_reason": reason})
        if len(chosen) >= sample_size:
            break
    for row in sorted(valid, key=identity):
        key = identity(row)
        chosen.setdefault(key, {**row, "selection_reason": "deterministic_fill"})
        if len(chosen) >= sample_size:
            break
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-selected", required=True)
    parser.add_argument("--output-rejected", required=True)
    parser.add_argument("--sample-size", type=int, help="Override configured sample size")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    source_manifest = config["input"]["source_manifest"]
    source_categories = set(config["input"]["source_categories"])
    sample_size = int(args.sample_size or config["selection"]["sample_size"])
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    rows = read_csv(source_manifest)
    pool = []
    seen = set()
    for row in rows:
        if row.get("category") not in source_categories:
            continue
        if row.get("population") != "failure" or int(float(row.get("count_veto", 0))) <= 0:
            continue
        key = identity(row)
        if key in seen:
            continue
        seen.add(key)
        pool.append(row)
    pool.sort(key=lambda row: (int(row["source_file_index"]), int(row["source_event_index"])))
    if not pool:
        raise RuntimeError("No veto-tagged Muon-DIS candidates in source manifest")

    track_config = config["track"]
    pattern = config["input"]["track_input_pattern"]
    valid, rejected = [], []
    open_path = None
    source = tree = None
    try:
        for row in pool:
            path = track_input_path(pattern, row)
            try:
                if path != open_path:
                    if source:
                        source.Close()
                    source, tree = open_track_tree(path, config["input"].get("track_tree", "sndData"))
                    open_path = path
                entry = find_track_entry(tree, *identity(row))
                selected, _ = select_track(
                    tree,
                    [int(value) for value in track_config["type_priority"]],
                    int(track_config.get("min_ndf", 1)),
                    int(track_config.get("min_scifi_planes_per_projection", 2)),
                )
                z_reference = 0.5 * (
                    min(item["z_cm"] for item in selected["collapsed_scifi_points"])
                    + max(item["z_cm"] for item in selected["collapsed_scifi_points"])
                )
                output = dict(row)
                output.update({
                    "track_input_path": path,
                    "track_entry": entry,
                    "track_index": selected["track_index"],
                    "track_type": selected["track_type"],
                    "track_ndf": selected["track_ndf"],
                    "track_chi2ndf": selected["track_chi2ndf"],
                    "track_slope_xz": selected["projection_x"].slope,
                    "track_slope_yz": selected["projection_y"].slope,
                    "track_fit_max_residual_x_cm": selected["projection_x"].max_abs_residual_cm,
                    "track_fit_max_residual_y_cm": selected["projection_y"].max_abs_residual_cm,
                    "reference_z_cm": z_reference,
                    "reference_x_cm": selected["projection_x"].at(z_reference),
                    "reference_y_cm": selected["projection_y"].at(z_reference),
                    "scifi_occupancy": sum(int(float(row.get(f"count_scifi{i}", 0))) for i in range(1, 6)),
                })
                valid.append(output)
            except Exception as error:
                rejected.append({**row, "rejection_reason": str(error)})
    finally:
        if source:
            source.Close()

    selected = rank_sample(valid, sample_size)
    selected_keys = {identity(row) for row in selected}
    rejected.extend(
        {**row, "rejection_reason": "not selected after deterministic ranking"}
        for row in valid if identity(row) not in selected_keys
    )
    if len(selected) < sample_size:
        raise RuntimeError(
            f"Only {len(selected)} usable events found, requested {sample_size}; "
            f"see {args.output_rejected}"
        )
    for sequence, row in enumerate(selected, start=1):
        row["event_sequence"] = sequence
    atomic_csv(selected, args.output_selected, ["event_sequence", "selection_reason"])
    atomic_csv(rejected, args.output_rejected, ["rejection_reason"])
    print(
        f"Selected {len(selected)} of {len(pool)} Muon-DIS candidates; "
        f"rejected/not-selected={len(rejected)}"
    )


if __name__ == "__main__":
    main()
