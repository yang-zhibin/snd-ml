#!/usr/bin/env python3
"""Build one source-aligned RealDataDQ manifest for Run-8329 transforms."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml

from detid_channel_shift import choose_transform_axis
from scifi_edge_transform_sndsw import select_track


ROOT.gROOT.SetBatch(True)


FIELDS = [
    "runId", "eventId", "source_file_index", "original_event_index",
    "original_feature_entry", "original_count_veto", "original_passes_loose_v7",
    "simple_eligible", "reco_scifi_eligible", "track_entry", "track_index",
    "track_type", "track_flag", "track_ndf", "track_chi2ndf",
    "track_slope_xz", "track_slope_yz", "reference_z_cm", "reference_x_cm",
    "reference_y_cm", "track_mom_x", "track_mom_y", "track_mom_z",
    "selected_axis_preview", "entry_side_preview", "axis_dominance_ratio_preview",
]


def read_source_row(path: str, source_index: int) -> dict:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["source_file_index"]) == source_index
        ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one metadata row for source {source_index}, found {len(rows)}"
        )
    return rows[0]


def atomic_gzip_csv(rows, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def valid_average(tree, station: int, axis: str) -> bool:
    value = float(getattr(tree, f"avg_scifi{station}_{axis}"))
    return math.isfinite(value) and value > -900.0


def passes_loose_v7(tree) -> bool:
    veto = [int(getattr(tree, f"count_veto{i}")) for i in range(1, 4)]
    scifi = [int(getattr(tree, f"count_scifi{i}")) for i in range(1, 6)]
    upstream = [int(getattr(tree, f"count_us{i}")) for i in range(1, 6)]
    downstream = [int(getattr(tree, f"count_ds{i}")) for i in range(1, 5)]
    return (
        all(value > 0 for value in veto)
        and int(tree.scifi_n_active_stations_xy) >= 5
        and all(value > 0 for value in upstream)
        and sum(value > 0 for value in downstream) >= 4
        and all(value < 20 for value in scifi)
        and all(value <= 2 for value in veto)
        and int(tree.count_veto) <= 4
        and all(value <= 2 for value in upstream)
        and int(tree.count_us) <= 6
    )


def blank_track() -> dict:
    return {name: "" for name in FIELDS if name.startswith("track_")} | {
        "reference_z_cm": "", "reference_x_cm": "", "reference_y_cm": "",
        "selected_axis_preview": "", "entry_side_preview": "",
        "axis_dominance_ratio_preview": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    source_row = read_source_row(config["inputs"]["real_dq_metadata"], args.source_index)
    feature_path = source_row["feature_full_path"]
    track_path = config["inputs"]["track_input_pattern"].format(
        source_file_index=args.source_index
    )

    feature_file = ROOT.TFile.Open(feature_path)
    if not feature_file or feature_file.IsZombie():
        raise OSError(f"Cannot open RealDataDQ features {feature_path}")
    feature_tree = feature_file.Get(config.get("tree_name", "sndData"))
    if not feature_tree:
        raise RuntimeError(f"No sndData tree in {feature_path}")

    track_file = ROOT.TFile.Open(track_path)
    track_tree = None
    if track_file and not track_file.IsZombie():
        candidate = track_file.Get(config.get("tree_name", "sndData"))
        if candidate and candidate.GetEntries() > 0 and candidate.GetBranch("reco_track_type"):
            if candidate.BuildIndex("runId", "eventId") < 0:
                raise RuntimeError(f"Cannot index track input {track_path}")
            track_tree = candidate

    track_config = config["methods"]["reco_scifi"]["track"]
    transform_config = config["methods"]["reco_scifi"]["transform"]
    rows = []
    counts = {
        "real_dq": int(feature_tree.GetEntries()), "simple_eligible": 0,
        "reco_scifi_eligible": 0, "loose_v7": 0, "no_veto": 0,
    }
    for feature_entry in range(int(feature_tree.GetEntries())):
        if feature_tree.GetEntry(feature_entry) <= 0:
            raise RuntimeError(f"Cannot read feature entry {feature_entry}")
        run_id, event_id = int(feature_tree.runId), int(feature_tree.eventId)
        simple = all(
            valid_average(feature_tree, station, axis)
            for station in range(1, 6) for axis in ("x", "y")
        )
        loose = passes_loose_v7(feature_tree)
        no_veto = int(feature_tree.count_veto) == 0
        counts["simple_eligible"] += int(simple)
        counts["loose_v7"] += int(loose)
        counts["no_veto"] += int(no_veto)

        row = {
            "runId": run_id,
            "eventId": event_id,
            "source_file_index": args.source_index,
            "original_event_index": int(feature_tree.eventIndex),
            "original_feature_entry": feature_entry,
            "original_count_veto": int(feature_tree.count_veto),
            "original_passes_loose_v7": int(loose),
            "simple_eligible": int(simple),
            "reco_scifi_eligible": 0,
            **blank_track(),
        }
        if track_tree is not None:
            track_entry = int(track_tree.GetEntryNumberWithIndex(run_id, event_id))
            if track_entry >= 0 and track_tree.GetEntry(track_entry) > 0:
                try:
                    track, _ = select_track(
                        track_tree,
                        [int(value) for value in track_config["type_priority"]],
                        int(track_config.get("min_ndf", 1)),
                        int(track_config.get("min_scifi_planes_per_projection", 2)),
                    )
                    pz = track.get("track_mom_z")
                    pz_sign = 1 if pz is not None and float(pz) > 0 else -1
                    if pz is None or not math.isfinite(float(pz)) or float(pz) == 0:
                        pz_sign = int(transform_config["direction"]["assumed_pz_sign"])
                    decision = choose_transform_axis(
                        track["projection_x"].slope, track["projection_y"].slope,
                        pz_sign, "track_pz" if pz not in (None, 0) else "assumed_pz",
                        float(transform_config.get("min_abs_slope", 0.0)),
                        float(transform_config.get("min_axis_dominance_ratio", 1.0)),
                    )
                    z_reference = 0.5 * (
                        min(value["z_cm"] for value in track["collapsed_scifi_points"])
                        + max(value["z_cm"] for value in track["collapsed_scifi_points"])
                    )
                    row.update({
                        "reco_scifi_eligible": 1,
                        "track_entry": track_entry,
                        "track_index": track["track_index"],
                        "track_type": track["track_type"],
                        "track_flag": track["track_flag"],
                        "track_ndf": track["track_ndf"],
                        "track_chi2ndf": track["track_chi2ndf"],
                        "track_slope_xz": track["projection_x"].slope,
                        "track_slope_yz": track["projection_y"].slope,
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
                    counts["reco_scifi_eligible"] += 1
                except (RuntimeError, ValueError):
                    pass
        rows.append(row)

    atomic_gzip_csv(rows, args.output_manifest)
    summary = {
        "version": config["version"], "source_file_index": args.source_index,
        "source_path": source_row["source_path"], "feature_path": feature_path,
        "track_path": track_path, "geometry_path": source_row["geo_path"],
        "counts": counts,
    }
    atomic_json(summary, args.output_summary)
    feature_file.Close()
    if track_file:
        track_file.Close()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
