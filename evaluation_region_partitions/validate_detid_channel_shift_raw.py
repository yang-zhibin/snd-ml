#!/usr/bin/env python3
"""Validate a transformed SNDSW digi file against its source event and audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import ROOT

from detid_channel_shift import (
    build_all_plane_activity_line,
    build_two_station_activity_line,
    choose_target_edge,
    choose_transform_axis,
)
from detid_channel_shift_sndsw import (
    decode_scifi_detid,
    digi_payload,
    event_identity,
    mufilter_hit_key,
    public_eos_path,
)


ROOT.gROOT.SetBatch(True)


def tree_from(source, name):
    tree = source.Get(name)
    if not tree:
        raise RuntimeError(f"Missing tree {name}")
    return tree


def payloads(collection):
    values = {int(hit.GetDetectorID()): digi_payload(hit) for hit in collection}
    if len(values) != int(collection.GetEntriesFast()):
        raise RuntimeError("Duplicate detector IDs in digi collection")
    return values


def write_marker(
    path: str, event_id: int, input_hits: int, output_hits: int,
    input_mufilter: int, output_mufilter: int, version: str,
):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"version={version}\n")
            stream.write(f"event={event_id}\n")
            stream.write(f"input_scifi_hits={input_hits}\n")
            stream.write(f"output_scifi_hits={output_hits}\n")
            stream.write(f"input_mufilter_hits={input_mufilter}\n")
            stream.write(f"output_mufilter_hits={output_mufilter}\n")
            stream.write("status=passed\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    with open(args.audit_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    source_info = audit["source"]
    source_file = ROOT.TFile.Open(public_eos_path(source_info["path"]))
    output_file = ROOT.TFile.Open(args.output_root)
    if not source_file or source_file.IsZombie() or not output_file or output_file.IsZombie():
        raise OSError("Cannot open source or transformed ROOT file")
    tree_name = source_info["tree"]
    source_tree = tree_from(source_file, tree_name)
    output_tree = tree_from(output_file, tree_name)
    expected_branches = {"EventHeader.", "Digi_ScifiHits", "Digi_MuFilterHits"}
    actual_branches = {branch.GetName() for branch in output_tree.GetListOfBranches()}
    if actual_branches != expected_branches:
        raise RuntimeError(f"Unexpected output branches: {sorted(actual_branches)}")
    if output_tree.GetEntries() != 1:
        raise RuntimeError(f"Expected one transformed event, found {output_tree.GetEntries()}")
    source_tree.GetEntry(int(source_info["entry"]))
    output_tree.GetEntry(0)
    identity = (int(audit["identity"]["runId"]), int(audit["identity"]["eventId"]))
    if event_identity(source_tree) != identity or event_identity(output_tree) != identity:
        raise RuntimeError("Source/output event identity mismatch")

    source_scifi = payloads(source_tree.Digi_ScifiHits)
    output_scifi = payloads(output_tree.Digi_ScifiHits)
    scifi_rows = [row for row in rows if row["detector"] == "Scifi"]
    mapped = [row for row in scifi_rows if row["status"] == "mapped"]
    unchanged = [row for row in scifi_rows if row["status"] == "unchanged_orthogonal"]
    dropped = [
        row for row in scifi_rows
        if row["status"] not in {"mapped", "unchanged_orthogonal"}
    ]
    retained = mapped + unchanged
    if len(source_scifi) != len(scifi_rows) or len(output_scifi) != len(retained):
        raise RuntimeError("SciFi input/output/audit count closure failed")
    seen_destinations = set()
    for row in mapped:
        source_id = int(row["source_det_id"])
        destination_id = int(row["destination_det_id"])
        if destination_id in seen_destinations:
            raise RuntimeError(f"Destination collision at {destination_id}")
        seen_destinations.add(destination_id)
        if int(row["destination_ordinal"]) != int(row["source_ordinal"]) + int(row["channel_offset"]):
            raise RuntimeError("Ordinal mapping equation failed")
        source_decoded = decode_scifi_detid(source_id)
        destination_decoded = decode_scifi_detid(destination_id)
        if source_decoded[:2] != destination_decoded[:2]:
            raise RuntimeError("SciFi mapping changed station or view")
        source_payload = source_scifi[source_id]
        destination_payload = output_scifi[destination_id]
        for field in ("valid", "n_sipms", "n_sides", "signals", "times", "daq"):
            if source_payload[field] != destination_payload[field]:
                raise RuntimeError(f"Payload mismatch for {source_id}->{destination_id}: {field}")
    for row in unchanged:
        source_id = int(row["source_det_id"])
        destination_id = int(row["destination_det_id"])
        if source_id != destination_id or int(row["channel_offset"]) != 0:
            raise RuntimeError("Orthogonal SciFi detector ID was changed")
        if source_scifi[source_id] != output_scifi[destination_id]:
            raise RuntimeError(f"Orthogonal SciFi payload changed for {source_id}")

    source_mufi = payloads(source_tree.Digi_MuFilterHits)
    output_mufi = payloads(output_tree.Digi_MuFilterHits)
    source_mufi_keys = {
        int(hit.GetDetectorID()): mufilter_hit_key(hit)
        for hit in source_tree.Digi_MuFilterHits
    }
    output_mufi_keys = {
        int(hit.GetDetectorID()): mufilter_hit_key(hit)
        for hit in output_tree.Digi_MuFilterHits
    }
    mufi_rows = [row for row in rows if row["detector"] == "MuFilter"]
    mapped_mufi = [row for row in mufi_rows if row["status"] == "mapped"]
    unchanged_mufi = [row for row in mufi_rows if row["status"] == "unchanged_orthogonal"]
    dropped_mufi = [
        row for row in mufi_rows
        if row["status"] not in {"mapped", "unchanged_orthogonal"}
    ]
    retained_mufi = mapped_mufi + unchanged_mufi
    if len(source_mufi) != len(mufi_rows) or len(output_mufi) != len(retained_mufi):
        raise RuntimeError("MuFilter input/output/audit count closure failed")
    seen_mufi_destinations = set()
    for row in mapped_mufi:
        source_id = int(row["source_det_id"])
        destination_id = int(row["destination_det_id"])
        if destination_id in seen_mufi_destinations:
            raise RuntimeError(f"MuFilter destination collision at {destination_id}")
        seen_mufi_destinations.add(destination_id)
        if int(row["destination_ordinal"]) != int(row["source_ordinal"]) + int(row["channel_offset"]):
            raise RuntimeError("MuFilter ordinal mapping equation failed")
        if source_mufi_keys[source_id] != output_mufi_keys[destination_id]:
            raise RuntimeError("MuFilter mapping changed system, plane, or view")
        for field in ("valid", "n_sipms", "n_sides", "signals", "times", "daq"):
            if source_mufi[source_id][field] != output_mufi[destination_id][field]:
                raise RuntimeError(
                    f"MuFilter payload mismatch for {source_id}->{destination_id}: {field}"
                )
    for row in unchanged_mufi:
        source_id = int(row["source_det_id"])
        destination_id = int(row["destination_det_id"])
        if source_id != destination_id or int(row["channel_offset"]) != 0:
            raise RuntimeError("Orthogonal MuFilter detector ID was changed")
        if source_mufi[source_id] != output_mufi[destination_id]:
            raise RuntimeError(f"Orthogonal MuFilter payload changed for {source_id}")

    event_policy = audit["event"].get("policy")
    if event_policy in {
        "slope_selected_single_axis", "proximity_constrained_single_axis",
        "two_station_average_proximity_single_axis",
        "all_plane_average_proximity_single_axis",
    }:
        stored = audit["event"]["axis_decision"]
        recomputed = choose_transform_axis(
            float(stored["slope_xz"]), float(stored["slope_yz"]),
            int(stored["pz_sign"]), str(stored["direction_source"]),
            float(audit.get("transform", {}).get("min_abs_slope", 0.0)),
            float(audit.get("transform", {}).get("min_axis_dominance_ratio", 1.0)),
        )
        if recomputed.selected_axis != stored["selected_axis"] or recomputed.entry_side != stored["entry_side"]:
            raise RuntimeError("Stored v3 axis/side decision is not reproducible")
        selected_view = int(stored["selected_view"])
        if any(int(row["view"]) != selected_view for row in mapped + dropped):
            raise RuntimeError("A SciFi orthogonal view was remapped or dropped")
        if any(int(row["view"]) == selected_view for row in unchanged):
            raise RuntimeError("A selected SciFi view was marked unchanged")
        if any(int(row["view"]) != selected_view for row in mapped_mufi + dropped_mufi):
            raise RuntimeError("A MuFilter orthogonal view was remapped or dropped")
        if any(int(row["view"]) == selected_view for row in unchanged_mufi):
            raise RuntimeError("A selected MuFilter view was marked unchanged")
        source_scifi_orthogonal = [
            int(hit.GetDetectorID()) for hit in source_tree.Digi_ScifiHits
            if int(bool(hit.isVertical())) != selected_view
        ]
        output_scifi_orthogonal = [
            int(hit.GetDetectorID()) for hit in output_tree.Digi_ScifiHits
            if int(bool(hit.isVertical())) != selected_view
        ]
        if source_scifi_orthogonal != output_scifi_orthogonal:
            raise RuntimeError("Orthogonal SciFi collection order changed")
        source_mufi_orthogonal = [
            int(hit.GetDetectorID()) for hit in source_tree.Digi_MuFilterHits
            if int(bool(hit.isVertical())) != selected_view
        ]
        output_mufi_orthogonal = [
            int(hit.GetDetectorID()) for hit in output_tree.Digi_MuFilterHits
            if int(bool(hit.isVertical())) != selected_view
        ]
        if source_mufi_orthogonal != output_mufi_orthogonal:
            raise RuntimeError("Orthogonal MuFilter collection order changed")
        reference = audit["event"]["reference"]
        tolerance = float(audit.get("validation", {}).get("track_edge_tolerance_cm", 1.0e-8))
        if event_policy in {
            "two_station_average_proximity_single_axis",
            "all_plane_average_proximity_single_axis",
        }:
            stored_activity = audit["event"].get("activity_line")
            if not stored_activity:
                raise RuntimeError("Missing v5 activity-line audit")
            coordinates = {}
            for row in scifi_rows:
                key = (int(row["plane"]), int(row["view"]))
                coordinates.setdefault(key, []).append(float(row["source_coordinate_cm"]))
            plane_z = {}
            if event_policy == "all_plane_average_proximity_single_axis":
                for points_name in ("x_points", "y_points"):
                    for point in stored_activity[points_name]:
                        plane_z[(int(point["station"]), int(point["view"]))] = float(point["z_cm"])
                recomputed_activity = build_all_plane_activity_line(
                    coordinates, plane_z, int(stored["pz_sign"]),
                )
            else:
                for point_name in ("first", "second"):
                    point = stored_activity[point_name]
                    station = int(point["station"])
                    plane_z[(station, 1)] = float(point["z_x_cm"])
                    plane_z[(station, 0)] = float(point["z_y_cm"])
                recomputed_activity = build_two_station_activity_line(
                    coordinates, plane_z, int(stored["pz_sign"]),
                )
            recomputed_value = asdict(recomputed_activity)
            if event_policy == "all_plane_average_proximity_single_axis":
                for points_name in ("x_points", "y_points"):
                    expected_points = stored_activity[points_name]
                    actual_points = recomputed_value[points_name]
                    if len(actual_points) != len(expected_points):
                        raise RuntimeError(f"v5 activity point count differs for {points_name}")
                    for index, (actual_point, expected_point) in enumerate(zip(actual_points, expected_points)):
                        for name in ("station", "view", "n_hits"):
                            if int(actual_point[name]) != int(expected_point[name]):
                                raise RuntimeError(f"v5 activity point differs for {points_name}[{index}].{name}")
                        for name in ("coordinate_cm", "z_cm"):
                            if abs(float(actual_point[name]) - float(expected_point[name])) > 1.0e-10:
                                raise RuntimeError(f"v5 activity point differs for {points_name}[{index}].{name}")
            else:
                for point_name in ("first", "second"):
                    expected_point = stored_activity[point_name]
                    actual_point = recomputed_value[point_name]
                    for name in ("station", "n_x_hits", "n_y_hits"):
                        if int(actual_point[name]) != int(expected_point[name]):
                            raise RuntimeError(f"v5 activity point differs for {point_name}.{name}")
                    for name in ("x_cm", "y_cm", "z_x_cm", "z_y_cm"):
                        if abs(float(actual_point[name]) - float(expected_point[name])) > 1.0e-10:
                            raise RuntimeError(f"v5 activity point differs for {point_name}.{name}")
            for name in (
                "reference_z_cm", "reference_x_cm", "reference_y_cm",
                "slope_xz", "slope_yz",
            ):
                if abs(
                    float(recomputed_value["line"][name])
                    - float(stored_activity["line"][name])
                ) > 1.0e-10:
                    raise RuntimeError(f"v5 activity line differs for {name}")
            if stored.get("direction_source") not in {
                "assumed_positive_z", "assumed_negative_z",
            }:
                raise RuntimeError("v5 activity line unexpectedly used track momentum")
        if event_policy in {
            "proximity_constrained_single_axis",
            "two_station_average_proximity_single_axis",
            "all_plane_average_proximity_single_axis",
        }:
            if audit["transform"].get("edge_policy") != "closest_within_threshold_else_slope":
                raise RuntimeError("Unexpected v4 edge policy configuration")
            stored_edge = audit["event"].get("edge_decision")
            if not stored_edge:
                raise RuntimeError("Missing v4 edge decision audit")
            recomputed_edge = choose_target_edge(
                float(reference["track_coordinate_before_cm"]),
                float(stored_edge["low_edge_coordinate_cm"]),
                float(stored_edge["high_edge_coordinate_cm"]),
                recomputed.entry_side,
                float(audit["transform"]["near_edge_threshold_cm"]),
            )
            for name in (
                "selection_policy", "slope_entry_side", "closest_side",
                "target_side", "near_edge_override",
            ):
                if getattr(recomputed_edge, name) != stored_edge[name]:
                    raise RuntimeError(f"Stored v4 edge decision differs for {name}")
            for name in (
                "distance_to_low_edge_cm", "distance_to_high_edge_cm",
                "closest_edge_distance_cm", "near_edge_threshold_cm",
            ):
                if abs(getattr(recomputed_edge, name) - float(stored_edge[name])) > tolerance:
                    raise RuntimeError(f"Stored v4 edge distance differs for {name}")
            if reference.get("target_side") != recomputed_edge.target_side:
                raise RuntimeError("Reference target side differs from v4 edge decision")
            expected_edge = (
                recomputed_edge.low_edge_coordinate_cm
                if recomputed_edge.target_side == "low"
                else recomputed_edge.high_edge_coordinate_cm
            )
            if abs(float(reference["edge_coordinate_cm"]) - expected_edge) > tolerance:
                raise RuntimeError("v4 reference edge is not the selected target edge")
        if abs(float(reference["track_coordinate_after_cm"]) - float(reference["edge_coordinate_cm"])) > tolerance:
            raise RuntimeError("Translated track does not land on the requested edge")
        requested_shift = float(reference["requested_shift_cm"])
        for plane in audit["event"].get("mufilter_planes", []):
            if plane.get("action") != "remapped":
                continue
            shift = plane.get("shift", {})
            if shift.get("selection_policy") != "closest_physical_translation":
                raise RuntimeError("MuFilter plane did not use physical-translation mapping")
            if abs(float(shift["requested_shift_cm"]) - requested_shift) > tolerance:
                raise RuntimeError("MuFilter plane changed the requested global translation")
            quantization = float(shift["quantization_tolerance_cm"])
            translation_error = abs(float(shift["translation_error_cm"]))
            if translation_error > quantization + tolerance:
                raise RuntimeError(
                    "MuFilter realized translation exceeds channel quantization: "
                    f"{translation_error} > {quantization} cm"
                )
            if int(shift["mapped_hits"]) != int(plane["counts"].get("mapped", 0)):
                raise RuntimeError("MuFilter translation audit mapped-hit count mismatch")
            plane_dropped = sum(
                int(plane["counts"].get(name, 0))
                for name in ("outside_low", "outside_high")
            )
            if int(shift["dropped_hits"]) != plane_dropped:
                raise RuntimeError("MuFilter translation audit dropped-hit count mismatch")
        planes = audit["event"]["track_planes_in_travel_order"]
        for left, right in zip(planes, planes[1:]):
            step = float(right["after_cm"]) - float(left["after_cm"])
            if step * float(stored["travel_slope"]) <= 0:
                raise RuntimeError("Translated track does not move inward along travel order")
    metadata = output_file.Get("detid_channel_shift_metadata")
    if not metadata:
        raise RuntimeError("Missing detid_channel_shift_metadata")
    metadata_value = json.loads(metadata.GetTitle())
    if not metadata_value.get("synthetic_daq_warning"):
        raise RuntimeError("Synthetic DAQ warning missing from output metadata")
    for object_name in ("BranchList", "TimeBasedBranchList", "FileHeader", "cbmout"):
        if not output_file.GetKey(object_name):
            raise RuntimeError(f"Missing required FairRoot object {object_name}")

    write_marker(
        args.marker, identity[1], len(source_scifi), len(output_scifi),
        len(source_mufi), len(output_mufi), audit["version"],
    )
    print(
        f"detID channel-shift validation passed: event {identity[1]}, "
        f"SciFi {len(source_scifi)} -> {len(output_scifi)} "
        f"({len(dropped)} edge drops), MuFilter {len(source_mufi)} -> "
        f"{len(output_mufi)} ({len(dropped_mufi)} edge drops)"
    )
    source_file.Close()
    output_file.Close()


if __name__ == "__main__":
    main()
