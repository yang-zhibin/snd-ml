#!/usr/bin/env python3
"""Write a one-event SNDSW raw-digi file with channel-shifted SciFi detIDs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import ROOT
import yaml

from detid_channel_shift import (
    PlaneKey,
    TrackLine,
    build_all_plane_activity_line,
    build_five_station_activity_line,
    choose_anchor,
    choose_target_edge,
    choose_translation_offset,
    choose_transform_axis,
    map_detector_ids,
    summarize_mapping,
)
from detid_channel_shift_sndsw import (
    SciFiChannelGeometry,
    decode_scifi_detid,
    digi_payload,
    event_identity,
    gap_audit,
    layout_matches_axis,
    mufilter_hit_key,
    public_eos_path,
)


ROOT.gROOT.SetBatch(True)


def atomic_json(value, output_path: str) -> None:
    output = Path(output_path)
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


def atomic_csv(rows: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = []
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
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_selected(path: str, sequence: int) -> dict:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if int(row["event_sequence"]) == sequence]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one manifest row for event_sequence={sequence}")
    return selected[0]


def resolve_tree(source, candidates: list[str]):
    for name in candidates:
        tree = source.Get(name)
        if tree:
            return name, tree
    raise RuntimeError(f"No event tree from {candidates}")


def closest_side(layout, coordinate: float) -> str:
    low = layout.channels[0].centre_cm
    high = layout.channels[-1].centre_cm
    return "low" if abs(coordinate - low) <= abs(high - coordinate) else "high"


def optional_float(row: dict, name: str) -> float | None:
    raw = row.get(name, "")
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def transform_event_v3(tree, layouts, mufilter_layouts, row, config):
    """Apply the v3/v4 single-axis policy while preserving other views."""
    transform_config = config["transform"]
    direction_config = transform_config.get("direction", {})
    momentum_z = optional_float(row, "track_mom_z")
    if direction_config.get("use_track_pz", True) and momentum_z not in (None, 0.0):
        pz_sign = 1 if momentum_z > 0 else -1
        direction_source = "track_pz"
    else:
        pz_sign = int(direction_config["assumed_pz_sign"])
        direction_source = "assumed_positive_z" if pz_sign > 0 else "assumed_negative_z"
    decision = choose_transform_axis(
        float(row["track_slope_xz"]),
        float(row["track_slope_yz"]),
        pz_sign,
        direction_source,
        float(transform_config.get("min_abs_slope", 0.0)),
        float(transform_config.get("min_axis_dominance_ratio", 1.0)),
    )
    track = TrackLine(
        float(row["reference_z_cm"]),
        float(row["reference_x_cm"]),
        float(row["reference_y_cm"]),
        float(row["track_slope_xz"]),
        float(row["track_slope_yz"]),
    )
    collection = tree.Digi_ScifiHits
    original_payload = {int(hit.GetDetectorID()): digi_payload(hit) for hit in collection}
    if len(original_payload) != int(collection.GetEntriesFast()):
        raise RuntimeError("Input event contains duplicate SciFi detector IDs")

    stations = range(1, 6) if decision.pz_sign > 0 else range(5, 0, -1)
    reference_station = None
    reference_hits = []
    for station in stations:
        candidates = [
            int(hit.GetDetectorID()) for hit in collection
            if int(hit.GetStation()) == station
            and int(bool(hit.isVertical())) == decision.selected_view
        ]
        if candidates:
            reference_station, reference_hits = station, candidates
            break
    if reference_station is None:
        raise RuntimeError("No SciFi reference-plane hits in the selected view")
    reference_key = PlaneKey("Scifi", 0, reference_station, decision.selected_view)
    reference_layout = layouts[reference_key]
    reference_z = sum(channel.z_cm for channel in reference_layout.channels) / len(
        reference_layout.channels
    )
    track_coordinate = track.coordinate_at(decision.selected_axis, reference_z)
    edge_policy = transform_config.get("edge_policy", "slope_entry_side")
    edge_decision = None
    if edge_policy == "closest_within_threshold_else_slope":
        edge_decision = choose_target_edge(
            track_coordinate,
            reference_layout.channels[0].centre_cm,
            reference_layout.channels[-1].centre_cm,
            decision.entry_side,
            float(transform_config["near_edge_threshold_cm"]),
        )
        target_side = edge_decision.target_side
    elif edge_policy == "slope_entry_side":
        target_side = decision.entry_side
    else:
        raise ValueError(f"Unsupported edge policy {edge_policy!r}")
    edge = reference_layout.channels[0 if target_side == "low" else -1]
    requested_shift = edge.centre_cm - track_coordinate
    selected_shift = choose_anchor(
        reference_layout, reference_hits, target_side, requested_shift,
    )

    decision_fields = {
        "selected_axis": decision.selected_axis,
        "selected_view": decision.selected_view,
        "slope_xz": decision.slope_xz,
        "slope_yz": decision.slope_yz,
        "axis_dominance_ratio": decision.axis_dominance_ratio,
        "pz": momentum_z,
        "pz_sign": decision.pz_sign,
        "direction_source": decision.direction_source,
        "travel_slope": decision.travel_slope,
        "entry_side": decision.entry_side,
        "target_side": target_side,
        "edge_selection_policy": (
            edge_decision.selection_policy if edge_decision else "slope_entry_side"
        ),
        "reference_plane": reference_station,
        "reference_z_cm": reference_z,
        "track_coordinate_before_cm": track_coordinate,
        "track_coordinate_after_cm": track_coordinate + requested_shift,
        "requested_shift_cm": requested_shift,
    }
    audit_rows, mapped_by_source, plane_summaries = [], {}, []
    for station in range(1, 6):
        for view in (0, 1):
            key = PlaneKey("Scifi", 0, station, view)
            layout = layouts[key]
            source_ids = [
                int(hit.GetDetectorID()) for hit in collection
                if int(hit.GetStation()) == station and int(bool(hit.isVertical())) == view
            ]
            if layout_matches_axis(key, decision.selected_axis):
                mapped = map_detector_ids(layout, source_ids, selected_shift)
                plane_summaries.append({
                    "key": asdict(key), "action": "remapped",
                    **summarize_mapping(mapped),
                })
                values = mapped
            else:
                plane_summaries.append({
                    "key": asdict(key), "action": "unchanged_orthogonal",
                    "input_hits": len(source_ids),
                    "counts": {"unchanged_orthogonal": len(source_ids)},
                    "count_closure": True,
                })
                values = []
            for item in values:
                mapped_by_source[item.source_det_id] = item
                source_mat, source_sipm, source_channel = decode_scifi_detid(item.source_det_id)[2:]
                destination = (
                    decode_scifi_detid(item.destination_det_id)[2:]
                    if item.destination_det_id is not None else (None, None, None)
                )
                audit_rows.append({
                    "runId": int(row["runId"]), "eventId": int(row["eventId"]),
                    "detector": "Scifi", "system": 0, "plane": station, "view": view,
                    "source_det_id": item.source_det_id,
                    "destination_det_id": item.destination_det_id,
                    "source_ordinal": item.source_ordinal,
                    "destination_ordinal": item.destination_ordinal,
                    "status": item.status, "mapping_action": item.status,
                    "source_coordinate_cm": item.source_coordinate_cm,
                    "destination_coordinate_cm": item.destination_coordinate_cm,
                    "realized_shift_cm": item.realized_shift_cm,
                    "anchor_det_id": selected_shift.anchor_det_id,
                    "edge_det_id": selected_shift.edge_det_id,
                    "channel_offset": selected_shift.offset,
                    "source_mat": source_mat, "source_sipm": source_sipm,
                    "source_channel": source_channel,
                    "destination_mat": destination[0], "destination_sipm": destination[1],
                    "destination_channel": destination[2], **decision_fields,
                })
            if not layout_matches_axis(key, decision.selected_axis):
                by_id = layout.by_det_id
                for det_id in source_ids:
                    address = by_id[det_id]
                    mapped_by_source[det_id] = None
                    mat, sipm, channel = decode_scifi_detid(det_id)[2:]
                    audit_rows.append({
                        "runId": int(row["runId"]), "eventId": int(row["eventId"]),
                        "detector": "Scifi", "system": 0, "plane": station, "view": view,
                        "source_det_id": det_id, "destination_det_id": det_id,
                        "source_ordinal": address.ordinal,
                        "destination_ordinal": address.ordinal,
                        "status": "unchanged_orthogonal",
                        "mapping_action": "unchanged_orthogonal",
                        "source_coordinate_cm": address.centre_cm,
                        "destination_coordinate_cm": address.centre_cm,
                        "realized_shift_cm": 0.0, "anchor_det_id": None,
                        "edge_det_id": None, "channel_offset": 0,
                        "source_mat": mat, "source_sipm": sipm, "source_channel": channel,
                        "destination_mat": mat, "destination_sipm": sipm,
                        "destination_channel": channel, **decision_fields,
                    })

    for index in reversed(range(int(collection.GetEntriesFast()))):
        hit = collection.At(index)
        mapping = mapped_by_source[int(hit.GetDetectorID())]
        if mapping is None:
            continue
        if mapping.destination_det_id is None:
            collection.RemoveAt(index)
        else:
            hit.SetDetectorID(int(mapping.destination_det_id))
    collection.Compress()

    mufilter_collection = tree.Digi_MuFilterHits
    original_mufilter_payload = {
        int(hit.GetDetectorID()): digi_payload(hit) for hit in mufilter_collection
    }
    if len(original_mufilter_payload) != int(mufilter_collection.GetEntriesFast()):
        raise RuntimeError("Input event contains duplicate MuFilter detector IDs")
    mufilter_mapped_by_source, mufilter_plane_summaries = {}, []
    for key, layout in sorted(mufilter_layouts.items()):
        source_ids = [
            int(hit.GetDetectorID()) for hit in mufilter_collection
            if mufilter_hit_key(hit) == key
        ]
        if not source_ids:
            continue
        if layout_matches_axis(key, decision.selected_axis):
            plane_shift = choose_translation_offset(
                layout, source_ids, requested_shift,
            )
            mapped = map_detector_ids(layout, source_ids, plane_shift)
            mufilter_plane_summaries.append({
                "key": asdict(key), "action": "remapped", "shift": asdict(plane_shift),
                **summarize_mapping(mapped),
            })
            for item in mapped:
                mufilter_mapped_by_source[item.source_det_id] = item
                destination_bar = (
                    int(item.destination_det_id) % 1000
                    if item.destination_det_id is not None else None
                )
                audit_rows.append({
                    "runId": int(row["runId"]), "eventId": int(row["eventId"]),
                    "detector": "MuFilter", "system": key.system,
                    "plane": key.plane, "view": key.view,
                    "source_det_id": item.source_det_id,
                    "destination_det_id": item.destination_det_id,
                    "source_ordinal": item.source_ordinal,
                    "destination_ordinal": item.destination_ordinal,
                    "status": item.status, "mapping_action": item.status,
                    "source_coordinate_cm": item.source_coordinate_cm,
                    "destination_coordinate_cm": item.destination_coordinate_cm,
                    "realized_shift_cm": item.realized_shift_cm,
                    "anchor_det_id": None,
                    "edge_det_id": None,
                    "channel_offset": plane_shift.offset,
                    "shift_selection_policy": plane_shift.selection_policy,
                    "reference_source_det_id": plane_shift.reference_source_det_id,
                    "reference_destination_det_id": plane_shift.reference_destination_det_id,
                    "plane_translation_error_cm": plane_shift.translation_error_cm,
                    "plane_quantization_tolerance_cm": plane_shift.quantization_tolerance_cm,
                    "source_bar": int(item.source_det_id) % 1000,
                    "destination_bar": destination_bar, **decision_fields,
                })
        else:
            mufilter_plane_summaries.append({
                "key": asdict(key), "action": "unchanged_orthogonal",
                "input_hits": len(source_ids),
                "counts": {"unchanged_orthogonal": len(source_ids)},
                "count_closure": True,
            })
            by_id = layout.by_det_id
            for det_id in source_ids:
                address = by_id[det_id]
                mufilter_mapped_by_source[det_id] = None
                audit_rows.append({
                    "runId": int(row["runId"]), "eventId": int(row["eventId"]),
                    "detector": "MuFilter", "system": key.system,
                    "plane": key.plane, "view": key.view,
                    "source_det_id": det_id, "destination_det_id": det_id,
                    "source_ordinal": address.ordinal,
                    "destination_ordinal": address.ordinal,
                    "status": "unchanged_orthogonal",
                    "mapping_action": "unchanged_orthogonal",
                    "source_coordinate_cm": address.centre_cm,
                    "destination_coordinate_cm": address.centre_cm,
                    "realized_shift_cm": 0.0, "anchor_det_id": None,
                    "edge_det_id": None, "channel_offset": 0,
                    "source_bar": det_id % 1000, "destination_bar": det_id % 1000,
                    **decision_fields,
                })
    if len(mufilter_mapped_by_source) != len(original_mufilter_payload):
        raise RuntimeError("Not every MuFilter digi was assigned to a layout")
    for index in reversed(range(int(mufilter_collection.GetEntriesFast()))):
        hit = mufilter_collection.At(index)
        mapping = mufilter_mapped_by_source[int(hit.GetDetectorID())]
        if mapping is None:
            continue
        if mapping.destination_det_id is None:
            mufilter_collection.RemoveAt(index)
        else:
            hit.SetDetectorID(int(mapping.destination_det_id))
    mufilter_collection.Compress()

    transformed_payload = {int(hit.GetDetectorID()): digi_payload(hit) for hit in collection}
    transformed_mufilter_payload = {
        int(hit.GetDetectorID()): digi_payload(hit) for hit in mufilter_collection
    }
    kept_rows = [item for item in audit_rows if item["status"] in {"mapped", "unchanged_orthogonal"}]
    if len(transformed_payload) + len(transformed_mufilter_payload) != len(kept_rows):
        raise RuntimeError("Output detector IDs do not close against retained audit rows")
    for item in kept_rows:
        source_values = original_payload if item["detector"] == "Scifi" else original_mufilter_payload
        destination_values = transformed_payload if item["detector"] == "Scifi" else transformed_mufilter_payload
        source_value = source_values[int(item["source_det_id"])]
        destination_value = destination_values[int(item["destination_det_id"])]
        for field in ("valid", "n_sipms", "n_sides", "signals", "times", "daq"):
            if source_value[field] != destination_value[field]:
                raise RuntimeError(f"Digi payload field {field} changed for {item['source_det_id']}")

    track_planes = []
    for station in stations:
        layout = layouts[PlaneKey("Scifi", 0, station, decision.selected_view)]
        z_cm = sum(channel.z_cm for channel in layout.channels) / len(layout.channels)
        before = track.coordinate_at(decision.selected_axis, z_cm)
        track_planes.append({
            "station": station, "z_cm": z_cm, "before_cm": before,
            "after_cm": before + requested_shift,
        })
    if len(track_planes) > 1:
        step = track_planes[1]["after_cm"] - track_planes[0]["after_cm"]
        if step * decision.travel_slope <= 0:
            raise RuntimeError("Transformed track does not move in its travel-slope direction")
    return audit_rows, {
        "policy": (
            "proximity_constrained_single_axis"
            if edge_decision is not None else "slope_selected_single_axis"
        ),
        "axis_decision": asdict(decision),
        "edge_decision": asdict(edge_decision) if edge_decision else None,
        "track": asdict(track),
        "track_momentum_z": momentum_z,
        "reference": {
            "station": reference_station, "view": decision.selected_view,
            "z_cm": reference_z, "track_coordinate_before_cm": track_coordinate,
            "track_coordinate_after_cm": track_coordinate + requested_shift,
            "edge_coordinate_cm": edge.centre_cm,
            "target_side": target_side,
            "requested_shift_cm": requested_shift,
        },
        "selected_shift": asdict(selected_shift),
        "track_planes_in_travel_order": track_planes,
        "planes": plane_summaries,
        "mufilter_planes": mufilter_plane_summaries,
        "input_scifi_hits": len(original_payload),
        "output_scifi_hits": len(transformed_payload),
        "dropped_scifi_hits": len(original_payload) - len(transformed_payload),
        "input_mufilter_hits": len(original_mufilter_payload),
        "output_mufilter_hits": len(transformed_mufilter_payload),
        "dropped_mufilter_hits": len(original_mufilter_payload) - len(transformed_mufilter_payload),
        "payload_validation_passed": True,
    }


def transform_event_v5(tree, layouts, mufilter_layouts, row, config):
    """Fit all active SciFi plane averages, then apply the v4 transform."""
    direction_config = config["transform"].get("direction", {})
    pz_sign = int(direction_config.get("assumed_pz_sign", 1))
    collection = tree.Digi_ScifiHits
    coordinates = {}
    for hit in collection:
        station = int(hit.GetStation())
        view = int(bool(hit.isVertical()))
        key = PlaneKey("Scifi", 0, station, view)
        det_id = int(hit.GetDetectorID())
        coordinates.setdefault((station, view), []).append(
            layouts[key].by_det_id[det_id].centre_cm
        )
    plane_z = {}
    for station in range(1, 6):
        for view in (0, 1):
            layout = layouts[PlaneKey("Scifi", 0, station, view)]
            plane_z[(station, view)] = sum(
                channel.z_cm for channel in layout.channels
            ) / len(layout.channels)
    strict_five_station = bool(
        config["transform"].get("require_all_five_stations_both_views", False)
    )
    builder = (
        build_five_station_activity_line
        if strict_five_station else build_all_plane_activity_line
    )
    activity = builder(coordinates, plane_z, pz_sign)
    line = activity.line
    activity_row = dict(row)
    activity_row.update({
        "reference_z_cm": line.reference_z_cm,
        "reference_x_cm": line.reference_x_cm,
        "reference_y_cm": line.reference_y_cm,
        "track_slope_xz": line.slope_xz,
        "track_slope_yz": line.slope_yz,
        # Deliberately prevent the v3/v4 helper from reading reconstructed
        # momentum: the activity method has no reconstructed-track dependency.
        "track_mom_z": "",
    })
    audit_rows, event_audit = transform_event_v3(
        tree, layouts, mufilter_layouts, activity_row, config,
    )
    event_audit.update({
        "policy": "all_plane_average_proximity_single_axis",
        "line_source": "all_active_scifi_plane_average_linear_fit",
        "activity_line": asdict(activity),
        "track_momentum_z": None,
        "strict_five_station_both_views": strict_five_station,
    })
    return audit_rows, event_audit


def transform_event(tree, layouts, mufilter_layouts, row, config):
    collection = tree.Digi_ScifiHits
    original_payload = {int(hit.GetDetectorID()): digi_payload(hit) for hit in collection}
    if len(original_payload) != int(collection.GetEntriesFast()):
        raise RuntimeError("Input event contains duplicate SciFi detector IDs")

    transform_config = config["transform"]
    reference_station = int(transform_config.get("reference_station", 3))
    shifts = {}
    for view, coordinate_name in ((1, "x"), (0, "y")):
        key = PlaneKey("Scifi", 0, reference_station, view)
        layout = layouts[key]
        reference_coordinate = float(row[f"reference_{coordinate_name}_cm"])
        configured_side = transform_config.get(f"{coordinate_name}_side", "auto")
        side = closest_side(layout, reference_coordinate) if configured_side == "auto" else configured_side
        edge = layout.channels[0 if side == "low" else -1]
        configured_shift = transform_config.get(f"requested_shift_{coordinate_name}_cm", "auto")
        requested_shift = (
            edge.centre_cm - reference_coordinate
            if configured_shift == "auto" else float(configured_shift)
        )
        reference_hits = [
            int(hit.GetDetectorID()) for hit in collection
            if int(hit.GetStation()) == reference_station and int(bool(hit.isVertical())) == view
        ]
        shifts[view] = choose_anchor(layout, reference_hits, side, requested_shift)

    audit_rows = []
    mapped_by_source = {}
    plane_summaries = []
    for station in range(1, 6):
        for view in (0, 1):
            key = PlaneKey("Scifi", 0, station, view)
            layout = layouts[key]
            source_ids = [
                int(hit.GetDetectorID()) for hit in collection
                if int(hit.GetStation()) == station and int(bool(hit.isVertical())) == view
            ]
            mapped = map_detector_ids(layout, source_ids, shifts[view])
            summary = summarize_mapping(mapped)
            plane_summaries.append({"key": asdict(key), **summary})
            for item in mapped:
                mapped_by_source[item.source_det_id] = item
                source_mat, source_sipm, source_channel = decode_scifi_detid(item.source_det_id)[2:]
                destination = (
                    decode_scifi_detid(item.destination_det_id)[2:]
                    if item.destination_det_id is not None else (None, None, None)
                )
                audit_rows.append({
                    "runId": int(row["runId"]),
                    "eventId": int(row["eventId"]),
                    "detector": "Scifi",
                    "system": 0,
                    "plane": station,
                    "view": view,
                    "source_det_id": item.source_det_id,
                    "destination_det_id": item.destination_det_id,
                    "source_ordinal": item.source_ordinal,
                    "destination_ordinal": item.destination_ordinal,
                    "status": item.status,
                    "source_coordinate_cm": item.source_coordinate_cm,
                    "destination_coordinate_cm": item.destination_coordinate_cm,
                    "realized_shift_cm": item.realized_shift_cm,
                    "anchor_det_id": shifts[view].anchor_det_id,
                    "edge_det_id": shifts[view].edge_det_id,
                    "channel_offset": shifts[view].offset,
                    "source_mat": source_mat,
                    "source_sipm": source_sipm,
                    "source_channel": source_channel,
                    "destination_mat": destination[0],
                    "destination_sipm": destination[1],
                    "destination_channel": destination[2],
                })

    for index in reversed(range(int(collection.GetEntriesFast()))):
        hit = collection.At(index)
        mapping = mapped_by_source[int(hit.GetDetectorID())]
        if mapping.destination_det_id is None:
            collection.RemoveAt(index)
        else:
            hit.SetDetectorID(int(mapping.destination_det_id))
    collection.Compress()

    transformed_payload = {int(hit.GetDetectorID()): digi_payload(hit) for hit in collection}
    mapped_rows = [row_value for row_value in audit_rows if row_value["status"] == "mapped"]
    if len(transformed_payload) != len(mapped_rows):
        raise RuntimeError("Output SciFi detector IDs are not unique")
    for item in mapped_rows:
        source = original_payload[int(item["source_det_id"])]
        destination = transformed_payload[int(item["destination_det_id"])]
        for field in ("valid", "n_sipms", "n_sides", "signals", "times", "daq"):
            if source[field] != destination[field]:
                raise RuntimeError(
                    f"Digi payload field {field} changed for {item['source_det_id']}"
                )

    mufilter_collection = tree.Digi_MuFilterHits
    original_mufilter_payload = {
        int(hit.GetDetectorID()): digi_payload(hit) for hit in mufilter_collection
    }
    if len(original_mufilter_payload) != int(mufilter_collection.GetEntriesFast()):
        raise RuntimeError("Input event contains duplicate MuFilter detector IDs")
    mufilter_mapped_by_source = {}
    mufilter_plane_summaries = []
    for key, layout in sorted(mufilter_layouts.items()):
        source_ids = [
            int(hit.GetDetectorID()) for hit in mufilter_collection
            if mufilter_hit_key(hit) == key
        ]
        if not source_ids:
            continue
        orientation_shift = shifts[key.view]
        plane_shift = choose_anchor(
            layout, source_ids, orientation_shift.side,
            orientation_shift.requested_shift_cm,
        )
        mapped = map_detector_ids(layout, source_ids, plane_shift)
        mufilter_plane_summaries.append({
            "key": asdict(key), "shift": asdict(plane_shift),
            **summarize_mapping(mapped),
        })
        for item in mapped:
            mufilter_mapped_by_source[item.source_det_id] = item
            source_bar = int(item.source_det_id) % 1000
            destination_bar = (
                int(item.destination_det_id) % 1000
                if item.destination_det_id is not None else None
            )
            audit_rows.append({
                "runId": int(row["runId"]),
                "eventId": int(row["eventId"]),
                "detector": "MuFilter",
                "system": key.system,
                "plane": key.plane,
                "view": key.view,
                "source_det_id": item.source_det_id,
                "destination_det_id": item.destination_det_id,
                "source_ordinal": item.source_ordinal,
                "destination_ordinal": item.destination_ordinal,
                "status": item.status,
                "source_coordinate_cm": item.source_coordinate_cm,
                "destination_coordinate_cm": item.destination_coordinate_cm,
                "realized_shift_cm": item.realized_shift_cm,
                "anchor_det_id": plane_shift.anchor_det_id,
                "edge_det_id": plane_shift.edge_det_id,
                "channel_offset": plane_shift.offset,
                "source_mat": None,
                "source_sipm": None,
                "source_channel": None,
                "destination_mat": None,
                "destination_sipm": None,
                "destination_channel": None,
                "source_bar": source_bar,
                "destination_bar": destination_bar,
            })
    if len(mufilter_mapped_by_source) != len(original_mufilter_payload):
        raise RuntimeError("Not every MuFilter digi was assigned to a layout")
    for index in reversed(range(int(mufilter_collection.GetEntriesFast()))):
        hit = mufilter_collection.At(index)
        mapping = mufilter_mapped_by_source[int(hit.GetDetectorID())]
        if mapping.destination_det_id is None:
            mufilter_collection.RemoveAt(index)
        else:
            hit.SetDetectorID(int(mapping.destination_det_id))
    mufilter_collection.Compress()
    transformed_mufilter_payload = {
        int(hit.GetDetectorID()): digi_payload(hit) for hit in mufilter_collection
    }
    mapped_mufilter_rows = [
        item for item in audit_rows
        if item["detector"] == "MuFilter" and item["status"] == "mapped"
    ]
    if len(transformed_mufilter_payload) != len(mapped_mufilter_rows):
        raise RuntimeError("Output MuFilter detector IDs are not unique")
    for item in mapped_mufilter_rows:
        source_payload = original_mufilter_payload[int(item["source_det_id"])]
        destination_payload = transformed_mufilter_payload[int(item["destination_det_id"])]
        for field in ("valid", "n_sipms", "n_sides", "signals", "times", "daq"):
            if source_payload[field] != destination_payload[field]:
                raise RuntimeError(
                    f"MuFilter payload field {field} changed for {item['source_det_id']}"
                )
    return audit_rows, {
        "shifts": {"vertical_x": asdict(shifts[1]), "horizontal_y": asdict(shifts[0])},
        "planes": plane_summaries,
        "input_scifi_hits": len(original_payload),
        "output_scifi_hits": len(transformed_payload),
        "dropped_scifi_hits": len(original_payload) - len(transformed_payload),
        "mufilter_planes": mufilter_plane_summaries,
        "input_mufilter_hits": len(original_mufilter_payload),
        "output_mufilter_hits": len(transformed_mufilter_payload),
        "dropped_mufilter_hits": (
            len(original_mufilter_payload) - len(transformed_mufilter_payload)
        ),
        "payload_validation_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--event-sequence", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-audit-csv", required=True)
    parser.add_argument("--output-audit-json", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    row = read_selected(args.manifest, args.event_sequence)

    source_path = public_eos_path(row["event_source_path"])
    source = ROOT.TFile.Open(source_path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {source_path}")
    tree_name, tree = resolve_tree(source, config["input"]["event_tree_candidates"])
    entry = int(row["source_event_index"])
    if tree.GetEntry(entry) <= 0:
        raise RuntimeError(f"Cannot read entry {entry}")
    if event_identity(tree) != (int(row["runId"]), int(row["eventId"])):
        raise RuntimeError(f"Source identity mismatch at entry {entry}")

    geometry = SciFiChannelGeometry(row["geo_path"])
    geometry.initialize_event(tree.EventHeader)
    layouts = geometry.build_layouts()
    mufilter_layouts = geometry.build_mufilter_layouts()

    output = Path(args.output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".root", dir=output.parent,
    )
    os.close(descriptor)
    os.unlink(temporary)
    audit_rows = []
    try:
        destination = ROOT.TFile.Open(temporary, "RECREATE")
        destination.cd()
        # FairRoot/SNDSW needs these file-level objects to initialize tasks
        # even though the transformed event tree itself has only raw branches.
        for object_name in ("BranchList", "TimeBasedBranchList", "FileHeader", "cbmout"):
            source_key = source.GetKey(object_name)
            if not source_key:
                raise RuntimeError(f"Required FairRoot object {object_name} is missing")
            source_object = source_key.ReadObj()
            destination.cd()
            write_option = (
                ROOT.TObject.kSingleKey
                if source_object.InheritsFrom("TCollection") else 0
            )
            source_object.Write(object_name, write_option)
        output_tree = tree.CloneTree(0)
        # CloneTree(0) changes branch addresses and clears the current branch
        # buffers. Reload the selected entry before transforming those buffers.
        if tree.GetEntry(entry) <= 0:
            raise RuntimeError(f"Cannot reload entry {entry} after CloneTree")
        axis_policy = config["transform"].get("axis_policy")
        if axis_policy == "largest_absolute_scifi_track_slope":
            audit_rows, event_audit = transform_event_v3(
                tree, layouts, mufilter_layouts, row, config,
            )
        elif axis_policy == "all_active_scifi_plane_averages_linear_fit":
            audit_rows, event_audit = transform_event_v5(
                tree, layouts, mufilter_layouts, row, config,
            )
        else:
            audit_rows, event_audit = transform_event(
                tree, layouts, mufilter_layouts, row, config,
            )
        output_tree.Fill()
        output_tree.Write(tree_name, ROOT.TObject.kOverwrite)
        metadata = {
            "transform_version": config["version"],
            "source_file": row["event_source_path"],
            "source_tree": tree_name,
            "source_entry": entry,
            "geometry_file": row["geo_path"],
            "run_id": int(row["runId"]),
            "event_id": int(row["eventId"]),
            "synthetic_daq_warning": True,
        }
        if "axis_decision" in event_audit:
            metadata.update(event_audit["axis_decision"])
            metadata.update({
                "axis_policy": config["transform"]["axis_policy"],
                "reference_plane": event_audit["reference"]["station"],
                "reference_z_cm": event_audit["reference"]["z_cm"],
                "requested_shift_cm": event_audit["reference"]["requested_shift_cm"],
                "selected_channel_offset": event_audit["selected_shift"]["offset"],
            })
            if event_audit.get("edge_decision"):
                metadata["edge_decision"] = event_audit["edge_decision"]
        ROOT.TNamed("detid_channel_shift_metadata", json.dumps(metadata, sort_keys=True)).Write()
        destination.Close()
        os.replace(temporary, output)
    finally:
        source.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)

    audit = {
        "version": config["version"],
        "identity": {"runId": int(row["runId"]), "eventId": int(row["eventId"])},
        "source": {"path": row["event_source_path"], "tree": tree_name, "entry": entry},
        "output_root": str(output.resolve()),
        "transform": config.get("transform", {}),
        "validation": config.get("validation", {}),
        "geometry": {
            "path": row["geo_path"],
            "template_validation": geometry.template_validation,
            "reference_gap_audit": {
                "horizontal": gap_audit(layouts[PlaneKey("Scifi", 0, 3, 0)]),
                "vertical": gap_audit(layouts[PlaneKey("Scifi", 0, 3, 1)]),
            },
            "mufilter_layout_channel_counts": {
                f"system_{key.system}_plane_{key.plane}_view_{key.view}": len(layout.channels)
                for key, layout in sorted(mufilter_layouts.items())
            },
        },
        "event": event_audit,
        "synthetic_daq_warning": (
            "Detector IDs are remapped while original DAQ identifiers are retained."
        ),
    }
    atomic_csv(audit_rows, args.output_audit_csv)
    atomic_json(audit, args.output_audit_json)
    print(
        f"Wrote {output}: {event_audit['input_scifi_hits']} -> "
        f"{event_audit['output_scifi_hits']} SciFi hits; "
        f"MuFilter {event_audit['input_mufilter_hits']} -> "
        f"{event_audit['output_mufilter_hits']}"
    )


if __name__ == "__main__":
    main()
