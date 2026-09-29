#!/usr/bin/env python3
"""Render original/transformed SciFi planes for Run-8329 Muon-DIS events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import yaml

from plot_passing_muon_v6_event_displays import source_batches
from scifi_edge_transform import (
    PlaneKey,
    build_plane_transforms,
    classify_transformed_coordinate,
    transform_hits,
    validate_transform,
)
from scifi_edge_transform_sndsw import (
    GeometryAdapter,
    event_identity,
    extract_digi_hits,
    extract_mufilter_hits,
    find_track_entry,
    open_track_tree,
    open_tree,
    select_track,
    serializable_track,
    transform_mufilter_hits,
)
from sndsw_event_display_common import (
    displayable_event_tree,
    find_sndsw_2d_display_script,
    merge_pages_to_pdf,
    run_sndsw_batch,
)


PAGE_TYPES = (
    "overview", "full_event_transform_comparison",
    "scifi_horizontal_comparison", "scifi_vertical_comparison",
)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(rows: list[dict], path: str):
    output = Path(path)
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
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: dict, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_safe(value):
    """Convert ROOT/numpy scalars and non-finite diagnostics to strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def pdf_page_count(path: str) -> int:
    with open(path, "rb") as stream:
        return len(re.findall(rb"/Type\s*/Page\b", stream.read()))


def annotations(row: dict) -> list[str]:
    return [
        "Population: Muon-DIS candidate (Veto hits present)",
        f"Selection reason: {row.get('selection_reason', '')}",
        f"Veto hit count: {row.get('count_veto', '')}",
        f"Track type/index: {row.get('track_type', '')} / {row.get('track_index', '')}",
        f"Source index / entry: {row['source_file_index']} / {row['source_event_index']}",
    ]


def _qdc_norm(hits):
    values = [max(0.0, hit.qdc) for hit in hits if hit.valid and math.isfinite(hit.qdc)]
    return Normalize(0.0, max(values) if values and max(values) > 0 else 1.0)


def _draw_geometry(axis, plane):
    axis.broken_barh(
        [(item.low_cm, item.high_cm - item.low_cm) for item in plane.intervals],
        (0.18, 0.64), facecolors="#f2f2f2", edgecolors="#d0d0d0", linewidth=0.05,
    )
    axis.axvline(plane.low_edge_cm, color="black", linewidth=0.8)
    axis.axvline(plane.high_edge_cm, color="black", linewidth=0.8)


def _scatter_original(axis, hits, norm):
    valid = [hit for hit in hits if hit.valid]
    invalid = [hit for hit in hits if not hit.valid]
    if valid:
        axis.scatter(
            [hit.coordinate_cm for hit in valid], [0.5] * len(valid),
            c=[max(0.0, hit.qdc) for hit in valid], cmap="viridis", norm=norm,
            marker="o", s=24, edgecolors="black", linewidths=0.25, zorder=4,
        )
    if invalid:
        axis.scatter(
            [hit.coordinate_cm for hit in invalid], [0.5] * len(invalid),
            color="red", marker="x", s=30, linewidths=0.9, zorder=5,
        )


def _scatter_transformed(axis, hits, norm):
    edge_color = {
        "inside_active": "black", "inside_gap": "#ff8c00", "outside_geometry": "red",
    }
    for location in edge_color:
        selected = [hit for hit in hits if hit.location == location and hit.source.valid]
        if selected:
            axis.scatter(
                [hit.transformed_coordinate_cm for hit in selected], [0.5] * len(selected),
                c=[max(0.0, hit.source.qdc) for hit in selected], cmap="viridis", norm=norm,
                marker="o", s=28, edgecolors=edge_color[location], linewidths=1.0,
                zorder=4,
            )
    invalid = [hit for hit in hits if not hit.source.valid]
    if invalid:
        axis.scatter(
            [hit.transformed_coordinate_cm for hit in invalid], [0.5] * len(invalid),
            color="red", marker="x", s=30, linewidths=0.9, zorder=5,
        )


def draw_comparison_page(
    pdf: PdfPages, orientation: str, planes, original_hits, transformed_hits,
    transforms, title: str, page_width: float,
):
    view = 1 if orientation == "vertical" else 0
    coordinate = "x" if view == 1 else "y"
    figure, axes = plt.subplots(
        5, 2, figsize=(page_width, 13.5), squeeze=False, sharey=True,
    )
    norm = _qdc_norm(original_hits)
    for station in range(1, 6):
        key = PlaneKey(station, view)
        plane = planes[key]
        transform = transforms[key]
        source_plane_hits = [hit for hit in original_hits if hit.key == key]
        target_plane_hits = [hit for hit in transformed_hits if hit.source.key == key]
        all_coordinates = [plane.low_edge_cm, plane.high_edge_cm]
        all_coordinates.extend(hit.coordinate_cm for hit in source_plane_hits)
        all_coordinates.extend(hit.transformed_coordinate_cm for hit in target_plane_hits)
        span = max(all_coordinates) - min(all_coordinates)
        padding = max(0.5, 0.03 * span)
        limits = (min(all_coordinates) - padding, max(all_coordinates) + padding)

        left, right = axes[station - 1]
        for axis in (left, right):
            _draw_geometry(axis, plane)
            axis.set_ylim(0, 1)
            axis.set_yticks([])
            axis.set_xlim(*limits)
            axis.grid(axis="x", alpha=0.15)
        _scatter_original(left, source_plane_hits, norm)
        left.axvline(
            transform.track_original_cm, color="#0066cc", linewidth=1.5,
            label="original track",
        )
        left.set_ylabel(f"Station {station}\nz={plane.z_cm:.2f} cm")
        left.set_title(
            f"Original: track={transform.track_original_cm:.3f} cm, "
            f"local closest={transform.plane_local_closest_side}", fontsize=9,
        )

        _scatter_transformed(right, target_plane_hits, norm)
        right.axvline(
            transform.target_cm, color="#cc0077", linewidth=1.6,
            label="target / transformed track",
        )
        right.set_title(
            f"Transformed: {transform.locked_side} edge, "
            f"delta {coordinate}={transform.delta_cm:+.3f} cm", fontsize=9,
        )
        if station == 5:
            left.set_xlabel(f"Original {coordinate} [cm]")
            right.set_xlabel(f"Transformed {coordinate} [cm]")
    figure.suptitle(
        f"{title}\nSciFi {orientation} planes ({coordinate} measurement)", fontsize=14,
    )
    figure.text(
        0.5, 0.006,
        "QDC: Viridis fill; active/gap/outside transformed coordinates: "
        "black/orange/red outline; invalid digi: red x",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0.015, 0.025, 0.995, 0.95))
    pdf.savefig(figure)
    plt.close(figure)


def build_sndsw_transform_payload(
    row, scifi_details, transformed_scifi, transformed_mufilter,
    scifi_transforms, mufilter_audit, track, scifi_half_width_cm,
):
    groups = {
        projection: {system: [] for system in ("Veto", "Scifi", "US", "DS")}
        for projection in ("X", "Y")
    }
    converted_scifi = {
        item.source.detector_id: item for item in transformed_scifi
    }
    for hit in scifi_details:
        view = int(hit["view"])
        projection = "X" if view == 1 else "Y"
        converted = converted_scifi[int(hit["detector_id"])]
        groups[projection]["Scifi"].append({
            "z_cm": float(hit["z_mid_cm"]),
            "u_cm": float(converted.transformed_coordinate_cm),
            "z_error_cm": max(0.05, 0.5 * abs(float(hit["z_b_cm"]) - float(hit["z_a_cm"]))),
            "u_error_cm": float(scifi_half_width_cm),
            "location": converted.location,
        })
    for hit in transformed_mufilter:
        projection = "X" if int(hit["view"]) == 1 else "Y"
        groups[projection][hit["system_label"]].append({
            "z_cm": float(hit["z_mid_cm"]),
            "u_cm": float(hit["transformed_coordinate_cm"]),
            "z_error_cm": max(0.1, 0.5 * abs(float(hit["z_b_cm"]) - float(hit["z_a_cm"]))),
            "u_error_cm": 0.5,
            "location": hit["location"],
        })

    original_tracks, transformed_tracks = {}, {}
    for projection, coordinate, view in (("X", "x", 1), ("Y", "y", 0)):
        fit = track[f"projection_{coordinate}"]
        original = [{"z_cm": 250.0, "u_cm": fit.at(250.0)}]
        original.extend(
            {"z_cm": float(item["z_cm"]), "u_cm": float(item[f"{coordinate}_cm"])}
            for item in sorted(track["collapsed_scifi_points"], key=lambda value: value["z_cm"])
        )
        original.append({"z_cm": 600.0, "u_cm": fit.at(600.0)})
        original_tracks[projection] = original
        targets = [
            {"z_cm": float(value.z_cm), "u_cm": float(value.target_cm)}
            for key, value in scifi_transforms.items() if key.view == view
        ]
        targets.extend(
            {"z_cm": float(value["z_cm"]), "u_cm": float(value["target_cm"])}
            for value in mufilter_audit["planes"] if int(value["key"]["view"]) == view
        )
        transformed_tracks[projection] = sorted(
            targets, key=lambda value: (value["z_cm"], value["u_cm"]),
        )
    return {
        "run_id": int(row["runId"]),
        "event_id": int(row["eventId"]),
        "track_type": int(track["track_type"]),
        "track_index": int(track["track_index"]),
        "transformed_hits": groups,
        "tracks": {"original": original_tracks, "transformed": transformed_tracks},
    }


def process_event(
    row: dict, config: dict, output_pdf: str, geometry_cache: dict,
    display_script: str,
) -> tuple[dict, list[dict]]:
    input_config = config["input"]
    raw_source, tree_name, raw_tree = open_tree(
        row["event_source_path"],
        list(input_config.get("event_tree_candidates", ["rawConv", "cbmsim"])),
    )
    track_source = None
    try:
        event_index = int(row["source_event_index"])
        if raw_tree.GetEntry(event_index) <= 0:
            raise RuntimeError(f"Could not load raw event entry {event_index}")
        expected_identity = (int(row["runId"]), int(row["eventId"]))
        observed_identity = event_identity(raw_tree)
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"Raw event identity mismatch: {observed_identity} != {expected_identity}"
            )
        geofile = row["geo_path"]
        geometry = geometry_cache.get(geofile)
        if geometry is None:
            geometry = GeometryAdapter(geofile)
            geometry_cache[geofile] = geometry
        geometry.initialize_event(raw_tree.EventHeader)
        planes = geometry.build_planes()
        hits, hit_details = extract_digi_hits(raw_tree, geometry)
        mufilter_hits = extract_mufilter_hits(raw_tree, geometry)
        original_location_counts = Counter()
        for hit in hits:
            location, _ = classify_transformed_coordinate(hit.coordinate_cm, planes[hit.key])
            original_location_counts[location] += 1
        if original_location_counts["inside_gap"] or original_location_counts["outside_geometry"]:
            raise RuntimeError(
                f"Original digis do not map to active channels: {dict(original_location_counts)}"
            )

        track_source, track_tree = open_track_tree(
            row["track_input_path"], input_config.get("track_tree", "sndData")
        )
        track_entry = find_track_entry(track_tree, *expected_identity)
        if track_entry != int(row["track_entry"]):
            raise RuntimeError(
                f"Track entry changed since selection: {track_entry} != {row['track_entry']}"
            )
        track_config = config["track"]
        track, eligible_tracks = select_track(
            track_tree,
            [int(value) for value in track_config["type_priority"]],
            int(track_config.get("min_ndf", 1)),
            int(track_config.get("min_scifi_planes_per_projection", 2)),
        )
        if track["track_index"] != int(row["track_index"]):
            raise RuntimeError(
                f"Selected track changed: {track['track_index']} != {row['track_index']}"
            )
        track_positions = {
            key: (
                track["projection_x"].at(plane.z_cm)
                if key.view == 1 else track["projection_y"].at(plane.z_cm)
            )
            for key, plane in planes.items()
        }
        transform_config = config["transform"]
        transforms = build_plane_transforms(
            list(planes.values()), track_positions,
            float(transform_config.get("inward_margin_cm", 0.0)),
            bool(transform_config.get("lock_side_per_orientation", True)),
        )
        converted = transform_hits(hits, planes, transforms)
        validation = validate_transform(
            hits, converted, transforms,
            float(transform_config.get("residual_tolerance_cm", 1e-9)),
            float(transform_config.get("track_edge_tolerance_cm", 1e-8)),
            bool(transform_config.get("lock_side_per_orientation", True)),
        )
        if not validation["passed"]:
            raise RuntimeError("Transform validation failed: " + "; ".join(validation["failures"]))

        mufilter_planes = geometry.build_mufilter_planes()
        side_by_view = {}
        for view in (0, 1):
            sides = {value.locked_side for key, value in transforms.items() if key.view == view}
            if len(sides) != 1:
                raise RuntimeError(f"SciFi side lock failed for view {view}: {sorted(sides)}")
            side_by_view[view] = next(iter(sides))
        transformed_mufilter, mufilter_validation = transform_mufilter_hits(
            mufilter_hits, mufilter_planes,
            track["projection_x"], track["projection_y"], side_by_view,
            float(transform_config.get("inward_margin_cm", 0.0)),
            float(transform_config.get("residual_tolerance_cm", 1e-9)),
        )

        title = (
            f"Run {expected_identity[0]} / Event {expected_identity[1]} — "
            f"Muon-DIS candidate — track type {track['track_type']} index {track['track_index']}"
        )
        payload = build_sndsw_transform_payload(
            row, hit_details, converted, transformed_mufilter, transforms,
            mufilter_validation, track, geometry.half_width_cm,
        )
        comparison_renderer = Path(__file__).with_name(
            "render_sndsw_edge_transform_comparison.py"
        )
        with tempfile.TemporaryDirectory(prefix="sndsw_full_comparison_") as workdir:
            payload_path = str(Path(workdir) / "payload.json")
            full_pdf = str(Path(workdir) / "full_event_comparison.pdf")
            detail_pdf = str(Path(workdir) / "plane_comparisons.pdf")
            with open(payload_path, "w", encoding="utf-8") as stream:
                json.dump(_json_safe(payload), stream, allow_nan=False)
            completed = subprocess.run([
                sys.executable, str(comparison_renderer),
                "--display-script", display_script,
                "--input", row["event_source_path"],
                "--geo", geofile,
                "--event-index", str(event_index),
                "--payload", payload_path,
                "--output-pdf", full_pdf,
            ], check=False)
            if completed.returncode != 0:
                if not (
                    completed.returncode == -signal.SIGTERM
                    and os.path.isfile(full_pdf) and os.path.getsize(full_pdf) > 0
                ):
                    raise subprocess.CalledProcessError(
                        completed.returncode, completed.args
                    )
                print("[sndsw] SIGTERM after transformed comparison output; accepting PDF")
            with PdfPages(detail_pdf) as pdf:
                for orientation in ("horizontal", "vertical"):
                    draw_comparison_page(
                        pdf, orientation, planes, hits, converted, transforms, title,
                        float(config["display"].get("plane_page_width", 30.0)),
                    )
            merge_pages_to_pdf([full_pdf, detail_pdf], output_pdf)

        plane_rows = []
        plane_audits = []
        for key in sorted(planes):
            transform = transforms[key]
            source_hits = [hit for hit in hits if hit.key == key]
            target_hits = [hit for hit in converted if hit.source.key == key]
            locations = Counter(hit.location for hit in target_hits)
            residuals = [
                abs(
                    (item.transformed_coordinate_cm - transform.track_transformed_cm)
                    - (item.source.coordinate_cm - transform.track_original_cm)
                )
                for item in target_hits
            ]
            values = {
                "runId": expected_identity[0], "eventId": expected_identity[1],
                "source_file_index": row["source_file_index"],
                "source_event_index": event_index,
                "track_input_path": row["track_input_path"], "track_entry": track_entry,
                "track_index": track["track_index"], "track_type": track["track_type"],
                "track_chi2ndf": track["track_chi2ndf"],
                "station": key.station, "view": key.view,
                "orientation": key.orientation, "plane_z_cm": transform.z_cm,
                "plane_z_spread_cm": planes[key].z_spread_cm,
                "locked_side": transform.locked_side,
                "plane_local_closest_side": transform.plane_local_closest_side,
                "low_edge_cm": transform.low_edge_cm, "high_edge_cm": transform.high_edge_cm,
                "target_cm": transform.target_cm,
                "track_original_cm": transform.track_original_cm,
                "track_transformed_cm": transform.track_transformed_cm,
                "delta_cm": transform.delta_cm,
                "n_original_hits": len(source_hits),
                "n_valid_original_hits": sum(hit.valid for hit in source_hits),
                "n_transformed_hits": len(target_hits),
                "n_inside_active": locations["inside_active"],
                "n_inside_gap": locations["inside_gap"],
                "n_outside": locations["outside_geometry"],
                "max_residual_difference_cm": max(residuals, default=0.0),
                "validation_passed": True,
            }
            plane_rows.append(values)
            plane_audits.append(values)
        audit = {
            "event_sequence": int(row["event_sequence"]),
            "identity": {"runId": expected_identity[0], "eventId": expected_identity[1]},
            "raw": {
                "path": row["event_source_path"], "tree": tree_name, "entry": event_index,
            },
            "geometry": {
                "path": geofile, "template_validation": geometry.template_validation,
            },
            "track": serializable_track(track),
            "eligible_and_rejected_tracks": eligible_tracks,
            "hit_details": hit_details,
            "mufilter_hit_count": len(mufilter_hits),
            "mufilter_valid_hit_count": sum(item["valid"] for item in mufilter_hits),
            "mufilter_hits": mufilter_hits,
            "transformed_mufilter_hits": transformed_mufilter,
            "mufilter_transform_validation": mufilter_validation,
            "transform_validation": validation,
            "planes": plane_audits,
        }
        return audit, plane_rows
    finally:
        if track_source:
            track_source.Close()
        raw_source.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-pages", required=True)
    parser.add_argument("--output-audit-csv", required=True)
    parser.add_argument("--output-audit-json", required=True)
    parser.add_argument("--sndsw-display-script")
    parser.add_argument("--limit-events", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = read_csv(args.manifest)
    rows.sort(key=lambda row: int(row["event_sequence"]))
    if args.limit_events is not None:
        if args.limit_events < 1:
            raise ValueError("--limit-events must be positive")
        rows = rows[:args.limit_events]
    if not rows:
        raise RuntimeError("Selected event manifest is empty")
    for expected, row in enumerate(rows, start=1):
        if int(row["event_sequence"]) != expected:
            raise RuntimeError("Selected manifest event sequence is not contiguous")
        tree_name, entries = displayable_event_tree(
            row["event_source_path"],
            list(config["input"].get("event_tree_candidates", ["rawConv", "cbmsim"])),
        )
        if int(row["source_event_index"]) >= entries:
            raise IndexError(f"Raw entry is outside source tree: {row}")
        row["event_tree_name"] = tree_name

    output = Path(args.output_pdf).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_output = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_output)
    display = config["display"]
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script
        or str(display.get("sndsw_display_script", "") or "").strip()
        or None
    )
    extension = "eps" if display.get("output_mode", "vector_pdf") == "vector_pdf" else "png"
    pages, audits, plane_rows, page_rows = [], [], [], []
    geometry_cache = {}
    try:
        with tempfile.TemporaryDirectory(prefix="scifi_edge_transform_") as workdir:
            overview_by_sequence = {}
            for batch in source_batches(rows):
                rendered = run_sndsw_batch(
                    batch, display_script, workdir, extension,
                    [annotations(row) for row in batch],
                    int(display.get("canvas_width", 1600)),
                    int(display.get("canvas_height", 1000)),
                )
                if len(rendered) != len(batch):
                    raise RuntimeError(
                        f"SNDSW produced {len(rendered)} pages for {len(batch)} events"
                    )
                for row, page in zip(batch, rendered):
                    overview_pdf = str(
                        Path(workdir) / f"overview_{int(row['event_sequence']):04d}.pdf"
                    )
                    merge_pages_to_pdf([page], overview_pdf)
                    overview_by_sequence[int(row["event_sequence"])] = overview_pdf

            for row in rows:
                sequence = int(row["event_sequence"])
                comparison_pdf = str(Path(workdir) / f"comparison_{sequence:04d}.pdf")
                audit, event_plane_rows = process_event(
                    row, config, comparison_pdf, geometry_cache, display_script,
                )
                audits.append(audit)
                plane_rows.extend(event_plane_rows)
                pages.extend([overview_by_sequence[sequence], comparison_pdf])
                for offset, page_type in enumerate(PAGE_TYPES):
                    page_rows.append({
                        "event_sequence": sequence,
                        "document_page": len(PAGE_TYPES) * (sequence - 1) + offset + 1,
                        "page_type": page_type,
                        "runId": row["runId"], "eventId": row["eventId"],
                        "selection_reason": row.get("selection_reason", ""),
                        "track_type": row["track_type"], "track_index": row["track_index"],
                        "output_pdf": str(output), "status": "rendered",
                    })
            merge_pages_to_pdf(pages, temporary_output)
            expected_pages = len(PAGE_TYPES) * len(rows)
            observed_pages = pdf_page_count(temporary_output)
            if observed_pages != expected_pages:
                raise RuntimeError(
                    f"Merged PDF has {observed_pages} pages, expected {expected_pages}"
                )
        os.replace(temporary_output, output)
        atomic_csv(page_rows, args.output_pages)
        atomic_csv(plane_rows, args.output_audit_csv)
        atomic_json({
            "version": config["version"], "event_count": len(rows),
            "expected_pages": len(PAGE_TYPES) * len(rows), "output_pdf": str(output),
            "all_events_passed": all(
                event["transform_validation"]["passed"] for event in audits
            ),
            "events": audits,
        }, args.output_audit_json)
    finally:
        if os.path.exists(temporary_output):
            os.unlink(temporary_output)
    print(
        f"Rendered {len(rows)} events and {len(PAGE_TYPES) * len(rows)} "
        f"pages to {output}"
    )


if __name__ == "__main__":
    main()
