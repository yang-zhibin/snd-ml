#!/usr/bin/env python3
"""Regenerate one frozen V7 category with SNDSW + SciFi plane pages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from plot_passing_muon_v6_event_displays import annotation_lines, source_batches
from scifi_sipm_event_display import (
    create_sipm_geometry_context,
    render_plane_pages,
)
from sndsw_event_display_common import (
    displayable_event_tree,
    find_sndsw_2d_display_script,
    merge_pages_to_pdf,
    run_sndsw_batch,
)


PAGE_TYPES = ("overview", "scifi_horizontal", "scifi_vertical")
SUMMARY_FIELDS = [
    "category", "event_sequence", "document_page", "page_type", "sample",
    "population", "veto_category", "source_file_index", "source_feature_path",
    "feature_entry", "event_source_path", "event_tree_name",
    "source_event_index", "geo_path", "runId", "eventId",
    "total_recorded_scifi_hits", "total_valid_scifi_hits",
    "active_horizontal_stations", "active_vertical_stations",
    "coordinate_validation_passed", "unmapped_scifi_hits", "output_pdf",
    "status", "message",
]


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def identity(row: dict) -> tuple[str, str, str, int]:
    return (
        row["category"], row["sample"], row["source_feature_path"],
        int(row["source_event_index"]),
    )


def atomic_csv(rows: list[dict], output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: dict, output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def enrich_source_rows(
    source_rows: list[dict], selected: list[dict], category: str,
    selection_category_aliases: dict[str, str] | None = None,
) -> list[dict]:
    selected_by_identity = {}
    for row in selected:
        key = identity(row)
        if key in selected_by_identity:
            raise RuntimeError(f"Duplicate selection-manifest identity: {key}")
        selected_by_identity[key] = row
    category_rows = [row for row in source_rows if row["category"] == category]
    category_rows.sort(key=lambda row: int(row["page"]))
    pages = [int(row["page"]) for row in category_rows]
    if pages != list(range(1, len(category_rows) + 1)):
        raise RuntimeError(f"Source category {category} has invalid event-page sequence: {pages}")
    output = []
    selection_category = (selection_category_aliases or {}).get(category, category)
    for source_row in category_rows:
        key = (
            selection_category, source_row["sample"],
            source_row["source_feature_path"], int(source_row["source_event_index"]),
        )
        selected_row = selected_by_identity.get(key)
        if selected_row is None:
            raise RuntimeError(
                f"Source event is missing from selection manifest: source_category={category}, "
                f"selection_category={selection_category}, identity={key[1:]}"
            )
        merged = dict(selected_row)
        merged.update(source_row)
        output.append(merged)
    return output


def rows_from_selection_manifest(
    selected: list[dict], category: str, target: int,
    tree_candidates: list[str],
) -> list[dict]:
    """Freeze the first displayable target events from a reserve manifest."""
    tree_cache = {}
    rows = []
    seen = set()
    for selected_row in selected:
        if selected_row["category"] != category:
            continue
        key = identity(selected_row)
        if key in seen:
            raise RuntimeError(f"Duplicate selection-manifest identity: {key}")
        seen.add(key)
        source = selected_row["event_source_path"]
        try:
            if source not in tree_cache:
                tree_cache[source] = displayable_event_tree(source, tree_candidates)
            tree_name, entries = tree_cache[source]
            source_index = int(selected_row["source_event_index"])
            if source_index < 0 or source_index >= entries:
                raise IndexError(
                    f"entry {source_index} outside 0..{entries - 1}"
                )
        except Exception as error:
            print(f"[skip] {category} source validation failed: {error}")
            continue
        row = dict(selected_row)
        row["event_tree_name"] = tree_name
        row["page"] = len(rows) + 1
        rows.append(row)
        if len(rows) >= target:
            break
    return rows


def all_coordinate_checks_pass(audit: dict) -> bool:
    checks = audit["global_coordinate_validation"]
    return all(
        value["passed"]
        for orientation in checks.values()
        for value in orientation.values()
    )


def pdf_page_count(path: str) -> int:
    # pdfunite writes each page dictionary as an ordinary PDF object.  Count
    # those dictionaries directly so batch workers do not require the optional
    # pdfinfo executable.  The word boundary excludes the /Type /Pages tree.
    with open(path, "rb") as stream:
        count = len(re.findall(rb"/Type\s*/Page\b", stream.read()))
    if count < 1:
        raise RuntimeError(f"No PDF page objects found in {path}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Existing per-category summary manifest")
    parser.add_argument("--selection-manifest", required=True, help="Full selected manifest for annotations")
    parser.add_argument(
        "--direct-selection-manifest",
        action="store_true",
        help="Validate and freeze events directly from --selection-manifest",
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--sndsw-display-script")
    parser.add_argument("--limit-events", type=int, help="Test-only prefix of the source category")
    args = parser.parse_args()

    config = load_yaml(args.config)
    categories = list(config["frozen_input"]["categories"])
    if args.category not in categories:
        raise ValueError(f"Category {args.category!r} is not in the configured category list")
    expected = int(config["frozen_input"]["events_per_category"])
    allow_fewer = bool(config["frozen_input"].get("allow_fewer_events", False))
    selected = read_csv(args.selection_manifest)
    if args.direct_selection_manifest:
        rows = rows_from_selection_manifest(
            selected,
            args.category,
            expected,
            list(config["input"].get("event_tree_candidates", ["cbmsim", "rawConv"])),
        )
    else:
        if not args.manifest:
            parser.error("--manifest is required unless --direct-selection-manifest is used")
        rows = enrich_source_rows(
            read_csv(args.manifest), selected, args.category,
            config["frozen_input"].get("selection_category_aliases", {}),
        )
    if not rows:
        raise RuntimeError(f"Source category {args.category} has no displayable events")
    if len(rows) != expected and not allow_fewer:
        raise RuntimeError(
            f"Source category {args.category} has {len(rows)} events, expected {expected}"
        )
    if len(rows) < expected:
        print(
            f"[warn] Source category {args.category} has {len(rows)} displayable "
            f"events; requested target={expected}"
        )
    if args.limit_events is not None:
        if args.limit_events < 1:
            raise ValueError("--limit-events must be positive")
        rows = rows[:args.limit_events]

    display = config["display"]
    extension = "eps" if display.get("output_mode", "vector_pdf") == "vector_pdf" else "png"
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script
        or str(display.get("sndsw_display_script", "") or "").strip()
        or None
    )
    page_width = float(display.get("plane_page_width", 66.0))
    max_lines = int(config.get("annotations", {}).get("max_lines", 10))
    output = Path(args.output_pdf).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_output = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_output)

    summaries = []
    event_audits = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"v7_scifi_{args.category}_") as workdir:
            overview_by_event = {}
            for batch in source_batches(rows):
                pages = run_sndsw_batch(
                    batch, display_script, workdir, extension,
                    [annotation_lines(row, max_lines) for row in batch],
                    int(display.get("canvas_width", 1600)),
                    int(display.get("canvas_height", 1000)),
                )
                if len(pages) != len(batch):
                    raise RuntimeError(
                        f"SNDSW returned {len(pages)} pages for a {len(batch)}-event batch"
                    )
                for row, page in zip(batch, pages):
                    page_pdf = str(Path(workdir) / f"overview_{int(row['page']):04d}.pdf")
                    merge_pages_to_pdf([page], page_pdf)
                    overview_by_event[int(row["page"])] = page_pdf

            document_parts = []
            context = None
            for event_sequence, row in enumerate(rows, start=1):
                geofile = row["geo_path"]
                if context is None or context.geofile != geofile:
                    context = create_sipm_geometry_context(geofile)
                plane_pdf = str(Path(workdir) / f"planes_{event_sequence:04d}.pdf")
                title = (
                    f"Run {row['runId']} / Event {row['eventId']} — "
                    f"{args.category.replace('__', ' / ')}"
                )
                audit = render_plane_pages(
                    row["event_source_path"], row["event_tree_name"],
                    int(row["source_event_index"]), geofile, plane_pdf, title,
                    plane_page_width=page_width, geometry_context=context,
                )
                expected_identity = (int(row["runId"]), int(row["eventId"]))
                observed_identity = (
                    int(audit["event"].get("run_id", -1)),
                    int(audit["event"].get("event_id", -1)),
                )
                if observed_identity != expected_identity:
                    raise RuntimeError(
                        f"Event identity mismatch for sequence {event_sequence}: "
                        f"manifest={expected_identity}, tree={observed_identity}"
                    )
                if audit["unmapped_hits"]:
                    raise RuntimeError(
                        f"Event sequence {event_sequence} has {len(audit['unmapped_hits'])} unmapped SciFi hits"
                    )
                if not all_coordinate_checks_pass(audit):
                    raise RuntimeError(f"Event sequence {event_sequence} failed coordinate validation")
                document_parts.extend([overview_by_event[event_sequence], plane_pdf])
                event_audits.append({"event_sequence": event_sequence, **audit})

                common = {
                    "category": args.category,
                    "event_sequence": event_sequence,
                    "sample": row["sample"],
                    "population": row["population"],
                    "veto_category": row["veto_category"],
                    "source_file_index": row["source_file_index"],
                    "source_feature_path": row["source_feature_path"],
                    "feature_entry": row["feature_entry"],
                    "event_source_path": row["event_source_path"],
                    "event_tree_name": row["event_tree_name"],
                    "source_event_index": row["source_event_index"],
                    "geo_path": geofile,
                    "runId": row["runId"],
                    "eventId": row["eventId"],
                    "total_recorded_scifi_hits": audit["total_recorded_scifi_hits"],
                    "total_valid_scifi_hits": audit["total_valid_scifi_hits"],
                    "active_horizontal_stations": json.dumps(
                        audit["active_stations_by_orientation"]["horizontal"]
                    ),
                    "active_vertical_stations": json.dumps(
                        audit["active_stations_by_orientation"]["vertical"]
                    ),
                    "coordinate_validation_passed": True,
                    "unmapped_scifi_hits": 0,
                    "output_pdf": str(output),
                    "status": "rendered",
                    "message": "",
                }
                for page_offset, page_type in enumerate(PAGE_TYPES):
                    summaries.append({
                        **common,
                        "document_page": 3 * (event_sequence - 1) + page_offset + 1,
                        "page_type": page_type,
                    })
            merge_pages_to_pdf(document_parts, temporary_output)
            expected_document_pages = 3 * len(rows)
            actual_document_pages = pdf_page_count(temporary_output)
            if actual_document_pages != expected_document_pages:
                raise RuntimeError(
                    f"Merged PDF has {actual_document_pages} pages, "
                    f"expected {expected_document_pages}"
                )
        os.replace(temporary_output, output)
    finally:
        if os.path.exists(temporary_output):
            os.unlink(temporary_output)

    expected_pages = 3 * len(rows)
    if len(summaries) != expected_pages:
        raise RuntimeError(f"Built {len(summaries)} summary rows, expected {expected_pages}")
    atomic_csv(summaries, args.output_summary)
    atomic_json({
        "category": args.category,
        "event_count": len(rows),
        "page_count": expected_pages,
        "page_types_per_event": list(PAGE_TYPES),
        "plane_page_width_inches": page_width,
        "limited_test_run": args.limit_events is not None,
        "direct_selection_manifest": args.direct_selection_manifest,
        "events": event_audits,
    }, args.output_audit)
    print(f"Wrote {expected_pages} pages for {len(rows)} events to {output}")


if __name__ == "__main__":
    main()
