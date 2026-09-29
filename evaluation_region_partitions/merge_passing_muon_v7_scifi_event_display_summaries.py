#!/usr/bin/env python3
"""Merge V7 SciFi display summaries and enforce three-page event closure."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml


# Kept local so the lightweight closure job does not import ROOT/SNDSW.
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


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != SUMMARY_FIELDS:
            raise RuntimeError(f"Unexpected summary schema in {path}")
        return list(reader)


def frozen_identity(row: dict) -> tuple[str, str, int]:
    return row["sample"], row["source_feature_path"], int(row["source_event_index"])


def validate_event_rows(category: str, event_sequence: int, event_rows: list[dict]):
    if [row["page_type"] for row in event_rows] != list(PAGE_TYPES):
        raise RuntimeError(
            f"Category {category} event {event_sequence} does not have "
            f"{list(PAGE_TYPES)} in order"
        )
    if any(
        row["coordinate_validation_passed"].lower() != "true"
        for row in event_rows
    ):
        raise RuntimeError(
            f"Category {category} event {event_sequence} failed coordinate validation"
        )
    if any(int(row["unmapped_scifi_hits"]) != 0 for row in event_rows):
        raise RuntimeError(
            f"Category {category} event {event_sequence} has unmapped SciFi hits"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    categories = list(config["frozen_input"]["categories"])
    expected_events = int(config["frozen_input"]["events_per_category"])
    allow_fewer = bool(config["frozen_input"].get("allow_fewer_events", False))
    manifest_mode = str(config["frozen_input"].get("manifest_mode", "frozen_summary"))
    if manifest_mode not in {"frozen_summary", "selection"}:
        raise ValueError(f"Unknown frozen_input.manifest_mode {manifest_mode!r}")
    frozen_rows = []
    for path in args.manifests:
        with open(path, newline="", encoding="utf-8") as stream:
            frozen_rows.extend(csv.DictReader(stream))
    frozen_by_category = defaultdict(list)
    for row in frozen_rows:
        if row["category"] in categories:
            frozen_by_category[row["category"]].append(row)

    rows = []
    for path in args.summaries:
        rows.extend(read_csv(path))
    audit_categories = set()
    event_counts = {}
    for path in args.audits:
        with open(path, encoding="utf-8") as stream:
            audit = json.load(stream)
        category = audit["category"]
        if category in audit_categories:
            raise RuntimeError(f"Duplicate category audit {category}")
        audit_categories.add(category)
        if audit.get("limited_test_run"):
            raise RuntimeError(f"Category {category} audit is marked as a limited test run")
        event_count = int(audit["event_count"])
        if event_count < 1 or event_count > expected_events:
            raise RuntimeError(f"Category {category} audit has the wrong event count")
        if event_count != expected_events and not allow_fewer:
            raise RuntimeError(f"Category {category} audit has the wrong event count")
        event_counts[category] = event_count
        if int(audit["page_count"]) != 3 * event_count:
            raise RuntimeError(f"Category {category} audit has the wrong page count")
        if list(audit["page_types_per_event"]) != list(PAGE_TYPES):
            raise RuntimeError(f"Category {category} audit has the wrong page types")
    if audit_categories != set(categories):
        raise RuntimeError(
            f"Audit categories do not close: expected {categories}, found {sorted(audit_categories)}"
        )
    by_category = defaultdict(list)
    for row in rows:
        if row["category"] not in categories:
            raise RuntimeError(f"Unexpected category {row['category']!r}")
        by_category[row["category"]].append(row)

    for category in categories:
        category_rows = sorted(by_category[category], key=lambda row: int(row["document_page"]))
        actual_events = event_counts[category]
        expected_pages = 3 * actual_events
        if len(category_rows) != expected_pages:
            raise RuntimeError(
                f"Category {category} has {len(category_rows)} pages, expected {expected_pages}"
            )
        pages = [int(row["document_page"]) for row in category_rows]
        if pages != list(range(1, expected_pages + 1)):
            raise RuntimeError(f"Category {category} has an invalid document-page sequence")
        frozen = frozen_by_category[category]
        if manifest_mode == "frozen_summary":
            frozen = sorted(frozen, key=lambda row: int(row["page"]))
            if len(frozen) != actual_events:
                raise RuntimeError(
                    f"Source summaries contain {len(frozen)} events for {category}, "
                    f"expected {actual_events}"
                )
        else:
            frozen_identities = {frozen_identity(row) for row in frozen}
            if len(frozen_identities) != len(frozen):
                raise RuntimeError(
                    f"Selection manifest contains duplicate identities for {category}"
                )
        for event_sequence in range(1, actual_events + 1):
            event_rows = [
                row for row in category_rows
                if int(row["event_sequence"]) == event_sequence
            ]
            validate_event_rows(category, event_sequence, event_rows)
            identities = {frozen_identity(row) for row in event_rows}
            if manifest_mode == "frozen_summary":
                expected_identity = frozen_identity(frozen[event_sequence - 1])
                if identities != {expected_identity}:
                    raise RuntimeError(
                        f"Category {category} event {event_sequence} identity mismatch: "
                        f"expected {expected_identity}, found {identities}"
                    )
            else:
                if len(identities) != 1 or not identities.issubset(frozen_identities):
                    raise RuntimeError(
                        f"Category {category} event {event_sequence} is not in the "
                        "selection manifest"
                    )

    identities = [
        (row["category"], row["event_sequence"], row["page_type"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Merged summaries contain duplicate category/event/page-type identities")

    category_order = {category: index for index, category in enumerate(categories)}
    rows.sort(key=lambda row: (category_order[row["category"]], int(row["document_page"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"Merged {len(rows)} pages for {sum(event_counts.values())} events "
        f"across {len(categories)} source categories into {output}"
    )


if __name__ == "__main__":
    main()
