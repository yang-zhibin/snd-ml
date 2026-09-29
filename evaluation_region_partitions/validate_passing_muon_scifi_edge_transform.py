#!/usr/bin/env python3
"""Validate SciFi edge-transform output closure and publish a success marker."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

import yaml


def rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pages(path):
    with open(path, "rb") as stream:
        return len(re.findall(rb"/Type\s*/Page\b", stream.read()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--rendered-pages", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    selected = rows(args.selected)
    rendered = rows(args.rendered_pages)
    audit_rows = rows(args.audit_csv)
    with open(args.audit_json, encoding="utf-8") as stream:
        audit = json.load(stream)
    expected_events = int(config["selection"]["sample_size"])
    if len(selected) != expected_events:
        raise RuntimeError(f"Selected {len(selected)} events, expected {expected_events}")
    pages_per_event = len(config["display"]["page_types"])
    if len(rendered) != pages_per_event * expected_events or pages(args.pdf) != pages_per_event * expected_events:
        raise RuntimeError(
            f"Rendered page count does not close to {pages_per_event} pages per event"
        )
    if len(audit_rows) != 10 * expected_events:
        raise RuntimeError("Plane audit does not contain ten planes per event")
    if not audit.get("all_events_passed") or len(audit.get("events", [])) != expected_events:
        raise RuntimeError("JSON audit is incomplete or contains failed events")
    for event in audit["events"]:
        mufilter = event.get("mufilter_transform_validation", {})
        if not mufilter.get("passed"):
            raise RuntimeError(
                f"MuonFilter transform failed for event {event.get('identity')}"
            )
        original = int(mufilter.get("n_original_hits", -1))
        transformed = int(mufilter.get("n_transformed_hits", -2))
        classified = sum(int(value) for value in mufilter.get("locations", {}).values())
        if original != transformed or transformed != classified:
            raise RuntimeError(
                f"MuonFilter hit-count closure failed for event {event.get('identity')}"
            )
    for row in audit_rows:
        if row["validation_passed"].lower() not in ("true", "1"):
            raise RuntimeError(f"Failed plane audit row: {row}")
        original = int(row["n_original_hits"])
        transformed = int(row["n_transformed_hits"])
        classified = sum(int(row[key]) for key in (
            "n_inside_active", "n_inside_gap", "n_outside",
        ))
        if original != transformed or transformed != classified:
            raise RuntimeError(f"Hit-count closure failure: {row}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(
                f"version={config['version']}\nevents={expected_events}\n"
                f"pages={pages_per_event * expected_events}\nstatus=passed\n"
            )
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"SciFi edge-transform validation passed for {expected_events} events")


if __name__ == "__main__":
    main()
