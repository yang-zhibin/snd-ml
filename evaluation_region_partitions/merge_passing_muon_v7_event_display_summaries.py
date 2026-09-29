#!/usr/bin/env python3
"""Merge V7 data or MC event-display summaries and enforce exact closure."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from passing_muon_v7_event_display_config import category_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {args.config}")
    target = int(config["sampling"]["events_per_category"])
    categories = category_names(config)
    expected = set(categories)

    rows = []
    fields = None
    for path in args.inputs:
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            current_fields = list(reader.fieldnames or [])
            if fields is None:
                fields = current_fields
            elif current_fields != fields:
                raise RuntimeError(f"Summary schema mismatch in {path}")
            rows.extend(reader)
    if not fields:
        raise RuntimeError("V7 event-display summaries have no CSV schema")

    counts = Counter(row["category"] for row in rows)
    bad = {
        category: counts.get(category, 0)
        for category in categories
        if not 1 <= counts.get(category, 0) <= target
    }
    unexpected = set(counts) - expected
    if bad or unexpected:
        raise RuntimeError(
            f"Expected {len(categories)} categories with 1..{target} pages each; "
            f"counts={dict(counts)}, unexpected={sorted(unexpected)}"
        )

    identities = [
        (
            row["category"], row["sample"], row["source_feature_path"],
            row["source_event_index"],
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError(
            "Rendered V7 summaries contain duplicate identities within a category"
        )
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        statuses = {row.get("status", "") or "rendered" for row in category_rows}
        if "no_events" in statuses and (
            statuses != {"no_events"} or len(category_rows) != 1
        ):
            raise RuntimeError(
                f"Category {category} mixes a no-event placeholder with events"
            )
        pages = sorted(int(row["page"]) for row in category_rows)
        if pages != list(range(1, len(category_rows) + 1)):
            raise RuntimeError(
                f"Category {category} has invalid page sequence: {pages}"
            )

    category_order = {category: index for index, category in enumerate(categories)}
    rows.sort(key=lambda row: (category_order[row["category"]], int(row["page"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
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
    print(
        f"Merged {len(rows)} rendered or placeholder pages across "
        f"{len(categories)} V7 categories into {output}"
    )


if __name__ == "__main__":
    main()
