#!/usr/bin/env python3
"""Merge V4 event-display summaries and enforce exact category page counts."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    target = int(config["sampling"]["events_per_category"])
    veto_categories = list(config["selection"]["categories"])
    expected_categories = {
        f"{population}__{category}"
        for population in ("passing", "missing")
        for category in veto_categories
    }
    rows, fields = [], None
    for path in args.inputs:
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fields:
                raise RuntimeError(f"Summary schema mismatch in {path}")
            rows.extend(reader)
    counts = Counter(row["category"] for row in rows)
    bad = {
        category: counts.get(category, 0)
        for category in expected_categories
        if counts.get(category, 0) != target
    }
    unexpected = set(counts) - expected_categories
    if bad or unexpected:
        raise RuntimeError(
            f"Expected {len(expected_categories)} categories with {target} pages each; "
            f"counts={dict(counts)}, unexpected={sorted(unexpected)}"
        )
    identities = [
        (row["source_file_index"], row["runId"], row["eventId"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Rendered V4 summaries contain duplicate event identities")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Merged {len(rows)} rendered V4 event rows into {output}")


if __name__ == "__main__":
    main()
