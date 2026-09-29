#!/usr/bin/env python3
"""Merge V6 category summaries and enforce the configured page count."""

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
    veto_categories = list(config["selection"]["failure_veto_categories"])
    expected_categories = {
        f"{sample}__candidate"
        for sample in ("run8329", "mc_muon_up")
    } | {
        f"{sample}__failure__{veto}"
        for sample in ("run8329", "mc_muon_up")
        for veto in veto_categories
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
    bad = {category: counts.get(category, 0) for category in expected_categories
           if counts.get(category, 0) != target}
    unexpected = set(counts) - expected_categories
    if bad or unexpected:
        raise RuntimeError(
            f"Expected {len(expected_categories)} categories with {target} pages each; "
            f"counts={dict(counts)}, unexpected={sorted(unexpected)}"
        )
    identities = [
        (row["sample"], row["source_feature_path"], row["source_event_index"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Rendered summaries contain duplicate event identities")
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
    print(f"Merged {len(rows)} rendered event rows into {output}")


if __name__ == "__main__":
    main()
