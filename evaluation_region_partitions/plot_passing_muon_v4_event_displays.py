#!/usr/bin/env python3
"""Render one V4 veto-time event-display category from its manifest."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from itertools import groupby
from pathlib import Path

import yaml

from sndsw_event_display_common import (
    find_sndsw_2d_display_script,
    merge_pages_to_pdf,
    run_sndsw_batch,
)


SUMMARY_FIELDS = [
    "category", "page", "population", "veto_time_category", "partition_id",
    "partition_path", "partition_entry", "source_file_index",
    "event_source_path", "event_tree_name", "source_event_index",
    "raw_event_index", "feature_event_index", "identity_resolution",
    "raw_identity_verified", "geo_path", "runId", "eventId", "eventIndex",
    "original_entry", "count_veto",
    "hitTime_veto_min_ns", "selection_expression", "output_pdf",
]


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def read_rows(path: str, category: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["category"] == category]


def event_tree_identity(
    path: str, candidates: list[str], entry: int,
) -> tuple[str, int, int, int]:
    """Read one direct raw-tree entry and return its stable identity."""
    import ROOT

    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open event-display source {path}")
    try:
        tree_name, tree = None, None
        for candidate in candidates:
            tree = source.Get(candidate)
            if tree:
                tree_name = candidate
                break
        if not tree:
            raise RuntimeError(f"No event tree {candidates} in {path}")
        if not (tree.GetBranch("EventHeader") or tree.GetBranch("EventHeader.")):
            raise RuntimeError(f"Event tree {tree_name} has no EventHeader in {path}")
        entries = int(tree.GetEntries())
        if entry < 0 or entry >= entries:
            raise IndexError(f"entry {entry} outside 0..{entries - 1}")
        if tree.GetEntry(int(entry)) <= 0:
            raise RuntimeError(f"Could not read entry {entry} from {path}")
        header = tree.EventHeader
        run_id = int(header.GetRunId())
        if hasattr(header, "GetEventNumber"):
            event_id = int(header.GetEventNumber())
        elif hasattr(header, "GetMCEntryNumber"):
            event_id = int(header.GetMCEntryNumber())
        else:
            raise RuntimeError(f"EventHeader has no event-number accessor in {path}")
        return str(tree_name), entries, run_id, event_id
    finally:
        source.Close()


def compact_counts(row: dict, prefix: str, stations: int) -> str:
    return ",".join(
        str(row[f"count_{prefix}{station}"])
        for station in range(1, stations + 1)
    )


def format_time(value: object) -> str:
    number = float(value)
    return f"{number:.3f} ns" if math.isfinite(number) else str(number)


def annotation_lines(row: dict, max_lines: int) -> list[str]:
    lines = [
        "Sample: Run 8329 RealDataDQ",
        f"Population: {row['population_label']}",
        f"Veto-time region: {row['veto_time_label']}",
        f"Earliest veto time: {format_time(row['hitTime_veto_min_ns'])}",
        f"Veto total/stations: {row['count_veto']} / {compact_counts(row, 'veto', 3)}",
        f"SciFi stations: {compact_counts(row, 'scifi', 5)}",
        f"US stations: {compact_counts(row, 'us', 5)}",
        f"DS stations: {compact_counts(row, 'ds', 4)}",
        f"Source index / entry: {row['source_file_index']} / {row['source_event_index']}",
    ]
    return lines[:max_lines]


def source_batches(rows: list[dict]):
    # Preserve manifest order. Adjacent events with a common source/geofile can
    # share one SNDSW process without changing cross-source round-robin order.
    key = lambda row: (row["event_source_path"], row["geo_path"])
    return (list(group) for _, group in groupby(rows, key=key))


def atomic_summary(rows: list[dict], output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in SUMMARY_FIELDS}
                for row in rows
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--sndsw-display-script")
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = read_rows(args.manifest, args.category)
    target = int(config["sampling"]["events_per_category"])
    if len(rows) < target:
        raise RuntimeError(
            f"Manifest has {len(rows)} rows for {args.category}, needs {target}"
        )
    candidates = list(config["input"].get("event_tree_candidates", ["cbmsim", "rawConv"]))
    display_cfg = config.get("display", {})
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script
        or str(display_cfg.get("sndsw_display_script", "") or "").strip()
        or None
    )
    output_mode = str(display_cfg.get("output_mode", "vector_pdf"))
    extension = "eps" if output_mode == "vector_pdf" else "png"
    max_lines = int(config.get("annotations", {}).get("max_lines", 10))

    identity_cache = {}
    valid_rows = []
    for row in rows:
        source = row["event_source_path"]
        try:
            index = int(row["source_event_index"])
            cache_key = (source, index)
            if cache_key not in identity_cache:
                identity_cache[cache_key] = event_tree_identity(
                    source, candidates, index
                )
            tree_name, _, observed_run, observed_event = identity_cache[cache_key]
            expected_run = int(row["runId"])
            expected_event = int(row["eventId"])
            if (observed_run, observed_event) != (expected_run, expected_event):
                raise RuntimeError(
                    "raw identity mismatch at entry "
                    f"{index}: observed=({observed_run}, {observed_event}), "
                    f"expected=({expected_run}, {expected_event})"
                )
            row["event_tree_name"] = tree_name
            row["raw_identity_verified"] = "1"
            valid_rows.append(row)
        except Exception as error:
            print(f"[skip] {args.category} source validation failed: {error}")
    if len(valid_rows) < target:
        raise RuntimeError(
            f"Only {len(valid_rows)} displayable rows remain for "
            f"{args.category}; target={target}"
        )

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_pdf = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_pdf)
    rendered_rows, pages = [], []
    try:
        with tempfile.TemporaryDirectory(prefix=f"v4_display_{args.category}_") as workdir:
            for batch in source_batches(valid_rows):
                remaining = target - len(rendered_rows)
                if remaining <= 0:
                    break
                batch = batch[:remaining]
                try:
                    new_pages = run_sndsw_batch(
                        batch, display_script, workdir, extension,
                        [annotation_lines(row, max_lines) for row in batch],
                        int(display_cfg.get("canvas_width", 1600)),
                        int(display_cfg.get("canvas_height", 1000)),
                    )
                except Exception as error:
                    print(
                        f"[skip] SNDSW batch failed for "
                        f"{batch[0]['event_source_path']}: {error}"
                    )
                    continue
                accepted = min(len(new_pages), len(batch), remaining)
                pages.extend(new_pages[:accepted])
                rendered_rows.extend(batch[:accepted])
            if len(rendered_rows) != target:
                raise RuntimeError(
                    f"Rendered {len(rendered_rows)} pages for "
                    f"{args.category}; target={target}"
                )
            merge_pages_to_pdf(pages, temporary_pdf)
        os.replace(temporary_pdf, output)
    finally:
        if os.path.exists(temporary_pdf):
            os.unlink(temporary_pdf)

    summary = []
    for page, row in enumerate(rendered_rows, start=1):
        item = dict(row)
        item.update({"page": page, "output_pdf": str(output)})
        summary.append(item)
    atomic_summary(summary, args.output_summary)
    print(f"Wrote {len(summary)} pages to {output}")


if __name__ == "__main__":
    main()
