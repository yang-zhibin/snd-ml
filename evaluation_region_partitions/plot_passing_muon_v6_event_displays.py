#!/usr/bin/env python3
"""Render one V6 loose-selection event-display category from its manifest."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from itertools import groupby
from pathlib import Path

import yaml
import ROOT

from sndsw_event_display_common import (
    displayable_event_tree,
    find_sndsw_2d_display_script,
    merge_pages_to_pdf,
    run_sndsw_batch,
)


SUMMARY_FIELDS = [
    "category", "page", "sample", "population", "veto_category",
    "production_key", "source_file_index", "source_feature_path", "feature_entry",
    "event_source_path", "event_tree_name", "source_event_index", "geo_path",
    "runId", "eventId", "failed_components", "selection_expression",
    "output_pdf", "status", "message",
]


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def read_rows(path: str, category: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["category"] == category]


def compact_counts(row: dict, prefix: str, stations: int) -> str:
    return ",".join(str(row[f"count_{prefix}{station}"]) for station in range(1, stations + 1))


def total_count(row: dict, prefix: str, stations: int) -> int:
    return sum(int(row[f"count_{prefix}{station}"]) for station in range(1, stations + 1))


def annotation_lines(row: dict, max_lines: int) -> list[str]:
    sample = row["sample_label"]
    if row.get("production_key"):
        sample += f" ({row['production_key']})"
    state = {
        "candidate": "PASS",
        "failure": "FAIL",
        "inclusive": "not used for category membership",
    }.get(row["population"], str(row["population"]))
    lines = [
        f"Sample: {sample}",
        f"Category: {row['veto_category_label']}",
        f"Loose selection: {state}",
        f"Veto total/stations: {row['count_veto']} / {compact_counts(row, 'veto', 3)}",
        "SciFi total/stations: "
        f"{total_count(row, 'scifi', 5)} / {compact_counts(row, 'scifi', 5)}",
        f"US stations: {compact_counts(row, 'us', 5)}",
        f"DS stations: {compact_counts(row, 'ds', 4)}",
    ]
    if row.get("failed_components"):
        lines.append("Failed: " + row["failed_components"].replace(";", ", "))
    lines.append(f"Source index / entry: {row['source_file_index']} / {row['source_event_index']}")
    return lines[:max_lines]


def source_batches(rows: list[dict]):
    # Preserve manifest round-robin order. Only adjacent events from the same
    # source are batched, so the first ten rendered MC pages remain diverse.
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
            writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_no_events_outputs(
    category: str, sample: str, output_pdf: str, output_summary: str,
    message: str,
):
    """Publish an auditable one-page PDF when a category has no events."""
    output = Path(output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_pdf = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_pdf)
    try:
        canvas = ROOT.TCanvas("no_events", "no events", 1200, 800)
        canvas.SetMargin(0.08, 0.08, 0.08, 0.08)
        text = ROOT.TLatex()
        text.SetTextAlign(22)
        text.SetTextFont(42)
        text.SetTextSize(0.055)
        text.DrawLatexNDC(0.5, 0.60, "No events found")
        text.SetTextSize(0.032)
        label = category.replace("__", " / ").replace("_", " ")
        text.DrawLatexNDC(0.5, 0.49, label)
        text.SetTextSize(0.026)
        text.DrawLatexNDC(0.5, 0.39, message)
        canvas.Print(temporary_pdf)
        os.replace(temporary_pdf, output)
    finally:
        if os.path.exists(temporary_pdf):
            os.unlink(temporary_pdf)
    atomic_summary([{
        "category": category,
        "page": 1,
        "sample": sample,
        "population": "candidate" if category.endswith("__candidate") else "failure",
        "output_pdf": str(output),
        "status": "no_events",
        "message": message,
    }], output_summary)


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
        print(
            f"[warn] Manifest has {len(rows)} rows for {args.category}; "
            f"requested target={target}"
        )
    candidates = list(config["input"].get("event_tree_candidates", ["cbmsim", "rawConv"]))
    display_cfg = config.get("display", {})
    output_mode = str(display_cfg.get("output_mode", "vector_pdf"))
    extension = "eps" if output_mode == "vector_pdf" else "png"
    max_lines = int(config.get("annotations", {}).get("max_lines", 10))

    # Validate and cache source-tree resolution before invoking SNDSW.
    tree_cache = {}
    valid_rows = []
    for row in rows:
        source = row["event_source_path"]
        try:
            if source not in tree_cache:
                tree_cache[source] = displayable_event_tree(source, candidates)
            tree_name, entries = tree_cache[source]
            index = int(row["source_event_index"])
            if index < 0 or index >= entries:
                raise IndexError(f"entry {index} outside 0..{entries - 1}")
            row["event_tree_name"] = tree_name
            valid_rows.append(row)
        except Exception as error:
            print(f"[skip] {args.category} source validation failed: {error}")
    if len(valid_rows) < target:
        print(
            f"[warn] Only {len(valid_rows)} displayable rows remain for "
            f"{args.category}; requested target={target}"
        )
    if not valid_rows:
        write_no_events_outputs(
            args.category,
            str(config["samples"][0]),
            args.output_pdf,
            args.output_summary,
            "No selected event is displayable for this category.",
        )
        print(f"Wrote no-event placeholder to {args.output_pdf}")
        return
    render_target = min(target, len(valid_rows))
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script
        or str(display_cfg.get("sndsw_display_script", "") or "").strip()
        or None
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
        with tempfile.TemporaryDirectory(prefix=f"v6_display_{args.category}_") as workdir:
            for batch in source_batches(valid_rows):
                remaining = render_target - len(rendered_rows)
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
                    print(f"[skip] SNDSW batch failed for {batch[0]['event_source_path']}: {error}")
                    continue
                accepted = min(len(new_pages), len(batch), remaining)
                pages.extend(new_pages[:accepted])
                rendered_rows.extend(batch[:accepted])
            if len(rendered_rows) < render_target:
                print(
                    f"[warn] Rendered {len(rendered_rows)} pages for "
                    f"{args.category}; available target={render_target}"
                )
            if rendered_rows:
                merge_pages_to_pdf(pages, temporary_pdf)
        if rendered_rows:
            os.replace(temporary_pdf, output)
    finally:
        if os.path.exists(temporary_pdf):
            os.unlink(temporary_pdf)

    if not rendered_rows:
        write_no_events_outputs(
            args.category,
            str(config["samples"][0]),
            args.output_pdf,
            args.output_summary,
            "Selected events could not be rendered by SNDSW.",
        )
        print(f"Wrote no-event placeholder to {args.output_pdf}")
        return

    summary = []
    for page, row in enumerate(rendered_rows, start=1):
        item = dict(row)
        item.update({
            "page": page,
            "output_pdf": str(output),
            "status": "rendered",
            "message": "",
        })
        summary.append(item)
    atomic_summary(summary, args.output_summary)
    print(f"Wrote {len(summary)} pages to {output}")


if __name__ == "__main__":
    main()
