#!/usr/bin/env python3
"""Validate and materialize the frozen sideband-only density merge policy."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_sideband_counts(
    path: str | Path, config: dict
) -> tuple[list[float], dict[str, str], dict[str, list[int]]]:
    source = config["source"]
    selected = {
        str(item["region"]): str(item["label"])
        for item in config["selection"]["regions"]
    }
    excluded = {str(region) for region in config["selection"]["excluded_regions"]}
    if not selected or set(selected) & excluded:
        raise ValueError("Selected sidebands must be nonempty and disjoint from exclusions")

    n_bins = int(source["regular_bins"])
    expected_indices = set(range(1, n_bins + 1))
    values: dict[str, dict[int, int]] = {region: {} for region in selected}
    lows: dict[int, float] = {}
    highs: dict[int, float] = {}
    table_versions: set[str] = set()
    binning_versions: set[str] = set()

    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "table_version",
            "binning_version",
            "region",
            "row_type",
            "root_bin_index",
            "bin_low",
            "bin_high",
            "data_raw_events",
            "source_validation_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Input table is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            region = row["region"]
            if region not in selected or row["row_type"] != "bin":
                continue
            if row["source_validation_status"] != "PASS":
                raise ValueError(f"Non-PASS source row for {region}")
            index = int(row["root_bin_index"])
            count_float = float(row["data_raw_events"])
            count = int(round(count_float))
            if count < 0 or not math.isclose(count_float, count, abs_tol=1e-9):
                raise ValueError(f"Invalid raw count for {region}, bin {index}")
            if index in values[region]:
                raise ValueError(f"Duplicate regular bin {index} for {region}")
            values[region][index] = count
            lows.setdefault(index, float(row["bin_low"]))
            highs.setdefault(index, float(row["bin_high"]))
            table_versions.add(row["table_version"])
            binning_versions.add(row["binning_version"])

    if table_versions != {str(source["table_version"])}:
        raise ValueError(f"Unexpected selected-row table versions: {table_versions}")
    if binning_versions != {str(source["binning_version"])}:
        raise ValueError(f"Unexpected selected-row binning versions: {binning_versions}")
    for region, region_values in values.items():
        if set(region_values) != expected_indices:
            raise ValueError(f"Expected {n_bins} regular bins for {region}")

    base_edges = [lows[1]] + [highs[index] for index in range(1, n_bins + 1)]
    expected_edges = [float(source["min"]) + i * float(source["bin_width"]) for i in range(n_bins + 1)]
    if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(base_edges, expected_edges)):
        raise ValueError("Source regular-bin edges do not match the configured base binning")
    if not math.isclose(base_edges[-1], float(source["max"]), abs_tol=1e-9):
        raise ValueError("Source maximum does not match the configured base binning")

    ordered = {
        region: [region_values[index] for index in range(1, n_bins + 1)]
        for region, region_values in values.items()
    }
    return base_edges, selected, ordered


def derive_edges(
    base_edges: list[float], counts: dict[str, list[int]], minimum: int
) -> list[float]:
    n_bins = len(base_edges) - 1
    first_sparse = next(
        (
            index
            for index in range(n_bins)
            if min(region_counts[index] for region_counts in counts.values()) < minimum
        ),
        n_bins,
    )
    merged_ranges: list[tuple[int, int]] = [(index, index + 1) for index in range(first_sparse)]
    tail_descending: list[tuple[int, int]] = []
    high = n_bins
    while high > first_sparse:
        low = high
        totals = {region: 0 for region in counts}
        while low > first_sparse and min(totals.values()) < minimum:
            low -= 1
            for region, region_counts in counts.items():
                totals[region] += region_counts[low]
        tail_descending.append((low, high))
        high = low

    if tail_descending:
        lowest_low, lowest_high = tail_descending[-1]
        lowest_totals = {
            region: sum(region_counts[lowest_low:lowest_high])
            for region, region_counts in counts.items()
        }
        if min(lowest_totals.values()) < minimum:
            if len(tail_descending) < 2:
                raise ValueError("All sparse bins together fail the minimum-count criterion")
            tail_descending.pop()
            adjacent_low, adjacent_high = tail_descending.pop()
            tail_descending.append((lowest_low, adjacent_high))

    merged_ranges.extend(reversed(tail_descending))
    edges = [base_edges[merged_ranges[0][0]]] if merged_ranges else [base_edges[0]]
    edges.extend(base_edges[high] for _, high in merged_ranges)
    return edges


def merged_counts(
    base_edges: list[float],
    merged_edges: list[float],
    counts: dict[str, list[int]],
) -> list[dict[str, int]]:
    edge_to_index = {edge: index for index, edge in enumerate(base_edges)}
    results = []
    for low, high in zip(merged_edges, merged_edges[1:]):
        if low not in edge_to_index or high not in edge_to_index:
            raise ValueError(f"Merged edge {low} or {high} is not a base edge")
        low_index, high_index = edge_to_index[low], edge_to_index[high]
        results.append(
            {
                region: sum(region_counts[low_index:high_index])
                for region, region_counts in counts.items()
            }
        )
    return results


def render_csv(
    config: dict,
    edges: list[float],
    labels: dict[str, str],
    counts_by_bin: list[dict[str, int]],
) -> str:
    fieldnames = [
        "policy_version",
        "merged_bin_index",
        "bin_low",
        "bin_high",
        "region",
        "region_label",
        "raw_data_count",
        "minimum_sideband_count_in_bin",
        "limiting_sideband_labels",
        "minimum_required_count",
        "validation_status",
    ]
    rows = []
    minimum_required = int(config["policy"]["minimum_raw_data_count_per_sideband"])
    for index, bin_counts in enumerate(counts_by_bin, start=1):
        minimum = min(bin_counts.values())
        limiting = ",".join(labels[region] for region, value in bin_counts.items() if value == minimum)
        for region, count in bin_counts.items():
            rows.append(
                {
                    "policy_version": config["version"],
                    "merged_bin_index": index,
                    "bin_low": f"{edges[index - 1]:.12g}",
                    "bin_high": f"{edges[index]:.12g}",
                    "region": region,
                    "region_label": labels[region],
                    "raw_data_count": count,
                    "minimum_sideband_count_in_bin": minimum,
                    "limiting_sideband_labels": limiting,
                    "minimum_required_count": minimum_required,
                    "validation_status": "PASS",
                }
            )
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_markdown(
    config: dict,
    edges: list[float],
    labels: dict[str, str],
    counts_by_bin: list[dict[str, int]],
    input_table: str,
) -> str:
    ordered_regions = list(labels)
    header = ["Bin", *[labels[region] for region in ordered_regions], "Minimum", "Limiting"]
    lines = [
        f"# {config['version']} sideband-count audit",
        "",
        "Status: **PASS**",
        "",
        f"Source table: `{Path(input_table).resolve()}`",
        "",
        "This audit uses raw regular-bin counts from SB1--SB7 only. It does not use the signal region, calculate an ABCD prediction, or calculate closure.",
        "",
        f"Frozen minimum: **{int(config['policy']['minimum_raw_data_count_per_sideband'])} raw events per sideband per merged regular bin**.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *(["---:"] * 7), "---:", "---"]) + " |",
    ]
    for index, bin_counts in enumerate(counts_by_bin):
        minimum = min(bin_counts.values())
        limiting = ", ".join(labels[region] for region, value in bin_counts.items() if value == minimum)
        values = [
            f"[{edges[index]:g}, {edges[index + 1]:g})",
            *[f"{bin_counts[region]:,}" for region in ordered_regions],
            f"{minimum:,}",
            limiting,
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Underflow remains separate and is not subject to the minimum-count rule. Overflow is already folded into the final regular bin.",
            "",
            "The edges are frozen before formal binned closure inspection and must not be retuned after viewing closure.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    base_edges, labels, counts = load_sideband_counts(args.input_table, config)
    minimum = int(config["policy"]["minimum_raw_data_count_per_sideband"])
    if minimum <= 0:
        raise ValueError("Minimum raw sideband count must be positive")
    target_proxy = float(
        config["policy"]["maximum_four_count_relative_statistical_design_proxy"]
    )
    if math.sqrt(4.0 / minimum) > target_proxy:
        raise ValueError("Minimum count does not satisfy the configured design proxy")

    derived = derive_edges(base_edges, counts, minimum)
    frozen = [float(edge) for edge in config["merged_regular_bin_edges"]]
    if derived != frozen:
        raise ValueError(f"Frozen edges {frozen} do not match derived edges {derived}")
    counts_by_bin = merged_counts(base_edges, frozen, counts)
    for index, bin_counts in enumerate(counts_by_bin, start=1):
        if min(bin_counts.values()) < minimum:
            raise ValueError(f"Merged bin {index} fails the minimum-count criterion")
    for region, region_counts in counts.items():
        if sum(bin_counts[region] for bin_counts in counts_by_bin) != sum(region_counts):
            raise ValueError(f"Merged-count conservation failed for {region}")

    atomic_text(args.output_csv, render_csv(config, frozen, labels, counts_by_bin))
    atomic_text(
        args.output_markdown,
        render_markdown(config, frozen, labels, counts_by_bin, args.input_table),
    )
    print(
        f"Validated {len(frozen) - 1} frozen regular bins for {len(labels)} sidebands; "
        f"minimum count {minimum}; no closure calculated"
    )


if __name__ == "__main__":
    main()
