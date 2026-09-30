#!/usr/bin/env python3
"""Audit inclusive Region V1 real-data inputs before ABCD estimation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml


REQUIRED_COLUMNS = {
    "partition_version",
    "region_version",
    "region",
    "particle_group",
    "partition_id",
    "feature_available",
    "n_events",
    "lumi_per_partition",
    "fraction",
    "n_partitions_requested",
    "n_partitions_written",
}

OUTPUT_COLUMNS = [
    "audit_version",
    "partition_version",
    "region_version",
    "label",
    "region",
    "particle_group",
    "expected_catalogue_rows",
    "observed_catalogue_rows",
    "catalogue_row_closure",
    "expected_n_events",
    "observed_n_events",
    "event_count_difference",
    "event_count_closure",
    "expected_exposure",
    "observed_exposure",
    "exposure_difference",
    "exposure_absolute_tolerance",
    "exposure_closure",
    "expected_fraction",
    "observed_fractions",
    "fraction_closure",
    "expected_partitions_requested",
    "observed_partitions_requested",
    "partitions_requested_closure",
    "expected_partitions_written",
    "observed_partitions_written",
    "partitions_written_closure",
    "feature_available_closure",
    "partition_id_unique",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-catalogue", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--audit-config", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def unique_numbers(rows: list[dict], column: str, converter) -> list:
    values = set()
    for row in rows:
        try:
            values.add(converter(row[column]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {column}={row.get(column)!r} in partition "
                f"{row.get('partition_id', '<unknown>')}"
            ) from error
    return sorted(values)


def one_value(values: list) -> str:
    return str(values[0]) if len(values) == 1 else ";".join(map(str, values))


def atomic_write(path: str | Path, writer) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer(stream)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_catalogue(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Partition catalogue is missing required columns: "
                + ", ".join(sorted(missing))
            )
        return list(reader)


def audit_rows(catalogue_rows: list[dict], region_config: dict, config: dict):
    catalogue = config["catalogue"]
    validation = config.get("validation", {})
    partition_version = str(catalogue["partition_version"])
    region_version = str(catalogue["region_version"])
    particle_group = str(catalogue["particle_group"])
    tolerance = float(validation.get("exposure_absolute_tolerance", 0.0))
    require_features = bool(validation.get("require_feature_available", True))
    require_unique_ids = bool(validation.get("require_unique_partition_ids", True))

    configured_regions = config.get("regions", {})
    defined_regions = set(region_config.get("regions", {}))
    if str(region_config.get("region_version")) != region_version:
        raise ValueError("Region config version does not match the audit config")
    if set(configured_regions) != defined_regions:
        raise ValueError(
            "Audit regions do not match the region config: "
            f"audit={sorted(configured_regions)}, region_config={sorted(defined_regions)}"
        )

    selected = [
        row
        for row in catalogue_rows
        if str(row["partition_version"]) == partition_version
        and str(row["region_version"]) == region_version
        and str(row["particle_group"]) == particle_group
    ]
    discovered_regions = {str(row["region"]) for row in selected}
    if discovered_regions != set(configured_regions):
        raise ValueError(
            "Catalogue regions do not match the audit config: "
            f"catalogue={sorted(discovered_regions)}, audit={sorted(configured_regions)}"
        )

    by_region: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_region[str(row["region"])].append(row)

    results = []
    for region, expected in configured_regions.items():
        rows = by_region[region]
        event_counts = unique_numbers(rows, "n_events", int)
        observed_events = sum(int(row["n_events"]) for row in rows)
        exposures = unique_numbers(rows, "lumi_per_partition", float)
        observed_exposure = math.fsum(float(row["lumi_per_partition"]) for row in rows)
        fractions = unique_numbers(rows, "fraction", float)
        partitions_requested = unique_numbers(rows, "n_partitions_requested", int)
        partitions_written = unique_numbers(rows, "n_partitions_written", int)
        partition_ids = [str(row["partition_id"]) for row in rows]
        # The unique-value calls also validate every numeric cell.
        assert event_counts and exposures

        row_closure = len(rows) == int(expected["expected_catalogue_rows"])
        count_closure = observed_events == int(expected["expected_n_events"])
        exposure_difference = observed_exposure - float(expected["expected_exposure"])
        exposure_closure = math.isclose(
            observed_exposure,
            float(expected["expected_exposure"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        fraction_closure = len(fractions) == 1 and math.isclose(
            fractions[0],
            float(expected["expected_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        requested_closure = partitions_requested == [
            int(expected["expected_partitions_requested"])
        ]
        written_closure = partitions_written == [
            int(expected["expected_partitions_written"])
        ]
        feature_closure = (not require_features) or all(
            truthy(row["feature_available"]) for row in rows
        )
        ids_unique = len(partition_ids) == len(set(partition_ids))
        id_closure = (not require_unique_ids) or ids_unique
        checks = (
            row_closure,
            count_closure,
            exposure_closure,
            fraction_closure,
            requested_closure,
            written_closure,
            feature_closure,
            id_closure,
        )
        results.append(
            {
                "audit_version": config["version"],
                "partition_version": partition_version,
                "region_version": region_version,
                "label": expected["label"],
                "region": region,
                "particle_group": particle_group,
                "expected_catalogue_rows": int(expected["expected_catalogue_rows"]),
                "observed_catalogue_rows": len(rows),
                "catalogue_row_closure": row_closure,
                "expected_n_events": int(expected["expected_n_events"]),
                "observed_n_events": observed_events,
                "event_count_difference": (
                    observed_events - int(expected["expected_n_events"])
                ),
                "event_count_closure": count_closure,
                "expected_exposure": f"{float(expected['expected_exposure']):.12f}",
                "observed_exposure": f"{observed_exposure:.12f}",
                "exposure_difference": f"{exposure_difference:.12g}",
                "exposure_absolute_tolerance": f"{tolerance:.12g}",
                "exposure_closure": exposure_closure,
                "expected_fraction": expected["expected_fraction"],
                "observed_fractions": one_value(fractions),
                "fraction_closure": fraction_closure,
                "expected_partitions_requested": expected[
                    "expected_partitions_requested"
                ],
                "observed_partitions_requested": one_value(partitions_requested),
                "partitions_requested_closure": requested_closure,
                "expected_partitions_written": expected[
                    "expected_partitions_written"
                ],
                "observed_partitions_written": one_value(partitions_written),
                "partitions_written_closure": written_closure,
                "feature_available_closure": feature_closure,
                "partition_id_unique": ids_unique,
                "status": "PASS" if all(checks) else "FAIL",
            }
        )
    return results


def write_csv(path: str | Path, rows: list[dict]) -> None:
    def writer(stream):
        output = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        output.writeheader()
        output.writerows(rows)

    atomic_write(path, writer)


def write_report(
    path: str | Path,
    rows: list[dict],
    catalogue_path: str | Path,
    audit_config_path: str | Path,
) -> None:
    overall = all(row["status"] == "PASS" for row in rows)

    def writer(stream):
        stream.write(f"# Region V1 ABCD input audit: {rows[0]['audit_version']}\n\n")
        stream.write(f"Overall status: **{'PASS' if overall else 'FAIL'}**\n\n")
        stream.write(f"- Partition catalogue: {Path(catalogue_path)}\n")
        stream.write(f"- Frozen expectations: {Path(audit_config_path)}\n")
        stream.write("- Event-count policy: inclusive sum of metadata n_events.\n")
        stream.write("- Exposure policy: sum of metadata lumi_per_partition.\n\n")
        stream.write(
            "| Region | Rows | Events (observed / expected) | "
            "Exposure (observed / expected) | Status |\n"
        )
        stream.write("| --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            stream.write(
                f"| {row['label']} | {row['observed_catalogue_rows']} / "
                f"{row['expected_catalogue_rows']} | {row['observed_n_events']} / "
                f"{row['expected_n_events']} | {row['observed_exposure']} / "
                f"{row['expected_exposure']} | {row['status']} |\n"
            )
        stream.write("\nThe CSV companion records every individual closure check.\n")

    atomic_write(path, writer)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.audit_config)
    region_config = load_yaml(args.region_config)
    rows = audit_rows(read_catalogue(args.partition_catalogue), region_config, config)
    write_csv(args.output_csv, rows)
    write_report(
        args.output_report,
        rows,
        args.partition_catalogue,
        args.audit_config,
    )
    passed = all(row["status"] == "PASS" for row in rows)
    print(
        f"Region V1 ABCD input audit: {'PASS' if passed else 'FAIL'}; "
        f"wrote {args.output_csv} and {args.output_report}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
