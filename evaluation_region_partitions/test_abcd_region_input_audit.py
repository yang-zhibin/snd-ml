#!/usr/bin/env python3
"""Synthetic end-to-end test for the Region V1 ABCD input audit."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITOR = REPO_ROOT / "evaluation_region_partitions/audit_abcd_region_inputs.py"


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_catalogue(path: Path, event_count: int = 7) -> None:
    fieldnames = [
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
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "partition_version": "partition_test",
                "region_version": "region_test",
                "region": "region_a",
                "particle_group": "real_data",
                "partition_id": "region_a__real_data__part001",
                "feature_available": "True",
                "n_events": event_count,
                "lumi_per_partition": "1.25",
                "fraction": "1.0",
                "n_partitions_requested": 1,
                "n_partitions_written": 1,
            }
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_input_audit_") as directory:
        directory = Path(directory)
        catalogue = directory / "catalogue.csv"
        region_config = directory / "regions.yaml"
        audit_config = directory / "audit.yaml"
        output_csv = directory / "audit.csv"
        output_report = directory / "audit.md"
        write_catalogue(catalogue)
        write_yaml(
            region_config,
            {"region_version": "region_test", "regions": {"region_a": {}}},
        )
        write_yaml(
            audit_config,
            {
                "version": "audit_test_v1",
                "catalogue": {
                    "partition_version": "partition_test",
                    "region_version": "region_test",
                    "particle_group": "real_data",
                },
                "validation": {
                    "exposure_absolute_tolerance": 1e-12,
                    "require_feature_available": True,
                    "require_unique_partition_ids": True,
                },
                "regions": {
                    "region_a": {
                        "label": "A",
                        "expected_catalogue_rows": 1,
                        "expected_n_events": 7,
                        "expected_exposure": 1.25,
                        "expected_fraction": 1.0,
                        "expected_partitions_requested": 1,
                        "expected_partitions_written": 1,
                    }
                },
            },
        )
        command = [
            sys.executable,
            str(AUDITOR),
            "--partition-catalogue",
            str(catalogue),
            "--region-config",
            str(region_config),
            "--audit-config",
            str(audit_config),
            "--output-csv",
            str(output_csv),
            "--output-report",
            str(output_report),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        with open(output_csv, encoding="utf-8", newline="") as stream:
            result = next(csv.DictReader(stream))
        assert result["status"] == "PASS"
        assert result["event_count_closure"] == "True"
        assert "Overall status: **PASS**" in output_report.read_text(encoding="utf-8")

        write_catalogue(catalogue, event_count=8)
        failed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        assert failed.returncode == 1
        with open(output_csv, encoding="utf-8", newline="") as stream:
            result = next(csv.DictReader(stream))
        assert result["status"] == "FAIL"
        assert result["event_count_difference"] == "1"
        assert "Overall status: **FAIL**" in output_report.read_text(encoding="utf-8")
    print("Region V1 ABCD input audit synthetic test: OK")


if __name__ == "__main__":
    main()
