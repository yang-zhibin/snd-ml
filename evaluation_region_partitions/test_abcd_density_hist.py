#!/usr/bin/env python3
"""Synthetic end-to-end test for the Region V1 ABCD density builder."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "evaluation_region_partitions/make_abcd_density_hist.py"
REGION = "region_test"


def make_feature_file(path: Path, values: list[float]) -> None:
    output = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic ABCD density")
    density = array("d", [0.0])
    tree.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")
    for value in values:
        density[0] = value
        tree.Fill()
    tree.Write()
    output.Close()


def write_catalogue(path: Path, inputs: list[Path], expected_events: int) -> None:
    fieldnames = [
        "partition_version",
        "region_version",
        "region",
        "particle_group",
        "partition_id",
        "feature_partition_path",
        "feature_available",
        "n_events",
        "lumi_per_partition",
    ]
    counts = [3, expected_events - 3]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (input_path, count) in enumerate(zip(inputs, counts), start=1):
            writer.writerow(
                {
                    "partition_version": "partition_test",
                    "region_version": "region_test",
                    "region": REGION,
                    "particle_group": "real_data",
                    "partition_id": f"part{index:03d}",
                    "feature_partition_path": input_path,
                    "feature_available": True,
                    "n_events": count,
                    "lumi_per_partition": 0.625,
                }
            )


def write_config(path: Path) -> None:
    config = {
        "version": "density_test_v1",
        "catalogue": {
            "partition_version": "partition_test",
            "region_version": "region_test",
            "particle_group": "real_data",
        },
        "input": {"tree_name": "sndData", "feature": "density_sndsw_scifi"},
        "histogram": {
            "name": "h_density_sndsw_scifi_raw",
            "bins": 80,
            "min": 0.0,
            "max": 40000.0,
            "bin_width": 500.0,
            "weight": "unit",
            "underflow": "preserve",
            "overflow": "fold_last",
            "sumw2": True,
        },
        "regions": {
            REGION: {
                "label": "Test",
                "expected_n_events": 7,
                "expected_exposure": 1.25,
            }
        },
        "validation": {
            "exposure_absolute_tolerance": 1e-12,
            "require_feature_available": True,
            "require_unique_partition_ids": True,
            "require_finite_feature": True,
            "require_count_closure": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def command(
    catalogue: Path, config: Path, output: Path, summary: Path
) -> list[str]:
    return [
        sys.executable,
        str(BUILDER),
        "--partition-catalogue",
        str(catalogue),
        "--config",
        str(config),
        "--region",
        REGION,
        "--particle-group",
        "real_data",
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--threads",
        "1",
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_density_hist_test.") as directory:
        workdir = Path(directory)
        first = workdir / "first.root"
        second = workdir / "second.root"
        values = [-1.0, 0.0, 499.9, 500.0, 39999.0, 40000.0, 45000.0]
        make_feature_file(first, values[:3])
        make_feature_file(second, values[3:])
        catalogue = workdir / "catalogue.csv"
        config = workdir / "config.yaml"
        output = workdir / "hist.root"
        summary = workdir / "summary.csv"
        write_catalogue(catalogue, [first, second], expected_events=7)
        write_config(config)

        subprocess.run(
            command(catalogue, config, output, summary),
            cwd=REPO_ROOT,
            check=True,
        )
        with open(summary, encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert row["status"] == "PASS"
        assert row["normalization"] == "raw_unit_event_counts"
        assert int(row["source_tree_entries"]) == 7
        assert float(row["underflow_entries"]) == 1.0
        assert float(row["overflow_entries_before_folding"]) == 2.0
        assert float(row["histogram_integral_with_flow"]) == 7.0

        root_file = ROOT.TFile.Open(str(output))
        histogram = root_file.Get("h_density_sndsw_scifi_raw")
        assert histogram
        assert histogram.GetNbinsX() == 80
        assert histogram.GetXaxis().GetXmin() == 0.0
        assert histogram.GetXaxis().GetXmax() == 40000.0
        assert histogram.GetBinContent(0) == 1.0
        assert histogram.GetBinContent(1) == 2.0
        assert histogram.GetBinContent(2) == 1.0
        assert histogram.GetBinContent(80) == 3.0
        assert histogram.GetBinContent(81) == 0.0
        assert histogram.Integral(0, 81) == 7.0
        assert root_file.Get("count_closure").GetVal() == 1
        root_file.Close()

        write_catalogue(catalogue, [first, second], expected_events=8)
        failed = subprocess.run(
            command(catalogue, config, output, summary),
            cwd=REPO_ROOT,
            check=False,
        )
        assert failed.returncode != 0

    print("Region V1 ABCD density histogram synthetic test: OK")


if __name__ == "__main__":
    main()
