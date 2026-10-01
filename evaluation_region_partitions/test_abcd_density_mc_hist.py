#!/usr/bin/env python3
"""Synthetic end-to-end test for normalized Region V1 MC density histograms."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "evaluation_region_partitions/make_abcd_density_mc_hist.py"
REGION = "region_test"
PARTICLE_GROUP = "CC_nue"


def make_feature_file(path: Path, values: list[float]) -> None:
    output = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic normalized MC density")
    density = array("d", [0.0])
    tree.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")
    for value in values:
        density[0] = value
        tree.Fill()
    tree.Write()
    output.Close()


def write_source_metadata(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["lumi_per_file"])
        writer.writeheader()
        writer.writerows([{"lumi_per_file": 5.0}, {"lumi_per_file": 5.0}])


def write_catalogue(
    path: Path, inputs: list[Path], source_metadata: Path, expected_events: int
) -> None:
    fieldnames = [
        "partition_version",
        "region_version",
        "region",
        "particle_group",
        "partition_id",
        "feature_partition_path",
        "feature_available",
        "n_events",
        "source_metadata_csv",
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
                    "particle_group": PARTICLE_GROUP,
                    "partition_id": f"part{index:03d}",
                    "feature_partition_path": input_path,
                    "feature_available": True,
                    "n_events": count,
                    "source_metadata_csv": source_metadata,
                }
            )


def write_config(path: Path, source_metadata: Path, expected_lumi: float = 10.0) -> None:
    config = {
        "version": "density_mc_test_v1",
        "binning_version": "density_test_v1",
        "catalogue": {
            "partition_version": "partition_test",
            "region_version": "region_test",
        },
        "input": {"tree_name": "sndData", "feature": "density_sndsw_scifi"},
        "histogram": {
            "raw_name": "h_density_sndsw_scifi_raw",
            "weighted_name": "h_density_sndsw_scifi_weighted",
            "sumw2_name": "h_density_sndsw_scifi_sumw2",
            "bins": 80,
            "min": 0.0,
            "max": 40000.0,
            "bin_width": 500.0,
            "underflow": "preserve",
            "overflow": "fold_last",
        },
        "normalization": {
            "mode": "constant_source_luminosity",
            "formula": "target_data_exposure / source_mc_luminosity",
            "source_luminosity_column": "lumi_per_file",
            "event_physics_weight": 1.0,
        },
        "samples": {
            PARTICLE_GROUP: {
                "role": "signal",
                "source_metadata_csv": str(source_metadata),
                "expected_source_luminosity": expected_lumi,
            }
        },
        "regions": {REGION: {"label": "Test", "target_data_exposure": 2.0}},
        "validation": {
            "luminosity_absolute_tolerance": 1e-12,
            "require_feature_available": True,
            "require_finite_feature": True,
            "require_count_closure": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def command(catalogue: Path, config: Path, output: Path, summary: Path) -> list[str]:
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
        PARTICLE_GROUP,
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--threads",
        "1",
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_density_mc_hist_test.") as directory:
        workdir = Path(directory)
        first = workdir / "first.root"
        second = workdir / "second.root"
        values = [-1.0, 0.0, 499.9, 500.0, 39999.0, 40000.0, 45000.0]
        make_feature_file(first, values[:3])
        make_feature_file(second, values[3:])
        source_metadata = workdir / "source.csv"
        catalogue = workdir / "catalogue.csv"
        config = workdir / "config.yaml"
        output = workdir / "hist.root"
        summary = workdir / "summary.csv"
        write_source_metadata(source_metadata)
        write_catalogue(catalogue, [first, second], source_metadata, 7)
        write_config(config, source_metadata)

        subprocess.run(command(catalogue, config, output, summary), check=True)
        with open(summary, encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert row["status"] == "PASS"
        assert int(row["source_tree_entries"]) == 7
        assert float(row["source_mc_luminosity"]) == 10.0
        assert float(row["target_data_exposure"]) == 2.0
        assert float(row["normalization_weight"]) == 0.2
        assert math.isclose(float(row["sum_weights"]), 1.4)
        assert math.isclose(float(row["sum_weights_squared"]), 0.28)
        assert math.isclose(float(row["effective_entries"]), 7.0)
        assert float(row["raw_underflow_entries"]) == 1.0
        assert float(row["raw_overflow_entries_before_folding"]) == 2.0

        root_file = ROOT.TFile.Open(str(output))
        raw = root_file.Get("h_density_sndsw_scifi_raw")
        weighted = root_file.Get("h_density_sndsw_scifi_weighted")
        sumw2 = root_file.Get("h_density_sndsw_scifi_sumw2")
        assert raw and weighted and sumw2
        assert raw.Integral(0, 81) == 7.0
        assert abs(weighted.Integral(0, 81) - 1.4) < 1e-12
        assert abs(sumw2.Integral(0, 81) - 0.28) < 1e-12
        assert raw.GetBinContent(80) == 3.0
        assert math.isclose(weighted.GetBinContent(80), 0.6)
        assert root_file.Get("count_closure").GetVal() == 1
        assert root_file.Get("weighted_closure").GetVal() == 1
        root_file.Close()

        write_config(config, source_metadata, expected_lumi=11.0)
        failed = subprocess.run(
            command(catalogue, config, output, summary), check=False
        )
        assert failed.returncode != 0

    print("Region V1 normalized MC density histogram synthetic test: OK")


if __name__ == "__main__":
    main()
