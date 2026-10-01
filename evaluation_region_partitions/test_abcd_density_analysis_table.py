#!/usr/bin/env python3
"""Synthetic end-to-end test for the Region V1 density analysis table."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "evaluation_region_partitions/make_abcd_density_analysis_table.py"
REGION = "region_test"
SAMPLES = {
    "CC_nue": ("signal", "cc_nue", 0.2),
    "CC_numu": ("neutrino_background", "cc_numu", 0.3),
    "NC_nue": ("neutrino_background", "nc_nue", 0.4),
    "NC_numu": ("neutrino_background", "nc_numu", 0.5),
}


def write_root(path: Path, names_and_values: dict[str, list[float]]) -> None:
    output = ROOT.TFile.Open(str(path), "RECREATE")
    for name, values in names_and_values.items():
        histogram = ROOT.TH1D(name, name, 2, 0.0, 2.0)
        for index, value in enumerate(values):
            histogram.SetBinContent(index, value)
        histogram.Write()
    output.Close()


def write_summary(path: Path, row: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_inputs(workdir: Path, break_weighted: bool = False) -> tuple[Path, Path, Path]:
    product = workdir / "products"
    (product / "hists").mkdir(parents=True, exist_ok=True)
    (product / "summaries").mkdir(exist_ok=True)
    data_values = [1.0, 2.0, 3.0]
    write_root(product / "hists" / f"{REGION}__real_data.root", {"h_raw": data_values})
    write_summary(
        product / "summaries" / f"{REGION}__real_data.csv",
        {
            "status": "PASS",
            "histogram_integral_with_flow": 6,
            "catalogue_exposure": 2.0,
        },
    )
    raw = [1.0, 2.0, 3.0]
    for sample, (role, _, weight) in SAMPLES.items():
        weighted = [value * weight for value in raw]
        if break_weighted and sample == "CC_nue":
            weighted[1] += 0.1
        sumw2 = [value * weight * weight for value in raw]
        write_root(
            product / "hists" / f"{REGION}__{sample}.root",
            {"h_raw": raw, "h_weighted": weighted, "h_sumw2": sumw2},
        )
        write_summary(
            product / "summaries" / f"{REGION}__{sample}.csv",
            {
                "status": "PASS",
                "sample_role": role,
                "normalization_weight": weight,
                "raw_integral_with_flow": sum(raw),
                "sum_weights": sum(weighted),
                "sum_weights_squared": sum(sumw2),
                "source_mc_luminosity": 10.0,
                "target_data_exposure": 2.0,
            },
        )

    data_config = {
        "version": "density_test_v1",
        "catalogue": {"partition_version": "p", "region_version": "r"},
        "histogram": {
            "name": "h_raw",
            "bins": 2,
            "min": 0.0,
            "max": 2.0,
            "bin_width": 1.0,
            "underflow": "preserve",
            "overflow": "fold_last",
        },
        "regions": {
            REGION: {"label": "Test", "expected_n_events": 6, "expected_exposure": 2.0}
        },
    }
    mc_config = {
        "version": "density_mc_test_v1",
        "binning_version": "density_test_v1",
        "histogram": {
            "raw_name": "h_raw",
            "weighted_name": "h_weighted",
            "sumw2_name": "h_sumw2",
            "bins": 2,
            "min": 0.0,
            "max": 2.0,
            "bin_width": 1.0,
            "underflow": "preserve",
            "overflow": "fold_last",
        },
        "samples": {sample: {"role": role} for sample, (role, _, _) in SAMPLES.items()},
        "regions": {REGION: {"label": "Test", "target_data_exposure": 2.0}},
    }
    table_config = {
        "version": "table_test_v1",
        "binning_version": "density_test_v1",
        "samples": {
            sample: {"role": role, "column_prefix": prefix}
            for sample, (role, prefix, _) in SAMPLES.items()
        },
        "validation": {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-10,
            "require_source_status_pass": True,
            "require_zero_overflow": True,
            "require_integer_raw_counts": True,
            "require_exact_table_reconstruction": True,
        },
    }
    paths = []
    for name, config in (
        ("data.yaml", data_config),
        ("mc.yaml", mc_config),
        ("table.yaml", table_config),
    ):
        path = workdir / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1], paths[2]


def command(
    data_config: Path, mc_config: Path, table_config: Path, product: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(BUILDER),
        "--config",
        str(table_config),
        "--data-config",
        str(data_config),
        "--mc-config",
        str(mc_config),
        "--product-directory",
        str(product),
        "--output",
        str(output),
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_density_table_test.") as directory:
        workdir = Path(directory)
        data_config, mc_config, table_config = write_inputs(workdir)
        output = workdir / "table.csv"
        subprocess.run(
            command(data_config, mc_config, table_config, workdir / "products", output),
            check=True,
        )
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        assert [row["row_type"] for row in rows] == ["underflow", "bin", "bin", "inclusive"]
        inclusive = rows[-1]
        assert int(inclusive["data_raw_events"]) == 6
        assert math.isclose(float(inclusive["cc_nue_sum_weights"]), 1.2)
        assert math.isclose(float(inclusive["cc_nue_sum_weights_squared"]), 0.24)
        assert math.isclose(float(inclusive["cc_nue_source_mc_luminosity"]), 10.0)
        assert math.isclose(float(inclusive["neutrino_background_sum_weights"]), 7.2)
        assert math.isclose(float(inclusive["standard_neutrino_sum_weights"]), 8.4)
        assert math.isclose(float(inclusive["cc_nue_signal_fraction_of_data"]), 0.2)
        assert inclusive["source_validation_status"] == "PASS"

        data_config, mc_config, table_config = write_inputs(workdir, break_weighted=True)
        failed = subprocess.run(
            command(data_config, mc_config, table_config, workdir / "products", output),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert failed.returncode != 0

    print("Region V1 density analysis table synthetic test: OK")


if __name__ == "__main__":
    main()
