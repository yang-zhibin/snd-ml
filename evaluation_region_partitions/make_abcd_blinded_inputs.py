#!/usr/bin/env python3
"""Build an ABCD input table that cannot contain observed signal-region data."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import ROOT

from fit_abcd_density_signal_region import atomic_text, csv_text, load_yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--mc-config", required=True)
    parser.add_argument("--product-directory", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("status") != "PASS":
        raise ValueError(f"Expected one PASS summary row in {path}")
    return rows[0]


def histogram_values(path: Path, name: str, bins: int) -> list[float]:
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise OSError(f"Could not open {path}")
    histogram = source.Get(name)
    if not histogram or int(histogram.GetNbinsX()) != bins:
        raise ValueError(f"Missing or invalid histogram {name!r} in {path}")
    if float(histogram.GetBinContent(bins + 1)) != 0.0:
        raise ValueError(f"Overflow was not folded in {path}")
    values = [float(histogram.GetBinContent(index)) for index in range(1, bins + 1)]
    source.Close()
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"Non-finite or negative content in {path}")
    return values


def validate_config(config: dict, data_config: dict, mc_config: dict) -> None:
    roles = config["source"]["roles"]
    if set(roles) != {"A", "B", "C", "D"} or len(set(roles.values())) != 4:
        raise ValueError("ABCD roles must map to four distinct regions")
    if config["blinding"]["signal_region_observation_allowed_in_expected_inputs"]:
        raise ValueError("Expected workflow must forbid signal-region observations")
    if config["source"]["binning_version"] != data_config["version"]:
        raise ValueError("Data binning version mismatch")
    if mc_config["binning_version"] != data_config["version"]:
        raise ValueError("MC binning version mismatch")
    samples = [config["source"]["signal_sample"], *config["source"]["background_samples"]]
    if set(samples) != set(mc_config["samples"]):
        raise ValueError("Configured MC samples do not match the histogram production")
    for region in roles.values():
        if region not in data_config["regions"] or region not in mc_config["regions"]:
            raise ValueError(f"Unknown configured region {region}")


def build_rows(
    config: dict, data_config: dict, mc_config: dict, product_directory: Path
) -> list[dict]:
    """Read B/C/D data and A/B/C/D MC; never resolve an A-data path."""
    validate_config(config, data_config, mc_config)
    roles = config["source"]["roles"]
    samples = [config["source"]["signal_sample"], *config["source"]["background_samples"]]
    histogram = data_config["histogram"]
    bins = int(histogram["bins"])
    minimum = float(histogram["min"])
    width = float(histogram["bin_width"])
    data_name = str(histogram["name"])
    mc_names = mc_config["histogram"]
    rows = []

    for role, region in roles.items():
        exposure = float(data_config["regions"][region]["expected_exposure"])
        observed = None
        if role != "A":
            data_path = product_directory / "hists" / f"{region}__real_data.root"
            summary_path = product_directory / "summaries" / f"{region}__real_data.csv"
            summary = read_summary(summary_path)
            if summary["region"] != region:
                raise ValueError(f"Data region mismatch in {summary_path}")
            if not math.isclose(float(summary["catalogue_exposure"]), exposure, abs_tol=1e-9):
                raise ValueError(f"Data exposure mismatch for {region}")
            observed = histogram_values(data_path, data_name, bins)

        sample_values = {}
        for sample in samples:
            mc_path = product_directory / "hists" / f"{region}__{sample}.root"
            summary_path = product_directory / "summaries" / f"{region}__{sample}.csv"
            summary = read_summary(summary_path)
            if summary["region"] != region or summary["particle_group"] != sample:
                raise ValueError(f"MC identity mismatch in {summary_path}")
            if not math.isclose(float(summary["target_data_exposure"]), exposure, abs_tol=1e-9):
                raise ValueError(f"MC target exposure mismatch for {region} {sample}")
            sample_values[sample] = {
                "sumw": histogram_values(mc_path, str(mc_names["weighted_name"]), bins),
                "sumw2": histogram_values(mc_path, str(mc_names["sumw2_name"]), bins),
            }

        for index in range(bins):
            row = {
                "input_version": config["version"],
                "blinding_status": config["blinding"]["status"],
                "role": role,
                "region": region,
                "bin_index": index + 1,
                "bin_low": f"{minimum + index * width:.17g}",
                "bin_high": f"{minimum + (index + 1) * width:.17g}",
                "exposure": f"{exposure:.17g}",
                "observed_data": "" if role == "A" else int(round(observed[index])),
            }
            for sample in samples:
                prefix = sample.lower()
                row[f"{prefix}_sum_weights"] = f"{sample_values[sample]['sumw'][index]:.17g}"
                row[f"{prefix}_sum_weights_squared"] = f"{sample_values[sample]['sumw2'][index]:.17g}"
            rows.append(row)
    return rows


def assert_blinded(rows: list[dict]) -> None:
    a_rows = [row for row in rows if row["role"] == "A"]
    if not a_rows or any(row["observed_data"] not in ("", None) for row in a_rows):
        raise ValueError("Sanitized inputs contain an observed signal-region value")
    if any(row["observed_data"] in ("", None) for row in rows if row["role"] != "A"):
        raise ValueError("A control-region observation is missing")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    mc_config = load_yaml(args.mc_config)
    rows = build_rows(config, data_config, mc_config, Path(args.product_directory))
    assert_blinded(rows)
    atomic_text(args.output, csv_text(rows))
    print(f"wrote {len(rows)} sanitized rows; signal-region observations present: 0")


if __name__ == "__main__":
    main()
