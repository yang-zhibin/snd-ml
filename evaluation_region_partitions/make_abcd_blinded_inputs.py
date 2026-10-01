#!/usr/bin/env python3
"""Build the strict one-bin blinded ABCD input contract."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import ROOT

from fit_abcd_density_signal_region import atomic_text, csv_text, load_yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

FORBIDDEN_SIGNAL_OBSERVATION_FIELDS = {
    "a_data", "data_a", "count_a", "observed_a", "a_observed",
    "signal_region_data", "signal_region_observed", "observed_data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--mc-config", required=True)
    parser.add_argument("--product-directory", required=True)
    parser.add_argument("--declared-data-inputs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("status") != "PASS":
        raise ValueError(f"Expected one PASS summary row in {path}")
    return rows[0]


def histogram_integral(path: Path, name: str, bins: int, first: int, last: int) -> float:
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise OSError(f"Could not open {path}")
    histogram = source.Get(name)
    if not histogram or int(histogram.GetNbinsX()) != bins:
        raise ValueError(f"Missing or invalid histogram {name!r} in {path}")
    if float(histogram.GetBinContent(bins + 1)) != 0.0:
        raise ValueError(f"Overflow was not folded in {path}")
    value = float(histogram.Integral(first, last))
    source.Close()
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Invalid integral in {path}")
    return value


def range_bins(config: dict, data_config: dict) -> tuple[int, int, float, float]:
    low, high = [float(value) for value in config["source"]["counting_range"]]
    histogram = data_config["histogram"]
    minimum = float(histogram["min"])
    width = float(histogram["bin_width"])
    bins = int(histogram["bins"])
    if low < minimum or high != float(histogram["max"]) or low % width or high % width:
        raise ValueError("Counting range must align with source bins and end at the histogram maximum")
    first = int(round((low - minimum) / width)) + 1
    return first, bins, low, high


def validate_config(config: dict, data_config: dict, mc_config: dict) -> None:
    roles = config["source"]["roles"]
    validation = config["source"]["validation_roles"]
    if set(roles) != {"A", "B", "C", "D"} or len(set(roles.values())) != 4:
        raise ValueError("ABCD roles must map to four distinct regions")
    if set(validation) != {"numerator", "transfer_b", "transfer_c", "anchor_d"}:
        raise ValueError("Validation roles are incomplete")
    if roles["A"] in set(validation.values()):
        raise ValueError("The independent validation plane must not contain signal-region A")
    if config["blinding"]["signal_region_observation_allowed_in_expected_inputs"]:
        raise ValueError("Expected workflow must forbid signal-region observations")
    if config["source"]["binning_version"] != data_config["version"] or mc_config["binning_version"] != data_config["version"]:
        raise ValueError("Histogram binning version mismatch")
    samples = [config["source"]["signal_sample"], *config["source"]["background_samples"]]
    if set(samples) != set(mc_config["samples"]):
        raise ValueError("Configured MC samples do not match production")
    for region in {*roles.values(), *validation.values()}:
        if region not in data_config["regions"] or region not in mc_config["regions"]:
            raise ValueError(f"Unknown configured region {region}")


def assert_no_signal_data_dependencies(paths: list[str], signal_region: str) -> None:
    forbidden = f"{signal_region}__real_data"
    offenders = [path for path in paths if forbidden in str(path)]
    if offenders:
        raise ValueError("Observed signal-region data entered blinded dependencies: " + ", ".join(offenders))


def assert_blinded_schema(row: dict) -> None:
    forbidden = FORBIDDEN_SIGNAL_OBSERVATION_FIELDS & {str(key).lower() for key in row}
    if forbidden:
        raise ValueError("Forbidden signal-observation fields: " + ", ".join(sorted(forbidden)))
    if not {"count_b", "count_c", "count_d"}.issubset(row):
        raise ValueError("Blinded input is missing control counts")


def make_input_row(
    config: dict,
    data_config: dict,
    mc_config: dict,
    product_directory: Path,
) -> dict:
    validate_config(config, data_config, mc_config)
    first, last, low, high = range_bins(config, data_config)
    bins = int(data_config["histogram"]["bins"])
    data_histogram = str(data_config["histogram"]["name"])
    mc_histograms = mc_config["histogram"]
    roles = config["source"]["roles"]
    validation_roles = config["source"]["validation_roles"]
    samples = [config["source"]["signal_sample"], *config["source"]["background_samples"]]
    backgrounds = config["source"]["background_samples"]
    regions = {*roles.values(), *validation_roles.values()}

    exposures = {
        region: float(data_config["regions"][region]["expected_exposure"])
        for region in regions
    }
    data_counts = {}
    for region in {*[roles[role] for role in ("B", "C", "D")], *validation_roles.values()}:
        summary_path = product_directory / "summaries" / f"{region}__real_data.csv"
        summary = read_summary(summary_path)
        if summary["region"] != region or not math.isclose(float(summary["catalogue_exposure"]), exposures[region], abs_tol=1e-9):
            raise ValueError(f"Data identity/exposure mismatch in {summary_path}")
        data_counts[region] = int(round(histogram_integral(
            product_directory / "hists" / f"{region}__real_data.root",
            data_histogram, bins, first, last,
        )))

    mc = {}
    for region in regions:
        mc[region] = {}
        for sample in samples:
            summary_path = product_directory / "summaries" / f"{region}__{sample}.csv"
            summary = read_summary(summary_path)
            if summary["region"] != region or summary["particle_group"] != sample:
                raise ValueError(f"MC identity mismatch in {summary_path}")
            if not math.isclose(float(summary["target_data_exposure"]), exposures[region], abs_tol=1e-9):
                raise ValueError(f"MC target exposure mismatch for {region} {sample}")
            path = product_directory / "hists" / f"{region}__{sample}.root"
            mc[region][sample] = {
                "yield": histogram_integral(path, str(mc_histograms["weighted_name"]), bins, first, last),
                "variance": histogram_integral(path, str(mc_histograms["sumw2_name"]), bins, first, last),
            }

    row: dict[str, object] = {
        "input_version": config["version"],
        "blinding_status": config["blinding"]["status"],
        "range_low": f"{low:.17g}",
        "range_high": f"{high:.17g}",
    }
    for role, region in roles.items():
        key = role.lower()
        row[f"region_{key}"] = region
        row[f"exposure_{key}"] = f"{exposures[region]:.17g}"
        if role != "A":
            row[f"count_{key}"] = data_counts[region]
        for sample in samples:
            prefix = sample.lower()
            row[f"{prefix}_{key}"] = f"{mc[region][sample]['yield']:.17g}"
            row[f"{prefix}_variance_{key}"] = f"{mc[region][sample]['variance']:.17g}"

    threshold = float(config["model"]["signal_contamination_negligible_threshold"])
    fractions = {}
    for role in ("B", "C", "D"):
        key = role.lower()
        fraction = float(row[f"cc_nue_{key}"]) / float(row[f"count_{key}"])
        fractions[role] = fraction
        row[f"signal_contamination_fraction_{key}"] = f"{fraction:.17g}"
    row["signal_contamination_threshold"] = f"{threshold:.17g}"
    row["signal_contamination_decision"] = (
        "NEGLIGIBLE_MU_FIXED_ZERO" if max(fractions.values()) < threshold
        else "NONNEGLIGIBLE_BACKGROUND_ONLY_CONDITIONAL_MU_ZERO"
    )

    validation_keys = {
        "numerator": "numerator", "transfer_b": "b",
        "transfer_c": "c", "anchor_d": "d",
    }
    residual = {}
    residual_variance = {}
    for name, suffix in validation_keys.items():
        region = validation_roles[name]
        observed = data_counts[region]
        q = sum(mc[region][sample]["yield"] for sample in backgrounds)
        q_variance = sum(mc[region][sample]["variance"] for sample in backgrounds)
        q_norm_variance = sum(
            (float(config["model"]["background_mc_normalization"][sample]) * mc[region][sample]["yield"]) ** 2
            for sample in backgrounds
        )
        residual[name] = observed - q
        residual_variance[name] = observed + q_variance + q_norm_variance
        if residual[name] <= 0:
            raise ValueError(f"Non-positive validation residual for {name}")
        row[f"validation_region_{suffix}"] = region
        row[f"validation_exposure_{suffix}"] = f"{exposures[region]:.17g}"
        row[f"validation_observed_{suffix}"] = observed
        row[f"validation_other_neutrino_{suffix}"] = f"{q:.17g}"
        row[f"validation_residual_{suffix}"] = f"{residual[name]:.17g}"
        row[f"validation_residual_variance_{suffix}"] = f"{residual_variance[name]:.17g}"
        signal_yield = mc[region][config["source"]["signal_sample"]]["yield"]
        row[f"validation_signal_{suffix}"] = f"{signal_yield:.17g}"
        row[f"validation_signal_fraction_{suffix}"] = f"{signal_yield / observed:.17g}"

    exposure_factor = (
        exposures[validation_roles["numerator"]] * exposures[validation_roles["anchor_d"]]
        / (exposures[validation_roles["transfer_b"]] * exposures[validation_roles["transfer_c"]])
    )
    prediction = residual["transfer_b"] * residual["transfer_c"] / residual["anchor_d"] * exposure_factor
    prediction_relative_variance = sum(
        residual_variance[name] / residual[name] ** 2
        for name in ("transfer_b", "transfer_c", "anchor_d")
    )
    prediction_uncertainty = prediction * math.sqrt(prediction_relative_variance)
    transfer = (
        residual["transfer_c"] / residual["anchor_d"]
        * exposures[validation_roles["anchor_d"]]
        / exposures[validation_roles["transfer_c"]]
    )
    transfer_uncertainty = transfer * math.sqrt(
        residual_variance["transfer_c"] / residual["transfer_c"] ** 2
        + residual_variance["anchor_d"] / residual["anchor_d"] ** 2
    )
    kappa = residual["numerator"] / prediction
    relative_variance = sum(
        residual_variance[name] / residual[name] ** 2 for name in validation_keys
    )
    sigma = kappa * math.sqrt(relative_variance)
    relative = sigma / kappa
    pull = (residual["numerator"] - prediction) / math.sqrt(
        residual_variance["numerator"] + prediction_uncertainty ** 2
    )
    row.update({
        "validation_exposure_factor": f"{exposure_factor:.17g}",
        "validation_transfer": f"{transfer:.17g}",
        "validation_transfer_uncertainty": f"{transfer_uncertainty:.17g}",
        "validation_prediction": f"{prediction:.17g}",
        "validation_prediction_uncertainty": f"{prediction_uncertainty:.17g}",
        "validation_fractional_nonclosure": f"{kappa - 1.0:.17g}",
        "validation_pull": f"{pull:.17g}",
        "kappa_central": f"{kappa:.17g}",
        "kappa_uncertainty": f"{sigma:.17g}",
        "kappa_relative_uncertainty": f"{relative:.17g}",
        "kappa_constraint_lo": f"{max(0.01, 1.0 - relative):.17g}",
        "kappa_constraint_hi": f"{1.0 + relative:.17g}",
        "kappa_constraint_form": config["model"]["nonclosure"]["constraint"],
        "kappa_constraint_source": config["model"]["nonclosure"]["source"],
        "kappa_transport_status": config["model"]["nonclosure"]["transport_status"],
    })
    assert_blinded_schema(row)
    return row


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    mc_config = load_yaml(args.mc_config)
    assert_no_signal_data_dependencies(args.declared_data_inputs, config["source"]["roles"]["A"])
    row = make_input_row(config, data_config, mc_config, Path(args.product_directory))
    atomic_text(args.output, csv_text([row]))
    print(
        "wrote strict one-bin blinded input; "
        f"kappa={float(row['kappa_central']):.6g} +/- {float(row['kappa_uncertainty']):.6g}; "
        "signal-region observations present: 0"
    )


if __name__ == "__main__":
    main()
