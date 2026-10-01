#!/usr/bin/env python3
"""Build raw and luminosity-normalized Region V1 neutrino-MC histograms."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.TH1.AddDirectory(False)


REQUIRED_CATALOGUE_COLUMNS = {
    "partition_version",
    "region_version",
    "region",
    "particle_group",
    "partition_id",
    "feature_partition_path",
    "feature_available",
    "n_events",
    "source_metadata_csv",
}

SUMMARY_COLUMNS = [
    "histogram_version",
    "binning_version",
    "partition_version",
    "region_version",
    "label",
    "region",
    "particle_group",
    "sample_role",
    "catalogue_rows",
    "input_files",
    "raw_events_catalogue",
    "source_tree_entries",
    "finite_feature_entries",
    "nonfinite_feature_entries",
    "raw_underflow_entries",
    "raw_in_range_entries_before_folding",
    "raw_overflow_entries_before_folding",
    "raw_integral_with_flow",
    "source_metadata_csv",
    "source_metadata_rows",
    "source_mc_luminosity",
    "target_data_exposure",
    "normalization_weight",
    "sum_weights",
    "sum_weights_squared",
    "effective_entries",
    "bins",
    "minimum",
    "maximum",
    "bin_width",
    "underflow_policy",
    "overflow_policy",
    "count_closure",
    "luminosity_closure",
    "weighted_closure",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-catalogue", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--particle-group", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: " + ", ".join(sorted(missing))
            )
        return list(reader)


def validate_config(config: dict, region: str, particle_group: str) -> None:
    if region not in config.get("regions", {}):
        raise KeyError(f"Region {region!r} is not configured")
    if particle_group not in config.get("samples", {}):
        raise KeyError(f"Particle group {particle_group!r} is not configured")
    histogram = config["histogram"]
    bins = int(histogram["bins"])
    minimum = float(histogram["min"])
    maximum = float(histogram["max"])
    width = float(histogram["bin_width"])
    if bins <= 0 or not minimum < maximum:
        raise ValueError("Histogram binning is invalid")
    if not math.isclose(
        (maximum - minimum) / bins, width, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Configured bin width is inconsistent")
    if histogram.get("underflow") != "preserve":
        raise ValueError("Only preserved underflow is supported")
    if histogram.get("overflow") != "fold_last":
        raise ValueError("Only last-bin overflow folding is supported")
    normalization = config["normalization"]
    if normalization.get("mode") != "constant_source_luminosity":
        raise ValueError("Unsupported MC normalization mode")
    if float(normalization.get("event_physics_weight", 0.0)) != 1.0:
        raise ValueError("Only unit event physics weights are validated")


def select_catalogue_rows(
    rows: list[dict[str, str]], config: dict, region: str, particle_group: str
) -> list[dict[str, str]]:
    catalogue = config["catalogue"]
    selected = [
        row
        for row in rows
        if row["partition_version"] == str(catalogue["partition_version"])
        and row["region_version"] == str(catalogue["region_version"])
        and row["region"] == region
        and row["particle_group"] == particle_group
    ]
    if not selected:
        raise ValueError(
            f"No catalogue rows for region={region!r}, "
            f"particle_group={particle_group!r}"
        )
    return selected


def inspect_catalogue_rows(
    rows: list[dict[str, str]], sample: dict, require_available: bool
) -> tuple[list[str], str]:
    partition_ids = [row["partition_id"] for row in rows]
    if len(partition_ids) != len(set(partition_ids)):
        raise ValueError("Partition IDs are not unique")
    if require_available and not all(truthy(row["feature_available"]) for row in rows):
        raise ValueError("At least one catalogue row has feature_available != true")
    paths = [row["feature_partition_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Feature partition paths are not unique")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} feature files; first missing path: {missing[0]}"
        )
    metadata_paths = {row["source_metadata_csv"] for row in rows}
    if len(metadata_paths) != 1:
        raise ValueError(f"Expected one source metadata path; found {metadata_paths}")
    metadata_path = metadata_paths.pop()
    configured_path = str(sample["source_metadata_csv"])
    if metadata_path != configured_path:
        raise ValueError(
            f"Catalogue source metadata {metadata_path!r} does not match "
            f"configured path {configured_path!r}"
        )
    return paths, metadata_path


def source_luminosity(
    path: str | Path, column: str, expected: float, tolerance: float
) -> tuple[int, float]:
    rows = read_csv(path, {column})
    values = [float(row[column]) for row in rows]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"Invalid {column} values in {path}")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError(f"Non-positive source luminosity in {path}")
    if not math.isclose(total, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"Source luminosity {total:.12g} does not match "
            f"configured expectation {expected:.12g}"
        )
    return len(rows), total


def fold_overflow(histogram) -> float:
    n_bins = histogram.GetNbinsX()
    overflow = float(histogram.GetBinContent(n_bins + 1))
    histogram.SetBinContent(
        n_bins, histogram.GetBinContent(n_bins) + overflow
    )
    histogram.SetBinError(
        n_bins,
        math.hypot(
            histogram.GetBinError(n_bins), histogram.GetBinError(n_bins + 1)
        ),
    )
    histogram.SetBinContent(n_bins + 1, 0.0)
    histogram.SetBinError(n_bins + 1, 0.0)
    return overflow


def make_sumw2_histogram(weighted, name: str):
    result = weighted.Clone(name)
    result.Reset("ICES")
    result.SetDirectory(0)
    for index in range(0, weighted.GetNbinsX() + 2):
        result.SetBinContent(index, weighted.GetBinError(index) ** 2)
        result.SetBinError(index, 0.0)
    result.SetTitle(";SciFi density (SNDSW);Sum of squared weights")
    return result


def atomic_csv(path: str | Path, row: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_named(output: ROOT.TFile, name: str, value: object) -> None:
    output.cd()
    if isinstance(value, bool):
        ROOT.TParameter("int")(name, int(value)).Write()
    elif isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, int(value)).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, float(value)).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def atomic_root(
    path: str | Path, histograms: list, metadata: dict[str, object]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".root", dir=destination.parent
    )
    os.close(descriptor)
    try:
        output = ROOT.TFile.Open(temporary, "RECREATE")
        if not output or output.IsZombie():
            raise OSError(f"Could not create temporary ROOT file: {temporary}")
        try:
            output.cd()
            for histogram in histograms:
                histogram.Write()
            for name, value in metadata.items():
                write_named(output, name, value)
        finally:
            output.Close()
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    validate_config(config, args.region, args.particle_group)
    catalogue_rows = read_csv(args.partition_catalogue, REQUIRED_CATALOGUE_COLUMNS)
    rows = select_catalogue_rows(
        catalogue_rows, config, args.region, args.particle_group
    )
    validation = config.get("validation", {})
    sample = config["samples"][args.particle_group]
    paths, metadata_path = inspect_catalogue_rows(
        rows, sample, bool(validation.get("require_feature_available", True))
    )
    tolerance = float(validation.get("luminosity_absolute_tolerance", 0.0))
    metadata_rows, source_mc_luminosity = source_luminosity(
        metadata_path,
        str(config["normalization"]["source_luminosity_column"]),
        float(sample["expected_source_luminosity"]),
        tolerance,
    )
    region_config = config["regions"][args.region]
    target_data_exposure = float(region_config["target_data_exposure"])
    if target_data_exposure <= 0:
        raise ValueError("Target data exposure must be positive")
    normalization_weight = target_data_exposure / source_mc_luminosity

    chain = ROOT.TChain(str(config["input"]["tree_name"]))
    for path in paths:
        if chain.Add(path) != 1:
            raise OSError(f"Could not add ROOT input to chain: {path}")
    branches = {str(branch.GetName()) for branch in chain.GetListOfBranches()}
    feature = str(config["input"]["feature"])
    if feature not in branches:
        raise KeyError(f"Feature branch {feature!r} is missing")

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    frame = ROOT.RDataFrame(chain).Define(
        "__abcd_density_is_finite", f"std::isfinite({feature})"
    )
    finite = frame.Filter("__abcd_density_is_finite")
    histogram_config = config["histogram"]
    model = ROOT.RDF.TH1DModel(
        str(histogram_config["raw_name"]),
        ";SciFi density (SNDSW);Raw MC events",
        int(histogram_config["bins"]),
        float(histogram_config["min"]),
        float(histogram_config["max"]),
    )
    actions = {
        "source": frame.Count(),
        "finite": finite.Count(),
        "histogram": finite.Histo1D(model, feature),
    }
    ROOT.RDF.RunGraphs(list(actions.values()))
    source_entries = int(actions["source"].GetValue())
    finite_entries = int(actions["finite"].GetValue())
    nonfinite_entries = source_entries - finite_entries
    expected_events = sum(int(row["n_events"]) for row in rows)
    raw = actions["histogram"].GetValue().Clone(str(histogram_config["raw_name"]))
    raw.SetDirectory(0)
    n_bins = raw.GetNbinsX()
    underflow = float(raw.GetBinContent(0))
    in_range_before = float(raw.Integral(1, n_bins))
    overflow = fold_overflow(raw)
    raw_integral = float(raw.Integral(0, n_bins + 1))

    count_closure = (
        source_entries == finite_entries == expected_events
        and math.isclose(raw_integral, float(expected_events), abs_tol=1e-9)
    )
    if bool(validation.get("require_finite_feature", True)) and nonfinite_entries:
        raise ValueError(f"Found {nonfinite_entries} non-finite {feature} values")
    if bool(validation.get("require_count_closure", True)) and not count_closure:
        raise ValueError(
            "Count closure failed: "
            f"catalogue={expected_events}, tree={source_entries}, "
            f"finite={finite_entries}, histogram={raw_integral:.12g}"
        )

    weighted = raw.Clone(str(histogram_config["weighted_name"]))
    weighted.SetDirectory(0)
    weighted.SetTitle(";SciFi density (SNDSW);Expected events")
    weighted.Scale(normalization_weight)
    sumw2_histogram = make_sumw2_histogram(
        weighted, str(histogram_config["sumw2_name"])
    )
    sum_weights = float(weighted.Integral(0, n_bins + 1))
    sum_weights_squared = float(sumw2_histogram.Integral(0, n_bins + 1))
    effective_entries = (
        sum_weights * sum_weights / sum_weights_squared
        if sum_weights_squared > 0
        else 0.0
    )
    expected_sum_weights = source_entries * normalization_weight
    expected_sum_weights_squared = source_entries * normalization_weight**2
    weighted_closure = math.isclose(
        sum_weights, expected_sum_weights, rel_tol=1e-12, abs_tol=1e-12
    ) and math.isclose(
        sum_weights_squared,
        expected_sum_weights_squared,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    if not weighted_closure:
        raise ValueError("Weighted sum or sum-of-squared-weights closure failed")

    width = (raw.GetXaxis().GetXmax() - raw.GetXaxis().GetXmin()) / n_bins
    summary = {
        "histogram_version": config["version"],
        "binning_version": config["binning_version"],
        "partition_version": config["catalogue"]["partition_version"],
        "region_version": config["catalogue"]["region_version"],
        "label": region_config["label"],
        "region": args.region,
        "particle_group": args.particle_group,
        "sample_role": sample["role"],
        "catalogue_rows": len(rows),
        "input_files": len(paths),
        "raw_events_catalogue": expected_events,
        "source_tree_entries": source_entries,
        "finite_feature_entries": finite_entries,
        "nonfinite_feature_entries": nonfinite_entries,
        "raw_underflow_entries": f"{underflow:.12g}",
        "raw_in_range_entries_before_folding": f"{in_range_before:.12g}",
        "raw_overflow_entries_before_folding": f"{overflow:.12g}",
        "raw_integral_with_flow": f"{raw_integral:.12g}",
        "source_metadata_csv": metadata_path,
        "source_metadata_rows": metadata_rows,
        "source_mc_luminosity": f"{source_mc_luminosity:.12g}",
        "target_data_exposure": f"{target_data_exposure:.12f}",
        "normalization_weight": f"{normalization_weight:.17g}",
        "sum_weights": f"{sum_weights:.17g}",
        "sum_weights_squared": f"{sum_weights_squared:.17g}",
        "effective_entries": f"{effective_entries:.12g}",
        "bins": n_bins,
        "minimum": f"{raw.GetXaxis().GetXmin():.12g}",
        "maximum": f"{raw.GetXaxis().GetXmax():.12g}",
        "bin_width": f"{width:.12g}",
        "underflow_policy": histogram_config["underflow"],
        "overflow_policy": histogram_config["overflow"],
        "count_closure": count_closure,
        "luminosity_closure": True,
        "weighted_closure": weighted_closure,
        "status": "PASS" if count_closure and weighted_closure else "FAIL",
    }
    metadata = {
        "histogram_version": config["version"],
        "binning_version": config["binning_version"],
        "partition_catalogue": str(Path(args.partition_catalogue).resolve()),
        "region": args.region,
        "region_label": region_config["label"],
        "particle_group": args.particle_group,
        "sample_role": sample["role"],
        "feature": feature,
        "normalization_formula": config["normalization"]["formula"],
        "source_metadata_csv": metadata_path,
        "source_mc_luminosity": source_mc_luminosity,
        "target_data_exposure": target_data_exposure,
        "normalization_weight": normalization_weight,
        "raw_events": source_entries,
        "sum_weights": sum_weights,
        "sum_weights_squared": sum_weights_squared,
        "effective_entries": effective_entries,
        "underflow_entries": underflow,
        "overflow_entries_before_folding": overflow,
        "count_closure": count_closure,
        "weighted_closure": weighted_closure,
    }
    atomic_root(args.output, [raw, weighted, sumw2_histogram], metadata)
    atomic_csv(args.summary, summary)
    print(
        f"{region_config['label']} {args.particle_group}: raw={source_entries}, "
        f"weight={normalization_weight:.12g}, sumw={sum_weights:.12g}, "
        f"sumw2={sum_weights_squared:.12g}, status={summary['status']}"
    )


if __name__ == "__main__":
    build(parse_args())
