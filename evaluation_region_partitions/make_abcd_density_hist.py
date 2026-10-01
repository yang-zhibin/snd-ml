#!/usr/bin/env python3
"""Build one raw-count Region V1 ABCD density histogram from the catalogue."""

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


REQUIRED_COLUMNS = {
    "partition_version",
    "region_version",
    "region",
    "particle_group",
    "partition_id",
    "feature_partition_path",
    "feature_available",
    "n_events",
    "lumi_per_partition",
}

SUMMARY_COLUMNS = [
    "histogram_version",
    "partition_version",
    "region_version",
    "label",
    "region",
    "particle_group",
    "catalogue_rows",
    "input_files",
    "expected_events_config",
    "expected_events_catalogue",
    "source_tree_entries",
    "finite_feature_entries",
    "nonfinite_feature_entries",
    "underflow_entries",
    "in_range_entries_before_folding",
    "overflow_entries_before_folding",
    "histogram_integral_with_flow",
    "expected_exposure_config",
    "catalogue_exposure",
    "bins",
    "minimum",
    "maximum",
    "bin_width",
    "underflow_policy",
    "overflow_policy",
    "normalization",
    "count_closure",
    "exposure_closure",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-catalogue", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--particle-group")
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


def read_catalogue(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Partition catalogue is missing required columns: "
                + ", ".join(sorted(missing))
            )
        return list(reader)


def selected_rows(
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


def validate_config(config: dict, region: str, particle_group: str) -> None:
    if region not in config.get("regions", {}):
        raise KeyError(f"Region {region!r} is not configured")
    configured_group = str(config["catalogue"]["particle_group"])
    if particle_group != configured_group:
        raise ValueError(
            f"This config is frozen for particle_group={configured_group!r}; "
            f"received {particle_group!r}"
        )
    histogram = config["histogram"]
    bins = int(histogram["bins"])
    minimum = float(histogram["min"])
    maximum = float(histogram["max"])
    width = float(histogram["bin_width"])
    if bins <= 0 or not minimum < maximum:
        raise ValueError("Histogram binning is invalid")
    calculated_width = (maximum - minimum) / bins
    if not math.isclose(calculated_width, width, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Configured bin width {width} does not match {calculated_width}"
        )
    if histogram.get("weight") != "unit":
        raise ValueError("ABCD data histograms must use unit event weights")
    if histogram.get("underflow") != "preserve":
        raise ValueError("Only preserved underflow is supported")
    if histogram.get("overflow") != "fold_last":
        raise ValueError("Only last-bin overflow folding is supported")


def inspect_inputs(rows: list[dict[str, str]], require_available: bool) -> list[str]:
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
    return paths


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


def atomic_root(path: str | Path, histogram, metadata: dict[str, object]) -> None:
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
    particle_group = args.particle_group or str(config["catalogue"]["particle_group"])
    validate_config(config, args.region, particle_group)
    rows = selected_rows(
        read_catalogue(args.partition_catalogue),
        config,
        args.region,
        particle_group,
    )
    validation = config.get("validation", {})
    paths = inspect_inputs(
        rows, bool(validation.get("require_feature_available", True))
    )

    expected_events_catalogue = sum(int(row["n_events"]) for row in rows)
    catalogue_exposure = math.fsum(float(row["lumi_per_partition"]) for row in rows)
    region_config = config["regions"][args.region]
    expected_events_config = int(region_config["expected_n_events"])
    expected_exposure_config = float(region_config["expected_exposure"])
    tolerance = float(validation.get("exposure_absolute_tolerance", 0.0))
    exposure_closure = math.isclose(
        catalogue_exposure,
        expected_exposure_config,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    if expected_events_catalogue != expected_events_config:
        raise ValueError(
            f"Catalogue event count {expected_events_catalogue} does not match "
            f"frozen expectation {expected_events_config}"
        )
    if not exposure_closure:
        raise ValueError(
            f"Catalogue exposure {catalogue_exposure:.12g} does not match "
            f"frozen expectation {expected_exposure_config:.12g}"
        )

    chain = ROOT.TChain(str(config["input"]["tree_name"]))
    for path in paths:
        if chain.Add(path) != 1:
            raise OSError(f"Could not add ROOT input to chain: {path}")
    columns = {str(branch.GetName()) for branch in chain.GetListOfBranches()}
    feature = str(config["input"]["feature"])
    if feature not in columns:
        raise KeyError(f"Feature branch {feature!r} is missing")

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    frame = ROOT.RDataFrame(chain).Define(
        "__abcd_density_is_finite", f"std::isfinite({feature})"
    )
    finite = frame.Filter("__abcd_density_is_finite")
    histogram_config = config["histogram"]
    model = ROOT.RDF.TH1DModel(
        str(histogram_config["name"]),
        ";SciFi density (SNDSW);Raw events",
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
    histogram = actions["histogram"].GetValue().Clone(str(histogram_config["name"]))
    histogram.SetDirectory(0)

    n_bins = histogram.GetNbinsX()
    underflow = float(histogram.GetBinContent(0))
    overflow = float(histogram.GetBinContent(n_bins + 1))
    overflow_error = float(histogram.GetBinError(n_bins + 1))
    in_range_before = float(histogram.Integral(1, n_bins))
    histogram.SetBinContent(
        n_bins, histogram.GetBinContent(n_bins) + overflow
    )
    histogram.SetBinError(
        n_bins, math.hypot(histogram.GetBinError(n_bins), overflow_error)
    )
    histogram.SetBinContent(n_bins + 1, 0.0)
    histogram.SetBinError(n_bins + 1, 0.0)
    integral_with_flow = float(histogram.Integral(0, n_bins + 1))

    count_closure = (
        source_entries == finite_entries == expected_events_catalogue
        and math.isclose(
            integral_with_flow,
            float(expected_events_catalogue),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if bool(validation.get("require_finite_feature", True)) and nonfinite_entries:
        raise ValueError(f"Found {nonfinite_entries} non-finite {feature} values")
    if bool(validation.get("require_count_closure", True)) and not count_closure:
        raise ValueError(
            "Count closure failed: "
            f"catalogue={expected_events_catalogue}, tree={source_entries}, "
            f"finite={finite_entries}, histogram={integral_with_flow:.12g}"
        )

    width = (histogram.GetXaxis().GetXmax() - histogram.GetXaxis().GetXmin()) / n_bins
    summary = {
        "histogram_version": config["version"],
        "partition_version": config["catalogue"]["partition_version"],
        "region_version": config["catalogue"]["region_version"],
        "label": region_config["label"],
        "region": args.region,
        "particle_group": particle_group,
        "catalogue_rows": len(rows),
        "input_files": len(paths),
        "expected_events_config": expected_events_config,
        "expected_events_catalogue": expected_events_catalogue,
        "source_tree_entries": source_entries,
        "finite_feature_entries": finite_entries,
        "nonfinite_feature_entries": nonfinite_entries,
        "underflow_entries": f"{underflow:.12g}",
        "in_range_entries_before_folding": f"{in_range_before:.12g}",
        "overflow_entries_before_folding": f"{overflow:.12g}",
        "histogram_integral_with_flow": f"{integral_with_flow:.12g}",
        "expected_exposure_config": f"{expected_exposure_config:.12f}",
        "catalogue_exposure": f"{catalogue_exposure:.12f}",
        "bins": n_bins,
        "minimum": f"{histogram.GetXaxis().GetXmin():.12g}",
        "maximum": f"{histogram.GetXaxis().GetXmax():.12g}",
        "bin_width": f"{width:.12g}",
        "underflow_policy": histogram_config["underflow"],
        "overflow_policy": histogram_config["overflow"],
        "normalization": "raw_unit_event_counts",
        "count_closure": count_closure,
        "exposure_closure": exposure_closure,
        "status": "PASS" if count_closure and exposure_closure else "FAIL",
    }
    metadata = {
        "histogram_version": config["version"],
        "partition_catalogue": str(Path(args.partition_catalogue).resolve()),
        "region": args.region,
        "region_label": region_config["label"],
        "particle_group": particle_group,
        "feature": feature,
        "normalization": "raw_unit_event_counts",
        "underflow_policy": histogram_config["underflow"],
        "overflow_policy": histogram_config["overflow"],
        "catalogue_rows": len(rows),
        "source_tree_entries": source_entries,
        "finite_feature_entries": finite_entries,
        "nonfinite_feature_entries": nonfinite_entries,
        "underflow_entries": underflow,
        "overflow_entries_before_folding": overflow,
        "histogram_integral_with_flow": integral_with_flow,
        "exposure": catalogue_exposure,
        "count_closure": count_closure,
        "exposure_closure": exposure_closure,
    }
    atomic_root(args.output, histogram, metadata)
    atomic_csv(args.summary, summary)
    print(
        f"{region_config['label']} {particle_group}: "
        f"events={source_entries}, underflow={underflow:.0f}, "
        f"overflow={overflow:.0f}, status={summary['status']}"
    )


if __name__ == "__main__":
    build(parse_args())
