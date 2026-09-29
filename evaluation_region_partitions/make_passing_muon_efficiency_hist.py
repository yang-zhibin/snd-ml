#!/usr/bin/env python3
"""Build all run-8329 real-DQ/passing-DQ histograms for one selection."""

import argparse
import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import ROOT
import yaml


ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def chain_for_group(metadata, group, tree_name):
    rows = metadata.loc[metadata["particle_group"].astype(str) == group].sort_values("partition_id")
    if rows.empty:
        raise ValueError(f"No partition rows for {group}")
    chain = ROOT.TChain(tree_name)
    for path in rows["feature_partition_path"].astype(str):
        if chain.Add(path) <= 0:
            raise RuntimeError(f"Could not add {path}")
    return chain, rows


def validate_hist_config(config):
    features = config.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("Histogram config must contain a non-empty features mapping")
    for name, feature in features.items():
        dimension = int(feature.get("dimension", 1))
        if dimension == 1:
            required = {"expression", "n_bins", "x_min", "x_max", "axis_title"}
        elif dimension == 2:
            required = {
                "x_expression", "y_expression", "x_bins", "x_min", "x_max",
                "y_bins", "y_min", "y_max", "x_title", "y_title",
            }
        else:
            raise ValueError(f"Feature {name!r} has unsupported dimension {dimension}")
        missing = required - set(feature)
        if missing:
            raise ValueError(f"Feature {name!r} lacks fields: {sorted(missing)}")


def _add_bin(histogram, target, source):
    histogram.SetBinContent(
        target, histogram.GetBinContent(target) + histogram.GetBinContent(source)
    )
    histogram.SetBinError(
        target,
        math.hypot(histogram.GetBinError(target), histogram.GetBinError(source)),
    )
    histogram.SetBinContent(source, 0.0)
    histogram.SetBinError(source, 0.0)


def fold_1d(histogram, underflow, overflow):
    if underflow:
        _add_bin(histogram, 1, 0)
    if overflow:
        last = histogram.GetNbinsX()
        _add_bin(histogram, last, last + 1)


def fold_2d_axis(histogram, axis, underflow, overflow):
    """Fold a TH2 axis, including edge/corner flow entries exactly once."""
    nx, ny = histogram.GetNbinsX(), histogram.GetNbinsY()
    if axis == "x":
        for iy in range(0, ny + 2):
            if underflow:
                _add_bin(histogram, histogram.GetBin(1, iy), histogram.GetBin(0, iy))
            if overflow:
                _add_bin(
                    histogram,
                    histogram.GetBin(nx, iy),
                    histogram.GetBin(nx + 1, iy),
                )
        return
    if axis == "y":
        for ix in range(1, nx + 1):
            if underflow:
                _add_bin(histogram, histogram.GetBin(ix, 1), histogram.GetBin(ix, 0))
            if overflow:
                _add_bin(
                    histogram,
                    histogram.GetBin(ix, ny),
                    histogram.GetBin(ix, ny + 1),
                )
        return
    raise ValueError(f"Unsupported TH2 axis {axis!r}")


def configured_features(config):
    defaults = config.get("defaults", {})
    result = {}
    for name, value in config["features"].items():
        feature = dict(defaults)
        feature.update(value)
        feature["dimension"] = int(feature.get("dimension", 1))
        result[name] = feature
    return result


def define_shared_columns(dataframe, definitions):
    columns = {str(column) for column in dataframe.GetColumnNames()}
    for name, expression in definitions.items():
        if name in columns:
            raise ValueError(f"Configured definition {name!r} already exists in input")
        dataframe = dataframe.Define(str(name), str(expression))
        columns.add(str(name))
    return dataframe


def feature_frame_and_columns(dataframe, feature_name, feature):
    frame = dataframe
    feature_cut = str(feature.get("feature_cut", "")).strip()
    if feature_cut:
        frame = frame.Filter(feature_cut, f"{feature_name}_feature_selection")
    if feature["dimension"] == 1:
        column = f"__feature_{feature_name}"
        return frame.Define(column, str(feature["expression"])), (column,)
    x_column = f"__feature_{feature_name}_x"
    y_column = f"__feature_{feature_name}_y"
    frame = frame.Define(x_column, str(feature["x_expression"]))
    frame = frame.Define(y_column, str(feature["y_expression"]))
    return frame, (x_column, y_column)


def book_histogram(dataframe, feature_name, label, feature, columns):
    name = f"{feature_name}__{label}"
    if feature["dimension"] == 1:
        model = (
            name, "", int(feature["n_bins"]),
            float(feature["x_min"]), float(feature["x_max"]),
        )
        return dataframe.Histo1D(model, columns[0])
    model = (
        name, "", int(feature["x_bins"]),
        float(feature["x_min"]), float(feature["x_max"]),
        int(feature["y_bins"]), float(feature["y_min"]), float(feature["y_max"]),
    )
    return dataframe.Histo2D(model, columns[0], columns[1])


def clone_and_fold(result, name, feature):
    histogram = result.GetValue().Clone(name)
    histogram.SetDirectory(0)
    if feature["dimension"] == 1:
        fold_1d(
            histogram,
            bool(feature.get("fold_underflow", False)),
            bool(feature.get("fold_overflow", False)),
        )
        histogram.GetXaxis().SetTitle(str(feature["axis_title"]))
        histogram.GetYaxis().SetTitle("Events")
    else:
        fold_2d_axis(
            histogram, "x",
            bool(feature.get("fold_x_underflow", False)),
            bool(feature.get("fold_x_overflow", False)),
        )
        fold_2d_axis(
            histogram, "y",
            bool(feature.get("fold_y_underflow", False)),
            bool(feature.get("fold_y_overflow", False)),
        )
        histogram.GetXaxis().SetTitle(str(feature["x_title"]))
        histogram.GetYaxis().SetTitle(str(feature["y_title"]))
        histogram.GetZaxis().SetTitle("Events")
    return histogram


def assert_subset(denominator, numerator, feature_name):
    if denominator.GetDimension() == 1:
        indices = range(0, denominator.GetNbinsX() + 2)
    else:
        indices = (
            denominator.GetBin(ix, iy)
            for ix in range(0, denominator.GetNbinsX() + 2)
            for iy in range(0, denominator.GetNbinsY() + 2)
        )
    for index in indices:
        if numerator.GetBinContent(index) > denominator.GetBinContent(index) + 1e-9:
            raise RuntimeError(
                f"Passing count exceeds real-data count for {feature_name} in bin {index}"
            )


def write_parameter(output, name, value):
    output.cd()
    if isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, value).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, value).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--selection", default="none")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    hist_config = load_yaml(args.hist_config)
    eval_config = load_yaml(args.eval_config)
    validate_hist_config(hist_config)
    if args.selection not in eval_config["selections"]:
        raise KeyError(f"Unknown selection {args.selection!r}")
    if not bool(eval_config["statistics"].get("require_subset", True)):
        raise ValueError("The consolidated comparison requires exact-subset statistics")
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)

    metadata = pd.read_csv(args.metadata)
    tree_name = str(eval_config.get("tree_name", "sndData"))
    groups = {
        "real_data": str(eval_config["groups"]["denominator"]),
        "passing_muon": str(eval_config["groups"]["numerator"]),
    }
    selection_expression = str(eval_config["selections"][args.selection])
    features = configured_features(hist_config)

    owners = []
    rows_by_label = {}
    actions = {}
    histogram_actions = {}
    for label, group in groups.items():
        chain, rows = chain_for_group(metadata, group, tree_name)
        owners.append(chain)
        rows_by_label[label] = rows
        dataframe = ROOT.RDataFrame(chain).Filter(
            selection_expression, f"{label}_{args.selection}_selection"
        )
        dataframe = define_shared_columns(
            dataframe, hist_config.get("definitions", {})
        )
        actions[f"{label}_selected"] = dataframe.Count()
        for feature_name, feature in features.items():
            feature_frame, columns = feature_frame_and_columns(
                dataframe, feature_name, feature
            )
            actions[f"{feature_name}__{label}_selected"] = feature_frame.Count()
            histogram_actions[(feature_name, label)] = book_histogram(
                feature_frame, feature_name, label, feature, columns
            )

    ROOT.RDF.RunGraphs(list(actions.values()) + list(histogram_actions.values()))
    selected_counts = {name: int(action.GetValue()) for name, action in actions.items()}
    if selected_counts["passing_muon_selected"] > selected_counts["real_data_selected"]:
        raise RuntimeError("Selected passing count exceeds selected real-data count")

    histograms = {}
    for feature_name, feature in features.items():
        for label in groups:
            histograms[(feature_name, label)] = clone_and_fold(
                histogram_actions[(feature_name, label)],
                f"{feature_name}__{label}",
                feature,
            )
        assert_subset(
            histograms[(feature_name, "real_data")],
            histograms[(feature_name, "passing_muon")],
            feature_name,
        )

    luminosity = float(rows_by_label["real_data"]["lumi_per_partition"].sum())
    passing_luminosity = float(
        rows_by_label["passing_muon"]["lumi_per_partition"].sum()
    )
    if abs(luminosity - passing_luminosity) > 1e-12:
        raise RuntimeError("Real-data and passing-muon luminosities differ")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    try:
        output = ROOT.TFile.Open(temporary_name, "RECREATE")
        if not output or output.IsZombie():
            raise OSError(f"Could not create {temporary_name}")
        try:
            for feature_name, feature in features.items():
                directory = output.mkdir(feature_name)
                directory.cd()
                for label in groups:
                    histograms[(feature_name, label)].Write(label)
                output.cd()
                write_parameter(
                    output, f"{feature_name}__dimension", int(feature["dimension"])
                )
                for label in groups:
                    write_parameter(
                        output,
                        f"{feature_name}__{label}_selected",
                        selected_counts[f"{feature_name}__{label}_selected"],
                    )
            write_parameter(output, "histogram_version", hist_config["version"])
            write_parameter(output, "selection_key", args.selection)
            write_parameter(output, "selection_expression", selection_expression)
            write_parameter(output, "real_data_selected", selected_counts["real_data_selected"])
            write_parameter(output, "passing_muon_selected", selected_counts["passing_muon_selected"])
            write_parameter(output, "luminosity_fb", luminosity)
            write_parameter(output, "subset_semantics", "exact")
            write_parameter(output, "feature_order", ",".join(features))
        finally:
            output.Close()
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    denominator = selected_counts["real_data_selected"]
    numerator = selected_counts["passing_muon_selected"]
    ratio = numerator / denominator if denominator else 0.0
    print(
        f"{args.selection}: wrote {len(features)} comparison features; "
        f"PassingMuonReco after DQ {numerator}/{denominator} = {ratio:.6%}"
    )


if __name__ == "__main__":
    main()
