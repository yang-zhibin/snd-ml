#!/usr/bin/env python3
"""Build and plot validated unified shower-feature distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_features = {
        "density_sndsw_scifi",
        "scifi_cluster_qdc_per_hit",
        "scifi_qdc_per_hit_direct",
    }
    if set(config.get("features", {})) != required_features:
        raise ValueError("distribution config must contain exactly the three frozen features")
    return config


def histogram_model(feature: str, feature_config: dict) -> tuple[int, float, float]:
    low = float(feature_config["x_min"])
    high = float(feature_config["x_max"])
    width = float(feature_config["bin_width"])
    if not high > low or width <= 0:
        raise ValueError(f"invalid binning for {feature}")
    bins_float = (high - low) / width
    bins = int(round(bins_float))
    if bins <= 0 or not math.isclose(bins_float, bins, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"bin width does not divide the range for {feature}")
    return bins, low, high


def normalization_scale(
    mode: str,
    integral: float,
    bin_width: float,
    exposure: float,
    exposure_kind: str,
    normalization_config: dict,
) -> float:
    if mode == "unit_area":
        if integral < 0:
            raise ValueError("unit-area integral cannot be negative")
        return 0.0 if integral == 0 else 1.0 / (integral * bin_width)
    if mode == "common_exposure_19fb":
        allowed = set(normalization_config[mode]["allowed_exposure_kinds"])
        if exposure_kind not in allowed:
            raise ValueError(
                f"normalization {mode} is not allowed for exposure kind {exposure_kind}"
            )
        if not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"normalization {mode} requires positive exposure")
        return float(normalization_config[mode]["target_exposure_fb"]) / exposure
    if mode == "raw_counts":
        return 1.0
    raise ValueError(f"unknown normalization mode {mode}")


def load_validated_rows(manifest_path: str | Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "family",
        "source_uid",
        "feature_scifi_cluster_v1_path",
        "validation_record",
        "lumi_per_file",
        "exposure_kind",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["family"].astype(str).eq(family)].copy()
    if frame.empty:
        raise ValueError(f"manifest contains no rows for family {family}")
    if frame["source_uid"].duplicated().any():
        raise ValueError(f"manifest contains duplicate source UIDs for {family}")

    for row in frame.to_dict("records"):
        marker = Path(str(row["validation_record"]))
        if not marker.is_file():
            raise RuntimeError(f"missing validation record: {marker}")
        with marker.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("validation_passed") is not True:
            raise RuntimeError(f"failed validation record: {marker}")
        if str(record.get("source_uid", "")) != str(row["source_uid"]):
            raise RuntimeError(f"source UID mismatch in {marker}")
        if str(record.get("feature_path", "")) != str(
            row["feature_scifi_cluster_v1_path"]
        ):
            raise RuntimeError(f"feature path mismatch in {marker}")
    return frame.sort_values("source_uid", kind="mergesort").reset_index(drop=True)


def audit_family(args: argparse.Namespace) -> None:
    rows = load_validated_rows(args.manifest, args.family)
    missing_features = [
        path
        for path in rows["feature_scifi_cluster_v1_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_features:
        preview = ", ".join(missing_features[:3])
        raise RuntimeError(
            f"{len(missing_features)} validated feature files are unreadable; first: {preview}"
        )
    exposure_kinds = sorted(set(rows["exposure_kind"].astype(str)))
    if len(exposure_kinds) != 1:
        raise ValueError(f"mixed exposure kinds for {args.family}: {exposure_kinds}")
    record = {
        "audit_passed": True,
        "distribution_version": load_config(args.config)["version"],
        "family": args.family,
        "source_count": int(len(rows)),
        "validated_feature_count": int(len(rows)),
        "exposure_kind": exposure_kinds[0],
        "manifest_path": str(args.manifest),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"audited {len(rows)} validated {args.family} feature files")


def combine_selection(selection: str, group_expression: str) -> str:
    terms = [str(value).strip() for value in (selection, group_expression) if str(value).strip()]
    return " && ".join(f"({term})" for term in terms)


def _write_named(root, name: str, value: object) -> None:
    import ROOT

    root.cd()
    if isinstance(value, bool):
        ROOT.TParameter("int")(name, int(value)).Write()
    elif isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, int(value)).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, float(value)).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def _numeric_exposure(rows: pd.DataFrame) -> float:
    values = pd.to_numeric(rows["lumi_per_file"], errors="coerce")
    if values.isna().all():
        return 0.0
    if values.isna().any() or (values < 0).any():
        raise ValueError("family has incomplete or invalid exposure values")
    return float(values.sum())


def make_histogram(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()
    ROOT.TH1.AddDirectory(False)

    config = load_config(args.config)
    if args.feature not in config["features"]:
        raise KeyError(f"unknown feature {args.feature}")
    if args.group not in config["groups"]:
        raise KeyError(f"unknown group {args.group}")
    group = config["groups"][args.group]
    family = str(group["family"])
    rows = load_validated_rows(args.manifest, family)
    source_subfolders = [str(value) for value in group.get("source_subfolders", [])]
    if source_subfolders:
        if "subfolder" not in rows:
            raise KeyError(f"manifest has no subfolder column required by {args.group}")
        rows = rows.loc[rows["subfolder"].astype(str).isin(source_subfolders)].copy()
        if rows.empty:
            raise ValueError(
                f"no {family} sources match subfolders {source_subfolders} for {args.group}"
            )
        rows = rows.reset_index(drop=True)
    exposure_kinds = set(rows["exposure_kind"].astype(str))
    if len(exposure_kinds) != 1:
        raise ValueError(f"mixed exposure kinds for {family}: {sorted(exposure_kinds)}")
    exposure_kind = next(iter(exposure_kinds))
    exposure = _numeric_exposure(rows)

    chain = ROOT.TChain(str(config.get("tree_name", "sndData")))
    for path in rows["feature_scifi_cluster_v1_path"].astype(str):
        if chain.Add(path) != 1:
            raise RuntimeError(f"could not add validated feature file {path}")
    selection = combine_selection(
        config["selection"]["expression"], group.get("expression", "")
    )
    dataframe = ROOT.RDataFrame(chain)
    if selection:
        dataframe = dataframe.Filter(selection, "unified_distribution_selection")
    feature_config = config["features"][args.feature]
    bins, low, high = histogram_model(args.feature, feature_config)
    selected = dataframe.Count()
    result = dataframe.Histo1D(
        (f"h_{args.feature}_{args.group}", "", bins, low, high),
        str(feature_config.get("expression", args.feature)),
    )
    selected_entries = int(selected.GetValue())
    histogram = result.GetValue().Clone("hist")
    histogram.SetDirectory(0)
    underflow = float(histogram.GetBinContent(0))
    in_range = float(histogram.Integral(1, histogram.GetNbinsX()))
    overflow = float(histogram.GetBinContent(histogram.GetNbinsX() + 1))
    integral_all = underflow + in_range + overflow
    if not math.isclose(integral_all, selected_entries, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(
            f"histogram flow closure failed: {integral_all} != {selected_entries}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not create {output}")
    histogram.Write("hist")
    metadata = {
        "distribution_version": config["version"],
        "family": family,
        "particle_group": args.group,
        "feature": args.feature,
        "feature_expression": feature_config.get("expression", args.feature),
        "selection_name": config["selection"]["name"],
        "selection_expression": selection,
        "source_count": int(len(rows)),
        "selected_entries": selected_entries,
        "underflow_raw": underflow,
        "in_range_raw": in_range,
        "overflow_raw": overflow,
        "integral_all_raw": integral_all,
        "sample_exposure": exposure,
        "exposure_kind": exposure_kind,
        "bin_width": float(feature_config["bin_width"]),
    }
    for name, value in metadata.items():
        _write_named(root_file, name, value)
    root_file.Close()
    print(f"wrote raw histogram {output}: entries={selected_entries}, sources={len(rows)}")


def _read_value(root_file, name: str, default=None):
    obj = root_file.Get(name)
    if obj is None:
        return default
    if hasattr(obj, "GetVal"):
        return obj.GetVal()
    if hasattr(obj, "GetTitle"):
        return str(obj.GetTitle())
    return default


def _load_histogram(path: str, expected_feature: str) -> dict:
    import ROOT

    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise RuntimeError(f"could not open histogram {path}")
    histogram = source.Get("hist")
    if histogram is None:
        raise RuntimeError(f"missing histogram in {path}")
    histogram = histogram.Clone(f"hist_{Path(path).stem}")
    histogram.SetDirectory(0)
    values = {
        "hist": histogram,
        "feature": _read_value(source, "feature", ""),
        "group": _read_value(source, "particle_group", ""),
        "selection": _read_value(source, "selection_expression", ""),
        "selected_entries": int(_read_value(source, "selected_entries", 0)),
        "underflow_raw": float(_read_value(source, "underflow_raw", 0.0)),
        "in_range_raw": float(_read_value(source, "in_range_raw", 0.0)),
        "overflow_raw": float(_read_value(source, "overflow_raw", 0.0)),
        "integral_all_raw": float(_read_value(source, "integral_all_raw", 0.0)),
        "exposure": float(_read_value(source, "sample_exposure", 0.0)),
        "exposure_kind": _read_value(source, "exposure_kind", ""),
    }
    source.Close()
    if values["feature"] != expected_feature:
        raise RuntimeError(f"feature mismatch in {path}")
    return values


def _fold_flows(histogram, feature_config: dict) -> None:
    if feature_config.get("fold_underflow_for_plot", False):
        content = histogram.GetBinContent(1) + histogram.GetBinContent(0)
        error = math.hypot(histogram.GetBinError(1), histogram.GetBinError(0))
        histogram.SetBinContent(1, content)
        histogram.SetBinError(1, error)
        histogram.SetBinContent(0, 0.0)
        histogram.SetBinError(0, 0.0)
    if feature_config.get("fold_overflow_for_plot", False):
        last = histogram.GetNbinsX()
        content = histogram.GetBinContent(last) + histogram.GetBinContent(last + 1)
        error = math.hypot(histogram.GetBinError(last), histogram.GetBinError(last + 1))
        histogram.SetBinContent(last, content)
        histogram.SetBinError(last, error)
        histogram.SetBinContent(last + 1, 0.0)
        histogram.SetBinError(last + 1, 0.0)


def make_plot(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    config = load_config(args.config)
    feature_config = config["features"][args.feature]
    normalization_config = config["normalizations"]
    requested = []
    for item in args.hist:
        if "=" not in item:
            raise ValueError(f"histogram argument must be GROUP=PATH: {item}")
        group, path = item.split("=", 1)
        if group not in config["groups"]:
            raise KeyError(f"unknown group {group}")
        loaded = _load_histogram(path, args.feature)
        if loaded["group"] != group:
            raise RuntimeError(f"group mismatch in {path}")
        requested.append((group, loaded))
    selections = {loaded["selection"] for _, loaded in requested}
    if len(selections) != 1:
        raise RuntimeError("cannot compare histograms with different selections")

    colors = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kMagenta + 2, ROOT.kBlack]
    curves = []
    for index, (group, loaded) in enumerate(requested):
        histogram = loaded["hist"].Clone(f"plot_{group}")
        histogram.SetDirectory(0)
        _fold_flows(histogram, feature_config)
        integral = float(histogram.Integral(1, histogram.GetNbinsX()))
        scale = normalization_scale(
            args.normalization,
            integral,
            float(feature_config["bin_width"]),
            loaded["exposure"],
            loaded["exposure_kind"],
            normalization_config,
        )
        histogram.Scale(scale)
        histogram.SetLineColor(colors[index % len(colors)])
        histogram.SetLineWidth(2)
        curves.append((group, loaded, histogram, scale))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("canvas", "", 900, 700)
    if feature_config.get("log_y", False):
        canvas.SetLogy()
    maximum = max((curve[2].GetMaximum() for curve in curves), default=1.0)
    legend = ROOT.TLegend(0.58, 0.62, 0.90, 0.88)
    legend.SetBorderSize(0)
    for index, (group, _, histogram, _) in enumerate(curves):
        histogram.SetTitle("")
        histogram.GetXaxis().SetTitle(str(feature_config["axis_title"]))
        if args.normalization == "raw_counts":
            y_title = "Events"
        else:
            y_title = str(normalization_config[args.normalization]["y_axis_title"])
        histogram.GetYaxis().SetTitle(y_title)
        histogram.SetMaximum(maximum * (20.0 if canvas.GetLogy() else 1.35))
        if canvas.GetLogy():
            histogram.SetMinimum(1e-8)
        histogram.Draw("HIST" if index == 0 else "HIST SAME")
        legend.AddEntry(histogram, str(config["groups"][group]["label"]), "l")
    legend.Draw()
    canvas.SaveAs(str(output))

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature", "group", "normalization", "exposure_kind", "sample_exposure",
        "target_exposure_fb", "scale_factor", "selected_entries", "underflow_raw",
        "in_range_raw", "overflow_raw", "integral_all_raw", "scaled_in_range_integral",
    ]
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for group, loaded, histogram, scale in curves:
            writer.writerow(
                {
                    "feature": args.feature,
                    "group": group,
                    "normalization": args.normalization,
                    "exposure_kind": loaded["exposure_kind"],
                    "sample_exposure": f"{loaded['exposure']:.17g}",
                    "target_exposure_fb": normalization_config.get(args.normalization, {}).get(
                        "target_exposure_fb", ""
                    ),
                    "scale_factor": f"{scale:.17g}",
                    "selected_entries": loaded["selected_entries"],
                    "underflow_raw": f"{loaded['underflow_raw']:.17g}",
                    "in_range_raw": f"{loaded['in_range_raw']:.17g}",
                    "overflow_raw": f"{loaded['overflow_raw']:.17g}",
                    "integral_all_raw": f"{loaded['integral_all_raw']:.17g}",
                    "scaled_in_range_integral": f"{histogram.Integral():.17g}",
                }
            )
    print(f"wrote {args.normalization} plot {output} and summary {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--family", required=True)
    audit.add_argument("--output", required=True)
    histogram = subparsers.add_parser("histogram")
    histogram.add_argument("--config", required=True)
    histogram.add_argument("--manifest", required=True)
    histogram.add_argument("--group", required=True)
    histogram.add_argument("--feature", required=True)
    histogram.add_argument("--output", required=True)
    plot = subparsers.add_parser("plot")
    plot.add_argument("--config", required=True)
    plot.add_argument("--feature", required=True)
    plot.add_argument("--normalization", required=True)
    plot.add_argument("--hist", action="append", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit_family(args)
    elif args.command == "histogram":
        make_histogram(args)
    else:
        make_plot(args)


if __name__ == "__main__":
    main()
