#!/usr/bin/env python3
"""Render PDF-only comparisons from validated ROOT histogram bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def load_plot_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _bundle_arguments(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"bundle argument must be GROUP=PATH: {value}")
        group, path = value.split("=", 1)
        if group in result:
            raise ValueError(f"duplicate bundle group {group}")
        result[group] = path
    return result


def _metadata(source) -> dict:
    obj = source.Get("metadata/bundle_json")
    if not obj:
        raise RuntimeError("bundle has no metadata/bundle_json")
    return json.loads(str(obj.GetString().Data()))


def load_histograms(args: argparse.Namespace, dimension: int):
    import ROOT

    config = load_plot_config(args.config)
    if args.cohort not in config["cohorts"]:
        raise KeyError(f"unknown cohort {args.cohort}")
    expected = list(config["cohorts"][args.cohort]["groups"])
    supplied = _bundle_arguments(args.bundle)
    if set(supplied) != set(expected):
        raise RuntimeError(
            f"bundle groups do not match cohort; missing={sorted(set(expected)-set(supplied))}, "
            f"extra={sorted(set(supplied)-set(expected))}"
        )
    directory = "features" if dimension == 1 else "features2d"
    loaded = []
    base_selection = None
    for group in expected:
        source = ROOT.TFile.Open(supplied[group])
        if not source or source.IsZombie():
            raise RuntimeError(f"cannot open {supplied[group]}")
        metadata = _metadata(source)
        if metadata["group"] != group:
            raise RuntimeError(f"group mismatch in {supplied[group]}")
        selection = metadata["base_selection"]
        if base_selection is None:
            base_selection = selection
        elif selection != base_selection:
            raise RuntimeError("cohort bundles have inconsistent base selections")
        histogram = source.Get(f"{directory}/{args.feature}")
        if not histogram or histogram.GetDimension() != dimension:
            raise RuntimeError(f"missing {directory}/{args.feature} in {supplied[group]}")
        histogram = histogram.Clone(f"plot_{dimension}_{group}_{args.feature}")
        histogram.SetDirectory(0)
        source.Close()
        loaded.append((group, metadata, histogram))
    return config, loaded


def _particle(metadata: dict) -> str:
    value = str(metadata.get("particle_charge", ""))
    return "electron" if value.startswith("e") else "pion"


def _label(metadata: dict) -> str:
    symbol = "e" if _particle(metadata) == "electron" else "#pi"
    charge = str(metadata.get("particle_charge", ""))
    if "+" in charge:
        symbol += "^{+}"
    elif "-" in charge:
        symbol += "^{-}"
    return f"{symbol}, {float(metadata['beam_energy_gev']):g} GeV"


def _cohort(config: dict, name: str) -> dict:
    return config["cohorts"][name]


def _group_label(group: str, metadata: dict, config: dict, cohort: str) -> str:
    configured = _cohort(config, cohort).get("group_labels", {})
    if group in configured:
        label = str(configured[group])
    elif metadata.get("beam_energy_gev", "") not in ("", None):
        label = _label(metadata)
    else:
        label = str(metadata.get("group_label", group))
    if _cohort(config, cohort).get("show_source_count", False):
        source_count = metadata.get("source_count")
        if source_count not in (None, ""):
            label += f" ({int(source_count):,} sources)"
    return label


def _style_name(group: str, metadata: dict, config: dict, cohort: str) -> str:
    configured = _cohort(config, cohort).get("group_styles", {})
    return str(configured.get(group, _particle(metadata)))


def _unit_area(histogram) -> None:
    integral = float(histogram.Integral(1, histogram.GetNbinsX()))
    if integral > 0:
        histogram.Scale(1.0 / integral)


def _style(histogram, group: str, metadata: dict, config: dict, cohort: str) -> None:
    style = config["styles"][_style_name(group, metadata, config, cohort)]
    histogram.SetLineColor(int(style["color"]))
    histogram.SetMarkerColor(int(style["color"]))
    histogram.SetMarkerStyle(int(style["marker"]))
    histogram.SetMarkerSize(0.55)
    histogram.SetLineWidth(1)
    histogram.SetStats(False)


def faceted_1d(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    config, loaded = load_histograms(args, 1)
    cohort = _cohort(config, args.cohort)
    group_size = int(cohort.get("facet_group_size", 2))
    facets = [loaded[index:index + group_size] for index in range(0, len(loaded), group_size)]
    columns = int(cohort.get("facet_columns", 1))
    rows = math.ceil(len(facets) / columns)
    canvas = ROOT.TCanvas("faceted_1d", "", 540 * columns, 430 * rows)
    canvas.Divide(columns, rows, 0.002, 0.002)
    keep = []
    x_range = config.get("display_x_ranges", {}).get(args.feature)
    for pad_index, facet in enumerate(facets, 1):
        pad = canvas.cd(pad_index)
        pad.SetLeftMargin(0.13)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.11)
        maximum = 0.0
        for group, metadata, histogram in facet:
            _unit_area(histogram)
            _style(histogram, group, metadata, config, args.cohort)
            if x_range:
                histogram.GetXaxis().SetRangeUser(float(x_range[0]), float(x_range[1]))
            maximum = max(maximum, histogram.GetMaximum())
        legend = ROOT.TLegend(0.50, 0.70, 0.91, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.032)
        keep.append(legend)
        for index, (group, metadata, histogram) in enumerate(facet):
            histogram.GetYaxis().SetTitle("Fraction of events / bin")
            histogram.GetYaxis().SetTitleSize(0.045)
            histogram.GetXaxis().SetTitleSize(0.045)
            histogram.SetMaximum(maximum * 1.3 if maximum else 1.0)
            histogram.SetMinimum(0.0)
            histogram.Draw("E1 P" if index == 0 else "E1 P SAME")
            legend.AddEntry(histogram, _group_label(group, metadata, config, args.cohort), "lep")
        legend.Draw()
        annotation = str(cohort.get("annotation", "")).strip()
        if annotation:
            label = ROOT.TLatex()
            label.SetNDC(True)
            label.SetTextSize(0.025)
            label.SetTextColor(15)
            label.DrawLatex(0.14, 0.93, annotation)
            keep.append(label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))


def energy_summary(args: argparse.Namespace) -> None:
    import ROOT
    from array import array

    ROOT.gROOT.SetBatch(True)
    config, loaded = load_histograms(args, 1)
    series = {"pion": [], "electron": []}
    for _, metadata, histogram in loaded:
        energy = float(metadata["beam_energy_gev"])
        mean = float(histogram.GetMean())
        error = float(histogram.GetMeanError())
        series[_particle(metadata)].append((energy, mean, error))
    canvas = ROOT.TCanvas("energy_summary", "", 850, 650)
    canvas.SetLeftMargin(0.13)
    canvas.SetBottomMargin(0.13)
    graphs = []
    all_values = [value for values in series.values() for value in values]
    if not all_values:
        raise RuntimeError("no points available for energy summary")
    x_values = [value[0] for value in all_values]
    y_low = min(value[1] - value[2] for value in all_values)
    y_high = max(value[1] + value[2] for value in all_values)
    padding = max(0.05 * (y_high - y_low), 0.05 * max(abs(y_high), 1.0))
    frame = canvas.DrawFrame(
        min(x_values) * 0.9, max(0.0, y_low - padding),
        max(x_values) * 1.05, y_high + padding,
        f";Beam energy [GeV];Mean {loaded[0][2].GetXaxis().GetTitle()}",
    )
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    legend = ROOT.TLegend(0.67, 0.74, 0.91, 0.89)
    legend.SetBorderSize(0)
    for index, particle in enumerate(("pion", "electron")):
        values = sorted(series[particle])
        graph = ROOT.TGraphErrors(
            len(values), array("d", [v[0] for v in values]),
            array("d", [v[1] for v in values]), array("d", [0.0] * len(values)),
            array("d", [v[2] for v in values]),
        )
        style = config["styles"][particle]
        graph.SetMarkerColor(int(style["color"]))
        graph.SetLineColor(int(style["color"]))
        graph.SetMarkerStyle(int(style["marker"]))
        graph.Draw("P SAME")
        legend.AddEntry(graph, particle.capitalize(), "lep")
        graphs.append(graph)
    legend.Draw()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))


def faceted_2d(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    config, loaded = load_histograms(args, 2)
    cohort = _cohort(config, args.cohort)
    if cohort.get("sort_2d_by_particle", str(args.cohort).startswith("data_")):
        loaded = sorted(
            loaded,
            key=lambda item: (
                1 if _particle(item[1]) == "electron" else 0,
                float(item[1]["beam_energy_gev"]),
            ),
        )
    columns = int(cohort.get("two_dimensional_columns", 4))
    rows = math.ceil(len(loaded) / columns)
    canvas = ROOT.TCanvas("faceted_2d", "", 450 * columns, 360 * rows)
    canvas.Divide(columns, rows, 0.002, 0.002)
    maximum = 0.0
    positive = []
    for _, _, histogram in loaded:
        integral = float(histogram.Integral())
        if integral > 0:
            histogram.Scale(1.0 / integral)
        maximum = max(maximum, histogram.GetMaximum())
        positive.extend(
            histogram.GetBinContent(ix, iy)
            for ix in range(1, histogram.GetNbinsX() + 1)
            for iy in range(1, histogram.GetNbinsY() + 1)
            if histogram.GetBinContent(ix, iy) > 0
        )
    floor_fraction = float(config["styles"].get("th2_z_floor_fraction", 1e-5))
    minimum = max(min(positive), maximum * floor_fraction) if positive else 1e-8
    labels = []
    for index, (group, metadata, histogram) in enumerate(loaded, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.12)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
        if config["styles"].get("th2_log_z", True):
            pad.SetLogz()
        histogram.SetMinimum(minimum)
        histogram.SetMaximum(maximum if maximum else 1.0)
        histogram.Draw("COLZ")
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextSize(0.042)
        label.DrawLatex(0.16, 0.92, _group_label(group, metadata, config, args.cohort))
        labels.append(label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))


def manifest(args: argparse.Namespace) -> None:
    config = load_plot_config(args.config)
    payload = {
        "plot_version": config["version"],
        "normalization": config["normalization"],
        "individual_csv_written": False,
        "bundles": _bundle_arguments(args.bundle),
        "outputs": sorted(str(value) for value in args.plot),
    }
    coverage_receipt = getattr(args, "coverage_receipt", None)
    if coverage_receipt:
        coverage_path = Path(coverage_receipt)
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if coverage.get("validation_passed") is not True:
            raise RuntimeError(f"coverage receipt did not pass: {coverage_path}")
        payload["feature_snapshot_coverage"] = coverage
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name, function in (
        ("faceted-1d", faceted_1d),
        ("energy-summary", energy_summary),
        ("faceted-2d", faceted_2d),
    ):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--cohort", required=True)
        command.add_argument("--feature", required=True)
        command.add_argument("--bundle", action="append", default=[], required=True)
        command.add_argument("--output", required=True)
        command.set_defaults(function=function)
    command = sub.add_parser("manifest")
    command.add_argument("--config", required=True)
    command.add_argument("--bundle", action="append", default=[], required=True)
    command.add_argument("--plot", action="append", default=[])
    command.add_argument("--coverage-receipt")
    command.add_argument("--output", required=True)
    command.set_defaults(function=manifest)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
