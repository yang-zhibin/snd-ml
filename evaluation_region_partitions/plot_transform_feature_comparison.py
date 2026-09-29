#!/usr/bin/env python3
"""Plot matched original/transformed Run-8329 feature distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import ROOT
import yaml

from make_passing_muon_binary_region_hist_v5 import configured_v5_features


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def clone(source, path: str):
    value = source.Get(path)
    if not value:
        raise RuntimeError(f"Missing histogram {path}")
    result = value.Clone(path.replace("/", "_"))
    result.SetDirectory(0)
    return result


def summary_page(canvas, counts, label: str, output: str) -> None:
    canvas.Clear()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextSize(0.035)
    lines = [
        label,
        f"Original selected events: {counts.GetBinContent(1):,.0f}",
        f"Matched successful transformations: {counts.GetBinContent(2):,.0f}",
        f"Events without transformed partner: {counts.GetBinContent(4):,.0f}",
        "Selections are frozen from the original RealDataDQ feature row.",
        "Every feature uses events valid in both original and transformed files.",
    ]
    for index, line in enumerate(lines):
        text.DrawLatex(0.08, 0.88 - 0.07 * index, line)
    canvas.Print(output)


def plot_1d(canvas, original, transformed, definition, label: str, output: str) -> None:
    canvas.Clear()
    canvas.Divide(1, 2)
    upper = canvas.cd(1)
    upper.SetPad(0, 0.30, 1, 1)
    upper.SetBottomMargin(0.02)
    upper.SetLogy(bool(definition.get("logy", False)))
    original.SetLineColor(ROOT.kBlack)
    original.SetLineWidth(2)
    transformed.SetLineColor(ROOT.kRed + 1)
    transformed.SetLineWidth(2)
    original.SetTitle(label)
    original.GetXaxis().SetLabelSize(0)
    original.GetYaxis().SetTitle("Matched events")
    maximum = max(original.GetMaximum(), transformed.GetMaximum())
    original.SetMaximum(maximum * (20 if upper.GetLogy() else 1.25))
    if upper.GetLogy():
        original.SetMinimum(0.5)
    original.Draw("HIST")
    transformed.Draw("HIST SAME")
    legend = ROOT.TLegend(0.62, 0.76, 0.89, 0.89)
    legend.AddEntry(original, "Original", "l")
    legend.AddEntry(transformed, "Transformed", "l")
    legend.Draw()

    lower = canvas.cd(2)
    lower.SetPad(0, 0, 1, 0.30)
    lower.SetTopMargin(0.03)
    lower.SetBottomMargin(0.30)
    ratio = transformed.Clone("ratio")
    ratio.Divide(original)
    ratio.SetTitle("")
    ratio.GetYaxis().SetTitle("Trans./orig.")
    ratio.GetYaxis().SetRangeUser(0.0, 2.0)
    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.10)
    ratio.GetYaxis().SetLabelSize(0.09)
    ratio.GetXaxis().SetTitle(definition.get("axis_title", ""))
    ratio.GetXaxis().SetTitleSize(0.11)
    ratio.GetXaxis().SetLabelSize(0.09)
    ratio.Draw("E1")
    line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1, ratio.GetXaxis().GetXmax(), 1)
    line.SetLineStyle(2)
    line.Draw()
    canvas.Print(output)


def plot_2d(canvas, original, transformed, definition, label: str, output: str) -> None:
    canvas.Clear()
    canvas.Divide(2, 1)
    for pad_index, histogram, title in (
        (1, original, "Original"), (2, transformed, "Transformed")
    ):
        pad = canvas.cd(pad_index)
        pad.SetRightMargin(0.14)
        pad.SetLogz(bool(definition.get("logz", False)))
        histogram.SetTitle(f"{label}: {title}")
        histogram.GetXaxis().SetTitle(definition.get("x_title", ""))
        histogram.GetYaxis().SetTitle(definition.get("y_title", ""))
        histogram.Draw("COLZ")
    canvas.Print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.comparison not in config["comparisons"]:
        raise ValueError(f"Unknown comparison {args.comparison}")
    v7 = load_yaml(config["inputs"]["v7_config"])
    feature_config = load_yaml(config["inputs"]["feature_config"])
    features = configured_v5_features(feature_config, v7)
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    root = f"comparisons/{args.comparison}"
    counts = clone(source, f"{root}/counts")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("transform_comparison", "", 1000, 900)
    pdf = str(output)
    canvas.Print(pdf + "[")
    label = config["comparisons"][args.comparison]["label"]
    summary_page(canvas, counts, label, pdf)
    for feature, definition in features.items():
        original = clone(source, f"{root}/features/{feature}/original")
        transformed = clone(source, f"{root}/features/{feature}/transformed")
        if int(definition.get("dimension", 1)) == 1:
            plot_1d(canvas, original, transformed, definition, f"{label}: {feature}", pdf)
        else:
            plot_2d(canvas, original, transformed, definition, f"{label}: {feature}", pdf)
    canvas.Print(pdf + "]")
    source.Close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
