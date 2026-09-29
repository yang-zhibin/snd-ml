#!/usr/bin/env python3
"""Plot natural and sampled Muon-DIS feature counts in separate side-by-side pads."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation_region_partitions"))

from make_passing_muon_binary_region_hist_v5 import configured_v5_features
from plot_passing_muon_region_study_v4 import (
    clone_histogram, positive_minimum_1d, positive_minimum_2d,
)


COLORS = {"natural": 600 + 1, "sampled": 632 + 1}  # ROOT blue/red offsets.


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def draw_label(ROOT, population, summary, feature_title):
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.031)
    heading = "Natural training events" if population == "natural" else (
        "Sampled training draws (repeats counted)"
    )
    text.DrawLatex(0.14, 0.965, heading)
    text.SetTextSize(0.025)
    text.DrawLatex(0.14, 0.925, feature_title)
    count = (
        int(summary["merged_natural_events"])
        if population == "natural" else int(summary["merged_sampled_draws"])
    )
    text.DrawLatex(0.14, 0.885, f"N={count:,}")
    return text


def draw_summary(ROOT, canvas, summary, feature_count):
    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.041)
    text.DrawLatex(0.07, 0.91, "Muon-DIS natural versus sampled feature counts")
    text.SetTextSize(0.026)
    repeated_fraction = (
        summary["muondis_repeated_draws"] / summary["muondis_draws"]
        if summary["muondis_draws"] else 0.0
    )
    lines = [
        f"CV fold: {summary['cv_fold']}   epoch: {summary['epoch']}   seed: {summary['seed']}",
        f"Configured epoch size: {summary['epoch_size']:,}",
        f"Natural eligible Muon-DIS events: {summary['merged_natural_events']:,}",
        f"Sampled Muon-DIS draws: {summary['merged_sampled_draws']:,}",
        f"Unique sampled events: {summary['merged_sampled_unique_events']:,}",
        f"Repeated draws: {summary['muondis_repeated_draws']:,} ({100 * repeated_fraction:.2f}%)",
        f"Source files: {summary['merged_source_files']:,}",
        f"Feature pages: {feature_count}",
        "",
        "Natural: every valid-density event in the non-held-out folds, weight 1.",
        "Sampled: exact deterministic hierarchical epoch draw, with multiplicity.",
        "Class sampling is uniform; domains are natural; populated density bins are uniform.",
        "The two panels use identical bins but independent raw-count axis ranges.",
        "No unit-area normalization and no natural/sampled overlay are used.",
        f"Sampling stage: {summary['sampling_stage']}",
        "",
        "Natural domain counts: " + ", ".join(
            f"{key}={value:,}" for key, value in sorted(
                summary["natural_domain_events"].items()
            )
        ),
        "Sampled domain draws: " + ", ".join(
            f"{key}={value:,}" for key, value in sorted(
                summary["domain_draws"].items()
            )
        ),
    ]
    y = 0.84
    for line in lines:
        text.DrawLatex(0.08, y, line)
        y -= 0.039 if line else 0.020
    canvas._keepalive = [text]
    canvas.Modified()
    canvas.Update()


def draw_1d(ROOT, canvas, source, summary, feature_name, feature):
    canvas.Clear()
    canvas.Divide(2, 1, 0.004, 0.004)
    retained = []
    for index, population in enumerate(("natural", "sampled"), start=1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.05)
        pad.SetBottomMargin(0.15)
        pad.SetTopMargin(0.15)
        logy = bool(feature.get("logy", False))
        pad.SetLogy(logy)
        histogram = clone_histogram(
            source, f"inclusive/{population}/{feature_name}",
            f"plot_{population}_{feature_name}",
        )
        color = COLORS[population]
        histogram.SetLineColor(color)
        histogram.SetFillColorAlpha(color, 0.28)
        histogram.SetLineWidth(2)
        histogram.SetTitle(f";{feature['axis_title']};Events")
        maximum = float(histogram.GetMaximum())
        if logy:
            positive = positive_minimum_1d([histogram]) or 1.0
            histogram.SetMinimum(max(positive * 0.5, 0.5))
            histogram.SetMaximum(max(maximum * 20.0, 10.0))
        else:
            histogram.SetMinimum(0.0)
            histogram.SetMaximum(max(maximum * 1.30, 1.0))
        histogram.Draw("HIST")
        label = draw_label(ROOT, population, summary, feature["axis_title"])
        retained.extend([pad, histogram, label])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def draw_2d(ROOT, canvas, source, summary, feature_name, feature):
    canvas.Clear()
    canvas.Divide(2, 1, 0.004, 0.004)
    retained = []
    for index, population in enumerate(("natural", "sampled"), start=1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.18)
        pad.SetBottomMargin(0.15)
        pad.SetTopMargin(0.15)
        logz = bool(feature.get("logz", False))
        pad.SetLogz(logz)
        histogram = clone_histogram(
            source, f"inclusive/{population}/{feature_name}",
            f"plot_{population}_{feature_name}",
        )
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Events"
        )
        maximum = float(histogram.GetMaximum())
        if logz:
            positive = positive_minimum_2d([histogram]) or 1.0
            histogram.SetMinimum(max(positive * 0.5, 0.5))
            histogram.SetMaximum(max(maximum * 2.0, 10.0))
        else:
            histogram.SetMinimum(0.0)
            histogram.SetMaximum(max(maximum * 1.05, 1.0))
        histogram.Draw("COLZ")
        label = draw_label(
            ROOT, population, summary,
            f"{feature['x_title']} vs {feature['y_title']}",
        )
        retained.extend([pad, histogram, label])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def plot_report(input_root, summary_path, hist_config, region_config, output_pdf):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    with open(summary_path, encoding="utf-8") as stream:
        summary = json.load(stream)
    features = configured_v5_features(hist_config, region_config)
    source = ROOT.TFile.Open(str(input_root))
    if not source or source.IsZombie():
        raise OSError(f"Could not open {input_root}")
    output = Path(output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("muondis_sampling", "muondis_sampling", 1500, 760)
    pages = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(ROOT, canvas, summary, len(features))
        canvas.Print(temporary)
        pages += 1
        for feature_name, feature in features.items():
            if int(feature["dimension"]) == 1:
                draw_1d(ROOT, canvas, source, summary, feature_name, feature)
            else:
                draw_2d(ROOT, canvas, source, summary, feature_name, feature)
            canvas.Print(temporary)
            pages += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    if pages != len(features) + 1:
        raise RuntimeError("Unexpected PDF page count")
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()
    pages = plot_report(
        args.input_root, args.summary, load_yaml(args.hist_config),
        load_yaml(args.region_config), args.output_pdf,
    )
    print(f"Wrote {pages} pages to {args.output_pdf}")


if __name__ == "__main__":
    main()
