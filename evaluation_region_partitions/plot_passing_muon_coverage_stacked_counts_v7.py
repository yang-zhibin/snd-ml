#!/usr/bin/env python3
"""Plot raw Run-8329 coverage-pass/fail counts for every V7 detector feature."""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from make_passing_muon_binary_region_hist_v5 import (
    POPULATIONS,
    configured_selections,
    configured_v5_features,
    validate_region_config,
)


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

PASS_COLOR = ROOT.kBlue + 1
FAIL_COLOR = ROOT.kRed + 1
INCLUSIVE_COLOR = ROOT.kBlack


def metadata(source, name):
    return common.object_value(source, name)


def histogram(source, selection, feature, population):
    return common.clone_histogram(
        source,
        f"selections/{selection}/features/{feature}/{POPULATIONS[population]}",
        f"coverage_stack_{selection}_{feature}_{population}",
    )


def selected_histograms(source, selection, feature):
    passed = histogram(source, selection, feature, "candidate")
    failed = histogram(source, selection, feature, "complement")
    inclusive = passed.Clone(f"coverage_stack_{selection}_{feature}_inclusive")
    inclusive.SetDirectory(0)
    inclusive.Add(failed)
    return passed, failed, inclusive


def style_histograms(passed, failed, inclusive):
    passed.SetLineColor(PASS_COLOR)
    passed.SetFillColorAlpha(PASS_COLOR, 0.48)
    passed.SetLineWidth(2)
    failed.SetLineColor(FAIL_COLOR)
    failed.SetFillColorAlpha(FAIL_COLOR, 0.48)
    failed.SetLineWidth(2)
    inclusive.SetLineColor(INCLUSIVE_COLOR)
    inclusive.SetMarkerColor(INCLUSIVE_COLOR)
    inclusive.SetMarkerStyle(20)
    inclusive.SetMarkerSize(0.50)
    inclusive.SetLineWidth(2)


def valid_count(source, selection, population, feature):
    return int(metadata(
        source,
        f"selection__{selection}__{population}__{feature}__valid_count",
    ))


def draw_header(selection, title):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.08, 0.972, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.95, 0.972, "Run 8329 RealDataDQ raw event counts")
    label.SetTextAlign(13)
    label.SetTextSize(0.025)
    label.DrawLatex(0.08, 0.935, f"{selection}: {title}")
    return label


def draw_summary(canvas, source, selection, definition, feature_count):
    total = int(metadata(source, "real_data_dq_count"))
    passed = int(metadata(source, f"selection__{selection}__candidate_count"))
    failed = int(metadata(source, f"selection__{selection}__complement_count"))
    closure = bool(int(metadata(
        source, f"selection__{selection}__population_closure"
    )))
    if not closure or passed + failed != total:
        raise RuntimeError(
            f"Coverage closure failed for {selection}: {passed} + {failed} != {total}"
        )

    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.039)
    text.DrawLatex(0.08, 0.87, "Coverage pass/fail stacked-count summary")
    text.SetTextSize(0.024)
    lines = [
        definition["label"],
        f"RealDataDQ events: {total:,}",
        f"Coverage pass: {passed:,} "
        f"({100.0 * passed / total if total else 0.0:.5f}%)",
        f"Coverage fail: {failed:,} "
        f"({100.0 * failed / total if total else 0.0:.5f}%)",
        f"Closure: {passed:,} + {failed:,} = {total:,}",
        f"Configured detector features: {feature_count}",
        "Blue and red histograms are stacked raw event counts.",
        "Black points are their inclusive sum; no unit-area normalization is used.",
        "Feature validity cuts are retained and their valid event counts are shown.",
        "All histogram axes use linear count scales.",
        "No reconstructed-track requirement or particle-truth label is used.",
        "",
        "Detector requirements:",
    ]
    for name, expression in definition["components"].items():
        lines.extend(textwrap.wrap(
            f"{name}: {expression}",
            width=105,
            subsequent_indent="    ",
            break_long_words=False,
        ))
    y = 0.81
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.032 if line else 0.016
    header = draw_header(selection, "coverage-only stacked counts")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def draw_1d(canvas, source, selection, feature_name, feature):
    passed, failed, inclusive = selected_histograms(
        source, selection, feature_name
    )
    style_histograms(passed, failed, inclusive)
    pass_valid = valid_count(source, selection, "candidate", feature_name)
    fail_valid = valid_count(source, selection, "complement", feature_name)

    canvas.Clear()
    upper = ROOT.TPad(f"upper_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetLeftMargin(0.14)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.13)
    upper.SetBottomMargin(0.025)
    upper.SetLogy(False)
    lower.SetLeftMargin(0.14)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.36)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()

    upper.cd()
    stack = ROOT.THStack(f"stack_{feature_name}", "")
    stack.Add(failed)
    stack.Add(passed)
    stack.Draw("HIST")
    stack.SetTitle(f";{feature['axis_title']};Events")
    stack.SetMinimum(0.0)
    stack.SetMaximum(max(float(inclusive.GetMaximum()) * 1.35, 1.0))
    inclusive.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.46, 0.66, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    legend.AddEntry(
        inclusive,
        f"Inclusive (valid N={pass_valid + fail_valid:,})",
        "lep",
    )
    legend.AddEntry(passed, f"Coverage pass (valid N={pass_valid:,})", "f")
    legend.AddEntry(failed, f"Coverage fail (valid N={fail_valid:,})", "f")
    legend.Draw()
    header = draw_header(selection, feature["axis_title"])

    lower.cd()
    fractions = []
    for index, (population, item, color, marker) in enumerate((
        ("pass", passed, PASS_COLOR, 20),
        ("fail", failed, FAIL_COLOR, 21),
    )):
        fraction = item.Clone(f"fraction_{feature_name}_{population}")
        fraction.SetDirectory(0)
        fraction.Divide(item, inclusive, 1.0, 1.0, "B")
        fraction.SetLineColor(color)
        fraction.SetMarkerColor(color)
        fraction.SetMarkerStyle(marker)
        fraction.SetMarkerSize(0.50)
        fraction.SetTitle(f";{feature['axis_title']};Class fraction")
        fraction.GetXaxis().SetTitleSize(0.13)
        fraction.GetXaxis().SetLabelSize(0.10)
        fraction.GetXaxis().SetTitleOffset(1.15)
        fraction.GetYaxis().SetTitleSize(0.095)
        fraction.GetYaxis().SetLabelSize(0.085)
        fraction.GetYaxis().SetTitleOffset(0.65)
        fraction.GetYaxis().SetNdivisions(505)
        fraction.SetMinimum(0.0)
        fraction.SetMaximum(1.05)
        fraction.Draw("E1 X0" if index == 0 else "E1 X0 SAME")
        fractions.append(fraction)
    fraction_legend = ROOT.TLegend(0.76, 0.70, 0.94, 0.94)
    fraction_legend.SetBorderSize(0)
    fraction_legend.SetFillStyle(0)
    fraction_legend.SetTextSize(0.075)
    fraction_legend.AddEntry(fractions[0], "pass", "lep")
    fraction_legend.AddEntry(fractions[1], "fail", "lep")
    fraction_legend.Draw()
    canvas._keepalive = [
        upper, lower, stack, passed, failed, inclusive, legend, header,
        fraction_legend, *fractions,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, selection, feature_name, feature):
    passed, failed, inclusive = selected_histograms(
        source, selection, feature_name
    )
    panels = (
        (passed, "Coverage pass", valid_count(
            source, selection, "candidate", feature_name
        )),
        (failed, "Coverage fail", valid_count(
            source, selection, "complement", feature_name
        )),
        (inclusive, "Inclusive", valid_count(
            source, selection, "candidate", feature_name
        ) + valid_count(source, selection, "complement", feature_name)),
    )
    canvas.Clear()
    canvas.Divide(3, 1, 0.002, 0.002)
    retained = [passed, failed, inclusive]
    for index, (item, label, count) in enumerate(panels, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.18)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.16)
        pad.SetLogz(False)
        item.SetMinimum(0.0)
        item.SetMaximum(max(float(item.GetMaximum()) * 1.05, 1.0))
        item.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Events"
        )
        item.Draw("COLZ")
        title = ROOT.TLatex()
        title.SetNDC(True)
        title.SetTextFont(42)
        title.SetTextSize(0.024)
        title.DrawLatex(0.15, 0.965, f"{label} (valid N={count:,})")
        retained.extend([pad, title])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument(
        "--selection",
        required=True,
        choices=("coverage_loose", "coverage_tight"),
    )
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    hist_config = common.load_yaml(args.hist_config)
    config = common.load_yaml(args.region_config)
    validate_region_config(config)
    features = configured_v5_features(hist_config, config)
    selections = configured_selections(config)
    definition = selections[args.selection]

    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if bool(int(metadata(source, "uses_reconstructed_tracks"))):
        raise RuntimeError("Coverage stacked counts require detector-only input")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("coverage_stacked_counts_v7", "", 1500, 900)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(
            canvas, source, args.selection, definition, len(features)
        )
        canvas.Print(temporary)
        page_count += 1
        for feature_name, feature in features.items():
            if int(feature.get("dimension", 1)) == 1:
                draw_1d(
                    canvas, source, args.selection, feature_name, feature
                )
            else:
                draw_2d(
                    canvas, source, args.selection, feature_name, feature
                )
            canvas.Print(temporary)
            page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} coverage pass/fail stacked-count pages to {output}")


if __name__ == "__main__":
    main()
