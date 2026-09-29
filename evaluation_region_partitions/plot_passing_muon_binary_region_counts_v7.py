#!/usr/bin/env python3
"""Plot raw data counts and detector-family N-minus-one diagnostics for V7."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from make_passing_muon_binary_region_hist_v5 import (
    POPULATIONS,
    configured_count_diagnostics,
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


def clone(source, path, name):
    return common.clone_histogram(source, path, name)


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


def style_histograms(passed, failed, inclusive):
    passed.SetLineColor(PASS_COLOR)
    passed.SetFillColorAlpha(PASS_COLOR, 0.45)
    passed.SetLineWidth(2)
    failed.SetLineColor(FAIL_COLOR)
    failed.SetFillColorAlpha(FAIL_COLOR, 0.45)
    failed.SetLineWidth(2)
    inclusive.SetLineColor(INCLUSIVE_COLOR)
    inclusive.SetMarkerColor(INCLUSIVE_COLOR)
    inclusive.SetMarkerStyle(20)
    inclusive.SetMarkerSize(0.55)
    inclusive.SetLineWidth(2)


def full_selection_histograms(source, selection, feature_name, tag):
    base = f"selections/{selection}/features/{feature_name}"
    passed = clone(
        source,
        f"{base}/{POPULATIONS['candidate']}",
        f"{tag}_{feature_name}_candidate",
    )
    failed = clone(
        source,
        f"{base}/{POPULATIONS['complement']}",
        f"{tag}_{feature_name}_complement",
    )
    inclusive = passed.Clone(f"{tag}_{feature_name}_inclusive")
    inclusive.SetDirectory(0)
    inclusive.Add(failed)
    style_histograms(passed, failed, inclusive)
    return passed, failed, inclusive


def nminus1_histograms(source, selection, family, feature_name, tag):
    base = (
        f"count_diagnostics/{selection}/{family}/features/{feature_name}"
    )
    passed = clone(source, f"{base}/pass", f"{tag}_{feature_name}_pass")
    failed = clone(source, f"{base}/fail", f"{tag}_{feature_name}_fail")
    inclusive = clone(
        source, f"{base}/inclusive", f"{tag}_{feature_name}_inclusive"
    )
    style_histograms(passed, failed, inclusive)
    return passed, failed, inclusive


def draw_stack_panel(
    pad, histograms, feature, title, pass_label, fail_label, compact=False,
    draw_legend=False,
):
    passed, failed, inclusive = histograms
    pad.cd()
    pad.SetLeftMargin(0.15 if compact else 0.13)
    pad.SetRightMargin(0.04)
    pad.SetTopMargin(0.13 if compact else 0.10)
    pad.SetBottomMargin(0.14 if compact else 0.13)
    pad.SetLogy(bool(feature.get("logy", True)))
    stack = ROOT.THStack(f"stack_{passed.GetName()}", "")
    stack.Add(failed)
    stack.Add(passed)
    stack.Draw("HIST")
    stack.SetTitle(f";{feature['axis_title']};Events")
    stack.GetXaxis().SetTitleSize(0.045 if compact else 0.050)
    stack.GetYaxis().SetTitleSize(0.045 if compact else 0.050)
    stack.GetXaxis().SetLabelSize(0.036 if compact else 0.040)
    stack.GetYaxis().SetLabelSize(0.036 if compact else 0.040)
    stack.GetYaxis().SetTitleOffset(1.45 if compact else 1.25)
    maximum = max(float(inclusive.GetMaximum()), 1.0)
    if pad.GetLogy():
        stack.SetMinimum(0.5)
        stack.SetMaximum(maximum * 20.0)
    else:
        stack.SetMinimum(0.0)
        stack.SetMaximum(maximum * 1.35)
    inclusive.Draw("E1 X0 SAME")
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.040 if compact else 0.032)
    text.DrawLatex(0.16, 0.92, title)
    note = ROOT.TLatex()
    note.SetNDC(True)
    note.SetTextFont(42)
    note.SetTextSize(0.025 if compact else 0.021)
    if bool(feature.get("fold_overflow", False)):
        note.DrawLatex(0.57, 0.92, "overflow #rightarrow last bin")
    legend = None
    if draw_legend:
        legend = ROOT.TLegend(0.50, 0.69, 0.94, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.027)
        legend.AddEntry(inclusive, "Inclusive data", "lep")
        legend.AddEntry(passed, pass_label, "f")
        legend.AddEntry(failed, fail_label, "f")
        legend.Draw()
    return [passed, failed, inclusive, stack, text, note, legend]


def draw_summary(canvas, source, selection, definition, diagnostics):
    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.038)
    text.DrawLatex(0.08, 0.87, "Raw-count and N-minus-one summary")
    total = int(metadata(source, "real_data_dq_count"))
    candidate = int(metadata(
        source, f"selection__{selection}__candidate_count"
    ))
    complement = int(metadata(
        source, f"selection__{selection}__complement_count"
    ))
    lines = [
        definition["label"],
        f"RealDataDQ events: {total:,}",
        f"Full-selection candidate: {candidate:,} "
        f"({100.0 * candidate / total if total else 0.0:.5f}%)",
        f"Full-selection topology failure: {complement:,}",
        "Full-selection detector plots are sculpted by their own requirements.",
        "N-1: all other detector-family requirements pass; this family is removed.",
        "Pass/fail are selection categories, not particle-truth labels.",
        "Pooled MC is omitted because its unweighted event sum has no data exposure.",
        "",
        "Detector-family N-minus-one closure:",
    ]
    for family_name, family in diagnostics["families"].items():
        prefix = f"selection__{selection}__count_diagnostic__{family_name}"
        parent = int(metadata(source, f"{prefix}__inclusive_count"))
        passed = int(metadata(source, f"{prefix}__pass_count"))
        failed = int(metadata(source, f"{prefix}__fail_count"))
        lines.append(
            f"{family['label']}: {passed:,} pass + {failed:,} fail = "
            f"{parent:,} parent; pass fraction "
            f"{100.0 * passed / parent if parent else 0.0:.4f}%"
        )
    text.SetTextSize(0.024)
    y = 0.81
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.039 if line else 0.020
    header = draw_header(selection, "data counts; no unit-area normalization")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def draw_overview(
    canvas, source, selection, feature_names, features, title,
    histogram_getter, pass_label, fail_label,
):
    canvas.Clear()
    canvas.Divide(3 if len(feature_names) > 4 else 2, 2, 0.002, 0.002)
    retained = []
    for index, feature_name in enumerate(feature_names, start=1):
        pad = canvas.cd(index)
        histograms = histogram_getter(feature_name, f"overview_{selection}_{index}")
        retained.extend(draw_stack_panel(
            pad,
            histograms,
            features[feature_name],
            features[feature_name]["axis_title"],
            pass_label,
            fail_label,
            compact=True,
            draw_legend=index == 1,
        ))
        retained.append(pad)
    canvas.cd(0)
    header = draw_header(selection, title)
    retained.append(header)
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def draw_detail(
    canvas, source, selection, family_name, family, feature_name, feature,
):
    canvas.Clear()
    upper = ROOT.TPad(f"upper_{family_name}_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{family_name}_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetBottomMargin(0.025)
    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.04)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.36)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()
    histograms = nminus1_histograms(
        source, selection, family_name, feature_name,
        f"detail_{selection}_{family_name}",
    )
    retained = draw_stack_panel(
        upper,
        histograms,
        feature,
        f"{family['label']} N-minus-one: {feature['axis_title']}",
        "Pass family requirement",
        "Fail family requirement",
        draw_legend=True,
    )

    passed, _, inclusive = histograms
    lower.cd()
    fraction = passed.Clone(f"fraction_{selection}_{family_name}_{feature_name}")
    fraction.SetDirectory(0)
    fraction.Divide(passed, inclusive, 1.0, 1.0, "B")
    fraction.SetLineColor(PASS_COLOR)
    fraction.SetMarkerColor(PASS_COLOR)
    fraction.SetMarkerStyle(20)
    fraction.SetMarkerSize(0.55)
    fraction.SetTitle(f";{feature['axis_title']};Pass / parent")
    fraction.GetXaxis().SetTitleSize(0.13)
    fraction.GetXaxis().SetLabelSize(0.10)
    fraction.GetXaxis().SetTitleOffset(1.15)
    fraction.GetYaxis().SetTitleSize(0.095)
    fraction.GetYaxis().SetLabelSize(0.085)
    fraction.GetYaxis().SetTitleOffset(0.65)
    fraction.GetYaxis().SetNdivisions(505)
    fraction.SetMinimum(0.0)
    fraction.SetMaximum(1.05)
    fraction.Draw("E1 X0")
    retained.extend([upper, lower, fraction])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument(
        "--selection", required=True,
        choices=("coverage_loose", "coverage_tight"),
    )
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    hist_config = common.load_yaml(args.hist_config)
    config = common.load_yaml(args.region_config)
    validate_region_config(config)
    features = configured_v5_features(hist_config, config)
    selections = configured_selections(config)
    diagnostics = configured_count_diagnostics(config, features, selections)
    if not diagnostics:
        raise ValueError("V7 configuration has no count_diagnostics")
    definition = selections[args.selection]

    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if not bool(int(metadata(source, "uses_count_diagnostics"))):
        raise RuntimeError("Histogram input does not contain count diagnostics")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("v7_counts", "v7_counts", 1400, 900)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(canvas, source, args.selection, definition, diagnostics)
        canvas.Print(temporary)
        page_count += 1

        overview = diagnostics["overview_features"]
        draw_overview(
            canvas,
            source,
            args.selection,
            overview,
            diagnostics["features"],
            "full-selection raw counts (selection-sculpted)",
            lambda feature_name, tag: full_selection_histograms(
                source, args.selection, feature_name, tag
            ),
            "Full-selection candidate",
            "Topology failure",
        )
        canvas.Print(temporary)
        page_count += 1

        for family_name, family in diagnostics["families"].items():
            draw_overview(
                canvas,
                source,
                args.selection,
                family["features"],
                diagnostics["features"],
                f"{family['label']} detector-family N-minus-one raw counts",
                lambda feature_name, tag, family_name=family_name: (
                    nminus1_histograms(
                        source,
                        args.selection,
                        family_name,
                        feature_name,
                        tag,
                    )
                ),
                "Pass family requirement",
                "Fail family requirement",
            )
            canvas.Print(temporary)
            page_count += 1

        for family_name, family in diagnostics["families"].items():
            for feature_name in family["features"]:
                draw_detail(
                    canvas,
                    source,
                    args.selection,
                    family_name,
                    family,
                    feature_name,
                    diagnostics["features"][feature_name],
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
    print(f"Wrote {page_count} raw-count/N-minus-one pages to {output}")


if __name__ == "__main__":
    main()
