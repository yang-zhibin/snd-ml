#!/usr/bin/env python3
"""Plot one combined Run-8329/pooled-MC PDF for a V6 coverage point."""

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

CURVES = (
    ("data", "candidate", "Run 8329 candidate", ROOT.kBlue + 1, 20, 1),
    ("data", "complement", "Run 8329 topology failure", ROOT.kRed + 1, 21, 1),
    ("mc", "candidate", "Combined MC candidate", ROOT.kCyan + 2, 24, 2),
    ("mc", "complement", "Combined MC topology failure", ROOT.kOrange + 7, 25, 2),
)


def metadata(source, name):
    return common.object_value(source, name)


def selection_metadata(source, selection, suffix):
    return metadata(source, f"selection__{selection}__{suffix}")


def histogram_path(selection, feature, sample, population):
    if sample == "data":
        return (
            f"selections/{selection}/features/{feature}/"
            f"{POPULATIONS[population]}"
        )
    return (
        f"mc_muon_up/all/selections/{selection}/features/{feature}/"
        f"{population}"
    )


def selected_histograms(source, selection, feature):
    result = []
    for sample, population, label, color, marker, line_style in CURVES:
        histogram = common.clone_histogram(
            source,
            histogram_path(selection, feature, sample, population),
            f"v6_{selection}_{feature}_{sample}_{population}",
        )
        count = common.normalize(histogram)
        result.append({
            "histogram": histogram,
            "count": count,
            "label": label,
            "color": color,
            "marker": marker,
            "line_style": line_style,
        })
    return result


def draw_header(selection, text):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.10, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, "Run 8329 and pooled MC muon up")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    label.DrawLatex(0.10, 0.925, f"{selection}: {text}")
    return label


def draw_summary(canvas, source, config, selection, definition):
    real_total = int(metadata(source, "real_data_dq_count"))
    real_candidate = int(selection_metadata(source, selection, "candidate_count"))
    real_complement = int(selection_metadata(source, selection, "complement_count"))
    mc_total = int(metadata(source, "mc_muon_up__all__baseline_count"))
    mc_candidate = int(selection_metadata(
        source, selection, "mc__all__candidate_count"
    ))
    mc_complement = int(selection_metadata(
        source, selection, "mc__all__complement_count"
    ))
    components = definition["components"]

    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.040)
    text.DrawLatex(0.08, 0.86, definition["label"])
    text.SetTextSize(0.024)
    lines = [
        f"Run 8329 RealDataDQ: {real_total:,}",
        f"  candidate: {real_candidate:,} "
        f"({100 * real_candidate / real_total if real_total else 0:.5f}%)",
        f"  topology failure: {real_complement:,}",
        f"Combined MC baseline: {mc_total:,}",
        f"  candidate: {mc_candidate:,} "
        f"({100 * mc_candidate / mc_total if mc_total else 0:.5f}%)",
        f"  topology failure: {mc_complement:,}",
        "MC aggregation: unweighted event sum of all four productions",
        "MC reconstructed tracks: not used and not plotted",
        "",
        "Detector requirements:",
    ]
    for name, expression in components.items():
        lines.extend(textwrap.wrap(
            f"{name}: {expression}", width=105,
            subsequent_indent="    ", break_long_words=False,
        ))
    y = 0.80
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.033 if line else 0.016
    header = draw_header(selection, "combined data/MC detector selection")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def style_curve(item):
    histogram = item["histogram"]
    histogram.SetLineColor(item["color"])
    histogram.SetMarkerColor(item["color"])
    histogram.SetMarkerStyle(item["marker"])
    histogram.SetMarkerSize(0.45)
    histogram.SetLineWidth(2)
    histogram.SetLineStyle(item["line_style"])


def ratio(numerator, denominator, name):
    result = numerator.Clone(name)
    result.SetDirectory(0)
    result.Divide(denominator)
    return result


def draw_1d(canvas, source, selection, feature_name, feature):
    items = selected_histograms(source, selection, feature_name)
    histograms = [item["histogram"] for item in items]
    for item in items:
        style_curve(item)

    canvas.Clear()
    canvas.cd()
    upper = ROOT.TPad(f"upper_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetLeftMargin(0.14)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.13)
    upper.SetBottomMargin(0.025)
    upper.SetLogy(bool(feature.get("logy", False)))
    lower.SetLeftMargin(0.14)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.36)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()

    upper.cd()
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = common.positive_minimum_1d(histograms)
    first = histograms[0]
    first.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    first.GetXaxis().SetLabelSize(0)
    if bool(feature.get("logy", False)):
        first.SetMinimum(max((positive or 1e-9) * 0.5, 1e-9))
        first.SetMaximum(max(maximum * 20, 1e-8))
    else:
        first.SetMinimum(0)
        first.SetMaximum(max(maximum * 1.35, 1e-8))
    first.Draw("E1 X0")
    for histogram in histograms[1:]:
        histogram.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.48, 0.69, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    for item in items:
        legend.AddEntry(
            item["histogram"],
            f"{item['label']} (N={common.format_count(item['count'])})",
            "lep",
        )
    legend.Draw()
    header = draw_header(selection, feature["axis_title"])

    lower.cd()
    ratios = (
        ratio(histograms[2], histograms[0], f"candidate_ratio_{feature_name}"),
        ratio(histograms[3], histograms[1], f"complement_ratio_{feature_name}"),
    )
    for result, color, marker in (
        (ratios[0], CURVES[2][3], CURVES[2][4]),
        (ratios[1], CURVES[3][3], CURVES[3][4]),
    ):
        result.SetLineColor(color)
        result.SetMarkerColor(color)
        result.SetMarkerStyle(marker)
        result.SetLineWidth(2)
    ratios[0].SetTitle(f";{feature['axis_title']};MC / data")
    ratios[0].GetXaxis().SetTitleSize(0.13)
    ratios[0].GetXaxis().SetLabelSize(0.10)
    ratios[0].GetXaxis().SetTitleOffset(1.15)
    ratios[0].GetYaxis().SetTitleSize(0.095)
    ratios[0].GetYaxis().SetLabelSize(0.085)
    ratios[0].GetYaxis().SetTitleOffset(0.65)
    ratios[0].GetYaxis().SetNdivisions(505)
    ratios[0].SetMinimum(0)
    ratios[0].SetMaximum(5)
    ratios[0].Draw("E1 X0")
    ratios[1].Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    ratio_legend = ROOT.TLegend(0.68, 0.72, 0.94, 0.94)
    ratio_legend.SetBorderSize(0)
    ratio_legend.SetFillStyle(0)
    ratio_legend.SetTextSize(0.075)
    ratio_legend.AddEntry(ratios[0], "candidate", "lep")
    ratio_legend.AddEntry(ratios[1], "topology failure", "lep")
    ratio_legend.Draw()
    canvas._keepalive = [
        *histograms, *ratios, upper, lower, legend, ratio_legend, header, unity,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, selection, feature_name, feature):
    items = selected_histograms(source, selection, feature_name)
    histograms = [item["histogram"] for item in items]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = common.positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    minimum = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0

    canvas.Clear()
    canvas.Divide(2, 2, 0.002, 0.002)
    retained = list(histograms)
    for index, item in enumerate(items, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.16)
        pad.SetLogz(log_z)
        histogram = item["histogram"]
        histogram.SetMinimum(minimum)
        histogram.SetMaximum(max(maximum * (2 if log_z else 1.05), 1e-12))
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Fraction of selected events"
        )
        histogram.Draw("COLZ")
        title = ROOT.TLatex()
        title.SetNDC(True)
        title.SetTextFont(42)
        title.SetTextSize(0.027)
        title.DrawLatex(
            0.13, 0.965,
            f"{item['label']} (N={common.format_count(item['count'])})",
        )
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
    definition = selections[args.selection]
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("v6", "v6", 1200, 850)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(canvas, source, config, args.selection, definition)
        canvas.Print(temporary)
        page_count += 1
        for feature_name, feature in features.items():
            if int(feature.get("dimension", 1)) == 1:
                draw_1d(canvas, source, args.selection, feature_name, feature)
            else:
                draw_2d(canvas, source, args.selection, feature_name, feature)
            canvas.Print(temporary)
            page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} combined data/MC pages to {output}")


if __name__ == "__main__":
    main()
