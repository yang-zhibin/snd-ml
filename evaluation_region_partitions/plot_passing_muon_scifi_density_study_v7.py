#!/usr/bin/env python3
"""Plot data/MC distributions for one V7 loose-failure SciFi-density category."""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from make_passing_muon_binary_region_hist_v5 import (
    configured_v5_features,
    validate_region_config,
)
from passing_muon_v7_scifi_density import (
    category_expression,
    configured_density_categories,
    configured_density_selection,
)


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

CURVES = (
    ("data", "Run 8329 RealDataDQ", ROOT.kBlue + 1, 20, 1),
    ("mc", "Pooled MC muon up", ROOT.kOrange + 7, 24, 2),
)


def metadata(source, name):
    return common.object_value(source, name)


def selected_histograms(source, category, feature):
    result = []
    for sample, label, color, marker, line_style in CURVES:
        histogram = common.clone_histogram(
            source,
            f"categories/{category}/features/{feature}/{sample}",
            f"density_{category}_{feature}_{sample}",
        )
        plotted_count = common.normalize(histogram)
        result.append({
            "sample": sample,
            "histogram": histogram,
            "count": plotted_count,
            "label": label,
            "color": color,
            "marker": marker,
            "line_style": line_style,
        })
    return result


def style_curve(item):
    histogram = item["histogram"]
    histogram.SetLineColor(item["color"])
    histogram.SetMarkerColor(item["color"])
    histogram.SetMarkerStyle(item["marker"])
    histogram.SetMarkerSize(0.45)
    histogram.SetLineWidth(2)
    histogram.SetLineStyle(item["line_style"])


def draw_header(category, title):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.10, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, "Run 8329 and pooled MC muon up")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    label.DrawLatex(0.10, 0.925, f"{category}: {title}")
    return label


def draw_summary(canvas, source, category_key, category, selection, study):
    prefix = f"category__{category_key}"
    data_count = int(metadata(source, f"{prefix}__data_count"))
    mc_count = int(metadata(source, f"{prefix}__mc_count"))
    data_veto_counts = [
        int(metadata(source, f"{prefix}__data__{key}__count"))
        for key in selection["veto_categories"]
    ]
    if sum(data_veto_counts) != data_count:
        raise RuntimeError(f"Data Veto-union closure failed for {category_key}")

    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.040)
    text.DrawLatex(0.08, 0.86, "Loose-failure SciFi-density distribution")
    text.SetTextSize(0.024)
    lines = [
        str(category["label"]),
        f"Run 8329 selected events: {data_count:,}",
        f"Pooled MC selected events: {mc_count:,}",
    ]
    for production, definition in study["mc_muon_up"]["productions"].items():
        count = int(metadata(source, f"{prefix}__mc__{production}__count"))
        lines.append(f"  {definition['label']}: {count:,}")
    lines.extend([
        "",
        "The Veto <3, =3, and >3 groups are combined into one population.",
        "The four density thresholds are cumulative, not exclusive bins.",
        "Histograms use the same features, binning, validity cuts, and log settings",
        "as coverage_loose.pdf. Each data/MC curve is normalized independently.",
        "Reconstructed-track information is not used.",
        "",
        "Selection expression:",
    ])
    lines.extend(textwrap.wrap(
        category_expression(selection, category),
        width=105,
        subsequent_indent="    ",
        break_long_words=False,
    ))
    y = 0.80
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.034 if line else 0.017
    header = draw_header(category_key, "selection summary")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def ratio(numerator, denominator, name):
    result = numerator.Clone(name)
    result.SetDirectory(0)
    result.Divide(denominator)
    return result


def draw_1d(canvas, source, category, feature_name, feature):
    items = selected_histograms(source, category, feature_name)
    for item in items:
        style_curve(item)
    histograms = [item["histogram"] for item in items]

    canvas.Clear()
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
    histograms[1].Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.48, 0.72, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    for item in items:
        legend.AddEntry(
            item["histogram"],
            f"{item['label']} (plotted N={common.format_count(item['count'])})",
            "lep",
        )
    legend.Draw()
    header = draw_header(category, feature["axis_title"])

    lower.cd()
    comparison = ratio(
        histograms[1], histograms[0], f"mc_data_ratio_{category}_{feature_name}"
    )
    comparison.SetLineColor(CURVES[1][2])
    comparison.SetMarkerColor(CURVES[1][2])
    comparison.SetMarkerStyle(CURVES[1][3])
    comparison.SetLineWidth(2)
    comparison.SetTitle(f";{feature['axis_title']};MC / data")
    comparison.GetXaxis().SetTitleSize(0.13)
    comparison.GetXaxis().SetLabelSize(0.10)
    comparison.GetXaxis().SetTitleOffset(1.15)
    comparison.GetYaxis().SetTitleSize(0.095)
    comparison.GetYaxis().SetLabelSize(0.085)
    comparison.GetYaxis().SetTitleOffset(0.65)
    comparison.GetYaxis().SetNdivisions(505)
    comparison.SetMinimum(0)
    comparison.SetMaximum(5)
    comparison.Draw("E1 X0")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive = [
        upper, lower, *histograms, comparison, unity, legend, header,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, category, feature_name, feature):
    items = selected_histograms(source, category, feature_name)
    histograms = [item["histogram"] for item in items]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = common.positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    minimum = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0

    canvas.Clear()
    canvas.Divide(2, 1, 0.002, 0.002)
    retained = list(histograms)
    for index, item in enumerate(items, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
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
            0.14, 0.965,
            f"{item['label']} (plotted N={common.format_count(item['count'])})",
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
    parser.add_argument("--event-display-config", required=True)
    parser.add_argument("--mc-event-display-config", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    hist_config = common.load_yaml(args.hist_config)
    study = common.load_yaml(args.region_config)
    data_display = common.load_yaml(args.event_display_config)
    mc_display = common.load_yaml(args.mc_event_display_config)
    validate_region_config(study)
    features = configured_v5_features(hist_config, study)
    categories = configured_density_categories(data_display, mc_display)
    selection = configured_density_selection(study, data_display, mc_display)
    if args.category not in categories:
        raise ValueError(
            f"Unknown SciFi-density category {args.category!r}; "
            f"expected one of {list(categories)}"
        )

    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    expected_features = str(metadata(source, "feature_order")).split(",")
    if expected_features != list(features):
        source.Close()
        raise RuntimeError("Histogram feature order does not match current configuration")
    expected_categories = str(metadata(source, "category_order")).split(",")
    if expected_categories != list(categories):
        source.Close()
        raise RuntimeError("Histogram category order does not match current configuration")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("density_v7", "density_v7", 1200, 850)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(
            canvas, source, args.category, categories[args.category], selection, study
        )
        canvas.Print(temporary)
        page_count += 1
        for feature_name, feature in features.items():
            if int(feature.get("dimension", 1)) == 1:
                draw_1d(canvas, source, args.category, feature_name, feature)
            else:
                draw_2d(canvas, source, args.category, feature_name, feature)
            canvas.Print(temporary)
            page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} SciFi-density pages to {output}")


if __name__ == "__main__":
    main()
