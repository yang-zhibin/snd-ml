#!/usr/bin/env python3
"""Plot matched veto-tagged transforms against the Run-8329 signal region."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import ROOT

from make_muon_dis_transform_comparison_hist_shard import (
    COUNTER_LABELS,
    VETO_TAGGED_POPULATIONS,
    load_yaml,
)
from make_passing_muon_binary_region_hist_v5 import configured_v5_features
from plot_passing_muon_region_study_v4 import (
    clone_histogram,
    format_count,
    histogram_integral,
    object_value,
    positive_minimum_1d,
    positive_minimum_2d,
)


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetEndErrorSize(0)
ROOT.gStyle.SetPaperSize(29.7, 21.0)

CURVES = (
    ("original", ROOT.kBlack, 20, 1),
    ("transformed", ROOT.kRed + 1, 21, 1),
    ("signal_region", ROOT.kAzure + 2, 22, 2),
)


def normalize(histogram) -> float:
    count = histogram_integral(histogram)
    if count > 0:
        histogram.Scale(1.0 / count)
    return count


def curve_label(config: dict, version: str) -> str:
    keys = {
        "original": "original_label",
        "transformed": "transformed_label",
        "signal_region": "signal_label",
    }
    return str(config["plot"][keys[version]])


def population_label(config: dict, population: str) -> str:
    return str(config["plot"]["populations"][population])


def histogram_path(method: str, population: str, feature: str, version: str) -> str:
    if version == "signal_region":
        return f"reference/signal_region/features/{feature}/original"
    return (
        f"methods/{method}/populations/{population}/features/{feature}/{version}"
    )


def feature_histograms(source, config, method: str, population: str, feature: str):
    result = []
    for version, color, marker, line_style in CURVES:
        histogram = clone_histogram(
            source,
            histogram_path(method, population, feature, version),
            f"{method}_{population}_{feature}_{version}",
        )
        count = normalize(histogram)
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.42)
        histogram.SetLineStyle(line_style)
        histogram.SetLineWidth(2)
        result.append({
            "version": version,
            "label": curve_label(config, version),
            "histogram": histogram,
            "count": count,
            "color": color,
            "marker": marker,
        })
    return result


def counter_values(counter) -> dict[str, int]:
    observed = {
        str(counter.GetXaxis().GetBinLabel(index)): int(
            round(counter.GetBinContent(index))
        )
        for index in range(1, counter.GetNbinsX() + 1)
    }
    missing = set(COUNTER_LABELS) - set(observed)
    if missing:
        raise RuntimeError(f"Counter is missing labels {sorted(missing)}")
    return observed


def draw_summary(canvas, source, config: dict, method: str) -> None:
    counter = clone_histogram(source, f"methods/{method}/counts", f"{method}_counts")
    counts = counter_values(counter)
    method_label = str(config["methods"][method]["label"])
    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.039)
    text.DrawLatex(0.07, 0.92, "Veto-tagged detector-ID transform comparison")
    text.SetTextSize(0.027)
    lines = [
        method_label,
        "",
        f"All RealDataDQ events: {counts['all RealDataDQ']:,}",
        f"Original no-Veto signal region: {counts['no-Veto signal region']:,}",
        "",
    ]
    for prefix, title in (
        ("passing-muon", "Passing-muon candidate (passes v7 loose)"),
        ("Muon-DIS", "Muon-DIS candidate (fails v7 loose)"),
    ):
        total = counts[f"{prefix} candidate"]
        eligible = counts[f"{prefix} eligible"]
        matched = counts[f"{prefix} matched"]
        ineligible = counts[f"{prefix} ineligible"]
        missing = counts[f"{prefix} eligible missing"]
        lines.extend([
            title,
            f"  total {total:,}; eligible {eligible:,}; matched {matched:,}",
            f"  ineligible {ineligible:,}; eligible but missing {missing:,}; "
            f"matched fraction {100.0 * matched / total if total else 0.0:.3f}%",
            "",
        ])
    lines.extend([
        "Population labels are frozen from the original event.",
        "Original and transformed curves use identical (runId, eventId) pairs.",
        "Signal-region curves use all original events with no veto hit.",
        "Shapes are normalized independently after paired feature-validity cuts.",
    ])
    y = 0.855
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.042 if line else 0.020
    footer = ROOT.TLatex()
    footer.SetNDC(True)
    footer.SetTextFont(42)
    footer.SetTextSize(0.026)
    footer.DrawLatex(0.07, 0.035, "SND@LHC work in progress")
    footer.SetTextAlign(31)
    footer.DrawLatex(0.94, 0.035, "Run 8329, real-data DQ")
    canvas._keepalive = [counter, text, footer]
    canvas.Modified()
    canvas.Update()


def safe_ratio(numerator, denominator, name: str):
    ratio = numerator.Clone(name)
    ratio.SetDirectory(0)
    ratio.Reset("ICES")
    for index in range(1, denominator.GetNbinsX() + 1):
        den = float(denominator.GetBinContent(index))
        num = float(numerator.GetBinContent(index))
        if den <= 0:
            continue
        ratio.SetBinContent(index, num / den)
        num_error = float(numerator.GetBinError(index))
        den_error = float(denominator.GetBinError(index))
        variance = (num_error / den) ** 2 + (num * den_error / den**2) ** 2
        ratio.SetBinError(index, math.sqrt(max(variance, 0.0)))
    return ratio


def draw_1d_column(
    parent, source, config, method: str, population: str, feature_name: str, feature: dict
):
    parent.cd()
    upper = ROOT.TPad(f"upper_{method}_{population}_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{method}_{population}_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetLeftMargin(0.19)
    upper.SetRightMargin(0.04)
    upper.SetTopMargin(0.14)
    upper.SetBottomMargin(0.025)
    upper.SetLogy(bool(feature.get("logy", False)))
    lower.SetLeftMargin(0.19)
    lower.SetRightMargin(0.04)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.34)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()

    items = feature_histograms(source, config, method, population, feature_name)
    histograms = [item["histogram"] for item in items]
    upper.cd()
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_1d(histograms)
    first = histograms[0]
    first.SetTitle(f";{feature['axis_title']};Fraction of feature-valid events")
    first.GetXaxis().SetLabelSize(0)
    if upper.GetLogy():
        first.SetMinimum(max((positive or 1e-9) * 0.5, 1e-9))
        first.SetMaximum(max(maximum * 30.0, 1e-8))
    else:
        first.SetMinimum(0.0)
        first.SetMaximum(max(maximum * 1.45, 1e-8))
    first.Draw("E1 X0")
    for histogram in histograms[1:]:
        histogram.Draw("E1 X0 SAME")

    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.035)
    title.DrawLatex(0.19, 0.965, population_label(config, population))
    legend = ROOT.TLegend(0.42, 0.67, 0.95, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    for item in items:
        legend.AddEntry(
            item["histogram"],
            f"{item['label']} (N={format_count(item['count'])})",
            "lep",
        )
    legend.Draw()

    lower.cd()
    ratios = []
    for item in items[1:]:
        ratio = safe_ratio(
            item["histogram"], items[0]["histogram"],
            f"ratio_{method}_{population}_{feature_name}_{item['version']}",
        )
        ratio.SetLineColor(item["color"])
        ratio.SetMarkerColor(item["color"])
        ratio.SetMarkerStyle(item["marker"])
        ratio.SetMarkerSize(0.38)
        ratio.SetLineWidth(2)
        ratios.append(ratio)
    ratios[0].SetTitle(f";{feature['axis_title']};Ratio to original")
    ratios[0].GetXaxis().SetTitleSize(0.12)
    ratios[0].GetXaxis().SetLabelSize(0.09)
    ratios[0].GetXaxis().SetTitleOffset(1.10)
    ratios[0].GetYaxis().SetTitleSize(0.085)
    ratios[0].GetYaxis().SetLabelSize(0.075)
    ratios[0].GetYaxis().SetTitleOffset(0.72)
    ratios[0].GetYaxis().SetNdivisions(505)
    ratios[0].SetMinimum(float(config["plot"].get("ratio_min", 0.0)))
    ratios[0].SetMaximum(float(config["plot"].get("ratio_max", 3.0)))
    ratios[0].Draw("E1 X0")
    ratios[1].Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1.0, float(feature["x_max"]), 1.0)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    ratio_legend = ROOT.TLegend(0.49, 0.68, 0.95, 0.94)
    ratio_legend.SetBorderSize(0)
    ratio_legend.SetFillStyle(0)
    ratio_legend.SetTextSize(0.059)
    ratio_legend.AddEntry(ratios[0], "Transformed / original", "lep")
    ratio_legend.AddEntry(ratios[1], "Signal / original", "lep")
    ratio_legend.Draw()
    return [
        *histograms, *ratios, upper, lower, title, legend, ratio_legend, unity,
    ]


def draw_1d(canvas, source, config, method, feature_name, feature):
    canvas.Clear()
    canvas.Divide(2, 1, 0.002, 0.002)
    retained = []
    for index, population in enumerate(VETO_TAGGED_POPULATIONS, 1):
        retained.extend(
            draw_1d_column(
                canvas.cd(index), source, config, method, population,
                feature_name, feature,
            )
        )
    canvas.cd(0)
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, config, method, feature_name, feature):
    population_items = {
        population: feature_histograms(
            source, config, method, population, feature_name
        )
        for population in VETO_TAGGED_POPULATIONS
    }
    all_histograms = [
        item["histogram"]
        for items in population_items.values()
        for item in items
    ]
    maximum = max((histogram.GetMaximum() for histogram in all_histograms), default=1.0)
    positive = positive_minimum_2d(all_histograms)
    log_z = bool(feature.get("logz", False))
    minimum = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    canvas.Clear()
    canvas.Divide(3, 2, 0.002, 0.002)
    retained = list(all_histograms)
    for row, population in enumerate(VETO_TAGGED_POPULATIONS):
        for column, item in enumerate(population_items[population]):
            pad = canvas.cd(row * 3 + column + 1)
            pad.SetLeftMargin(0.14)
            pad.SetRightMargin(0.17)
            pad.SetBottomMargin(0.14)
            pad.SetTopMargin(0.20)
            pad.SetLogz(log_z)
            histogram = item["histogram"]
            histogram.SetMinimum(minimum)
            histogram.SetMaximum(max(maximum * (2.0 if log_z else 1.05), 1e-12))
            histogram.SetTitle(
                f";{feature['x_title']};{feature['y_title']};"
                "Fraction of feature-valid events"
            )
            histogram.Draw("COLZ")
            title = ROOT.TLatex()
            title.SetNDC(True)
            title.SetTextFont(42)
            title.SetTextSize(0.027)
            title.DrawLatex(0.14, 0.970, population_label(config, population))
            title.SetTextSize(0.023)
            title.DrawLatex(0.14, 0.925, item["label"])
            title.DrawLatex(0.14, 0.885, f"N={format_count(item['count'])}")
            retained.extend([pad, title])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("reco_scifi", "simple"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    v7_config = load_yaml(config["inputs"]["v7_config"])
    histogram_config = load_yaml(config["inputs"]["feature_config"])
    features = configured_v5_features(histogram_config, v7_config)
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if str(object_value(source, "study_version")) != str(config["version"]):
        raise RuntimeError("Input ROOT file is not the configured transform study")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas(f"comparison_{args.method}", "", 1550, 880)
    canvas.SetCanvasSize(1550, 880)
    page_count = 0
    try:
        canvas.Print(temporary + "[")
        draw_summary(canvas, source, config, args.method)
        canvas.Print(temporary)
        page_count += 1
        for feature_name, feature in features.items():
            if int(feature.get("dimension", 1)) == 1:
                draw_1d(canvas, source, config, args.method, feature_name, feature)
                canvas.Print(temporary)
            else:
                wide = ROOT.TCanvas(
                    f"wide_{args.method}_{feature_name}", "", 1650, 1000
                )
                wide.SetCanvasSize(1650, 1000)
                draw_2d(wide, source, config, args.method, feature_name, feature)
                wide.Print(temporary)
                wide.Close()
            page_count += 1
        canvas.Print(temporary + "]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} pages to {output}")


if __name__ == "__main__":
    main()
