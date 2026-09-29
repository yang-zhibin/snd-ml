#!/usr/bin/env python3
"""Render one run-8329 target-type or all-track-types multi-page PDF."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml


TARGET_COLORS = (ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kBlack)
ALGORITHM_COLORS = (
    ROOT.kGray + 2,
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kBlack,
)
ALGORITHM_COMBINATION_ORDER = (
    "sf0_ds0",
    "sf1_ds0",
    "sf0_ds1",
    "sf1_ds1",
)
ALGORITHM_COMBINATION_LABELS = {
    "sf0_ds0": "SciFi=0, DS=0",
    "sf1_ds0": "SciFi=1, DS=0",
    "sf0_ds1": "SciFi=0, DS=1",
    "sf1_ds1": "SciFi=1, DS=1",
}
TYPE_COLORS = (
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kCyan + 2,
    ROOT.kViolet + 1,
)
REFERENCE_COLORS = (
    ROOT.kBlack,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kRed + 1,
)
REFERENCE_MARKERS = (20, 21, 22, 23)
REFERENCE_KEYS = (
    "mc_muon_up",
    "real_data_dq",
    "passing_muon_reco",
    "no_passing_muon_reco",
)
REFERENCE_LABELS = (
    "MC muon up, detector activity",
    "RealDataDQ",
    "PassingMuonReco",
    "NoPassingMuonReco",
)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def object_value(source, name):
    value = source.Get(name)
    if not value:
        raise RuntimeError(f"Missing ROOT metadata object {name}")
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def clone_histogram(source, path, suffix):
    histogram = source.Get(path)
    if not histogram:
        raise RuntimeError(f"Missing histogram {path}")
    result = histogram.Clone(f"{histogram.GetName()}__{suffix}")
    result.SetDirectory(0)
    return result


def histogram_integral(histogram):
    if histogram.GetDimension() == 1:
        return float(histogram.Integral(1, histogram.GetNbinsX()))
    return float(
        histogram.Integral(1, histogram.GetNbinsX(), 1, histogram.GetNbinsY())
    )


def normalize(histogram):
    integral = histogram_integral(histogram)
    if integral > 0:
        histogram.Scale(1.0 / integral)
    return integral


def format_count(value):
    """Use compact scientific notation for event counts on crowded plots."""
    return f"{float(value):.3e}"


def conditional_overlap_percent(intersection, row_count):
    """Return the percentage of the row category also in the column category."""
    return 100.0 * float(intersection) / float(row_count) if row_count else 0.0


def quality_features(config):
    quality = config["track_quality_2d"]
    score = quality["score"]
    result = {}
    for name, observable in quality["observables"].items():
        result[name] = {
            "x_min": float(score["min"]),
            "x_max": float(score["max"]),
            "x_title": "Best track #chi^{2}/NDF",
            "y_title": str(observable["axis_title"]),
            "logz": bool(observable.get("logz", True)),
        }
    return result


def quality_score_title(track):
    if len(track["components"]) > 1:
        return "Maximum component best #chi^{2}/NDF"
    return "Best track #chi^{2}/NDF"


def positive_minimum_1d(histograms):
    values = [
        histogram.GetBinContent(index)
        for histogram in histograms
        for index in range(1, histogram.GetNbinsX() + 1)
        if histogram.GetBinContent(index) > 0
    ]
    return min(values) if values else None


def positive_minimum_2d(histograms):
    values = [
        histogram.GetBinContent(ix, iy)
        for histogram in histograms
        for ix in range(1, histogram.GetNbinsX() + 1)
        for iy in range(1, histogram.GetNbinsY() + 1)
        if histogram.GetBinContent(ix, iy) > 0
    ]
    return min(values) if values else None


def draw_header(text, right="Run 8329, real-data DQ"):
    left = ROOT.TLatex()
    left.SetNDC(True)
    left.SetTextFont(42)
    left.SetTextSize(0.030)
    left.DrawLatex(0.12, 0.965, "SND@LHC work in progress")
    left.SetTextAlign(31)
    left.DrawLatex(0.94, 0.965, right)
    left.SetTextAlign(13)
    left.SetTextSize(0.026)
    left.DrawLatex(0.12, 0.925, text)
    return left


def target_histograms(source, target, feature_name):
    return [
        clone_histogram(
            source, f"targets/{target}/{feature_name}/has_target", "has_target"
        ),
        clone_histogram(
            source, f"targets/{target}/{feature_name}/other_passing", "other"
        ),
        clone_histogram(
            source, f"missing/{feature_name}/no_passing_muon_reco", "missing"
        ),
    ]


def draw_target_summary(canvas, source, config, target):
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    track = config["track_types"][target]
    passing = int(object_value(source, "passing_count"))
    missing = int(object_value(source, "missing_count"))
    real = int(object_value(source, "real_data_dq_count"))
    has_target = int(object_value(source, f"{target}__has_target_count"))
    other = int(object_value(source, f"{target}__other_passing_count"))
    converged = int(object_value(source, f"{target}__has_converged_count"))
    best = int(object_value(source, f"{target}__has_best_count"))
    unscorable = int(object_value(source, f"{target}__unscorable_count"))
    expression = str(object_value(source, f"{target}__target_expression"))
    working_point = str(object_value(source, "working_point"))
    working_label = str(object_value(source, "working_point_label"))

    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.045)
    title.DrawLatex(0.10, 0.855, f"Target study: {track['label']}")
    title.SetTextSize(0.032)
    lines = [
        f"Working point: {working_point}",
        working_label,
        f"Expression: {expression}",
        "",
        f"HasTargetTrack: {has_target:,} ({100.0 * has_target / real:.4f}% of DQ)",
        f"OtherPassingReco: {other:,} ({100.0 * other / real:.4f}% of DQ)",
        f"NoPassingMuonReco: {missing:,} ({100.0 * missing / real:.4f}% of DQ)",
        "",
        f"PassingMuonReco total: {passing:,}",
        f"RealDataDQ total: {real:,}",
        f"Converged target type: {converged:,}",
        f"Valid best target type: {best:,}",
        f"Converged but unscorable: {unscorable:,}",
        "",
        f"Closure: {has_target:,} + {other:,} + {missing:,} = {real:,}",
    ]
    y = 0.785
    for line in lines:
        title.DrawLatex(0.12, y, line)
        y -= 0.045 if line else 0.025
    header = draw_header("Three disjoint event populations")
    canvas._keepalive.extend([title, header])
    canvas.Modified()
    canvas.Update()


def draw_target_1d(canvas, source, config, target, feature_name, feature):
    histograms = target_histograms(source, target, feature_name)
    if any(histogram.GetDimension() != 1 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not one-dimensional")
    raw_integrals = [normalize(histogram) for histogram in histograms]
    labels = (
        f"Has target track (valid N={format_count(raw_integrals[0])})",
        f"Other passing reco (valid N={format_count(raw_integrals[1])})",
        f"No PassingMuonReco (valid N={format_count(raw_integrals[2])})",
    )
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"top_{target}_{feature_name}", "", 0.0, 0.30, 1.0, 1.0)
    bottom = ROOT.TPad(f"ratio_{target}_{feature_name}", "", 0.0, 0.0, 1.0, 0.30)
    canvas._keepalive.extend([top, bottom])
    top.SetLeftMargin(0.15)
    top.SetRightMargin(0.05)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    bottom.SetLeftMargin(0.15)
    bottom.SetRightMargin(0.05)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    log_y = bool(feature.get("logy", False))
    top.SetLogy(log_y)
    top.Draw()
    bottom.Draw()

    markers = (20, 24, 25)
    for histogram, color, marker in zip(histograms, TARGET_COLORS, markers):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.65)
        histogram.SetLineWidth(2)
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_1d(histograms)
    reference = histograms[0]
    if log_y:
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10.0, 1e-6))
    else:
        reference.SetMinimum(0.0)
        reference.SetMaximum(max(maximum * 1.40, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    reference.GetXaxis().SetLabelSize(0.0)
    reference.GetXaxis().SetTitleSize(0.0)
    reference.GetYaxis().SetTitleOffset(1.25)

    top.cd()
    reference.Draw("E1 X0")
    histograms[1].Draw("E1 X0 SAME")
    histograms[2].Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.54, 0.67, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    for histogram, label in zip(histograms, labels):
        legend.AddEntry(histogram, label, "lep")
    legend.Draw()
    track_label = config["track_types"][target]["label"]
    working_label = str(object_value(source, "working_point_label"))
    header = draw_header(f"{track_label}; {working_label}")
    feature_label = ROOT.TLatex()
    feature_label.SetNDC(True)
    feature_label.SetTextFont(42)
    feature_label.SetTextSize(0.035)
    feature_label.DrawLatex(0.14, 0.86, feature_name.replace("_", " "))
    canvas._keepalive.extend([legend, header, feature_label])

    bottom.cd()
    ratio_frame = reference.Clone(f"ratio_frame_{target}_{feature_name}")
    ratio_frame.Reset("ICES")
    ratio_frame.SetTitle("")
    ratio_frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    ratio_frame.GetYaxis().SetTitle("Normalized shape ratio")
    ratio_frame.GetYaxis().SetTitleSize(0.065)
    ratio_frame.GetYaxis().SetTitleOffset(0.90)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetXaxis().SetTitleSize(0.11)
    ratio_frame.GetXaxis().SetLabelSize(0.09)
    ratio_frame.GetXaxis().SetTitleOffset(1.0)
    ratio_frame.GetXaxis().CenterTitle(True)
    ratio_frame.GetYaxis().CenterTitle(True)
    ratios = []
    ratio_maximum = 1.0
    for index, numerator in enumerate(histograms[1:], start=1):
        ratio = numerator.Clone(f"shape_ratio_{target}_{feature_name}_{index}")
        ratio.Divide(reference)
        ratio.SetLineColor(TARGET_COLORS[index])
        ratio.SetMarkerColor(TARGET_COLORS[index])
        ratio.SetMarkerStyle(markers[index])
        ratio.SetMarkerSize(0.55)
        ratios.append(ratio)
        for bin_index in range(1, ratio.GetNbinsX() + 1):
            if reference.GetBinContent(bin_index) <= 0:
                ratio.SetBinContent(bin_index, 0.0)
                ratio.SetBinError(bin_index, 0.0)
            else:
                ratio_maximum = max(
                    ratio_maximum,
                    ratio.GetBinContent(bin_index) + ratio.GetBinError(bin_index),
                )
    ratio_frame.SetMinimum(0.0)
    ratio_frame.SetMaximum(min(max(2.0, 1.20 * ratio_maximum), 20.0))
    ratio_frame.Draw("AXIS")
    for ratio in ratios:
        ratio.Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1.0, float(feature["x_max"]), 1.0)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.extend([ratio_frame, *ratios, unity])
    canvas.Modified()
    canvas.Update()


def draw_target_2d(canvas, source, config, target, feature_name, feature):
    histograms = target_histograms(source, target, feature_name)
    if any(histogram.GetDimension() != 2 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not two-dimensional")
    raw_integrals = [normalize(histogram) for histogram in histograms]
    labels = ("Has target track", "Other passing reco", "No PassingMuonReco")
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(3, 1, 0.002, 0.002)
    for index, (label, histogram, count) in enumerate(
        zip(labels, histograms, raw_integrals), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Fraction of selected events"
        )
        histogram.GetXaxis().SetTitleOffset(1.15)
        histogram.GetYaxis().SetTitleOffset(1.35)
        histogram.GetZaxis().SetTitleOffset(1.25)
        histogram.Draw("COLZ")
        panel = ROOT.TLatex()
        panel.SetNDC(True)
        panel.SetTextFont(42)
        panel.SetTextSize(0.030)
        panel.DrawLatex(
            0.14, 0.965, f"{label} (valid N={format_count(count)})"
        )
        panel.SetTextSize(0.023)
        panel.DrawLatex(
            0.14, 0.925,
            f"{config['track_types'][target]['label']}; "
            f"{object_value(source, 'working_point')}",
        )
        canvas._keepalive.append(panel)
    canvas.Modified()
    canvas.Update()


def algorithm_histograms(source, algorithm, feature_name):
    histograms = [
        clone_histogram(
            source,
            "algorithm_comparisons/"
            f"{algorithm}/{combination}/{feature_name}/{combination}",
            f"{algorithm}_{combination}",
        )
        for combination in ALGORITHM_COMBINATION_ORDER
    ]
    histograms.append(
        clone_histogram(
            source,
            f"missing/{feature_name}/no_passing_muon_reco",
            f"{algorithm}_missing",
        )
    )
    return histograms


def draw_algorithm_summary(canvas, source, config, algorithm):
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    definition = config["algorithm_comparisons"][algorithm]
    passing = int(object_value(source, "passing_count"))
    missing = int(object_value(source, "missing_count"))
    real = int(object_value(source, "real_data_dq_count"))
    working_point = str(object_value(source, "working_point"))
    working_label = str(object_value(source, "working_point_label"))
    counts = {
        combination: int(
            object_value(
                source, f"algorithm__{algorithm}__{combination}__count"
            )
        )
        for combination in ALGORITHM_COMBINATION_ORDER
    }

    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.046)
    text.DrawLatex(0.09, 0.87, f"{definition['label']}: SciFi/DS combinations")
    text.SetTextSize(0.031)
    lines = [
        f"Working point: {working_point}",
        working_label,
        "1 = component passes the active working point; 0 = it does not pass",
        "",
    ]
    for combination in ALGORITHM_COMBINATION_ORDER:
        lines.append(
            f"{ALGORITHM_COMBINATION_LABELS[combination]}: "
            f"{counts[combination]:,} "
            f"({100.0 * counts[combination] / passing:.4f}% of passing)"
        )
    lines.extend(
        [
            "",
            f"NoPassingMuonReco: {missing:,} ({100.0 * missing / real:.4f}% of DQ)",
            f"PassingMuonReco total: {passing:,}",
            f"RealDataDQ total: {real:,}",
            "",
            "Four-way passing closure: "
            f"{sum(counts.values()):,} = {passing:,}",
            "Five-population DQ closure: "
            f"{sum(counts.values()):,} + {missing:,} = {real:,}",
        ]
    )
    if algorithm == "hough":
        lines.extend(
            [
                "",
                "Type-15 Hough combined tracks do not set the separate",
                "type-11 SciFi or type-13 DS bits used on this page.",
            ]
        )
    y = 0.80
    for line in lines:
        text.DrawLatex(0.11, y, line)
        y -= 0.042 if line else 0.023
    header = draw_header("Five disjoint event populations")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_algorithm_1d(canvas, source, config, algorithm, feature_name, feature):
    histograms = algorithm_histograms(source, algorithm, feature_name)
    if any(histogram.GetDimension() != 1 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not one-dimensional")
    integrals = [normalize(histogram) for histogram in histograms]
    labels = [
        ALGORITHM_COMBINATION_LABELS[value]
        for value in ALGORITHM_COMBINATION_ORDER
    ] + ["No PassingMuonReco"]
    labels = [
        f"{label} (valid N={format_count(count)})"
        for label, count in zip(labels, integrals)
    ]

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"algorithm_top_{algorithm}_{feature_name}", "", 0, 0.30, 1, 1)
    bottom = ROOT.TPad(
        f"algorithm_ratio_{algorithm}_{feature_name}", "", 0, 0, 1, 0.30
    )
    canvas._keepalive.extend([top, bottom])
    top.SetLeftMargin(0.15)
    top.SetRightMargin(0.05)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    bottom.SetLeftMargin(0.15)
    bottom.SetRightMargin(0.05)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    log_y = bool(feature.get("logy", False))
    top.SetLogy(log_y)
    top.Draw()
    bottom.Draw()

    markers = (20, 21, 22, 23, 25)
    for histogram, color, marker in zip(histograms, ALGORITHM_COLORS, markers):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.62)
        histogram.SetLineWidth(2)
    reference = histograms[3]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_1d(histograms)
    if log_y:
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10.0, 1e-6))
    else:
        reference.SetMinimum(0.0)
        reference.SetMaximum(max(maximum * 1.40, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    reference.GetXaxis().SetLabelSize(0.0)
    reference.GetXaxis().SetTitleSize(0.0)
    reference.GetYaxis().SetTitleOffset(1.25)

    top.cd()
    reference.Draw("E1 X0")
    for index, histogram in enumerate(histograms):
        if index != 3:
            histogram.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.55, 0.61, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.026)
    for histogram, label in zip(histograms, labels):
        legend.AddEntry(histogram, label, "lep")
    legend.Draw()
    header = draw_header(
        f"{config['algorithm_comparisons'][algorithm]['label']}; "
        f"{object_value(source, 'working_point_label')}"
    )
    canvas._keepalive.extend([legend, header])

    bottom.cd()
    ratio_frame = reference.Clone(f"algorithm_ratio_frame_{algorithm}_{feature_name}")
    ratio_frame.Reset("ICES")
    ratio_frame.SetTitle("")
    ratio_frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    ratio_frame.GetYaxis().SetTitle("Shape ratio to SF=1, DS=1")
    ratio_frame.GetYaxis().SetTitleSize(0.065)
    ratio_frame.GetYaxis().SetTitleOffset(0.90)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetXaxis().SetTitleSize(0.11)
    ratio_frame.GetXaxis().SetLabelSize(0.09)
    ratio_frame.GetXaxis().SetTitleOffset(1.0)
    ratio_frame.GetXaxis().CenterTitle(True)
    ratio_frame.GetYaxis().CenterTitle(True)
    ratios = []
    ratio_maximum = 1.0
    for index in (0, 1, 2, 4):
        ratio = histograms[index].Clone(
            f"algorithm_shape_ratio_{algorithm}_{feature_name}_{index}"
        )
        ratio.Divide(reference)
        ratio.SetLineColor(ALGORITHM_COLORS[index])
        ratio.SetMarkerColor(ALGORITHM_COLORS[index])
        ratio.SetMarkerStyle(markers[index])
        ratio.SetMarkerSize(0.52)
        for bin_index in range(1, ratio.GetNbinsX() + 1):
            if reference.GetBinContent(bin_index) <= 0:
                ratio.SetBinContent(bin_index, 0.0)
                ratio.SetBinError(bin_index, 0.0)
            else:
                ratio_maximum = max(
                    ratio_maximum,
                    ratio.GetBinContent(bin_index) + ratio.GetBinError(bin_index),
                )
        ratios.append(ratio)
    ratio_frame.SetMinimum(0.0)
    ratio_frame.SetMaximum(min(max(2.0, 1.20 * ratio_maximum), 20.0))
    ratio_frame.Draw("AXIS")
    for ratio in ratios:
        ratio.Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1.0, float(feature["x_max"]), 1.0)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.extend([ratio_frame, *ratios, unity])
    canvas.Modified()
    canvas.Update()


def draw_algorithm_2d(canvas, source, config, algorithm, feature_name, feature):
    histograms = algorithm_histograms(source, algorithm, feature_name)
    if any(histogram.GetDimension() != 2 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not two-dimensional")
    counts = [normalize(histogram) for histogram in histograms]
    labels = [
        ALGORITHM_COMBINATION_LABELS[value]
        for value in ALGORITHM_COMBINATION_ORDER
    ] + ["No PassingMuonReco"]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(3, 2, 0.002, 0.002)
    for index, (label, histogram, count) in enumerate(
        zip(labels, histograms, counts), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};"
            "Fraction of selected events"
        )
        histogram.Draw("COLZ")
        panel = ROOT.TLatex()
        panel.SetNDC(True)
        panel.SetTextFont(42)
        panel.SetTextSize(0.027)
        panel.DrawLatex(
            0.14, 0.965, f"{label} (valid N={format_count(count)})"
        )
        canvas._keepalive.append(panel)
    info_pad = canvas.cd(6)
    info = ROOT.TLatex()
    info.SetNDC(True)
    info.SetTextFont(42)
    info.SetTextSize(0.043)
    info.DrawLatex(0.10, 0.82, config["algorithm_comparisons"][algorithm]["label"])
    info.SetTextSize(0.034)
    info.DrawLatex(0.10, 0.70, str(object_value(source, "working_point_label")))
    info.DrawLatex(0.10, 0.58, "Four passing combinations + missing reference")
    info.DrawLatex(0.10, 0.46, "Unit-normalized with common z scale")
    if algorithm == "hough":
        info.DrawLatex(0.10, 0.34, "Type-15 combined track is a separate type")
    canvas._keepalive.extend([info_pad, info])
    canvas.Modified()
    canvas.Update()


def draw_all_summary(canvas, source, config):
    track_keys = list(config["track_types"])
    passing = int(object_value(source, "passing_count"))
    missing = int(object_value(source, "missing_count"))
    multiple = int(object_value(source, "multiple_types_count"))
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    canvas.Divide(2, 1, 0.002, 0.002)
    left = canvas.cd(1)
    left.SetLeftMargin(0.08)
    left.SetTopMargin(0.08)
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.042)
    text.DrawLatex(0.08, 0.94, "Inclusive track-type yields")
    text.SetTextSize(0.026)
    text.DrawLatex(0.08, 0.875, str(object_value(source, "working_point_label")))
    text.DrawLatex(0.08, 0.825, "Run 8329, real-data DQ")
    y = 0.755
    for key in track_keys:
        count = int(object_value(source, f"{key}__has_target_count"))
        label = config["track_types"][key]["label"]
        text.DrawLatex(
            0.10, y,
            f"{label}: {count:,} ({100.0 * count / passing:.3f}% of passing)",
        )
        y -= 0.058
    text.DrawLatex(
        0.10, y - 0.005,
        f"Multiple types: {multiple:,} ({100.0 * multiple / passing:.3f}%)",
    )
    text.DrawLatex(0.10, y - 0.065, f"Passing total: {passing:,}")
    text.DrawLatex(0.10, y - 0.12, f"NoPassingMuonReco reference: {missing:,}")

    right = canvas.cd(2)
    right.SetLeftMargin(0.27)
    right.SetRightMargin(0.16)
    right.SetBottomMargin(0.28)
    right.SetTopMargin(0.18)
    size = len(track_keys)
    matrix = ROOT.TH2D("track_overlap_summary", "", size, 0, size, size, 0, size)
    matrix.SetDirectory(0)
    for index, key in enumerate(track_keys, start=1):
        short = str(config["track_types"][key].get("short_label", key))
        matrix.GetXaxis().SetBinLabel(index, short)
        matrix.GetYaxis().SetBinLabel(index, short)
    for row_index, row_key in enumerate(track_keys, start=1):
        row_count = float(object_value(source, f"{row_key}__has_target_count"))
        for column_index, column_key in enumerate(track_keys, start=1):
            overlap = float(
                object_value(source, f"overlap__{row_key}__{column_key}")
            )
            matrix.SetBinContent(
                column_index,
                row_index,
                conditional_overlap_percent(overlap, row_count),
            )
    matrix.SetTitle(";Column category;;% of row category")
    matrix.SetMinimum(0.0)
    matrix.SetMaximum(100.0)
    matrix.GetXaxis().LabelsOption("v")
    matrix.GetXaxis().SetLabelSize(0.027)
    matrix.GetYaxis().SetLabelSize(0.027)
    matrix.GetXaxis().SetTitleOffset(3.2)
    matrix.GetZaxis().SetTitleOffset(1.1)
    ROOT.gStyle.SetPaintTextFormat(".2f")
    matrix.Draw("COLZ TEXT")
    note = ROOT.TLatex()
    note.SetNDC(True)
    note.SetTextFont(42)
    note.SetTextSize(0.034)
    note.DrawLatex(0.25, 0.965, "Track-type overlap")
    note.SetTextSize(0.023)
    note.DrawLatex(
        0.25, 0.925,
        "M(row,column) = 100 #times N(row AND column) / N(row)",
    )
    note.DrawLatex(
        0.25, 0.892,
        "Rows: y axis; columns: x axis; non-empty diagonal = 100%",
    )
    canvas._keepalive.extend([text, matrix, note, left, right])
    canvas.Modified()
    canvas.Update()


def all_type_histograms(source, config, feature_name):
    result = []
    for key in config["track_types"]:
        result.append(
            clone_histogram(
                source, f"targets/{key}/{feature_name}/has_target", f"all_{key}"
            )
        )
    return result


def draw_all_1d(canvas, source, config, feature_name, feature):
    histograms = all_type_histograms(source, config, feature_name)
    integrals = [normalize(histogram) for histogram in histograms]
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    pad = ROOT.TPad(f"all_{feature_name}", "", 0, 0, 1, 1)
    pad.SetLeftMargin(0.12)
    pad.SetRightMargin(0.05)
    pad.SetTopMargin(0.12)
    pad.SetBottomMargin(0.12)
    pad.SetLogy(bool(feature.get("logy", False)))
    pad.Draw()
    canvas._keepalive.append(pad)
    for index, (histogram, color) in enumerate(zip(histograms, TYPE_COLORS)):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(20 + index)
        histogram.SetMarkerSize(0.55)
        histogram.SetLineWidth(2)
    reference = histograms[0]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_1d(histograms)
    if bool(feature.get("logy", False)):
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10.0, 1e-6))
    else:
        reference.SetMinimum(0.0)
        reference.SetMaximum(max(maximum * 1.40, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    pad.cd()
    for index, histogram in enumerate(histograms):
        histogram.Draw("E1 X0" if index == 0 else "E1 X0 SAME")
    legend = ROOT.TLegend(0.51, 0.53, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.024)
    for key, histogram, count in zip(config["track_types"], histograms, integrals):
        legend.AddEntry(
            histogram,
            f"{config['track_types'][key]['label']} "
            f"(valid N={format_count(count)})",
            "lep",
        )
    legend.Draw()
    header = draw_header(f"All track types; {object_value(source, 'working_point_label')}")
    canvas._keepalive.extend([legend, header])
    canvas.Modified()
    canvas.Update()


def draw_all_2d(canvas, source, config, feature_name, feature):
    histograms = all_type_histograms(source, config, feature_name)
    integrals = [normalize(histogram) for histogram in histograms]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    columns = 4 if len(histograms) > 5 else 3
    rows = int(math.ceil((len(histograms) + 1) / columns))
    canvas.Divide(columns, rows, 0.002, 0.002)
    for index, (key, histogram, count) in enumerate(
        zip(config["track_types"], histograms, integrals), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Fraction of selected events"
        )
        histogram.Draw("COLZ")
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(0.028)
        label.DrawLatex(
            0.14, 0.965,
            f"{config['track_types'][key]['label']} "
            f"(valid N={format_count(count)})",
        )
        canvas._keepalive.append(label)
    info_pad = canvas.cd(len(histograms) + 1)
    info = ROOT.TLatex()
    info.SetNDC(True)
    info.SetTextFont(42)
    info.SetTextSize(0.045)
    info.DrawLatex(0.10, 0.82, "Run 8329 real-data DQ")
    info.SetTextSize(0.038)
    info.DrawLatex(0.10, 0.70, str(object_value(source, "working_point_label")))
    info.DrawLatex(0.10, 0.58, "Inclusive, overlapping track categories")
    info.DrawLatex(0.10, 0.46, "Common normalized z scale")
    canvas._keepalive.extend([info_pad, info])
    canvas.Modified()
    canvas.Update()


def reference_histograms(source, feature_name):
    return [
        clone_histogram(
            source,
            f"references/{feature_name}/{key}",
            f"reference_{key}",
        )
        for key in REFERENCE_KEYS
    ]


def draw_mc_reference_summary(canvas, source):
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    inclusive = int(object_value(source, "mc_muon_up_inclusive_count"))
    selected = int(object_value(source, "mc_muon_up_reference_count"))
    passing = int(object_value(source, "passing_count"))
    missing = int(object_value(source, "missing_count"))
    real = int(object_value(source, "real_data_dq_count"))
    expression = str(object_value(source, "mc_muon_up_reference_expression"))
    source_files = str(object_value(source, "mc_muon_up_source_files"))

    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.044)
    text.DrawLatex(0.09, 0.87, "Upward-muon MC detector reference")
    text.SetTextSize(0.029)
    lines = [
        f"MC inclusive muons: {inclusive:,}",
        f"MC detector-activity reference: {selected:,}",
        f"MC retained fraction: {100.0 * selected / inclusive if inclusive else 0.0:.4f}%",
        f"Selection: {expression}",
        "",
        f"RealDataDQ: {real:,}",
        f"PassingMuonReco: {passing:,}",
        f"NoPassingMuonReco: {missing:,}",
        f"Real-data closure: {passing:,} + {missing:,} = {real:,}",
        "",
        "MC is a unit-normalized shape reference, not part of data closure.",
        "No reconstructed-track branch is used by the MC selection.",
        "This MC production has no SciFi hits; do not tune SciFi cuts from it.",
        f"MC source: {source_files}",
    ]
    y = 0.79
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.050 if line else 0.025
    header = draw_header("Detector-only MC reference", "Run 8329 data + MC muon up")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_mc_reference_1d(canvas, source, feature_name, feature):
    histograms = reference_histograms(source, feature_name)
    if any(histogram.GetDimension() != 1 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not one-dimensional")
    integrals = [normalize(histogram) for histogram in histograms]
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"reference_top_{feature_name}", "", 0, 0.30, 1, 1)
    bottom = ROOT.TPad(f"reference_ratio_{feature_name}", "", 0, 0, 1, 0.30)
    top.SetLeftMargin(0.15)
    top.SetRightMargin(0.06)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    top.SetLogy(bool(feature.get("logy", False)))
    bottom.SetLeftMargin(0.15)
    bottom.SetRightMargin(0.06)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    top.Draw()
    bottom.Draw()
    canvas._keepalive.extend([top, bottom])

    for histogram, color, marker in zip(
        histograms, REFERENCE_COLORS, REFERENCE_MARKERS
    ):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.55)
        histogram.SetLineWidth(2)
    reference = histograms[0]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_1d(histograms)
    if bool(feature.get("logy", False)):
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10.0, 1e-6))
    else:
        reference.SetMinimum(0.0)
        reference.SetMaximum(max(maximum * 1.4, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    reference.GetXaxis().SetLabelSize(0)
    reference.GetXaxis().SetTitleSize(0)
    top.cd()
    for index, histogram in enumerate(histograms):
        histogram.Draw("E1 X0" if index == 0 else "E1 X0 SAME")
    legend = ROOT.TLegend(0.52, 0.60, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    for histogram, label, count in zip(histograms, REFERENCE_LABELS, integrals):
        legend.AddEntry(histogram, f"{label} (valid N={format_count(count)})", "lep")
    legend.Draw()
    header = draw_header("MC-muon-up detector reference", "Run 8329 data + MC muon up")

    bottom.cd()
    frame = reference.Clone(f"reference_ratio_frame_{feature_name}")
    frame.Reset("ICES")
    frame.SetTitle("")
    frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    frame.GetYaxis().SetTitle("Shape / MC shape")
    frame.GetYaxis().SetTitleSize(0.075)
    frame.GetYaxis().SetTitleOffset(0.78)
    frame.GetYaxis().SetLabelSize(0.08)
    frame.GetYaxis().SetNdivisions(505)
    frame.GetXaxis().SetTitleSize(0.11)
    frame.GetXaxis().SetLabelSize(0.09)
    ratios = []
    ratio_maximum = 1.0
    for index, histogram in enumerate(histograms[1:], start=1):
        ratio = histogram.Clone(f"reference_ratio_{feature_name}_{index}")
        ratio.Divide(reference)
        ratio.SetLineColor(REFERENCE_COLORS[index])
        ratio.SetMarkerColor(REFERENCE_COLORS[index])
        ratio.SetMarkerStyle(REFERENCE_MARKERS[index])
        for bin_index in range(1, ratio.GetNbinsX() + 1):
            if reference.GetBinContent(bin_index) <= 0:
                ratio.SetBinContent(bin_index, 0.0)
                ratio.SetBinError(bin_index, 0.0)
            ratio_maximum = max(ratio_maximum, ratio.GetBinContent(bin_index))
        ratios.append(ratio)
    frame.SetMinimum(0.0)
    frame.SetMaximum(min(max(2.0, 1.15 * ratio_maximum), 10.0))
    frame.Draw("AXIS")
    for ratio in ratios:
        ratio.Draw("E1 X0 SAME")
    unity = ROOT.TLine(reference.GetXaxis().GetXmin(), 1.0,
                       reference.GetXaxis().GetXmax(), 1.0)
    unity.SetLineStyle(2)
    unity.Draw()
    canvas._keepalive.extend([legend, header, frame, *ratios, unity])
    canvas.Modified()
    canvas.Update()


def draw_mc_reference_2d(canvas, source, feature_name, feature):
    histograms = reference_histograms(source, feature_name)
    if any(histogram.GetDimension() != 2 for histogram in histograms):
        raise RuntimeError(f"Feature {feature_name} is not two-dimensional")
    integrals = [normalize(histogram) for histogram in histograms]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(2, 2, 0.002, 0.002)
    for index, (histogram, label, count) in enumerate(
        zip(histograms, REFERENCE_LABELS, integrals), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Fraction of selected events"
        )
        histogram.Draw("COLZ")
        panel = ROOT.TLatex()
        panel.SetNDC(True)
        panel.SetTextFont(42)
        panel.SetTextSize(0.028)
        panel.DrawLatex(0.14, 0.965, f"{label} (valid N={format_count(count)})")
        canvas._keepalive.append(panel)
    canvas.Modified()
    canvas.Update()


def track_quality_histogram(source, target, feature_name, suffix):
    return clone_histogram(
        source,
        f"track_quality/{target}/{feature_name}/has_target",
        suffix,
    )


def draw_target_quality_2d(canvas, source, config, target, feature_name, feature):
    histogram = track_quality_histogram(
        source, target, feature_name, f"quality_{target}_{feature_name}"
    )
    if histogram.GetDimension() != 2:
        raise RuntimeError(f"Track-quality feature {feature_name} is not 2D")
    count = normalize(histogram)
    log_z = bool(feature.get("logz", True))
    positive = positive_minimum_2d([histogram])
    maximum = histogram.GetMaximum()
    histogram.SetMinimum(
        max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    )
    histogram.SetMaximum(max(maximum * (2.0 if log_z else 1.05), 1e-12))

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = [histogram]
    pad = ROOT.TPad(f"quality_{target}_{feature_name}", "", 0, 0, 1, 1)
    pad.SetLeftMargin(0.12)
    pad.SetRightMargin(0.16)
    pad.SetBottomMargin(0.13)
    pad.SetTopMargin(0.14)
    pad.SetLogz(log_z)
    pad.Draw()
    pad.cd()
    track = config["track_types"][target]
    histogram.SetTitle(
        f";{quality_score_title(track)};{feature['y_title']};"
        "Fraction of valid target events"
    )
    histogram.GetYaxis().SetTitleOffset(1.25)
    histogram.GetZaxis().SetTitleOffset(1.15)
    histogram.Draw("COLZ")
    header = draw_header(
        f"{track['label']}; {object_value(source, 'working_point_label')}"
    )
    subtitle = ROOT.TLatex()
    subtitle.SetNDC(True)
    subtitle.SetTextFont(42)
    subtitle.SetTextSize(0.029)
    subtitle.DrawLatex(
        0.14, 0.885,
        f"{feature['y_title']} versus target quality; "
        f"valid score N={format_count(count)}",
    )
    canvas._keepalive.extend([pad, header, subtitle])
    canvas.Modified()
    canvas.Update()


def draw_all_quality_2d(canvas, source, config, feature_name, feature):
    track_keys = list(config["track_types"])
    histograms = [
        track_quality_histogram(source, key, feature_name, f"quality_all_{key}")
        for key in track_keys
    ]
    counts = [normalize(histogram) for histogram in histograms]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", True))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    columns = 4 if len(histograms) > 5 else 3
    rows = int(math.ceil((len(histograms) + 1) / columns))
    canvas.Divide(columns, rows, 0.002, 0.002)
    for index, (key, histogram, count) in enumerate(
        zip(track_keys, histograms, counts), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        track = config["track_types"][key]
        histogram.SetTitle(
            f";{quality_score_title(track)};{feature['y_title']};"
            "Fraction of valid target events"
        )
        histogram.Draw("COLZ")
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(0.026)
        label.DrawLatex(
            0.14, 0.965,
            f"{track['label']} (valid N={format_count(count)})",
        )
        canvas._keepalive.append(label)

    info_pad = canvas.cd(len(histograms) + 1)
    info = ROOT.TLatex()
    info.SetNDC(True)
    info.SetTextFont(42)
    info.SetTextSize(0.041)
    info.DrawLatex(0.10, 0.82, "Run 8329 real-data DQ")
    info.SetTextSize(0.033)
    info.DrawLatex(0.10, 0.70, str(object_value(source, "working_point_label")))
    info.DrawLatex(0.10, 0.58, f"{feature['y_title']} versus track quality")
    info.DrawLatex(0.10, 0.46, "Inclusive categories; normalized z scale")
    info.DrawLatex(0.10, 0.34, "Paired categories use maximum component score")
    canvas._keepalive.extend([info_pad, info])
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    hist_config = load_yaml(args.hist_config)
    study_config = load_yaml(args.study_config)
    algorithm_plot_names = {
        f"{algorithm}_track_combinations": algorithm
        for algorithm in study_config["algorithm_comparisons"]
    }
    valid_comparisons = (
        set(study_config["track_types"])
        | {"all_track_types", "mc_muon_up_reference"}
        | set(algorithm_plot_names)
    )
    if args.comparison not in valid_comparisons:
        raise KeyError(f"Unknown comparison {args.comparison!r}")
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open {args.input}")
    working_point = str(object_value(source, "working_point"))
    if working_point not in study_config["working_points"]:
        raise RuntimeError(f"Unknown ROOT working point {working_point!r}")

    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".pdf", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    # ROOT fixes the page aspect ratio when the multipage PDF is opened. Use a
    # single landscape canvas so later three- and six-panel pages are not
    # compressed into the top of a portrait page.
    canvas = ROOT.TCanvas("track_study", "track_study", 1500, 900)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        algorithm = algorithm_plot_names.get(args.comparison)
        if args.comparison == "mc_muon_up_reference":
            draw_mc_reference_summary(canvas, source)
        elif args.comparison == "all_track_types":
            draw_all_summary(canvas, source, study_config)
        elif algorithm is not None:
            draw_algorithm_summary(canvas, source, study_config, algorithm)
        else:
            draw_target_summary(canvas, source, study_config, args.comparison)
        canvas.Print(temporary)
        page_count += 1
        for feature_name, raw_feature in hist_config["features"].items():
            feature = dict(hist_config.get("defaults", {}))
            feature.update(raw_feature)
            dimension = int(feature.get("dimension", 1))
            if args.comparison == "mc_muon_up_reference":
                if dimension == 1:
                    draw_mc_reference_1d(
                        canvas, source, feature_name, feature
                    )
                elif dimension == 2:
                    draw_mc_reference_2d(
                        canvas, source, feature_name, feature
                    )
                else:
                    raise ValueError(f"Unsupported dimension {dimension}")
            elif args.comparison == "all_track_types":
                if dimension == 1:
                    draw_all_1d(canvas, source, study_config, feature_name, feature)
                elif dimension == 2:
                    draw_all_2d(canvas, source, study_config, feature_name, feature)
                else:
                    raise ValueError(f"Unsupported dimension {dimension}")
            elif algorithm is not None:
                if dimension == 1:
                    draw_algorithm_1d(
                        canvas, source, study_config, algorithm,
                        feature_name, feature,
                    )
                elif dimension == 2:
                    draw_algorithm_2d(
                        canvas, source, study_config, algorithm,
                        feature_name, feature,
                    )
                else:
                    raise ValueError(f"Unsupported dimension {dimension}")
            else:
                if dimension == 1:
                    draw_target_1d(
                        canvas, source, study_config, args.comparison,
                        feature_name, feature,
                    )
                elif dimension == 2:
                    draw_target_2d(
                        canvas, source, study_config, args.comparison,
                        feature_name, feature,
                    )
                else:
                    raise ValueError(f"Unsupported dimension {dimension}")
            canvas.Print(temporary)
            page_count += 1
        # The four algorithm-bit populations do not share one meaningful
        # track score (the 0/0 group has none), so quality-observable pages
        # remain exclusive to target-type and all-type PDFs.
        if algorithm is None and args.comparison != "mc_muon_up_reference":
            for feature_name, feature in quality_features(study_config).items():
                if args.comparison == "all_track_types":
                    draw_all_quality_2d(
                        canvas, source, study_config, feature_name, feature
                    )
                else:
                    draw_target_quality_2d(
                        canvas,
                        source,
                        study_config,
                        args.comparison,
                        feature_name,
                        feature,
                    )
                canvas.Print(temporary)
                page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output_path)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} pages to {output_path}")


if __name__ == "__main__":
    main()
