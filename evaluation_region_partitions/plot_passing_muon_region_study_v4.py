#!/usr/bin/env python3
"""Render run-8329 v4 marginal, joint-summary, and selected-region PDFs."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml

from passing_muon_v4_features import expanded_v4_hist_config


COLORS = (ROOT.kGray + 2, ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2)
MARKERS = (20, 21, 22, 23)
POPULATION_COLORS = (ROOT.kBlue + 1, ROOT.kRed + 1)
POPULATION_MARKERS = (20, 21)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return result


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


def dq_histogram(source, base_path, suffix):
    passing = clone_histogram(
        source, f"{base_path}/passing_muon_reco", f"{suffix}_passing"
    )
    missing = clone_histogram(
        source, f"{base_path}/no_passing_muon_reco", f"{suffix}_missing"
    )
    passing.Add(missing)
    passing.SetName(f"dq__{suffix}")
    return passing


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
    return f"{float(value):.3e}"


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


def draw_header(text):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.10, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, "Run 8329, real-data DQ")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    label.DrawLatex(0.10, 0.925, text)
    return label


def axis_count(source, axis, category, population):
    return int(
        object_value(source, f"axis__{axis}__{category}__{population}__count")
    )


def joint_count(source, region, population):
    return int(object_value(source, f"joint__{region}__{population}__count"))


def marginal_histograms(source, axis, categories, feature_name):
    return [
        dq_histogram(
            source,
            f"marginal/{axis}/{category}/{feature_name}",
            f"{axis}_{category}_{feature_name}",
        )
        for category in categories
    ]


def draw_marginal_summary(canvas, source, config, axis_name):
    axis = config["axes"][axis_name]
    real = int(object_value(source, "real_data_dq_count"))
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.047)
    text.DrawLatex(0.09, 0.86, axis["label"])
    text.SetTextSize(0.030)
    y = 0.79
    for category, definition in axis["categories"].items():
        passing = axis_count(source, axis_name, category, "passing")
        missing = axis_count(source, axis_name, category, "missing")
        dq = passing + missing
        efficiency = 100.0 * passing / dq if dq else 0.0
        text.DrawLatex(0.11, y, definition["label"])
        text.DrawLatex(
            0.34,
            y,
            f"DQ={format_count(dq)}, passing={format_count(passing)}, "
            f"missing={format_count(missing)}, passing fraction={efficiency:.4f}%",
        )
        y -= 0.070
    text.DrawLatex(0.11, y - 0.005, f"All RealDataDQ events: {real:,}")
    y -= 0.080
    if axis_name == "veto_time":
        text.SetTextSize(0.028)
        text.DrawLatex(0.11, y, "Audited outside the 40 main joint regions:")
        y -= 0.052
        for category, definition in axis.get("audit_categories", {}).items():
            passing = axis_count(source, axis_name, category, "passing")
            missing = axis_count(source, axis_name, category, "missing")
            text.DrawLatex(
                0.13,
                y,
                f"{definition['label']}: DQ={passing + missing:,} "
                f"(passing={passing:,}, missing={missing:,})",
            )
            y -= 0.047
        text.DrawLatex(
            0.11,
            y - 0.010,
            "No-veto time means count_veto = 0; invalid positive-hit times are separate.",
        )
    header = draw_header("Four exclusive marginal regions")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_marginal_1d(canvas, source, config, axis_name, feature_name, feature):
    axis = config["axes"][axis_name]
    categories = list(axis["categories"])
    histograms = marginal_histograms(source, axis_name, categories, feature_name)
    integrals = [normalize(histogram) for histogram in histograms]
    reference_index = categories.index(axis["reference"])
    reference = histograms[reference_index]

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"top_{axis_name}_{feature_name}", "", 0, 0.30, 1, 1)
    bottom = ROOT.TPad(f"ratio_{axis_name}_{feature_name}", "", 0, 0, 1, 0.30)
    top.SetLeftMargin(0.15)
    top.SetRightMargin(0.07)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    top.SetLogy(bool(feature.get("logy", False)))
    bottom.SetLeftMargin(0.15)
    bottom.SetRightMargin(0.07)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    top.Draw()
    bottom.Draw()
    canvas._keepalive.extend([top, bottom])
    for histogram, color, marker in zip(histograms, COLORS, MARKERS):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.60)
        histogram.SetLineWidth(2)
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = positive_minimum_1d(histograms)
    if bool(feature.get("logy", False)):
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10, 1e-6))
    else:
        reference.SetMinimum(0)
        reference.SetMaximum(max(maximum * 1.4, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    reference.GetXaxis().SetLabelSize(0)
    reference.GetXaxis().SetTitleSize(0)
    top.cd()
    reference.Draw("E1 X0")
    for index, histogram in enumerate(histograms):
        if index != reference_index:
            histogram.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.56, 0.62, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.026)
    for category, histogram, count in zip(categories, histograms, integrals):
        legend.AddEntry(
            histogram,
            f"{axis['categories'][category]['label']} "
            f"(valid N={format_count(count)})",
            "lep",
        )
    legend.Draw()
    header = draw_header(axis["label"])
    canvas._keepalive.extend([legend, header])

    bottom.cd()
    frame = reference.Clone(f"ratio_frame_{axis_name}_{feature_name}")
    frame.Reset("ICES")
    frame.SetTitle("")
    frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    frame.GetYaxis().SetTitle(f"Shape ratio to {axis['categories'][axis['reference']]['label']}")
    frame.GetYaxis().SetTitleSize(0.065)
    frame.GetYaxis().SetTitleOffset(0.90)
    frame.GetYaxis().SetLabelSize(0.08)
    frame.GetYaxis().SetNdivisions(505)
    frame.GetXaxis().SetTitleSize(0.11)
    frame.GetXaxis().SetLabelSize(0.09)
    frame.GetXaxis().SetTitleOffset(1.0)
    ratios = []
    ratio_maximum = 1.0
    for index, histogram in enumerate(histograms):
        if index == reference_index:
            continue
        ratio = histogram.Clone(f"ratio_{axis_name}_{feature_name}_{index}")
        ratio.Divide(reference)
        ratio.SetLineColor(COLORS[index])
        ratio.SetMarkerColor(COLORS[index])
        ratio.SetMarkerStyle(MARKERS[index])
        ratio.SetMarkerSize(0.50)
        for bin_index in range(1, ratio.GetNbinsX() + 1):
            if reference.GetBinContent(bin_index) <= 0:
                ratio.SetBinContent(bin_index, 0)
                ratio.SetBinError(bin_index, 0)
            else:
                ratio_maximum = max(
                    ratio_maximum,
                    ratio.GetBinContent(bin_index) + ratio.GetBinError(bin_index),
                )
        ratios.append(ratio)
    frame.SetMinimum(0)
    frame.SetMaximum(min(max(2.0, 1.2 * ratio_maximum), 20.0))
    frame.Draw("AXIS")
    for ratio in ratios:
        ratio.Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.extend([frame, *ratios, unity])
    canvas.Modified()
    canvas.Update()


def draw_marginal_2d(canvas, source, config, axis_name, feature_name, feature):
    axis = config["axes"][axis_name]
    categories = list(axis["categories"])
    histograms = marginal_histograms(source, axis_name, categories, feature_name)
    counts = [normalize(histogram) for histogram in histograms]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0
    z_max = max(maximum * (2 if log_z else 1.05), 1e-12)
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(2, 2, 0.002, 0.002)
    for index, (category, histogram, count) in enumerate(
        zip(categories, histograms, counts), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
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
        label.SetTextSize(0.030)
        label.DrawLatex(
            0.13,
            0.965,
            f"{axis['categories'][category]['label']} "
            f"(valid N={format_count(count)})",
        )
        canvas._keepalive.extend([pad, label])
    canvas.Modified()
    canvas.Update()


def selected_histograms(source, region, feature_name):
    base = f"joint_regions/{region}/{feature_name}"
    return [
        clone_histogram(source, f"{base}/passing_muon_reco", "selected_passing"),
        clone_histogram(
            source, f"{base}/no_passing_muon_reco", "selected_missing"
        ),
    ]


def draw_selected_summary(canvas, source, config, region):
    passing = joint_count(source, region, "passing")
    missing = joint_count(source, region, "missing")
    dq = passing + missing
    efficiency = 100.0 * passing / dq if dq else 0.0
    real = int(object_value(source, "real_data_dq_count"))
    count_category, time_category, us_category = region.split("__")
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.043)
    text.DrawLatex(0.08, 0.85, "Selected joint region")
    text.SetTextSize(0.029)
    lines = [
        region,
        "",
        f"RealDataDQ in region: {dq:,} ({100.0 * dq / real if real else 0:.6f}% of DQ)",
        f"PassingMuonReco: {passing:,}",
        f"NoPassingMuonReco: {missing:,}",
        f"Passing fraction inside region: {efficiency:.6f}%",
        "",
        f"Closure: {passing:,} + {missing:,} = {dq:,}",
        "",
        "Region categories:",
        f"Veto count: {config['axes']['veto_count']['categories'][count_category]['label']}",
        f"Veto time: {config['axes']['veto_time']['categories'][time_category]['label']}",
        f"US count: {config['axes']['us_count']['categories'][us_category]['label']}",
        "The complete C++ expression is stored in the ROOT metadata.",
    ]
    y = 0.79
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.050 if line else 0.025
    header = draw_header("PassingMuonReco versus NoPassingMuonReco")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_selected_1d(canvas, source, region, feature_name, feature):
    histograms = selected_histograms(source, region, feature_name)
    integrals = [normalize(histogram) for histogram in histograms]
    labels = (
        f"PassingMuonReco (valid N={format_count(integrals[0])})",
        f"NoPassingMuonReco (valid N={format_count(integrals[1])})",
    )
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"selected_top_{feature_name}", "", 0, 0.30, 1, 1)
    bottom = ROOT.TPad(f"selected_ratio_{feature_name}", "", 0, 0, 1, 0.30)
    top.SetLeftMargin(0.15)
    top.SetRightMargin(0.07)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    top.SetLogy(bool(feature.get("logy", False)))
    bottom.SetLeftMargin(0.15)
    bottom.SetRightMargin(0.07)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    top.Draw()
    bottom.Draw()
    canvas._keepalive.extend([top, bottom])
    for histogram, color, marker in zip(
        histograms, POPULATION_COLORS, POPULATION_MARKERS
    ):
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.62)
        histogram.SetLineWidth(2)
    reference = histograms[0]
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = positive_minimum_1d(histograms)
    if bool(feature.get("logy", False)):
        reference.SetMinimum(max((positive or 1e-6) * 0.4, 1e-8))
        reference.SetMaximum(max(maximum * 10, 1e-6))
    else:
        reference.SetMinimum(0)
        reference.SetMaximum(max(maximum * 1.4, 1e-6))
    reference.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    reference.GetXaxis().SetLabelSize(0)
    reference.GetXaxis().SetTitleSize(0)
    top.cd()
    reference.Draw("E1 X0")
    histograms[1].Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.58, 0.72, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.028)
    for histogram, label in zip(histograms, labels):
        legend.AddEntry(histogram, label, "lep")
    legend.Draw()
    header = draw_header(region)
    canvas._keepalive.extend([legend, header])

    bottom.cd()
    frame = reference.Clone(f"selected_ratio_frame_{feature_name}")
    frame.Reset("ICES")
    frame.SetTitle("")
    frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    frame.GetYaxis().SetTitle("Missing / passing shape")
    frame.GetYaxis().SetTitleSize(0.075)
    frame.GetYaxis().SetTitleOffset(0.78)
    frame.GetYaxis().SetLabelSize(0.08)
    frame.GetYaxis().SetNdivisions(505)
    frame.GetXaxis().SetTitleSize(0.11)
    frame.GetXaxis().SetLabelSize(0.09)
    ratio = histograms[1].Clone(f"selected_ratio_{feature_name}")
    ratio.Divide(reference)
    ratio.SetLineColor(POPULATION_COLORS[1])
    ratio.SetMarkerColor(POPULATION_COLORS[1])
    ratio.SetMarkerStyle(POPULATION_MARKERS[1])
    ratio_maximum = 1.0
    for index in range(1, ratio.GetNbinsX() + 1):
        if reference.GetBinContent(index) <= 0:
            ratio.SetBinContent(index, 0)
            ratio.SetBinError(index, 0)
        else:
            ratio_maximum = max(
                ratio_maximum,
                ratio.GetBinContent(index) + ratio.GetBinError(index),
            )
    frame.SetMinimum(0)
    frame.SetMaximum(min(max(2.0, 1.2 * ratio_maximum), 20.0))
    frame.Draw("AXIS")
    ratio.Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.extend([frame, ratio, unity])
    canvas.Modified()
    canvas.Update()


def draw_selected_2d(canvas, source, region, feature_name, feature):
    histograms = selected_histograms(source, region, feature_name)
    counts = [normalize(histogram) for histogram in histograms]
    labels = ("PassingMuonReco", "NoPassingMuonReco")
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1)
    positive = positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0
    z_max = max(maximum * (2 if log_z else 1.05), 1e-12)
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(2, 1, 0.002, 0.002)
    for index, (histogram, label, count) in enumerate(
        zip(histograms, labels, counts), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
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
        panel.SetTextSize(0.030)
        panel.DrawLatex(0.13, 0.965, f"{label} (valid N={format_count(count)})")
        panel.SetTextSize(0.023)
        panel.DrawLatex(0.13, 0.925, region)
        canvas._keepalive.extend([pad, panel])
    canvas.Modified()
    canvas.Update()


def matrix(source, config, us_category, value_kind, name):
    count_categories = list(config["axes"]["veto_count"]["categories"])
    time_categories = list(config["axes"]["veto_time"]["categories"])
    result = ROOT.TH2D(name, "", 4, 0, 4, 4, 0, 4)
    result.SetDirectory(0)
    real_total = int(object_value(source, "real_data_dq_count"))
    for ix, count_category in enumerate(count_categories, start=1):
        result.GetXaxis().SetBinLabel(
            ix, config["axes"]["veto_count"]["categories"][count_category]["label"]
        )
    for iy, time_category in enumerate(time_categories, start=1):
        result.GetYaxis().SetBinLabel(
            iy, config["axes"]["veto_time"]["categories"][time_category]["label"]
        )
    for ix, count_category in enumerate(count_categories, start=1):
        count_state = config["axes"]["veto_count"]["categories"][count_category][
            "state"
        ]
        for iy, time_category in enumerate(time_categories, start=1):
            time_state = config["axes"]["veto_time"]["categories"][time_category][
                "state"
            ]
            if count_state != time_state:
                continue
            region = f"{count_category}__{time_category}__{us_category}"
            passing = joint_count(source, region, "passing")
            missing = joint_count(source, region, "missing")
            dq = passing + missing
            if value_kind == "dq_count":
                value = dq
            elif value_kind == "dq_fraction":
                value = 100.0 * dq / real_total if real_total else 0
            elif value_kind == "passing_fraction":
                value = 100.0 * passing / dq if dq else 0
            elif value_kind == "missing_count":
                value = missing
            else:
                raise ValueError(value_kind)
            result.SetBinContent(ix, iy, value)
    return result


def draw_joint_intro(canvas, source, config):
    passing = int(object_value(source, "passing_count"))
    missing = int(object_value(source, "missing_count"))
    invalid = sum(
        axis_count(source, "veto_time", "veto_time_invalid", population)
        for population in ("passing", "missing")
    )
    negative = sum(
        axis_count(source, "veto_time", "veto_time_negative", population)
        for population in ("passing", "missing")
    )
    eligible = passing + missing - invalid - negative
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.047)
    text.DrawLatex(0.09, 0.86, "Joint veto-count, veto-time, and US regions")
    text.SetTextSize(0.031)
    lines = [
        "40 physically compatible main regions",
        "4 US categories #times (1 no-veto state + 9 positive-veto states)",
        "",
        f"RealDataDQ: {passing + missing:,}",
        f"Main-region eligible: {eligible:,}",
        f"Positive-hit invalid/missing veto time: {invalid:,}",
        f"Valid negative veto time: {negative:,}",
        "",
        "Each following page fixes one US category.",
        "Cells incompatible with veto presence are intentionally empty.",
        "Passing fraction = 100 x N(PassingMuonReco) / N(RealDataDQ in cell).",
    ]
    y = 0.78
    for line in lines:
        text.DrawLatex(0.11, y, line)
        y -= 0.052 if line else 0.026
    header = draw_header("Joint-region population summary")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_joint_us_page(canvas, source, config, us_category):
    specifications = (
        ("dq_count", "RealDataDQ events", ".2e"),
        ("dq_fraction", "% of complete RealDataDQ", ".3f"),
        ("passing_fraction", "PassingMuonReco fraction [%]", ".3f"),
        ("missing_count", "NoPassingMuonReco events", ".2e"),
    )
    matrices = [
        matrix(source, config, us_category, kind, f"matrix_{us_category}_{kind}")
        for kind, _, _ in specifications
    ]
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(matrices)
    canvas.Divide(2, 2, 0.002, 0.002)
    labels = []
    for index, (histogram, (_, title, paint_format)) in enumerate(
        zip(matrices, specifications), start=1
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.24)
        pad.SetRightMargin(0.15)
        pad.SetBottomMargin(0.22)
        pad.SetTopMargin(0.17)
        histogram.SetTitle(f";Veto-count category;;{title}")
        histogram.GetXaxis().LabelsOption("v")
        histogram.GetXaxis().SetLabelSize(0.028)
        histogram.GetYaxis().SetLabelSize(0.028)
        histogram.GetXaxis().SetTitleOffset(2.9)
        histogram.GetZaxis().SetTitleOffset(1.1)
        ROOT.gStyle.SetPaintTextFormat(paint_format)
        histogram.Draw("COLZ TEXT")
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(0.032)
        label.DrawLatex(
            0.24,
            0.965,
            config["axes"]["us_count"]["categories"][us_category]["label"],
        )
        labels.append(label)
        canvas._keepalive.append(pad)
    canvas._keepalive.extend(labels)
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    hist_config = load_yaml(args.hist_config)
    config = load_yaml(args.region_config)
    hist_config = expanded_v4_hist_config(hist_config, config)
    axis_comparisons = {
        f"{axis}_regions": axis for axis in config["axes"]
    }
    selected_comparisons = {
        f"region__{region}": region
        for region in config.get("selected_joint_regions", [])
    }
    valid = set(axis_comparisons) | {"joint_region_summary"} | set(
        selected_comparisons
    )
    if args.comparison not in valid:
        raise KeyError(f"Unknown comparison {args.comparison!r}")
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open {args.input}")
    if str(object_value(source, "study_version")) != str(config["version"]):
        raise RuntimeError("ROOT/config study-version mismatch")

    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".pdf", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("region_study_v4", "region_study_v4", 1500, 900)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        if args.comparison == "joint_region_summary":
            draw_joint_intro(canvas, source, config)
            canvas.Print(temporary)
            page_count += 1
            for us_category in config["axes"]["us_count"]["categories"]:
                draw_joint_us_page(canvas, source, config, us_category)
                canvas.Print(temporary)
                page_count += 1
        else:
            axis_name = axis_comparisons.get(args.comparison)
            region = selected_comparisons.get(args.comparison)
            if axis_name is not None:
                draw_marginal_summary(canvas, source, config, axis_name)
            else:
                draw_selected_summary(canvas, source, config, region)
            canvas.Print(temporary)
            page_count += 1
            for feature_name, raw_feature in hist_config["features"].items():
                feature = dict(hist_config.get("defaults", {}))
                feature.update(raw_feature)
                dimension = int(feature.get("dimension", 1))
                if axis_name is not None and dimension == 1:
                    draw_marginal_1d(
                        canvas, source, config, axis_name, feature_name, feature
                    )
                elif axis_name is not None and dimension == 2:
                    draw_marginal_2d(
                        canvas, source, config, axis_name, feature_name, feature
                    )
                elif region is not None and dimension == 1:
                    draw_selected_1d(
                        canvas, source, region, feature_name, feature
                    )
                elif region is not None and dimension == 2:
                    draw_selected_2d(
                        canvas, source, region, feature_name, feature
                    )
                else:
                    raise ValueError(f"Unsupported dimension {dimension}")
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
