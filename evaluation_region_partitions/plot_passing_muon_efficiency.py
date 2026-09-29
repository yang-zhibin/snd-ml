#!/usr/bin/env python3
"""Render one multi-page run-8329 comparison PDF for a configured selection."""

import argparse
import os
import tempfile
from pathlib import Path

import ROOT
import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def clone_histogram(source, feature_name, label):
    histogram = source.Get(f"{feature_name}/{label}")
    if not histogram:
        raise RuntimeError(f"Missing histogram {feature_name}/{label}")
    result = histogram.Clone(f"{feature_name}__{label}__plot")
    result.SetDirectory(0)
    return result


def histogram_integral(histogram):
    if histogram.GetDimension() == 1:
        return float(histogram.Integral(0, histogram.GetNbinsX() + 1))
    return float(
        histogram.Integral(
            0, histogram.GetNbinsX() + 1,
            0, histogram.GetNbinsY() + 1,
        )
    )


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


def draw_page_header(selection_key, selection_expression, luminosity):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.12, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, f"Run 8329, {luminosity:.4f} fb^{{-1}}")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    label.DrawLatex(0.12, 0.925, f"Selection: {selection_key} ({selection_expression})")
    return label


def binomial_graph(numerator, denominator, confidence_level):
    if not ROOT.TEfficiency.CheckConsistency(numerator, denominator):
        raise RuntimeError("Histograms are inconsistent with exact-subset efficiency")
    efficiency = ROOT.TEfficiency(numerator, denominator)
    efficiency.SetConfidenceLevel(float(confidence_level))
    efficiency.SetStatisticOption(ROOT.TEfficiency.kFCP)
    graph = efficiency.CreateGraph()
    graph.SetName(f"ratio_{numerator.GetName()}_over_{denominator.GetName()}")
    return efficiency, graph


def graph_ratio_range(graph):
    lower_values = []
    for index in range(graph.GetN()):
        y_value = float(graph.GetPointY(index))
        lower_values.append(max(0.0, y_value - float(graph.GetErrorYlow(index))))
    if not lower_values:
        return 0.0, 1.05
    minimum = min(lower_values)
    if minimum >= 0.8:
        lower = max(0.0, minimum - 0.08)
    else:
        lower = max(0.0, minimum * 0.85)
    return lower, 1.05


def draw_1d_page(
    canvas, source, feature_name, feature, selection_key,
    selection_expression, luminosity, confidence_level,
):
    real_data = clone_histogram(source, feature_name, "real_data")
    passing = clone_histogram(source, feature_name, "passing_muon")
    if real_data.GetDimension() != 1 or passing.GetDimension() != 1:
        raise RuntimeError(f"Feature {feature_name} is not one-dimensional")

    canvas.Clear()
    canvas._keepalive = [real_data, passing]
    top = ROOT.TPad(f"top_{feature_name}", "", 0.0, 0.30, 1.0, 1.0)
    bottom = ROOT.TPad(f"ratio_{feature_name}", "", 0.0, 0.0, 1.0, 0.30)
    canvas._keepalive.extend([top, bottom])
    top.SetLeftMargin(0.12)
    top.SetRightMargin(0.05)
    top.SetTopMargin(0.12)
    top.SetBottomMargin(0.025)
    bottom.SetLeftMargin(0.12)
    bottom.SetRightMargin(0.05)
    bottom.SetTopMargin(0.035)
    bottom.SetBottomMargin(0.36)
    bottom.SetGridy(True)
    log_y = bool(feature.get("logy", False))
    top.SetLogy(log_y)
    top.Draw()
    bottom.Draw()

    real_data.SetLineColor(ROOT.kBlack)
    real_data.SetMarkerColor(ROOT.kBlack)
    real_data.SetMarkerStyle(20)
    real_data.SetLineWidth(2)
    passing.SetLineColor(ROOT.kRed + 1)
    passing.SetMarkerColor(ROOT.kRed + 1)
    passing.SetMarkerStyle(24)
    passing.SetLineWidth(2)
    maximum = max(real_data.GetMaximum(), passing.GetMaximum(), 1.0)
    positive = positive_minimum_1d([real_data, passing])
    if log_y:
        real_data.SetMinimum(max((positive or 1.0) * 0.4, 1e-3))
        real_data.SetMaximum(maximum * 8.0)
    else:
        real_data.SetMinimum(0.0)
        real_data.SetMaximum(maximum * 1.35)
    real_data.SetTitle(f";{feature['axis_title']};Events")
    real_data.GetXaxis().SetLabelSize(0.0)
    real_data.GetXaxis().SetTitleSize(0.0)
    real_data.GetYaxis().SetTitleOffset(1.2)

    top.cd()
    real_data.Draw("E1 X0")
    passing.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.57, 0.69, 0.93, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(real_data, f"Real data after DQ ({histogram_integral(real_data):.0f})", "lep")
    legend.AddEntry(
        passing,
        f"PassingMuonReco after DQ ({histogram_integral(passing):.0f})",
        "lep",
    )
    legend.Draw()
    header = draw_page_header(selection_key, selection_expression, luminosity)
    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.036)
    title.DrawLatex(0.14, 0.86, feature_name.replace("_", " "))
    canvas._keepalive.extend([legend, header, title])

    bottom.cd()
    ratio_frame = real_data.Clone(f"ratio_frame_{feature_name}")
    ratio_frame.Reset("ICES")
    ratio_frame.SetTitle("")
    ratio_frame.GetXaxis().SetTitle(str(feature["axis_title"]))
    ratio_frame.GetYaxis().SetTitle("Passing / real")
    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.50)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetXaxis().SetTitleSize(0.11)
    ratio_frame.GetXaxis().SetLabelSize(0.09)
    ratio_frame.GetXaxis().SetTitleOffset(1.0)
    ratio_frame.SetMinimum(0.0)
    ratio_frame.SetMaximum(1.05)
    ratio_frame.Draw("AXIS")
    canvas._keepalive.append(ratio_frame)

    if histogram_integral(real_data) > 0:
        efficiency, graph = binomial_graph(passing, real_data, confidence_level)
        lower, upper = graph_ratio_range(graph)
        ratio_frame.SetMinimum(lower)
        ratio_frame.SetMaximum(upper)
        ratio_frame.Draw("AXIS")
        graph.SetLineColor(ROOT.kBlue + 1)
        graph.SetMarkerColor(ROOT.kBlue + 1)
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.65)
        graph.Draw("P SAME")
        canvas._keepalive.extend([efficiency, graph])
    else:
        empty = ROOT.TLatex()
        empty.SetNDC(True)
        empty.SetTextAlign(22)
        empty.SetTextSize(0.10)
        empty.DrawLatex(0.5, 0.55, "No selected real-data events")
        canvas._keepalive.append(empty)

    unity = ROOT.TLine(
        float(feature["x_min"]), 1.0, float(feature["x_max"]), 1.0
    )
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.append(unity)
    canvas.Modified()
    canvas.Update()


def draw_2d_page(
    canvas, source, feature_name, feature, selection_key,
    selection_expression, luminosity,
):
    real_data = clone_histogram(source, feature_name, "real_data")
    passing = clone_histogram(source, feature_name, "passing_muon")
    if real_data.GetDimension() != 2 or passing.GetDimension() != 2:
        raise RuntimeError(f"Feature {feature_name} is not two-dimensional")

    canvas.Clear()
    canvas._keepalive = [real_data, passing]
    maximum = max(real_data.GetMaximum(), passing.GetMaximum(), 1.0)
    positive = positive_minimum_2d([real_data, passing])
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1.0) * 0.5, 1e-9) if log_z else 0.0
    z_max = maximum * (2.0 if log_z else 1.05)
    canvas.Divide(2, 1, 0.002, 0.002)
    for index, (label, histogram) in enumerate(
        (("Real data after DQ", real_data), ("PassingMuonReco after DQ", passing)),
        start=1,
    ):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.17)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.15)
        pad.SetLogz(log_z)
        histogram.SetMinimum(z_min)
        histogram.SetMaximum(z_max)
        histogram.SetTitle(f";{feature['x_title']};{feature['y_title']};Events")
        histogram.GetXaxis().SetTitleOffset(1.15)
        histogram.GetYaxis().SetTitleOffset(1.35)
        histogram.GetZaxis().SetTitleOffset(1.25)
        histogram.Draw("COLZ")
        panel_label = ROOT.TLatex()
        panel_label.SetNDC(True)
        panel_label.SetTextFont(42)
        panel_label.SetTextSize(0.032)
        panel_label.DrawLatex(
            0.14, 0.965,
            f"Run 8329 {label} ({histogram_integral(histogram):.0f})",
        )
        panel_label.SetTextSize(0.024)
        panel_label.DrawLatex(
            0.14, 0.925,
            f"{selection_key}: {selection_expression}; {luminosity:.4f} fb^{{-1}}",
        )
        canvas._keepalive.append(panel_label)
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    hist_config = load_yaml(args.hist_config)
    eval_config = load_yaml(args.eval_config)
    interval = str(eval_config["statistics"].get("interval", "clopper_pearson"))
    if interval != "clopper_pearson":
        raise ValueError(f"Unsupported exact-subset interval {interval!r}")
    if not bool(eval_config["statistics"].get("require_subset", True)):
        raise ValueError("The comparison plotter requires exact-subset statistics")
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open {args.input}")
    selection_key = source.Get("selection_key").GetTitle()
    selection_expression = source.Get("selection_expression").GetTitle()
    luminosity = float(source.Get("luminosity_fb").GetVal())
    if selection_key not in eval_config["selections"]:
        raise RuntimeError(f"Histogram file has unknown selection {selection_key!r}")

    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".pdf", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    canvas = ROOT.TCanvas("comparison", "comparison", 1000, 900)
    try:
        canvas.Print(f"{temporary_name}[")
        for feature_name, raw_feature in hist_config["features"].items():
            feature = dict(hist_config.get("defaults", {}))
            feature.update(raw_feature)
            dimension = int(feature.get("dimension", 1))
            if dimension == 1:
                draw_1d_page(
                    canvas, source, feature_name, feature, selection_key,
                    selection_expression, luminosity,
                    float(eval_config["statistics"].get(
                        "confidence_level", 0.682689492137
                    )),
                )
            elif dimension == 2:
                canvas.SetCanvasSize(1400, 700)
                draw_2d_page(
                    canvas, source, feature_name, feature, selection_key,
                    selection_expression, luminosity,
                )
                canvas.SetCanvasSize(1000, 900)
            else:
                raise ValueError(f"Unsupported dimension {dimension} for {feature_name}")
            canvas.Print(temporary_name)
        canvas.Print(f"{temporary_name}]")
        os.replace(temporary_name, output_path)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"Wrote {len(hist_config['features'])} pages to {output_path}")


if __name__ == "__main__":
    main()
