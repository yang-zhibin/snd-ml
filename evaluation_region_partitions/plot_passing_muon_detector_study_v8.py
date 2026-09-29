#!/usr/bin/env python3
"""Plot V8 passing-muon-selection or no-Veto signal-region comparisons."""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from passing_muon_v8 import configured_features, load_v8_config


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

STYLES = {
    ("data", "passing_muon_candidate"): (ROOT.kBlue + 1, 20, 1),
    ("data", "muon_dis_candidate"): (ROOT.kRed + 1, 21, 1),
    ("data", "signal_region"): (ROOT.kMagenta + 1, 20, 1),
    ("mc", "passing_muon_candidate"): (ROOT.kCyan + 2, 24, 2),
    ("mc", "muon_dis_candidate"): (ROOT.kOrange + 7, 25, 2),
    ("mc", "signal_region"): (ROOT.kGreen + 2, 24, 2),
}


def metadata(source, name):
    return common.object_value(source, name)


def population_label(source, population):
    return str(metadata(source, f"population__{population}__label"))


def mode_populations(mode):
    if mode == "passing_muon_selection":
        return ("passing_muon_candidate", "muon_dis_candidate")
    return ("signal_region",)


def curves(source, mode):
    result = []
    for sample in ("data", "mc"):
        sample_label = "Run 8329" if sample == "data" else "Combined MC"
        for population in mode_populations(mode):
            color, marker, line_style = STYLES[(sample, population)]
            result.append({
                "sample": sample,
                "population": population,
                "label": f"{sample_label} {population_label(source, population)}",
                "color": color,
                "marker": marker,
                "line_style": line_style,
            })
    return result


def histogram_path(source, sample, population, feature):
    if sample == "data":
        root = "data"
    else:
        root = f"mc_muon_up/{metadata(source, 'mc_muon_up_aggregate_key')}"
    return f"{root}/populations/{population}/features/{feature}/histogram"


def selected_histograms(source, mode, feature):
    result = curves(source, mode)
    for item in result:
        histogram = common.clone_histogram(
            source,
            histogram_path(
                source, item["sample"], item["population"], feature
            ),
            f"v8_{mode}_{feature}_{item['sample']}_{item['population']}",
        )
        item["count"] = common.normalize(histogram)
        item["histogram"] = histogram
    return result


def draw_header(mode, text):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.10, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, "Run 8329 and pooled MC muon up")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    title = "Passing-muon selection" if mode == "passing_muon_selection" else "No-Veto signal region"
    label.DrawLatex(0.10, 0.925, f"{title}: {text}")
    return label


def draw_summary(canvas, source, mode, base_selection):
    data_total = int(metadata(source, "data_event_count"))
    data_lumi = float(metadata(source, "data_luminosity_fb"))
    data_year = int(metadata(source, "data_year"))
    mc_total = int(metadata(source, "mc_muon_up_event_count"))
    mc_declared = int(metadata(source, "mc_muon_up_declared_event_count"))
    mc_lumi = float(metadata(source, "mc_muon_up_luminosity_fb"))
    production_order = str(metadata(source, "mc_muon_up_production_order")).split(",")

    canvas.Clear()
    canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.038)
    title = "V8 passing-muon selection" if mode == "passing_muon_selection" else "V8 no-Veto signal region"
    text.DrawLatex(0.07, 0.88, title)
    text.SetTextSize(0.023)
    lines = [
        "Data catalog",
        f"  Year {data_year}, Run 8329 RealDataDQ: {data_total:,} events, {data_lumi:.6g} fb^{{-1}}",
        "",
        "Upward-muon MC catalogs",
    ]
    for key in production_order:
        prefix = f"mc_muon_up__{key}"
        lines.append(
            "  "
            f"{metadata(source, prefix + '__catalog')}: "
            f"{int(metadata(source, prefix + '__file_count'))} files, "
            f"{int(metadata(source, prefix + '__declared_event_count')):,} generated events, "
            f"{int(metadata(source, prefix + '__baseline_event_count')):,} baseline events, "
            f"{float(metadata(source, prefix + '__luminosity_fb')):.6g} fb^{{-1}}"
        )
    lines.extend([
        f"  Total: {mc_declared:,} generated events, {mc_total:,} baseline events, {mc_lumi:.6g} fb^{{-1}}",
        "  MC curves are an unweighted event sum of the four catalogs.",
        "",
        "Exclusive V8 populations",
    ])
    for population in (
        "passing_muon_candidate", "muon_dis_candidate", "signal_region"
    ):
        data_count = int(metadata(source, f"population__{population}__data_count"))
        mc_count = int(metadata(source, f"population__{population}__mc_count"))
        lines.append(
            f"  {population_label(source, population)}: data {data_count:,}; MC {mc_count:,}"
        )
    if mode == "passing_muon_selection":
        lines.extend(["", "Passing-muon detector requirements:"])
        for name, expression in base_selection["components"].items():
            lines.extend(textwrap.wrap(
                f"  {name}: {expression}", width=112,
                subsequent_indent="      ", break_long_words=False,
            ))
    else:
        lines.extend([
            "",
            "Signal-region definition: count_veto == 0.",
            "The passing-muon detector requirements are not applied in this PDF.",
        ])
    y = 0.83
    for line in lines:
        text.DrawLatex(0.08, y, line)
        y -= 0.031 if line else 0.014
    header = draw_header(mode, "sample and luminosity summary")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def style(item):
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


def draw_1d(canvas, source, mode, feature_name, feature):
    items = selected_histograms(source, mode, feature_name)
    for item in items:
        style(item)
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
    maximum = max((hist.GetMaximum() for hist in histograms), default=1)
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
    legend = ROOT.TLegend(0.45, 0.67, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.023)
    for item in items:
        legend.AddEntry(
            item["histogram"],
            f"{item['label']} (N={common.format_count(item['count'])})", "lep",
        )
    legend.Draw()
    header = draw_header(mode, feature["axis_title"])

    lower.cd()
    population_count = len(mode_populations(mode))
    ratios = []
    for index, population in enumerate(mode_populations(mode)):
        result = ratio(
            histograms[population_count + index], histograms[index],
            f"ratio_{population}_{feature_name}",
        )
        result.SetLineColor(STYLES[("mc", population)][0])
        result.SetMarkerColor(STYLES[("mc", population)][0])
        result.SetMarkerStyle(STYLES[("mc", population)][1])
        result.SetLineWidth(2)
        ratios.append(result)
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
    for result in ratios[1:]:
        result.Draw("E1 X0 SAME")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    ratio_legend = ROOT.TLegend(0.62, 0.69, 0.94, 0.94)
    ratio_legend.SetBorderSize(0)
    ratio_legend.SetFillStyle(0)
    ratio_legend.SetTextSize(0.067)
    for result, population in zip(ratios, mode_populations(mode)):
        ratio_legend.AddEntry(result, population_label(source, population), "lep")
    ratio_legend.Draw()
    canvas._keepalive = [
        *histograms, *ratios, upper, lower, legend, ratio_legend, header, unity,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, mode, feature_name, feature):
    items = selected_histograms(source, mode, feature_name)
    histograms = [item["histogram"] for item in items]
    maximum = max((hist.GetMaximum() for hist in histograms), default=1)
    positive = common.positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    minimum = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0
    canvas.Clear()
    canvas.Divide(2, 2 if len(items) > 2 else 1, 0.002, 0.002)
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
        "--mode", required=True,
        choices=("passing_muon_selection", "signal_region"),
    )
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    overlay, base_config, base_selection = load_v8_config(args.region_config)
    hist_config = common.load_yaml(args.hist_config)
    features = configured_features(hist_config, base_config)
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if str(metadata(source, "study_version")) != str(overlay["version"]):
        raise RuntimeError("Input ROOT file is not the configured V8 study")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("v8", "v8", 1200, 850)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(canvas, source, args.mode, base_selection)
        canvas.Print(temporary)
        page_count += 1
        for feature_name, feature in features.items():
            if int(feature.get("dimension", 1)) == 1:
                draw_1d(canvas, source, args.mode, feature_name, feature)
            else:
                draw_2d(canvas, source, args.mode, feature_name, feature)
            canvas.Print(temporary)
            page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} V8 {args.mode} pages to {output}")


if __name__ == "__main__":
    main()
