#!/usr/bin/env python3
"""Render run-8329 detector-only through-going-muon study PDFs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from make_passing_muon_binary_region_hist_v5 import (
    MAXIMUM_FIELDS,
    POPULATIONS,
    configured_selections,
    configured_v5_features,
    validate_region_config,
)


COLORS = (ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2)
MARKERS = (20, 21, 24)


def selection_value(source, selection, suffix):
    return common.object_value(source, f"selection__{selection}__{suffix}")


def mc_production_value(source, production, suffix):
    return common.object_value(source, f"mc_muon_up__{production}__{suffix}")


def production_keys(config):
    return list(config["mc_muon_up"]["productions"])


def aggregate_comparisons(config):
    aggregate = config["mc_muon_up"].get("aggregate", {}) or {}
    if not aggregate:
        return set()
    return {
        f"{selection}__mc__{aggregate['key']}"
        for selection in aggregate.get("selections", [])
    }


def selected_histograms(source, selection, feature_name, production=None):
    base = f"selections/{selection}/features/{feature_name}"
    candidate = common.clone_histogram(
        source, f"{base}/{POPULATIONS['candidate']}",
        f"{selection}_candidate",
    )
    if production is None:
        complement = common.clone_histogram(
            source, f"{base}/{POPULATIONS['complement']}",
            f"{selection}_complement",
        )
        return [candidate, complement]
    mc_path = (
        f"mc_muon_up/{production}/selections/{selection}/features/"
        f"{feature_name}/candidate"
    )
    return [
        candidate,
        common.clone_histogram(
            source, mc_path, f"{selection}_{production}_mc_candidate"
        ),
    ]


def draw_selection_summary(canvas, source, selection_name):
    total = int(common.object_value(source, "real_data_dq_count"))
    candidate = int(selection_value(source, selection_name, "candidate_count"))
    complement = int(selection_value(source, selection_name, "complement_count"))
    label = str(selection_value(source, selection_name, "label"))
    candidate_label = str(selection_value(source, selection_name, "candidate_label"))
    complement_label = str(selection_value(source, selection_name, "complement_label"))
    cut_order = str(selection_value(source, selection_name, "cut_order")).split(",")

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.044)
    text.DrawLatex(0.08, 0.85, label)
    text.SetTextSize(0.029)
    lines = [
        f"Working point: {selection_name}",
        "Detector-only definition; no reconstructed-track information",
        "",
        f"RealDataDQ: {common.format_count(total)}",
        f"{candidate_label}: {common.format_count(candidate)} "
        f"({100.0 * candidate / total if total else 0.0:.4f}%)",
        f"{complement_label}: {common.format_count(complement)} "
        f"({100.0 * complement / total if total else 0.0:.4f}%)",
        f"Closure: {common.format_count(candidate)} + "
        f"{common.format_count(complement)} = {common.format_count(total)}",
        "",
        "Cumulative requirements:",
    ]
    for cut in cut_order:
        expression = str(selection_value(source, selection_name, f"cut__{cut}__expression"))
        lines.append(f"  {cut}: {expression}")
    y = 0.79
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.050 if line else 0.025
    header = common.draw_header("Detector-only population definition")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def style_1d(histogram, color, marker):
    histogram.SetLineColor(color)
    histogram.SetMarkerColor(color)
    histogram.SetMarkerStyle(marker)
    histogram.SetMarkerSize(0.55)
    histogram.SetLineWidth(2)


def draw_selection_1d(
    canvas, source, selection_name, feature_name, feature,
    histograms=None, labels=None, header_text=None,
    ratio_title="Complement / candidate", production=None,
):
    histograms = histograms or selected_histograms(
        source, selection_name, feature_name, production
    )
    integrals = [common.normalize(histogram) for histogram in histograms]
    if labels is None:
        candidate_label = str(selection_value(source, selection_name, "candidate_label"))
        complement_label = str(selection_value(source, selection_name, "complement_label"))
        if production is None:
            labels = (candidate_label, complement_label)
        else:
            labels = (
                candidate_label,
                f"{mc_production_value(source, production, 'label')}, passes selection",
            )
    labels = tuple(
        f"{label} (valid N={common.format_count(integral)})"
        for label, integral in zip(labels, integrals)
    )

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    top = ROOT.TPad(f"top_{selection_name}_{feature_name}", "", 0, 0.30, 1, 1)
    bottom = ROOT.TPad(f"ratio_{selection_name}_{feature_name}", "", 0, 0, 1, 0.30)
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

    top.cd()
    for histogram, color, marker in zip(histograms, COLORS, MARKERS):
        style_1d(histogram, color, marker)
        histogram.SetTitle(f";{feature['axis_title']};Fraction of selected events")
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = common.positive_minimum_1d(histograms)
    if bool(feature.get("logy", False)):
        histograms[0].SetMinimum(max((positive or 1e-9) * 0.5, 1e-9))
        histograms[0].SetMaximum(max(maximum * 20.0, 1e-8))
    else:
        histograms[0].SetMinimum(0.0)
        histograms[0].SetMaximum(max(maximum * 1.35, 1e-8))
    histograms[0].GetXaxis().SetLabelSize(0)
    histograms[0].Draw("E1 X0")
    for histogram in histograms[1:]:
        histogram.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.48, 0.74, 0.93, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.026)
    for histogram, label in zip(histograms, labels):
        legend.AddEntry(histogram, label, "lep")
    legend.Draw()
    header = common.draw_header(
        header_text or str(selection_value(source, selection_name, "label"))
    )

    bottom.cd()
    ratio = histograms[1].Clone(f"ratio_{selection_name}_{feature_name}")
    ratio.SetDirectory(0)
    ratio.Divide(histograms[0])
    ratio.SetTitle(f";{feature['axis_title']};{ratio_title}")
    ratio.GetXaxis().SetTitleSize(0.13)
    ratio.GetXaxis().SetLabelSize(0.10)
    ratio.GetXaxis().SetTitleOffset(1.15)
    ratio.GetYaxis().SetTitleSize(0.095)
    ratio.GetYaxis().SetLabelSize(0.085)
    ratio.GetYaxis().SetTitleOffset(0.65)
    ratio.GetYaxis().SetNdivisions(505)
    ratio.SetMinimum(0.0)
    finite_max = max(
        (ratio.GetBinContent(i) + ratio.GetBinError(i)
         for i in range(1, ratio.GetNbinsX() + 1)
         if histograms[0].GetBinContent(i) > 0),
        default=2.0,
    )
    ratio.SetMaximum(min(max(2.0, 1.2 * finite_max), 20.0))
    ratio.Draw("E1 X0")
    unity = ROOT.TLine(float(feature["x_min"]), 1.0, float(feature["x_max"]), 1.0)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive.extend([legend, header, ratio, unity])
    canvas.Modified()
    canvas.Update()


def draw_nminus1(canvas, source, selection_name, field, feature):
    base = f"diagnostics/nminus1/{selection_name}/{field}"
    histograms = [
        common.clone_histogram(
            source, f"{base}/passes_requirement", f"{selection_name}_{field}_pass"
        ),
        common.clone_histogram(
            source, f"{base}/fails_requirement", f"{selection_name}_{field}_fail"
        ),
    ]
    expression = str(
        selection_value(source, selection_name, f"cut__{field}__expression")
    )
    draw_selection_1d(
        canvas,
        source,
        selection_name,
        MAXIMUM_FIELDS[field],
        feature,
        histograms=histograms,
        labels=(f"Pass: {expression}", f"Fail: !({expression})"),
        header_text=f"N-1 diagnostic for {selection_name}: {field}",
        ratio_title="Fail / pass",
    )


def draw_selection_2d(
    canvas, source, selection_name, feature_name, feature, production=None
):
    histograms = selected_histograms(
        source, selection_name, feature_name, production
    )
    integrals = [common.normalize(histogram) for histogram in histograms]
    if production is None:
        labels = (
            str(selection_value(source, selection_name, "candidate_label")),
            str(selection_value(source, selection_name, "complement_label")),
        )
    else:
        labels = (
            str(selection_value(source, selection_name, "candidate_label")),
            f"{mc_production_value(source, production, 'label')}, passes selection",
        )
    maximum = max((histogram.GetMaximum() for histogram in histograms), default=1.0)
    positive = common.positive_minimum_2d(histograms)
    log_z = bool(feature.get("logz", False))
    z_min = max((positive or 1e-9) * 0.5, 1e-12) if log_z else 0.0
    z_max = max(maximum * (2.0 if log_z else 1.05), 1e-12)

    canvas.Clear()
    canvas.cd()
    canvas._keepalive = list(histograms)
    canvas.Divide(len(histograms), 1, 0.002, 0.002)
    for index, (histogram, label, count) in enumerate(
        zip(histograms, labels, integrals), start=1
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
        title = ROOT.TLatex()
        title.SetNDC(True)
        title.SetTextFont(42)
        title.SetTextSize(0.028)
        title.DrawLatex(0.13, 0.965, f"{label} (valid N={common.format_count(count)})")
        canvas._keepalive.extend([pad, title])
    canvas.Modified()
    canvas.Update()


def draw_mc_selection_summary(canvas, source, selection_name, production):
    total = int(mc_production_value(source, production, "baseline_count"))
    candidate = int(selection_value(
        source, selection_name, f"mc__{production}__candidate_count"
    ))
    complement = int(selection_value(
        source, selection_name, f"mc__{production}__complement_count"
    ))
    label = str(mc_production_value(source, production, "label"))
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.044)
    text.DrawLatex(0.08, 0.85, label)
    text.SetTextSize(0.030)
    lines = [
        f"Working point: {selection_name}",
        f"Baseline detector-activity events: {common.format_count(total)}",
        f"Candidate: {common.format_count(candidate)} "
        f"({100.0 * candidate / total if total else 0.0:.4f}%)",
        f"Complement: {common.format_count(complement)}",
        f"Closure: {common.format_count(candidate + complement)} = "
        f"{common.format_count(total)}",
        "",
        (
            "Unweighted event-count sum across every configured MC-muon-up production."
            if production == "all" else
            "This production is evaluated separately; no cross-production weight is applied."
        ),
    ]
    y = 0.77
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.055 if line else 0.028
    header = common.draw_header("MC-muon-up detector selection")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_mc_summary_intro(canvas, source, config):
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.044)
    text.DrawLatex(0.08, 0.87, "MC-muon-up production summary")
    text.SetTextSize(0.028)
    lines = [
        "Each production is validated, chained, and selected independently.",
        "No combined-production histogram or luminosity weight is applied.",
        "",
    ]
    for production in production_keys(config):
        label = str(mc_production_value(source, production, "label"))
        files = int(mc_production_value(source, production, "partition_count"))
        declared = int(mc_production_value(source, production, "declared_event_count"))
        baseline = int(mc_production_value(source, production, "baseline_count"))
        lines.append(
            f"{label}: {files} files, {common.format_count(declared)} entries, "
            f"{common.format_count(baseline)} baseline events"
        )
    y = 0.79
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.055 if line else 0.027
    header = common.draw_header("Separate production provenance")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_mc_efficiency_summary(canvas, source, config, selection_name):
    productions = production_keys(config)
    histogram = ROOT.TH1D(
        f"mc_efficiency_{selection_name}", "", len(productions),
        0.5, len(productions) + 0.5,
    )
    histogram.SetDirectory(0)
    for index, production in enumerate(productions, start=1):
        total = int(mc_production_value(source, production, "baseline_count"))
        candidate = int(selection_value(
            source, selection_name, f"mc__{production}__candidate_count"
        ))
        histogram.GetXaxis().SetBinLabel(
            index, str(mc_production_value(source, production, "label"))
        )
        histogram.SetBinContent(
            index, 100.0 * candidate / total if total else 0.0
        )
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = [histogram]
    canvas.SetBottomMargin(0.25)
    canvas.SetLeftMargin(0.12)
    histogram.SetMinimum(0.0)
    histogram.SetMaximum(max(105.0, 1.25 * histogram.GetMaximum()))
    histogram.SetLineColor(ROOT.kBlue + 1)
    histogram.SetFillColorAlpha(ROOT.kBlue - 9, 0.55)
    histogram.GetXaxis().LabelsOption("v")
    histogram.SetTitle(";MC production;Candidate efficiency [%]")
    histogram.Draw("HIST TEXT0")
    header = common.draw_header(f"MC selection efficiency: {selection_name}")
    canvas._keepalive.append(header)
    canvas.Modified()
    canvas.Update()
    canvas.SetBottomMargin(0.10)
    canvas.SetLeftMargin(0.10)


def draw_diagnostics_intro(canvas, source, config, selections):
    total = int(common.object_value(source, "real_data_dq_count"))
    mc_total = int(common.object_value(source, "mc_muon_up_baseline_count"))
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = []
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.045)
    text.DrawLatex(0.08, 0.85, "Detector-only v5 diagnostics")
    text.SetTextSize(0.030)
    lines = [
        f"Complete RealDataDQ sample: {common.format_count(total)} events",
        f"MC-muon-up detector-activity reference: {common.format_count(mc_total)} events",
        "No reconstructed-track branch is used in any selection.",
        "",
        "Topology working points:",
    ]
    lines.extend(f"  {name}: {definition['label']}" for name, definition in selections.items())
    lines.extend((
        "",
        "Threshold scans apply one upper count limit after the stated base topology.",
        "They are diagnostics only; their values are not final physics cuts.",
        "MC productions are validated and reported separately.",
        "Empty MC candidate selections are retained as diagnostics.",
    ))
    y = 0.78
    for line in lines:
        text.DrawLatex(0.10, y, line)
        y -= 0.052 if line else 0.026
    header = common.draw_header("Selection and threshold diagnostics")
    canvas._keepalive.extend([text, header])
    canvas.Modified()
    canvas.Update()


def draw_cutflow(canvas, source, selection_name):
    histogram = common.clone_histogram(
        source,
        f"diagnostics/cutflow/{selection_name}/real_data_fraction_pct",
        f"cutflow_{selection_name}",
    )
    counts = common.clone_histogram(
        source, f"diagnostics/cutflow/{selection_name}/counts", f"counts_{selection_name}"
    )
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = [histogram, counts]
    canvas.SetBottomMargin(0.26)
    canvas.SetLeftMargin(0.12)
    histogram.SetLineColor(ROOT.kBlue + 1)
    histogram.SetMarkerColor(ROOT.kBlue + 1)
    histogram.SetMarkerStyle(20)
    histogram.SetLineWidth(2)
    histogram.SetMinimum(0.0)
    histogram.SetMaximum(110.0)
    histogram.GetXaxis().LabelsOption("v")
    histogram.SetTitle(";Cumulative detector requirement;Fraction of RealDataDQ [%]")
    histogram.Draw("HIST TEXT0")
    legend = ROOT.TLegend(0.56, 0.78, 0.92, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(
        histogram,
        f"Final N={common.format_count(counts.GetBinContent(counts.GetNbinsX()))}",
        "lp",
    )
    legend.Draw()
    header = common.draw_header(f"Cumulative cutflow: {selection_name}")
    canvas._keepalive.extend([legend, header])
    canvas.Modified()
    canvas.Update()
    canvas.SetBottomMargin(0.10)
    canvas.SetLeftMargin(0.10)


def draw_threshold_scan(canvas, source, config, scan_name):
    base_fraction = common.clone_histogram(
        source,
        f"diagnostics/threshold_scans/{scan_name}/base_coverage_fraction_pct",
        f"scan_base_{scan_name}",
    )
    real_fraction = common.clone_histogram(
        source,
        f"diagnostics/threshold_scans/{scan_name}/real_data_fraction_pct",
        f"scan_real_{scan_name}",
    )
    counts = common.clone_histogram(
        source, f"diagnostics/threshold_scans/{scan_name}/counts", f"scan_counts_{scan_name}"
    )
    scan = config["multiplicity_scans"][scan_name]
    canvas.Clear()
    canvas.cd()
    canvas._keepalive = [base_fraction, real_fraction, counts]
    canvas.SetBottomMargin(0.16)
    canvas.SetLeftMargin(0.12)
    for histogram, color, marker in zip(
        (base_fraction, real_fraction), COLORS, MARKERS
    ):
        style_1d(histogram, color, marker)
    base_fraction.SetMinimum(0.0)
    base_fraction.SetMaximum(110.0)
    base_fraction.SetTitle(
        f";{scan['label']};Surviving fraction [%]"
    )
    base_fraction.Draw("E1")
    real_fraction.Draw("E1 SAME")
    legend = ROOT.TLegend(0.50, 0.73, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(
        base_fraction,
        f"Of {scan['base_coverage']} "
        f"(final N={common.format_count(counts.GetBinContent(counts.GetNbinsX()))})",
        "lp",
    )
    legend.AddEntry(real_fraction, "Of complete RealDataDQ", "lp")
    legend.Draw()
    header = common.draw_header(f"Provisional one-variable scan: {scan_name}")
    canvas._keepalive.extend([legend, header])
    canvas.Modified()
    canvas.Update()
    canvas.SetBottomMargin(0.10)
    canvas.SetLeftMargin(0.10)


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
    hist_config = common.load_yaml(args.hist_config)
    config = common.load_yaml(args.region_config)
    validate_region_config(config)
    features = configured_v5_features(hist_config, config)
    selections = configured_selections(config)
    mc_comparisons = {
        f"{selection}__mc__{production}"
        for selection in selections
        for production in production_keys(config)
    }
    mc_comparisons.update(aggregate_comparisons(config))
    valid = {"diagnostics", "mc_muon_up_summary", *selections, *mc_comparisons}
    if args.comparison not in valid:
        raise KeyError(f"Unknown comparison {args.comparison!r}; choose from {sorted(valid)}")

    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open {args.input}")
    if str(common.object_value(source, "study_version")) != str(config["version"]):
        raise RuntimeError("ROOT/config study-version mismatch")
    if int(common.object_value(source, "uses_reconstructed_tracks")) != 0:
        raise RuntimeError("Detector-only v5 file unexpectedly uses tracks")
    if int(common.object_value(source, "uses_mc_muon_up")) != 1:
        raise RuntimeError("Detector-only v5 file is missing its MC-muon-up dataset")

    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".pdf", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("detector_selection_v5", "detector_selection_v5", 1500, 900)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        if args.comparison == "diagnostics":
            draw_diagnostics_intro(canvas, source, config, selections)
            canvas.Print(temporary)
            page_count += 1
            for selection_name in selections:
                draw_cutflow(canvas, source, selection_name)
                canvas.Print(temporary)
                page_count += 1
            for scan_name in config["multiplicity_scans"]:
                draw_threshold_scan(canvas, source, config, scan_name)
                canvas.Print(temporary)
                page_count += 1
        elif args.comparison == "mc_muon_up_summary":
            draw_mc_summary_intro(canvas, source, config)
            canvas.Print(temporary)
            page_count += 1
            for selection_name in selections:
                draw_mc_efficiency_summary(
                    canvas, source, config, selection_name
                )
                canvas.Print(temporary)
                page_count += 1
        else:
            if "__mc__" in args.comparison:
                selection_name, production = args.comparison.split("__mc__", 1)
                draw_mc_selection_summary(
                    canvas, source, selection_name, production
                )
            else:
                selection_name, production = args.comparison, None
                draw_selection_summary(canvas, source, selection_name)
            canvas.Print(temporary)
            page_count += 1
            for feature_name, feature in features.items():
                if int(feature.get("dimension", 1)) == 1:
                    draw_selection_1d(
                        canvas, source, selection_name, feature_name, feature,
                        production=production,
                        ratio_title=(
                            "MC candidate / data candidate"
                            if production else "Complement / candidate"
                        ),
                    )
                else:
                    draw_selection_2d(
                        canvas, source, selection_name, feature_name, feature,
                        production=production,
                    )
                canvas.Print(temporary)
                page_count += 1
            selection = selections[selection_name]
            if production is None and selection["kind"] == "selection":
                for field, variable in MAXIMUM_FIELDS.items():
                    if field not in selection["components"]:
                        continue
                    draw_nminus1(
                        canvas, source, selection_name, field, features[variable]
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
