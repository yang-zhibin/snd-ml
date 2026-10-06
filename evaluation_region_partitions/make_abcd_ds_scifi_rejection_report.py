#!/usr/bin/env python3
"""Create the validated DS--SciFi rejection tables and PyROOT report."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import ROOT

from abcd_ds_scifi_selection import (
    configured_cutflow,
    configured_selection,
    cutflow_field,
    load_yaml,
)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-summary", required=True)
    parser.add_argument("--mc-root", action="append", default=[])
    parser.add_argument("--mc-summary", action="append", default=[])
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def rows_from(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as stream: rows.extend(csv.DictReader(stream))
    return rows


def atomic_text(path, text):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False)
    temporary = handle.name
    try:
        with handle: handle.write(text)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_csv(path, rows):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False)
    temporary = handle.name
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def latex(x, y, text, size=0.035, align=11):
    item = ROOT.TLatex(); item.SetNDC(); item.SetTextSize(size); item.SetTextAlign(align); item.DrawLatex(x, y, text); return item


def table_page(canvas, title, headers, body, widths=None):
    canvas.Clear(); canvas.cd(); latex(0.06, 0.94, title, 0.045)
    widths = widths or [1 / len(headers)] * len(headers)
    left, usable = 0.05, 0.90
    positions, cursor = [], left
    for width in widths:
        positions.append(cursor); cursor += usable * width
    y = 0.875
    for x, header in zip(positions, headers): latex(x, y, header, 0.026)
    multiline_header = any("#splitline" in str(header) for header in headers)
    line_offset = 0.040 if multiline_header else 0.015
    body_offset = 0.078 if multiline_header else 0.052
    line = ROOT.TLine(left, y - line_offset, left + usable, y - line_offset); line.SetLineColor(ROOT.kGray + 1); line.Draw()
    y -= body_offset
    size = min(0.024, 0.70 / max(len(body), 1))
    keep = [line]
    for row in body:
        for x, value in zip(positions, row): keep.append(latex(x, y, str(value), size))
        y -= max(size * 1.7, 0.038)
    canvas.Update(); return keep


def clone(source, name):
    obj = source.Get(name)
    if not obj: raise KeyError(name)
    result = obj.Clone(name); result.SetDirectory(0); return result


def aggregate_cutflow(rows, cutflow):
    return [
        sum(int(row[cutflow_field(stage)]) for row in rows)
        for stage in cutflow
    ]


def validate_cutflow(counts, tagged, label):
    if (
        not counts
        or counts[-1] != tagged
        or any(left < right for left, right in zip(counts, counts[1:]))
    ):
        raise ValueError(f"Invalid {label} cutflow: {counts}; tagged={tagged}")


def efficiency(numerator, denominator):
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.6g}%"


def scientific_three_significant(value):
    return f"{float(value):.2e}"


def cutflow_table_body(cutflow, counts):
    rows = []
    for index, (stage, count) in enumerate(zip(cutflow, counts)):
        previous = counts[index - 1] if index else None
        rows.append([
            index, stage["label"], f"{count:,}",
            "--" if previous is None else f"{previous - count:,}",
            "--" if previous is None else efficiency(count, previous),
            efficiency(count, counts[0]),
        ])
    return rows


def luminosity_normalized_data_body(config, data_rows):
    reference = float(config["input"]["expected_full_data_luminosity"])
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError(f"Invalid reference luminosity: {reference}")
    body = []
    for row in data_rows:
        if row["population"] == "total":
            continue
        exposure = float(row["exposure_fb"])
        if not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(
                f"Invalid exposure for {row['population']}: {exposure}"
            )
        scale = reference / exposure
        body.append([
            row["label"],
            scientific_three_significant(int(row["n_before"]) * scale),
            scientific_three_significant(int(row["n_tagged"]) * scale),
            scientific_three_significant(int(row["n_retained"]) * scale),
            f"{float(row['removed_fraction']):.4f}",
        ])
    return reference, body


def cutflow_histogram(name, title, cutflow, counts):
    hist = ROOT.TH1D(name, title, len(cutflow), 0.0, float(len(cutflow)))
    hist.SetDirectory(0)
    for index, (stage, count) in enumerate(zip(cutflow, counts), start=1):
        hist.SetBinContent(index, count)
        hist.SetBinError(index, math.sqrt(count))
        hist.GetXaxis().SetBinLabel(index, stage["key"])
    return hist


def region_description(region, label):
    veto = "no veto" if "no_veto" in region else "has veto"
    us = "no US" if "no_us" in region else "has US"
    fiducial = "inside" if region.endswith("inside") else "outside"
    return f"{label}: {veto}, {us}, {fiducial}"


def histogram_event_count(histogram):
    return histogram.Integral(0, histogram.GetNbinsX() + 1)


def legend_event_count(value):
    return "0" if value == 0 else f"{value:.3g}"


def draw_region_overlay(
    canvas, pdf, data_file, plot_file, config, colors, state, title,
    histogram_family,
):
    canvas.Clear(); canvas.cd()
    plot_pad = ROOT.TPad(f"{state}_region_plot", "", 0.0, 0.0, 0.72, 1.0)
    legend_pad = ROOT.TPad(
        f"{state}_region_legend_pad", "", 0.73, 0.0, 1.0, 1.0
    )
    plot_pad.SetLeftMargin(0.14); plot_pad.SetRightMargin(0.08)
    plot_pad.SetLogy(True); legend_pad.SetLeftMargin(0.02)
    plot_pad.Draw(); legend_pad.Draw()
    legend = ROOT.TLegend(0.02, 0.48, 0.98, 0.92)
    legend.SetBorderSize(0); legend.SetTextSize(0.036)
    histograms = []
    for index, (region, label) in enumerate(config["regions"].items()):
        hist = clone(data_file, f"h_{histogram_family}_{region}_{state}")
        hist.SetLineColor(colors[index]); hist.SetMarkerColor(colors[index])
        hist.SetMarkerStyle(20 + index % 8)
        hist.SetTitle(f"{title};SciFi density (SNDSW);Events / bin")
        count = histogram_event_count(hist)
        legend.AddEntry(
            hist,
            f"{region_description(region, label)} ({legend_event_count(count)})",
            "lep",
        )
        histograms.append((region, hist))
    positive_maximum = max(
        (hist.GetMaximum() for _, hist in histograms), default=0
    )
    plot_pad.cd()
    for index, (_, hist) in enumerate(histograms):
        if index == 0:
            hist.SetMinimum(0.5)
            hist.SetMaximum(max(5.0, 10.0 * positive_maximum))
            hist.Draw("E1")
        else:
            hist.Draw("E1 SAME")
    legend_pad.cd(); legend.Draw(); canvas.Print(str(pdf))
    plot_file.cd()
    for region, hist in histograms:
        hist.Write(f"{state}_overlay_{region}")
    legend.Write(f"{state}_region_legend")
    return [plot_pad, legend_pad, legend, *(hist for _, hist in histograms)]


def selection_definition_rows(selection, cutflow):
    scifi = "; ".join(stage["label"] for stage in cutflow[1:3])
    ds = "; ".join(
        stage["label"] for stage in cutflow if stage["key"].startswith("ds_")
    )
    return [
        ["Base", selection["base_expression"]],
        ["SciFi tag", scifi],
        ["DS tag", ds],
        ["Removed", "SciFi tag AND DS tag"],
        ["Retained", "NOT(removed tag)"],
    ]


def append_markdown_cutflow(lines, title, cutflow, counts):
    lines += [
        f"## {title}", "",
        "Counts are raw and requirements are cumulative. Removal-tag step "
        "efficiency is `N_i/N_(i-1)`; removal-tag cumulative efficiency is "
        "`N_i/N_0`. `N_(i-1)-N_i` counts events failing the newly applied "
        "tag requirement, not events removed from the analysis.", "",
        "| Step | Newly applied requirement | Events remaining | Failing requirement | Removal-tag step efficiency | Removal-tag cumulative efficiency |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in cutflow_table_body(cutflow, counts):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def markdown(
    config, selection, data_rows, mc_rows, cutflow, data_cut_counts,
    signal_sample, signal_cut_counts,
):
    data_scope = str(config["input"].get("data_scope_label", "Full-2024 data"))
    lines = [
        "# Region V1 DS--SciFi passing-muon-like rejection", "",
        "## Scope and frozen selection", "",
        "This diagnostic removes events tagged by SciFi and DS only; Veto and US remain exclusive ABCD axes.", "",
        f"- Base population: `{selection['base_expression']}`",
        f"- SciFi component: `{selection['scifi_expression']}`",
        f"- DS component: `{selection['ds_expression']}`",
        f"- Removed tag: `{selection['tag_expression']}`",
        f"- Retained population: `{selection['keep_expression']}`", "",
        "For every population, `N_before = N_tagged + N_retained`. The removal fraction is `N_tagged/N_before`; the keep efficiency is `N_retained/N_before`. Bounds are exact 68.27% Clopper--Pearson binomial intervals.", "",
    ]
    append_markdown_cutflow(
        lines, f"{data_scope} sequential cutflow", cutflow, data_cut_counts
    )
    append_markdown_cutflow(
        lines, f"{signal_sample} signal-MC sequential cutflow",
        cutflow, signal_cut_counts,
    )
    lines += [
        f"## {data_scope} efficiency", "",
        "The aggregate row combines stored regions with their configured exposures; it is a processing summary, not a single-luminosity physics yield.", "",
        "| Population | Exposure [fb^-1] | N before | N tagged | N retained | Removed | Keep efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data_rows:
        lines.append(f"| {row['label']} | {row['exposure_fb']} | {row['n_before']} | {row['n_tagged']} | {row['n_retained']} | {float(row['removed_fraction']):.4g} | {float(row['keep_efficiency']):.4g} [{float(row['keep_low']):.4g}, {float(row['keep_high']):.4g}] |")
    lines += ["", "## Neutrino-MC acceptance", "", "MC yields are normalized with `w = target data exposure / source MC luminosity`; statistical variances use `sum(w^2)`.", "", "| Region | Sample | Role | Raw before | Raw retained | Keep efficiency | Weighted retained |", "|---|---|---|---:|---:|---:|---:|"]
    for row in mc_rows:
        lines.append(f"| {row['label']} | {row['dataset']} | {row['role']} | {row['n_before']} | {row['n_retained']} | {float(row['keep_efficiency']):.4g} | {float(row['sumw_retained']):.4g} |")
    lines += ["", "## Interpretation boundary", "", "This report measures rejection efficiency and the retained density shapes. It does not by itself approve the selection: sparse-bin sufficiency and post-selection ABCD closure must still be rerun before using it in the background estimate.", ""]
    return "\n".join(lines)


def build(args):
    config = load_yaml(args.config); selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    data_scope = str(config["input"].get("data_scope_label", "Full-2024 data"))
    data_pdf_label = (
        "Region V1 data partitions" if "partitions" in data_scope else data_scope
    )
    if len(args.mc_root) != len(args.mc_summary): raise ValueError("MC ROOT/summary counts differ")
    data_rows = rows_from([args.data_summary]); mc_rows = rows_from(args.mc_summary)
    all_rows = data_rows + mc_rows
    if not all_rows or any(row["status"] != "PASS" or row["selection_fingerprint"] != selection["fingerprint"] for row in all_rows): raise ValueError("Input validation/fingerprint failure")
    expected_mc = len(config["regions"]) * len(config["mc_samples"])
    if len(mc_rows) != expected_mc: raise ValueError(f"Expected {expected_mc} MC summaries, found {len(mc_rows)}")
    data_total_rows = [row for row in data_rows if row["population"] == "total"]
    if len(data_total_rows) != 1: raise ValueError("Expected one total-data summary row")
    signal_samples = [
        sample for sample, definition in config["mc_samples"].items()
        if definition["role"] == "signal"
    ]
    if len(signal_samples) != 1: raise ValueError("Expected exactly one configured signal sample")
    signal_sample = signal_samples[0]
    signal_rows = [row for row in mc_rows if row["dataset"] == signal_sample]
    if {row["population"] for row in signal_rows} != set(config["regions"]):
        raise ValueError(f"Incomplete {signal_sample} region coverage")
    data_cut_counts = aggregate_cutflow(data_total_rows, cutflow)
    signal_cut_counts = aggregate_cutflow(signal_rows, cutflow)
    validate_cutflow(
        data_cut_counts, int(data_total_rows[0]["n_tagged"]), "data"
    )
    validate_cutflow(
        signal_cut_counts, sum(int(row["n_tagged"]) for row in signal_rows),
        signal_sample,
    )
    atomic_csv(args.output_summary, all_rows)
    atomic_text(args.output_markdown, markdown(
        config, selection, data_rows, mc_rows, cutflow, data_cut_counts,
        signal_sample, signal_cut_counts,
    ))

    data_file = ROOT.TFile.Open(args.data_root, "READ")
    if not data_file or data_file.IsZombie(): raise OSError(args.data_root)
    output = Path(args.output_root); output.parent.mkdir(parents=True, exist_ok=True)
    plot_file = ROOT.TFile.Open(str(output), "RECREATE")
    if not plot_file or plot_file.IsZombie(): raise OSError(output)
    plot_file.cd()
    cutflow_objects = [
        cutflow_histogram(
            "cutflow_data", f"{data_pdf_label} sequential cutflow",
            cutflow, data_cut_counts,
        ),
        cutflow_histogram(
            "cutflow_signal_mc", f"{signal_sample} signal-MC sequential cutflow",
            cutflow, signal_cut_counts,
        ),
    ]
    for histogram in cutflow_objects: histogram.Write()
    pdf = Path(args.output_pdf); pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("c_ds_scifi", "", 1100, 800)
    canvas.Print(str(pdf) + "[")
    keep = table_page(
        canvas, "Region V1 DS--SciFi rejection: frozen definition",
        ["Quantity", "Definition"], selection_definition_rows(selection, cutflow),
        [0.22, 0.78],
    ); canvas.Print(str(pdf))
    cutflow_headers = [
        "Step", "New requirement", "Remaining",
        "#splitline{Failing}{requirement}",
        "#splitline{Removal-tag}{step eff.}",
        "#splitline{Removal-tag}{cumulative eff.}",
    ]
    cutflow_widths = [0.05, 0.35, 0.14, 0.12, 0.15, 0.19]
    keep = table_page(
        canvas, f"{data_pdf_label} sequential cutflow", cutflow_headers,
        cutflow_table_body(cutflow, data_cut_counts), cutflow_widths,
    ); canvas.Print(str(pdf))
    keep = table_page(
        canvas, f"{signal_sample} signal-MC sequential cutflow (all regions)",
        cutflow_headers, cutflow_table_body(cutflow, signal_cut_counts),
        cutflow_widths,
    ); canvas.Print(str(pdf))
    reference_luminosity, data_body = luminosity_normalized_data_body(
        config, data_rows
    )
    keep = table_page(
        canvas,
        f"Data rejection efficiency (counts at L_{{ref}} = "
        f"{reference_luminosity:.3f} fb^{{-1}})",
        ["Region", "Before", "Tagged", "Retained", "Removed fraction"],
        data_body,
        [0.22, 0.18, 0.18, 0.18, 0.24],
    ); canvas.Print(str(pdf))

    colors = [ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kAzure + 7, ROOT.kOrange + 7, ROOT.kRed + 1, ROOT.kOrange + 1, ROOT.kMagenta + 1, ROOT.kSpring + 5]
    retained_canvas = ROOT.TCanvas("c_ds_scifi_retained", "", 1500, 800)
    retained_objects = draw_region_overlay(
        retained_canvas, pdf, data_file, plot_file, config, colors, "retained",
        f"Retained {data_pdf_label} by category", "original_plot",
    )
    tagged_canvas = ROOT.TCanvas("c_ds_scifi_tagged", "", 1500, 800)
    tagged_objects = draw_region_overlay(
        tagged_canvas, pdf, data_file, plot_file, config, colors, "tagged",
        f"Removed {data_pdf_label} by category", "analysis",
    )

    # Preserve the existing diagnostic ROOT objects without drawing their
    # former eight individual PDF pages.
    for region in config["regions"]:
        before = clone(data_file, f"h_original_plot_{region}_before")
        tagged = clone(data_file, f"h_original_plot_{region}_tagged")
        retained = clone(data_file, f"h_original_plot_{region}_retained")
        ratio = retained.Clone(f"retention_{region}")
        ratio.Divide(retained, before, 1.0, 1.0, "B")
        plot_file.cd(); before.Write(f"before_{region}"); tagged.Write(f"tagged_{region}"); retained.Write(f"retained_{region}"); ratio.Write()
    canvas.Print(str(pdf) + "]")
    plot_file.Close(); data_file.Close()
    print(f"Wrote {args.output_summary}, {args.output_root}, {args.output_pdf}, and {args.output_markdown}")


if __name__ == "__main__":
    build(parse_args())
