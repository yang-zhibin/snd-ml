#!/usr/bin/env python3
"""Create the Region V1 Veto edge-muon selection tables and PyROOT report."""

from __future__ import annotations

import argparse
from pathlib import Path

import ROOT

from abcd_ds_scifi_selection import configured_cutflow, configured_selection, load_yaml
from make_abcd_ds_scifi_rejection_report import (
    aggregate_cutflow,
    atomic_csv,
    atomic_text,
    clone,
    cutflow_histogram,
    cutflow_table_body,
    draw_region_overlay,
    rows_from,
    scientific_three_significant,
    table_page,
    validate_cutflow,
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


def percent(numerator, denominator):
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.6g}%"


def definition_rows(config):
    edge = config["selection"]["edge"]
    boundaries = edge["boundaries_cm"]
    return [
        ["Selected population", "edge_muon (the population kept by this study)"],
        ["Valid position", f"plane hit count > 0 and average position > {edge['valid_position_min_exclusive']:g} cm"],
        ["Veto 1 y edges", f"y <= {boundaries['veto1_y']['low_edge_max']:.6f} or y >= {boundaries['veto1_y']['high_edge_min']:.6f} cm"],
        ["Veto 2 y edges", f"y <= {boundaries['veto2_y']['low_edge_max']:.6f} or y >= {boundaries['veto2_y']['high_edge_min']:.6f} cm"],
        ["Veto 3 x edges", f"x <= {boundaries['veto3_x']['low_edge_max']:.6f} or x >= {boundaries['veto3_x']['high_edge_min']:.6f} cm"],
        ["Both y planes", "Both must identify the same low or high y edge"],
        ["One y plane", "Use the one available y plane"],
        ["Missing orientation", "Neutral: the available orientation decides"],
        ["All positions missing", "Not selected because any_valid is false"],
        ["Final selection", "edge_muon = any_valid AND y_edge AND x_edge"],
    ]


def availability_body(data_rows):
    fields = [
        "category_both_orientations", "category_y_only",
        "category_x_only", "category_no_position",
    ]
    return [
        [row["label"], *(f"{int(row[field]):,}" for field in fields)]
        for row in data_rows if row["population"] != "total"
    ]


def data_efficiency_body(config, data_rows):
    reference = float(config["input"]["expected_full_data_luminosity"])
    body = []
    for row in data_rows:
        if row["population"] == "total":
            continue
        scale = reference / float(row["exposure_fb"])
        before = int(row["n_before"])
        edge = int(row["n_selected_edge"])
        nonedge = int(row["n_nonedge"])
        body.append([
            row["label"],
            scientific_three_significant(before * scale),
            scientific_three_significant(edge * scale),
            scientific_three_significant(nonedge * scale),
            percent(edge, before),
        ])
    return reference, body


def mc_efficiency_body(mc_rows):
    return [
        [
            row["label"], row["dataset"],
            f"{int(row['n_before']):,}", f"{int(row['n_selected_edge']):,}",
            percent(int(row["n_selected_edge"]), int(row["n_before"])),
            f"{float(row['sumw_selected_edge']):.4g}",
        ]
        for row in mc_rows
    ]


def append_cutflow(lines, title, cutflow, counts):
    lines += [
        f"## {title}", "",
        "Requirements are cumulative. The final row is the selected edge-muon population.", "",
        "| Step | Requirement | Remaining | Failing requirement | Step efficiency | Cumulative efficiency |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in cutflow_table_body(cutflow, counts):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def markdown(config, selection, data_rows, mc_rows, cutflow, data_counts, mc_counts):
    lines = [
        "# Region V1 Veto edge-muon selection", "",
        "The selected/kept population is edge_muon. The complementary non_edge "
        "population is stored only for closure and comparison.", "",
        "## Selection", "",
        f"- Base: {selection['base_expression']}",
        f"- Edge selection: {selection['tag_expression']}",
        f"- Complement: {selection['keep_expression']}", "",
        "Missing one orientation is neutral and lets the other orientation "
        "decide. Events with no valid Veto position are not selected.", "",
        "For every population, N_before = N_edge + N_nonedge. Edge selection "
        "efficiency is N_edge/N_before. No confidence interval is reported.", "",
    ]
    append_cutflow(lines, "Data sequential selection", cutflow, data_counts)
    append_cutflow(lines, "CC_nue sequential selection", cutflow, mc_counts)
    lines += [
        "## Data availability and selection", "",
        "| Region | Both orientations | y only | x only | No position | Before | Edge selected | Non-edge | Edge efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data_rows:
        if row["population"] == "total":
            continue
        lines.append(
            f"| {row['label']} | {row['category_both_orientations']} | "
            f"{row['category_y_only']} | {row['category_x_only']} | "
            f"{row['category_no_position']} | {row['n_before']} | "
            f"{row['n_selected_edge']} | {row['n_nonedge']} | "
            f"{float(row['edge_efficiency']):.6g} |"
        )
    lines += [
        "", "## CC_nue acceptance", "",
        "MC weighted yields use target region exposure/source MC luminosity.",
        "", "| Region | Before | Edge selected | Edge efficiency | Weighted edge yield |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in mc_rows:
        lines.append(
            f"| {row['label']} | {row['n_before']} | {row['n_selected_edge']} | "
            f"{float(row['edge_efficiency']):.6g} | "
            f"{float(row['sumw_selected_edge']):.6g} |"
        )
    lines += [
        "", "## Interpretation boundary", "",
        "The existing no-Veto Region V1 populations have no valid Veto "
        "position and therefore cannot enter the edge-muon sample. This "
        "product is a control-sample diagnostic, not a new validated ABCD "
        "construction.", "",
    ]
    return "\n".join(lines)


def build(args):
    config = load_yaml(args.config)
    selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    if config["selection"].get("mode") != "veto_edge_muon":
        raise ValueError("Report requires selection.mode=veto_edge_muon")
    if len(args.mc_root) != len(args.mc_summary):
        raise ValueError("MC ROOT/summary counts differ")

    data_rows = rows_from([args.data_summary])
    mc_rows = rows_from(args.mc_summary)
    all_rows = data_rows + mc_rows
    if not all_rows or any(
        row["status"] != "PASS"
        or row["selection_fingerprint"] != selection["fingerprint"]
        for row in all_rows
    ):
        raise ValueError("Input validation/fingerprint failure")
    expected_mc = len(config["regions"]) * len(config["mc_samples"])
    if len(mc_rows) != expected_mc:
        raise ValueError(f"Expected {expected_mc} MC summaries, found {len(mc_rows)}")
    total = [row for row in data_rows if row["population"] == "total"]
    if len(total) != 1:
        raise ValueError("Expected one total-data row")
    data_counts = aggregate_cutflow(total, cutflow)
    mc_counts = aggregate_cutflow(mc_rows, cutflow)
    validate_cutflow(data_counts, int(total[0]["n_tagged"]), "data")
    validate_cutflow(
        mc_counts, sum(int(row["n_tagged"]) for row in mc_rows), "MC"
    )
    for row in all_rows:
        if int(row["n_before"]) != int(row["n_selected_edge"]) + int(row["n_nonedge"]):
            raise ValueError(f"Count closure failure for {row['population']}")
        categories = sum(
            int(row[f"category_{key}"])
            for key in ("both_orientations", "y_only", "x_only", "no_position")
        )
        if categories != int(row["n_before"]):
            raise ValueError(f"Availability closure failure for {row['population']}")

    atomic_csv(args.output_summary, all_rows)
    atomic_text(
        args.output_markdown,
        markdown(config, selection, data_rows, mc_rows, cutflow, data_counts, mc_counts),
    )

    data_file = ROOT.TFile.Open(args.data_root, "READ")
    if not data_file or data_file.IsZombie():
        raise OSError(args.data_root)
    output = Path(args.output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_file = ROOT.TFile.Open(str(output), "RECREATE")
    if not plot_file or plot_file.IsZombie():
        raise OSError(output)
    data_cutflow = cutflow_histogram(
        "cutflow_data", "Data edge-muon selection", cutflow, data_counts
    )
    mc_cutflow = cutflow_histogram(
        "cutflow_cc_nue", "CC_nue edge-muon selection", cutflow, mc_counts
    )
    plot_file.cd()
    data_cutflow.Write()
    mc_cutflow.Write()

    pdf = Path(args.output_pdf)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("c_veto_edge", "", 1100, 800)
    canvas.Print(str(pdf) + "[")
    keep = table_page(
        canvas, "Region V1 Veto edge-muon selection",
        ["Quantity", "Definition"], definition_rows(config), [0.25, 0.75],
    )
    canvas.Print(str(pdf))
    headers = [
        "Step", "New requirement", "Remaining", "Failing",
        "#splitline{Step}{efficiency}", "#splitline{Cumulative}{efficiency}",
    ]
    widths = [0.05, 0.39, 0.13, 0.12, 0.14, 0.17]
    keep = table_page(
        canvas, "Region V1 data sequential edge selection", headers,
        cutflow_table_body(cutflow, data_counts), widths,
    )
    canvas.Print(str(pdf))
    keep = table_page(
        canvas, "CC_{#nu_{e}} sequential edge selection (all regions)", headers,
        cutflow_table_body(cutflow, mc_counts), widths,
    )
    canvas.Print(str(pdf))
    keep = table_page(
        canvas, "Data Veto-position availability",
        ["Region", "y and x", "y only", "x only", "No position"],
        availability_body(data_rows), [0.24, 0.19, 0.17, 0.17, 0.23],
    )
    canvas.Print(str(pdf))
    reference, body = data_efficiency_body(config, data_rows)
    keep = table_page(
        canvas,
        f"Data edge selection (counts at L_{{ref}} = {reference:.3f} fb^{{-1}})",
        ["Region", "Before", "Edge selected", "Non-edge", "Edge efficiency"],
        body, [0.22, 0.18, 0.21, 0.18, 0.21],
    )
    canvas.Print(str(pdf))
    keep = table_page(
        canvas, "CC_{#nu_{e}} edge-selection acceptance",
        ["Region", "Sample", "Before", "Edge selected", "Edge efficiency", "Weighted edge"],
        mc_efficiency_body(mc_rows), [0.18, 0.15, 0.15, 0.18, 0.18, 0.16],
    )
    canvas.Print(str(pdf))

    colors = [
        ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kAzure + 7, ROOT.kOrange + 7,
        ROOT.kRed + 1, ROOT.kOrange + 1, ROOT.kMagenta + 1, ROOT.kSpring + 5,
    ]
    edge_canvas = ROOT.TCanvas("c_veto_edge_selected", "", 1500, 800)
    edge_objects = draw_region_overlay(
        edge_canvas, pdf, data_file, plot_file, config, colors, "tagged",
        "Selected Veto edge-muon data by category", "analysis",
    )
    nonedge_canvas = ROOT.TCanvas("c_veto_edge_nonedge", "", 1500, 800)
    nonedge_objects = draw_region_overlay(
        nonedge_canvas, pdf, data_file, plot_file, config, colors, "retained",
        "Complementary non-edge data by category", "original_plot",
    )

    for region in config["regions"]:
        before = clone(data_file, f"h_analysis_{region}_before")
        edge = clone(data_file, f"h_analysis_{region}_tagged")
        efficiency = edge.Clone(f"edge_efficiency_{region}")
        efficiency.Divide(edge, before, 1.0, 1.0, "B")
        plot_file.cd()
        before.Write(f"before_{region}")
        edge.Write(f"edge_selected_{region}")
        efficiency.Write()
    canvas.Print(str(pdf) + "]")
    plot_file.Close()
    data_file.Close()
    print(
        f"Wrote {args.output_summary}, {args.output_root}, "
        f"{args.output_pdf}, and {args.output_markdown}"
    )


if __name__ == "__main__":
    build(parse_args())
