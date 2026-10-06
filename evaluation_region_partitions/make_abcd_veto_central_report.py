#!/usr/bin/env python3
"""Create the Region V1 central-muon rejection PyROOT report."""

from __future__ import annotations

import argparse
from pathlib import Path

import ROOT

from abcd_ds_scifi_selection import configured_selection, load_yaml
from make_abcd_ds_scifi_rejection_report import (
    atomic_csv,
    atomic_text,
    clone,
    draw_region_overlay,
    rows_from,
    scientific_three_significant,
    table_page,
)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)


DEFINITION_LINES = [
    "valid_veto1 = (",
    "    count_veto1 > 0",
    "    and avg_veto1_y > -900",
    ")",
    "",
    "valid_veto2 = (",
    "    count_veto2 > 0",
    "    and avg_veto2_y > -900",
    ")",
    "",
    "valid_veto3 = (",
    "    count_veto3 > 0",
    "    and avg_veto3_x > -900",
    ")",
    "",
    "v1_central = (",
    "    valid_veto1",
    "    and 24.835139 < avg_veto1_y < 46.867571",
    ")",
    "",
    "v2_central = (",
    "    valid_veto2",
    "    and 22.784356 < avg_veto2_y < 44.816788",
    ")",
    "",
    "x_central = (",
    "    valid_veto3",
    "    and -37.370290 < avg_veto3_x < -15.755074",
    ")",
    "",
    "if valid_veto1 and valid_veto2:",
    "    # Both measured y positions must be central.",
    "    y_central = v1_central and v2_central",
    "elif valid_veto1:",
    "    y_central = v1_central",
    "elif valid_veto2:",
    "    y_central = v2_central",
    "else:",
    "    # Central classification requires y information.",
    "    y_central = False",
    "",
    "# Both orientations are mandatory.",
    "central_muon = y_central and x_central",
    "",
    "reject_event = central_muon",
    "keep_event = not central_muon",
]


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


def definition_page(canvas):
    canvas.Clear()
    title = ROOT.TText()
    title.SetNDC(True)
    title.SetTextFont(62)
    title.SetTextSize(0.040)
    title.DrawText(0.055, 0.950, "Exact central-muon rejection definition")
    objects = [title]
    y = 0.900
    for line in DEFINITION_LINES:
        text = ROOT.TText()
        text.SetNDC(True)
        text.SetTextFont(82)
        text.SetTextSize(0.014)
        text.DrawText(0.065, y, line)
        objects.append(text)
        y -= 0.0165
    canvas.Update()
    return objects


def availability_body(data_rows):
    fields = [
        "category_both_orientations", "category_y_only",
        "category_x_only", "category_no_position",
    ]
    return [
        [row["label"], *(f"{int(row[field]):,}" for field in fields)]
        for row in data_rows if row["population"] != "total"
    ]


def data_body(config, data_rows):
    reference = float(config["input"]["expected_full_data_luminosity"])
    body = []
    for row in data_rows:
        if row["population"] == "total":
            continue
        scale = reference / float(row["exposure_fb"])
        before = int(row["n_before"])
        rejected = int(row["n_central_muon"])
        kept = int(row["n_kept"])
        body.append([
            row["label"],
            scientific_three_significant(before * scale),
            scientific_three_significant(rejected * scale),
            scientific_three_significant(kept * scale),
            percent(rejected, before),
        ])
    return reference, body


def mc_body(rows):
    return [
        [
            row["label"], row["dataset"],
            f"{int(row['n_before']):,}", f"{int(row['n_central_muon']):,}",
            f"{int(row['n_kept']):,}",
            percent(int(row["n_central_muon"]), int(row["n_before"])),
        ]
        for row in rows
    ]


def markdown(config, selection, data_rows, mc_rows):
    total = next(row for row in data_rows if row["population"] == "total")
    lines = [
        "# Region V1 central-muon rejection", "",
        "## Exact definition", "", "~~~python", *DEFINITION_LINES, "~~~", "",
        "## Single central_muon cut efficiency", "",
        "| Cut | Before | Central muon | Efficiency | Kept after rejection |",
        "|---|---:|---:|---:|---:|",
        f"| central_muon | {total['n_before']} | {total['n_central_muon']} | "
        f"{float(total['central_muon_efficiency']):.6g} | {total['n_kept']} |",
        "", "The aggregate combines stored Region V1 partitions with their "
        "configured exposures and is a processing diagnostic.", "",
        "## Per-region data results", "",
        "| Region | Before | Central muon rejected | Kept | Central-muon efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in data_rows:
        if row["population"] == "total":
            continue
        lines.append(
            f"| {row['label']} | {row['n_before']} | {row['n_central_muon']} | "
            f"{row['n_kept']} | {float(row['central_muon_efficiency']):.6g} |"
        )
    lines += [
        "", "## CC_nue acceptance", "",
        "| Region | Before | Central muon rejected | Kept | Central-muon efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in mc_rows:
        lines.append(
            f"| {row['label']} | {row['n_before']} | {row['n_central_muon']} | "
            f"{row['n_kept']} | {float(row['central_muon_efficiency']):.6g} |"
        )
    lines += [
        "", "## Interpretation boundary", "",
        "A central muon requires both central y information and a central x "
        "measurement. Missing x, missing all y, edge-like, and inconsistent "
        "events are kept. No confidence interval is reported.", "",
    ]
    return "\n".join(lines)


def build(args):
    config = load_yaml(args.config)
    selection = configured_selection(config)
    if config["selection"].get("mode") != "veto_central_muon_rejection":
        raise ValueError("Report requires central-muon rejection mode")
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
    if len(mc_rows) != len(config["regions"]) * len(config["mc_samples"]):
        raise ValueError("Incomplete MC coverage")
    total = [row for row in data_rows if row["population"] == "total"]
    if len(total) != 1:
        raise ValueError("Expected one total-data row")
    for row in all_rows:
        if int(row["n_before"]) != int(row["n_central_muon"]) + int(row["n_kept"]):
            raise ValueError(f"Count closure failure for {row['population']}")
        availability = sum(
            int(row[f"category_{key}"])
            for key in ("both_orientations", "y_only", "x_only", "no_position")
        )
        if availability != int(row["n_before"]):
            raise ValueError(f"Availability closure failure for {row['population']}")

    atomic_csv(args.output_summary, all_rows)
    atomic_text(
        args.output_markdown,
        markdown(config, selection, data_rows, mc_rows),
    )
    data_file = ROOT.TFile.Open(args.data_root, "READ")
    if not data_file or data_file.IsZombie():
        raise OSError(args.data_root)
    output = Path(args.output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_file = ROOT.TFile.Open(str(output), "RECREATE")
    if not plot_file or plot_file.IsZombie():
        raise OSError(output)
    plot_file.cd()
    ROOT.TNamed("central_muon_expression", selection["tag_expression"]).Write()
    ROOT.TNamed("keep_expression", selection["keep_expression"]).Write()

    pdf = Path(args.output_pdf)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("c_veto_central", "", 1100, 800)
    canvas.Print(str(pdf) + "[")
    keep = definition_page(canvas)
    canvas.Print(str(pdf))
    total_row = total[0]
    keep = table_page(
        canvas, "Single central_muon cut efficiency",
        ["Cut", "Before", "Central muon", "Efficiency", "Kept"],
        [[
            "central_muon", f"{int(total_row['n_before']):,}",
            f"{int(total_row['n_central_muon']):,}",
            percent(
                int(total_row["n_central_muon"]), int(total_row["n_before"])
            ),
            f"{int(total_row['n_kept']):,}",
        ]],
        [0.23, 0.19, 0.21, 0.18, 0.19],
    )
    canvas.Print(str(pdf))
    keep = table_page(
        canvas, "Data Veto-position availability",
        ["Region", "y and x", "y only", "x only", "No position"],
        availability_body(data_rows), [0.24, 0.19, 0.17, 0.17, 0.23],
    )
    canvas.Print(str(pdf))
    reference, rows = data_body(config, data_rows)
    keep = table_page(
        canvas,
        f"Data central-muon rejection (counts at L_{{ref}} = {reference:.3f} fb^{{-1}})",
        ["Region", "Before", "Central rejected", "Kept", "Central efficiency"],
        rows, [0.22, 0.18, 0.22, 0.18, 0.20],
    )
    canvas.Print(str(pdf))
    keep = table_page(
        canvas, "CC_{#nu_{e}} central-muon rejection",
        ["Region", "Sample", "Before", "Central rejected", "Kept", "Central efficiency"],
        mc_body(mc_rows), [0.18, 0.14, 0.15, 0.20, 0.14, 0.19],
    )
    canvas.Print(str(pdf))

    colors = [
        ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kAzure + 7, ROOT.kOrange + 7,
        ROOT.kRed + 1, ROOT.kOrange + 1, ROOT.kMagenta + 1, ROOT.kSpring + 5,
    ]
    kept_canvas = ROOT.TCanvas("c_central_kept", "", 1500, 800)
    kept_objects = draw_region_overlay(
        kept_canvas, pdf, data_file, plot_file, config, colors, "retained",
        "Data kept after central-muon rejection", "original_plot",
    )
    rejected_canvas = ROOT.TCanvas("c_central_rejected", "", 1500, 800)
    rejected_objects = draw_region_overlay(
        rejected_canvas, pdf, data_file, plot_file, config, colors, "tagged",
        "Rejected central-muon data", "analysis",
    )
    for region in config["regions"]:
        before = clone(data_file, f"h_analysis_{region}_before")
        central = clone(data_file, f"h_analysis_{region}_tagged")
        kept_hist = clone(data_file, f"h_analysis_{region}_retained")
        plot_file.cd()
        before.Write(f"before_{region}")
        central.Write(f"central_muon_{region}")
        kept_hist.Write(f"kept_{region}")
    canvas.Print(str(pdf) + "]")
    plot_file.Close()
    data_file.Close()
    print(
        f"Wrote {args.output_summary}, {args.output_root}, "
        f"{args.output_pdf}, and {args.output_markdown}"
    )


if __name__ == "__main__":
    build(parse_args())
