#!/usr/bin/env python3
"""Create a Region V1 distribution overview and six ratio comparisons."""

from __future__ import annotations

import argparse
import math
import os
from array import array
from pathlib import Path

import ROOT

from plot_abcd_original_binning_transfer_comparison import (
    atomic_csv,
    graph_from_rows,
    load_yaml,
    read_histogram,
    validate_binning,
)


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetEndErrorSize(4)


def default_transfer(numerator, numerator_lumi, denominator, denominator_lumi):
    """Ratio with symmetric propagation of standard sqrt(N) count errors."""
    if denominator <= 0.0:
        return None
    scale = denominator_lumi / numerator_lumi
    value = scale * numerator / denominator
    variance = scale**2 * (numerator / denominator**2 + numerator**2 / denominator**3)
    error = math.sqrt(variance)
    return value, value - error, value + error


def default_double_ratio(counts, luminosities):
    """Ratio of ratios with independent-Poisson error propagation."""
    roles = ("signal", "sb2", "sb4", "sb6")
    if any(counts[role] <= 0.0 for role in roles):
        return None
    scale = luminosities["sb2"] * luminosities["sb4"] / (luminosities["signal"] * luminosities["sb6"])
    value = counts["signal"] * counts["sb6"] / (counts["sb2"] * counts["sb4"]) * scale
    error = value * math.sqrt(sum(1.0 / counts[role] for role in roles))
    return value, value - error, value + error


def calculate_comparison_rows(histograms: dict, luminosities: dict, comparison: dict) -> list[dict]:
    left = comparison["left"]
    right = comparison["right"]
    rows = []
    axis = histograms["signal"].GetXaxis()
    for index in range(1, histograms["signal"].GetNbinsX() + 1):
        roles = [left["numerator"], left["denominator"], right["numerator"], right["denominator"]]
        counts = {role: float(histograms[role].GetBinContent(index)) for role in roles}
        left_result = default_transfer(
            counts[left["numerator"]], luminosities[left["numerator"]],
            counts[left["denominator"]], luminosities[left["denominator"]],
        )
        right_result = default_transfer(
            counts[right["numerator"]], luminosities[right["numerator"]],
            counts[right["denominator"]], luminosities[right["denominator"]],
        )
        remapped_counts = {
            "signal": counts[left["numerator"]],
            "sb2": counts[left["denominator"]],
            "sb4": counts[right["numerator"]],
            "sb6": counts[right["denominator"]],
        }
        remapped_luminosities = {
            "signal": luminosities[left["numerator"]],
            "sb2": luminosities[left["denominator"]],
            "sb4": luminosities[right["numerator"]],
            "sb6": luminosities[right["denominator"]],
        }
        double_result = default_double_ratio(remapped_counts, remapped_luminosities)

        def values(result):
            return ("", "", "") if result is None else result

        left_value, left_low, left_high = values(left_result)
        right_value, right_low, right_high = values(right_result)
        double_value, double_low, double_high = values(double_result)
        rows.append({
            "comparison_id": comparison["id"],
            "bin_index": index,
            "bin_low": float(axis.GetBinLowEdge(index)),
            "bin_high": float(axis.GetBinUpEdge(index)),
            "left_numerator_region": left["numerator"],
            "left_denominator_region": left["denominator"],
            "right_numerator_region": right["numerator"],
            "right_denominator_region": right["denominator"],
            "count_left_numerator": counts[left["numerator"]],
            "count_left_denominator": counts[left["denominator"]],
            "count_right_numerator": counts[right["numerator"]],
            "count_right_denominator": counts[right["denominator"]],
            "left_ratio": left_value,
            "left_interval_low": left_low,
            "left_interval_high": left_high,
            "right_ratio": right_value,
            "right_interval_low": right_low,
            "right_interval_high": right_high,
            "double_ratio": double_value,
            "double_interval_low": double_low,
            "double_interval_high": double_high,
            "left_status": "PASS" if left_result is not None else "UNDEFINED_ZERO_DENOMINATOR",
            "right_status": "PASS" if right_result is not None else "UNDEFINED_ZERO_DENOMINATOR",
            "double_status": "PASS" if double_result is not None else "UNDEFINED_NONPOSITIVE_COUNT",
        })
    return rows


def zero_floor_graph(rows, value_key, floor, name):
    selected = [row for row in rows if row[value_key] != "" and float(row[value_key]) == 0.0]
    if not selected:
        return None
    x = array("d", [(row["bin_low"] + row["bin_high"]) / 2.0 for row in selected])
    ex = array("d", [(row["bin_high"] - row["bin_low"]) / 2.0 for row in selected])
    y = array("d", [floor] * len(selected))
    ey = array("d", [0.0] * len(selected))
    graph = ROOT.TGraphErrors(len(selected), x, y, ex, ey)
    graph.SetName(name)
    return graph


def draw_distribution_page(canvas, histograms, luminosities, edges, page_config, root_file):
    canvas.cd()
    canvas.SetLeftMargin(0.12); canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.08); canvas.SetBottomMargin(0.12)
    canvas.SetLogy(True)

    frame = ROOT.TH1D("frame_region_distributions", page_config["title"], len(edges) - 1, array("d", edges))
    frame.SetDirectory(0)
    frame.SetMinimum(float(page_config["y_min"])); frame.SetMaximum(float(page_config["y_max"]))
    frame.GetXaxis().SetTitle(page_config["x_axis_title"])
    frame.GetYaxis().SetTitle(page_config["y_axis_title"])
    frame.GetXaxis().SetTitleSize(0.045); frame.GetXaxis().SetLabelSize(0.040)
    frame.GetYaxis().SetTitleSize(0.045); frame.GetYaxis().SetTitleOffset(1.18)
    frame.Draw("AXIS")

    legend = ROOT.TLegend(0.51, 0.48, 0.96, 0.88)
    legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(0.024)
    scaled_histograms = []
    reference_lumi = float(page_config["reference_lumi_fb"])
    for region in page_config["regions"]:
        role = region["role"]
        hist = histograms[role].Clone(f"distribution_{role}")
        hist.SetDirectory(0)
        hist.Scale(reference_lumi / luminosities[role])
        color = ROOT.TColor.GetColor(region["color"])
        hist.SetLineColor(color); hist.SetMarkerColor(color)
        hist.SetLineWidth(1); hist.SetMarkerStyle(int(region["marker"])); hist.SetMarkerSize(0.75)
        hist.Draw("E1 SAME")
        legend.AddEntry(hist, f"{region['label']} ({hist.Integral():.3g})", "lep")
        root_file.cd(); hist.Write()
        scaled_histograms.append(hist)
    legend.Draw()

    info = ROOT.TLatex(); info.SetNDC(True); info.SetTextFont(62); info.SetTextSize(0.030)
    info.DrawLatex(0.14, 0.85, f"L_{{ref}} = {reference_lumi:.4g} fb^{{-1}}")
    info.DrawLatex(0.14, 0.80, page_config["preselection_label"])
    canvas.Modified(); canvas.Update()
    ROOT.TNamed("distribution_reference_lumi_fb", f"{reference_lumi:.12g}").Write()
    return frame, legend, info, *scaled_histograms


def draw_comparison(canvas, rows, edges, comparison, plot_config, root_file):
    identifier = comparison["id"]
    canvas.cd()
    header = ROOT.TPad(f"header_{identifier}", "", 0.0, 0.80, 1.0, 1.0)
    top = ROOT.TPad(f"top_{identifier}", "", 0.0, 0.34, 1.0, 0.80)
    bottom = ROOT.TPad(f"bottom_{identifier}", "", 0.0, 0.0, 1.0, 0.34)
    header.SetLeftMargin(0.12); header.SetRightMargin(0.04); header.SetTopMargin(0.05); header.SetBottomMargin(0.02)
    top.SetLeftMargin(0.12); top.SetRightMargin(0.04); top.SetTopMargin(0.025); top.SetBottomMargin(0.025)
    bottom.SetLeftMargin(0.12); bottom.SetRightMargin(0.04); bottom.SetTopMargin(0.025); bottom.SetBottomMargin(0.28)
    header.Draw(); top.Draw(); bottom.Draw()

    left_graph = graph_from_rows(rows, "left_ratio", "left_interval_low", "left_interval_high", f"g_{identifier}_left")
    right_graph = graph_from_rows(rows, "right_ratio", "right_interval_low", "right_interval_high", f"g_{identifier}_right")
    double_graph = graph_from_rows(rows, "double_ratio", "double_interval_low", "double_interval_high", f"g_{identifier}_double")
    for graph, color, marker in (
        (left_graph, ROOT.kBlue + 1, 20),
        (right_graph, ROOT.kRed + 1, 24),
    ):
        graph.SetLineColor(color); graph.SetMarkerColor(color); graph.SetMarkerStyle(marker)
        graph.SetMarkerSize(0.9); graph.SetLineWidth(2)
    double_graph.SetLineColor(ROOT.kBlack); double_graph.SetMarkerColor(ROOT.kBlack)
    double_graph.SetMarkerStyle(20); double_graph.SetMarkerSize(0.85); double_graph.SetLineWidth(2)

    header.cd()
    title = ROOT.TLatex(); title.SetNDC(True); title.SetTextFont(62); title.SetTextSize(0.18)
    title.DrawLatex(0.12, 0.78, comparison["transfer_title"])
    subtitle = ROOT.TLatex(); subtitle.SetNDC(True); subtitle.SetTextFont(42); subtitle.SetTextSize(0.10)
    subtitle.DrawLatex(0.12, 0.56, comparison["subtitle"])
    legend = ROOT.TLegend(0.10, 0.02, 0.98, 0.48)
    legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(0.105)
    legend.AddEntry(left_graph, comparison["left"]["label"], "lep")
    legend.AddEntry(right_graph, comparison["right"]["label"], "lep")
    legend.Draw()

    top.cd()
    central_values = [float(row[key]) for row in rows for key in ("left_ratio", "right_ratio") if row[key] != ""]
    positive_values = [value for value in central_values if value > 0.0]
    requested_top_scale = str(comparison.get("top_scale", "auto"))
    if requested_top_scale not in {"auto", "linear", "log"}:
        raise ValueError(f"{identifier}.top_scale must be auto, linear, or log")
    top_log = requested_top_scale == "log" or (
        requested_top_scale == "auto" and bool(positive_values) and
        max(positive_values) / min(positive_values)
        > float(plot_config["top_log_dynamic_range_threshold"])
    )
    y_max = float(comparison["transfer_y_max"])
    if y_max <= max(central_values, default=0.0):
        raise ValueError(f"{identifier}.transfer_y_max does not contain every central value")
    frame_top = ROOT.TH1D(f"frame_{identifier}_transfer", "", len(edges) - 1, array("d", edges))
    if top_log:
        top.SetLogy(True)
        top_floor = max(min(positive_values) * float(plot_config["log_floor_fraction"]), 1.0e-12)
        frame_top.SetMinimum(top_floor)
    else:
        top_floor = 0.0
        frame_top.SetMinimum(0.0)
    frame_top.SetDirectory(0); frame_top.SetMaximum(y_max)
    frame_top.GetYaxis().SetTitle("Transfer factor")
    frame_top.GetYaxis().SetTitleSize(0.052); frame_top.GetYaxis().SetTitleOffset(1.05); frame_top.GetYaxis().SetLabelSize(0.043)
    frame_top.GetXaxis().SetLabelSize(0.0)
    frame_top.Draw("AXIS"); left_graph.Draw("P SAME"); right_graph.Draw("P SAME")
    zero_graphs = []
    if top_log:
        for key, name, color in (
            ("left_ratio", f"g_{identifier}_left_zero", ROOT.kBlue + 1),
            ("right_ratio", f"g_{identifier}_right_zero", ROOT.kRed + 1),
        ):
            zero_graph = zero_floor_graph(rows, key, top_floor * 1.15, name)
            if zero_graph is None:
                continue
            zero_graph.SetMarkerStyle(23); zero_graph.SetMarkerSize(1.0)
            zero_graph.SetMarkerColor(color); zero_graph.SetLineColor(color)
            zero_graph.Draw("P SAME")
            zero_graphs.append(zero_graph)

    bottom.cd()
    double_values = [float(row["double_ratio"]) for row in rows if row["double_ratio"] != ""]
    positive_double = [value for value in double_values if value > 0.0]
    lower_log = bool(positive_double) and (
        min(positive_double) < float(plot_config["lower_log_value_threshold"])
        or max(positive_double) / min(positive_double)
        > float(plot_config["lower_log_dynamic_range_threshold"])
    )
    frame_bottom = ROOT.TH1D(f"frame_{identifier}_double", "", len(edges) - 1, array("d", edges))
    if lower_log:
        bottom.SetLogy(True)
        lower_min = max(min(positive_double) * float(plot_config["log_floor_fraction"]), 1.0e-12)
        lower_max = max(2.0, max(positive_double) * 2.0)
        frame_bottom.SetMinimum(lower_min)
    else:
        lower_max = max(1.35, max(double_values, default=1.0) * 1.25)
        frame_bottom.SetMinimum(0.0)
    frame_bottom.SetDirectory(0); frame_bottom.SetMaximum(lower_max)
    frame_bottom.GetYaxis().SetTitle("Ratio of transfer factors")
    frame_bottom.GetXaxis().SetTitle("Sum of SciFi Density Weight (SNDSW)")
    frame_bottom.GetYaxis().SetTitleSize(0.085); frame_bottom.GetYaxis().SetTitleOffset(0.63)
    frame_bottom.GetYaxis().SetLabelSize(0.070); frame_bottom.GetYaxis().SetNdivisions(505)
    frame_bottom.GetXaxis().SetTitleSize(0.095); frame_bottom.GetXaxis().SetTitleOffset(1.05); frame_bottom.GetXaxis().SetLabelSize(0.075)
    frame_bottom.Draw("AXIS")
    unity = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
    unity.SetLineStyle(2); unity.SetLineWidth(2); unity.SetLineColor(ROOT.kGray + 2); unity.Draw("SAME")
    double_graph.Draw("P SAME")

    canvas.cd(); canvas.Modified(); canvas.Update()
    keepalive = (
        header, top, bottom, left_graph, right_graph, double_graph, *zero_graphs,
        frame_top, frame_bottom, legend, title, subtitle, unity,
    )
    root_file.cd(); left_graph.Write(); right_graph.Write(); double_graph.Write()
    for zero_graph in zero_graphs:
        zero_graph.Write()
    ROOT.TNamed(f"{identifier}_top_scale", "log" if top_log else "linear").Write()
    ROOT.TNamed(f"{identifier}_lower_scale", "log" if lower_log else "linear").Write()
    return keepalive


def create_signal_ratio_comparisons(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    region_names = list(config["source"]["roles"])
    loaded = {role: read_histogram(getattr(args, role), config["source"]["histogram_name"]) for role in region_names}
    histograms = {role: value[0] for role, value in loaded.items()}
    luminosities = {role: value[1] for role, value in loaded.items()}
    edges = validate_binning(histograms, config)

    all_rows = []
    rows_by_id = {}
    for comparison in config["comparisons"]:
        rows = calculate_comparison_rows(histograms, luminosities, comparison)
        rows_by_id[comparison["id"]] = rows
        all_rows.extend(rows)
    atomic_csv(all_rows, args.output_csv)

    output_pdf = Path(args.output_pdf)
    output_root = Path(args.output_root)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = output_pdf.with_name(f".{output_pdf.stem}.tmp.pdf")
    temporary_root = output_root.with_name(f".{output_root.stem}.tmp.root")
    root_file = ROOT.TFile(str(temporary_root), "RECREATE")
    ROOT.TNamed("analysis_scope", "explicitly unblinded distribution and six-comparison development diagnostic").Write()
    ROOT.TNamed("flow_policy", str(config["source"]["flow_policy"])).Write()
    comparisons = config["comparisons"]
    if len(comparisons) != 6:
        raise ValueError("Exactly six comparisons are required")
    canvases = []
    distribution_canvas = ROOT.TCanvas("c_region_distributions", "", 1100, 850)
    distribution_canvas._python_keepalive = draw_distribution_page(
        distribution_canvas, histograms, luminosities, edges, config["front_page"], root_file
    )
    root_file.cd(); distribution_canvas.Write()
    canvases.append(distribution_canvas)
    distribution_canvas.Print(f"{temporary_pdf}(")
    for page_index, comparison in enumerate(comparisons):
        canvas = ROOT.TCanvas(f"c_{comparison['id']}", "", 1100, 850)
        canvas._python_keepalive = draw_comparison(
            canvas, rows_by_id[comparison["id"]], edges, comparison, config["plot"], root_file
        )
        canvas.cd(); canvas.Modified(); canvas.Update()
        root_file.cd(); canvas.Write()
        canvases.append(canvas)
        suffix = ")" if page_index == len(comparisons) - 1 else ""
        canvas.Print(f"{temporary_pdf}{suffix}")
    root_file.Close()
    os.replace(temporary_pdf, output_pdf)
    os.replace(temporary_root, output_root)
    for output in (args.output_csv, args.output_root, args.output_pdf):
        print(f"Wrote {output}")
    for comparison in config["comparisons"]:
        valid = sum(row["double_status"] == "PASS" for row in rows_by_id[comparison["id"]])
        print(f"{comparison['id']} comparable bins: {valid}/{len(rows_by_id[comparison['id']])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for role in ("signal", "sb1", "sb2", "sb3", "sb4", "sb5", "sb6", "sb7"):
        parser.add_argument(f"--{role}", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    create_signal_ratio_comparisons(parser.parse_args())


if __name__ == "__main__":
    main()
