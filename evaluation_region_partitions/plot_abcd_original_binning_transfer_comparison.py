#!/usr/bin/env python3
"""Plot Signal/SB2 and SB4/SB6 transfer factors with PyROOT."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from array import array
from pathlib import Path

import ROOT
import yaml
from scipy.optimize import brentq


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

ROLE_ORDER = ("signal", "sb2", "sb4", "sb6")
ONE_SIGMA_CONFIDENCE_LEVEL = 0.682689492137


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_histogram(path: str | Path, histogram_name: str) -> tuple[ROOT.TH1, float]:
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise RuntimeError(f"Could not open ROOT input: {path}")
    try:
        obj = source.Get(histogram_name)
        if obj is None or not obj.InheritsFrom("TH1"):
            raise KeyError(f"Missing TH1 {histogram_name!r} in {path}")
        lumi_obj = source.Get("lumi")
        if lumi_obj is None or not hasattr(lumi_obj, "GetVal"):
            raise KeyError(f"Missing luminosity metadata in {path}")
        hist = obj.Clone(f"source_{Path(path).parent.name}")
        hist.SetDirectory(0)
        lumi = float(lumi_obj.GetVal())
    finally:
        source.Close()
    if not math.isfinite(lumi) or lumi <= 0.0:
        raise ValueError(f"Non-positive luminosity in {path}: {lumi}")
    return hist, lumi


def validate_binning(histograms: dict[str, ROOT.TH1], config: dict) -> list[float]:
    source_cfg = config["source"]
    expected_n = int(source_cfg["expected_n_bins"])
    expected_min = float(source_cfg["expected_x_min"])
    expected_max = float(source_cfg["expected_x_max"])
    reference = histograms[ROLE_ORDER[0]]
    if reference.GetNbinsX() != expected_n:
        raise ValueError(f"Expected {expected_n} bins, found {reference.GetNbinsX()}")
    axis = reference.GetXaxis()
    if not math.isclose(axis.GetXmin(), expected_min, abs_tol=1e-12):
        raise ValueError(f"Unexpected x minimum: {axis.GetXmin()}")
    if not math.isclose(axis.GetXmax(), expected_max, abs_tol=1e-12):
        raise ValueError(f"Unexpected x maximum: {axis.GetXmax()}")
    edges = [float(axis.GetBinLowEdge(1))]
    edges.extend(float(axis.GetBinUpEdge(i)) for i in range(1, expected_n + 1))
    for role, hist in histograms.items():
        if hist.GetNbinsX() != expected_n:
            raise ValueError(f"Binning mismatch for {role}")
        other = hist.GetXaxis()
        for index, edge in enumerate(edges):
            candidate = float(other.GetBinLowEdge(1)) if index == 0 else float(other.GetBinUpEdge(index))
            if not math.isclose(candidate, edge, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Bin-edge mismatch for {role} at edge {index}")
    return edges


def integer_count(value: float, label: str) -> int:
    rounded = int(round(value))
    if value < 0.0 or not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} is not a non-negative integer count: {value}")
    return rounded


def transfer(
    numerator: float,
    numerator_lumi: float,
    denominator: float,
    denominator_lumi: float,
) -> tuple[float, float, float] | None:
    numerator_count = integer_count(numerator, "transfer numerator")
    denominator_count = integer_count(denominator, "transfer denominator")
    if denominator_count == 0:
        return None
    total = numerator_count + denominator_count
    probability_low = ROOT.TEfficiency.ClopperPearson(
        total, numerator_count, ONE_SIGMA_CONFIDENCE_LEVEL, False
    )
    probability_high = ROOT.TEfficiency.ClopperPearson(
        total, numerator_count, ONE_SIGMA_CONFIDENCE_LEVEL, True
    )
    exposure_scale = denominator_lumi / numerator_lumi
    value = numerator_count * exposure_scale / denominator_count
    interval_low = exposure_scale * probability_low / (1.0 - probability_low)
    interval_high = exposure_scale * probability_high / (1.0 - probability_high)
    return value, interval_low, interval_high


def double_ratio_interval(
    counts: dict[str, float], luminosities: dict[str, float]
) -> tuple[float, float, float] | None:
    integer_counts = {
        role: integer_count(counts[role], f"double-ratio {role}")
        for role in ROLE_ORDER
    }
    if any(value == 0 for value in integer_counts.values()):
        return None
    exposure_scale = (
        luminosities["sb2"] * luminosities["sb4"]
        / (luminosities["signal"] * luminosities["sb6"])
    )
    value = (
        integer_counts["signal"] * integer_counts["sb6"]
        / (integer_counts["sb2"] * integer_counts["sb4"])
        * exposure_scale
    )
    observed = [
        float(integer_counts["signal"]),
        float(integer_counts["sb2"]),
        float(integer_counts["sb4"]),
        float(integer_counts["sb6"]),
    ]
    row_one = observed[0] + observed[1]
    column_one = observed[0] + observed[2]
    total = sum(observed)
    x_min = max(0.0, row_one + column_one - total)
    x_max = min(row_one, column_one)

    def profile_deviance(log_odds_ratio: float) -> float:
        span = x_max - x_min
        epsilon = max(1.0e-10, span * 1.0e-12)

        def constraint(cell_a: float) -> float:
            fitted = [
                cell_a,
                row_one - cell_a,
                column_one - cell_a,
                total - row_one - column_one + cell_a,
            ]
            return (
                math.log(fitted[0]) + math.log(fitted[3])
                - math.log(fitted[1]) - math.log(fitted[2])
                - log_odds_ratio
            )

        cell_a = brentq(
            constraint,
            x_min + epsilon,
            x_max - epsilon,
            xtol=1.0e-10,
            rtol=1.0e-12,
        )
        fitted = [
            cell_a,
            row_one - cell_a,
            column_one - cell_a,
            total - row_one - column_one + cell_a,
        ]
        return 2.0 * sum(
            expected - count + count * math.log(count / expected)
            for count, expected in zip(observed, fitted)
        )

    odds_ratio_hat = value / exposure_scale
    log_hat = math.log(odds_ratio_hat)

    def endpoint(direction: float) -> float:
        inside = log_hat
        outside = log_hat + direction
        while profile_deviance(outside) < 1.0:
            outside += direction
        lower, upper = sorted((inside, outside))
        return brentq(
            lambda candidate: profile_deviance(candidate) - 1.0,
            lower,
            upper,
            xtol=1.0e-10,
            rtol=1.0e-10,
        )

    interval_low = math.exp(endpoint(-1.0)) * exposure_scale
    interval_high = math.exp(endpoint(1.0)) * exposure_scale
    return value, interval_low, interval_high


def calculate_rows(histograms: dict[str, ROOT.TH1], luminosities: dict[str, float]) -> list[dict]:
    rows = []
    n_bins = histograms["signal"].GetNbinsX()
    for index in range(1, n_bins + 1):
        counts = {role: float(histograms[role].GetBinContent(index)) for role in ROLE_ORDER}
        no_veto = transfer(counts["signal"], luminosities["signal"], counts["sb2"], luminosities["sb2"])
        veto = transfer(counts["sb4"], luminosities["sb4"], counts["sb6"], luminosities["sb6"])
        comparison = double_ratio_interval(counts, luminosities)
        axis = histograms["signal"].GetXaxis()
        rows.append({
            "bin_index": index,
            "bin_low": float(axis.GetBinLowEdge(index)),
            "bin_high": float(axis.GetBinUpEdge(index)),
            **{f"count_{role}": counts[role] for role in ROLE_ORDER},
            **{f"exposure_{role}": luminosities[role] for role in ROLE_ORDER},
            "transfer_signal_over_sb2": "" if no_veto is None else no_veto[0],
            "transfer_signal_over_sb2_interval_low": "" if no_veto is None else no_veto[1],
            "transfer_signal_over_sb2_interval_high": "" if no_veto is None else no_veto[2],
            "transfer_sb4_over_sb6": "" if veto is None else veto[0],
            "transfer_sb4_over_sb6_interval_low": "" if veto is None else veto[1],
            "transfer_sb4_over_sb6_interval_high": "" if veto is None else veto[2],
            "double_ratio": "" if comparison is None else comparison[0],
            "double_ratio_interval_low": "" if comparison is None else comparison[1],
            "double_ratio_interval_high": "" if comparison is None else comparison[2],
            "status_signal_over_sb2": "PASS" if no_veto is not None else "UNDEFINED_ZERO_DENOMINATOR",
            "status_sb4_over_sb6": "PASS" if veto is not None else "UNDEFINED_ZERO_DENOMINATOR",
            "status_double_ratio": "PASS" if comparison is not None else "UNDEFINED_NONPOSITIVE_COUNT",
        })
    return rows


def atomic_csv(rows: list[dict], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def graph_from_rows(
    rows: list[dict], value_key: str, low_key: str, high_key: str, name: str
) -> ROOT.TGraphAsymmErrors:
    selected = [row for row in rows if row[value_key] != ""]
    x = array("d", [(row["bin_low"] + row["bin_high"]) / 2.0 for row in selected])
    ex = array("d", [(row["bin_high"] - row["bin_low"]) / 2.0 for row in selected])
    y = array("d", [float(row[value_key]) for row in selected])
    ey_low = array("d", [float(row[value_key]) - float(row[low_key]) for row in selected])
    ey_high = array("d", [float(row[high_key]) - float(row[value_key]) for row in selected])
    graph = ROOT.TGraphAsymmErrors(len(selected), x, y, ex, ex, ey_low, ey_high)
    graph.SetName(name)
    return graph


def draw_plot(rows: list[dict], edges: list[float], config: dict, pdf: str | Path, png: str | Path, root_path: str | Path) -> None:
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetEndErrorSize(4)
    canvas = ROOT.TCanvas("c_transfer_factor_comparison", "", 1100, 850)
    top = ROOT.TPad("top", "", 0.0, 0.34, 1.0, 1.0)
    bottom = ROOT.TPad("bottom", "", 0.0, 0.0, 1.0, 0.34)
    top.SetLeftMargin(0.12); top.SetRightMargin(0.04); top.SetTopMargin(0.10); top.SetBottomMargin(0.025)
    bottom.SetLeftMargin(0.12); bottom.SetRightMargin(0.04); bottom.SetTopMargin(0.025); bottom.SetBottomMargin(0.28)
    top.Draw(); bottom.Draw()

    graph_no_veto = graph_from_rows(rows, "transfer_signal_over_sb2", "transfer_signal_over_sb2_interval_low", "transfer_signal_over_sb2_interval_high", "g_signal_over_sb2")
    graph_veto = graph_from_rows(rows, "transfer_sb4_over_sb6", "transfer_sb4_over_sb6_interval_low", "transfer_sb4_over_sb6_interval_high", "g_sb4_over_sb6")
    graph_double = graph_from_rows(rows, "double_ratio", "double_ratio_interval_low", "double_ratio_interval_high", "g_double_ratio")
    for graph, color, marker in [(graph_no_veto, ROOT.kBlue + 1, 20), (graph_veto, ROOT.kRed + 1, 24)]:
        graph.SetLineColor(color); graph.SetMarkerColor(color); graph.SetMarkerStyle(marker); graph.SetMarkerSize(0.9); graph.SetLineWidth(2)
    graph_double.SetLineColor(ROOT.kBlack); graph_double.SetMarkerColor(ROOT.kBlack); graph_double.SetMarkerStyle(20); graph_double.SetMarkerSize(0.85); graph_double.SetLineWidth(2)

    top.cd()
    top_central_values = [float(row[key]) for row in rows for key in [
        "transfer_signal_over_sb2", "transfer_sb4_over_sb6"
    ] if row[key] != ""]
    transfer_y_max = float(config["plot"]["transfer_y_max"])
    if transfer_y_max <= max(top_central_values, default=0.0):
        raise ValueError(
            "plot.transfer_y_max must exceed every transfer-factor central value"
        )
    frame_top = ROOT.TH1D("frame_transfer", "", len(edges) - 1, array("d", edges))
    frame_top.SetDirectory(0); frame_top.SetMinimum(0.0); frame_top.SetMaximum(transfer_y_max)
    frame_top.GetYaxis().SetTitle("Inside / outside transfer factor")
    frame_top.GetYaxis().SetTitleSize(0.052); frame_top.GetYaxis().SetTitleOffset(1.05); frame_top.GetYaxis().SetLabelSize(0.043)
    frame_top.GetXaxis().SetLabelSize(0.0)
    frame_top.Draw("AXIS")
    graph_no_veto.Draw("P SAME"); graph_veto.Draw("P SAME")
    legend = ROOT.TLegend(0.57, 0.70, 0.94, 0.87)
    legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(0.038)
    legend.AddEntry(graph_no_veto, "Signal / SB2 (no veto, US)", "lep")
    legend.AddEntry(graph_veto, "SB4 / SB6 (veto, US)", "lep")
    legend.Draw()
    title = ROOT.TLatex(); title.SetNDC(True); title.SetTextFont(42); title.SetTextSize(0.048)
    title.DrawLatex(0.12, 0.93, "Fiducial transfer-factor comparison")
    warning = ROOT.TLatex(); warning.SetNDC(True); warning.SetTextFont(42); warning.SetTextColor(ROOT.kRed + 1); warning.SetTextSize(0.033)
    warning.DrawLatex(0.13, 0.84, "Explicitly unblinded development diagnostic")
    support = ROOT.TLatex(); support.SetNDC(True); support.SetTextFont(42); support.SetTextSize(0.028)
    support.DrawLatex(
        0.13,
        0.79,
        f"68.27% asymmetric intervals; upper errors clipped at y = {transfer_y_max:g} where needed",
    )

    bottom.cd()
    frame_bottom = ROOT.TH1D("frame_double_ratio", "", len(edges) - 1, array("d", edges))
    frame_bottom.SetDirectory(0); frame_bottom.SetMinimum(0.0)
    bottom_values = [float(row["double_ratio_interval_high"]) for row in rows if row["double_ratio"] != ""]
    frame_bottom.SetMaximum(max(1.35, max(bottom_values, default=1.0) * 1.20))
    frame_bottom.GetYaxis().SetTitle("(Signal/SB2) / (SB4/SB6)")
    frame_bottom.GetXaxis().SetTitle("Sum of SciFi Density Weight (SNDSW)")
    frame_bottom.GetYaxis().SetTitleSize(0.085); frame_bottom.GetYaxis().SetTitleOffset(0.63); frame_bottom.GetYaxis().SetLabelSize(0.070); frame_bottom.GetYaxis().SetNdivisions(505)
    frame_bottom.GetXaxis().SetTitleSize(0.095); frame_bottom.GetXaxis().SetTitleOffset(1.05); frame_bottom.GetXaxis().SetLabelSize(0.075)
    frame_bottom.Draw("AXIS")
    unity = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0); unity.SetLineStyle(2); unity.SetLineWidth(2); unity.SetLineColor(ROOT.kGray + 2); unity.Draw("SAME")
    graph_double.Draw("P SAME")

    canvas.cd(); canvas.Modified(); canvas.Update()
    outputs = [Path(pdf), Path(png), Path(root_path)]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = Path(pdf).with_name(f".{Path(pdf).stem}.tmp.pdf")
    temporary_png = Path(png).with_name(f".{Path(png).stem}.tmp.png")
    temporary_root = Path(root_path).with_name(f".{Path(root_path).stem}.tmp.root")
    canvas.SaveAs(str(temporary_pdf)); canvas.SaveAs(str(temporary_png))
    root_file = ROOT.TFile(str(temporary_root), "RECREATE")
    canvas.Write(); graph_no_veto.Write(); graph_veto.Write(); graph_double.Write(); frame_top.Write(); frame_bottom.Write()
    ROOT.TNamed("analysis_scope", "explicitly unblinded development diagnostic").Write()
    ROOT.TNamed("flow_policy", str(config["source"]["flow_policy"])).Write()
    root_file.Close()
    os.replace(temporary_pdf, pdf); os.replace(temporary_png, png); os.replace(temporary_root, root_path)


def create_transfer_factor_comparison(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    paths = {role: getattr(args, role) for role in ROLE_ORDER}
    loaded = {role: read_histogram(path, str(config["source"]["histogram_name"])) for role, path in paths.items()}
    histograms = {role: item[0] for role, item in loaded.items()}
    luminosities = {role: item[1] for role, item in loaded.items()}
    edges = validate_binning(histograms, config)
    rows = calculate_rows(histograms, luminosities)
    atomic_csv(rows, args.output_csv)
    draw_plot(rows, edges, config, args.output_pdf, args.output_png, args.output_root)
    valid = sum(row["status_double_ratio"] == "PASS" for row in rows)
    for output in (args.output_csv, args.output_pdf, args.output_png, args.output_root):
        print(f"Wrote {output}")
    print(f"Comparable bins: {valid}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for role in ROLE_ORDER:
        parser.add_argument(f"--{role}", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-png", required=True)
    create_transfer_factor_comparison(parser.parse_args())


if __name__ == "__main__":
    main()
