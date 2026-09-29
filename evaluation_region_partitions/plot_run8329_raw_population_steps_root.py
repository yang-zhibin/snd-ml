#!/usr/bin/env python3
"""Create the Run-8329 raw-population step plot with ROOT."""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

import ROOT


RAW_EVENTS = 298_285_666
PASSING_MUON_RECO_EVENTS = 82_145_120
EXPECTED_REAL_DQ_EVENTS = 11_115_982
LUMINOSITY_FB = 0.5323492746290897


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing all 299 v8 source manifest summaries.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def read_real_dq_total(manifest_dir: Path) -> int:
    paths = sorted(manifest_dir.glob("source*.json"))
    if len(paths) != 299:
        raise RuntimeError(
            f"Expected 299 manifest summaries in {manifest_dir}, found {len(paths)}"
        )

    sources: set[int] = set()
    total = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = int(payload["source_file_index"])
        if source in sources:
            raise RuntimeError(f"Duplicate source index {source}")
        sources.add(source)
        total += int(payload["counts"]["real_dq"])

    if sources != set(range(299)):
        raise RuntimeError("Manifest summaries do not cover source indices 000--298")
    if total != EXPECTED_REAL_DQ_EVENTS:
        raise RuntimeError(
            f"Unexpected RealDataDQ total: {total:,}; expected {EXPECTED_REAL_DQ_EVENTS:,}"
        )
    return total


def draw_plot(real_dq: int) -> tuple[ROOT.TCanvas, ROOT.TH1D, ROOT.TGraph]:
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetTextFont(42)
    ROOT.gStyle.SetEndErrorSize(0)

    canvas = ROOT.TCanvas("run8329_raw_population_canvas", "", 1250, 780)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.035)
    canvas.SetTopMargin(0.105)
    canvas.SetBottomMargin(0.18)
    canvas.SetLogy(True)
    canvas.SetGridy(True)

    histogram = ROOT.TH1D(
        "run8329_raw_population",
        "Populations derived from the raw run;;Events",
        3,
        0.0,
        3.0,
    )
    values = [RAW_EVENTS, PASSING_MUON_RECO_EVENTS, real_dq]
    labels = ["Complete raw run", "PassingMuonReco", "RealDataDQ used by v8"]
    for index, (label, value) in enumerate(zip(labels, values), start=1):
        histogram.SetBinContent(index, value)
        histogram.GetXaxis().SetBinLabel(index, label)

    histogram.SetMinimum(5.0e6)
    histogram.SetMaximum(5.8e8)
    histogram.SetLineColor(ROOT.kAzure + 2)
    histogram.SetLineWidth(4)
    histogram.SetFillColorAlpha(ROOT.kAzure - 9, 0.48)
    histogram.GetXaxis().SetLabelSize(0.043)
    histogram.GetXaxis().SetTitleSize(0.047)
    histogram.GetXaxis().SetTitleOffset(1.55)
    histogram.GetYaxis().SetLabelSize(0.041)
    histogram.GetYaxis().SetTitleSize(0.047)
    histogram.GetYaxis().SetTitleOffset(1.15)
    histogram.GetYaxis().SetMoreLogLabels(True)
    histogram.GetYaxis().SetNoExponent(False)
    histogram.SetTitleFont(42)
    histogram.Draw("HIST")

    graph = ROOT.TGraph(3, array("d", [0.5, 1.5, 2.5]), array("d", values))
    graph.SetName("run8329_raw_population_markers")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.45)
    graph.SetMarkerColor(ROOT.kAzure + 2)
    graph.Draw("P SAME")

    text = ROOT.TLatex()
    text.SetTextFont(42)
    text.SetTextAlign(22)
    text.SetTextSize(0.037)
    for x, value in zip([0.5, 1.5, 2.5], values):
        fraction = 100.0 * value / RAW_EVENTS
        text.DrawLatex(x, value * 1.34, f"#bf{{{value:,}}}")
        text.DrawLatex(x, value * 1.15, f"{fraction:.3f}% of raw")

    note = ROOT.TLatex()
    note.SetNDC(True)
    note.SetTextFont(42)
    note.SetTextColor(ROOT.kGray + 2)
    note.SetTextSize(0.031)
    note.DrawLatex(
        0.125,
        0.075,
        "PassingMuonReco and RealDataDQ are separate branches derived from the same raw run.",
    )
    note.SetTextAlign(22)
    note.DrawLatex(
        0.54,
        0.032,
        f"Run 8329  |  299 source files  |  {LUMINOSITY_FB:.6f} fb^{{-1}}",
    )

    canvas.Modified()
    canvas.Update()
    return canvas, histogram, graph


def main() -> None:
    args = parse_args()
    real_dq = read_real_dq_total(args.manifest_dir)
    canvas, histogram, graph = draw_plot(real_dq)

    for path in (args.output_root, args.output_pdf, args.output_png):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_file = ROOT.TFile.Open(str(args.output_root), "RECREATE")
    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Cannot create {args.output_root}")
    histogram.Write()
    graph.Write()
    canvas.Write()
    output_file.Close()

    canvas.SaveAs(str(args.output_pdf))
    canvas.SaveAs(str(args.output_png))

    print(
        json.dumps(
            {
                "raw": RAW_EVENTS,
                "passing_muon_reco": PASSING_MUON_RECO_EVENTS,
                "real_dq": real_dq,
                "output_root": str(args.output_root),
                "output_pdf": str(args.output_pdf),
                "output_png": str(args.output_png),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
