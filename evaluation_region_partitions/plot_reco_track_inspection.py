#!/usr/bin/env python3
"""Render reconstructed-track inspection histograms as one-page PDF files."""

import argparse
from pathlib import Path

import ROOT

from reco_track_features import TRACK_TYPE_PREFIXES


def draw_label(text):
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.035)
    label.DrawLatex(0.14, 0.94, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--done", required=True)
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open {args.input}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix in TRACK_TYPE_PREFIXES.values():
        for suffix in ("n_all", "best_chi2ndf"):
            name = f"{prefix}_{suffix}"
            histogram = source.Get(name)
            if not histogram:
                raise RuntimeError(f"Missing histogram {name}")
            canvas = ROOT.TCanvas(f"c_{name}", name, 900, 700)
            canvas.SetLeftMargin(0.13)
            canvas.SetBottomMargin(0.12)
            canvas.SetLogy()
            histogram.SetLineWidth(2)
            histogram.SetMarkerStyle(20)
            histogram.Draw("E1")
            draw_label("SND@LHC work in progress")
            canvas.SaveAs(str(output_dir / f"{name}.pdf"))

        for suffix in (
            "veto_count_vs_n_all",
            "veto_time_vs_n_all",
            "veto_count_vs_chi2ndf",
            "veto_time_vs_chi2ndf",
            "start_xy_2d",
        ):
            name = f"{prefix}_{suffix}"
            histogram = source.Get(name)
            if not histogram:
                raise RuntimeError(f"Missing histogram {name}")
            canvas = ROOT.TCanvas(f"c_{name}", name, 1000, 750)
            canvas.SetLeftMargin(0.14)
            canvas.SetRightMargin(0.16)
            canvas.SetBottomMargin(0.12)
            canvas.SetLogz()
            histogram.GetYaxis().SetTitleOffset(1.45)
            histogram.Draw("COLZ")
            draw_label("SND@LHC work in progress")
            canvas.SaveAs(str(output_dir / f"{name}.pdf"))

        name = f"{prefix}_start_xy_chi2ndf_3d"
        histogram = source.Get(name)
        if not histogram:
            raise RuntimeError(f"Missing histogram {name}")
        canvas = ROOT.TCanvas(f"c_{name}", name, 1100, 850)
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.16)
        canvas.SetBottomMargin(0.12)
        canvas.SetTheta(25)
        canvas.SetPhi(35)
        histogram.GetXaxis().SetTitleOffset(1.7)
        histogram.GetYaxis().SetTitleOffset(1.8)
        histogram.GetZaxis().SetTitleOffset(1.35)
        histogram.Draw("BOX2Z")
        draw_label("SND@LHC work in progress")
        canvas.SaveAs(str(output_dir / f"{name}.pdf"))

    for method in ("simple", "hough"):
        for suffix in (
            "ds_n_all_vs_scifi_n_all",
            "ds_best_chi2ndf_vs_scifi_best_chi2ndf",
        ):
            name = f"reco_{method}_{suffix}"
            histogram = source.Get(name)
            if not histogram:
                raise RuntimeError(f"Missing histogram {name}")
            canvas = ROOT.TCanvas(f"c_{name}", name, 1000, 750)
            canvas.SetLeftMargin(0.14)
            canvas.SetRightMargin(0.16)
            canvas.SetBottomMargin(0.12)
            canvas.SetLogz()
            histogram.GetYaxis().SetTitleOffset(1.45)
            histogram.Draw("COLZ")
            draw_label("SND@LHC work in progress")
            canvas.SaveAs(str(output_dir / f"{name}.pdf"))

    Path(args.done).write_text(
        "passing_muon_tracks_v1 plotting complete\n", encoding="utf-8"
    )
    source.Close()


if __name__ == "__main__":
    main()
