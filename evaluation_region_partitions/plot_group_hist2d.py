#!/usr/bin/env python3
"""Plot four-class 2D densities and profile overlays from grouped histograms."""

from __future__ import annotations

import argparse
import csv
import math
import json
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

COLORS = {
    "CC_nue": ROOT.kBlue + 1,
    "CC_numu": ROOT.kRed + 1,
    "NC_nue": ROOT.kGreen + 2,
    "NC_numu": ROOT.kMagenta + 1,
}


def parse_hist(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--hist must use CLASS=PATH")
    return tuple(value.split("=", 1))


def plot(args: argparse.Namespace) -> None:
    pair_cfg = getattr(args, 'pair_config', None)
    if pair_cfg is None:
        with open(args.hist_config, "r", encoding="utf-8") as handle:
            pair_cfg = yaml.safe_load(handle)["features2d"][args.pair]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opened = []
    items = []
    rows = []
    mode = getattr(args, "normalization", "shape")
    if mode not in ("shape", "yield"):
        raise ValueError("Unknown 2D normalization")
    scales = getattr(args, "scales", {})
    styles = getattr(args, "styles", {})
    labels = getattr(args, "labels", {})
    for particle_group, path in args.hist:
        root_file = ROOT.TFile.Open(path)
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not open {path}")
        opened.append(root_file)
        hist = root_file.Get("hist2d")
        profile = root_file.Get("profile_x")
        if not hist or not profile:
            raise KeyError(f"Missing hist2d/profile_x in {path}")
        hist = hist.Clone(f"hist2d_{particle_group}")
        profile = profile.Clone(f"profile_{particle_group}")
        hist.SetDirectory(0)
        profile.SetDirectory(0)
        integral = float(hist.Integral())
        scale = (1.0 / integral if integral > 0 else 0.0) if mode == "shape" else float(scales[particle_group])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Invalid audited 2D scale")
        hist.Scale(scale)
        items.append((particle_group, hist, profile))
        rows.append(
            {
                "region": args.region,
                "pair": args.pair,
                "particle_group": particle_group,
                "entries": int(hist.GetEntries()),
                "normalization": mode, "raw_integral": integral,
                "scale_factor": scale, "scaled_integral": float(hist.Integral()),
                "exposure": json.dumps(getattr(args, "exposures", {}).get(particle_group, {}), sort_keys=True),
                "mean_x": hist.GetMean(1),
                "mean_y": hist.GetMean(2),
                "correlation": hist.GetCorrelationFactor(),
            }
        )

    canvas = ROOT.TCanvas("density", "density", 1400, 1000)
    if not items:
        raise ValueError("No groups supplied for 2D plot")
    canvas.Divide(2, max(2, math.ceil(len(items) / 2)))
    for index, (particle_group, hist, _) in enumerate(items, start=1):
        canvas.cd(index)
        ROOT.gPad.SetRightMargin(0.15)
        ROOT.gPad.SetLogz(bool(pair_cfg.get("logz", True)) and hist.Integral() > 0)
        ROOT.gPad.SetLogx(bool(pair_cfg.get("logx", False)))
        ROOT.gPad.SetLogy(bool(pair_cfg.get("logy", False)))
        hist.SetTitle(f"{labels.get(particle_group, particle_group)}: {args.region}")
        hist.GetXaxis().SetTitle(str(pair_cfg["x_title"]))
        hist.GetYaxis().SetTitle(str(pair_cfg["y_title"]))
        hist.GetZaxis().SetTitle("Fraction of events" if mode == "shape" else "Events per bin")
        hist.Draw("COLZ")
        if hist.Integral() == 0:
            annotation = ROOT.TLatex()
            annotation.DrawLatexNDC(.2, .5, "No in-range events")
    canvas.Print(f"{output}(")

    profile_canvas = ROOT.TCanvas("profiles", "profiles", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.68, 0.89, 0.89)
    maximum = max((profile.GetMaximum() for _, _, profile in items), default=1.0)
    for index, (particle_group, _, profile) in enumerate(items):
        from plot_comparison import color_from_config
        style = styles.get(particle_group, {})
        color = color_from_config(style['color'], index) if 'color' in style else COLORS.get(particle_group, ROOT.kBlack)
        profile.SetLineColor(color)
        profile.SetMarkerColor(color)
        profile.SetLineWidth(2)
        profile.SetMarkerStyle(int(style.get("marker", 20 + index)))
        profile.SetTitle(f"{args.pair}: {args.region}")
        profile.GetXaxis().SetTitle(str(pair_cfg["x_title"]))
        profile.GetYaxis().SetTitle(f"Mean {pair_cfg['y_title']}")
        profile.SetMaximum(maximum * 1.2 if maximum > 0 else 1.0)
        profile.Draw("E1" if index == 0 else "E1 SAME")
        legend.AddEntry(profile, labels.get(particle_group, particle_group), "lp")
    legend.Draw()
    profile_canvas.Print(f"{output})")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for root_file in opened:
        root_file.Close()
    print(f"Wrote {output} and {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--hist", action="append", type=parse_hist, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
