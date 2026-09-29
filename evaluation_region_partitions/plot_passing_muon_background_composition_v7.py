#!/usr/bin/env python3
"""Plot raw-count PassingMuonReco-only observable composition for V7."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from make_passing_muon_binary_region_hist_v5 import configured_selections
from passing_muon_background_composition_v7 import (
    configured_background_composition,
)


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

STYLES = {
    "no_valid_track": (ROOT.kGray + 1, 24),
    "track_like_low_activity": (ROOT.kBlue + 1, 20),
    "track_plus_activity": (ROOT.kRed + 1, 21),
}


def metadata(source, name):
    return common.object_value(source, name)


def histogram(source, selection, feature, class_name):
    return common.clone_histogram(
        source,
        f"selections/{selection}/features/{feature}/{class_name}",
        f"composition_{selection}_{feature}_{class_name}",
    )


def class_histograms(source, selection, feature, composition):
    result = []
    for class_name in composition["class_order"]:
        item = histogram(source, selection, feature, class_name)
        color, marker = STYLES[class_name]
        item.SetLineColor(color)
        item.SetFillColorAlpha(color, 0.48)
        item.SetMarkerColor(color)
        item.SetMarkerStyle(marker)
        item.SetLineWidth(2)
        result.append((class_name, item))
    inclusive = histogram(source, selection, feature, "inclusive")
    combined = result[0][1].Clone(f"combined_{selection}_{feature}")
    combined.SetDirectory(0)
    for _, item in result[1:]:
        combined.Add(item)
    if inclusive.GetNcells() != combined.GetNcells():
        raise RuntimeError(f"Histogram-cell mismatch for {selection}/{feature}")
    for cell in range(inclusive.GetNcells()):
        if abs(inclusive.GetBinContent(cell) - combined.GetBinContent(cell)) > 1e-9:
            raise RuntimeError(
                f"Histogram closure failed for {selection}/{feature}, cell {cell}"
            )
    inclusive.SetLineColor(ROOT.kBlack)
    inclusive.SetMarkerColor(ROOT.kBlack)
    inclusive.SetMarkerStyle(20)
    inclusive.SetMarkerSize(0.50)
    inclusive.SetLineWidth(2)
    return result, inclusive


def header(selection, title):
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.029)
    text.DrawLatex(0.08, 0.972, "SND@LHC work in progress")
    text.SetTextAlign(31)
    text.DrawLatex(0.95, 0.972, "Run 8329 PassingMuonReco, flat_v2")
    text.SetTextAlign(13)
    text.SetTextSize(0.024)
    text.DrawLatex(0.08, 0.935, f"{selection}: {title}")
    return text


def draw_summary(canvas, source, selection, composition):
    canvas.Clear()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.036)
    text.DrawLatex(0.08, 0.87, "Observable background-composition summary")
    total = int(metadata(source, f"selection__{selection}__total_count"))
    lines = [
        f"Scope: {metadata(source, 'analysis_scope')}",
        f"Input events: {total:,}",
        f"Configured track: {metadata(source, 'track_definition_label')}",
        "Classes are exhaustive observable categories, not particle-truth labels.",
        "Low activity means the named detector coverage selection passes.",
        "No-valid-track is restricted to PassingMuonReco and is not an inclusive-data rate.",
        "All plotted bins contain raw event counts; no unit-area normalization is used.",
        "",
        "Class closure:",
    ]
    class_sum = 0
    for class_name in composition["class_order"]:
        count = int(metadata(
            source, f"selection__{selection}__class__{class_name}__count"
        ))
        class_sum += count
        fraction = 100.0 * count / total if total else 0.0
        lines.append(
            f"{composition['class_labels'][class_name]}: {count:,} ({fraction:.4f}%)"
        )
    lines.append(f"Sum: {class_sum:,} = {total:,}")
    text.SetTextSize(0.024)
    y = 0.81
    for line in lines:
        text.DrawLatex(0.09, y, str(line))
        y -= 0.043 if line else 0.022
    top = header(selection, "PassingMuonReco-only composition")
    canvas._keepalive = [text, top]
    canvas.Modified()
    canvas.Update()


def draw_section(canvas, selection, section, composition):
    canvas.Clear()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.045)
    text.DrawLatex(0.10, 0.78, section["label"])
    text.SetTextSize(0.027)
    y = 0.69
    for feature_name in section["features"]:
        feature = composition["features"][feature_name]
        title = feature.get("axis_title") or (
            f"{feature['x_title']} versus {feature['y_title']}"
        )
        text.DrawLatex(0.13, y, f"#bullet  {title}")
        y -= 0.050
    top = header(selection, "section overview")
    canvas._keepalive = [text, top]
    canvas.Modified()
    canvas.Update()


def draw_1d(canvas, source, selection, feature_name, feature, composition):
    items, inclusive = class_histograms(
        source, selection, feature_name, composition
    )
    canvas.Clear()
    upper = ROOT.TPad(f"upper_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetLeftMargin(0.13)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.13)
    upper.SetBottomMargin(0.025)
    # Composition plots intentionally use a linear event-count scale. Keep
    # this local to the composition report so detector, count, and reco plots
    # retain their independently configured axis scales.
    upper.SetLogy(False)
    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.36)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()

    upper.cd()
    stack = ROOT.THStack(f"stack_{feature_name}", "")
    for _, item in items:
        stack.Add(item)
    stack.Draw("HIST")
    stack.SetTitle(f";{feature['axis_title']};Events")
    maximum = max(float(inclusive.GetMaximum()), 1.0)
    if upper.GetLogy():
        stack.SetMinimum(0.5)
        stack.SetMaximum(maximum * 20.0)
    else:
        stack.SetMinimum(0.0)
        stack.SetMaximum(maximum * 1.35)
    inclusive.Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.47, 0.64, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.024)
    legend.AddEntry(inclusive, "Inclusive PassingMuonReco", "lep")
    for class_name, item in items:
        valid = int(metadata(
            source,
            f"selection__{selection}__class__{class_name}__feature__{feature_name}__valid_count",
        ))
        legend.AddEntry(
            item, f"{composition['class_labels'][class_name]} (valid N={valid:,})", "f"
        )
    legend.Draw()
    top = header(selection, feature["axis_title"])

    lower.cd()
    fractions = []
    first = True
    for class_name, item in items:
        fraction = item.Clone(f"fraction_{feature_name}_{class_name}")
        fraction.SetDirectory(0)
        fraction.Divide(item, inclusive, 1.0, 1.0, "B")
        color, marker = STYLES[class_name]
        fraction.SetLineColor(color)
        fraction.SetMarkerColor(color)
        fraction.SetMarkerStyle(marker)
        fraction.SetTitle(f";{feature['axis_title']};Class fraction")
        fraction.GetXaxis().SetTitleSize(0.13)
        fraction.GetXaxis().SetLabelSize(0.10)
        fraction.GetXaxis().SetTitleOffset(1.15)
        fraction.GetYaxis().SetTitleSize(0.095)
        fraction.GetYaxis().SetLabelSize(0.085)
        fraction.GetYaxis().SetTitleOffset(0.65)
        fraction.GetYaxis().SetNdivisions(505)
        fraction.SetMinimum(0.0)
        fraction.SetMaximum(1.05)
        fraction.Draw("E1 X0" if first else "E1 X0 SAME")
        first = False
        fractions.append(fraction)
    canvas._keepalive = [
        upper, lower, stack, inclusive, legend, top,
        *(item for _, item in items), *fractions,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(canvas, source, selection, feature_name, feature, composition):
    items, inclusive = class_histograms(
        source, selection, feature_name, composition
    )
    panels = [*items, ("inclusive", inclusive)]
    maximum = max(float(item.GetMaximum()) for _, item in panels)
    logz = False
    canvas.Clear()
    canvas.Divide(2, 2, 0.002, 0.002)
    retained = [inclusive, *(item for _, item in items)]
    for index, (name, item) in enumerate(panels, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.16)
        pad.SetLogz(logz)
        item.SetMinimum(0.0)
        item.SetMaximum(max(maximum * 1.05, 1.0))
        item.SetTitle(f";{feature['x_title']};{feature['y_title']};Events")
        item.Draw("COLZ")
        title = ROOT.TLatex()
        title.SetNDC(True)
        title.SetTextFont(42)
        title.SetTextSize(0.024)
        label = "Inclusive PassingMuonReco" if name == "inclusive" else composition["class_labels"][name]
        title.DrawLatex(0.13, 0.965, label)
        retained.extend([pad, title])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    config = common.load_yaml(args.region_config)
    composition = configured_background_composition(config)
    selections = configured_selections(config)
    if args.selection not in selections:
        raise ValueError(f"Unknown composition selection {args.selection!r}")
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if str(metadata(source, "analysis_scope")) != composition["scope"]:
        raise RuntimeError("Composition ROOT scope does not match configuration")
    if str(metadata(source, "track_definition_key")) != composition["track_key"]:
        raise RuntimeError("Composition ROOT track definition does not match configuration")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("composition_v7", "composition_v7", 1400, 900)
    pages = 0
    try:
        canvas.Print(f"{temporary}[")
        draw_summary(canvas, source, args.selection, composition)
        canvas.Print(temporary)
        pages += 1
        for section in composition["sections"].values():
            draw_section(canvas, args.selection, section, composition)
            canvas.Print(temporary)
            pages += 1
            for feature_name in section["features"]:
                feature = composition["features"][feature_name]
                if int(feature["dimension"]) == 1:
                    draw_1d(
                        canvas, source, args.selection, feature_name,
                        feature, composition,
                    )
                else:
                    draw_2d(
                        canvas, source, args.selection, feature_name,
                        feature, composition,
                    )
                canvas.Print(temporary)
                pages += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {pages} PassingMuonReco-only composition pages to {output}")


if __name__ == "__main__":
    main()
