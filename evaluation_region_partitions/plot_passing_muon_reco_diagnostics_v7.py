#!/usr/bin/env python3
"""Plot Run-8329 flat-v2 track diagnostics for one V7 detector selection."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ROOT

import plot_passing_muon_region_study_v4 as common
from passing_muon_v7_reco import configured_reco_diagnostics


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

POPULATIONS = (
    ("candidate", "Detector candidate", ROOT.kBlue + 1, 20),
    ("complement", "Detector topology failure", ROOT.kRed + 1, 21),
)


def metadata(source, name):
    return common.object_value(source, name)


def clone(source, path, name):
    return common.clone_histogram(source, path, name)


def histogram_path(selection, owner, feature, population, event_group=None):
    if owner == "event":
        if event_group:
            return (
                f"selections/{selection}/plot_groups/{event_group}/"
                f"event_features/{feature}/{population}"
            )
        return (
            f"selections/{selection}/event_features/{feature}/{population}"
        )
    return (
        f"selections/{selection}/tracks/{owner}/features/{feature}/{population}"
    )


def valid_count(source, selection, owner, feature, population, event_group=None):
    prefix = f"selection__{selection}__{population}"
    if owner == "event":
        if event_group:
            name = (
                f"{prefix}__plot_group__{event_group}__event__"
                f"{feature}__valid_count"
            )
        else:
            name = f"{prefix}__event__{feature}__valid_count"
    else:
        name = f"{prefix}__{owner}__{feature}__valid_count"
    return int(metadata(source, name))


def draw_header(selection, title):
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.030)
    label.DrawLatex(0.10, 0.965, "SND@LHC work in progress")
    label.SetTextAlign(31)
    label.DrawLatex(0.94, 0.965, "Run 8329 PassingMuonReco, flat_v2")
    label.SetTextAlign(13)
    label.SetTextSize(0.026)
    label.DrawLatex(0.10, 0.925, f"{selection}: {title}")
    return label


def draw_summary(
    canvas, source, selection, diagnostics, track_keys=None, group_label=None
):
    canvas.Clear()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextSize(0.036)
    summary_title = (
        f"{group_label} reconstructed-track summary"
        if group_label else "Reconstructed-track diagnostic summary"
    )
    text.DrawLatex(0.08, 0.88, summary_title)
    text.SetTextSize(0.023)
    candidate = int(metadata(source, f"selection__{selection}__candidate__count"))
    complement = int(metadata(source, f"selection__{selection}__complement__count"))
    lines = [
        f"Detector candidate PassingMuonReco events: {candidate:,}",
        f"Detector topology-failure PassingMuonReco events: {complement:,}",
        "All track-property shapes are conditional on a valid best track of that type.",
        "Associated measurements are hits for Hough tracks and may be clusters for Simple tracks.",
        "Measured-plane counts do not mean every geometrically crossed plane.",
        "MC tracks are not shown because the configured MC was not reconstructed.",
        "",
        "Best-track availability: candidate / topology failure",
    ]
    selected_track_keys = track_keys or list(diagnostics["track_types"])
    for track_key in selected_track_keys:
        track = diagnostics["track_types"][track_key]
        values = []
        for population, denominator in (
            ("candidate", candidate), ("complement", complement)
        ):
            count = int(metadata(
                source,
                f"selection__{selection}__{population}__{track_key}__has_best_count",
            ))
            fraction = 100.0 * count / denominator if denominator else 0.0
            values.append(f"{count:,} ({fraction:.3f}%)")
        lines.append(f"{track['label']}: {values[0]} / {values[1]}")
    y = 0.82
    for line in lines:
        text.DrawLatex(0.09, y, line)
        y -= 0.040 if line else 0.020
    header = draw_header(
        selection, f"{group_label} track diagnostics" if group_label else "track diagnostics"
    )
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def draw_section(canvas, selection, group_label, section_label, track_labels):
    canvas.Clear()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(42)
    text.SetTextAlign(22)
    text.SetTextSize(0.052)
    text.DrawLatex(0.50, 0.62, group_label)
    text.SetTextSize(0.044)
    text.DrawLatex(0.50, 0.51, section_label)
    text.SetTextSize(0.027)
    text.DrawLatex(0.50, 0.42, "Best-track distributions")
    text.SetTextSize(0.022)
    text.DrawLatex(0.50, 0.35, ", ".join(track_labels))
    header = draw_header(selection, f"{group_label}: {section_label}")
    canvas._keepalive = [text, header]
    canvas.Modified()
    canvas.Update()


def styled_items(source, selection, owner, feature_name, event_group=None):
    result = []
    for population, label, color, marker in POPULATIONS:
        histogram = clone(
            source,
            histogram_path(
                selection, owner, feature_name, population, event_group
            ),
            f"v7_{selection}_{owner}_{feature_name}_{population}",
        )
        common.normalize(histogram)
        histogram.SetLineColor(color)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker)
        histogram.SetMarkerSize(0.45)
        histogram.SetLineWidth(2)
        result.append({
            "population": population,
            "label": label,
            "color": color,
            "marker": marker,
            "histogram": histogram,
            "valid_count": valid_count(
                source, selection, owner, feature_name, population, event_group
            ),
        })
    return result


def draw_1d(
    canvas, source, selection, owner, feature_name, feature, owner_label,
    event_group=None,
):
    items = styled_items(
        source, selection, owner, feature_name, event_group
    )
    histograms = [item["histogram"] for item in items]
    canvas.Clear()
    upper = ROOT.TPad(f"upper_{owner}_{feature_name}", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad(f"lower_{owner}_{feature_name}", "", 0, 0, 1, 0.30)
    upper.SetLeftMargin(0.14)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.13)
    upper.SetBottomMargin(0.025)
    upper.SetLogy(bool(feature.get("logy", False)))
    lower.SetLeftMargin(0.14)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.035)
    lower.SetBottomMargin(0.36)
    lower.SetGridy(True)
    upper.Draw()
    lower.Draw()

    upper.cd()
    maximum = max((hist.GetMaximum() for hist in histograms), default=1.0)
    positive = common.positive_minimum_1d(histograms)
    first = histograms[0]
    first.SetTitle(f";{feature['axis_title']};Fraction of valid events")
    first.GetXaxis().SetLabelSize(0)
    if feature.get("logy", False):
        first.SetMinimum(max((positive or 1e-9) * 0.5, 1e-9))
        first.SetMaximum(max(maximum * 20.0, 1e-8))
    else:
        first.SetMinimum(0.0)
        first.SetMaximum(max(maximum * 1.35, 1e-8))
    first.Draw("E1 X0")
    histograms[1].Draw("E1 X0 SAME")
    legend = ROOT.TLegend(0.51, 0.73, 0.94, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    for item in items:
        legend.AddEntry(
            item["histogram"],
            f"{item['label']} (valid N={item['valid_count']:,})",
            "lep",
        )
    legend.Draw()
    header = draw_header(selection, f"{owner_label}: {feature['axis_title']}")

    lower.cd()
    ratio = histograms[1].Clone(f"ratio_{owner}_{feature_name}")
    ratio.SetDirectory(0)
    ratio.Divide(histograms[0])
    ratio.SetLineColor(POPULATIONS[1][2])
    ratio.SetMarkerColor(POPULATIONS[1][2])
    ratio.SetMarkerStyle(POPULATIONS[1][3])
    ratio.SetTitle(f";{feature['axis_title']};Failure / candidate")
    ratio.GetXaxis().SetTitleSize(0.13)
    ratio.GetXaxis().SetLabelSize(0.10)
    ratio.GetXaxis().SetTitleOffset(1.15)
    ratio.GetYaxis().SetTitleSize(0.095)
    ratio.GetYaxis().SetLabelSize(0.085)
    ratio.GetYaxis().SetTitleOffset(0.65)
    ratio.GetYaxis().SetNdivisions(505)
    ratio.SetMinimum(0.0)
    ratio.SetMaximum(5.0)
    ratio.Draw("E1 X0")
    unity = ROOT.TLine(float(feature["x_min"]), 1, float(feature["x_max"]), 1)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw("SAME")
    canvas._keepalive = [
        *histograms, ratio, unity, upper, lower, legend, header,
    ]
    canvas.Modified()
    canvas.Update()


def draw_2d(
    canvas, source, selection, owner, feature_name, feature, owner_label,
    event_group=None,
):
    items = styled_items(
        source, selection, owner, feature_name, event_group
    )
    histograms = [item["histogram"] for item in items]
    maximum = max((hist.GetMaximum() for hist in histograms), default=1.0)
    positive = common.positive_minimum_2d(histograms)
    logz = bool(feature.get("logz", False))
    minimum = max((positive or 1e-9) * 0.5, 1e-12) if logz else 0.0
    canvas.Clear()
    canvas.Divide(2, 1, 0.002, 0.002)
    retained = list(histograms)
    for index, item in enumerate(items, 1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.16)
        pad.SetLogz(logz)
        histogram = item["histogram"]
        histogram.SetMinimum(minimum)
        histogram.SetMaximum(max(maximum * (2 if logz else 1.05), 1e-12))
        histogram.SetTitle(
            f";{feature['x_title']};{feature['y_title']};Fraction of valid events"
        )
        histogram.Draw("COLZ")
        title = ROOT.TLatex()
        title.SetNDC(True)
        title.SetTextFont(42)
        title.SetTextSize(0.026)
        title.DrawLatex(
            0.13,
            0.965,
            f"{owner_label}, {item['label']} (valid N={item['valid_count']:,})",
        )
        retained.extend([pad, title])
    canvas._keepalive = retained
    canvas.Modified()
    canvas.Update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--plot-group",
        choices=("simple", "hough"),
        help=(
            "Write only one configured tracking-method group. If omitted, "
            "retain the original all-track report."
        ),
    )
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    config = common.load_yaml(args.region_config)
    diagnostics = configured_reco_diagnostics(config)
    if args.selection not in config["coverage_working_points"]:
        raise ValueError(f"Unknown V7 selection {args.selection!r}")
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    if str(metadata(source, "track_feature_schema")) != "flat_v2":
        raise RuntimeError("V7 track histogram input is not flat_v2")

    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    canvas = ROOT.TCanvas("v7_reco", "v7_reco", 1200, 850)
    page_count = 0
    try:
        canvas.Print(f"{temporary}[")
        plot_group = (
            diagnostics["plot_groups"][args.plot_group]
            if args.plot_group else None
        )
        draw_summary(
            canvas,
            source,
            args.selection,
            diagnostics,
            track_keys=plot_group["track_types"] if plot_group else None,
            group_label=plot_group["label"] if plot_group else None,
        )
        canvas.Print(temporary)
        page_count += 1
        event_features = (
            plot_group["event_features"]
            if plot_group else diagnostics["event_features"]
        )
        for feature_name, feature in event_features.items():
            if feature["dimension"] == 1:
                draw_1d(
                    canvas, source, args.selection, "event", feature_name,
                    feature,
                    f"{plot_group['label']} event-level tracks"
                    if plot_group else "Event-level tracks",
                    event_group=args.plot_group,
                )
            else:
                draw_2d(
                    canvas, source, args.selection, "event", feature_name,
                    feature,
                    f"{plot_group['label']} event-level tracks"
                    if plot_group else "Event-level tracks",
                    event_group=args.plot_group,
                )
            canvas.Print(temporary)
            page_count += 1
        sections = (
            plot_group["sections"]
            if plot_group else {
                "all": {
                    "label": "All reconstructed-track types",
                    "track_types": list(diagnostics["track_types"]),
                }
            }
        )
        for section in sections.values():
            if plot_group:
                draw_section(
                    canvas,
                    args.selection,
                    plot_group["label"],
                    section["label"],
                    [
                        diagnostics["track_types"][key]["label"]
                        for key in section["track_types"]
                    ],
                )
                canvas.Print(temporary)
                page_count += 1
            for track_key in section["track_types"]:
                track = diagnostics["track_types"][track_key]
                for feature_name, feature in diagnostics["track_features"][track_key].items():
                    if feature["dimension"] == 1:
                        draw_1d(
                            canvas, source, args.selection, track_key, feature_name,
                            feature, str(track["label"]),
                        )
                    else:
                        draw_2d(
                            canvas, source, args.selection, track_key, feature_name,
                            feature, str(track["label"]),
                        )
                    canvas.Print(temporary)
                    page_count += 1
        canvas.Print(f"{temporary}]")
        os.replace(temporary, output)
    finally:
        source.Close()
        canvas.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {page_count} V7 reconstructed-track pages to {output}")


if __name__ == "__main__":
    main()
