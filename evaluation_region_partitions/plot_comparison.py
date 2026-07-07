#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

import pandas as pd
import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str, repo_root: str | Path) -> str:
    if path.startswith("root://") or os.path.isabs(path):
        return path
    return str(Path(repo_root) / path)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def read_tnamed(root_file: ROOT.TFile, name: str, default: str = "") -> str:
    obj = root_file.Get(name)
    if obj is None:
        return default
    if hasattr(obj, "GetTitle"):
        return str(obj.GetTitle())
    return default



def read_tparam(root_file: ROOT.TFile, name: str, default: float = 0.0) -> float:
    obj = root_file.Get(name)
    if obj is None:
        return default
    if hasattr(obj, "GetVal"):
        return float(obj.GetVal())
    return default


def get_feature_config(hist_cfg: dict, feature: str) -> dict:
    features = hist_cfg.get("features", {})
    if feature not in features:
        raise KeyError(f"feature={feature!r} is not defined in histogram config")

    merged = dict(hist_cfg.get("defaults", {}))
    merged.update(features[feature])
    return merged


def output_path(config_value: str, repo_root: str | Path) -> str:
    if os.path.isabs(config_value) or str(config_value).startswith("root://"):
        return str(config_value)
    return str(Path(repo_root) / config_value)


def grouped_hist_path(
    hist_dir: str,
    partition_version: str,
    region_version: str,
    hist_version: str,
    eval_version: str,
    feature: str,
    base_cut_key: str,
    extra_cut_key: str,
    region: str,
    particle_group: str,
) -> str:
    return str(
        Path(hist_dir)
        / f"{partition_version}_{region_version}"
        / f"{hist_version}__{eval_version}"
        / feature
        / base_cut_key
        / extra_cut_key
        / region
        / f"{particle_group}.root"
    )


def plot_output_dir(
    plot_dir: str,
    partition_version: str,
    region_version: str,
    hist_version: str,
    eval_version: str,
    feature: str,
    base_cut_key: str,
    extra_cut_key: str,
) -> Path:
    return (
        Path(plot_dir)
        / f"{partition_version}_{region_version}"
        / f"{hist_version}__{eval_version}"
        / feature
        / base_cut_key
        / extra_cut_key
    )


def reference_lumi(metadata: pd.DataFrame, eval_cfg: dict) -> float:
    ref = eval_cfg["normalization"]["reference"]
    selected = metadata.loc[
        (metadata["region"].astype(str) == str(ref["region"]))
        & (metadata["particle_group"].astype(str) == str(ref["particle_group"]))
    ]

    lumi = pd.to_numeric(selected.get("lumi_per_partition"), errors="coerce").fillna(0.0).sum()
    lumi = float(lumi)
    if lumi <= 0 and eval_cfg["normalization"].get("fail_on_missing_reference", True):
        raise RuntimeError(
            "Reference luminosity is zero or missing for "
            f"region={ref['region']}, particle_group={ref['particle_group']}"
        )
    return lumi


def configured_regions(eval_cfg: dict) -> set[str]:
    return set(str(v) for v in eval_cfg.get("comparisons", {}).get("regions", []))


def configured_particle_groups(eval_cfg: dict) -> set[str]:
    return set(str(v) for v in eval_cfg.get("comparisons", {}).get("particle_groups", []))


def is_eventbuilder_particle_group(particle_group: object) -> bool:
    return "EventBuilder" in str(particle_group)


def is_real_data_particle_group(particle_group: object) -> bool:
    return str(particle_group) == "real_data"


def page_keys(metadata: pd.DataFrame, eval_cfg: dict, mode: str) -> list[str]:
    allowed_regions = configured_regions(eval_cfg)
    allowed_particles = configured_particle_groups(eval_cfg)

    if mode == "particles_in_region":
        regions = sorted(str(v) for v in metadata["region"].dropna().unique())
        return [region for region in regions if not allowed_regions or region in allowed_regions]

    if mode == "regions_for_particle":
        particles = sorted(str(v) for v in metadata["particle_group"].dropna().unique())
        return [particle for particle in particles if not allowed_particles or particle in allowed_particles]

    raise ValueError(f"Unknown comparison mode: {mode}")


def particle_comparison_pages(metadata: pd.DataFrame, eval_cfg: dict) -> list[dict]:
    allowed_regions = configured_regions(eval_cfg)
    allowed_particles = configured_particle_groups(eval_cfg)
    particle_labels = plot_config(eval_cfg).get("particle_labels", {})
    region_labels = plot_config(eval_cfg).get("region_labels", {})
    pages = []

    regions = sorted(str(v) for v in metadata["region"].dropna().unique())
    for region in regions:
        if allowed_regions and region not in allowed_regions:
            continue

        rows = metadata.loc[metadata["region"].astype(str) == region].copy()
        if allowed_particles:
            rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]

        particles = sorted(str(v) for v in rows["particle_group"].dropna().unique())
        for page_kind, page_label, keep_eventbuilder in [
            ("no_eventbuilder", "No EventBuilder", False),
            ("eventbuilder", "EventBuilder", True),
        ]:
            page_particles = [
                particle
                for particle in particles
                if is_real_data_particle_group(particle)
                or is_eventbuilder_particle_group(particle) == keep_eventbuilder
            ]
            if not page_particles:
                continue

            page_key = f"{region}::{page_kind}"
            region_label = region_labels.get(region, region)
            pages.append(
                {
                    "comparison_mode": f"particles_in_region_{page_kind}",
                    "page_key": page_key,
                    "page_label": f"{page_label} particles in {region_label}",
                    "items": [
                        {
                            "page_key": page_key,
                            "page_label": f"{page_label} particles in {region_label}",
                            "group_name": particle,
                            "group_label": particle_labels.get(particle, particle),
                            "region": region,
                            "particle_group": particle,
                        }
                        for particle in page_particles
                    ],
                }
            )

    return pages


def comparison_items(metadata: pd.DataFrame, eval_cfg: dict, mode: str, page_key: str) -> list[dict[str, str]]:
    allowed_regions = configured_regions(eval_cfg)
    allowed_particles = configured_particle_groups(eval_cfg)

    if mode == "particles_in_region":
        rows = metadata.loc[metadata["region"].astype(str) == str(page_key)].copy()
        if allowed_particles:
            rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]
        particles = sorted(str(v) for v in rows["particle_group"].dropna().unique())
        return [
            {
                "page_key": str(page_key),
                "group_name": particle,
                "region": str(page_key),
                "particle_group": particle,
            }
            for particle in particles
        ]

    if mode == "regions_for_particle":
        rows = metadata.loc[metadata["particle_group"].astype(str) == str(page_key)].copy()
        if allowed_regions:
            rows = rows.loc[rows["region"].astype(str).isin(allowed_regions)]
        regions = sorted(str(v) for v in rows["region"].dropna().unique())
        return [
            {
                "page_key": str(page_key),
                "group_name": region,
                "region": region,
                "particle_group": str(page_key),
            }
            for region in regions
        ]

    raise ValueError(f"Unknown comparison mode: {mode}")


def load_group_hist(hist_path: str) -> dict | None:
    if not os.path.exists(hist_path):
        print(f"[skip] missing histogram file: {hist_path}")
        return None

    root_file = ROOT.TFile.Open(hist_path)
    if not root_file or root_file.IsZombie():
        print(f"[skip] could not open histogram file: {hist_path}")
        return None

    hist_obj = root_file.Get("hist")
    if hist_obj is None:
        print(f"[skip] missing 'hist' in {hist_path}")
        root_file.Close()
        return None

    hist = hist_obj.Clone(f"{safe_name(Path(hist_path).stem)}_clone")
    hist.SetDirectory(0)

    result = {
        "hist": hist,
        "hist_path": hist_path,
        "lumi": read_tparam(root_file, "lumi", 0.0),
        "raw_integral": read_tparam(root_file, "raw_integral", float(hist.Integral())),
        "selected_entries": int(read_tparam(root_file, "selected_entries", 0.0)),
        "n_partitions": int(read_tparam(root_file, "n_partitions", 0.0)),
        "source_partition_ids": read_tnamed(root_file, "source_partition_ids", ""),
        "selection": read_tnamed(root_file, "selection", ""),
    }
    root_file.Close()
    return result


def color_for_index(index: int) -> int:
    colors = [
        ROOT.kBlack,
        ROOT.kRed + 1,
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 2,
        ROOT.kOrange + 7,
        ROOT.kCyan + 2,
        ROOT.kViolet + 1,
        ROOT.kAzure + 6,
        ROOT.kPink + 6,
        ROOT.kTeal + 3,
        ROOT.kGray + 2,
    ]
    return colors[index % len(colors)]


def color_from_config(value: object, fallback_index: int = 0) -> int:
    if isinstance(value, int):
        return value

    name = str(value or "").strip()
    if name.startswith("#"):
        return ROOT.TColor.GetColor(name)

    named_colors = {
        "black": ROOT.kBlack,
        "blue": ROOT.kBlue + 1,
        "blue_dark": ROOT.kBlue + 2,
        "blue_light": ROOT.kAzure + 6,
        "red": ROOT.kRed + 1,
        "red_dark": ROOT.kRed + 2,
        "red_light": ROOT.kPink + 6,
        "green": ROOT.kGreen + 2,
        "green_dark": ROOT.kGreen + 3,
        "green_light": ROOT.kSpring + 5,
        "magenta": ROOT.kMagenta + 2,
        "orange": ROOT.kOrange + 7,
        "orange_dark": ROOT.kOrange + 7,
        "orange_light": ROOT.kOrange - 2,
        "cyan": ROOT.kCyan + 2,
        "gray": ROOT.kGray + 2,
    }
    return named_colors.get(name, color_for_index(fallback_index))


def scifi_position_feature(feature_name: str) -> tuple[int | None, str | None]:
    match = re.fullmatch(r"(?:avg|qdcAvg)_scifi([1-5])?_(x|y)", str(feature_name))
    if not match:
        return None, None

    plane = int(match.group(1)) if match.group(1) else None
    axis = match.group(2)
    return plane, axis


def fiducial_overlays_for_feature(feature_name: str, eval_cfg: dict) -> list[dict]:
    _, axis = scifi_position_feature(feature_name)
    if axis is None:
        return []

    cfg = plot_config(eval_cfg).get("fiducial_overlays", {})
    if not cfg.get("enabled", False):
        return []

    overlay_keys = ["original_sndsw", "separated_s12", "separated_s345"]

    overlays = []
    for key in overlay_keys:
        overlay_cfg = cfg.get(key, {})
        values = overlay_cfg.get(axis)
        if not values or len(values) != 2:
            continue
        overlays.append(
            {
                "key": key,
                "label": str(overlay_cfg.get("label", key)),
                "values": [float(values[0]), float(values[1])],
                "color": overlay_cfg.get("color"),
            }
        )
    return overlays


def draw_fiducial_overlays(
    canvas: ROOT.TCanvas,
    feature_name: str,
    eval_cfg: dict,
    y_min: float,
    y_max: float,
    legend: ROOT.TLegend | None = None,
) -> None:
    for index, overlay in enumerate(fiducial_overlays_for_feature(feature_name, eval_cfg)):
        color = color_from_config(overlay.get("color"), index)
        legend_line = None
        for value in overlay["values"]:
            line = ROOT.TLine(value, y_min, value, y_max)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("SAME")
            canvas._region_eval_keepalive.append(line)
            if legend_line is None:
                legend_line = line
        if legend is not None and legend_line is not None:
            legend.AddEntry(legend_line, overlay["label"], "l")


def plot_config(eval_cfg: dict) -> dict:
    return eval_cfg.get("plot", {})


def group_label(eval_cfg: dict, comparison_mode: str, group_name: str) -> str:
    cfg = plot_config(eval_cfg)
    if comparison_mode.startswith("particles_in_region"):
        return str(cfg.get("particle_labels", {}).get(group_name, group_name))
    if comparison_mode == "regions_for_particle":
        return str(cfg.get("region_labels", {}).get(group_name, group_name))
    if comparison_mode in {"feature_pair", "overall_feature_pair"}:
        return str(group_name)
    return group_name


def page_label(eval_cfg: dict, comparison_mode: str, page_key: str) -> str:
    cfg = plot_config(eval_cfg)
    if comparison_mode.startswith("particles_in_region"):
        region = str(page_key).split("::", 1)[0]
        return str(cfg.get("region_labels", {}).get(region, region))
    if comparison_mode == "regions_for_particle":
        return str(cfg.get("particle_labels", {}).get(page_key, page_key))
    return page_key


def style_for_group(eval_cfg: dict, comparison_mode: str, group_name: str, index: int) -> dict:
    cfg = plot_config(eval_cfg)
    if comparison_mode.startswith("particles_in_region"):
        styles = cfg.get("particle_styles", {})
    elif comparison_mode == "regions_for_particle":
        styles = cfg.get("region_styles", {})
    elif comparison_mode == "eventbuilder_veto_timing":
        styles = cfg.get("particle_styles", {})
    elif comparison_mode.startswith("overall_particle_overlay"):
        styles = dict(cfg.get("particle_styles", {}))
        styles.update(
            {
                "real_data_no_veto": {"color": "black", "marker": 20},
                "real_data_has_veto": {"color": "gray", "marker": 24},
            }
        )
    elif comparison_mode in {"feature_pair", "overall_feature_pair"}:
        styles = {
            "average": {"color": "black", "marker": 20},
            "qdc_weighted": {"color": "red", "marker": 24},
        }
    else:
        styles = {}
    return dict(styles.get(group_name, {}))


def style_hist(hist: ROOT.TH1, style_cfg: dict, index: int) -> None:
    color = color_from_config(style_cfg.get("color"), index)
    marker = int(style_cfg.get("marker", 20 + (index % 14)))
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetLineWidth(1)
    hist.SetMarkerStyle(marker)
    hist.SetMarkerSize(float(style_cfg.get("marker_size", 0.9)))


def get_ymax(hists: list[ROOT.TH1], logy: bool) -> float:
    ymax = max((hist.GetMaximum() for hist in hists), default=1.0)
    ymax = max(ymax, 1.0)
    return ymax * (20.0 if logy else 1.35)


def configured_ymax(feature_cfg: dict, default_value: float) -> float:
    if feature_cfg.get("y_max") is None:
        return default_value
    return float(feature_cfg["y_max"])


def get_ymin(hists: list[ROOT.TH1], logy: bool) -> float:
    if not logy:
        return 0.0

    positive_values = []
    for hist in hists:
        for ibin in range(1, hist.GetNbinsX() + 1):
            value = float(hist.GetBinContent(ibin))
            if math.isfinite(value) and value > 0:
                positive_values.append(value)

    if not positive_values:
        return 0.1

    return min(0.1, min(positive_values) * 0.5)


def ratio_range(hist: ROOT.TH1) -> tuple[float, float]:
    values = []
    for ibin in range(1, hist.GetNbinsX() + 1):
        value = float(hist.GetBinContent(ibin))
        if math.isfinite(value) and value > 0:
            values.append(value)
    if not values:
        return 0.0, 2.0

    values.sort()
    if len(values) >= 10:
        low_index = int(0.05 * (len(values) - 1))
        high_index = int(0.95 * (len(values) - 1))
        ymin = values[low_index]
        ymax = values[high_index]
    else:
        ymin = min(values)
        ymax = max(values)

    if ymin <= 1.0 <= ymax or (0.5 <= ymin <= 2.0) or (0.5 <= ymax <= 2.0):
        ymin = min(ymin, 1.0)
        ymax = max(ymax, 1.0)

    if ymin == ymax:
        return max(0.0, ymin * 0.8), ymax * 1.2 if ymax > 0 else 2.0
    padding = 0.2 * (ymax - ymin)
    return max(0.0, ymin - padding), ymax + padding


def make_ratio_hist(numerator: ROOT.TH1, denominator: ROOT.TH1, name: str) -> ROOT.TH1:
    ratio = numerator.Clone(name)
    ratio.SetDirectory(0)
    ratio.Divide(denominator)
    for ibin in range(1, ratio.GetNbinsX() + 1):
        denominator_value = float(denominator.GetBinContent(ibin))
        value = float(ratio.GetBinContent(ibin))
        error = float(ratio.GetBinError(ibin))
        if denominator_value == 0.0 or not math.isfinite(value):
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)
        elif not math.isfinite(error):
            ratio.SetBinError(ibin, 0.0)
    return ratio


def plot_normalization(feature_cfg: dict) -> str:
    return str(feature_cfg.get("plot_normalization", "lumi") or "lumi").strip()


def normalized_hist_for_plot(
    source_hist: ROOT.TH1,
    clone_name: str,
    feature_cfg: dict,
    reference_lumi_value: float,
    lumi: float,
) -> tuple[ROOT.TH1, float | str]:
    hist = source_hist.Clone(clone_name)
    hist.SetDirectory(0)

    normalization = plot_normalization(feature_cfg)
    if normalization == "unit_area":
        integral = float(hist.Integral())
        scale_factor = 1.0 / integral if integral > 0.0 else 1.0
    elif normalization == "lumi":
        scale_factor = reference_lumi_value / lumi
    else:
        raise ValueError(f"Unknown plot_normalization={normalization!r}")

    hist.Scale(scale_factor)
    return hist, scale_factor


def legend_value_label(hist: ROOT.TH1, feature_cfg: dict, scaled_integral: float) -> str:
    if plot_normalization(feature_cfg) == "unit_area":
        return f"mean={hist.GetMean():.3g}, std={hist.GetStdDev():.3g}"
    return f"{scaled_integral:.3g}"


def draw_overlay(
    canvas: ROOT.TCanvas,
    output_pdf: Path,
    feature_name: str,
    feature_cfg: dict,
    eval_cfg: dict,
    comparison_mode: str,
    page_key: str,
    title: str,
    reference_lumi_value: float,
    curves: list[dict],
) -> None:
    ROOT.gStyle.SetOptStat(0)

    x_min = float(feature_cfg["x_min"])
    x_max = float(feature_cfg["x_max"])
    bin_width = float(feature_cfg["bin_width"])
    n_bins = int(round((x_max - x_min) / bin_width))
    logy = bool(feature_cfg.get("logy", False))
    axis_title = str(feature_cfg.get("axis_title", ""))
    draw_ratio = len(curves) == 2

    canvas.Clear()
    canvas._region_eval_keepalive = []
    if draw_ratio:
        top_pad = ROOT.TPad(f"top_{safe_name(title)}", "", 0.0, 0.30, 1.0, 1.0)
        bottom_pad = ROOT.TPad(f"ratio_{safe_name(title)}", "", 0.0, 0.0, 1.0, 0.30)
        canvas._region_eval_keepalive.extend([top_pad, bottom_pad])
        top_pad.SetLeftMargin(0.12)
        top_pad.SetRightMargin(0.06)
        top_pad.SetTopMargin(0.08)
        top_pad.SetBottomMargin(0.03)
        top_pad.SetLogy(1 if logy else 0)
        bottom_pad.SetLeftMargin(0.12)
        bottom_pad.SetRightMargin(0.06)
        bottom_pad.SetTopMargin(0.04)
        bottom_pad.SetBottomMargin(0.35)
        bottom_pad.SetGridy(True)
        top_pad.Draw()
        bottom_pad.Draw()
        top_pad.cd()
    else:
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.06)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        canvas.SetLogy(1 if logy else 0)

    frame = ROOT.TH1D(f"frame_{safe_name(title)}", "", n_bins, x_min, x_max)
    frame.SetDirectory(0)
    frame.SetTitle(title)
    frame.GetXaxis().SetTitle(axis_title)
    y_axis_quantity = "Fraction" if plot_normalization(feature_cfg) == "unit_area" else "Events"
    frame.GetYaxis().SetTitle(f"{y_axis_quantity} / {bin_width:.6g}")
    frame.GetXaxis().SetTitleSize(0.0 if draw_ratio else 0.045)
    frame.GetXaxis().SetLabelSize(0.0 if draw_ratio else 0.04)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleOffset(1.2)
    hists = [curve["hist_scaled"] for curve in curves]
    frame.SetMinimum(get_ymin(hists, logy))
    frame.SetMaximum(configured_ymax(feature_cfg, get_ymax(hists, logy)))
    frame.Draw()

    legend = ROOT.TLegend(0.58, 0.50, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    for index, curve in enumerate(curves):
        hist = curve["hist_scaled"]
        style_cfg = style_for_group(eval_cfg, comparison_mode, curve["group_name"], index)
        style_hist(hist, style_cfg, index)
        hist.Draw("E1 SAME")
        label = curve.get("group_label") or group_label(eval_cfg, comparison_mode, curve["group_name"])
        legend.AddEntry(
            hist,
            f"{label} ({legend_value_label(hist, feature_cfg, curve['scaled_integral'])})",
            "lep",
        )

    draw_fiducial_overlays(
        canvas=canvas,
        feature_name=feature_name,
        eval_cfg=eval_cfg,
        y_min=frame.GetMinimum(),
        y_max=frame.GetMaximum(),
        legend=legend,
    )
    legend.Draw()
    draw_info_box(
        eval_cfg=eval_cfg,
        feature_cfg=feature_cfg,
        comparison_mode=comparison_mode,
        page_key=page_key,
        base_cut_key=curves[0]["base_cut_key"],
        extra_cut_key=curves[0]["extra_cut_key"],
        reference_lumi_value=reference_lumi_value,
    )

    if draw_ratio:
        bottom_pad.cd()
        numerator = curves[1]["hist_scaled"]
        denominator = curves[0]["hist_scaled"]
        ratio = make_ratio_hist(
            numerator,
            denominator,
            f"ratio_{safe_name(title)}_{safe_name(curves[1]['group_name'])}_over_{safe_name(curves[0]['group_name'])}",
        )
        ratio_title = (
            f"{curves[1].get('group_label') or curves[1]['group_name']} / "
            f"{curves[0].get('group_label') or curves[0]['group_name']}"
        )
        ymin, ymax = ratio_range(ratio)
        ratio.SetTitle("")
        ratio.GetXaxis().SetTitle(axis_title)
        ratio.GetYaxis().SetTitle("Ratio")
        ratio.GetYaxis().SetTitleSize(0.10)
        ratio.GetYaxis().SetTitleOffset(0.48)
        ratio.GetYaxis().SetLabelSize(0.08)
        ratio.GetYaxis().SetNdivisions(505)
        ratio.GetXaxis().SetTitleSize(0.11)
        ratio.GetXaxis().SetLabelSize(0.09)
        ratio.GetXaxis().SetTitleOffset(1.0)
        ratio.SetMinimum(ymin)
        ratio.SetMaximum(ymax)
        ratio.SetLineColor(ROOT.kBlack)
        ratio.SetMarkerColor(ROOT.kBlack)
        ratio.SetMarkerStyle(20)
        ratio.SetMarkerSize(0.7)
        ratio.Draw("E1")

        unity = ROOT.TLine(x_min, 1.0, x_max, 1.0)
        unity.SetLineStyle(2)
        unity.SetLineColor(ROOT.kGray + 2)
        unity.Draw("SAME")

        ratio_text = ROOT.TText()
        ratio_text.SetTextSize(0.08)
        ratio_text.SetTextAlign(13)
        ratio_text.DrawTextNDC(0.16, 0.88, ratio_title)
        canvas._region_eval_keepalive.extend([ratio, unity, ratio_text])

    canvas.Modified()
    canvas.Update()
    canvas.Print(str(output_pdf))


def write_summary(summary_csv: Path, rows: list[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison_mode",
        "page_key",
        "page_label",
        "feature",
        "feature_for_curve",
        "base_cut_key",
        "extra_cut_key",
        "group_name",
        "group_label",
        "region",
        "particle_group",
        "hist_path",
        "lumi",
        "reference_lumi",
        "scale_factor",
        "raw_integral",
        "scaled_integral",
        "selected_entries",
        "n_partitions",
        "selection",
        "source_partition_ids",
        "source_regions",
        "source_hist_paths",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote summary: {summary_csv}")


def clean_title(eval_cfg: dict, comparison_mode: str, page_key: str) -> str:
    label = page_label(eval_cfg, comparison_mode, page_key)
    if comparison_mode.startswith("particles_in_region"):
        if str(page_key).endswith("::eventbuilder"):
            return "Particle comparison: EventBuilder"
        return "Particle comparison: no EventBuilder"
    if comparison_mode == "regions_for_particle":
        return f"Region comparison for {label}"
    if comparison_mode == "eventbuilder_veto_timing":
        return label
    if comparison_mode.startswith("overall_particle_overlay"):
        if "no_veto" in str(page_key):
            if str(page_key).endswith("::eventbuilder"):
                return "Overall no-veto particle overlay: EventBuilder"
            return "Overall no-veto particle overlay: no EventBuilder"
        if str(page_key).endswith("::eventbuilder"):
            return "Overall particle overlay: EventBuilder"
        return "Overall particle overlay: no EventBuilder"
    if comparison_mode in {"feature_pair", "overall_feature_pair"}:
        return label
    return label


def cut_label(eval_cfg: dict, cut_type: str, cut_key: str) -> str:
    cfg = plot_config(eval_cfg).get(f"{cut_type}_cut_labels", {})
    return str(cfg.get(cut_key, cut_key))


def draw_info_box(
    eval_cfg: dict,
    feature_cfg: dict,
    comparison_mode: str,
    page_key: str,
    base_cut_key: str,
    extra_cut_key: str,
    reference_lumi_value: float,
) -> None:
    cfg = plot_config(eval_cfg)
    if plot_normalization(feature_cfg) == "unit_area":
        lines = ["Normalization: unit area"]
    else:
        lines = [f"L_ref = {reference_lumi_value:.4g} fb^-1"]

    if comparison_mode.startswith("particles_in_region"):
        region_label = page_label(eval_cfg, comparison_mode, page_key)
        lines.append(f"Region: {region_label}")
    elif comparison_mode == "eventbuilder_veto_timing":
        region = str(page_key).split("::", 1)[0]
        region_label = plot_config(eval_cfg).get("region_labels", {}).get(region, region)
        lines.append(f"Region: {region_label}")
    elif comparison_mode.startswith("overall_particle_overlay"):
        if "no_veto" in str(page_key):
            lines.append("Regions: no veto combined")
        else:
            lines.append("Regions: combined")
    elif comparison_mode == "feature_pair":
        region = str(page_key).split("::", 1)[0]
        particle = str(page_key).split("::", 1)[1] if "::" in str(page_key) else ""
        region_label = plot_config(eval_cfg).get("region_labels", {}).get(region, region)
        particle_label = plot_config(eval_cfg).get("particle_labels", {}).get(particle, particle)
        lines.append(f"Region: {region_label}")
        lines.append(f"Particle: {particle_label}")
    elif comparison_mode == "overall_feature_pair":
        label = str(page_key).rsplit("::", 1)[1] if "::" in str(page_key) else str(page_key)
        if "no_veto" in str(page_key):
            lines.append("Regions: no veto combined")
        else:
            lines.append("Regions: combined")
        lines.append(f"Sample: {label}")

    lines.append(str(cfg.get("region_preselection_label", "Preselection: SciFi count > 35")))
    if base_cut_key != "none":
        lines.append(f"Base cut: {cut_label(eval_cfg, 'base', base_cut_key)}")
    if extra_cut_key != "none":
        lines.append(f"Extra cut: {cut_label(eval_cfg, 'extra', extra_cut_key)}")
    feature_cut = str(feature_cfg.get("feature_cut", "") or "").strip()
    if feature_cut:
        lines.append(f"Feature cut: {feature_cut}")

    text = ROOT.TText()
    text.SetTextSize(0.027)
    text.SetTextAlign(13)
    y = 0.86
    for line in lines:
        text.DrawTextNDC(0.14, y, line)
        y -= 0.038

def build_curves_for_page(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    hist_cfg: dict,
    eval_cfg: dict,
    page_key: str,
) -> tuple[list[dict], float]:
    repo_root = Path(args.repo_root).resolve()
    hist_dir = output_path(eval_cfg["outputs"]["hist_dir"], repo_root)
    reference_lumi_value = reference_lumi(metadata, eval_cfg)
    hist_version = hist_cfg["version"]
    eval_version = eval_cfg["version"]

    items = comparison_items(metadata, eval_cfg, args.comparison_mode, page_key)
    curves = []

    for item in items:
        hist_path = grouped_hist_path(
            hist_dir=hist_dir,
            partition_version=args.partition_version,
            region_version=args.region_version,
            hist_version=hist_version,
            eval_version=eval_version,
            feature=args.feature,
            base_cut_key=args.base_cut_key,
            extra_cut_key=args.extra_cut_key,
            region=item["region"],
            particle_group=item["particle_group"],
        )
        loaded = load_group_hist(hist_path)
        if loaded is None:
            continue

        lumi = float(loaded["lumi"])
        if lumi <= 0:
            raise RuntimeError(f"Histogram has non-positive lumi: {hist_path}")

        hist_scaled, scale_factor = normalized_hist_for_plot(
            source_hist=loaded["hist"],
            clone_name=f"h_scaled_{safe_name(item['group_name'])}",
            feature_cfg=hist_cfg["features"][args.feature],
            reference_lumi_value=reference_lumi_value,
            lumi=lumi,
        )

        raw_integral = float(loaded["raw_integral"])
        scaled_integral = float(hist_scaled.Integral())
        curves.append(
            {
                **item,
                **loaded,
                "hist_scaled": hist_scaled,
                "scale_factor": scale_factor,
                "reference_lumi": reference_lumi_value,
                "raw_integral": raw_integral,
                "scaled_integral": scaled_integral,
                "base_cut_key": args.base_cut_key,
                "extra_cut_key": args.extra_cut_key,
                "group_label": group_label(eval_cfg, args.comparison_mode, item["group_name"]),
                "page_label": page_label(eval_cfg, args.comparison_mode, item["page_key"]),
            }
        )

    if not curves:
        raise RuntimeError(
            f"No histograms were loaded for mode={args.comparison_mode}, page_key={page_key}"
        )

    return curves, reference_lumi_value


def build_curves_for_items(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    hist_cfg: dict,
    eval_cfg: dict,
    items: list[dict[str, str]],
    comparison_mode: str,
) -> tuple[list[dict], float]:
    repo_root = Path(args.repo_root).resolve()
    hist_dir = output_path(eval_cfg["outputs"]["hist_dir"], repo_root)
    reference_lumi_value = reference_lumi(metadata, eval_cfg)
    hist_version = hist_cfg["version"]
    eval_version = eval_cfg["version"]
    curves = []

    for index, item in enumerate(items):
        hist_path = grouped_hist_path(
            hist_dir=hist_dir,
            partition_version=args.partition_version,
            region_version=args.region_version,
            hist_version=hist_version,
            eval_version=eval_version,
            feature=args.feature,
            base_cut_key=args.base_cut_key,
            extra_cut_key=args.extra_cut_key,
            region=item["region"],
            particle_group=item["particle_group"],
        )
        loaded = load_group_hist(hist_path)
        if loaded is None:
            continue

        lumi = float(loaded["lumi"])
        if lumi <= 0:
            raise RuntimeError(f"Histogram has non-positive lumi: {hist_path}")

        hist_scaled, scale_factor = normalized_hist_for_plot(
            source_hist=loaded["hist"],
            clone_name=f"h_scaled_{safe_name(item['group_name'])}_{index}",
            feature_cfg=hist_cfg["features"][args.feature],
            reference_lumi_value=reference_lumi_value,
            lumi=lumi,
        )

        raw_integral = float(loaded["raw_integral"])
        scaled_integral = float(hist_scaled.Integral())
        curves.append(
            {
                **item,
                **loaded,
                "hist_scaled": hist_scaled,
                "scale_factor": scale_factor,
                "reference_lumi": reference_lumi_value,
                "raw_integral": raw_integral,
                "scaled_integral": scaled_integral,
                "base_cut_key": args.base_cut_key,
                "extra_cut_key": args.extra_cut_key,
                "group_label": item.get("group_label")
                or group_label(eval_cfg, comparison_mode, item["group_name"]),
                "page_label": item.get("page_label", item["page_key"]),
            }
        )

    return curves, reference_lumi_value


def eventbuilder_veto_timing_pages(eval_cfg: dict, feature: str) -> list[dict]:
    cfg = eval_cfg.get("eventbuilder_veto_timing", {})
    timing_features = set(str(v) for v in cfg.get("features", []))
    if feature not in timing_features:
        return []

    particle_labels = plot_config(eval_cfg).get("particle_labels", {})
    region_labels = plot_config(eval_cfg).get("region_labels", {})
    pages = []
    for region in cfg.get("regions", []):
        region = str(region)
        for pair in cfg.get("pairs", []):
            normal = str(pair["normal"])
            eventbuilder = str(pair["eventbuilder"])
            page_key = f"{region}::{normal}_vs_{eventbuilder}"
            normal_label = particle_labels.get(normal, normal)
            eventbuilder_label = particle_labels.get(eventbuilder, eventbuilder)
            region_label = region_labels.get(region, region)
            pages.append(
                {
                    "page_key": page_key,
                    "page_label": f"{normal_label} vs {eventbuilder_label} in {region_label}",
                    "items": [
                        {
                            "page_key": page_key,
                            "page_label": f"{normal_label} vs {eventbuilder_label} in {region_label}",
                            "group_name": normal,
                            "group_label": normal_label,
                            "region": region,
                            "particle_group": normal,
                        },
                        {
                            "page_key": page_key,
                            "page_label": f"{normal_label} vs {eventbuilder_label} in {region_label}",
                            "group_name": eventbuilder,
                            "group_label": eventbuilder_label,
                            "region": region,
                            "particle_group": eventbuilder,
                        },
                    ],
                }
            )
    return pages


def overall_particle_overlay_pages(metadata: pd.DataFrame, eval_cfg: dict) -> list[dict]:
    cfg = eval_cfg.get("overall_particle_overlay", {})
    if not cfg.get("enabled", False):
        return []

    allowed_regions = configured_regions(eval_cfg)
    allowed_particles = configured_particle_groups(eval_cfg)
    particle_labels = plot_config(eval_cfg).get("particle_labels", {})

    rows = metadata.copy()
    if allowed_regions:
        rows = rows.loc[rows["region"].astype(str).isin(allowed_regions)]
    if allowed_particles:
        rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]

    all_regions = sorted(str(v) for v in rows["region"].dropna().unique())
    particles = sorted(str(v) for v in rows["particle_group"].dropna().unique())

    data_items = []
    if "real_data" in particles:
        for group_name, group_cfg in cfg.get("data_region_groups", {}).items():
            regions = [
                str(region)
                for region in group_cfg.get("regions", [])
                if (not allowed_regions or str(region) in allowed_regions)
            ]
            if not regions:
                continue
            data_items.append(
                {
                    "group_name": str(group_name),
                    "group_label": str(group_cfg.get("label", group_name)),
                    "particle_group": "real_data",
                    "regions": regions,
                }
            )

    normal_items = []
    eventbuilder_items = []
    for particle in particles:
        if is_real_data_particle_group(particle):
            continue

        particle_regions = sorted(
            str(v)
            for v in rows.loc[rows["particle_group"].astype(str) == particle, "region"]
            .dropna()
            .unique()
        )
        if not particle_regions:
            continue

        item = {
            "group_name": particle,
            "group_label": particle_labels.get(particle, particle),
            "particle_group": particle,
            "regions": particle_regions or all_regions,
        }
        if is_eventbuilder_particle_group(particle):
            eventbuilder_items.append(item)
        else:
            normal_items.append(item)

    pages = []
    if normal_items or data_items:
        pages.append(
            {
                "comparison_mode": "overall_particle_overlay_no_eventbuilder",
                "page_key": "overall::no_eventbuilder",
                "page_label": "Overall particle overlay: no EventBuilder",
                "items": data_items + normal_items,
            }
        )

    if cfg.get("split_eventbuilder", True):
        if eventbuilder_items:
            pages.append(
                {
                    "comparison_mode": "overall_particle_overlay_eventbuilder",
                    "page_key": "overall::eventbuilder",
                    "page_label": "Overall particle overlay: EventBuilder",
                    "items": data_items + eventbuilder_items,
                }
            )
    elif eventbuilder_items:
        pages.append(
            {
                "comparison_mode": "overall_particle_overlay_all",
                "page_key": "overall::all",
                "page_label": "Overall particle overlay",
                "items": data_items + normal_items + eventbuilder_items,
            }
        )

    no_veto_cfg = cfg.get("no_veto_particle_overlay", {})
    no_veto_regions = [
        str(region)
        for region in no_veto_cfg.get("regions", [])
        if (not allowed_regions or str(region) in allowed_regions)
    ]
    if no_veto_cfg.get("enabled", False) and no_veto_regions:
        no_veto_data_items = []
        if "real_data" in particles:
            no_veto_data_items.append(
                {
                    "group_name": "real_data_no_veto",
                    "group_label": str(no_veto_cfg.get("data_label", "Real data no veto")),
                    "particle_group": "real_data",
                    "regions": no_veto_regions,
                }
            )

        no_veto_normal_items = []
        no_veto_eventbuilder_items = []
        for particle in particles:
            if is_real_data_particle_group(particle):
                continue

            available_regions = set(
                str(v)
                for v in rows.loc[rows["particle_group"].astype(str) == particle, "region"]
                .dropna()
                .unique()
            )
            particle_no_veto_regions = [
                region for region in no_veto_regions if region in available_regions
            ]
            if not particle_no_veto_regions:
                continue

            item = {
                "group_name": particle,
                "group_label": particle_labels.get(particle, particle),
                "particle_group": particle,
                "regions": particle_no_veto_regions,
            }
            if is_eventbuilder_particle_group(particle):
                no_veto_eventbuilder_items.append(item)
            else:
                no_veto_normal_items.append(item)

        if no_veto_normal_items or no_veto_data_items:
            pages.append(
                {
                    "comparison_mode": "overall_particle_overlay_no_veto_no_eventbuilder",
                    "page_key": "overall_no_veto::no_eventbuilder",
                    "page_label": "Overall no-veto particle overlay: no EventBuilder",
                    "items": no_veto_data_items + no_veto_normal_items,
                }
            )

        if cfg.get("split_eventbuilder", True):
            if no_veto_eventbuilder_items:
                pages.append(
                    {
                        "comparison_mode": "overall_particle_overlay_no_veto_eventbuilder",
                        "page_key": "overall_no_veto::eventbuilder",
                        "page_label": "Overall no-veto particle overlay: EventBuilder",
                        "items": no_veto_data_items + no_veto_eventbuilder_items,
                    }
                )
        elif no_veto_eventbuilder_items:
            pages.append(
                {
                    "comparison_mode": "overall_particle_overlay_no_veto_all",
                    "page_key": "overall_no_veto::all",
                    "page_label": "Overall no-veto particle overlay",
                    "items": no_veto_data_items
                    + no_veto_normal_items
                    + no_veto_eventbuilder_items,
                }
            )

    return pages


def build_curves_for_overall_overlay(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    hist_cfg: dict,
    eval_cfg: dict,
    items: list[dict],
    feature: str | None = None,
) -> tuple[list[dict], float]:
    repo_root = Path(args.repo_root).resolve()
    hist_dir = output_path(eval_cfg["outputs"]["hist_dir"], repo_root)
    reference_lumi_value = reference_lumi(metadata, eval_cfg)
    hist_version = hist_cfg["version"]
    eval_version = eval_cfg["version"]
    feature = feature or args.feature
    curves = []

    for index, item in enumerate(items):
        combined_hist = None
        raw_integral = 0.0
        selected_entries = 0
        n_partitions = 0
        combined_lumi = 0.0
        source_regions = []
        source_hist_paths = []
        source_partition_ids = []
        selections = []

        for region in item["regions"]:
            hist_path = grouped_hist_path(
                hist_dir=hist_dir,
                partition_version=args.partition_version,
                region_version=args.region_version,
                hist_version=hist_version,
                eval_version=eval_version,
                feature=feature,
                base_cut_key=args.base_cut_key,
                extra_cut_key=args.extra_cut_key,
                region=region,
                particle_group=item["particle_group"],
            )
            loaded = load_group_hist(hist_path)
            if loaded is None:
                continue

            lumi = float(loaded["lumi"])
            if lumi <= 0:
                raise RuntimeError(f"Histogram has non-positive lumi: {hist_path}")

            scaled_hist = loaded["hist"].Clone(
                f"h_overall_source_{safe_name(item['group_name'])}_{safe_name(region)}_{index}"
            )
            scaled_hist.SetDirectory(0)
            scaled_hist.Scale(reference_lumi_value / lumi)

            if combined_hist is None:
                combined_hist = scaled_hist.Clone(
                    f"h_overall_{safe_name(item['group_name'])}_{index}"
                )
                combined_hist.SetDirectory(0)
            else:
                combined_hist.Add(scaled_hist)

            raw_integral += float(loaded["raw_integral"])
            selected_entries += int(loaded["selected_entries"])
            n_partitions += int(loaded["n_partitions"])
            combined_lumi += lumi
            source_regions.append(region)
            source_hist_paths.append(hist_path)
            source_partition_ids.extend(
                [
                    value
                    for value in str(loaded.get("source_partition_ids", "")).split(",")
                    if value
                ]
            )
            selection = str(loaded.get("selection", ""))
            if selection and selection not in selections:
                selections.append(selection)

        if combined_hist is None:
            continue

        scale_factor = "per_source_lumi"
        if plot_normalization(hist_cfg["features"][feature]) == "unit_area":
            integral = float(combined_hist.Integral())
            unit_area_scale = 1.0 / integral if integral > 0.0 else 1.0
            combined_hist.Scale(unit_area_scale)
            scale_factor = unit_area_scale

        curves.append(
            {
                "hist_scaled": combined_hist,
                "hist_path": "\n".join(source_hist_paths),
                "lumi": combined_lumi,
                "reference_lumi": reference_lumi_value,
                "scale_factor": scale_factor,
                "raw_integral": raw_integral,
                "scaled_integral": float(combined_hist.Integral()),
                "selected_entries": selected_entries,
                "n_partitions": n_partitions,
                "source_partition_ids": ",".join(source_partition_ids),
                "selection": " || ".join(selections),
                "base_cut_key": args.base_cut_key,
                "extra_cut_key": args.extra_cut_key,
                "group_name": item["group_name"],
                "group_label": item["group_label"],
                "region": "combined",
                "particle_group": item["particle_group"],
                "source_regions": ",".join(source_regions),
                "source_hist_paths": "\n".join(source_hist_paths),
            }
        )

    if not curves:
        raise RuntimeError("No histograms were loaded for overall particle overlay")

    return curves, reference_lumi_value


def feature_pair_configs(eval_cfg: dict, feature: str) -> list[dict]:
    matches = []
    for study_name, study_cfg in eval_cfg.get("feature_pair_comparisons", {}).items():
        if not study_cfg.get("enabled", True):
            continue
        for pair_cfg in study_cfg.get("pairs", []):
            left = str(pair_cfg["left"])
            right = str(pair_cfg["right"])
            trigger = str(pair_cfg.get("trigger", right))
            if feature != trigger:
                continue
            matches.append(
                {
                    "study": str(study_name),
                    "left": left,
                    "right": right,
                    "left_label": str(pair_cfg.get("left_label", left)),
                    "right_label": str(pair_cfg.get("right_label", right)),
                }
            )
    return matches


def feature_pair_pages(metadata: pd.DataFrame, eval_cfg: dict, feature: str) -> list[dict]:
    allowed_regions = configured_regions(eval_cfg)
    allowed_particles = configured_particle_groups(eval_cfg)
    particle_labels = plot_config(eval_cfg).get("particle_labels", {})
    region_labels = plot_config(eval_cfg).get("region_labels", {})
    pairs = feature_pair_configs(eval_cfg, feature)
    if not pairs:
        return []

    rows = metadata.copy()
    if allowed_regions:
        rows = rows.loc[rows["region"].astype(str).isin(allowed_regions)]
    if allowed_particles:
        rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]

    pages = []
    group_keys = rows[["region", "particle_group"]].drop_duplicates()
    for _, row in group_keys.sort_values(["region", "particle_group"]).iterrows():
        region = str(row["region"])
        particle_group = str(row["particle_group"])
        region_label = region_labels.get(region, region)
        particle_label = particle_labels.get(particle_group, particle_group)
        for pair in pairs:
            page_key = f"{region}::{particle_group}"
            pages.append(
                {
                    "comparison_mode": "feature_pair",
                    "page_key": page_key,
                    "page_label": (
                        f"{pair['right_label']} vs {pair['left_label']} "
                        f"for {particle_label} in {region_label}"
                    ),
                    "pair": pair,
                    "region": region,
                    "particle_group": particle_group,
                }
            )
    return pages


def overall_feature_pair_pages(metadata: pd.DataFrame, eval_cfg: dict, feature: str) -> list[dict]:
    pairs = feature_pair_configs(eval_cfg, feature)
    if not pairs:
        return []

    pages = []
    for overlay_page in overall_particle_overlay_pages(metadata, eval_cfg):
        for item in overlay_page["items"]:
            for pair in pairs:
                page_key = f"{overlay_page['page_key']}::{item['group_name']}"
                pages.append(
                    {
                        "comparison_mode": "overall_feature_pair",
                        "page_key": page_key,
                        "page_label": (
                            f"{overlay_page['page_label']}: "
                            f"{pair['right_label']} vs {pair['left_label']} "
                            f"for {item['group_label']}"
                        ),
                        "pair": pair,
                        "item": item,
                    }
                )
    return pages


def build_curves_for_feature_pair(
    args: argparse.Namespace,
    hist_cfg: dict,
    eval_cfg: dict,
    page: dict,
) -> tuple[list[dict], float]:
    repo_root = Path(args.repo_root).resolve()
    hist_dir = output_path(eval_cfg["outputs"]["hist_dir"], repo_root)
    metadata = pd.read_csv(resolve_path(args.metadata_csv, repo_root))
    reference_lumi_value = reference_lumi(metadata, eval_cfg)
    hist_version = hist_cfg["version"]
    eval_version = eval_cfg["version"]
    pair = page["pair"]
    curves = []

    for index, item in enumerate(
        [
            {
                "feature": pair["left"],
                "group_name": "average",
                "group_label": pair["left_label"],
            },
            {
                "feature": pair["right"],
                "group_name": "qdc_weighted",
                "group_label": pair["right_label"],
            },
        ]
    ):
        hist_path = grouped_hist_path(
            hist_dir=hist_dir,
            partition_version=args.partition_version,
            region_version=args.region_version,
            hist_version=hist_version,
            eval_version=eval_version,
            feature=item["feature"],
            base_cut_key=args.base_cut_key,
            extra_cut_key=args.extra_cut_key,
            region=page["region"],
            particle_group=page["particle_group"],
        )
        loaded = load_group_hist(hist_path)
        if loaded is None:
            continue

        lumi = float(loaded["lumi"])
        if lumi <= 0:
            raise RuntimeError(f"Histogram has non-positive lumi: {hist_path}")

        scale_factor = reference_lumi_value / lumi
        hist_scaled = loaded["hist"].Clone(
            f"h_scaled_{safe_name(item['group_name'])}_{index}"
        )
        hist_scaled.SetDirectory(0)
        hist_scaled.Scale(scale_factor)

        curves.append(
            {
                **loaded,
                "hist_scaled": hist_scaled,
                "scale_factor": scale_factor,
                "reference_lumi": reference_lumi_value,
                "raw_integral": float(loaded["raw_integral"]),
                "scaled_integral": float(hist_scaled.Integral()),
                "base_cut_key": args.base_cut_key,
                "extra_cut_key": args.extra_cut_key,
                "page_key": page["page_key"],
                "page_label": page["page_label"],
                "group_name": item["group_name"],
                "group_label": item["group_label"],
                "region": page["region"],
                "particle_group": page["particle_group"],
                "feature_for_curve": item["feature"],
            }
        )

    return curves, reference_lumi_value


def build_curves_for_overall_feature_pair(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    hist_cfg: dict,
    eval_cfg: dict,
    page: dict,
) -> tuple[list[dict], float]:
    pair = page["pair"]
    item = page["item"]
    curves = []
    reference_lumi_value = reference_lumi(metadata, eval_cfg)

    for feature, group_name, group_label in [
        (pair["left"], "average", pair["left_label"]),
        (pair["right"], "qdc_weighted", pair["right_label"]),
    ]:
        feature_curves, reference_lumi_value = build_curves_for_overall_overlay(
            args=args,
            metadata=metadata,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            items=[item],
            feature=feature,
        )
        if not feature_curves:
            continue

        curve = feature_curves[0]
        curve.update(
            {
                "page_key": page["page_key"],
                "page_label": page["page_label"],
                "group_name": group_name,
                "group_label": group_label,
                "feature_for_curve": feature,
                "particle_group": item["particle_group"],
            }
        )
        curves.append(curve)

    return curves, reference_lumi_value


def make_plot(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    hist_config = resolve_path(args.hist_config, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    metadata = pd.read_csv(metadata_csv)
    hist_cfg = load_yaml(hist_config)
    eval_cfg = load_yaml(eval_config)
    feature_cfg = get_feature_config(hist_cfg, args.feature)

    plot_dir = output_path(eval_cfg["outputs"]["plot_dir"], repo_root)
    outdir = plot_output_dir(
        plot_dir=plot_dir,
        partition_version=args.partition_version,
        region_version=args.region_version,
        hist_version=hist_cfg["version"],
        eval_version=eval_cfg["version"],
        feature=args.feature,
        base_cut_key=args.base_cut_key,
        extra_cut_key=args.extra_cut_key,
    )

    pdf_path = outdir / "plot.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = ROOT.TCanvas("c_region_eval_comparison", "", 900, 700)
    canvas.Print(f"{pdf_path}[")
    summary_rows = []

    for page in particle_comparison_pages(metadata, eval_cfg):
        comparison_mode = page["comparison_mode"]
        curves, reference_lumi_value = build_curves_for_items(
            args=args,
            metadata=metadata,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            items=page["items"],
            comparison_mode=comparison_mode,
        )
        title = clean_title(eval_cfg, comparison_mode, page["page_key"])
        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=page["page_key"],
            title=title,
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": page["page_key"],
                    "page_label": page["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": args.feature,
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                }
            )

    comparison_mode = "regions_for_particle"
    args.comparison_mode = comparison_mode
    for key in page_keys(metadata, eval_cfg, comparison_mode):
        curves, reference_lumi_value = build_curves_for_page(args, metadata, hist_cfg, eval_cfg, key)
        title = clean_title(eval_cfg, comparison_mode, key)
        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=key,
            title=title,
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": key,
                    "page_label": curve["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": args.feature,
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                }
            )

    for page in overall_particle_overlay_pages(metadata, eval_cfg):
        comparison_mode = page["comparison_mode"]
        curves, reference_lumi_value = build_curves_for_overall_overlay(
            args=args,
            metadata=metadata,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            items=page["items"],
        )
        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=page["page_key"],
            title=page["page_label"],
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": page["page_key"],
                    "page_label": page["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": args.feature,
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                    "source_regions": curve["source_regions"],
                    "source_hist_paths": curve["source_hist_paths"],
                }
            )

    comparison_mode = "eventbuilder_veto_timing"
    for page in eventbuilder_veto_timing_pages(eval_cfg, args.feature):
        curves, reference_lumi_value = build_curves_for_items(
            args=args,
            metadata=metadata,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            items=page["items"],
            comparison_mode=comparison_mode,
        )
        if len(curves) < 2:
            print(f"[skip] incomplete EventBuilder comparison page: {page['page_key']}")
            continue

        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=page["page_key"],
            title=page["page_label"],
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": page["page_key"],
                    "page_label": page["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": args.feature,
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                }
            )

    comparison_mode = "feature_pair"
    for page in feature_pair_pages(metadata, eval_cfg, args.feature):
        curves, reference_lumi_value = build_curves_for_feature_pair(
            args=args,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            page=page,
        )
        if len(curves) < 2:
            print(f"[skip] incomplete feature-pair comparison page: {page['page_key']}")
            continue

        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=page["page_key"],
            title=page["page_label"],
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": page["page_key"],
                    "page_label": page["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": curve["feature_for_curve"],
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                }
            )

    comparison_mode = "overall_feature_pair"
    for page in overall_feature_pair_pages(metadata, eval_cfg, args.feature):
        curves, reference_lumi_value = build_curves_for_overall_feature_pair(
            args=args,
            metadata=metadata,
            hist_cfg=hist_cfg,
            eval_cfg=eval_cfg,
            page=page,
        )
        if len(curves) < 2:
            print(f"[skip] incomplete overall feature-pair comparison page: {page['page_key']}")
            continue

        draw_overlay(
            canvas=canvas,
            output_pdf=pdf_path,
            feature_name=args.feature,
            feature_cfg=feature_cfg,
            eval_cfg=eval_cfg,
            comparison_mode=comparison_mode,
            page_key=page["page_key"],
            title=page["page_label"],
            reference_lumi_value=reference_lumi_value,
            curves=curves,
        )

        for curve in curves:
            summary_rows.append(
                {
                    "comparison_mode": comparison_mode,
                    "page_key": page["page_key"],
                    "page_label": page["page_label"],
                    "feature": args.feature,
                    "feature_for_curve": curve["feature_for_curve"],
                    "base_cut_key": args.base_cut_key,
                    "extra_cut_key": args.extra_cut_key,
                    "group_name": curve["group_name"],
                    "group_label": curve["group_label"],
                    "region": curve["region"],
                    "particle_group": curve["particle_group"],
                    "hist_path": curve["hist_path"],
                    "lumi": curve["lumi"],
                    "reference_lumi": curve["reference_lumi"],
                    "scale_factor": curve["scale_factor"],
                    "raw_integral": curve["raw_integral"],
                    "scaled_integral": curve["scaled_integral"],
                    "selected_entries": curve["selected_entries"],
                    "n_partitions": curve["n_partitions"],
                    "selection": curve["selection"],
                    "source_partition_ids": curve["source_partition_ids"],
                    "source_regions": curve["source_regions"],
                    "source_hist_paths": curve["source_hist_paths"],
                }
            )

    canvas.Print(f"{pdf_path}]")
    print(f"Wrote plot: {pdf_path}")
    write_summary(outdir / "summary.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grouped region-partition histogram comparisons.")
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--hist-config", required=True, help="Histogram feature config YAML")
    parser.add_argument("--eval-config", required=True, help="Evaluation options config YAML")
    parser.add_argument("--feature", required=True, help="Feature key from hist config")
    parser.add_argument("--base-cut-key", required=True, help="Base cut option key")
    parser.add_argument("--extra-cut-key", required=True, help="Extra cut option key")
    parser.add_argument("--partition-version", required=True, help="Partition version label")
    parser.add_argument("--region-version", required=True, help="Region version label")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative paths")
    make_plot(parser.parse_args())


if __name__ == "__main__":
    main()
