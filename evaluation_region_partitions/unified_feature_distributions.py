#!/usr/bin/env python3
"""Build and plot validated unified shower-feature distributions."""

from __future__ import annotations

import argparse
from array import array
import csv
import json
import math
import os
from pathlib import Path

import pandas as pd
import yaml


def expand_group_series(config: dict) -> None:
    """Expand compact energy-series definitions into ordinary plot groups."""
    groups = config.setdefault("groups", {})
    templates = config.get("group_templates", {})
    for series_name, series in config.get("group_series", {}).items():
        template_name = str(series["template"])
        if template_name not in templates:
            raise KeyError(f"unknown group template {template_name}")
        template = dict(templates[template_name])
        for energy in series["energies"]:
            name = f"{series_name}_{int(energy)}"
            if name in groups:
                raise ValueError(f"duplicate expanded group {name}")
            particle = str(template["particle_charge"])
            label_template = str(template.get("label_template", "{particle} {energy} GeV"))
            groups[name] = {
                "family": str(template["family"]),
                "geometry_setup_id": str(template["geometry_setup_id"]),
                "source_filters": {
                    "source_pdg_code": int(template["source_pdg_code"]),
                    "source_energy_gev": int(energy),
                },
                "expression": "",
                "max_events": int(template["max_events"]),
                "sampling_policy": "deterministic_chain_prefix_v1",
                "particle_charge": particle,
                "beam_energy_gev": int(energy),
                "label": label_template.format(
                    particle=particle, energy=int(energy)
                ),
            }


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expand_group_series(config)
    core_features = {
        "density_sndsw_scifi",
        "scifi_cluster_qdc_per_hit",
        "scifi_qdc_per_hit_direct",
    }
    supported_features = core_features | {
        "scifi_vertical_cluster1_qdc_std",
        "scifi_horizontal_cluster1_qdc_std",
    }
    configured_features = set(config.get("features", {}))
    if not core_features.issubset(configured_features):
        raise ValueError("distribution config is missing a core frozen feature")
    unsupported = configured_features - supported_features
    if unsupported:
        raise ValueError(f"distribution config has unsupported features: {sorted(unsupported)}")
    for plot_name, plot in config.get("plot_sets", {}).items():
        missing_groups = set(plot["groups"]) - set(config.get("groups", {}))
        if missing_groups:
            raise ValueError(
                f"plot set {plot_name} has unknown groups: {sorted(missing_groups)}"
            )
        geometries = {
            config["groups"][group].get("geometry_setup_id", "")
            for group in plot["groups"]
        }
        if "" in geometries:
            raise ValueError(f"plot set {plot_name} has missing geometry metadata")
        geometry_policy = str(plot.get("geometry_policy", "same_geometry"))
        if geometry_policy not in {"same_geometry", "cross_geometry_diagnostic"}:
            raise ValueError(
                f"plot set {plot_name} has unknown geometry policy {geometry_policy}"
            )
        if geometry_policy == "same_geometry" and len(geometries) != 1:
            raise ValueError(f"plot set {plot_name} mixes geometries: {sorted(geometries)}")
        if geometry_policy == "cross_geometry_diagnostic" and len(geometries) < 2:
            raise ValueError(
                f"cross-geometry plot set {plot_name} does not mix geometries"
            )
        style_policy = str(plot.get("style_policy", "default"))
        if style_policy not in {"default", "paired_particle_energy"}:
            raise ValueError(
                f"plot set {plot_name} has unknown style policy {style_policy}"
            )
        legend_columns = int(plot.get("legend_columns", 1))
        if legend_columns < 1:
            raise ValueError(f"plot set {plot_name} has invalid legend_columns")
        legend_label_mode = str(plot.get("legend_label_mode", "configured"))
        if legend_label_mode not in {
            "configured", "compact_particle_energy",
            "compact_particle_energy_geometry",
        }:
            raise ValueError(f"plot set {plot_name} has invalid legend_label_mode")
        draw_style = str(plot.get("draw_style", "points_errors"))
        if draw_style not in {"points_errors", "histogram_lines"}:
            raise ValueError(f"plot set {plot_name} has invalid draw_style")
        unit_area_mode = str(plot.get("unit_area_mode", "probability_density"))
        if unit_area_mode not in {"probability_density", "bin_fraction"}:
            raise ValueError(f"plot set {plot_name} has invalid unit_area_mode")
        annotation_lines = plot.get("annotation_lines", [])
        if not isinstance(annotation_lines, list) or any(
            not isinstance(line, str) or not line.strip()
            for line in annotation_lines
        ):
            raise ValueError(f"plot set {plot_name} has invalid annotation_lines")
        if style_policy == "paired_particle_energy":
            groups = list(plot["groups"])
            if len(groups) % 2:
                raise ValueError(
                    f"paired plot set {plot_name} must contain complete pairs"
                )
            for index in range(0, len(groups), 2):
                first = str(config["groups"][groups[index]]["particle_charge"])
                second = str(config["groups"][groups[index + 1]]["particle_charge"])
                particles = {first, second}
                if not (
                    any(value.startswith("pi") for value in particles)
                    and any(value.startswith("e") for value in particles)
                ):
                    raise ValueError(
                        f"paired plot set {plot_name} pair {index // 2} is not pion/electron"
                    )
    return config


def histogram_model(feature: str, feature_config: dict) -> tuple[int, float, float]:
    low = float(feature_config["x_min"])
    high = float(feature_config["x_max"])
    width = float(feature_config["bin_width"])
    if not high > low or width <= 0:
        raise ValueError(f"invalid binning for {feature}")
    bins_float = (high - low) / width
    bins = int(round(bins_float))
    if bins <= 0 or not math.isclose(bins_float, bins, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"bin width does not divide the range for {feature}")
    return bins, low, high


def normalization_scale(
    mode: str,
    integral: float,
    bin_width: float,
    exposure: float,
    exposure_kind: str,
    normalization_config: dict,
) -> float:
    if mode == "unit_area":
        if integral < 0:
            raise ValueError("unit-area integral cannot be negative")
        return 0.0 if integral == 0 else 1.0 / (integral * bin_width)
    if mode == "common_exposure_19fb":
        allowed = set(normalization_config[mode]["allowed_exposure_kinds"])
        if exposure_kind not in allowed:
            raise ValueError(
                f"normalization {mode} is not allowed for exposure kind {exposure_kind}"
            )
        if not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"normalization {mode} requires positive exposure")
        return float(normalization_config[mode]["target_exposure_fb"]) / exposure
    if mode == "raw_counts":
        return 1.0
    raise ValueError(f"unknown normalization mode {mode}")


def filter_source_rows(frame: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply exact source-level filters before touching validation records."""
    filtered = frame.copy()
    for column, expected in (filters or {}).items():
        if column not in filtered.columns:
            raise KeyError(f"manifest has no source-filter column {column}")
        values = expected if isinstance(expected, list) else [expected]
        if all(isinstance(value, (int, float)) for value in values):
            observed = pd.to_numeric(filtered[column], errors="coerce")
            mask = observed.map(
                lambda item: any(
                    math.isclose(float(item), float(value), rel_tol=0.0, abs_tol=1e-9)
                    for value in values
                ) if pd.notna(item) else False
            )
        else:
            mask = filtered[column].astype(str).isin([str(value) for value in values])
        filtered = filtered.loc[mask].copy()
    return filtered.reset_index(drop=True)


def load_validated_rows(
    manifest_path: str | Path,
    family: str,
    source_filters: dict | None = None,
    allow_missing_validated_sources: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "family",
        "source_uid",
        "feature_scifi_cluster_v1_path",
        "validation_record",
        "lumi_per_file",
        "exposure_kind",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["family"].astype(str).eq(family)].copy()
    frame = filter_source_rows(frame, source_filters)
    if frame.empty:
        raise ValueError(
            f"manifest contains no rows for family {family} and filters "
            f"{source_filters or {}}"
        )
    if frame["source_uid"].duplicated().any():
        if "feature_unit_uid" not in frame.columns:
            raise ValueError(
                f"manifest contains duplicate source UIDs without feature_unit_uid "
                f"for {family}"
            )
        if frame["feature_unit_uid"].astype(str).duplicated().any():
            raise ValueError(f"manifest contains duplicate feature unit UIDs for {family}")

    available_indices = []
    omitted_missing_source_count = 0
    for index, row in frame.iterrows():
        feature = Path(str(row["feature_scifi_cluster_v1_path"]))
        marker = Path(str(row["validation_record"]))
        if allow_missing_validated_sources and (
            not feature.is_file() or not marker.is_file()
        ):
            omitted_missing_source_count += 1
            continue
        if not marker.is_file():
            raise RuntimeError(f"missing validation record: {marker}")
        with marker.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("validation_passed") is not True:
            raise RuntimeError(f"failed validation record: {marker}")
        if str(record.get("source_uid", "")) != str(row["source_uid"]):
            raise RuntimeError(f"source UID mismatch in {marker}")
        if str(record.get("feature_path", "")) != str(
            row["feature_scifi_cluster_v1_path"]
        ):
            raise RuntimeError(f"feature path mismatch in {marker}")
        available_indices.append(index)
    frame = frame.loc[available_indices].copy()
    if frame.empty:
        raise RuntimeError(
            f"no available validated feature files remain for {family}"
        )
    frame.attrs["omitted_missing_source_count"] = omitted_missing_source_count
    sort_columns = ["source_uid"]
    if "selected_entry_start" in frame.columns:
        sort_columns.append("selected_entry_start")
    elif "feature_unit_uid" in frame.columns:
        sort_columns.append("feature_unit_uid")
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def audit_family(args: argparse.Namespace) -> None:
    rows = load_validated_rows(args.manifest, args.family)
    missing_features = [
        path
        for path in rows["feature_scifi_cluster_v1_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_features:
        preview = ", ".join(missing_features[:3])
        raise RuntimeError(
            f"{len(missing_features)} validated feature files are unreadable; first: {preview}"
        )
    exposure_kinds = sorted(set(rows["exposure_kind"].astype(str)))
    if len(exposure_kinds) != 1:
        raise ValueError(f"mixed exposure kinds for {args.family}: {exposure_kinds}")
    record = {
        "audit_passed": True,
        "distribution_version": load_config(args.config)["version"],
        "family": args.family,
        "source_count": int(rows["source_uid"].nunique()),
        "validated_feature_count": int(len(rows)),
        "exposure_kind": exposure_kinds[0],
        "manifest_path": str(args.manifest),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"audited {len(rows)} validated {args.family} feature files")


def combine_selection(*expressions: str) -> str:
    terms = [str(value).strip() for value in expressions if str(value).strip()]
    return " && ".join(f"({term})" for term in terms)


def _write_named(root, name: str, value: object) -> None:
    import ROOT

    root.cd()
    if isinstance(value, bool):
        ROOT.TParameter("int")(name, int(value)).Write()
    elif isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, int(value)).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, float(value)).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def _numeric_exposure(rows: pd.DataFrame) -> float:
    sources = rows.drop_duplicates("source_uid", keep="first")
    values = pd.to_numeric(sources["lumi_per_file"], errors="coerce")
    if values.isna().all():
        return 0.0
    if values.isna().any() or (values < 0).any():
        raise ValueError("family has incomplete or invalid exposure values")
    return float(values.sum())


def make_histogram(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()
    ROOT.TH1.AddDirectory(False)

    config = load_config(args.config)
    if args.feature not in config["features"]:
        raise KeyError(f"unknown feature {args.feature}")
    if args.group not in config["groups"]:
        raise KeyError(f"unknown group {args.group}")
    group = config["groups"][args.group]
    family = str(group["family"])
    rows = load_validated_rows(
        args.manifest,
        family,
        group.get("source_filters", {}),
        allow_missing_validated_sources=bool(
            getattr(args, "allow_missing_validated_sources", False)
        ),
    )
    omitted_missing_source_count = int(
        rows.attrs.get("omitted_missing_source_count", 0)
    )
    source_subfolders = [str(value) for value in group.get("source_subfolders", [])]
    if source_subfolders:
        if "subfolder" not in rows:
            raise KeyError(f"manifest has no subfolder column required by {args.group}")
        rows = rows.loc[rows["subfolder"].astype(str).isin(source_subfolders)].copy()
        if rows.empty:
            raise ValueError(
                f"no {family} sources match subfolders {source_subfolders} for {args.group}"
            )
        rows = rows.reset_index(drop=True)
    exposure_kinds = set(rows["exposure_kind"].astype(str))
    if len(exposure_kinds) != 1:
        raise ValueError(f"mixed exposure kinds for {family}: {sorted(exposure_kinds)}")
    exposure_kind = next(iter(exposure_kinds))
    exposure = _numeric_exposure(rows)

    expected_geometry = str(group.get("geometry_setup_id", ""))
    geometry_ids = (
        sorted(set(rows["source_geometry_id"].astype(str)))
        if "source_geometry_id" in rows else [""]
    )
    geometry_checksums = (
        sorted(set(rows["source_geometry_checksum"].astype(str)))
        if "source_geometry_checksum" in rows else [""]
    )
    if expected_geometry:
        if len(geometry_ids) != 1 or len(geometry_checksums) != 1:
            raise ValueError(
                f"group {args.group} mixes geometry IDs/checksums: "
                f"{geometry_ids}, {geometry_checksums}"
            )
        if geometry_ids != [expected_geometry]:
            raise ValueError(
                f"group {args.group} geometry {geometry_ids} != {expected_geometry}"
            )
    chain = ROOT.TChain(str(config.get("tree_name", "sndData")))
    for path in rows["feature_scifi_cluster_v1_path"].astype(str):
        if chain.Add(path) != 1:
            raise RuntimeError(f"could not add validated feature file {path}")
    feature_config = config["features"][args.feature]
    selection = combine_selection(
        config["selection"]["expression"],
        group.get("expression", ""),
        feature_config.get("feature_cut", ""),
    )
    dataframe = ROOT.RDataFrame(chain)
    if selection:
        dataframe = dataframe.Filter(selection, "unified_distribution_selection")
    available_entries = int(dataframe.Count().GetValue())
    max_events = int(group.get("max_events", 0) or 0)
    if max_events < 0:
        raise ValueError(f"group {args.group} has negative max_events")
    if max_events:
        dataframe = dataframe.Range(min(max_events, available_entries))
    bins, low, high = histogram_model(args.feature, feature_config)
    selected = dataframe.Count()
    result = dataframe.Histo1D(
        (f"h_{args.feature}_{args.group}", "", bins, low, high),
        str(feature_config.get("expression", args.feature)),
    )
    selected_entries = int(selected.GetValue())
    histogram = result.GetValue().Clone("hist")
    histogram.SetDirectory(0)
    underflow = float(histogram.GetBinContent(0))
    in_range = float(histogram.Integral(1, histogram.GetNbinsX()))
    overflow = float(histogram.GetBinContent(histogram.GetNbinsX() + 1))
    integral_all = underflow + in_range + overflow
    if not math.isclose(integral_all, selected_entries, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(
            f"histogram flow closure failed: {integral_all} != {selected_entries}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not create {output}")
    histogram.Write("hist")
    metadata = {
        "distribution_version": config["version"],
        "family": family,
        "particle_group": args.group,
        "feature": args.feature,
        "feature_expression": feature_config.get("expression", args.feature),
        "selection_name": config["selection"]["name"],
        "selection_base_expression": config["selection"]["expression"],
        "selection_expression": selection,
        "source_count": int(rows["source_uid"].nunique()),
        "omitted_missing_source_count": omitted_missing_source_count,
        "available_selected_entries": available_entries,
        "selected_entries": selected_entries,
        "sampling_policy": str(group.get("sampling_policy", "all_selected_events")),
        "event_cap": max_events,
        "source_geometry_id": geometry_ids[0],
        "source_geometry_checksum": geometry_checksums[0],
        "particle_charge": str(group.get("particle_charge", "")),
        "beam_energy_gev": str(group.get("beam_energy_gev", "")),
        "underflow_raw": underflow,
        "in_range_raw": in_range,
        "overflow_raw": overflow,
        "integral_all_raw": integral_all,
        "sample_exposure": exposure,
        "exposure_kind": exposure_kind,
        "bin_width": float(feature_config["bin_width"]),
    }
    for name, value in metadata.items():
        _write_named(root_file, name, value)
    root_file.Close()
    print(
        f"wrote raw histogram {output}: entries={selected_entries}, "
        f"sources={rows['source_uid'].nunique()}, files={len(rows)}"
    )


def _read_value(root_file, name: str, default=None):
    obj = root_file.Get(name)
    if not obj:
        return default
    if hasattr(obj, "GetVal"):
        return obj.GetVal()
    if hasattr(obj, "GetTitle"):
        return str(obj.GetTitle())
    return default


def _load_histogram(path: str, expected_feature: str) -> dict:
    import ROOT

    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise RuntimeError(f"could not open histogram {path}")
    histogram = source.Get("hist")
    if histogram is None:
        raise RuntimeError(f"missing histogram in {path}")
    histogram = histogram.Clone(f"hist_{Path(path).stem}")
    histogram.SetDirectory(0)
    values = {
        "hist": histogram,
        "feature": _read_value(source, "feature", ""),
        "group": _read_value(source, "particle_group", ""),
        "base_selection": _read_value(source, "selection_base_expression", ""),
        "selection": _read_value(source, "selection_expression", ""),
        "selected_entries": int(_read_value(source, "selected_entries", 0)),
        "available_selected_entries": int(
            _read_value(source, "available_selected_entries", 0)
        ),
        "source_count": int(_read_value(source, "source_count", 0)),
        "omitted_missing_source_count": int(_read_value(source, "omitted_missing_source_count", 0)),
        "sampling_policy": _read_value(source, "sampling_policy", ""),
        "event_cap": int(_read_value(source, "event_cap", 0)),
        "source_geometry_id": _read_value(source, "source_geometry_id", ""),
        "source_geometry_checksum": _read_value(
            source, "source_geometry_checksum", ""
        ),
        "particle_charge": _read_value(source, "particle_charge", ""),
        "beam_energy_gev": _read_value(source, "beam_energy_gev", ""),
        "underflow_raw": float(_read_value(source, "underflow_raw", 0.0)),
        "in_range_raw": float(_read_value(source, "in_range_raw", 0.0)),
        "overflow_raw": float(_read_value(source, "overflow_raw", 0.0)),
        "integral_all_raw": float(_read_value(source, "integral_all_raw", 0.0)),
        "exposure": float(_read_value(source, "sample_exposure", 0.0)),
        "exposure_kind": _read_value(source, "exposure_kind", ""),
    }
    source.Close()
    if values["feature"] != expected_feature:
        raise RuntimeError(f"feature mismatch in {path}")
    return values


def _fold_flows(histogram, feature_config: dict) -> None:
    if feature_config.get("fold_underflow_for_plot", False):
        content = histogram.GetBinContent(1) + histogram.GetBinContent(0)
        error = math.hypot(histogram.GetBinError(1), histogram.GetBinError(0))
        histogram.SetBinContent(1, content)
        histogram.SetBinError(1, error)
        histogram.SetBinContent(0, 0.0)
        histogram.SetBinError(0, 0.0)
    if feature_config.get("fold_overflow_for_plot", False):
        last = histogram.GetNbinsX()
        content = histogram.GetBinContent(last) + histogram.GetBinContent(last + 1)
        error = math.hypot(histogram.GetBinError(last), histogram.GetBinError(last + 1))
        histogram.SetBinContent(last, content)
        histogram.SetBinError(last, error)
        histogram.SetBinContent(last + 1, 0.0)
        histogram.SetBinError(last + 1, 0.0)


def _legend_label(group: str, loaded: dict, config: dict, mode: str) -> str:
    if mode == "configured":
        return str(config["groups"][group]["label"])
    particle = str(loaded["particle_charge"])
    energy_value = str(loaded["beam_energy_gev"])
    try:
        numeric_energy = float(energy_value)
        energy = f"{numeric_energy:g}"
    except ValueError:
        energy = energy_value
    label = f"{particle} {energy} GeV"
    if mode == "compact_particle_energy_geometry":
        geometry_aliases = {
            "Fe_3walls_H8_2023": "Fe 3-wall",
            "W_2walls_28plates": "W 2-wall",
        }
        geometry = geometry_aliases.get(
            str(loaded["source_geometry_id"]),
            str(loaded["source_geometry_id"]),
        )
        label += f" ({geometry})"
    return label


def make_plot(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetErrorX(0.0)
    config = load_config(args.config)
    feature_config = config["features"][args.feature]
    normalization_config = config["normalizations"]
    plot_set = str(getattr(args, "plot_set", "") or "")
    if plot_set and plot_set not in config.get("plot_sets", {}):
        raise KeyError(f"unknown plot set {plot_set}")
    plot_config = config.get("plot_sets", {}).get(plot_set, {})
    style_policy = str(plot_config.get("style_policy", "default"))
    legend_columns = int(plot_config.get("legend_columns", 1))
    legend_label_mode = str(plot_config.get("legend_label_mode", "configured"))
    draw_style = str(plot_config.get("draw_style", "points_errors"))
    unit_area_mode = str(plot_config.get("unit_area_mode", "probability_density"))
    annotation_lines = [str(line) for line in plot_config.get("annotation_lines", [])]
    requested = []
    for item in args.hist:
        if "=" not in item:
            raise ValueError(f"histogram argument must be GROUP=PATH: {item}")
        group, path = item.split("=", 1)
        if group not in config["groups"]:
            raise KeyError(f"unknown group {group}")
        loaded = _load_histogram(path, args.feature)
        if loaded["group"] != group:
            raise RuntimeError(f"group mismatch in {path}")
        requested.append((group, loaded))
    expected_base_selection = str(config["selection"]["expression"])
    for _, loaded in requested:
        if loaded["base_selection"] != expected_base_selection:
            raise RuntimeError(
                "cannot compare histogram with a different base selection: "
                f"{loaded['base_selection']!r} != {expected_base_selection!r}"
            )
    geometry_policy = str(getattr(args, "geometry_policy", "not_enforced"))
    geometry_pairs = {
        (str(loaded["source_geometry_id"]), str(loaded["source_geometry_checksum"]))
        for _, loaded in requested
    }
    if geometry_policy != "not_enforced" and any(not all(pair) for pair in geometry_pairs):
        raise RuntimeError("plot geometry policy requires complete geometry metadata")
    if geometry_policy == "same_geometry" and len(geometry_pairs) != 1:
        raise RuntimeError("same_geometry plot received mixed geometry inputs")
    if geometry_policy == "cross_geometry_diagnostic":
        if args.normalization != "unit_area":
            raise RuntimeError("cross-geometry plots require unit-area normalization")
        if len(geometry_pairs) < 2:
            raise RuntimeError("cross_geometry_diagnostic plot requires mixed geometries")

    colors = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kMagenta + 2, ROOT.kOrange + 7, ROOT.kCyan + 2]
    curves = []
    for index, (group, loaded) in enumerate(requested):
        histogram = loaded["hist"].Clone(f"plot_{group}")
        histogram.SetDirectory(0)
        _fold_flows(histogram, feature_config)
        integral = float(histogram.Integral(1, histogram.GetNbinsX()))
        if args.normalization == "unit_area" and unit_area_mode == "bin_fraction":
            scale = 0.0 if integral == 0.0 else 1.0 / integral
        else:
            scale = normalization_scale(
                args.normalization,
                integral,
                float(feature_config["bin_width"]),
                loaded["exposure"],
                loaded["exposure_kind"],
                normalization_config,
            )
        histogram.Scale(scale)
        if style_policy == "paired_particle_energy":
            color = colors[(index // 2) % len(colors)]
            particle = str(loaded["particle_charge"])
            line_style = 2 if particle.startswith("e") else 1
            marker_style = 24 if particle.startswith("e") else 20
        else:
            color = colors[index % len(colors)]
            line_style = 1
            marker_style = 20 + (index % 5)
        histogram.SetLineColor(color)
        histogram.SetLineStyle(line_style)
        histogram.SetLineWidth(2 if draw_style == "histogram_lines" else 1)
        histogram.SetMarkerColor(color)
        histogram.SetMarkerStyle(marker_style)
        histogram.SetMarkerSize(0.65)
        legend_label = _legend_label(group, loaded, config, legend_label_mode)
        curves.append(
            (group, loaded, histogram, scale, color, line_style,
             marker_style, legend_label)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    use_header_pad = bool(annotation_lines)
    canvas = ROOT.TCanvas("canvas", "", 900, 1050 if use_header_pad else 700)
    if use_header_pad:
        header_pad = ROOT.TPad("header_pad", "", 0.0, 0.68, 1.0, 1.0)
        plot_pad = ROOT.TPad("plot_pad", "", 0.0, 0.0, 1.0, 0.68)
        header_pad.SetBottomMargin(0.0)
        plot_pad.SetLeftMargin(0.13)
        plot_pad.SetRightMargin(0.04)
        plot_pad.SetBottomMargin(0.13)
        plot_pad.SetTopMargin(0.03)
        header_pad.Draw()
        plot_pad.Draw()
    else:
        header_pad = None
        plot_pad = canvas
    plot_pad.cd()
    if feature_config.get("log_y", False):
        plot_pad.SetLogy()

    if draw_style == "histogram_lines":
        maximum = max(histogram.GetMaximum() for _, _, histogram, _, _, _, _, _ in curves)
    else:
        maximum = max(
            histogram.GetBinContent(bin_index) + histogram.GetBinError(bin_index)
            for _, _, histogram, _, _, _, _, _ in curves
            for bin_index in range(1, histogram.GetNbinsX() + 1)
        )
    draw_option = "HIST" if draw_style == "histogram_lines" else "E1 P"
    legend_option = "l" if draw_style == "histogram_lines" else "lep"
    if use_header_pad:
        legend = ROOT.TLegend(0.06, 0.04, 0.96, 0.50)
    elif legend_columns > 1:
        legend = ROOT.TLegend(0.34, 0.50, 0.94, 0.88)
    else:
        legend = ROOT.TLegend(0.58, 0.62, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetNColumns(legend_columns)
    if use_header_pad:
        legend.SetTextSize(0.055)
    elif legend_columns > 1:
        legend.SetTextSize(0.020)
    for index, (_, _, histogram, _, _, _, _, legend_label) in enumerate(curves):
        histogram.SetTitle("")
        histogram.SetStats(False)
        histogram.GetXaxis().SetTitle(str(feature_config["axis_title"]))
        if args.normalization == "raw_counts":
            y_title = "Events"
        elif args.normalization == "unit_area" and unit_area_mode == "bin_fraction":
            y_title = "Fraction of events / bin"
        else:
            y_title = str(normalization_config[args.normalization]["y_axis_title"])
        histogram.GetYaxis().SetTitle(y_title)
        histogram.SetMaximum(maximum * (20.0 if plot_pad.GetLogy() else 1.35))
        if plot_pad.GetLogy():
            histogram.SetMinimum(1e-8)
        histogram.Draw(draw_option if index == 0 else f"{draw_option} SAME")
        legend.AddEntry(histogram, legend_label, legend_option)

    if header_pad:
        header_pad.cd()
        annotation = ROOT.TLatex()
        annotation.SetNDC(True)
        annotation.SetTextFont(42)
        annotation.SetTextSize(0.055)
        annotation_y = 0.92
        for line in annotation_lines:
            annotation.DrawLatex(0.06, annotation_y, line)
            annotation_y -= 0.13
        legend.Draw()
    else:
        plot_pad.cd()
        legend.Draw()
    canvas.cd()
    canvas.SaveAs(str(output))

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature", "group", "normalization", "exposure_kind", "sample_exposure",
        "target_exposure_fb", "scale_factor", "selected_entries", "underflow_raw",
        "in_range_raw", "overflow_raw", "integral_all_raw", "scaled_in_range_integral",
        "available_selected_entries", "source_count", "sampling_policy", "event_cap",
        "omitted_missing_source_count",
        "source_geometry_id", "source_geometry_checksum", "particle_charge",
        "beam_energy_gev", "geometry_policy", "plot_set", "style_policy",
        "draw_style", "unit_area_mode", "line_color", "line_style", "marker_style", "legend_label", "draw_option",
    ]
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (group, loaded, histogram, scale, color, line_style,
             marker_style, legend_label) in curves:
            writer.writerow(
                {
                    "feature": args.feature,
                    "group": group,
                    "normalization": args.normalization,
                    "exposure_kind": loaded["exposure_kind"],
                    "sample_exposure": f"{loaded['exposure']:.17g}",
                    "target_exposure_fb": normalization_config.get(args.normalization, {}).get(
                        "target_exposure_fb", ""
                    ),
                    "scale_factor": f"{scale:.17g}",
                    "selected_entries": loaded["selected_entries"],
                    "available_selected_entries": loaded["available_selected_entries"],
                    "source_count": loaded["source_count"],
                    "omitted_missing_source_count": loaded["omitted_missing_source_count"],
                    "sampling_policy": loaded["sampling_policy"],
                    "event_cap": loaded["event_cap"],
                    "source_geometry_id": loaded["source_geometry_id"],
                    "source_geometry_checksum": loaded["source_geometry_checksum"],
                    "particle_charge": loaded["particle_charge"],
                    "beam_energy_gev": loaded["beam_energy_gev"],
                    "geometry_policy": geometry_policy,
                    "plot_set": plot_set,
                    "style_policy": style_policy,
                    "draw_style": draw_style,
                    "unit_area_mode": unit_area_mode,
                    "line_color": color,
                    "line_style": line_style,
                    "marker_style": marker_style,
                    "legend_label": legend_label,
                    "draw_option": draw_option,
                    "underflow_raw": f"{loaded['underflow_raw']:.17g}",
                    "in_range_raw": f"{loaded['in_range_raw']:.17g}",
                    "overflow_raw": f"{loaded['overflow_raw']:.17g}",
                    "integral_all_raw": f"{loaded['integral_all_raw']:.17g}",
                    "scaled_in_range_integral": f"{histogram.Integral():.17g}",
                }
            )
    print(f"wrote {args.normalization} plot {output} and summary {summary}")


def _comparison_inputs(args: argparse.Namespace):
    config = load_config(args.config)
    if args.plot_set not in config.get("plot_sets", {}):
        raise KeyError(f"unknown plot set {args.plot_set}")
    plot_config = config["plot_sets"][args.plot_set]
    requested = []
    for item in args.hist:
        if "=" not in item:
            raise ValueError(f"histogram argument must be GROUP=PATH: {item}")
        group, path = item.split("=", 1)
        loaded = _load_histogram(path, args.feature)
        if loaded["group"] != group:
            raise RuntimeError(f"group mismatch in {path}")
        requested.append((group, loaded))
    if len(requested) % 2:
        raise ValueError("paired comparison requires an even number of histograms")
    expected_selection = str(config["selection"]["expression"])
    if any(item[1]["base_selection"] != expected_selection for item in requested):
        raise RuntimeError("cannot compare histograms with different base selections")
    geometries = {
        (str(item[1]["source_geometry_id"]), str(item[1]["source_geometry_checksum"]))
        for item in requested
    }
    if args.geometry_policy == "same_geometry" and len(geometries) != 1:
        raise RuntimeError("same_geometry plot received mixed geometry inputs")
    if args.geometry_policy == "cross_geometry_diagnostic" and len(geometries) < 2:
        raise RuntimeError("cross_geometry_diagnostic plot requires mixed geometries")
    return config, plot_config, requested


def _unit_area_histogram(loaded: dict, feature_config: dict, name: str):
    histogram = loaded["hist"].Clone(name)
    histogram.SetDirectory(0)
    _fold_flows(histogram, feature_config)
    integral = float(histogram.Integral(1, histogram.GetNbinsX()))
    histogram.Scale(0.0 if integral == 0.0 else 1.0 / integral)
    return histogram


def _apply_display_range(histogram, plot_config: dict, feature: str) -> None:
    values = plot_config.get("display_x_ranges", {}).get(feature)
    if values:
        histogram.GetXaxis().SetRangeUser(float(values[0]), float(values[1]))


def _histogram_quantiles(histogram) -> tuple[float, float, float]:
    probabilities = array("d", [0.16, 0.50, 0.84])
    quantiles = array("d", [0.0, 0.0, 0.0])
    histogram.GetQuantiles(3, quantiles, probabilities)
    return tuple(float(value) for value in quantiles)


def make_faceted_plot(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetErrorX(0.0)
    config, plot_config, requested = _comparison_inputs(args)
    feature_config = config["features"][args.feature]
    pairs = [requested[index:index + 2] for index in range(0, len(requested), 2)]
    columns = 3 if len(pairs) > 4 else 2
    rows = int(math.ceil(len(pairs) / columns))
    canvas = ROOT.TCanvas("faceted_canvas", "", 520 * columns, 460 * rows)
    canvas.Divide(columns, rows, 0.002, 0.002)
    keepalive = []
    summary_rows = []
    for panel_index, pair in enumerate(pairs, start=1):
        pad = canvas.cd(panel_index)
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.04)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.11)
        curves = []
        for curve_index, (group, loaded) in enumerate(pair):
            histogram = _unit_area_histogram(
                loaded, feature_config, f"facet_{panel_index}_{curve_index}"
            )
            particle = str(loaded["particle_charge"])
            is_electron = particle.startswith("e")
            color = ROOT.kRed + 1 if is_electron else ROOT.kBlue + 1
            marker = 24 if is_electron else 20
            histogram.SetLineColor(color)
            histogram.SetMarkerColor(color)
            histogram.SetMarkerStyle(marker)
            histogram.SetMarkerSize(0.70)
            _apply_display_range(histogram, plot_config, args.feature)
            curves.append((group, loaded, histogram))
        maximum = max(
            histogram.GetBinContent(bin_index) + histogram.GetBinError(bin_index)
            for _, _, histogram in curves
            for bin_index in range(1, histogram.GetNbinsX() + 1)
            if histogram.GetXaxis().GetFirst() <= bin_index <= histogram.GetXaxis().GetLast()
        )
        legend = ROOT.TLegend(0.53, 0.70, 0.94, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.034)
        energies = []
        for curve_index, (group, loaded, histogram) in enumerate(curves):
            histogram.GetXaxis().SetTitle(str(feature_config["axis_title"]))
            histogram.GetYaxis().SetTitle("Fraction of events / bin")
            histogram.GetXaxis().SetTitleSize(0.044)
            histogram.GetYaxis().SetTitleSize(0.044)
            histogram.GetXaxis().SetLabelSize(0.036)
            histogram.GetYaxis().SetLabelSize(0.036)
            histogram.SetMinimum(0.0)
            histogram.SetMaximum(maximum * 1.35)
            histogram.Draw("E1 P" if curve_index == 0 else "E1 P SAME")
            particle = str(loaded["particle_charge"])
            energy = float(loaded["beam_energy_gev"])
            energies.append((particle, energy))
            legend.AddEntry(histogram, f"{particle} {energy:g} GeV", "lep")
            q16, median, q84 = _histogram_quantiles(histogram)
            summary_rows.append({
                "feature": args.feature,
                "plot_set": args.plot_set,
                "panel": panel_index,
                "group": group,
                "particle_charge": particle,
                "beam_energy_gev": f"{energy:g}",
                "source_geometry_id": loaded["source_geometry_id"],
                "q16": f"{q16:.17g}",
                "median": f"{median:.17g}",
                "q84": f"{q84:.17g}",
                "draw_option": "E1 P",
            })
        legend.Draw()
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(0.040)
        pion_energy = next(value for particle, value in energies if particle.startswith("pi"))
        electron_energy = next(value for particle, value in energies if particle.startswith("e"))
        if math.isclose(pion_energy, electron_energy):
            title = f"{pion_energy:g} GeV pion vs electron"
        else:
            title = f"{pion_energy:g} GeV pion vs {electron_energy:g} GeV electron"
        label.DrawLatex(0.15, 0.93, title)
        if args.geometry_policy == "cross_geometry_diagnostic":
            label.SetTextSize(0.027)
            label.DrawLatex(0.15, 0.875, "Cross-geometry diagnostic")
        keepalive.extend([legend, label, *[item[2] for item in curves]])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary_rows[0])
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote faceted plot {output} and summary {summary}")


def make_energy_summary_plot(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    config, plot_config, requested = _comparison_inputs(args)
    feature_config = config["features"][args.feature]
    rows = []
    for group, loaded in requested:
        histogram = _unit_area_histogram(loaded, feature_config, f"summary_{group}")
        q16, median, q84 = _histogram_quantiles(histogram)
        rows.append({
            "feature": args.feature,
            "plot_set": args.plot_set,
            "group": group,
            "particle_charge": str(loaded["particle_charge"]),
            "beam_energy_gev": float(loaded["beam_energy_gev"]),
            "source_geometry_id": str(loaded["source_geometry_id"]),
            "q16": q16,
            "median": median,
            "q84": q84,
        })
    canvas = ROOT.TCanvas("energy_summary_canvas", "", 950, 760)
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.05)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.14)
    energies = [row["beam_energy_gev"] for row in rows]
    upper = max(row["q84"] for row in rows)
    frame = ROOT.TH1D("energy_summary_frame", "", 100, min(energies) - 15, max(energies) + 15)
    frame.SetMinimum(0.0)
    frame.SetMaximum(upper * 1.28)
    frame.GetXaxis().SetTitle("Beam energy [GeV]")
    frame.GetYaxis().SetTitle(f"Median {feature_config['axis_title']} (16-84% interval)")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.041)
    frame.Draw("AXIS")
    legend = ROOT.TLegend(0.16, 0.76, 0.43, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    keepalive = [frame, legend]
    for particle_index, particle_prefix in enumerate(("pi", "e")):
        selected = sorted(
            (row for row in rows if row["particle_charge"].startswith(particle_prefix)),
            key=lambda row: row["beam_energy_gev"],
        )
        graph = ROOT.TGraphAsymmErrors(len(selected))
        for index, row in enumerate(selected):
            graph.SetPoint(index, row["beam_energy_gev"], row["median"])
            graph.SetPointError(
                index, 0.0, 0.0,
                row["median"] - row["q16"], row["q84"] - row["median"],
            )
        is_electron = particle_prefix == "e"
        color = ROOT.kRed + 1 if is_electron else ROOT.kBlue + 1
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(24 if is_electron else 20)
        graph.SetMarkerSize(1.0)
        graph.SetLineWidth(2)
        graph.Draw("P L Z SAME")
        legend.AddEntry(graph, "electron" if is_electron else "pion", "lp")
        keepalive.append(graph)
    legend.Draw()
    annotation = ROOT.TLatex()
    annotation.SetNDC(True)
    annotation.SetTextFont(42)
    annotation.SetTextSize(0.034)
    title = "MC test beam - shared W 2-wall geometry"
    if args.geometry_policy == "cross_geometry_diagnostic":
        title = "Data test beam - cross-geometry diagnostic"
    annotation.DrawLatex(0.15, 0.93, title)
    keepalive.append(annotation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote energy-summary plot {output} and summary {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--family", required=True)
    audit.add_argument("--output", required=True)
    histogram = subparsers.add_parser("histogram")
    histogram.add_argument("--config", required=True)
    histogram.add_argument("--manifest", required=True)
    histogram.add_argument("--group", required=True)
    histogram.add_argument("--feature", required=True)
    histogram.add_argument("--output", required=True)
    histogram.add_argument(
        "--allow-missing-validated-sources", action="store_true"
    )
    plot = subparsers.add_parser("plot")
    plot.add_argument("--config", required=True)
    plot.add_argument("--plot-set", default="")
    plot.add_argument("--feature", required=True)
    plot.add_argument("--normalization", required=True)
    plot.add_argument(
        "--geometry-policy",
        choices=("not_enforced", "same_geometry", "cross_geometry_diagnostic"),
        default="not_enforced",
    )
    plot.add_argument("--hist", action="append", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--summary", required=True)
    for command in ("faceted-plot", "energy-summary-plot"):
        derived = subparsers.add_parser(command)
        derived.add_argument("--config", required=True)
        derived.add_argument("--plot-set", required=True)
        derived.add_argument("--feature", required=True)
        derived.add_argument(
            "--geometry-policy",
            choices=("same_geometry", "cross_geometry_diagnostic"),
            required=True,
        )
        derived.add_argument("--hist", action="append", required=True)
        derived.add_argument("--output", required=True)
        derived.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit_family(args)
    elif args.command == "histogram":
        make_histogram(args)
    elif args.command == "faceted-plot":
        make_faceted_plot(args)
    elif args.command == "energy-summary-plot":
        make_energy_summary_plot(args)
    else:
        make_plot(args)


if __name__ == "__main__":
    main()
