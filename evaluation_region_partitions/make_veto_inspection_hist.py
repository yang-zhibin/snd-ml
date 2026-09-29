#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.TH1.AddDirectory(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build veto-inspection histograms for one consolidated dataset."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def resolve_path(path: str, repo_root: str | Path) -> str:
    if path.startswith("root://") or os.path.isabs(path):
        return path
    return str(Path(repo_root) / path)


def select_manifest_row(path: str | Path, dataset: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("dataset_name", "")) == dataset
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for dataset={dataset!r}; found {len(rows)}"
        )
    return rows[0]


def dataset_config(config: dict, dataset: str) -> dict:
    rows = [item for item in config.get("datasets", []) if item.get("name") == dataset]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one dataset config for dataset={dataset!r}; found {len(rows)}"
        )
    return rows[0]


def uniform_model_1d(name: str, cfg: dict):
    return ROOT.RDF.TH1DModel(
        name,
        "",
        int(cfg["bins"]),
        float(cfg["min"]),
        float(cfg["max"]),
    )


def uniform_model_2d(name: str, cfg: dict):
    return ROOT.RDF.TH2DModel(
        name,
        "",
        int(cfg["x_bins"]),
        float(cfg["x_min"]),
        float(cfg["x_max"]),
        int(cfg["y_bins"]),
        float(cfg["y_min"]),
        float(cfg["y_max"]),
    )


def fold_axis(hist, axis: str, underflow: bool, overflow: bool) -> None:
    if axis == "x":
        n_primary = hist.GetNbinsX()
        n_secondary = hist.GetNbinsY() if hist.InheritsFrom("TH2") else 1

        def get_bin(primary: int, secondary: int) -> int:
            return hist.GetBin(primary, secondary) if hist.InheritsFrom("TH2") else primary

    elif axis == "y" and hist.InheritsFrom("TH2"):
        n_primary = hist.GetNbinsY()
        n_secondary = hist.GetNbinsX()

        def get_bin(primary: int, secondary: int) -> int:
            return hist.GetBin(secondary, primary)

    else:
        raise ValueError(f"Unsupported histogram axis {axis!r}")

    for secondary in range(1, n_secondary + 1):
        if underflow:
            source = get_bin(0, secondary)
            target = get_bin(1, secondary)
            hist.SetBinContent(target, hist.GetBinContent(target) + hist.GetBinContent(source))
            hist.SetBinError(
                target,
                math.hypot(hist.GetBinError(target), hist.GetBinError(source)),
            )
            hist.SetBinContent(source, 0.0)
            hist.SetBinError(source, 0.0)
        if overflow:
            source = get_bin(n_primary + 1, secondary)
            target = get_bin(n_primary, secondary)
            hist.SetBinContent(target, hist.GetBinContent(target) + hist.GetBinContent(source))
            hist.SetBinError(
                target,
                math.hypot(hist.GetBinError(target), hist.GetBinError(source)),
            )
            hist.SetBinContent(source, 0.0)
            hist.SetBinError(source, 0.0)


def clone_result(result, name: str):
    hist = result.GetValue().Clone(name)
    hist.SetDirectory(0)
    return hist


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def write_named(output: ROOT.TFile, name: str, value: object) -> None:
    output.cd()
    if isinstance(value, bool):
        ROOT.TParameter("int")(name, int(value)).Write()
    elif isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, int(value)).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, float(value)).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def build(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    manifest_path = resolve_path(args.manifest, args.repo_root)
    manifest = select_manifest_row(manifest_path, args.dataset)
    dataset_cfg = dataset_config(config, args.dataset)
    input_path = resolve_path(manifest["output_path"], args.repo_root)
    tree_name = str(config["input"]["tree_name"])
    weight_name = str(config["input"]["weight"])

    root_file = ROOT.TFile.Open(input_path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open input ROOT file: {input_path}")
    tree = root_file.Get(tree_name)
    if not tree:
        raise KeyError(f"Tree {tree_name!r} is missing in {input_path}")

    required = {
        str(config["selection"]["count_expression"]),
        str(config["selection"]["earliest_time_expression"]),
        str(config["selection"]["scifi_count_expression"]),
        str(config["selection"]["scifi_density_expression"]),
        weight_name,
    }
    columns = {str(branch.GetName()) for branch in tree.GetListOfBranches()}
    missing = required - columns
    if missing:
        raise KeyError(f"Input tree is missing required branches: {sorted(missing)}")

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)

    selection = config["selection"]
    count_expr = str(selection["count_expression"])
    time_expr = str(selection["earliest_time_expression"])
    scifi_count_expr = str(selection["scifi_count_expression"])
    scifi_density_expr = str(selection["scifi_density_expression"])
    no_veto_expr = str(selection["no_veto_expression"])
    valid_expr = " ".join(str(selection["valid_time_expression"]).split())
    invalid_expr = " ".join(str(selection["invalid_time_expression"]).split())
    display_time = float(selection["no_veto_display_time_ns"])
    reserve_no_veto_time_bin = bool(
        selection.get("reserve_no_veto_time_bin", True)
    )

    rdf_all = ROOT.RDataFrame(tree)
    rdf_all = (
        rdf_all.Define("__veto_raw_weight", "1.0")
        .Define("__veto_count", count_expr)
        .Define("__veto_time_ns", time_expr)
        .Define("__scifi_count", scifi_count_expr)
        .Define("__scifi_density", scifi_density_expr)
        .Define("__veto_no_hit", no_veto_expr)
        .Define("__veto_time_valid", valid_expr)
        .Define("__veto_time_invalid", invalid_expr)
        .Define(
            "__veto_time_plot_ns",
            f"__veto_no_hit ? {display_time:.17g} : __veto_time_ns",
        )
    )
    analysis_event_expr = " ".join(
        str(selection.get("analysis_event_expression", "true")).split()
    )
    rdf = rdf_all.Filter(analysis_event_expr, "inspection base event selection")

    included_time = rdf.Filter("__veto_no_hit || __veto_time_valid")
    positive_count = rdf.Filter("__veto_count > 0")
    valid_time = rdf.Filter("__veto_time_valid")

    count_cfg = config["histograms"]["veto_count"]
    time_cfg = config["histograms"]["veto_earliest_time"]
    two_dimensional = {
        "veto_time_vs_count": {
            "frame": included_time,
            "x": "__veto_time_plot_ns",
            "y": "__veto_count",
        },
        "veto_time_vs_scifi_count": {
            "frame": included_time,
            "x": "__veto_time_plot_ns",
            "y": "__scifi_count",
        },
        "veto_time_vs_scifi_density": {
            "frame": included_time,
            "x": "__veto_time_plot_ns",
            "y": "__scifi_density",
        },
        "veto_count_vs_scifi_count": {
            "frame": rdf,
            "x": "__veto_count",
            "y": "__scifi_count",
        },
        "veto_count_vs_scifi_density": {
            "frame": rdf,
            "x": "__veto_count",
            "y": "__scifi_density",
        },
    }

    actions = {
        "source_total": rdf_all.Count(),
        "total": rdf.Count(),
        "weighted_total": rdf.Sum(weight_name),
        "no_veto": rdf.Filter("__veto_no_hit").Count(),
        "weighted_no_veto": rdf.Filter("__veto_no_hit").Sum(weight_name),
        "valid_time": valid_time.Count(),
        "weighted_valid_time": valid_time.Sum(weight_name),
        "invalid_time": rdf.Filter("__veto_time_invalid").Count(),
        "weighted_invalid_time": rdf.Filter("__veto_time_invalid").Sum(weight_name),
        "negative_count": rdf.Filter("__veto_count < 0").Count(),
        "zero_with_valid_time": rdf.Filter("__veto_no_hit && __veto_time_valid").Count(),
        "valid_time_min": valid_time.Min("__veto_time_ns"),
        "valid_time_max": valid_time.Max("__veto_time_ns"),
    }
    if reserve_no_veto_time_bin:
        actions["valid_collision"] = valid_time.Filter(
            f"__veto_time_ns >= {float(time_cfg['min']):.17g} && "
            f"__veto_time_ns < "
            f"{float(time_cfg['min']) + (float(time_cfg['max']) - float(time_cfg['min'])) / int(time_cfg['bins']):.17g}"
        ).Count()

    hist_actions = {}
    for suffix, weight in (("raw", "__veto_raw_weight"), ("weighted", weight_name)):
        hist_actions[f"h_veto_count_all_{suffix}"] = rdf.Histo1D(
            uniform_model_1d(f"h_veto_count_all_{suffix}", count_cfg),
            "__veto_count",
            weight,
        )
        hist_actions[f"h_veto_count_positive_{suffix}"] = positive_count.Histo1D(
            uniform_model_1d(f"h_veto_count_positive_{suffix}", count_cfg),
            "__veto_count",
            weight,
        )
        hist_actions[f"h_veto_earliest_time_all_{suffix}"] = included_time.Histo1D(
            uniform_model_1d(f"h_veto_earliest_time_all_{suffix}", time_cfg),
            "__veto_time_plot_ns",
            weight,
        )
        hist_actions[f"h_veto_earliest_time_valid_{suffix}"] = valid_time.Histo1D(
            uniform_model_1d(f"h_veto_earliest_time_valid_{suffix}", time_cfg),
            "__veto_time_ns",
            weight,
        )
        for study_name, study in two_dimensional.items():
            hist_name = f"h2_{study_name}_{suffix}"
            hist_actions[hist_name] = study["frame"].Histo2D(
                uniform_model_2d(hist_name, config["histograms"][study_name]),
                study["x"],
                study["y"],
                weight,
            )

    ROOT.RDF.RunGraphs(list(actions.values()) + list(hist_actions.values()))
    values = {name: action.GetValue() for name, action in actions.items()}

    if int(values["negative_count"]) != 0:
        raise ValueError(f"Found {values['negative_count']} events with count_veto < 0")
    if int(values["zero_with_valid_time"]) != 0:
        raise ValueError(
            f"Found {values['zero_with_valid_time']} zero-veto events with valid veto time"
        )
    if reserve_no_veto_time_bin and int(values["valid_collision"]) != 0:
        raise ValueError(
            f"Found {values['valid_collision']} valid veto times in the dedicated "
            "no-veto display bin"
        )

    total = int(values["total"])
    if total <= 0:
        raise ValueError(f"Dataset {args.dataset} has no events")
    invalid_fraction = int(values["invalid_time"]) / total
    warning_threshold = float(selection["invalid_time_warning_fraction"])
    status = "WARN" if invalid_fraction > warning_threshold else "PASS"

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    output = ROOT.TFile.Open(str(output_path), "RECREATE")
    if not output or output.IsZombie():
        raise OSError(f"Could not create output ROOT file: {output_path}")

    try:
        for name, result in hist_actions.items():
            hist = clone_result(result, name)
            if hist.InheritsFrom("TH2"):
                suffix = "_weighted" if name.endswith("_weighted") else "_raw"
                study_name = name.removeprefix("h2_").removesuffix(suffix)
                two_cfg = config["histograms"][study_name]
                fold_axis(
                    hist,
                    "x",
                    bool(two_cfg.get("fold_x_underflow", False)),
                    bool(two_cfg.get("fold_x_overflow", False)),
                )
                fold_axis(
                    hist,
                    "y",
                    bool(two_cfg.get("fold_y_underflow", False)),
                    bool(two_cfg.get("fold_y_overflow", False)),
                )
                hist.GetXaxis().SetTitle(str(two_cfg["x_title"]))
                hist.GetYaxis().SetTitle(str(two_cfg["y_title"]))
            elif "count" in name:
                fold_axis(
                    hist,
                    "x",
                    bool(count_cfg.get("fold_underflow", False)),
                    bool(count_cfg.get("fold_overflow", False)),
                )
                hist.GetXaxis().SetTitle(str(count_cfg["x_title"]))
            else:
                fold_axis(
                    hist,
                    "x",
                    bool(time_cfg.get("fold_underflow", False)),
                    bool(time_cfg.get("fold_overflow", False)),
                )
                hist.GetXaxis().SetTitle(str(time_cfg["x_title"]))
            output.cd()
            hist.Write()

        metadata = {
            "inspection_version": config["version"],
            "dataset_name": args.dataset,
            "dataset_label": dataset_cfg["label"],
            "dataset_category": dataset_cfg["category"],
            "input_path": input_path,
            "tree_name": tree_name,
            "weight_branch": weight_name,
            "no_veto_display_time_ns": display_time,
            "analysis_event_expression": analysis_event_expr,
            "source_events": int(values["source_total"]),
            "total_events": total,
            "weighted_total": float(values["weighted_total"]),
            "no_veto_events": int(values["no_veto"]),
            "valid_timed_veto_events": int(values["valid_time"]),
            "invalid_timed_veto_events": int(values["invalid_time"]),
            "status": status,
        }
        for name, value in metadata.items():
            write_named(output, name, value)
    finally:
        output.Close()
        root_file.Close()

    fieldnames = [
        "inspection_version",
        "dataset_name",
        "dataset_label",
        "dataset_category",
        "input_path",
        "analysis_event_expression",
        "source_events",
        "selected_fraction",
        "total_events",
        "weighted_total",
        "no_veto_events",
        "weighted_no_veto",
        "no_veto_fraction",
        "valid_timed_veto_events",
        "weighted_valid_timed_veto",
        "valid_timed_veto_fraction",
        "invalid_timed_veto_events",
        "weighted_invalid_timed_veto",
        "invalid_timed_veto_fraction",
        "minimum_valid_time_ns",
        "maximum_valid_time_ns",
        "status",
        "message",
    ]
    row = {
        "inspection_version": config["version"],
        "dataset_name": args.dataset,
        "dataset_label": dataset_cfg["label"],
        "dataset_category": dataset_cfg["category"],
        "input_path": input_path,
        "analysis_event_expression": analysis_event_expr,
        "source_events": int(values["source_total"]),
        "selected_fraction": total / int(values["source_total"]),
        "total_events": total,
        "weighted_total": float(values["weighted_total"]),
        "no_veto_events": int(values["no_veto"]),
        "weighted_no_veto": float(values["weighted_no_veto"]),
        "no_veto_fraction": int(values["no_veto"]) / total,
        "valid_timed_veto_events": int(values["valid_time"]),
        "weighted_valid_timed_veto": float(values["weighted_valid_time"]),
        "valid_timed_veto_fraction": int(values["valid_time"]) / total,
        "invalid_timed_veto_events": int(values["invalid_time"]),
        "weighted_invalid_timed_veto": float(values["weighted_invalid_time"]),
        "invalid_timed_veto_fraction": invalid_fraction,
        "minimum_valid_time_ns": float(values["valid_time_min"]),
        "maximum_valid_time_ns": float(values["valid_time_max"]),
        "status": status,
        "message": (
            f"invalid timing fraction {invalid_fraction:.6g} exceeds "
            f"{warning_threshold:.6g}"
            if status == "WARN"
            else "ok"
        ),
    }
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(
        f"{args.dataset}: total={total}, no_veto={row['no_veto_events']}, "
        f"valid_time={row['valid_timed_veto_events']}, "
        f"invalid_time={row['invalid_timed_veto_events']}, status={status}"
    )


if __name__ == "__main__":
    build(parse_args())
