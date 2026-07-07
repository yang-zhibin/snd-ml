#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def resolve_path(path: str, repo_root: str | Path) -> str:
    if path.startswith("root://") or os.path.isabs(path):
        return path
    return str(Path(repo_root) / path)


def load_partition_row(metadata_csv: str | Path, partition_id: str) -> dict:
    df = pd.read_csv(metadata_csv)
    if "partition_id" not in df.columns:
        raise KeyError(f"partition_id column is missing from {metadata_csv}")

    matched = df.loc[df["partition_id"].astype(str) == str(partition_id)]
    if matched.empty:
        raise ValueError(f"partition_id={partition_id!r} was not found in {metadata_csv}")
    if len(matched) > 1:
        raise ValueError(f"partition_id={partition_id!r} appears {len(matched)} times in {metadata_csv}")

    return matched.iloc[0].to_dict()


def is_real_data(row: dict) -> bool:
    particle_group = str(row.get("particle_group", ""))
    particle_family = str(row.get("particle_family", ""))
    return particle_group == "real_data" or particle_family == "real_data"


def cut_applies(applies_to: str, row: dict) -> bool:
    applies_to = str(applies_to or "all")
    if applies_to == "all":
        return True
    if applies_to == "real_data_only":
        return is_real_data(row)
    if applies_to == "mc_only":
        return not is_real_data(row)
    raise ValueError(f"Unknown cut applies_to value: {applies_to}")


def build_selection(
    row: dict,
    feature_cfg: dict,
    eval_cfg: dict,
    base_cut_key: str,
    extra_cut_key: str,
) -> tuple[str, list[str]]:
    cuts_cfg = eval_cfg["cuts"]
    cut_definitions = cuts_cfg.get("definitions", {})
    base_options = cuts_cfg.get("base_cut_options", {})
    extra_options = cuts_cfg.get("extra_cut_options", {})

    if base_cut_key not in base_options:
        raise KeyError(f"base_cut_key={base_cut_key!r} is not defined")
    if extra_cut_key not in extra_options:
        raise KeyError(f"extra_cut_key={extra_cut_key!r} is not defined")

    expressions = []
    labels = []

    for cut_name in base_options[base_cut_key]:
        if cut_name not in cut_definitions:
            raise KeyError(f"Cut {cut_name!r} is listed in {base_cut_key!r} but not defined")

        cut_cfg = cut_definitions[cut_name]
        if not cut_applies(cut_cfg.get("applies_to", "all"), row):
            continue

        expression = str(cut_cfg.get("expression", "")).strip()
        if expression:
            expressions.append(f"({expression})")
            labels.append(cut_name)

    extra_expression = str(extra_options[extra_cut_key] or "").strip()
    if extra_expression:
        expressions.append(f"({extra_expression})")
        labels.append(f"extra:{extra_cut_key}")

    feature_cut = str(feature_cfg.get("feature_cut", "") or "").strip()
    if feature_cut:
        expressions.append(f"({feature_cut})")
        labels.append("feature_cut")

    return " && ".join(expressions), labels


def selection_needs_friend_tree(applied_cuts: list[str]) -> bool:
    return any(
        label != "feature_cut" and not str(label).startswith("extra:")
        for label in applied_cuts
    )


def get_feature_config(hist_cfg: dict, feature: str) -> dict:
    features = hist_cfg.get("features", {})
    if feature not in features:
        raise KeyError(f"feature={feature!r} is not defined in histogram config")

    defaults = hist_cfg.get("defaults", {})
    merged = dict(defaults)
    merged.update(features[feature])
    return merged


def get_feature_expression(feature_cfg: dict, feature: str, row: dict) -> str:
    if is_real_data(row) and feature_cfg.get("real_data_expression"):
        return str(feature_cfg["real_data_expression"]).strip()
    return str(feature_cfg.get("expression", feature)).strip()


def histogram_model(feature: str, feature_cfg: dict) -> tuple[str, int, float, float]:
    x_min = float(feature_cfg["x_min"])
    x_max = float(feature_cfg["x_max"])
    bin_width = float(feature_cfg["bin_width"])
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min for feature={feature}")
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive for feature={feature}")

    n_bins = int(round((x_max - x_min) / bin_width))
    if n_bins <= 0:
        raise ValueError(f"Computed non-positive number of bins for feature={feature}")

    hist_name = f"h_{safe_name(feature)}"
    return hist_name, n_bins, x_min, x_max


def fold_underflow(hist: ROOT.TH1) -> None:
    hist.SetBinContent(1, hist.GetBinContent(1) + hist.GetBinContent(0))
    hist.SetBinError(1, math.sqrt(hist.GetBinError(1) ** 2 + hist.GetBinError(0) ** 2))
    hist.SetBinContent(0, 0.0)
    hist.SetBinError(0, 0.0)


def fold_overflow(hist: ROOT.TH1) -> None:
    last = hist.GetNbinsX()
    hist.SetBinContent(last, hist.GetBinContent(last) + hist.GetBinContent(last + 1))
    hist.SetBinError(last, math.sqrt(hist.GetBinError(last) ** 2 + hist.GetBinError(last + 1) ** 2))
    hist.SetBinContent(last + 1, 0.0)
    hist.SetBinError(last + 1, 0.0)


def sanitize_hist_bins(hist: ROOT.TH1) -> None:
    for ibin in range(0, hist.GetNbinsX() + 2):
        content = hist.GetBinContent(ibin)
        error = hist.GetBinError(ibin)
        if not math.isfinite(content):
            hist.SetBinContent(ibin, 0.0)
        if not math.isfinite(error):
            hist.SetBinError(ibin, 0.0)


def write_metadata(output_file: ROOT.TFile, values: dict[str, object]) -> None:
    output_file.cd()
    for key, value in values.items():
        if value is None:
            value = ""

        if isinstance(value, bool):
            ROOT.TParameter("int")(key, int(value)).Write()
        elif isinstance(value, int):
            ROOT.TParameter("int")(key, int(value)).Write()
        elif isinstance(value, float):
            ROOT.TParameter("double")(key, float(value)).Write()
        else:
            ROOT.TNamed(key, str(value)).Write()


def root_file_has_tree(path: str, tree_name: str) -> bool:
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        return False
    try:
        return bool(root_file.Get(tree_name))
    finally:
        root_file.Close()


def build_friend_indices(chain: ROOT.TChain, friend_chain: ROOT.TChain) -> None:
    old_error_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    try:
        chain.BuildIndex("runId", "eventId")
        friend_chain.BuildIndex("runId", "eventId")
    finally:
        ROOT.gErrorIgnoreLevel = old_error_level


def make_histogram(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    hist_config = resolve_path(args.hist_config, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    hist_cfg = load_yaml(hist_config)
    eval_cfg = load_yaml(eval_config)

    row = load_partition_row(metadata_csv, args.partition_id)
    feature_cfg = get_feature_config(hist_cfg, args.feature)

    feature_path = row.get("feature_partition_path")
    if not feature_path or pd.isna(feature_path):
        raise ValueError(f"feature_partition_path is missing for partition_id={args.partition_id}")

    selection, applied_cuts = build_selection(
        row=row,
        feature_cfg=feature_cfg,
        eval_cfg=eval_cfg,
        base_cut_key=args.base_cut_key,
        extra_cut_key=args.extra_cut_key,
    )

    tree_name = args.tree or eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    friend_tree_name = args.friend_tree or eval_cfg.get("metadata", {}).get("friend_tree_name")
    feature_path = resolve_path(str(feature_path), repo_root)

    chain = ROOT.TChain(tree_name)
    added = chain.Add(feature_path)
    if added <= 0 or chain.GetEntries() < 0:
        raise RuntimeError(f"Could not add feature partition file to TChain: {feature_path}")
    if friend_tree_name and selection_needs_friend_tree(applied_cuts):
        if not root_file_has_tree(feature_path, friend_tree_name):
            raise RuntimeError(
                f"Required friend tree {friend_tree_name!r} is missing in {feature_path}. "
                "Regenerate region partitions with cutFlowSummary preservation enabled."
            )
        friend_chain = ROOT.TChain(friend_tree_name)
        friend_added = friend_chain.Add(feature_path)
        if friend_added <= 0:
            raise RuntimeError(
                f"Could not add friend tree {friend_tree_name!r} from {feature_path}"
            )
        build_friend_indices(chain, friend_chain)
        chain.AddFriend(friend_chain)

    hist_name, n_bins, x_min, x_max = histogram_model(args.feature, feature_cfg)
    expression = get_feature_expression(feature_cfg, args.feature, row)
    draw_expr = f"{expression}>>{hist_name}({n_bins},{x_min},{x_max})"

    selected = chain.Draw(draw_expr, selection, "goff")
    if selected < 0:
        raise RuntimeError(
            "TChain::Draw failed for "
            f"partition_id={args.partition_id}, feature={args.feature}, selection={selection!r}"
        )

    hist_tmp = chain.GetHistogram()
    if hist_tmp is None:
        raise RuntimeError(f"ROOT did not return a histogram for feature={args.feature}")

    hist = hist_tmp.Clone("hist")
    hist.SetDirectory(0)
    hist.SetTitle(str(feature_cfg.get("axis_title", args.feature)))
    hist.GetXaxis().SetTitle(str(feature_cfg.get("axis_title", args.feature)))
    hist.GetYaxis().SetTitle(f"Events / {float(feature_cfg['bin_width']):.6g}")

    if bool(feature_cfg.get("fold_underflow", False)):
        fold_underflow(hist)
    if bool(feature_cfg.get("fold_overflow", False)):
        fold_overflow(hist)
    sanitize_hist_bins(hist)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_file = ROOT.TFile(str(output_path), "RECREATE")
    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Could not create output ROOT file: {output_path}")

    hist.Write("hist")
    raw_integral = float(hist.Integral())
    lumi = row.get("lumi_per_partition", 0.0)
    lumi = 0.0 if pd.isna(lumi) else float(lumi)
    n_events = row.get("n_events", 0)
    n_events = 0 if pd.isna(n_events) else int(n_events)

    write_metadata(
        output_file,
        {
            "partition_id": args.partition_id,
            "region": row.get("region", ""),
            "particle_group": row.get("particle_group", ""),
            "particle_family": row.get("particle_family", ""),
            "particle_id": int(row["particle_id"]) if "particle_id" in row and not pd.isna(row["particle_id"]) else "",
            "feature": args.feature,
            "feature_expression": expression,
            "base_cut_key": args.base_cut_key,
            "extra_cut_key": args.extra_cut_key,
            "selection": selection,
            "applied_cuts": ",".join(applied_cuts),
            "feature_partition_path": feature_path,
            "lumi_per_partition": lumi,
            "n_events": n_events,
            "selected_entries": int(selected),
            "raw_integral": raw_integral,
            "x_min": float(feature_cfg["x_min"]),
            "x_max": float(feature_cfg["x_max"]),
            "bin_width": float(feature_cfg["bin_width"]),
            "logy": bool(feature_cfg.get("logy", False)),
        },
    )
    output_file.Close()

    print(f"Wrote histogram: {output_path}")
    print(f"  partition_id   : {args.partition_id}")
    print(f"  feature        : {args.feature}")
    print(f"  selected       : {selected}")
    print(f"  raw_integral   : {raw_integral}")
    print(f"  lumi           : {lumi}")
    print(f"  selection      : {selection if selection else '1'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one region-partition histogram for one partition, feature, and cut option."
    )
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--hist-config", required=True, help="Histogram feature config YAML")
    parser.add_argument("--eval-config", required=True, help="Evaluation options config YAML")
    parser.add_argument("--partition-id", required=True, help="Partition ID from the metadata CSV")
    parser.add_argument("--feature", required=True, help="Feature key from hist_features config")
    parser.add_argument("--base-cut-key", required=True, help="Base cut option key")
    parser.add_argument("--extra-cut-key", required=True, help="Extra cut option key")
    parser.add_argument("--output", required=True, help="Output ROOT histogram file")
    parser.add_argument("--tree", default=None, help="Override input tree name")
    parser.add_argument("--friend-tree", default=None, help="Override optional friend tree name")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve relative metadata/config paths",
    )

    make_histogram(parser.parse_args())


if __name__ == "__main__":
    main()
