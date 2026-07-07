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


def get_feature_config(hist_cfg: dict, feature: str) -> dict:
    features = hist_cfg.get("features", {})
    if feature not in features:
        raise KeyError(f"feature={feature!r} is not defined in histogram config")

    merged = dict(hist_cfg.get("defaults", {}))
    merged.update(features[feature])
    return merged


def get_feature_expression(feature_cfg: dict, feature: str, row: dict) -> str:
    if is_real_data(row) and feature_cfg.get("real_data_expression"):
        return str(feature_cfg["real_data_expression"]).strip()
    return str(feature_cfg.get("expression", feature)).strip()


def build_selection(
    representative_row: dict,
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
        if not cut_applies(cut_cfg.get("applies_to", "all"), representative_row):
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


def histogram_model(feature: str, feature_cfg: dict):
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

    return ROOT.RDF.TH1DModel(f"h_{safe_name(feature)}", "", n_bins, x_min, x_max)


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


def select_group_rows(metadata_csv: str | Path, region: str, particle_group: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"region", "particle_group", "feature_partition_path"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    selected = df.loc[
        (df["region"].astype(str) == str(region))
        & (df["particle_group"].astype(str) == str(particle_group))
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No metadata rows found for region={region!r}, particle_group={particle_group!r}"
        )

    return selected.reset_index(drop=True)


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


def build_chain(
    rows: pd.DataFrame,
    tree_name: str,
    repo_root: str | Path,
    friend_tree_name: str | None = None,
) -> tuple[ROOT.TChain, list[str], list[str]]:
    chain = ROOT.TChain(tree_name)
    friend_chain = ROOT.TChain(friend_tree_name) if friend_tree_name else None
    added_paths = []
    skipped_paths = []

    for _, row in rows.iterrows():
        raw_path = row.get("feature_partition_path")
        if not raw_path or pd.isna(raw_path):
            skipped_paths.append("")
            continue

        path = resolve_path(str(raw_path), repo_root)
        if friend_tree_name and not root_file_has_tree(path, friend_tree_name):
            skipped_paths.append(f"{path} (missing tree {friend_tree_name})")
            continue

        added = chain.Add(path)
        friend_added = friend_chain.Add(path) if friend_chain is not None else added
        if added <= 0 or friend_added <= 0:
            skipped_paths.append(path)
            continue
        added_paths.append(path)

    if not added_paths:
        message = "No feature partition files could be added to the TChain"
        if friend_tree_name and any(f"missing tree {friend_tree_name}" in path for path in skipped_paths):
            message += (
                f"; required friend tree {friend_tree_name!r} is missing. "
                "Regenerate region partitions with cutFlowSummary preservation enabled."
            )
        raise RuntimeError(message)

    if friend_chain is not None:
        build_friend_indices(chain, friend_chain)
        chain.AddFriend(friend_chain)

    return chain, added_paths, skipped_paths


def define_eval_feature(df: ROOT.RDataFrame, expression: str):
    expression = expression.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression):
        return df, expression
    return df.Define("__eval_feature", expression), "__eval_feature"


def sum_numeric_column(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return 0.0
    values = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    return float(values.sum())


def make_group_histogram(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    hist_config = resolve_path(args.hist_config, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    hist_cfg = load_yaml(hist_config)
    eval_cfg = load_yaml(eval_config)
    feature_cfg = get_feature_config(hist_cfg, args.feature)

    rows = select_group_rows(metadata_csv, args.region, args.particle_group)
    representative_row = rows.iloc[0].to_dict()

    selection, applied_cuts = build_selection(
        representative_row=representative_row,
        feature_cfg=feature_cfg,
        eval_cfg=eval_cfg,
        base_cut_key=args.base_cut_key,
        extra_cut_key=args.extra_cut_key,
    )

    tree_name = args.tree or eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    friend_tree_name = args.friend_tree or eval_cfg.get("metadata", {}).get("friend_tree_name")
    chain, added_paths, skipped_paths = build_chain(
        rows,
        tree_name,
        repo_root,
        friend_tree_name=friend_tree_name if selection_needs_friend_tree(applied_cuts) else None,
    )

    df = ROOT.RDataFrame(chain)
    if selection:
        df = df.Filter(selection, "region_eval_selection")

    expression = get_feature_expression(feature_cfg, args.feature, representative_row)
    df, hist_column = define_eval_feature(df, expression)

    model = histogram_model(args.feature, feature_cfg)
    selected_count = df.Count()
    hist_result = df.Histo1D(model, hist_column)

    selected_entries = int(selected_count.GetValue())
    hist = hist_result.GetValue().Clone("hist")
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

    partition_ids = [str(v) for v in rows.get("partition_id", pd.Series(dtype=str)).tolist()]
    raw_integral = float(hist.Integral())
    lumi = sum_numeric_column(rows, "lumi_per_partition")
    sum_n_events = int(sum_numeric_column(rows, "n_events"))

    write_metadata(
        output_file,
        {
            "region": args.region,
            "particle_group": args.particle_group,
            "particle_family": representative_row.get("particle_family", ""),
            "particle_id": int(representative_row["particle_id"])
            if "particle_id" in representative_row and not pd.isna(representative_row["particle_id"])
            else "",
            "feature": args.feature,
            "feature_expression": expression,
            "hist_column": hist_column,
            "base_cut_key": args.base_cut_key,
            "extra_cut_key": args.extra_cut_key,
            "selection": selection,
            "applied_cuts": ",".join(applied_cuts),
            "n_partitions": int(len(rows)),
            "n_added_files": int(len(added_paths)),
            "n_skipped_files": int(len(skipped_paths)),
            "sum_n_events": sum_n_events,
            "lumi": lumi,
            "selected_entries": selected_entries,
            "raw_integral": raw_integral,
            "source_partition_ids": ",".join(partition_ids),
            "source_feature_partition_paths": "\n".join(added_paths),
            "skipped_feature_partition_paths": "\n".join(skipped_paths),
            "x_min": float(feature_cfg["x_min"]),
            "x_max": float(feature_cfg["x_max"]),
            "bin_width": float(feature_cfg["bin_width"]),
            "logy": bool(feature_cfg.get("logy", False)),
        },
    )
    output_file.Close()

    print(f"Wrote grouped histogram: {output_path}")
    print(f"  region         : {args.region}")
    print(f"  particle_group : {args.particle_group}")
    print(f"  feature        : {args.feature}")
    print(f"  partitions     : {len(rows)}")
    print(f"  files added    : {len(added_paths)}")
    print(f"  selected       : {selected_entries}")
    print(f"  raw_integral   : {raw_integral}")
    print(f"  lumi           : {lumi}")
    print(f"  selection      : {selection if selection else '1'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one grouped region-partition histogram for one region, particle, feature, and cut option."
    )
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--hist-config", required=True, help="Histogram feature config YAML")
    parser.add_argument("--eval-config", required=True, help="Evaluation options config YAML")
    parser.add_argument("--region", required=True, help="Region name")
    parser.add_argument("--particle-group", required=True, help="Particle group name")
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
    make_group_histogram(parser.parse_args())


if __name__ == "__main__":
    main()
