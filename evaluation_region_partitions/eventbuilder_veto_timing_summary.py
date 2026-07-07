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


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str, repo_root: str | Path) -> str:
    if path.startswith("root://") or os.path.isabs(path):
        return path
    return str(Path(repo_root) / path)


def clean_expression(expression: object) -> str:
    return " ".join(str(expression or "").split())


def safe_column_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def build_chain(rows: pd.DataFrame, tree_name: str, repo_root: str | Path) -> tuple[ROOT.TChain, list[str], list[str]]:
    chain = ROOT.TChain(tree_name)
    added_paths = []
    skipped_paths = []

    for _, row in rows.iterrows():
        raw_path = row.get("feature_partition_path")
        if not raw_path or pd.isna(raw_path):
            skipped_paths.append("")
            continue

        path = resolve_path(str(raw_path), repo_root)
        added = chain.Add(path)
        if added <= 0:
            skipped_paths.append(path)
            continue
        added_paths.append(path)

    if not added_paths:
        raise RuntimeError("No feature partition files could be added to the TChain")

    return chain, added_paths, skipped_paths


def define_eval_column(rdf: ROOT.RDataFrame, expression: str, name_hint: str) -> tuple[ROOT.RDataFrame, str]:
    expression = clean_expression(expression)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression):
        return rdf, expression
    column_name = f"__eval_{safe_column_name(name_hint)}"
    return rdf.Define(column_name, expression), column_name


def summarize_feature(
    rows: pd.DataFrame,
    tree_name: str,
    repo_root: str | Path,
    feature_name: str,
    feature_cfg: dict,
) -> dict:
    chain, added_paths, skipped_paths = build_chain(rows, tree_name, repo_root)
    rdf = ROOT.RDataFrame(chain)

    expression = clean_expression(feature_cfg.get("expression", feature_name))
    feature_cut = clean_expression(feature_cfg.get("feature_cut", ""))
    rdf, value_column = define_eval_column(rdf, expression, feature_name)

    selected_rdf = rdf.Filter(feature_cut) if feature_cut else rdf
    selected_entries = int(selected_rdf.Count().GetValue())

    if selected_entries > 0:
        mean = float(selected_rdf.Mean(value_column).GetValue())
        stddev = float(selected_rdf.StdDev(value_column).GetValue())
    else:
        mean = float("nan")
        stddev = float("nan")

    return {
        "selected_entries": selected_entries,
        "mean": mean,
        "stddev": stddev,
        "n_partitions": len(rows),
        "n_added_files": len(added_paths),
        "n_skipped_files": len(skipped_paths),
        "feature_expression": expression,
        "feature_cut": feature_cut,
        "skipped_feature_partition_paths": "\n".join(skipped_paths),
    }


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def make_summary(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    hist_config = resolve_path(args.hist_config, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    metadata = pd.read_csv(metadata_csv)
    hist_cfg = load_yaml(hist_config)
    eval_cfg = load_yaml(eval_config)
    timing_cfg = eval_cfg.get("eventbuilder_veto_timing", {})
    if not timing_cfg:
        raise KeyError("eventbuilder_veto_timing section is missing from the evaluation config")

    tree_name = args.tree or eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    features_cfg = hist_cfg.get("features", {})
    timing_features = [str(v) for v in timing_cfg.get("features", [])]
    regions = [str(v) for v in timing_cfg.get("regions", [])]
    pairs = timing_cfg.get("pairs", [])

    if not timing_features:
        raise ValueError("eventbuilder_veto_timing.features is empty")
    if not regions:
        raise ValueError("eventbuilder_veto_timing.regions is empty")
    if not pairs:
        raise ValueError("eventbuilder_veto_timing.pairs is empty")

    rows_out = []
    for region in regions:
        for pair in pairs:
            normal_group = str(pair["normal"])
            eventbuilder_group = str(pair["eventbuilder"])

            normal_rows = metadata.loc[
                (metadata["region"].astype(str) == region)
                & (metadata["particle_group"].astype(str) == normal_group)
            ].reset_index(drop=True)
            eventbuilder_rows = metadata.loc[
                (metadata["region"].astype(str) == region)
                & (metadata["particle_group"].astype(str) == eventbuilder_group)
            ].reset_index(drop=True)

            if normal_rows.empty or eventbuilder_rows.empty:
                rows_out.append(
                    {
                        "region": region,
                        "normal_particle_group": normal_group,
                        "eventbuilder_particle_group": eventbuilder_group,
                        "feature": "",
                        "status": "missing_metadata_rows",
                    }
                )
                continue

            for feature in timing_features:
                if feature not in features_cfg:
                    raise KeyError(f"Timing feature {feature!r} is not defined in hist config")

                normal_summary = summarize_feature(
                    rows=normal_rows,
                    tree_name=tree_name,
                    repo_root=repo_root,
                    feature_name=feature,
                    feature_cfg=features_cfg[feature],
                )
                eventbuilder_summary = summarize_feature(
                    rows=eventbuilder_rows,
                    tree_name=tree_name,
                    repo_root=repo_root,
                    feature_name=feature,
                    feature_cfg=features_cfg[feature],
                )

                rows_out.append(
                    {
                        "region": region,
                        "normal_particle_group": normal_group,
                        "eventbuilder_particle_group": eventbuilder_group,
                        "feature": feature,
                        "status": "ok",
                        "normal_selected_entries": normal_summary["selected_entries"],
                        "eventbuilder_selected_entries": eventbuilder_summary["selected_entries"],
                        "eventbuilder_over_normal_entries": ratio(
                            eventbuilder_summary["selected_entries"],
                            normal_summary["selected_entries"],
                        ),
                        "normal_mean": normal_summary["mean"],
                        "eventbuilder_mean": eventbuilder_summary["mean"],
                        "eventbuilder_minus_normal_mean": (
                            eventbuilder_summary["mean"] - normal_summary["mean"]
                            if math.isfinite(eventbuilder_summary["mean"]) and math.isfinite(normal_summary["mean"])
                            else float("nan")
                        ),
                        "normal_stddev": normal_summary["stddev"],
                        "eventbuilder_stddev": eventbuilder_summary["stddev"],
                        "normal_n_partitions": normal_summary["n_partitions"],
                        "eventbuilder_n_partitions": eventbuilder_summary["n_partitions"],
                        "normal_n_added_files": normal_summary["n_added_files"],
                        "eventbuilder_n_added_files": eventbuilder_summary["n_added_files"],
                        "normal_n_skipped_files": normal_summary["n_skipped_files"],
                        "eventbuilder_n_skipped_files": eventbuilder_summary["n_skipped_files"],
                        "feature_expression": normal_summary["feature_expression"],
                        "feature_cut": normal_summary["feature_cut"],
                        "normal_skipped_feature_partition_paths": normal_summary["skipped_feature_partition_paths"],
                        "eventbuilder_skipped_feature_partition_paths": eventbuilder_summary["skipped_feature_partition_paths"],
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "region",
        "normal_particle_group",
        "eventbuilder_particle_group",
        "feature",
        "status",
        "normal_selected_entries",
        "eventbuilder_selected_entries",
        "eventbuilder_over_normal_entries",
        "normal_mean",
        "eventbuilder_mean",
        "eventbuilder_minus_normal_mean",
        "normal_stddev",
        "eventbuilder_stddev",
        "normal_n_partitions",
        "eventbuilder_n_partitions",
        "normal_n_added_files",
        "eventbuilder_n_added_files",
        "normal_n_skipped_files",
        "eventbuilder_n_skipped_files",
        "feature_expression",
        "feature_cut",
        "normal_skipped_feature_partition_paths",
        "eventbuilder_skipped_feature_partition_paths",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Wrote EventBuilder veto timing summary: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare veto timing between normal MC and EventBuilder MC.")
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--hist-config", required=True, help="Histogram feature config YAML")
    parser.add_argument("--eval-config", required=True, help="Evaluation options config YAML")
    parser.add_argument("--output", required=True, help="Output summary CSV")
    parser.add_argument("--tree", default=None, help="Override input tree name")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative paths")
    make_summary(parser.parse_args())


if __name__ == "__main__":
    main()
