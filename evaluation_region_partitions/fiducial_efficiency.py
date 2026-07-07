#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
from itertools import combinations
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


def count_selection(rdf: ROOT.RDataFrame, expression: str) -> int:
    expression = clean_expression(expression)
    if not expression:
        return int(rdf.Count().GetValue())
    return int(rdf.Filter(expression).Count().GetValue())


def binomial_error(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    efficiency = float(numerator) / float(denominator)
    return math.sqrt(max(efficiency * (1.0 - efficiency), 0.0) / float(denominator))


def select_eval_groups(metadata: pd.DataFrame, eval_cfg: dict) -> list[tuple[str, str, pd.DataFrame]]:
    allowed_regions = set(str(v) for v in eval_cfg.get("comparisons", {}).get("regions", []))
    allowed_particles = set(str(v) for v in eval_cfg.get("comparisons", {}).get("particle_groups", []))

    rows = metadata.copy()
    if allowed_regions:
        rows = rows.loc[rows["region"].astype(str).isin(allowed_regions)]
    if allowed_particles:
        rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]

    groups = []
    for (region, particle_group), group_rows in rows.groupby(["region", "particle_group"], sort=True):
        groups.append((str(region), str(particle_group), group_rows.reset_index(drop=True)))
    return groups


def make_summary(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    metadata = pd.read_csv(metadata_csv)
    eval_cfg = load_yaml(eval_config)
    study_cfg = eval_cfg.get("fiducial_study", {})
    if not study_cfg:
        raise KeyError("fiducial_study section is missing from the evaluation config")

    tree_name = args.tree or eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    friend_tree_name = args.friend_tree or eval_cfg.get("metadata", {}).get("friend_tree_name")
    denominator_cfg = study_cfg["denominator"]
    denominator_name = str(denominator_cfg["name"])
    denominator_expr = clean_expression(denominator_cfg["expression"])
    definitions = {
        str(name): clean_expression(cfg["expression"])
        for name, cfg in study_cfg.get("definitions", {}).items()
    }
    if not definitions:
        raise ValueError("fiducial_study.definitions is empty")

    summary_rows = []
    overlap_rows = []

    for region, particle_group, rows in select_eval_groups(metadata, eval_cfg):
        chain, added_paths, skipped_paths = build_chain(
            rows,
            tree_name,
            repo_root,
            friend_tree_name=friend_tree_name,
        )
        rdf = ROOT.RDataFrame(chain)

        denominator_count = count_selection(rdf, denominator_expr)
        pass_counts = {}
        for name, expression in definitions.items():
            numerator_expr = f"({denominator_expr}) && ({expression})"
            numerator = count_selection(rdf, numerator_expr)
            pass_counts[name] = numerator
            efficiency = float(numerator) / float(denominator_count) if denominator_count > 0 else 0.0
            summary_rows.append(
                {
                    "region": region,
                    "particle_group": particle_group,
                    "denominator_name": denominator_name,
                    "definition": name,
                    "n_partitions": len(rows),
                    "n_added_files": len(added_paths),
                    "n_skipped_files": len(skipped_paths),
                    "denominator": denominator_count,
                    "numerator": numerator,
                    "efficiency": efficiency,
                    "efficiency_error": binomial_error(numerator, denominator_count),
                    "denominator_expression": denominator_expr,
                    "definition_expression": expression,
                    "skipped_feature_partition_paths": "\n".join(skipped_paths),
                }
            )

        for first, second in combinations(definitions.keys(), 2):
            first_expr = definitions[first]
            second_expr = definitions[second]
            both = count_selection(rdf, f"({denominator_expr}) && ({first_expr}) && ({second_expr})")
            first_only = count_selection(rdf, f"({denominator_expr}) && ({first_expr}) && !({second_expr})")
            second_only = count_selection(rdf, f"({denominator_expr}) && !({first_expr}) && ({second_expr})")
            neither = count_selection(rdf, f"({denominator_expr}) && !({first_expr}) && !({second_expr})")
            overlap_rows.append(
                {
                    "region": region,
                    "particle_group": particle_group,
                    "denominator_name": denominator_name,
                    "first_definition": first,
                    "second_definition": second,
                    "denominator": denominator_count,
                    "first_pass": pass_counts[first],
                    "second_pass": pass_counts[second],
                    "both_pass": both,
                    "first_only": first_only,
                    "second_only": second_only,
                    "neither_pass": neither,
                    "both_fraction": float(both) / float(denominator_count) if denominator_count > 0 else 0.0,
                    "first_only_fraction": float(first_only) / float(denominator_count) if denominator_count > 0 else 0.0,
                    "second_only_fraction": float(second_only) / float(denominator_count) if denominator_count > 0 else 0.0,
                    "neither_fraction": float(neither) / float(denominator_count) if denominator_count > 0 else 0.0,
                }
            )

    summary_path = Path(args.summary)
    overlap_path = Path(args.overlap)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_path.parent.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "region",
        "particle_group",
        "denominator_name",
        "definition",
        "n_partitions",
        "n_added_files",
        "n_skipped_files",
        "denominator",
        "numerator",
        "efficiency",
        "efficiency_error",
        "denominator_expression",
        "definition_expression",
        "skipped_feature_partition_paths",
    ]
    overlap_fields = [
        "region",
        "particle_group",
        "denominator_name",
        "first_definition",
        "second_definition",
        "denominator",
        "first_pass",
        "second_pass",
        "both_pass",
        "first_only",
        "second_only",
        "neither_pass",
        "both_fraction",
        "first_only_fraction",
        "second_only_fraction",
        "neither_fraction",
    ]

    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(overlap_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overlap_fields)
        writer.writeheader()
        writer.writerows(overlap_rows)

    print(f"Wrote fiducial efficiency summary: {summary_path}")
    print(f"Wrote fiducial overlap summary: {overlap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SciFi fiducial efficiencies and overlaps.")
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--eval-config", required=True, help="Evaluation options config YAML")
    parser.add_argument("--summary", required=True, help="Output efficiency summary CSV")
    parser.add_argument("--overlap", required=True, help="Output overlap summary CSV")
    parser.add_argument("--tree", default=None, help="Override input tree name")
    parser.add_argument("--friend-tree", default=None, help="Override optional friend tree name")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative paths")
    make_summary(parser.parse_args())


if __name__ == "__main__":
    main()
