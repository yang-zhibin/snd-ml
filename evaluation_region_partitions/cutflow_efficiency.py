#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import pandas as pd
import ROOT
import yaml


ROOT.gROOT.SetBatch(True)

ALL_REGION = "ALL"
ALL_PARTICLE_GROUP = "ALL"
SUPPORTED_AGGREGATIONS = {
    "per_region_particle",
    "fiducial_merged_region_particle",
    "overall_particle",
    "veto_category_particle",
    "us_category_particle",
    "overall_all",
}
SUPPORTED_STUDY_MODES = {
    "sequential_cutflow",
    "independent_cuts",
}

OUTPUT_FIELDS = [
    "study",
    "aggregation_mode",
    "region",
    "particle_group",
    "cut_index",
    "cut_name",
    "cut_label",
    "applied",
    "applies_to",
    "skip_reason",
    "initial_count",
    "cumulative_count",
    "relative_count",
    "cumulative_efficiency",
    "relative_efficiency",
    "cumulative_efficiency_error",
    "relative_efficiency_error",
    "denominator_name",
    "denominator_expression",
    "cumulative_expression",
    "relative_expression",
    "cut_expression",
    "n_partitions",
    "n_added_files",
    "n_skipped_files",
    "lumi",
    "skipped_feature_partition_paths",
]

SUMMARY_BY_STUDY_FIELDS = [
    "study",
    "aggregation_mode",
    "region",
    "particle_group",
    "n_cuts",
    "n_applied_cuts",
    "initial_count",
    "final_cut_index",
    "final_cut_name",
    "final_cut_label",
    "final_cumulative_count",
    "final_cumulative_efficiency",
    "final_cumulative_efficiency_error",
    "denominator_name",
    "denominator_expression",
    "final_cumulative_expression",
    "n_partitions",
    "n_added_files",
    "n_skipped_files",
    "lumi",
]

EVENTBUILDER_COMPARISON_FIELDS = [
    "study",
    "aggregation_mode",
    "region",
    "base_particle_group",
    "eventbuilder_particle_group",
    "cut_index",
    "cut_name",
    "cut_label",
    "base_initial_count",
    "eventbuilder_initial_count",
    "base_count",
    "eventbuilder_count",
    "base_efficiency",
    "eventbuilder_efficiency",
    "efficiency_difference",
    "efficiency_ratio",
    "base_efficiency_error",
    "eventbuilder_efficiency_error",
]


def log_progress(message: str) -> None:
    print(f"[cutflow_efficiency] {message}", flush=True)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str, repo_root: str | Path) -> str:
    if path.startswith("root://") or os.path.isabs(path):
        return path
    return str(Path(repo_root) / path)


def clean_expression(expression: object) -> str:
    return " ".join(str(expression or "").split())


def combine_expressions(*expressions: object) -> str:
    selected = [clean_expression(expr) for expr in expressions if clean_expression(expr)]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return " && ".join(f"({expr})" for expr in selected)


def study_mode(study_cfg: dict) -> str:
    return str(study_cfg.get("mode", "sequential_cutflow"))


def binomial_error(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    efficiency = float(numerator) / float(denominator)
    return math.sqrt(max(efficiency * (1.0 - efficiency), 0.0) / float(denominator))


def efficiency(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def count_selection(rdf: ROOT.RDataFrame, expression: str) -> int:
    expression = clean_expression(expression)
    if not expression or expression == "1":
        return int(rdf.Count().GetValue())
    return int(rdf.Filter(expression).Count().GetValue())


def book_count_selection(
    rdf: ROOT.RDataFrame,
    expression: str,
    count_handles: dict[str, object],
) -> str:
    expression = clean_expression(expression)
    count_key = expression if expression and expression != "1" else "1"
    if count_key not in count_handles:
        if count_key == "1":
            count_handles[count_key] = rdf.Count()
        else:
            count_handles[count_key] = rdf.Filter(count_key).Count()
    return count_key


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
        start = time.monotonic()
        log_progress("building sndData and friend-tree indices on runId/eventId")
        chain.BuildIndex("runId", "eventId")
        friend_chain.BuildIndex("runId", "eventId")
        log_progress(f"finished friend-tree indices in {time.monotonic() - start:.1f}s")
    finally:
        ROOT.gErrorIgnoreLevel = old_error_level


def is_real_data_rows(rows: pd.DataFrame) -> pd.Series:
    particle_group = rows.get("particle_group", pd.Series(dtype=str)).astype(str)
    particle_family = rows.get("particle_family", pd.Series(dtype=str)).astype(str)
    return (particle_group == "real_data") | (particle_family == "real_data")


def group_composition(rows: pd.DataFrame) -> str:
    real_mask = is_real_data_rows(rows)
    has_real = bool(real_mask.any())
    has_mc = bool((~real_mask).any())
    if has_real and has_mc:
        return "mixed"
    if has_real:
        return "real_data"
    return "mc"


def cut_applies(applies_to: str, rows: pd.DataFrame) -> tuple[bool, str]:
    applies_to = str(applies_to or "all")
    composition = group_composition(rows)

    if applies_to == "all":
        return True, ""
    if applies_to == "real_data_only":
        if composition == "real_data":
            return True, ""
        if composition == "mixed":
            return False, "mixed_real_data_and_mc"
        return False, "real_data_only_cut_on_mc"
    if applies_to == "mc_only":
        if composition == "mc":
            return True, ""
        if composition == "mixed":
            return False, "mixed_real_data_and_mc"
        return False, "mc_only_cut_on_real_data"

    raise ValueError(f"Unknown applies_to value: {applies_to}")


def select_eval_rows(metadata: pd.DataFrame, eval_cfg: dict) -> pd.DataFrame:
    required = {"region", "particle_group", "feature_partition_path"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    allowed_regions = set(str(v) for v in eval_cfg.get("comparisons", {}).get("regions", []))
    allowed_particles = set(str(v) for v in eval_cfg.get("comparisons", {}).get("particle_groups", []))

    rows = metadata.copy()
    if allowed_regions:
        rows = rows.loc[rows["region"].astype(str).isin(allowed_regions)]
    if allowed_particles:
        rows = rows.loc[rows["particle_group"].astype(str).isin(allowed_particles)]

    if "feature_available" in rows.columns:
        feature_available = rows["feature_available"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        rows = rows.loc[feature_available]

    return rows.reset_index(drop=True)


def deduplicate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    subset = [
        column
        for column in ("partition_id", "feature_partition_path")
        if column in rows.columns
    ]
    if not subset:
        return rows.reset_index(drop=True)
    return rows.drop_duplicates(subset=subset).reset_index(drop=True)


def veto_category(region: object) -> str:
    region = str(region)
    if "_no_veto_" in region:
        return "no_veto"
    if "_has_veto_" in region:
        return "has_veto"
    return "unknown_veto"


def us_category(region: object) -> str:
    region = str(region)
    if "_has_us_" in region:
        return "has_us"
    if "_no_us_" in region:
        return "no_us"
    return "unknown_us"


def fiducial_merged_region(region: object) -> str:
    region = str(region)
    if "_no_veto_" in region and "_has_us_" in region:
        return "merged_01_signal_sb2_no_veto_has_us"
    if "_no_veto_" in region and "_no_us_" in region:
        return "merged_02_sb1_sb3_no_veto_no_us"
    if "_has_veto_" in region and "_has_us_" in region:
        return "merged_03_sb4_sb6_has_veto_has_us"
    if "_has_veto_" in region and "_no_us_" in region:
        return "merged_04_sb5_sb7_has_veto_no_us"
    return "merged_unknown"


def build_aggregation_groups(
    rows: pd.DataFrame,
    aggregation_mode: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    if aggregation_mode == "per_region_particle":
        groups = []
        for (region, particle_group), group_rows in rows.groupby(
            ["region", "particle_group"], sort=True
        ):
            groups.append(
                (
                    str(region),
                    str(particle_group),
                    deduplicate_rows(group_rows),
                )
        )
        return groups

    if aggregation_mode == "fiducial_merged_region_particle":
        rows = rows.copy()
        rows["_fiducial_merged_region"] = rows["region"].map(fiducial_merged_region)
        groups = []
        for (region, particle_group), group_rows in rows.groupby(
            ["_fiducial_merged_region", "particle_group"], sort=True
        ):
            groups.append(
                (
                    str(region),
                    str(particle_group),
                    deduplicate_rows(group_rows.drop(columns=["_fiducial_merged_region"])),
                )
            )
        return groups

    if aggregation_mode == "overall_particle":
        groups = []
        for particle_group, group_rows in rows.groupby("particle_group", sort=True):
            groups.append(
                (
                    ALL_REGION,
                    str(particle_group),
                    deduplicate_rows(group_rows),
                )
            )
        return groups

    if aggregation_mode == "veto_category_particle":
        rows = rows.copy()
        rows["_veto_category"] = rows["region"].map(veto_category)
        groups = []
        for (category, particle_group), group_rows in rows.groupby(
            ["_veto_category", "particle_group"], sort=True
        ):
            groups.append(
                (
                    str(category),
                    str(particle_group),
                    deduplicate_rows(group_rows.drop(columns=["_veto_category"])),
                )
            )
        return groups

    if aggregation_mode == "us_category_particle":
        rows = rows.copy()
        rows["_us_category"] = rows["region"].map(us_category)
        groups = []
        for (category, particle_group), group_rows in rows.groupby(
            ["_us_category", "particle_group"], sort=True
        ):
            groups.append(
                (
                    str(category),
                    str(particle_group),
                    deduplicate_rows(group_rows.drop(columns=["_us_category"])),
                )
            )
        return groups

    if aggregation_mode == "overall_all":
        return [(ALL_REGION, ALL_PARTICLE_GROUP, deduplicate_rows(rows))]

    raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")


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
    start = time.monotonic()
    log_progress(
        f"building chain tree={tree_name}"
        + (f" friend={friend_tree_name}" if friend_tree_name else "")
        + f" from {len(rows)} metadata rows"
    )

    for row_index, (_, row) in enumerate(rows.iterrows(), start=1):
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
        if row_index == 1 or row_index % 10 == 0 or row_index == len(rows):
            log_progress(
                f"chain progress {row_index}/{len(rows)} rows, "
                f"added={len(added_paths)}, skipped={len(skipped_paths)}"
            )

    if friend_chain is not None and added_paths:
        build_friend_indices(chain, friend_chain)
        chain.AddFriend(friend_chain)

    log_progress(
        f"finished chain in {time.monotonic() - start:.1f}s: "
        f"added={len(added_paths)}, skipped={len(skipped_paths)}"
    )
    return chain, added_paths, skipped_paths


def sum_numeric_column(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return 0.0
    return float(pd.to_numeric(rows[column], errors="coerce").fillna(0.0).sum())


def enabled_studies(cutflow_cfg: dict, only_study: str | None = None) -> dict:
    studies = cutflow_cfg.get("studies", {})
    if not studies:
        raise ValueError("cutflow config has no studies")

    selected = {}
    for study_name, study_cfg in studies.items():
        study_name = str(study_name)
        if only_study and study_name != only_study:
            continue
        if not study_cfg.get("enabled", True):
            continue
        selected[study_name] = study_cfg

    if only_study and not selected:
        raise ValueError(f"Requested study {only_study!r} is not enabled or not defined")
    if not selected:
        raise ValueError("No enabled cutflow efficiency studies selected")

    return selected


def validate_cutflow_config(cutflow_cfg: dict) -> None:
    cut_definitions = cutflow_cfg.get("cuts", {}).get("definitions", {})
    if not cut_definitions:
        raise ValueError("cutflow config cuts.definitions is empty")

    defaults = cutflow_cfg.get("defaults", {})
    default_denominator = defaults.get("denominator", {})
    if "expression" not in default_denominator:
        raise KeyError("cutflow defaults.denominator.expression is required")

    for study_name, study_cfg in cutflow_cfg.get("studies", {}).items():
        if not study_cfg.get("enabled", True):
            continue

        cuts = study_cfg.get("cuts", [])
        if not cuts:
            raise ValueError(f"cutflow study {study_name!r} has no cuts")

        mode = study_mode(study_cfg)
        if mode not in SUPPORTED_STUDY_MODES:
            raise ValueError(
                f"cutflow study {study_name!r} has unsupported mode {mode!r}"
            )

        for cut_name in cuts:
            if str(cut_name) not in cut_definitions:
                raise KeyError(
                    f"cutflow study {study_name!r} references undefined cut {cut_name!r}"
                )

        denominator_cfg = study_cfg.get(
            "denominator",
            defaults.get("denominator", {}),
        )
        for cut_name in denominator_cfg.get("cuts", []):
            if str(cut_name) not in cut_definitions:
                raise KeyError(
                    f"cutflow study {study_name!r} denominator references "
                    f"undefined cut {cut_name!r}"
                )

        aggregation_modes = study_cfg.get(
            "aggregation_modes",
            defaults.get("aggregation_modes", []),
        )
        for aggregation_mode in aggregation_modes:
            if str(aggregation_mode) not in SUPPORTED_AGGREGATIONS:
                raise ValueError(
                    f"cutflow study {study_name!r} has unsupported aggregation mode "
                    f"{aggregation_mode!r}"
                )


def build_denominator_expression(
    study_cfg: dict,
    cutflow_cfg: dict,
    rows: pd.DataFrame,
) -> tuple[str, str, list[str], list[str]]:
    cut_definitions = cutflow_cfg["cuts"]["definitions"]
    denominator_cfg = study_cfg.get(
        "denominator",
        cutflow_cfg.get("defaults", {}).get("denominator", {}),
    )
    denominator_name = str(denominator_cfg.get("name", "denominator"))
    denominator_expr = clean_expression(denominator_cfg.get("expression", "1"))
    expressions = [denominator_expr]
    applied_cuts = []
    skipped_cuts = []

    for cut_name in denominator_cfg.get("cuts", []):
        cut_name = str(cut_name)
        cut_cfg = cut_definitions[cut_name]
        applies, skip_reason = cut_applies(str(cut_cfg.get("applies_to", "all")), rows)
        if applies:
            cut_expr = clean_expression(cut_cfg.get("expression", ""))
            if not cut_expr:
                raise ValueError(f"Denominator cut {cut_name!r} has an empty expression")
            expressions.append(cut_expr)
            applied_cuts.append(cut_name)
        else:
            skipped_cuts.append(f"{cut_name}:{skip_reason}")

    return (
        denominator_name,
        combine_expressions(*expressions),
        applied_cuts,
        skipped_cuts,
    )


def zero_rows_for_group(
    study_name: str,
    study_cfg: dict,
    cutflow_cfg: dict,
    aggregation_mode: str,
    region: str,
    particle_group: str,
    rows: pd.DataFrame,
    skipped_paths: list[str],
    status_reason: str,
) -> list[dict]:
    cut_definitions = cutflow_cfg["cuts"]["definitions"]
    denominator_name, denominator_expr, _, _ = build_denominator_expression(
        study_cfg,
        cutflow_cfg,
        rows,
    )

    output_rows = []
    for cut_index, cut_name in enumerate(study_cfg.get("cuts", []), start=1):
        cut_cfg = cut_definitions[str(cut_name)]
        cut_expr = clean_expression(cut_cfg.get("expression", ""))
        output_rows.append(
            {
                "study": study_name,
                "aggregation_mode": aggregation_mode,
                "region": region,
                "particle_group": particle_group,
                "cut_index": cut_index,
                "cut_name": str(cut_name),
                "cut_label": str(cut_cfg.get("label", cut_name)),
                "applied": False,
                "applies_to": str(cut_cfg.get("applies_to", "all")),
                "skip_reason": status_reason,
                "initial_count": 0,
                "cumulative_count": 0,
                "relative_count": 0,
                "cumulative_efficiency": 0.0,
                "relative_efficiency": 0.0,
                "cumulative_efficiency_error": 0.0,
                "relative_efficiency_error": 0.0,
                "denominator_name": denominator_name,
                "denominator_expression": denominator_expr,
                "cumulative_expression": "",
                "relative_expression": "",
                "cut_expression": cut_expr,
                "n_partitions": int(len(rows)),
                "n_added_files": 0,
                "n_skipped_files": int(len(skipped_paths)),
                "lumi": sum_numeric_column(rows, "lumi_per_partition"),
                "skipped_feature_partition_paths": "\n".join(skipped_paths),
            }
        )
    return output_rows


def summarize_group(
    study_name: str,
    study_cfg: dict,
    cutflow_cfg: dict,
    aggregation_mode: str,
    region: str,
    particle_group: str,
    rows: pd.DataFrame,
    tree_name: str,
    friend_tree_name: str | None,
    repo_root: str | Path,
) -> list[dict]:
    chain, added_paths, skipped_paths = build_chain(
        rows,
        tree_name,
        repo_root,
        friend_tree_name=friend_tree_name,
    )
    if not added_paths:
        status_reason = "no_feature_files_added"
        if friend_tree_name and any(f"missing tree {friend_tree_name}" in path for path in skipped_paths):
            raise RuntimeError(
                f"No usable feature partition files for {region}/{particle_group}: "
                f"required friend tree {friend_tree_name!r} is missing. "
                "Regenerate region partitions with cutFlowSummary preservation enabled."
            )
        return zero_rows_for_group(
            study_name=study_name,
            study_cfg=study_cfg,
            cutflow_cfg=cutflow_cfg,
            aggregation_mode=aggregation_mode,
            region=region,
            particle_group=particle_group,
            rows=rows,
            skipped_paths=skipped_paths,
            status_reason=status_reason,
        )

    cut_definitions = cutflow_cfg["cuts"]["definitions"]
    denominator_cfg = study_cfg.get(
        "denominator",
        cutflow_cfg.get("defaults", {}).get("denominator", {}),
    )
    denominator_name, denominator_expr, _, _ = build_denominator_expression(
        study_cfg,
        cutflow_cfg,
        rows,
    )

    rdf = ROOT.RDataFrame(chain)
    count_handles = {}
    initial_count_key = book_count_selection(rdf, denominator_expr, count_handles)
    current_cumulative_expr = denominator_expr
    output_specs = []
    mode = study_mode(study_cfg)

    for cut_index, cut_name in enumerate(study_cfg.get("cuts", []), start=1):
        cut_name = str(cut_name)
        cut_cfg = cut_definitions[cut_name]
        cut_expr = clean_expression(cut_cfg.get("expression", ""))
        applies_to = str(cut_cfg.get("applies_to", "all"))
        applies, skip_reason = cut_applies(applies_to, rows)

        if applies:
            if not cut_expr:
                raise ValueError(f"Cut {cut_name!r} has an empty expression")
            relative_expr = combine_expressions(denominator_expr, cut_expr)
            relative_count_key = book_count_selection(rdf, relative_expr, count_handles)
            if mode == "independent_cuts":
                cumulative_expr = relative_expr
                cumulative_count_key = relative_count_key
            else:
                cumulative_expr = combine_expressions(current_cumulative_expr, cut_expr)
                cumulative_count_key = book_count_selection(
                    rdf,
                    cumulative_expr,
                    count_handles,
                )
                current_cumulative_expr = cumulative_expr
        else:
            cumulative_expr = current_cumulative_expr
            relative_expr = denominator_expr
            cumulative_count_key = book_count_selection(
                rdf,
                cumulative_expr,
                count_handles,
            )
            relative_count_key = initial_count_key

        output_specs.append(
            {
                "study": study_name,
                "aggregation_mode": aggregation_mode,
                "region": region,
                "particle_group": particle_group,
                "cut_index": cut_index,
                "cut_name": cut_name,
                "cut_label": str(cut_cfg.get("label", cut_name)),
                "applied": bool(applies),
                "applies_to": applies_to,
                "skip_reason": skip_reason,
                "denominator_name": denominator_name,
                "denominator_expression": denominator_expr,
                "cumulative_expression": cumulative_expr,
                "relative_expression": relative_expr,
                "cut_expression": cut_expr,
                "_initial_count_key": initial_count_key,
                "_cumulative_count_key": cumulative_count_key,
                "_relative_count_key": relative_count_key,
                "n_partitions": int(len(rows)),
                "n_added_files": int(len(added_paths)),
                "n_skipped_files": int(len(skipped_paths)),
                "lumi": sum_numeric_column(rows, "lumi_per_partition"),
                "skipped_feature_partition_paths": "\n".join(skipped_paths),
            }
        )

    count_values = {key: int(handle.GetValue()) for key, handle in count_handles.items()}

    output_rows = []
    for spec in output_specs:
        initial_count = count_values[spec.pop("_initial_count_key")]
        cumulative_count = count_values[spec.pop("_cumulative_count_key")]
        relative_count = count_values[spec.pop("_relative_count_key")]
        spec.update(
            {
                "initial_count": int(initial_count),
                "cumulative_count": int(cumulative_count),
                "relative_count": int(relative_count),
                "cumulative_efficiency": efficiency(cumulative_count, initial_count),
                "relative_efficiency": efficiency(relative_count, initial_count),
                "cumulative_efficiency_error": binomial_error(cumulative_count, initial_count),
                "relative_efficiency_error": binomial_error(relative_count, initial_count),
            }
        )
        output_rows.append(spec)

    return output_rows


def summarize_by_study(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    summary_rows = []
    data = pd.DataFrame(rows)
    group_columns = ["study", "aggregation_mode", "region", "particle_group"]
    for group_key, group_rows in data.groupby(group_columns, sort=True):
        group_rows = group_rows.sort_values("cut_index")
        final_row = group_rows.iloc[-1]
        study, aggregation_mode, region, particle_group = group_key
        summary_rows.append(
            {
                "study": study,
                "aggregation_mode": aggregation_mode,
                "region": region,
                "particle_group": particle_group,
                "n_cuts": int(len(group_rows)),
                "n_applied_cuts": int(group_rows["applied"].astype(bool).sum()),
                "initial_count": int(final_row["initial_count"]),
                "final_cut_index": int(final_row["cut_index"]),
                "final_cut_name": str(final_row["cut_name"]),
                "final_cut_label": str(final_row["cut_label"]),
                "final_cumulative_count": int(final_row["cumulative_count"]),
                "final_cumulative_efficiency": float(final_row["cumulative_efficiency"]),
                "final_cumulative_efficiency_error": float(
                    final_row["cumulative_efficiency_error"]
                ),
                "denominator_name": str(final_row["denominator_name"]),
                "denominator_expression": str(final_row["denominator_expression"]),
                "final_cumulative_expression": str(final_row["cumulative_expression"]),
                "n_partitions": int(final_row["n_partitions"]),
                "n_added_files": int(final_row["n_added_files"]),
                "n_skipped_files": int(final_row["n_skipped_files"]),
                "lumi": float(final_row["lumi"]),
            }
        )
    return summary_rows


def filter_summary_rows(
    rows: list[dict],
    aggregation_modes: set[str],
    study_names: set[str] | None = None,
) -> list[dict]:
    selected = []
    for row in rows:
        if str(row["aggregation_mode"]) not in aggregation_modes:
            continue
        if study_names and str(row["study"]) not in study_names:
            continue
        selected.append(row)
    return selected


def convenience_study_names(cutflow_cfg: dict) -> set[str]:
    configured = cutflow_cfg.get("outputs", {}).get("convenience_studies", [])
    if configured:
        return {str(name) for name in configured}
    if "fiducial_cut_comparison" in cutflow_cfg.get("studies", {}):
        return {"fiducial_cut_comparison"}
    return set()


def eventbuilder_base_particle(particle_group: object) -> str | None:
    particle_group = str(particle_group)
    suffix = "_EventBuilder"
    if not particle_group.endswith(suffix):
        return None
    return particle_group[: -len(suffix)]


def make_eventbuilder_comparison(rows: list[dict], study_names: set[str]) -> list[dict]:
    if not rows:
        return []

    comparison_rows = []
    data = pd.DataFrame(rows)
    if study_names:
        data = data.loc[data["study"].astype(str).isin(study_names)]

    lookup_columns = ["study", "aggregation_mode", "region", "particle_group", "cut_name"]
    lookup = {
        tuple(str(row[column]) for column in lookup_columns): row
        for _, row in data.iterrows()
    }

    for _, eventbuilder_row in data.iterrows():
        eventbuilder_particle = str(eventbuilder_row["particle_group"])
        base_particle = eventbuilder_base_particle(eventbuilder_particle)
        if not base_particle:
            continue

        key = (
            str(eventbuilder_row["study"]),
            str(eventbuilder_row["aggregation_mode"]),
            str(eventbuilder_row["region"]),
            base_particle,
            str(eventbuilder_row["cut_name"]),
        )
        base_row = lookup.get(key)
        if base_row is None:
            continue

        base_eff = float(base_row["relative_efficiency"])
        eventbuilder_eff = float(eventbuilder_row["relative_efficiency"])
        ratio = eventbuilder_eff / base_eff if base_eff else 0.0
        comparison_rows.append(
            {
                "study": str(eventbuilder_row["study"]),
                "aggregation_mode": str(eventbuilder_row["aggregation_mode"]),
                "region": str(eventbuilder_row["region"]),
                "base_particle_group": base_particle,
                "eventbuilder_particle_group": eventbuilder_particle,
                "cut_index": int(eventbuilder_row["cut_index"]),
                "cut_name": str(eventbuilder_row["cut_name"]),
                "cut_label": str(eventbuilder_row["cut_label"]),
                "base_initial_count": int(base_row["initial_count"]),
                "eventbuilder_initial_count": int(eventbuilder_row["initial_count"]),
                "base_count": int(base_row["relative_count"]),
                "eventbuilder_count": int(eventbuilder_row["relative_count"]),
                "base_efficiency": base_eff,
                "eventbuilder_efficiency": eventbuilder_eff,
                "efficiency_difference": eventbuilder_eff - base_eff,
                "efficiency_ratio": ratio,
                "base_efficiency_error": float(base_row["relative_efficiency_error"]),
                "eventbuilder_efficiency_error": float(
                    eventbuilder_row["relative_efficiency_error"]
                ),
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            row["study"],
            row["aggregation_mode"],
            row["region"],
            row["base_particle_group"],
            row["cut_index"],
        )
    )
    return comparison_rows


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def write_summary_outputs(args: argparse.Namespace, rows_out: list[dict], cutflow_cfg: dict) -> None:
    write_csv(args.output, rows_out, OUTPUT_FIELDS)
    convenience_studies = convenience_study_names(cutflow_cfg)

    if args.summary_by_study:
        write_csv(
            args.summary_by_study,
            summarize_by_study(rows_out),
            SUMMARY_BY_STUDY_FIELDS,
        )
        print(f"Wrote cutflow efficiency study summary: {args.summary_by_study}")

    convenience_outputs = [
        (
            args.regional_cut_efficiency,
            {"per_region_particle", "fiducial_merged_region_particle"},
            "regional cut efficiency",
        ),
        (
            args.overall_cut_efficiency,
            {"overall_particle", "overall_all"},
            "overall cut efficiency",
        ),
        (
            args.veto_cut_efficiency,
            {"veto_category_particle"},
            "veto-category cut efficiency",
        ),
        (
            args.us_cut_efficiency,
            {"us_category_particle"},
            "US-category cut efficiency",
        ),
    ]
    for output, aggregation_modes, label in convenience_outputs:
        if not output:
            continue
        write_csv(
            output,
            filter_summary_rows(rows_out, aggregation_modes, convenience_studies),
            OUTPUT_FIELDS,
        )
        print(f"Wrote {label}: {output}")

    if args.eventbuilder_comparison:
        write_csv(
            args.eventbuilder_comparison,
            make_eventbuilder_comparison(rows_out, convenience_studies),
            EVENTBUILDER_COMPARISON_FIELDS,
        )
        print(f"Wrote EventBuilder comparison: {args.eventbuilder_comparison}")

    print(f"Wrote cutflow efficiency summary: {args.output}")


def make_summary(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    cutflow_config = resolve_path(args.cutflow_config, repo_root)
    cutflow_cfg = load_yaml(cutflow_config)
    validate_cutflow_config(cutflow_cfg)

    if args.merge_input:
        write_summary_outputs(args, read_csv_rows(args.merge_input), cutflow_cfg)
        return

    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    eval_config = resolve_path(args.eval_config, repo_root)

    metadata = pd.read_csv(metadata_csv)
    eval_cfg = load_yaml(eval_config)

    tree_name = (
        args.tree
        or cutflow_cfg.get("metadata", {}).get("tree_name")
        or eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    )
    friend_tree_name = args.friend_tree or cutflow_cfg.get("metadata", {}).get(
        "friend_tree_name",
        "cutFlowSummary",
    )
    selected_rows = select_eval_rows(metadata, eval_cfg)
    if selected_rows.empty:
        raise RuntimeError("No region-evaluation metadata rows selected")

    rows_out = []
    for study_name, study_cfg in enabled_studies(cutflow_cfg, args.study).items():
        aggregation_modes = study_cfg.get(
            "aggregation_modes",
            cutflow_cfg.get("defaults", {}).get("aggregation_modes", []),
        )
        for aggregation_mode in aggregation_modes:
            aggregation_mode = str(aggregation_mode)
            if args.aggregation_mode and aggregation_mode != args.aggregation_mode:
                continue
            groups = build_aggregation_groups(selected_rows, aggregation_mode)
            for region, particle_group, group_rows in groups:
                if args.group_region and str(region) != str(args.group_region):
                    continue
                if args.group_particle and str(particle_group) != str(args.group_particle):
                    continue
                rows_out.extend(
                    summarize_group(
                        study_name=study_name,
                        study_cfg=study_cfg,
                        cutflow_cfg=cutflow_cfg,
                        aggregation_mode=aggregation_mode,
                        region=region,
                        particle_group=particle_group,
                        rows=group_rows,
                        tree_name=tree_name,
                        friend_tree_name=friend_tree_name,
                        repo_root=repo_root,
                    )
                )

    if not rows_out and (args.aggregation_mode or args.group_region or args.group_particle):
        raise RuntimeError(
            "No cutflow rows were produced for the requested filters: "
            f"study={args.study!r}, aggregation_mode={args.aggregation_mode!r}, "
            f"group_region={args.group_region!r}, group_particle={args.group_particle!r}"
        )

    write_summary_outputs(args, rows_out, cutflow_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize region cutflow efficiencies.")
    parser.add_argument("--metadata-csv", required=True, help="Region partition metadata CSV")
    parser.add_argument("--eval-config", required=True, help="Region evaluation options YAML")
    parser.add_argument("--cutflow-config", required=True, help="Cutflow efficiency config YAML")
    parser.add_argument("--output", required=True, help="Output summary CSV")
    parser.add_argument(
        "--summary-by-study",
        default=None,
        help="Optional compact output CSV with one final row per study/group",
    )
    parser.add_argument(
        "--regional-cut-efficiency",
        default=None,
        help="Optional per-region cut efficiency CSV",
    )
    parser.add_argument(
        "--overall-cut-efficiency",
        default=None,
        help="Optional all-region cut efficiency CSV",
    )
    parser.add_argument(
        "--veto-cut-efficiency",
        default=None,
        help="Optional veto/no-veto category cut efficiency CSV",
    )
    parser.add_argument(
        "--us-cut-efficiency",
        default=None,
        help="Optional US/no-US category cut efficiency CSV",
    )
    parser.add_argument(
        "--eventbuilder-comparison",
        default=None,
        help="Optional standard-vs-EventBuilder comparison CSV",
    )
    parser.add_argument("--tree", default=None, help="Override input tree name")
    parser.add_argument("--friend-tree", default=None, help="Optional friend tree name")
    parser.add_argument("--study", default=None, help="Run only one enabled study")
    parser.add_argument("--aggregation-mode", default=None, help="Run only one aggregation mode")
    parser.add_argument("--group-region", default=None, help="Run only one aggregation group region/category")
    parser.add_argument("--group-particle", default=None, help="Run only one aggregation group particle")
    parser.add_argument(
        "--merge-input",
        action="append",
        default=[],
        help="Shard CSV to merge. Can be repeated.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for relative paths")
    make_summary(parser.parse_args())


if __name__ == "__main__":
    main()
