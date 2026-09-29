#!/usr/bin/env python3
"""Validate and standardize every configured V5 upward-muon MC feature file."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import ROOT
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))
from make_passing_muon_binary_region_hist_v5 import (  # noqa: E402
    configured_selections,
    define_shared_columns,
)


IGNORED_TOKENS = {
    "std", "isfinite", "abs", "max", "min", "int", "true", "false",
}
BASE_BRANCHES = {
    "pdgCode", "count_veto", "count_scifi", "count_us", "count_ds",
    *(f"count_veto{i}" for i in range(1, 4)),
    *(f"count_scifi{i}" for i in range(1, 6)),
    *(f"count_us{i}" for i in range(1, 6)),
    *(f"count_ds{i}" for i in range(1, 5)),
    "scifi_n_active_stations_xy",
}


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def full_feature_path(row):
    path = Path(str(row["feature_path"]))
    return path if path.is_absolute() else Path(str(row["output_base_path"])) / path


def configured_sources(metadata, config):
    mc_config = config["mc_muon_up"]
    productions = mc_config["productions"]
    partition_to_key = {
        str(definition["source_partition"]): key
        for key, definition in productions.items()
    }
    if len(partition_to_key) != len(productions):
        raise ValueError("Each MC production must have a unique source_partition")
    selected = metadata.loc[
        metadata["data_type"].astype(str).eq("MC_muon")
        & metadata["subfolder"].astype(str).eq(str(mc_config["subfolder"]))
        & metadata["partition"].astype(str).isin(partition_to_key)
    ].copy()
    selected["production_key"] = selected["partition"].astype(str).map(
        partition_to_key
    )
    expected = int(mc_config.get("expected_source_files", len(selected)))
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} configured MC-muon-up source rows, found {len(selected)}"
        )
    found = set(selected["production_key"])
    missing = set(productions) - found
    if missing:
        raise RuntimeError(f"No source rows for MC productions: {sorted(missing)}")
    return selected.sort_index()


def expression_tokens(expressions):
    return {
        token
        for expression in expressions
        for token in re.findall(r"\b[A-Za-z_]\w*\b", str(expression))
    }


def required_branches(hist_config, config):
    definitions = hist_config.get("definitions", {})
    derived = config.get("derived_columns", {})
    expressions = [
        *definitions.values(),
        *derived.values(),
        config["mc_muon_up"]["baseline_expression"],
    ]
    features = {
        **hist_config.get("features", {}),
        **config.get("diagnostic_features", {}),
    }
    for feature in features.values():
        expressions.extend(
            feature[field]
            for field in (
                "expression", "x_expression", "y_expression", "feature_cut"
            )
            if feature.get(field)
        )
    for selection in configured_selections(config).values():
        expressions.extend(selection["components"].values())
    return (
        expression_tokens(expressions)
        - set(definitions)
        - set(derived)
        - IGNORED_TOKENS
        | BASE_BRANCHES
    )


def inspect_file(path, tree_name, required):
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open MC-muon-up feature file {path}")
    try:
        tree = source.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Missing tree {tree_name!r} in {path}")
        entries = int(tree.GetEntries())
        if entries <= 0:
            raise RuntimeError(f"Empty tree {tree_name!r} in {path}")
        branches = {branch.GetName() for branch in tree.GetListOfBranches()}
        missing = sorted(required - branches)
        if missing:
            raise RuntimeError(f"{path} lacks required branches: {missing}")
        return entries
    finally:
        source.Close()


def event_counts(path, tree_name, config):
    frame = ROOT.RDataFrame(tree_name, str(path))
    frame = define_shared_columns(frame, config["derived_columns"])
    muons = frame.Filter("std::abs(pdgCode) == 13", "truth_muon")
    actions = {
        "inclusive_muon_events": muons.Count(),
        "detector_activity_events": muons.Filter(
            "(count_veto + count_scifi + count_us + count_ds) > 0"
        ).Count(),
        "veto_positive_events": muons.Filter("count_veto > 0").Count(),
        "scifi_positive_events": muons.Filter("count_scifi > 0").Count(),
        "us_positive_events": muons.Filter("count_us > 0").Count(),
        "ds_positive_events": muons.Filter("count_ds > 0").Count(),
    }
    baseline = frame.Filter(
        str(config["mc_muon_up"]["baseline_expression"]), "v5_mc_baseline"
    )
    for name, selection in configured_selections(config).items():
        actions[f"{name}_candidate_events"] = baseline.Filter(
            selection["expression"], f"{name}_candidate"
        ).Count()
    ROOT.RDF.RunGraphs(list(actions.values()))
    return {name: int(action.GetValue()) for name, action in actions.items()}


def atomic_csv(frame, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--output-validation", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    config = load_yaml(args.study_config)
    hist_config = load_yaml(args.hist_config)
    mc_config = config.get("mc_muon_up")
    if not isinstance(mc_config, dict) or not isinstance(
        mc_config.get("productions"), dict
    ):
        raise ValueError("V5 config requires mc_muon_up.productions")
    metadata = pd.read_csv(args.source_metadata, keep_default_na=False)
    required_columns = {
        "data_type", "subfolder", "partition", "n_event", "geo_path",
        "output_base_path", "feature_path", "lumi_per_file",
    }
    missing_columns = required_columns - set(metadata)
    if missing_columns:
        raise ValueError(f"Source metadata lacks columns: {sorted(missing_columns)}")
    sources = configured_sources(metadata, config)
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)
    required = required_branches(hist_config, config)
    tree_name = str(config.get("tree_name", "sndData"))
    standardized_rows, validation_rows = [], []
    for sequence, (source_index, row) in enumerate(sources.iterrows(), start=1):
        production_key = str(row["production_key"])
        production = mc_config["productions"][production_key]
        path = full_feature_path(row)
        entries = inspect_file(path, tree_name, required)
        declared = int(row["n_event"])
        if entries != declared:
            raise RuntimeError(
                f"MC metadata/tree count mismatch for {path}: {declared} != {entries}"
            )
        counts = event_counts(path, tree_name, config)
        geometry_warning = (
            str(row["partition"]) == "scoring_2"
            and "scoring_2.5" in str(row["geo_path"])
        )
        standardized_rows.append({
            "region": "all_events",
            "particle_group": str(production["group"]),
            "particle_family": "muon",
            "partition_id": f"all_events__{production['group']}__part{sequence:03d}",
            "feature_partition_path": str(path),
            "feature_available": True,
            "n_events": entries,
            "lumi_per_partition": float(row["lumi_per_file"]),
            "source_file_indices": str(source_index),
            "source_file_count": 1,
            "production_key": production_key,
            "production_subfolder": str(row["subfolder"]),
            "production_partition": str(row["partition"]),
            "source_feature_name": str(row["feature_path"]),
        })
        validation_rows.append({
            "source_file_index": source_index,
            "production_key": production_key,
            "particle_group": str(production["group"]),
            "feature_partition_path": str(path),
            "tree_name": tree_name,
            "metadata_events": declared,
            "tree_events": entries,
            "event_count_match": True,
            "required_branch_count": len(required),
            **counts,
            "geometry_path": str(row["geo_path"]),
            "geometry_partition_warning": geometry_warning,
            "status": "warning" if geometry_warning else "ok",
        })
    atomic_csv(pd.DataFrame(standardized_rows), args.output_metadata)
    atomic_csv(pd.DataFrame(validation_rows), args.output_validation)
    print(
        f"Validated {len(standardized_rows)} MC-muon-up feature files with "
        f"{sum(row['n_events'] for row in standardized_rows)} events"
    )


if __name__ == "__main__":
    main()
