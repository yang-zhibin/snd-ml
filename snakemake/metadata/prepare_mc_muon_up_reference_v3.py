#!/usr/bin/env python3
"""Validate and standardize the configured v3 upward-muon MC reference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import ROOT
import yaml


SELECTION_BRANCHES = {
    "pdgCode",
    "count_veto",
    "count_scifi",
    "count_us",
    "count_ds",
}
IGNORED_EXPRESSION_TOKENS = {
    "std", "isfinite", "abs", "max", "min", "true", "false",
}


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def selected_source(metadata, config):
    source = config["mc_muon_up"]["source"]
    selected = metadata.loc[
        metadata["subfolder"].astype(str).eq(str(source["subfolder"]))
        & metadata["partition"].astype(str).eq(str(source["partition"]))
        & metadata["feature_path"].astype(str).eq(str(source["feature_name"]))
    ].copy()
    if len(selected) != 1:
        raise RuntimeError(
            "Configured MC-muon-up source must match exactly one metadata row; "
            f"matched {len(selected)}"
        )
    return selected.iloc[0]


def feature_path(row):
    path = Path(str(row["feature_path"]))
    return path if path.is_absolute() else Path(str(row["output_base_path"])) / path


def required_feature_branches(hist_config):
    definitions = hist_config.get("definitions", {})
    expressions = list(definitions.values())
    for feature in hist_config.get("features", {}).values():
        for field in ("expression", "x_expression", "y_expression", "feature_cut"):
            if feature.get(field):
                expressions.append(str(feature[field]))
    tokens = {
        token
        for expression in expressions
        for token in re.findall(r"\b[A-Za-z_]\w*\b", str(expression))
    }
    return (
        tokens
        - set(definitions)
        - IGNORED_EXPRESSION_TOKENS
        | SELECTION_BRANCHES
    )


def inspect_root(path, tree_name, required_branches):
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
        missing = required_branches - branches
        if missing:
            raise RuntimeError(f"MC feature file lacks branches: {sorted(missing)}")
        track_branches = sorted(name for name in branches if name.startswith("reco_"))
        return entries, track_branches
    finally:
        source.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--output-validation", required=True)
    args = parser.parse_args()

    config = load_yaml(args.study_config)
    hist_config = load_yaml(args.hist_config)
    mc_config = config.get("mc_muon_up")
    if not isinstance(mc_config, dict):
        raise ValueError("Study configuration requires mc_muon_up")
    for field in (
        "group", "label", "source", "inclusive_expression",
        "reference_expression",
    ):
        if not mc_config.get(field):
            raise ValueError(f"mc_muon_up lacks {field!r}")

    metadata = pd.read_csv(args.source_metadata, keep_default_na=False)
    required_columns = {
        "data_type", "subfolder", "partition", "output_base_path",
        "feature_path",
    }
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Source metadata lacks columns: {sorted(missing_columns)}"
        )
    row = selected_source(metadata, config)
    if str(row["data_type"]) != "MC_muon" or str(row["subfolder"]) != "up":
        raise RuntimeError("Configured source is not upward MC_muon data")

    path = feature_path(row)
    required_branches = required_feature_branches(hist_config)
    entries, track_branches = inspect_root(
        path, str(config.get("tree_name", "sndData")), required_branches
    )
    dataframe = ROOT.RDataFrame(str(config.get("tree_name", "sndData")), str(path))
    inclusive_action = dataframe.Filter(
        str(mc_config["inclusive_expression"])
    ).Count()
    reference_action = dataframe.Filter(
        str(mc_config["reference_expression"])
    ).Count()
    ROOT.RDF.RunGraphs([inclusive_action, reference_action])
    inclusive_events = int(inclusive_action.GetValue())
    reference_events = int(reference_action.GetValue())
    if inclusive_events <= 0 or reference_events <= 0:
        raise RuntimeError(
            "Configured MC-muon-up inclusive/reference selection is empty: "
            f"inclusive={inclusive_events}, reference={reference_events}"
        )
    declared_entries = int(row.get("n_event", entries))
    if declared_entries != entries:
        raise RuntimeError(
            f"MC metadata/tree count mismatch: {declared_entries} != {entries}"
        )

    standardized = pd.DataFrame([{
        "region": "all_events",
        "particle_group": str(mc_config["group"]),
        "particle_family": "muon",
        "partition_id": "all_events__mc_muon_up__part001",
        "feature_partition_path": str(path),
        "feature_available": True,
        "n_events": entries,
        "lumi_per_partition": float(row.get("lumi_per_file", 0.0)),
        "source_file_indices": str(row.name),
        "source_file_count": 1,
        "production_subfolder": str(row["subfolder"]),
        "production_partition": str(row["partition"]),
        "source_feature_name": str(row["feature_path"]),
    }])
    validation = pd.DataFrame([{
        "particle_group": str(mc_config["group"]),
        "feature_partition_path": str(path),
        "tree_name": str(config.get("tree_name", "sndData")),
        "metadata_events": declared_entries,
        "tree_events": entries,
        "inclusive_muon_events": inclusive_events,
        "detector_activity_reference_events": reference_events,
        "reference_fraction": reference_events / inclusive_events,
        "event_count_match": True,
        "required_branch_count": len(required_branches),
        "reco_track_branch_count": len(track_branches),
        "reco_track_branches_required": False,
        "production_subfolder": str(row["subfolder"]),
        "production_partition": str(row["partition"]),
        "status": "ok",
    }])

    for output in (args.output_metadata, args.output_validation):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(args.output_metadata, index=False)
    validation.to_csv(args.output_validation, index=False)
    print(
        f"Validated {entries} upward-muon MC events from {path}; "
        f"reco-track branches present={len(track_branches)}, required=0"
    )


if __name__ == "__main__":
    main()
