#!/usr/bin/env python3
"""Split one aligned neutrino feature/Hit3D pair into four truth classes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from build_muon_loose_partitions import (
    cpp_string, output_identity_mismatches, snapshot, snapshot_columns,
    validate_thread_count,
)
from data_driven_3d_dataset import load_yaml, truthy


def selected_source(source_catalog: str, source_uid: str) -> pd.Series:
    catalog = pd.read_csv(source_catalog, keep_default_na=False, low_memory=False)
    rows = catalog.loc[catalog["source_uid"].astype(str).eq(str(source_uid))]
    if len(rows) != 1:
        raise ValueError(f"Expected one source row for {source_uid}, found {len(rows)}")
    row = rows.iloc[0]
    if str(row["role"]) != "neutrino_split":
        raise ValueError(f"Source {source_uid} does not have role=neutrino_split")
    if not truthy(row["input_available"]):
        raise FileNotFoundError(f"Source {source_uid} does not have both input products")
    return row


def expression_for_codes(codes) -> str:
    values = sorted({int(value) for value in codes})
    if not values:
        raise ValueError("A neutrino class has no configured PDG codes")
    return " || ".join(f"pdgCode == {value}" for value in values)


def define_metadata(rdf, class_id: int, source_uid: str, version: str):
    definitions = {
        "ml_source_entry": "(Long64_t)rdfentry_",
        "ml_class_id": f"int({int(class_id)})",
        "ml_domain_id": "int(1)",
        "ml_source_uid": f"std::string({cpp_string(source_uid)})",
        "ml_selection_fingerprint": f"std::string({cpp_string(version)})",
    }
    columns = {str(value) for value in rdf.GetColumnNames()}
    for name, expression in definitions.items():
        if name in columns:
            raise ValueError(f"Input unexpectedly contains reserved column {name}")
        rdf = rdf.Define(name, expression)
        columns.add(name)
    return rdf, list(definitions)


def process_source(args) -> dict:
    import ROOT

    validate_thread_count(args.threads)
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    config = load_yaml(args.config)
    partition = config["neutrino_partition"]
    version = str(partition.get("version", "truth_pdg_partition_v1"))
    row = selected_source(args.source_catalog, args.source_uid)
    feature_tree_name = str(config["trees"]["feature"])
    hit_tree_name = str(config["trees"]["hit3d"])

    feature_file = ROOT.TFile.Open(str(row["feature_path"]))
    hit_file = ROOT.TFile.Open(str(row["hit3d_path"]))
    files = [feature_file, hit_file]
    try:
        if any(not item or item.IsZombie() for item in files):
            raise OSError("Could not open a neutrino input ROOT file")
        feature_tree = feature_file.Get(feature_tree_name)
        hit_tree = hit_file.Get(hit_tree_name)
        if not feature_tree or not hit_tree:
            raise KeyError("Input feature or Hit3D tree is missing")
        n_feature = int(feature_tree.GetEntries())
        n_hit = int(hit_tree.GetEntries())
        if n_feature != n_hit:
            raise RuntimeError(f"Entry mismatch: feature={n_feature}, hit3D={n_hit}")
        feature_columns = {str(branch.GetName()) for branch in feature_tree.GetListOfBranches()}
        hit_columns = {str(branch.GetName()) for branch in hit_tree.GetListOfBranches()}
    finally:
        for item in files:
            if item:
                item.Close()

    identity = [str(value) for value in partition.get(
        "identity_branches", ["runId", "eventId", "eventIndex"]
    )]
    missing = set(identity + ["pdgCode"]) - feature_columns.intersection(hit_columns)
    if missing:
        raise KeyError(f"Missing shared neutrino branches: {sorted(missing)}")

    feature_chain = ROOT.TChain(feature_tree_name)
    hit_chain = ROOT.TChain(hit_tree_name)
    feature_chain.Add(str(row["feature_path"]))
    hit_chain.Add(str(row["hit3d_path"]))
    hit_chain.AddFriend(feature_chain, "feat")
    feature_rdf = ROOT.RDataFrame(feature_chain)
    hit_rdf = ROOT.RDataFrame(hit_chain)
    mismatch_expression = " || ".join(f"({name} != feat.{name})" for name in identity)
    mismatch_count = int(hit_rdf.Filter(mismatch_expression or "false").Count().GetValue())
    if mismatch_count:
        raise RuntimeError(f"Found {mismatch_count} entrywise identity mismatches")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    classes = config["classes"]
    configured = partition["classes"]
    partition_rows = {}
    output_mismatches = 0
    total_selected = 0
    all_code_expressions = []
    for class_name in ("CC_numu", "CC_nue", "CC_nutau", "NC"):
        expression = expression_for_codes(configured[class_name])
        all_code_expressions.append(f"({expression})")
        feature_selected = feature_rdf.Filter(expression, class_name)
        hit_selected = hit_rdf.Filter(expression, class_name)
        feature_selected, added = define_metadata(
            feature_selected, classes[class_name]["class_id"], args.source_uid, version
        )
        hit_selected, _ = define_metadata(
            hit_selected, classes[class_name]["class_id"], args.source_uid, version
        )
        count = int(feature_selected.Count().GetValue())
        total_selected += count
        feature_path = output_root / f"{class_name}_feature.root"
        hit_path = output_root / f"{class_name}_hit3d.root"
        snapshot(feature_selected, feature_tree_name, str(feature_path),
                 snapshot_columns(feature_selected, sorted(feature_columns), added))
        snapshot(hit_selected, hit_tree_name, str(hit_path),
                 snapshot_columns(hit_selected, sorted(hit_columns), added))
        output_mismatches += output_identity_mismatches(
            str(row["hit3d_path"]), str(feature_path), str(hit_path),
            feature_tree_name, hit_tree_name, identity,
        )
        partition_rows[class_name] = {
            "entries": count,
            "feature_path": str(feature_path),
            "hit3d_path": str(hit_path),
            "expression": expression,
        }

    classified_expression = " || ".join(all_code_expressions)
    unclassified = int(feature_rdf.Filter(f"!({classified_expression})").Count().GetValue())
    if total_selected + unclassified != n_feature:
        raise RuntimeError("Neutrino partition closure failed")
    if bool(partition.get("require_complete", True)) and unclassified:
        raise RuntimeError(f"Found {unclassified} neutrino events with unconfigured pdgCode")
    if output_mismatches:
        raise RuntimeError(f"Found {output_mismatches} output identity mismatches")

    result = {
        "dataset_version": config["dataset_version"],
        "source_uid": args.source_uid,
        "fold_group_uid": row["fold_group_uid"],
        "lumi_source_uid": row["lumi_source_uid"],
        "dataset_key": row["dataset_key"],
        "domain": row["domain"],
        "run": row["run"],
        "production": row["production"],
        "energy_bin": row["energy_bin"],
        "lumi_per_file": float(row["lumi_per_file"]),
        "input_entries": n_feature,
        "partitioned_entries": total_selected,
        "unclassified_entries": unclassified,
        "identity_mismatches": mismatch_count,
        "output_identity_mismatches": output_mismatches,
        "partition_version": version,
        "partitions": json.dumps(partition_rows, sort_keys=True),
    }
    audit = Path(args.audit)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--source-uid", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    result = process_source(args)
    print(f"{args.source_uid}: partitioned={result['partitioned_entries']}")


if __name__ == "__main__":
    main()
