#!/usr/bin/env python3
"""Split one aligned data/MC-muon source into V7 loose pass and fail ROOT files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import (
    configured_density_bins, load_yaml, resolve_repo_path, truthy,
)
from passing_muon_selection import load_configured_coverage_selection


CPP_RESERVED = {"int", "float", "double", "std", "isfinite", "abs", "true", "false"}


def identifiers(expression: str) -> set[str]:
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(expression)))
    return names - CPP_RESERVED


def cpp_string(value: object) -> str:
    return json.dumps(str(value))


def snapshot_columns(rdf, original_columns: list[str], added: list[str]):
    result = list(original_columns)
    for name in added:
        if name not in result:
            result.append(name)
    vector = __import__("ROOT").std.vector("string")()
    for name in result:
        vector.push_back(str(name))
    return vector


def snapshot(rdf, tree_name: str, output: str, columns, compression: int = 1):
    import ROOT
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = ROOT.RDF.RSnapshotOptions()
    options.fCompressionLevel = int(compression)
    action = rdf.Snapshot(tree_name, str(output_path), columns, options)
    if hasattr(action, "GetValue"):
        action.GetValue()


def define_metadata(rdf, class_id: int, domain_id: int, passed: bool, source_uid: str,
                    fingerprint: str):
    definitions = {
        "ml_source_entry": "(Long64_t)rdfentry_",
        "ml_class_id": f"int({int(class_id)})",
        "ml_domain_id": f"int({int(domain_id)})",
        "ml_selection_pass": "true" if passed else "false",
        "ml_source_uid": f"std::string({cpp_string(source_uid)})",
        "ml_selection_fingerprint": f"std::string({cpp_string(fingerprint)})",
    }
    columns = {str(value) for value in rdf.GetColumnNames()}
    for name, expression in definitions.items():
        if name in columns:
            raise ValueError(f"Input unexpectedly contains reserved column {name}")
        rdf = rdf.Define(name, expression)
        columns.add(name)
    return rdf, list(definitions)


def replace_identifiers(expression: str, replacements: dict[str, str]) -> str:
    result = str(expression)
    for source in sorted(replacements, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(source)}\b", replacements[source], result)
    return result


def define_v7_columns(
    rdf, input_columns: set[str], selection: dict, friend_prefix=None,
    extra_expressions=(),
):
    """Expose required feature columns and construct V7 derived columns."""
    derived_names = set(selection["derived_columns"])
    raw_names = identifiers(selection["expression"]) - derived_names
    for expression in selection["derived_columns"].values():
        raw_names.update(identifiers(expression) - derived_names)
    for expression in extra_expressions:
        raw_names.update(identifiers(expression) - derived_names)
    raw_names.add("density_sndsw_scifi")

    columns = set(input_columns)
    replacements = {}
    for name in sorted(raw_names):
        if friend_prefix is None:
            if name not in columns:
                raise KeyError(f"Required feature branch {name!r} is missing")
            continue
        alias = f"mlfeat__{name}"
        rdf = rdf.Alias(alias, f"{friend_prefix}.{name}")
        columns.add(alias)
        replacements[name] = alias
    for name, expression in selection["derived_columns"].items():
        if name in columns:
            raise ValueError(f"Derived V7 column {name!r} already exists")
        rdf = rdf.Define(name, replace_identifiers(expression, replacements))
        columns.add(name)
    return rdf, replace_identifiers(selection["expression"], replacements), replacements


def density_expression(item: dict, variable: str) -> str:
    low = float(item["low"])
    high = float(item["high"])
    lower = f"{variable} >= {low:.17g}"
    if math.isinf(high):
        return f"std::isfinite({variable}) && ({lower})"
    return (
        f"std::isfinite({variable}) && ({lower}) && "
        f"({variable} < {high:.17g})"
    )


def count_density_bins(frame, config: dict) -> dict[str, int]:
    variable = str(config["density_sampling"]["variable"])
    bins = configured_density_bins(config)
    actions = {
        item["name"]: frame.Filter(density_expression(item, variable)).Count()
        for item in bins
    }
    invalid = frame.Filter(
        f"!std::isfinite({variable}) || {variable} < {bins[0]['low']:.17g}"
    ).Count()
    result = {name: int(action.GetValue()) for name, action in actions.items()}
    result["invalid"] = int(invalid.GetValue())
    return result


def selected_source(source_catalog: str, source_uid: str) -> pd.Series:
    catalog = pd.read_csv(source_catalog, keep_default_na=False, low_memory=False)
    rows = catalog.loc[catalog["source_uid"].astype(str) == str(source_uid)]
    if len(rows) != 1:
        raise ValueError(f"Expected one source row for {source_uid}, found {len(rows)}")
    row = rows.iloc[0]
    if str(row["role"]) != "muon_split":
        raise ValueError(f"Source {source_uid} does not have role=muon_split")
    if not truthy(row["input_available"]):
        raise FileNotFoundError(f"Source {source_uid} does not have both input products")
    return row


def validate_thread_count(threads: int) -> None:
    if int(threads) not in (0, 1):
        raise ValueError(
            "Muon feature/Hit3D partitioning must be single-threaded: ROOT "
            "Snapshot does not guarantee identical output ordering with implicit MT"
        )


def output_identity_mismatches(input_hit3d: str, feature_path: str, hit3d_path: str,
                               feature_tree_name: str, hit_tree_name: str,
                               identity: list[str]) -> int:
    """Check output pair alignment and original-entry lineage after Snapshot."""
    import ROOT

    input_file = ROOT.TFile.Open(str(input_hit3d))
    feature_file = ROOT.TFile.Open(str(feature_path))
    hit_file = ROOT.TFile.Open(str(hit3d_path))
    files = [input_file, feature_file, hit_file]
    try:
        if any(not item or item.IsZombie() for item in files):
            raise OSError("Could not open an input or output ROOT file for identity validation")
        original = input_file.Get(hit_tree_name)
        feature = feature_file.Get(feature_tree_name)
        hit = hit_file.Get(hit_tree_name)
        if not original or not feature or not hit:
            raise KeyError("Identity validation tree is missing")
        if feature.GetEntries() != hit.GetEntries():
            return abs(int(feature.GetEntries()) - int(hit.GetEntries())) or 1
        if int(hit.GetEntries()) == 0:
            return 0
        if not hit.GetBranch("ml_source_entry"):
            raise KeyError("Output Hit3D tree lacks ml_source_entry")
        mismatches = 0
        original_entries = int(original.GetEntries())
        for output_entry in range(int(hit.GetEntries())):
            feature.GetEntry(output_entry)
            hit.GetEntry(output_entry)
            source_entry = int(hit.ml_source_entry)
            if source_entry < 0 or source_entry >= original_entries:
                mismatches += 1
                continue
            original.GetEntry(source_entry)
            pair_matches = all(
                int(getattr(feature, name)) == int(getattr(hit, name))
                for name in identity
            )
            source_matches = all(
                int(getattr(original, name)) == int(getattr(hit, name))
                for name in identity
            )
            if not pair_matches or not source_matches:
                mismatches += 1
        return mismatches
    finally:
        for item in files:
            if item:
                item.Close()


def process_source(args) -> dict:
    import ROOT

    validate_thread_count(args.threads)
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()

    config = load_yaml(args.config)
    repo_root = Path(args.config).resolve().parents[4]
    muon = config["muon_selection"]
    study_path = resolve_repo_path(muon["study_config"], repo_root)
    study = load_yaml(study_path)
    selection = load_configured_coverage_selection(study_path, muon["working_point"])
    row = selected_source(args.source_catalog, args.source_uid)

    feature_file = ROOT.TFile.Open(str(row["feature_path"]))
    hit_file = ROOT.TFile.Open(str(row["hit3d_path"]))
    if not feature_file or feature_file.IsZombie():
        raise OSError(f"Could not open feature file {row['feature_path']}")
    if not hit_file or hit_file.IsZombie():
        raise OSError(f"Could not open Hit3D file {row['hit3d_path']}")
    feature_tree_name = str(config["trees"]["feature"])
    hit_tree_name = str(config["trees"]["hit3d"])
    feature_tree = feature_file.Get(feature_tree_name)
    hit_tree = hit_file.Get(hit_tree_name)
    if not feature_tree or not hit_tree:
        raise KeyError("Input feature or Hit3D tree is missing")
    n_feature = int(feature_tree.GetEntries())
    n_hit = int(hit_tree.GetEntries())
    if bool(muon.get("require_equal_entries", True)) and n_feature != n_hit:
        raise RuntimeError(f"Entry mismatch: feature={n_feature}, hit3D={n_hit}")
    feature_columns = {str(branch.GetName()) for branch in feature_tree.GetListOfBranches()}
    hit_columns = {str(branch.GetName()) for branch in hit_tree.GetListOfBranches()}
    identity = [str(value) for value in muon.get("identity_branches", [])]
    missing_identity = set(identity) - feature_columns.intersection(hit_columns)
    if missing_identity:
        raise KeyError(f"Missing identity branches: {sorted(missing_identity)}")
    feature_file.Close()
    hit_file.Close()

    feature_chain = ROOT.TChain(feature_tree_name)
    hit_chain = ROOT.TChain(hit_tree_name)
    if feature_chain.Add(str(row["feature_path"])) == 0:
        raise OSError(f"Could not chain {row['feature_path']}")
    if hit_chain.Add(str(row["hit3d_path"])) == 0:
        raise OSError(f"Could not chain {row['hit3d_path']}")
    hit_chain.AddFriend(feature_chain, "feat")

    domain = str(row["domain"])
    blinding = muon.get("data_blinding", {})
    blinding_enabled = domain == "data" and bool(blinding.get("enabled", False))
    blinded_expression = (
        str(blinding["signal_region_expression"]) if blinding_enabled else "false"
    )
    baseline = "true"
    if blinding_enabled:
        baseline = str(blinding["training_parent_expression"])
    elif domain == "mc" and bool(muon.get("use_v7_mc_baseline", True)):
        baseline = str(study["mc_muon_up"]["baseline_expression"])

    feature_rdf = ROOT.RDataFrame(feature_chain)
    feature_rdf, feature_selection_expression, _ = define_v7_columns(
        feature_rdf, feature_columns, selection,
        extra_expressions=[baseline, blinded_expression]
    )
    hit_rdf = ROOT.RDataFrame(hit_chain)
    hit_rdf, hit_selection_expression, hit_replacements = define_v7_columns(
        hit_rdf, hit_columns, selection, friend_prefix="feat",
        extra_expressions=[baseline, blinded_expression],
    )

    mismatch_expression = " || ".join(
        f"({name} != feat.{name})" for name in identity
    ) or "false"
    mismatch_count = int(hit_rdf.Filter(mismatch_expression).Count().GetValue())
    if bool(muon.get("require_identity_match", True)) and mismatch_count:
        raise RuntimeError(f"Found {mismatch_count} entrywise identity mismatches")

    parent_feature = feature_rdf.Filter(baseline, "muon_parent")
    hit_baseline = replace_identifiers(baseline, hit_replacements)
    parent_hit = hit_rdf.Filter(hit_baseline, "muon_parent")
    expression = feature_selection_expression
    pass_feature = parent_feature.Filter(expression, "coverage_loose_pass")
    fail_feature = parent_feature.Filter(f"!({expression})", "coverage_loose_fail")
    pass_hit = parent_hit.Filter(hit_selection_expression, "coverage_loose_pass")
    fail_hit = parent_hit.Filter(f"!({hit_selection_expression})", "coverage_loose_fail")
    hit_blinded_expression = replace_identifiers(blinded_expression, hit_replacements)
    signal_feature = feature_rdf.Filter(blinded_expression, "sealed_signal_region")
    signal_hit = hit_rdf.Filter(hit_blinded_expression, "sealed_signal_region")

    n_blinded = int(feature_rdf.Filter(blinded_expression).Count().GetValue())
    overlap = int(
        feature_rdf.Filter(f"({baseline}) && ({blinded_expression})").Count().GetValue()
    )
    if overlap:
        raise RuntimeError(f"Training parent overlaps the blinded data region by {overlap} events")

    classes = config["classes"]
    pass_name = str(muon["pass_class"])
    fail_name = str(muon["fail_class"])
    domain_id = 0 if str(row["domain"]) == "data" else 1
    pass_feature, added = define_metadata(
        pass_feature, classes[pass_name]["class_id"], domain_id, True,
        args.source_uid, selection["fingerprint"],
    )
    fail_feature, _ = define_metadata(
        fail_feature, classes[fail_name]["class_id"], domain_id, False,
        args.source_uid, selection["fingerprint"],
    )
    pass_hit, _ = define_metadata(
        pass_hit, classes[pass_name]["class_id"], domain_id, True,
        args.source_uid, selection["fingerprint"],
    )
    fail_hit, _ = define_metadata(
        fail_hit, classes[fail_name]["class_id"], domain_id, False,
        args.source_uid, selection["fingerprint"],
    )
    signal_feature, signal_added = define_metadata(
        signal_feature, -1, domain_id, False, args.source_uid, selection["fingerprint"],
    )
    signal_hit, _ = define_metadata(
        signal_hit, -1, domain_id, False, args.source_uid, selection["fingerprint"],
    )

    n_parent = int(parent_feature.Count().GetValue())
    n_pass = int(pass_feature.Count().GetValue())
    n_fail = int(fail_feature.Count().GetValue())
    if n_parent != n_pass + n_fail:
        raise RuntimeError(f"Pass/fail closure failed: {n_parent} != {n_pass} + {n_fail}")
    if blinding_enabled and n_feature != n_parent + n_blinded:
        raise RuntimeError(
            "Data blinding closure failed: input is not exactly training parent plus blinded region"
        )
    leaked_blinded = int(fail_feature.Filter(blinded_expression).Count().GetValue())
    if leaked_blinded:
        raise RuntimeError("Blinded data events leaked into the muon-DIS output")
    pass_density = count_density_bins(pass_feature, config)
    fail_density = count_density_bins(fail_feature, config)
    if sum(pass_density.values()) != n_pass or sum(fail_density.values()) != n_fail:
        raise RuntimeError("Density-bin counts do not close to selected populations")

    feature_original = sorted(feature_columns)
    hit_original = sorted(hit_columns)
    hit_added = [*added, str(config["density_sampling"]["variable"])]
    snapshot(pass_feature, feature_tree_name, args.pass_feature,
             snapshot_columns(pass_feature, feature_original, added))
    snapshot(fail_feature, feature_tree_name, args.fail_feature,
             snapshot_columns(fail_feature, feature_original, added))
    snapshot(pass_hit, hit_tree_name, args.pass_hit3d,
             snapshot_columns(pass_hit, hit_original, hit_added))
    snapshot(fail_hit, hit_tree_name, args.fail_hit3d,
             snapshot_columns(fail_hit, hit_original, hit_added))
    snapshot(signal_feature, feature_tree_name, args.signal_feature,
             snapshot_columns(signal_feature, feature_original, signal_added))
    snapshot(signal_hit, hit_tree_name, args.signal_hit3d,
             snapshot_columns(signal_hit, hit_original, signal_added))

    output_mismatches = sum(
        output_identity_mismatches(
            str(row["hit3d_path"]), feature_path, hit_path,
            feature_tree_name, hit_tree_name, identity,
        )
        for feature_path, hit_path in (
            (args.pass_feature, args.pass_hit3d),
            (args.fail_feature, args.fail_hit3d),
            (args.signal_feature, args.signal_hit3d),
        )
    )
    if bool(muon.get("require_identity_match", True)) and output_mismatches:
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
        "lumi_per_file": float(row["lumi_per_file"]),
        "input_feature_path": row["feature_path"],
        "input_hit3d_path": row["hit3d_path"],
        "input_entries": n_feature,
        "parent_entries": n_parent,
        "excluded_by_parent_baseline": n_feature - n_parent,
        "blinding_applied": blinding_enabled,
        "blinding_expression": blinded_expression,
        "blinded_entries": n_blinded,
        "blinded_fail_leakage": leaked_blinded,
        "signal_region_entries": n_blinded,
        "signal_region_feature_path": args.signal_feature,
        "signal_region_hit3d_path": args.signal_hit3d,
        "identity_mismatches": mismatch_count,
        "output_identity_mismatches": output_mismatches,
        "selection_name": selection["name"],
        "selection_fingerprint": selection["fingerprint"],
        "selection_expression": expression,
        "parent_expression": baseline,
        "pass_class": pass_name,
        "pass_entries": n_pass,
        "pass_feature_path": args.pass_feature,
        "pass_hit3d_path": args.pass_hit3d,
        "pass_density_counts": json.dumps(pass_density, sort_keys=True),
        "fail_class": fail_name,
        "fail_entries": n_fail,
        "fail_feature_path": args.fail_feature,
        "fail_hit3d_path": args.fail_hit3d,
        "fail_density_counts": json.dumps(fail_density, sort_keys=True),
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--source-uid", required=True)
    parser.add_argument("--pass-feature", required=True)
    parser.add_argument("--pass-hit3d", required=True)
    parser.add_argument("--fail-feature", required=True)
    parser.add_argument("--fail-hit3d", required=True)
    parser.add_argument("--signal-feature", required=True)
    parser.add_argument("--signal-hit3d", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    result = process_source(args)
    print(
        f"{args.source_uid}: parent={result['parent_entries']}, "
        f"pass={result['pass_entries']}, fail={result['fail_entries']}"
    )


if __name__ == "__main__":
    main()
