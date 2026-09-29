#!/usr/bin/env python3
"""Validate or combine region-v2 inspection metadata products."""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

from region_v2_inspection_common import (
    AUDIT_FIELDS, MANIFEST_FIELDS, load_config, load_selected_rows,
    source_luminosity, target_luminosity,
)

VALIDATION_FIELDS = ["dataset_name", "status", "check", "message"]

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def close(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)

def root_number(root_file, name):
    obj = root_file.Get(name)
    if not obj:
        raise ValueError(f"Missing ROOT object {name}")
    return float(obj.GetVal())

def root_text(root_file, name):
    obj = root_file.Get(name)
    if not obj:
        raise ValueError(f"Missing ROOT object {name}")
    return str(obj.GetTitle())

def validate(args):
    import ROOT
    started = time.monotonic()
    print(f"[validation 1/4] Loading configuration and expected metadata for {args.dataset}", flush=True)
    config = load_config(args.config)
    spec, metadata_path, selected = load_selected_rows(config, args.dataset)
    manifest_rows, audits = read_csv(args.manifest_row), read_csv(args.source_audit)
    if len(manifest_rows) != 1:
        raise ValueError("Per-dataset manifest must contain exactly one row")
    manifest = manifest_rows[0]
    expected_rows = [str(int(v)) for v in selected["source_metadata_row"]]
    if [r["source_metadata_row"] for r in audits] != expected_rows:
        raise ValueError("Source audit rows do not exactly match deterministic metadata selection")
    if any(r["file_exists"] != "1" or r["file_opened"] != "1" for r in audits):
        raise ValueError("Source audit contains a missing or unreadable selected file")
    print(f"[validation 2/4] Opening output ROOT file and checking required branches", flush=True)
    root_file = ROOT.TFile.Open(args.input, "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"Cannot open {args.input}")
    tree = root_file.Get(config["output"].get("tree_name", "sndData"))
    if not tree:
        raise ValueError("Output lacks sndData")
    branches = {b.GetName() for b in tree.GetListOfBranches()}
    required = {"inspection_event_weight", "inspection_source_row"}
    if missing := required.difference(branches):
        raise ValueError(f"Output lacks branches {sorted(missing)}")
    expected_source = source_luminosity(selected)
    expected_target = target_luminosity(config)
    expected_weight = expected_target / expected_source
    source_file = ROOT.TFile.Open(selected.iloc[0]["resolved_feature_path"], "READ")
    source_tree = source_file.Get(config["output"].get("tree_name", "sndData"))
    source_branches = {b.GetName() for b in source_tree.GetListOfBranches()}
    audited_input_count = sum(int(row["sndData_entries"]) for row in audits)
    print("[validation 3/4] Checking normalization, provenance, counts, and selections", flush=True)
    checks = {
        "inspection_version": root_text(root_file, "inspection_version") == config["version"],
        "inspection_dataset_name": root_text(root_file, "inspection_dataset_name") == args.dataset,
        "source_luminosity": close(root_number(root_file, "inspection_source_luminosity_fb"), expected_source),
        "target_luminosity": close(root_number(root_file, "inspection_target_luminosity_fb"), expected_target),
        "normalization_weight": close(root_number(root_file, "inspection_normalization_weight"), expected_weight),
        "selected_file_count": close(root_number(root_file, "inspection_selected_file_count"), len(selected)),
        "output_entry_count": close(root_number(root_file, "inspection_output_entry_count"), tree.GetEntries()),
        "input_entry_count": close(root_number(root_file, "inspection_input_entry_count"), audited_input_count),
        "source_branches_preserved": source_branches.issubset(branches),
        "event_selection": root_text(root_file, "inspection_event_selection") == spec.get("event_selection", "true"),
        "particle_filter": root_text(root_file, "inspection_particle_filter") == spec.get("particle_filter", "true"),
        "manifest_output_entries": int(manifest["output_entry_count"]) == tree.GetEntries(),
        "manifest_source_luminosity": close(manifest["source_luminosity_fb"], expected_source),
        "manifest_target_luminosity": close(manifest["target_luminosity_fb"], expected_target),
        "manifest_weight": close(manifest["normalization_weight"], expected_weight),
    }
    if tree.GetEntries():
        print("    scanning event weights and source-row identifiers", flush=True)
        rdf = ROOT.RDataFrame(tree)
        checks["constant_event_weight"] = close(rdf.Min("inspection_event_weight").GetValue(), expected_weight) and close(rdf.Max("inspection_event_weight").GetValue(), expected_weight)
        observed_rows = {str(int(x)) for x in rdf.Take["int"]("inspection_source_row").GetValue()}
        checks["source_rows_are_selected"] = observed_rows.issubset(set(expected_rows))
    failures = [name for name, passed in checks.items() if not passed]
    source_file.Close()
    root_file.Close()
    rows = [{"dataset_name": args.dataset, "status": "PASS" if not failures else "FAIL", "check": name, "message": "ok" if passed else "failed"} for name, passed in checks.items()]
    print(f"[validation 4/4] Writing {len(checks)} validation results", flush=True)
    write_csv(args.output, VALIDATION_FIELDS, rows)
    if failures:
        raise ValueError(f"Validation failed: {failures}")
    print(
        f"Validated {args.dataset}: {len(checks)} checks passed in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )

def combine(args):
    manifests, audits, validations = [], [], []
    for path in args.manifest_rows:
        manifests.extend(read_csv(path))
    for path in args.source_audits:
        audits.extend(read_csv(path))
    for path in args.validation_rows:
        validations.extend(read_csv(path))
    names = [row["dataset_name"] for row in manifests]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dataset in combined manifest")
    write_csv(args.manifest, MANIFEST_FIELDS, manifests)
    write_csv(args.source_audit, AUDIT_FIELDS, audits)
    write_csv(args.validation, VALIDATION_FIELDS, validations)

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("validate")
    for option in ("config", "dataset", "input", "manifest-row", "source-audit", "output"):
        one.add_argument(f"--{option}", required=True)
    many = sub.add_parser("combine")
    many.add_argument("--manifest-rows", nargs="+", required=True)
    many.add_argument("--source-audits", nargs="+", required=True)
    many.add_argument("--validation-rows", nargs="+", required=True)
    many.add_argument("--manifest", required=True)
    many.add_argument("--source-audit", required=True)
    many.add_argument("--validation", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        arguments = parse_args()
        validate(arguments) if arguments.command == "validate" else combine(arguments)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
