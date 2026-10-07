#!/usr/bin/env python3
"""Merge validated unified shower-feature shards in selected-entry order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import ROOT


TREE_NAMES = ("sndData", "cutFlowSummary")
SHARD_METADATA_NAMES = {
    "metadata_selected_entry_start",
    "metadata_selected_entry_stop",
}
COMMON_RECEIPT_FIELDS = (
    "manifest_version",
    "unified_schema_version",
    "scifi_cluster_schema_version",
    "source_uid",
    "source_domain_id",
    "source_domain",
    "source_geometry_id",
    "source_geometry_checksum",
    "input_digi_path",
    "input_cutflow_path",
    "expected_input_event_count",
    "source_selection_expression",
    "available_selected_event_count",
    "conversion_max_events",
    "converter_commit",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipts(paths: list[str]) -> list[dict]:
    receipts = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            receipt = json.load(source)
        receipt["validation_record"] = str(path)
        receipts.append(receipt)
    return receipts


def validate_receipts(receipts: list[dict]) -> list[dict]:
    if not receipts:
        raise RuntimeError("at least one shard validation record is required")
    for receipt in receipts:
        if receipt.get("validation_passed") is not True:
            raise RuntimeError(
                f"shard validation did not pass: {receipt['validation_record']}"
            )
        for field in ("selected_entry_start", "selected_entry_stop"):
            if field not in receipt:
                raise RuntimeError(
                    f"shard receipt lacks {field}: {receipt['validation_record']}"
                )
    ordered = sorted(receipts, key=lambda item: int(item["selected_entry_start"]))
    reference = ordered[0]
    expected_start = 0
    for receipt in ordered:
        for field in COMMON_RECEIPT_FIELDS:
            if receipt.get(field) != reference.get(field):
                raise RuntimeError(
                    f"shard field {field} differs between validation records"
                )
        start = int(receipt["selected_entry_start"])
        stop = int(receipt["selected_entry_stop"])
        if start != expected_start:
            raise RuntimeError(
                f"shard coverage gap or overlap: expected start {expected_start}, "
                f"observed {start}"
            )
        if stop <= start:
            raise RuntimeError(f"empty or reversed shard range [{start}, {stop})")
        if int(receipt["written_event_count"]) != stop - start:
            raise RuntimeError("shard written count does not match its range")
        feature_path = str(receipt["feature_path"])
        if not os.path.isfile(feature_path):
            raise RuntimeError(f"shard feature is not readable: {feature_path}")
        if sha256(feature_path) != str(receipt["feature_sha256"]):
            raise RuntimeError(f"shard checksum mismatch: {feature_path}")
        expected_start = stop
    available = int(reference["available_selected_event_count"])
    if expected_start != available:
        raise RuntimeError(
            f"shard coverage ends at {expected_start}, expected {available}"
        )
    if int(reference["conversion_max_events"]) != 0:
        raise RuntimeError("production shard receipts must have max-events zero")
    return ordered


def copy_metadata(source: ROOT.TFile, output: ROOT.TFile) -> None:
    copied = set()
    for key in source.GetListOfKeys():
        name = str(key.GetName())
        if name in copied or name in TREE_NAMES or name in SHARD_METADATA_NAMES:
            continue
        value = source.Get(name)
        if not value or value.InheritsFrom("TTree"):
            continue
        output.cd()
        value.Write(name)
        copied.add(name)


def merge_trees(
    shard_paths: list[str], output: ROOT.TFile
) -> dict[str, int]:
    counts = {}
    for tree_name in TREE_NAMES:
        chain = ROOT.TChain(tree_name)
        for shard_path in shard_paths:
            source = ROOT.TFile.Open(shard_path)
            if not source or source.IsZombie():
                raise RuntimeError(f"cannot open shard feature {shard_path}")
            try:
                tree = source.Get(tree_name)
                if not tree:
                    raise RuntimeError(f"missing {tree_name} in {shard_path}")
            finally:
                source.Close()
            if chain.Add(shard_path) != 1:
                raise RuntimeError(f"cannot add {tree_name} shard {shard_path}")
        expected_entries = int(chain.GetEntries())
        output.cd()
        merged = chain.CloneTree(-1, "fast")
        if not merged:
            raise RuntimeError(f"failed to merge shard tree {tree_name}")
        merged.SetName(tree_name)
        if int(merged.GetEntries()) != expected_entries:
            raise RuntimeError(f"merged {tree_name} entry count mismatch")
        output.cd()
        merged.Write()
        counts[tree_name] = expected_entries
    return counts


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def merge(args: argparse.Namespace) -> dict:
    receipts = validate_receipts(load_receipts(args.shard_record))
    shard_paths = [str(receipt["feature_path"]) for receipt in receipts]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = ROOT.TFile.Open(shard_paths[0])
    if not first or first.IsZombie():
        raise RuntimeError(f"cannot open first shard feature {shard_paths[0]}")
    output = ROOT.TFile(str(destination), "RECREATE")
    if not output or output.IsZombie():
        first.Close()
        raise RuntimeError(f"cannot create merged feature {destination}")
    try:
        copy_metadata(first, output)
        counts = merge_trees(shard_paths, output)
        output.cd()
        ROOT.TParameter("Long64_t")(
            "metadata_feature_shard_count", len(receipts)
        ).Write()
        output.Write()
    finally:
        output.Close()
        first.Close()
    if counts["sndData"] != counts["cutFlowSummary"]:
        raise RuntimeError("merged sndData/cutFlowSummary entry-count mismatch")
    expected_entries = int(receipts[0]["available_selected_event_count"])
    if counts["sndData"] != expected_entries:
        raise RuntimeError(
            f"merged entry count {counts['sndData']} != expected {expected_entries}"
        )
    report = {
        "merge_passed": True,
        "source_uid": receipts[0]["source_uid"],
        "shard_count": len(receipts),
        "merged_event_count": counts["sndData"],
        "output_path": str(destination),
        "output_size_bytes": destination.stat().st_size,
        "output_sha256": sha256(destination),
        "shard_validation_records": [
            receipt["validation_record"] for receipt in receipts
        ],
        "shard_feature_sha256": [receipt["feature_sha256"] for receipt in receipts],
        "shard_selected_entry_ranges": [
            [
                int(receipt["selected_entry_start"]),
                int(receipt["selected_entry_stop"]),
            ]
            for receipt in receipts
        ],
    }
    write_json(args.merge_record, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-record", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge-record", required=True)
    return parser.parse_args()


def main() -> None:
    report = merge(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
