#!/usr/bin/env python3
"""Write one multi-event Run-8329 detector-ID transform shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path

import ROOT
import yaml

from detid_channel_shift_sndsw import SciFiChannelGeometry, event_identity
from write_detid_channel_shift_raw import resolve_tree, transform_event_v3, transform_event_v5


ROOT.gROOT.SetBatch(True)


def read_manifest(path: str, method: str) -> list[dict]:
    eligibility = "simple_eligible" if method == "simple" else "reco_scifi_eligible"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if int(row[eligibility]) == 1]


def atomic_gzip_csv(rows: list[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "runId", "eventId", "source_file_index", "original_event_index",
        "output_event_index", "method", "status", "reason", "selected_axis",
        "target_side", "track_type", "input_scifi_hits", "output_scifi_hits",
        "dropped_scifi_hits", "input_mufilter_hits", "output_mufilter_hits",
        "dropped_mufilter_hits",
    ]
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(value: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("simple", "reco_scifi"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = read_manifest(args.manifest, args.method)
    if args.max_events > 0:
        rows = rows[: args.max_events]
    method_config = {
        "version": config["version"],
        "input": {"event_tree_candidates": ["cbmsim", "rawConv"]},
        "transform": config["methods"][args.method]["transform"],
    }

    source = ROOT.TFile.Open(args.source)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open source {args.source}")
    tree_name, tree = resolve_tree(source, method_config["input"]["event_tree_candidates"])

    output = Path(args.output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".root", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    audits = []
    succeeded = 0
    try:
        destination = ROOT.TFile.Open(temporary, "RECREATE")
        if not destination or destination.IsZombie():
            raise OSError(f"Cannot create {temporary}")
        destination.cd()
        for object_name in ("BranchList", "TimeBasedBranchList", "FileHeader", "cbmout"):
            source_key = source.GetKey(object_name)
            if not source_key:
                raise RuntimeError(f"Required FairRoot object {object_name} is missing")
            source_object = source_key.ReadObj()
            destination.cd()
            option = ROOT.TObject.kSingleKey if source_object.InheritsFrom("TCollection") else 0
            source_object.Write(object_name, option)
        output_tree = tree.CloneTree(0)
        geometry = SciFiChannelGeometry(args.geometry)
        layouts = mufilter_layouts = None
        for row in rows:
            base = {
                "runId": int(row["runId"]), "eventId": int(row["eventId"]),
                "source_file_index": int(row["source_file_index"]),
                "original_event_index": int(row["original_event_index"]),
                "output_event_index": -1, "method": args.method, "status": "failed",
                "reason": "", "track_type": row.get("track_type", ""),
            }
            try:
                entry = int(row["original_event_index"])
                if tree.GetEntry(entry) <= 0:
                    raise RuntimeError(f"Cannot read source entry {entry}")
                expected = (int(row["runId"]), int(row["eventId"]))
                if event_identity(tree) != expected:
                    raise RuntimeError(
                        f"Source identity mismatch {event_identity(tree)} != {expected}"
                    )
                geometry.initialize_event(tree.EventHeader)
                if layouts is None:
                    layouts = geometry.build_layouts()
                    mufilter_layouts = geometry.build_mufilter_layouts()
                if args.method == "simple":
                    _, event_audit = transform_event_v5(
                        tree, layouts, mufilter_layouts, row, method_config
                    )
                else:
                    _, event_audit = transform_event_v3(
                        tree, layouts, mufilter_layouts, row, method_config
                    )
                output_tree.Fill()
                axis = event_audit.get("axis_decision", {})
                reference = event_audit.get("reference", {})
                audits.append({
                    **base, "output_event_index": succeeded, "status": "success",
                    "selected_axis": axis.get("selected_axis", ""),
                    "target_side": reference.get("target_side", ""),
                    "input_scifi_hits": event_audit["input_scifi_hits"],
                    "output_scifi_hits": event_audit["output_scifi_hits"],
                    "dropped_scifi_hits": event_audit["dropped_scifi_hits"],
                    "input_mufilter_hits": event_audit["input_mufilter_hits"],
                    "output_mufilter_hits": event_audit["output_mufilter_hits"],
                    "dropped_mufilter_hits": event_audit["dropped_mufilter_hits"],
                })
                succeeded += 1
            except Exception as error:
                audits.append({**base, "reason": str(error)})
        destination.cd()
        output_tree.Write(tree_name, ROOT.TObject.kOverwrite)
        metadata = {
            "version": config["version"], "method": args.method,
            "source": args.source, "geometry": args.geometry,
            "selected_events": len(rows), "successful_events": succeeded,
        }
        ROOT.TNamed("detid_channel_shift_shard_metadata", json.dumps(metadata, sort_keys=True)).Write()
        destination.Close()
        os.replace(temporary, output)
    finally:
        source.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)

    atomic_gzip_csv(audits, args.output_audit)
    summary = {
        "version": config["version"], "method": args.method,
        "source": args.source, "manifest": args.manifest,
        "eligible_events": len(rows), "successful_events": succeeded,
        "failed_events": len(rows) - succeeded, "output_root": str(output),
    }
    write_json(summary, args.output_summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
