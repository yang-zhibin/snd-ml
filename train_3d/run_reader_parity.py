#!/usr/bin/env python3
"""Read a frozen production sample with PyROOT/uproot or compare reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from train_3d.data.data_driven_fold_dataset import EventReference
from train_3d.data.scifi_hit_event_dataset import make_scifi_reader


ARRAY_FIELDS = ("position", "qdc", "detector_type")
SCALAR_FIELDS = ("runId", "eventId", "eventIndex", "pdgCode", "energy")


def array_record(value):
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def json_scalar(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def read_report(args):
    manifest = json.loads(Path(args.manifest).read_text())
    reader = make_scifi_reader(args.backend, max_open_files=args.max_open_files)
    records = []
    max_cached = 0
    try:
        for payload in manifest["references"]:
            reference = EventReference(
                dataset_file_uid=payload["dataset_file_uid"],
                hit3d_path=payload["hit3d_path"],
                source_entry=int(payload["source_entry"]),
                class_id=int(payload["class_id"]),
                class_name=payload["class_name"],
                domain=payload["domain"],
                density_bin=payload.get("density_bin", ""),
            )
            event = reader.read(reference)
            max_cached = max(max_cached, len(reader._files))
            records.append({
                "dataset_file_uid": reference.dataset_file_uid,
                "source_entry": reference.source_entry,
                "class_name": reference.class_name,
                "domain": reference.domain,
                "density_bin": reference.density_bin,
                "categories": payload.get("categories", []),
                "n_candidates": int(len(event["qdc"])),
                "scalars": {
                    name: json_scalar(event[name])
                    for name in SCALAR_FIELDS if name in event
                },
                "arrays": {name: array_record(event[name]) for name in ARRAY_FIELDS},
            })
        for payload in manifest["references"][:min(10, len(records))]:
            reader.read(EventReference(
                payload["dataset_file_uid"], payload["hit3d_path"],
                int(payload["source_entry"]), int(payload["class_id"]),
                payload["class_name"], payload["domain"],
                payload.get("density_bin", ""),
            ))
            max_cached = max(max_cached, len(reader._files))
    finally:
        reader.close()
    package = "uproot" if args.backend == "uproot" else "ROOT"
    if package == "ROOT":
        import ROOT
        version = str(ROOT.gROOT.GetVersion())
    else:
        version = importlib.metadata.version("uproot")
    return {
        "schema_version": 1,
        "backend": args.backend,
        "backend_version": version,
        "python": platform.python_version(),
        "manifest": str(Path(args.manifest).resolve()),
        "reference_count": len(records),
        "max_open_files": int(args.max_open_files),
        "max_cached_files_observed": max_cached,
        "records": records,
    }


def compare_reports(args):
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    left_records = {
        (row["dataset_file_uid"], int(row["source_entry"])): row
        for row in left["records"]
    }
    right_records = {
        (row["dataset_file_uid"], int(row["source_entry"])): row
        for row in right["records"]
    }
    mismatches = []
    if set(left_records) != set(right_records):
        mismatches.append({
            "field": "reference_keys",
            "left_only": sorted(set(left_records) - set(right_records)),
            "right_only": sorted(set(right_records) - set(left_records)),
        })
    compared = sorted(set(left_records) & set(right_records))
    fields = ("class_name", "domain", "density_bin", "n_candidates", "scalars", "arrays")
    for key in compared:
        for field in fields:
            if left_records[key][field] != right_records[key][field]:
                mismatches.append({
                    "dataset_file_uid": key[0], "source_entry": key[1],
                    "field": field, "left": left_records[key][field],
                    "right": right_records[key][field],
                })
    result = {
        "schema_version": 1,
        "valid": not mismatches,
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "reference_count": len(compared),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"Compared {len(compared)} references: "
        f"{len(mismatches)} mismatches; valid={result['valid']}"
    )
    return int(bool(mismatches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    read = subparsers.add_parser("read")
    read.add_argument("--manifest", required=True)
    read.add_argument("--backend", choices=("pyroot", "uproot"), required=True)
    read.add_argument("--max-open-files", type=int, default=3)
    read.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "read":
        report = read_report(args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            f"Read {report['reference_count']} events with {args.backend}; "
            f"max cached files={report['max_cached_files_observed']}"
        )
        return 0
    return compare_reports(args)


if __name__ == "__main__":
    raise SystemExit(main())
