#!/usr/bin/env python3
"""Intersect one source's passing-muon features with its real-data DQ feature file."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

def root_input(path: str) -> str:
    return f"root://eospublic.cern.ch/{path}" if path.startswith("/eos/") else path


def event_keys(run_values, event_values):
    """Return sortable exact `(runId,eventId)` records without lossy packing."""
    import numpy as np

    return np.rec.fromarrays(
        [np.asarray(run_values, dtype=np.int64), np.asarray(event_values, dtype=np.int64)],
        names=["runId", "eventId"],
    )


def write_audit(path: str, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-feature", required=True)
    parser.add_argument("--passing-features", nargs="+", required=True)
    parser.add_argument("--source-file-index", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--expected-passing-events", type=int, required=True)
    parser.add_argument("--tree-name", default="sndData")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--published-root")
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args()

    import numpy as np
    import ROOT

    real_rdf = ROOT.RDataFrame(args.tree_name, root_input(args.real_feature))
    real_columns = {str(column) for column in real_rdf.GetColumnNames()}
    missing = {"runId", "eventId"} - real_columns
    if missing:
        raise RuntimeError(f"Real DQ feature schema lacks identity branches: {sorted(missing)}")
    real_arrays = real_rdf.AsNumpy(["runId", "eventId"])
    real_keys = event_keys(real_arrays["runId"], real_arrays["eventId"])
    real_events = len(real_keys)
    duplicate_real = real_events - len(np.unique(real_keys))
    wrong_run_real = int(np.count_nonzero(np.asarray(real_arrays["runId"]) != args.run))
    if duplicate_real or wrong_run_real:
        raise RuntimeError(
            f"Invalid real DQ keys for source {args.source_file_index}: "
            f"duplicates={duplicate_real}, wrong_run={wrong_run_real}"
        )

    paths = ROOT.std.vector("string")()
    for path in args.passing_features:
        paths.push_back(root_input(path))
    rdf = ROOT.RDataFrame(args.tree_name, paths)
    columns = {str(column) for column in rdf.GetColumnNames()}
    missing = {"runId", "eventId"} - columns
    if missing:
        raise RuntimeError(f"Passing feature schema lacks identity branches: {sorted(missing)}")
    passing_arrays = rdf.AsNumpy(["runId", "eventId"])
    passing_keys = event_keys(passing_arrays["runId"], passing_arrays["eventId"])
    passing_seen = len(passing_keys)
    duplicate_passing = passing_seen - len(np.unique(passing_keys))
    wrong_run_passing = int(
        np.count_nonzero(np.asarray(passing_arrays["runId"]) != args.run)
    )
    if passing_seen != args.expected_passing_events:
        raise RuntimeError(
            f"Passing input count mismatch for source {args.source_file_index}: "
            f"observed={passing_seen}, expected={args.expected_passing_events}"
        )
    if duplicate_passing or wrong_run_passing:
        raise RuntimeError(
            f"Invalid passing keys for source {args.source_file_index}: "
            f"duplicates={duplicate_passing}, wrong_run={wrong_run_passing}"
        )
    keep_mask = np.isin(passing_keys, real_keys)
    passing_dq = int(np.count_nonzero(keep_mask))
    passing_rejected = passing_seen - passing_dq

    # A friend tree supplies an entry-aligned boolean mask.  This keeps the
    # expensive event matching vectorized while Snapshot preserves the full
    # passing-muon feature/track schema.
    Path(args.output_root).parent.mkdir(parents=True, exist_ok=True)
    mask_fd, mask_path = tempfile.mkstemp(
        prefix="run8329_dq_mask_", suffix=".root", dir=Path(args.output_root).parent
    )
    os.close(mask_fd)
    os.unlink(mask_path)
    # PyROOT in LCG_107 cannot map NumPy's bool dtype (reported as ``b1``)
    # through RDF.FromNumpy.  Store the entry-aligned mask as an unsigned
    # 32-bit integer (a dtype supported by this ROOT build) and apply it
    # explicitly as a boolean expression below.
    mask_rdf = ROOT.RDF.FromNumpy({"dq_accept": keep_mask.astype(np.uint32)})
    mask_snapshot = mask_rdf.Snapshot("dqMask", mask_path)
    if hasattr(mask_snapshot, "GetValue"):
        mask_snapshot.GetValue()
    passing_chain = ROOT.TChain(args.tree_name)
    for path in args.passing_features:
        if passing_chain.Add(root_input(path)) <= 0:
            raise RuntimeError(f"Could not add passing feature file: {path}")
    mask_chain = ROOT.TChain("dqMask")
    if mask_chain.Add(mask_path) <= 0:
        raise RuntimeError("Could not add temporary DQ mask tree")
    passing_chain.AddFriend(mask_chain, "dqMask")
    selected = ROOT.RDataFrame(passing_chain).Filter(
        "dqMask.dq_accept != 0", "real_data_dq_membership"
    )
    options = ROOT.RDF.RSnapshotOptions()
    options.fCompressionLevel = args.compression_level
    output_columns = ROOT.std.vector("string")()
    for column in rdf.GetColumnNames():
        output_columns.push_back(str(column))
    try:
        snapshot = selected.Snapshot(
            args.tree_name, args.output_root, output_columns, options
        )
        if hasattr(snapshot, "GetValue"):
            snapshot.GetValue()
    finally:
        if os.path.exists(mask_path):
            os.unlink(mask_path)

    published = args.published_root or args.output_root
    write_audit(args.output_audit, {
        "status": "pass",
        "run": args.run,
        "source_file_index": args.source_file_index,
        "real_feature_path": args.real_feature,
        "filtered_passing_feature_path": published,
        "real_dq_events": real_events,
        "passing_before_dq": passing_seen,
        "passing_after_dq": passing_dq,
        "passing_rejected_by_dq": passing_rejected,
        "passing_dq_acceptance": passing_dq / passing_seen if passing_seen else 0.0,
        "passing_reco_efficiency_after_dq": passing_dq / real_events if real_events else 0.0,
        "duplicate_real_keys": duplicate_real,
        "duplicate_passing_keys": duplicate_passing,
        "wrong_run_real": wrong_run_real,
        "wrong_run_passing": wrong_run_passing,
    })
    print(
        f"Source {args.source_file_index:03d}: DQ real={real_events}, "
        f"passing={passing_dq}/{passing_seen}"
    )


if __name__ == "__main__":
    main()
