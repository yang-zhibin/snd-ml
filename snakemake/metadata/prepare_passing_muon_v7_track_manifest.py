#!/usr/bin/env python3
"""Validate DQ-intersected flat-v2 track files and write the V7 manifest."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "convertData"))
from reco_track_features import (  # noqa: E402
    FLAT_V2,
    scalar_dtypes_for_schema,
    vector_dtypes_for_schema,
)


def root_input(path):
    path = str(path)
    # This rule is local and lxplus exposes experiment EOS through FUSE.  Use
    # that mount when available: it avoids making validation depend on DNS or
    # an XRootD redirector after all expensive upstream jobs have completed.
    # Keep the redirector fallback for worker environments without the mount.
    if path.startswith("/eos/") and not Path(path).is_file():
        return f"root://eospublic.cern.ch/{path}"
    return path


def atomic_csv(frame, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect(path, tree_name, expected_entries, required_branches):
    source = ROOT.TFile.Open(root_input(path))
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open V7 flat-v2 input {path}")
    try:
        tree = source.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Missing {tree_name} in {path}")
        entries = int(tree.GetEntries())
        if entries != expected_entries:
            raise RuntimeError(
                f"V7 track input count mismatch for {path}: "
                f"{entries} != {expected_entries}"
            )
        branches = {branch.GetName() for branch in tree.GetListOfBranches()}
        missing = required_branches - branches
        if missing:
            # ROOT RDataFrame Snapshot produces a branchless TTree when the
            # selected frame has zero entries.  Its event count has already
            # been checked against the DQ audit above, and it contributes no
            # data to the downstream chain.  Keep strict full-schema checking
            # for every non-empty source.
            if expected_entries == 0 and not branches:
                return 0
            raise RuntimeError(
                f"V7 track input {path} lacks flat_v2 branches: {sorted(missing)}"
            )
        return len(branches)
    finally:
        source.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-metadata", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", type=int, default=8329)
    parser.add_argument("--tree-name", default="sndData")
    args = parser.parse_args()

    metadata = pd.read_csv(args.track_metadata, keep_default_na=False)
    required_columns = {
        "run",
        "source_file_index",
        "n_saved_event",
        "track_feature_schema",
    }
    missing = required_columns - set(metadata)
    if missing:
        raise ValueError(f"Track metadata lacks columns: {sorted(missing)}")
    metadata = metadata.loc[pd.to_numeric(metadata["run"]) == args.run].copy()
    if metadata.empty:
        raise ValueError(f"No flat-v2 track metadata for run {args.run}")
    schemas = set(metadata["track_feature_schema"].astype(str))
    if schemas != {FLAT_V2}:
        raise RuntimeError(f"Unexpected source track schemas: {sorted(schemas)}")
    expected_by_source = (
        metadata.assign(
            source_file_index=pd.to_numeric(metadata["source_file_index"]).astype(int),
            n_saved_event=pd.to_numeric(metadata["n_saved_event"]).astype(int),
        )
        .groupby("source_file_index")["n_saved_event"]
        .sum()
        .to_dict()
    )

    audit_frames = [pd.read_csv(path, keep_default_na=False) for path in args.audits]
    audits = pd.concat(audit_frames, ignore_index=True)
    required_audit_columns = {
        "status",
        "run",
        "source_file_index",
        "passing_before_dq",
        "passing_after_dq",
        "filtered_passing_feature_path",
        "passing_dq_acceptance",
    }
    missing = required_audit_columns - set(audits)
    if missing:
        raise ValueError(f"V7 DQ audits lack columns: {sorted(missing)}")
    if len(audits) != len(expected_by_source):
        raise RuntimeError(
            f"Expected {len(expected_by_source)} V7 audits, found {len(audits)}"
        )
    audits["source_file_index"] = pd.to_numeric(
        audits["source_file_index"]
    ).astype(int)
    if audits["source_file_index"].duplicated().any():
        raise RuntimeError("Duplicate source_file_index in V7 audits")
    if set(audits["source_file_index"]) != set(expected_by_source):
        raise RuntimeError("V7 audit source coverage does not match track metadata")
    if set(audits["status"].astype(str)) != {"pass"}:
        raise RuntimeError("One or more V7 DQ audits did not pass")
    if set(pd.to_numeric(audits["run"]).astype(int)) != {args.run}:
        raise RuntimeError("V7 DQ audits contain an unexpected run")

    required_branches = set(scalar_dtypes_for_schema(FLAT_V2)) | set(
        vector_dtypes_for_schema(FLAT_V2)
    )
    rows = []
    for audit in audits.sort_values("source_file_index").itertuples(index=False):
        source_index = int(audit.source_file_index)
        before = int(audit.passing_before_dq)
        after = int(audit.passing_after_dq)
        expected = int(expected_by_source[source_index])
        if before != expected:
            raise RuntimeError(
                f"V7 source {source_index} pre-DQ mismatch: {before} != {expected}"
            )
        if after < 0 or after > before:
            raise RuntimeError(
                f"V7 source {source_index} has invalid DQ counts: {after}/{before}"
            )
        acceptance = float(audit.passing_dq_acceptance)
        expected_acceptance = float(after) / before if before else 0.0
        if abs(acceptance - expected_acceptance) > 1e-12:
            raise RuntimeError(
                f"V7 source {source_index} has inconsistent DQ acceptance: "
                f"{acceptance} != {expected_acceptance}"
            )
        path = str(audit.filtered_passing_feature_path)
        branch_count = inspect(
            path, args.tree_name, after, required_branches
        )
        rows.append({
            "run": args.run,
            "source_file_index": source_index,
            "particle_group": "passing_muon_dq_run8329_flat_v2",
            "feature_path": path,
            "tree_name": args.tree_name,
            "n_events": after,
            "n_events_before_dq": before,
            "track_feature_schema": FLAT_V2,
            "branch_count": branch_count,
            "dq_acceptance": acceptance,
            "status": "pass",
        })
    result = pd.DataFrame(rows)
    atomic_csv(result, args.output)
    print(
        f"Validated {len(result)} V7 flat-v2 inputs with "
        f"{int(result['n_events'].sum())} DQ-intersected events"
    )


if __name__ == "__main__":
    main()
