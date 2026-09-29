#!/usr/bin/env python3
"""Build one exact RealDataDQ minus PassingMuonReco feature partition."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


KEY_COLUMNS = ("source_file_index", "runId", "eventId")


def root_input(path: str) -> str:
    return f"root://eospublic.cern.ch/{path}" if path.startswith("/eos/") else path


def event_keys(arrays):
    import numpy as np

    return np.rec.fromarrays(
        [np.asarray(arrays[name], dtype=np.int64) for name in KEY_COLUMNS],
        names=KEY_COLUMNS,
    )


def atomic_csv(path: str, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-partition", required=True)
    parser.add_argument("--passing-partition", required=True)
    parser.add_argument("--part-index", required=True, type=int)
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--tree-name", default="sndData")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--published-root")
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--compression-level", default=1, type=int)
    args = parser.parse_args()

    import numpy as np
    import ROOT

    real = ROOT.RDataFrame(args.tree_name, root_input(args.real_partition))
    passing = ROOT.RDataFrame(args.tree_name, root_input(args.passing_partition))
    real_columns = {str(column) for column in real.GetColumnNames()}
    passing_columns = {str(column) for column in passing.GetColumnNames()}
    if set(KEY_COLUMNS) - real_columns:
        raise RuntimeError(
            f"Real partition lacks keys: {sorted(set(KEY_COLUMNS) - real_columns)}"
        )
    if set(KEY_COLUMNS) - passing_columns:
        raise RuntimeError(
            f"Passing partition lacks keys: {sorted(set(KEY_COLUMNS) - passing_columns)}"
        )

    real_arrays = real.AsNumpy(list(KEY_COLUMNS))
    passing_arrays = passing.AsNumpy(list(KEY_COLUMNS))
    real_keys = event_keys(real_arrays)
    passing_keys = event_keys(passing_arrays)
    real_count = len(real_keys)
    passing_count = len(passing_keys)
    duplicate_real = real_count - len(np.unique(real_keys))
    duplicate_passing = passing_count - len(np.unique(passing_keys))
    wrong_run_real = int(np.count_nonzero(real_arrays["runId"] != args.run))
    wrong_run_passing = int(np.count_nonzero(passing_arrays["runId"] != args.run))
    if duplicate_real or duplicate_passing or wrong_run_real or wrong_run_passing:
        raise RuntimeError(
            "Invalid partition keys: "
            f"duplicate_real={duplicate_real}, duplicate_passing={duplicate_passing}, "
            f"wrong_run_real={wrong_run_real}, wrong_run_passing={wrong_run_passing}"
        )

    passing_in_real = np.isin(passing_keys, real_keys, assume_unique=True)
    unmatched_passing = int(np.count_nonzero(~passing_in_real))
    if unmatched_passing:
        raise RuntimeError(
            f"Part {args.part_index:03d} has {unmatched_passing} passing keys outside real DQ"
        )
    missing_mask = ~np.isin(real_keys, passing_keys, assume_unique=True)
    missing_count = int(np.count_nonzero(missing_mask))
    if passing_count + missing_count != real_count:
        raise RuntimeError("Real = passing + missing count closure failed")

    output_path = Path(args.output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Do not attach an entry-aligned mask as a friend. RDataFrame snapshots
    # produced with implicit multithreading carry TTree::kEntriesReshuffled,
    # and ROOT intentionally rejects entry-aligned friends for those trees.
    # The key anti-join above already gives the exact physical entry indices;
    # copy only those entries into a schema-preserving clone instead.
    input_file = ROOT.TFile.Open(root_input(args.real_partition), "READ")
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open real partition {args.real_partition}")
    input_tree = input_file.Get(args.tree_name)
    if not input_tree:
        input_file.Close()
        raise RuntimeError(
            f"Could not find {args.tree_name} in {args.real_partition}"
        )

    output_file = ROOT.TFile.Open(
        str(output_path), "RECREATE", "", args.compression_level
    )
    if not output_file or output_file.IsZombie():
        input_file.Close()
        raise RuntimeError(f"Could not create {output_path}")
    try:
        output_file.cd()
        output_tree = input_tree.CloneTree(0)
        output_tree.SetName(args.tree_name)
        copied = 0
        for entry_index in np.flatnonzero(missing_mask):
            if input_tree.GetEntry(int(entry_index)) < 0:
                raise RuntimeError(
                    f"Could not read real entry {int(entry_index)} in "
                    f"part {args.part_index:03d}"
                )
            if output_tree.Fill() < 0:
                raise RuntimeError(
                    f"Could not write real entry {int(entry_index)} in "
                    f"part {args.part_index:03d}"
                )
            copied += 1
        if copied != missing_count or int(output_tree.GetEntries()) != missing_count:
            raise RuntimeError(
                "Missing-event copy count mismatch: "
                f"copied={copied}, tree_entries={int(output_tree.GetEntries())}, "
                f"expected={missing_count}"
            )
        output_tree.Write("", ROOT.TObject.kOverwrite)
    finally:
        output_file.Close()
        input_file.Close()

    published = args.published_root or str(output_path)
    atomic_csv(
        args.output_audit,
        {
            "status": "pass",
            "run": args.run,
            "part_index": args.part_index,
            "real_partition_path": args.real_partition,
            "passing_partition_path": args.passing_partition,
            "missing_partition_path": published,
            "real_dq_events": real_count,
            "passing_muon_dq_events": passing_count,
            "no_passing_muon_reco_events": missing_count,
            "duplicate_real_keys": duplicate_real,
            "duplicate_passing_keys": duplicate_passing,
            "wrong_run_real": wrong_run_real,
            "wrong_run_passing": wrong_run_passing,
            "unmatched_passing_events": unmatched_passing,
            "count_closure": int(real_count == passing_count + missing_count),
        },
    )
    print(
        f"Part {args.part_index:03d}: real={real_count}, passing={passing_count}, "
        f"NoPassingMuonReco={missing_count}"
    )


if __name__ == "__main__":
    main()
