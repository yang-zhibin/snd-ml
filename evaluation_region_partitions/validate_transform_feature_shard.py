#!/usr/bin/env python3
"""Validate transformed feature identities against the successful digi audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    with gzip.open(args.audit, "rt", newline="", encoding="utf-8") as stream:
        expected = [
            (int(row["runId"]), int(row["eventId"]))
            for row in csv.DictReader(stream) if row["status"] == "success"
        ]
    source = ROOT.TFile.Open(args.feature)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.feature}")
    tree = source.Get("sndData")
    if not tree:
        raise RuntimeError("Feature shard has no sndData tree")
    required = {"runId", "eventId", "count_veto", "count_scifi", "count_us", "count_ds"}
    branches = {branch.GetName() for branch in tree.GetListOfBranches()}
    if not required.issubset(branches):
        raise RuntimeError(f"Feature shard lacks branches {sorted(required - branches)}")
    if int(tree.GetEntries()) != len(expected):
        raise RuntimeError(f"Feature entries {tree.GetEntries()} != audit successes {len(expected)}")
    observed = []
    for entry in range(int(tree.GetEntries())):
        tree.GetEntry(entry)
        observed.append((int(tree.runId), int(tree.eventId)))
    source.Close()
    if observed != expected:
        raise RuntimeError("Feature identities/order do not match transformed digi audit")
    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"status": "pass", "events": len(observed)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(observed)} transformed feature rows")


if __name__ == "__main__":
    main()
