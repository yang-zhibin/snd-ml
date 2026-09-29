#!/usr/bin/env python3
"""Validate identity and count closure for a transformed Run-8329 shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import ROOT

from detid_channel_shift_sndsw import event_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()

    with open(args.summary, encoding="utf-8") as stream:
        summary = json.load(stream)
    with gzip.open(args.audit, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    successes = [row for row in rows if row["status"] == "success"]
    failures = [row for row in rows if row["status"] == "failed"]
    if len(rows) != int(summary["eligible_events"]):
        raise RuntimeError("Audit rows do not close to eligible event count")
    if len(successes) != int(summary["successful_events"]):
        raise RuntimeError("Audit successes do not close to summary")
    if len(failures) != int(summary["failed_events"]):
        raise RuntimeError("Audit failures do not close to summary")
    identities = [(int(row["runId"]), int(row["eventId"])) for row in successes]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Successful transformed identities are not unique")

    source = ROOT.TFile.Open(args.root)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.root}")
    tree = source.Get("cbmsim") or source.Get("rawConv")
    if not tree:
        raise RuntimeError("Transformed shard has no cbmsim/rawConv tree")
    if int(tree.GetEntries()) != len(successes):
        raise RuntimeError("Transformed tree entries do not close to successful audit rows")
    for entry, expected in enumerate(identities):
        if tree.GetEntry(entry) <= 0:
            raise RuntimeError(f"Cannot read transformed entry {entry}")
        if event_identity(tree) != expected:
            raise RuntimeError(
                f"Transformed identity mismatch at {entry}: {event_identity(tree)} != {expected}"
            )
    source.Close()
    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"status": "pass", **summary}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(successes)} successful and {len(failures)} failed events")


if __name__ == "__main__":
    main()
