#!/usr/bin/env python3
"""Create an inclusive cutFlowSummary mapping one-to-one to an event tree."""

from __future__ import annotations

import argparse
from array import array

import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-tree", default="cbmsim")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open input file {args.input}")
    try:
        event_tree = source.Get(args.input_tree)
        if not event_tree or not isinstance(event_tree, ROOT.TTree):
            raise RuntimeError(
                f"Missing input tree {args.input_tree!r} in {args.input}"
            )
        entries = int(event_tree.GetEntries())
    finally:
        source.Close()
    if entries <= 0:
        raise RuntimeError(f"Input tree {args.input_tree!r} is empty")

    output = ROOT.TFile.Open(args.output, "RECREATE")
    if not output or output.IsZombie():
        raise RuntimeError(f"Cannot create output file {args.output}")
    try:
        tree = ROOT.TTree(
            "cutFlowSummary", "Inclusive one-to-one input-entry mapping"
        )
        entry = array("q", [0])
        tree.Branch("entry", entry, "entry/L")
        for index in range(entries):
            entry[0] = index
            tree.Fill()
        tree.Write()
        ROOT.TNamed("selection_mode", "inclusive_identity").Write()
        ROOT.TParameter("Long64_t")("source_tree_entries", entries).Write()
    finally:
        output.Close()
    print(f"Wrote {entries} inclusive cutflow rows to {args.output}")


if __name__ == "__main__":
    main()
