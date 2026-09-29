#!/usr/bin/env python3
"""Compare legacy-consolidated and partition-input track histograms bin by bin."""

import argparse
import csv
from pathlib import Path

import ROOT


def histogram_names(root_file):
    return sorted(
        key.GetName()
        for key in root_file.GetListOfKeys()
        if key.GetClassName().startswith(("TH1", "TH2"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    reference = ROOT.TFile.Open(args.reference)
    candidate = ROOT.TFile.Open(args.candidate)
    if not reference or reference.IsZombie():
        raise RuntimeError(f"Cannot open reference file: {args.reference}")
    if not candidate or candidate.IsZombie():
        raise RuntimeError(f"Cannot open candidate file: {args.candidate}")
    reference_names = histogram_names(reference)
    candidate_names = histogram_names(candidate)
    if reference_names != candidate_names:
        raise RuntimeError(
            f"Histogram-name mismatch: reference={reference_names}, "
            f"candidate={candidate_names}"
        )

    rows = []
    failed = []
    for name in reference_names:
        left = reference.Get(name)
        right = candidate.Get(name)
        if left.GetNcells() != right.GetNcells():
            raise RuntimeError(f"Cell-count mismatch for {name}")
        max_content_difference = 0.0
        max_error_difference = 0.0
        for index in range(left.GetNcells()):
            max_content_difference = max(
                max_content_difference,
                abs(left.GetBinContent(index) - right.GetBinContent(index)),
            )
            max_error_difference = max(
                max_error_difference,
                abs(left.GetBinError(index) - right.GetBinError(index)),
            )
        status = (
            "pass"
            if max(max_content_difference, max_error_difference) <= args.tolerance
            else "fail"
        )
        rows.append(
            {
                "histogram": name,
                "status": status,
                "max_content_difference": max_content_difference,
                "max_error_difference": max_error_difference,
            }
        )
        if status == "fail":
            failed.append(name)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reference.Close()
    candidate.Close()
    if failed:
        raise RuntimeError(f"Histogram equivalence failed: {failed}")
    print(f"Exactly matched {len(rows)} histograms")


if __name__ == "__main__":
    main()
