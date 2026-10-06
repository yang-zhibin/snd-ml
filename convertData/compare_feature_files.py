#!/usr/bin/env python3
"""Compare two feature ROOT files entrywise and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import ROOT


IDENTITY_BRANCHES = ("runId", "eventId", "eventIndex")


def branch_names(tree) -> set[str]:
    return {str(branch.GetName()) for branch in tree.GetListOfBranches()}


def branch_value(tree, branch_name: str):
    branch = tree.GetBranch(branch_name)
    value = getattr(tree, branch_name)
    if str(branch.GetClassName()):
        try:
            return list(value)
        except TypeError:
            return str(value)
    return value


def values_equal(left, right, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            values_equal(a, b, rel_tol, abs_tol) for a, b in zip(left, right)
        )
    if isinstance(left, (int, bool)) and isinstance(right, (int, bool)):
        return int(left) == int(right)
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def compare_tree(
    reference_tree,
    candidate_tree,
    tree_name: str,
    max_entries: int,
    allow_candidate_extra_entries: bool,
    rel_tol: float,
    abs_tol: float,
    expected_candidate_only: set[str],
) -> dict:
    reference_entries = int(reference_tree.GetEntries())
    candidate_entries = int(candidate_tree.GetEntries())
    reference_branches = branch_names(reference_tree)
    candidate_branches = branch_names(candidate_tree)
    reference_only = sorted(reference_branches - candidate_branches)
    candidate_only = sorted(candidate_branches - reference_branches)
    unexpected_candidate_only = sorted(set(candidate_only) - expected_candidate_only)
    missing_expected_additions = sorted(expected_candidate_only - set(candidate_only))
    mismatches = []
    if reference_only:
        mismatches.append({"kind": "reference_only_branches", "values": reference_only})
    if unexpected_candidate_only:
        mismatches.append(
            {"kind": "unexpected_candidate_only_branches", "values": unexpected_candidate_only}
        )
    if missing_expected_additions:
        mismatches.append(
            {"kind": "missing_expected_candidate_branches", "values": missing_expected_additions}
        )
    if allow_candidate_extra_entries:
        if candidate_entries < reference_entries:
            mismatches.append(
                {"kind": "entry_count", "reference": reference_entries, "candidate": candidate_entries}
            )
    elif candidate_entries != reference_entries:
        mismatches.append(
            {"kind": "entry_count", "reference": reference_entries, "candidate": candidate_entries}
        )

    entries = min(reference_entries, candidate_entries)
    if max_entries > 0:
        entries = min(entries, max_entries)
    common = sorted(reference_branches & candidate_branches)
    compared_values = 0
    for entry in range(entries):
        reference_tree.GetEntry(entry)
        candidate_tree.GetEntry(entry)
        for branch in common:
            left = branch_value(reference_tree, branch)
            right = branch_value(candidate_tree, branch)
            compared_values += 1
            if not values_equal(left, right, rel_tol, abs_tol):
                mismatches.append(
                    {
                        "kind": "value",
                        "tree": tree_name,
                        "entry": entry,
                        "branch": branch,
                        "reference": str(left)[:500],
                        "candidate": str(right)[:500],
                    }
                )
                if len(mismatches) >= 25:
                    break
        if len(mismatches) >= 25:
            break
    return {
        "tree": tree_name,
        "reference_entries": reference_entries,
        "candidate_entries": candidate_entries,
        "compared_entries": entries,
        "common_branch_count": len(common),
        "reference_only_branches": reference_only,
        "candidate_only_branches": candidate_only,
        "compared_values": compared_values,
        "mismatches": mismatches,
    }


def compare(args: argparse.Namespace) -> dict:
    reference = ROOT.TFile.Open(args.reference)
    candidate = ROOT.TFile.Open(args.candidate)
    if not reference or reference.IsZombie():
        raise RuntimeError(f"cannot open reference {args.reference}")
    if not candidate or candidate.IsZombie():
        raise RuntimeError(f"cannot open candidate {args.candidate}")
    expected_additions = {
        value for value in args.expected_candidate_only.split(",") if value
    }
    results = []
    for tree_name in args.tree:
        reference_tree = reference.Get(tree_name)
        candidate_tree = candidate.Get(tree_name)
        if not reference_tree or not candidate_tree:
            results.append(
                {
                    "tree": tree_name,
                    "mismatches": [{"kind": "missing_tree"}],
                }
            )
            continue
        additions = expected_additions if tree_name == "sndData" else set()
        results.append(
            compare_tree(
                reference_tree,
                candidate_tree,
                tree_name,
                args.max_entries,
                args.allow_candidate_extra_entries,
                args.rel_tol,
                args.abs_tol,
                additions,
            )
        )
    reference.Close()
    candidate.Close()
    mismatch_count = sum(len(result["mismatches"]) for result in results)
    return {
        "comparison_passed": mismatch_count == 0,
        "reference": args.reference,
        "candidate": args.candidate,
        "relative_tolerance": args.rel_tol,
        "absolute_tolerance": args.abs_tol,
        "allow_candidate_extra_entries": args.allow_candidate_extra_entries,
        "expected_candidate_only_branches": sorted(expected_additions),
        "trees": results,
        "mismatch_count": mismatch_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree", action="append", default=[])
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--allow-candidate-extra-entries", action="store_true")
    parser.add_argument("--expected-candidate-only", default="")
    parser.add_argument("--rel-tol", type=float, default=1e-12)
    parser.add_argument("--abs-tol", type=float, default=1e-12)
    args = parser.parse_args()
    if not args.tree:
        args.tree = ["sndData", "cutFlowSummary"]
    report = compare(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"comparison_passed": report["comparison_passed"], "mismatch_count": report["mismatch_count"]}))
    if not report["comparison_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
