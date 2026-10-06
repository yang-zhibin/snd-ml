#!/usr/bin/env python3

from __future__ import annotations

from argparse import Namespace
from array import array
import tempfile
import unittest
import sys
from pathlib import Path

import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_feature_files import compare


def write_file(path: Path, extra_branch: bool, offset: float = 0.0, entries: int = 3):
    output = ROOT.TFile(str(path), "RECREATE")
    for tree_name in ("sndData", "cutFlowSummary"):
        tree = ROOT.TTree(tree_name, "")
        run = array("i", [0])
        event = array("i", [0])
        index = array("i", [0])
        value = array("d", [0.0])
        source_domain = array("i", [1])
        tree.Branch("runId", run, "runId/I")
        tree.Branch("eventId", event, "eventId/I")
        tree.Branch("eventIndex", index, "eventIndex/I")
        tree.Branch("value", value, "value/D")
        if extra_branch and tree_name == "sndData":
            tree.Branch("source_domain_id", source_domain, "source_domain_id/I")
        for entry in range(entries):
            run[0] = 1
            event[0] = entry
            index[0] = entry
            value[0] = entry + offset
            tree.Fill()
        tree.Write()
    output.Close()


class CompareFeatureFilesTest(unittest.TestCase):
    def arguments(self, reference: Path, candidate: Path, **overrides):
        values = {
            "reference": str(reference),
            "candidate": str(candidate),
            "tree": ["sndData", "cutFlowSummary"],
            "max_entries": 0,
            "allow_candidate_extra_entries": False,
            "expected_candidate_only": "source_domain_id",
            "rel_tol": 1e-12,
            "abs_tol": 1e-12,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_expected_addition_and_common_values_pass(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            reference = Path(directory) / "reference.root"
            candidate = Path(directory) / "candidate.root"
            write_file(reference, False)
            write_file(candidate, True)
            report = compare(self.arguments(reference, candidate))
            self.assertTrue(report["comparison_passed"])
            self.assertEqual(report["trees"][0]["candidate_only_branches"], ["source_domain_id"])

    def test_value_mismatch_fails(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            reference = Path(directory) / "reference.root"
            candidate = Path(directory) / "candidate.root"
            write_file(reference, False)
            write_file(candidate, True, offset=0.1)
            report = compare(self.arguments(reference, candidate))
            self.assertFalse(report["comparison_passed"])
            self.assertGreater(report["mismatch_count"], 0)

    def test_prefix_comparison_allows_full_remote_file(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            reference = Path(directory) / "reference.root"
            candidate = Path(directory) / "candidate.root"
            write_file(reference, True, entries=2)
            write_file(candidate, True, entries=3)
            report = compare(
                self.arguments(
                    reference,
                    candidate,
                    max_entries=2,
                    allow_candidate_extra_entries=True,
                    expected_candidate_only="",
                )
            )
            self.assertTrue(report["comparison_passed"])


if __name__ == "__main__":
    unittest.main()
