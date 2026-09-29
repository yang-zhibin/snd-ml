import sys
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from assign_3d_cnn_folds import assign_groups, group_vectors


class FoldAssignmentTest(unittest.TestCase):
    def make_catalog(self):
        rows = []
        classes = ["CC_numu", "CC_nue", "NC", "neutral_kaon", "neutron",
                   "passing_muon", "muon_DIS"]
        for group_index in range(35):
            class_name = classes[group_index % len(classes)]
            rows.append({
                "fold_group_uid": f"group_{group_index:03d}",
                "class_name": class_name,
                "domain": "data" if class_name in {"passing_muon", "muon_DIS"} and group_index % 2 else "mc",
                "n_events": 10 + group_index,
                "energy_bin": "(10, 20)" if class_name in {"neutral_kaon", "neutron"} else "",
                "density_counts": '{"density_0_100": 3, "density_ge5000": 2}' if class_name == "muon_DIS" else "",
                "production": "production_a" if group_index % 2 else "production_b",
                "lumi_source_uid": f"lumi_{group_index:03d}",
                "lumi_per_file": 0.01 * (group_index + 1),
            })
        return pd.DataFrame(rows)

    def test_assignment_is_deterministic_and_group_exclusive(self):
        catalog = self.make_catalog()
        first = assign_groups(catalog, 5, 123)
        second = assign_groups(catalog.sample(frac=1, random_state=7), 5, 123)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), set(range(5)))
        self.assertEqual(len(first), catalog.fold_group_uid.nunique())
        assigned = catalog.assign(
            fold=catalog.fold_group_uid.map(first),
        )
        for class_name, rows in assigned.groupby("class_name"):
            self.assertEqual(
                set(rows["fold"]), set(range(5)),
                f"{class_name} lacks complete fold coverage",
            )

    def test_sparse_seed_groups_do_not_strand_folds(self):
        catalog = self.make_catalog()
        # Reproduce the shape that exposed the old greedy cold-start bug:
        # two multi-class groups are much larger than all single-class groups.
        extra = pd.DataFrame([
            {
                "fold_group_uid": "large_muon_mc", "class_name": class_name,
                "domain": "mc", "n_events": n_events, "energy_bin": "",
                "density_counts": "", "production": "muon_mc",
                "lumi_source_uid": "large_muon_mc", "lumi_per_file": 1.0,
            }
            for class_name, n_events in [("passing_muon", 3_000), ("muon_DIS", 90_000)]
        ])
        catalog = pd.concat([catalog, extra], ignore_index=True)
        assignments = assign_groups(catalog, 5, 123)
        assigned = catalog.assign(fold=catalog.fold_group_uid.map(assignments))
        coverage = assigned.groupby("class_name")["fold"].nunique()
        self.assertTrue(coverage.eq(5).all(), coverage.to_dict())

    def test_class_with_too_few_independent_groups_fails(self):
        catalog = self.make_catalog()
        catalog = catalog.loc[
            ~catalog["class_name"].eq("CC_nue")
            | catalog["fold_group_uid"].isin(["group_001", "group_008"])
        ]
        with self.assertRaisesRegex(ValueError, "CC_nue.*2 independent fold groups"):
            assign_groups(catalog, 5, 123)

    def test_too_few_groups_fails(self):
        with self.assertRaises(ValueError):
            assign_groups(self.make_catalog().iloc[:3], 5, 123)

    def test_luminosity_is_counted_once_per_original_source(self):
        catalog = pd.DataFrame([
            {
                "fold_group_uid": "group", "class_name": "CC_nue", "domain": "mc",
                "n_events": 20, "lumi_source_uid": "source", "lumi_per_file": 2.5,
            },
            {
                "fold_group_uid": "group", "class_name": "NC", "domain": "mc",
                "n_events": 10, "lumi_source_uid": "source", "lumi_per_file": 2.5,
            },
        ])
        self.assertEqual(group_vectors(catalog)["group"]["luminosity::mc"], 2.5)


if __name__ == "__main__":
    unittest.main()
