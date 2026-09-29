import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from data_driven_3d_dataset import configured_density_bins, density_bin_index, stable_uid
from passing_muon_selection import load_configured_coverage_selection


CONFIG = REPO / "snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml"
V7 = REPO / "snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v7.yaml"


class DatasetUtilityTest(unittest.TestCase):
    def test_density_bins_are_half_open_and_exhaustive(self):
        config = yaml.safe_load(CONFIG.read_text())
        bins = configured_density_bins(config)
        self.assertEqual(density_bin_index(-1, bins), -1)
        self.assertEqual(density_bin_index(float("nan"), bins), -1)
        self.assertEqual(density_bin_index(0, bins), 0)
        self.assertEqual(density_bin_index(499.999, bins), 0)
        self.assertEqual(density_bin_index(500, bins), 1)
        self.assertEqual(density_bin_index(2000, bins), 2)
        self.assertEqual(density_bin_index(5000, bins), 3)
        self.assertEqual(density_bin_index(10000, bins), 4)
        self.assertEqual(density_bin_index(1e12, bins), 4)

    def test_stable_uid(self):
        first = stable_uid("a", 1, "/tmp/file.root")
        self.assertEqual(first, stable_uid("a", 1, "/tmp/file.root"))
        self.assertNotEqual(first, stable_uid("a", 2, "/tmp/file.root"))

    def test_new_selection_matches_v7_implementation(self):
        selected = load_configured_coverage_selection(V7, "coverage_loose")
        evaluation_dir = REPO / "evaluation_region_partitions"
        sys.path.insert(0, str(evaluation_dir))
        try:
            old_path = evaluation_dir / "make_passing_muon_binary_region_hist_v5.py"
            spec = importlib.util.spec_from_file_location("old_v7_selection", old_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            old_config = yaml.safe_load(V7.read_text())
            old = module.configured_selections(old_config)["coverage_loose"]
        finally:
            sys.path.remove(str(evaluation_dir))
        self.assertEqual(selected["expression"], old["expression"])
        self.assertEqual(list(selected["components"].items()), list(old["components"].items()))


if __name__ == "__main__":
    unittest.main()
