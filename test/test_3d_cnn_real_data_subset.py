import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from build_real_data_run_subset import build_subset


class RealDataRunSubsetTest(unittest.TestCase):
    def make_metadata(self, directory: Path):
        rows = []
        for run in (100, 200):
            for file_number in range(12):
                feature = directory / f"feature_run{run}_sndsw_raw-{file_number:04d}.root"
                hit3d = directory / f"hit3d_run{run}_sndsw_raw-{file_number:04d}.root"
                feature.touch()
                hit3d.touch()
                rows.append({
                    "data_type": "real_data",
                    "run": run,
                    "partition": f"run_{run}",
                    "subfolder": "test",
                    "n_event": 1000,
                    "lumi_per_file": 0.1,
                    "output_base_path": str(directory),
                    "digi_path": f"sndsw_raw-{file_number:04d}.root",
                    "feature_path": str(feature),
                    "hit3d_path": str(hit3d),
                })
        return pd.DataFrame(rows)

    def test_exact_deterministic_stratified_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.make_metadata(Path(tmp))
            first, summary = build_subset(
                metadata, files_per_run=4, seed=17, workers=2, expected_runs=2
            )
            second, _ = build_subset(
                metadata.sample(frac=1, random_state=9),
                files_per_run=4, seed=17, workers=2, expected_runs=2,
            )
            self.assertEqual(len(first), 8)
            self.assertTrue(first.groupby("run").size().eq(4).all())
            self.assertTrue(first.groupby("run")["ml_run_stratum"].nunique().eq(4).all())
            self.assertEqual(
                set(first["feature_path"].astype(str)),
                set(second["feature_path"].astype(str)),
            )
            self.assertTrue(summary["selected_files"].eq(4).all())
            self.assertTrue(summary["selected_original_events"].eq(4000).all())

    def test_missing_products_are_excluded_before_stratification(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.make_metadata(Path(tmp))
            Path(metadata.loc[0, "hit3d_path"]).unlink()
            selected, summary = build_subset(
                metadata, files_per_run=4, seed=17, workers=1, expected_runs=2
            )
            self.assertNotIn(metadata.loc[0, "feature_path"], set(selected["feature_path"]))
            run_summary = summary.set_index("run")
            self.assertEqual(run_summary.loc["100", "available_feature_hit3d_pairs"], 11)

    def test_insufficient_run_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.make_metadata(Path(tmp))
            with self.assertRaisesRegex(ValueError, "only 12 available"):
                build_subset(metadata, files_per_run=13, seed=17, workers=1)


if __name__ == "__main__":
    unittest.main()
