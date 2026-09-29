import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))
sys.path.insert(0, str(REPO / "train_3d/src"))

from build_3d_cnn_fold_views import build_fold_views
from summarize_muondis_density_sampling import build_report
from train_3d.data.muondis_density_sampler import UniformDensityBinSampler


class FoldViewsAndSamplerTest(unittest.TestCase):
    def test_views_keep_validation_natural_and_training_uniform(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = []
            density_rows = []
            for fold in range(5):
                uid = f"dsf_{fold}"
                index_path = directory / f"{uid}.npz"
                np.savez_compressed(
                    index_path,
                    bin_0=np.asarray([0], dtype=np.int64),
                    bin_1=np.asarray([1, 2, 3], dtype=np.int64),
                    invalid=np.asarray([], dtype=np.int64),
                )
                rows.append({
                    "dataset_file_uid": uid,
                    "fold": fold,
                    "class_name": "muon_DIS",
                    "domain": "data" if fold < 3 else "mc",
                    "hit3d_path": f"hits_{fold}.root",
                })
                density_rows.append({
                    "dataset_file_uid": uid,
                    "index_path": str(index_path),
                })
            views = build_fold_views(pd.DataFrame(rows), pd.DataFrame(density_rows), 5)
            selected = views.loc[views["cv_fold"].eq(0)]
            held_out = selected["fold"].eq(0)
            self.assertEqual(set(selected.loc[held_out, "split"]), {"validation"})
            self.assertEqual(set(selected.loc[held_out, "sampling_strategy"]), {"natural"})
            training = selected.loc[selected["split"].eq("train")]
            self.assertEqual(
                set(training["sampling_strategy"]),
                {"uniform_populated_density_bins"},
            )

            first = UniformDensityBinSampler(training, seed=7).sample(30)
            second = UniformDensityBinSampler(training, seed=7).sample(30)
            self.assertEqual(first, second)
            self.assertEqual({item.density_bin for item in first}, {"bin_0", "bin_1"})

            large_sample = UniformDensityBinSampler(training, seed=11).sample(20_000)
            low_fraction = sum(item.density_bin == "bin_0" for item in large_sample) / len(large_sample)
            self.assertAlmostEqual(low_fraction, 0.5, delta=0.02)

            data_only = UniformDensityBinSampler(
                training,
                seed=9,
                domain_probabilities={"data": 1.0, "mc": 0.0},
            ).sample(20)
            self.assertEqual({item.domain for item in data_only}, {"data"})

    def test_run8329_sampling_report(self):
        config = {
            "density_sampling": {"bins": [
                {"name": "low", "low": 0, "high": 100},
                {"name": "high", "low": 100, "high": None},
            ]}
        }
        audits = pd.DataFrame([
            {"domain": "data", "run": 8329, "fail_density_counts": json.dumps(
                {"low": 90, "high": 10, "invalid": 0}
            )},
            {"domain": "data", "run": 9000, "fail_density_counts": json.dumps(
                {"low": 1000, "high": 0, "invalid": 0}
            )},
            {"domain": "mc", "run": "", "fail_density_counts": json.dumps(
                {"low": 20, "high": 20, "invalid": 0}
            )},
        ])
        report = build_report(audits, config)
        data = report.loc[report["cohort"].eq("data_run8329")].set_index("density_bin")
        self.assertEqual(data.loc["low", "n_events"], 90)
        self.assertAlmostEqual(data.loc["low", "natural_fraction"], 0.9)
        self.assertAlmostEqual(data.loc["low", "uniform_training_fraction"], 0.5)

        with self.assertRaisesRegex(ValueError, "data_run8329"):
            build_report(audits.loc[audits["domain"].eq("mc")], config)


if __name__ == "__main__":
    unittest.main()
