import importlib.util
import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation_data_driven_3d_cnn"))

from prepare_muondis_feature_sampling import prepare_manifests


CLASSES = (
    "CC_numu", "CC_nue", "NC", "neutral_kaon", "neutron",
    "passing_muon", "muon_DIS",
)


class MuonDISFeatureSamplingTest(unittest.TestCase):
    def training_config(self):
        with (REPO / "train_3d/configs/classifier_pyg_gravnet_v1.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        config["data"]["epoch_size"] = 700
        return config

    def test_prepare_manifest_is_reproducible_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            density = tmp / "density.npz"
            np.savez_compressed(
                density,
                bin_0=np.arange(5, dtype=np.int32),
                bin_1=np.arange(5, 10, dtype=np.int32),
                invalid=np.asarray([], dtype=np.int32),
            )
            rows = []
            for class_id, class_name in enumerate(CLASSES):
                rows.append({
                    "cv_fold": 0, "split": "train", "fold": 1,
                    "dataset_file_uid": f"dsf_{class_id:020x}",
                    "source_uid": f"src_{class_id}",
                    "hit3d_path": f"{class_name}.root",
                    "feature_path": f"{class_name}_feature.root",
                    "class_id": class_id, "class_name": class_name,
                    "domain": "data" if class_name == "muon_DIS" else "mc",
                    "n_events": 10, "event_filter": "true",
                    "entry_index_path": "",
                    "density_index_path": str(density) if class_name == "muon_DIS" else "",
                    "sampling_strategy": (
                        "uniform_populated_density_bins"
                        if class_name == "muon_DIS" else "natural"
                    ),
                })
            views = pd.DataFrame(rows)
            first = prepare_manifests(self.training_config(), views, 0, 3)
            second = prepare_manifests(self.training_config(), views, 0, 3)
            pd.testing.assert_frame_equal(first[0], second[0])
            pd.testing.assert_frame_equal(first[1], second[1])
            self.assertEqual(first[2], second[2])
            sources, sampled, summary = first
            self.assertEqual(len(sources), 1)
            self.assertEqual(int(sources.iloc[0]["valid_density_events"]), 10)
            self.assertEqual(int(sampled["multiplicity"].sum()), summary["muondis_draws"])
            self.assertEqual(sum(summary["class_counts"].values()), 700)
            self.assertEqual(summary["natural_invalid_events"], 0)

    @unittest.skipUnless(importlib.util.find_spec("ROOT"), "ROOT unavailable")
    def test_root_shard_merge_and_two_panel_pdf(self):
        import ROOT
        from build_muondis_feature_hist_shard import build_shard
        from merge_muondis_feature_hist_shards import merge_shards
        from plot_muondis_feature_sampling import plot_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            feature_path = tmp / "feature.root"
            root_file = ROOT.TFile.Open(str(feature_path), "RECREATE")
            tree = ROOT.TTree("sndData", "sndData")
            x, y, density = array("d", [0]), array("d", [0]), array("d", [0])
            tree.Branch("x", x, "x/D")
            tree.Branch("y", y, "y/D")
            tree.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")
            for index, value in enumerate((10.0, 20.0, 30.0, 40.0, -1.0)):
                x[0], y[0], density[0] = value, value / 10.0, value
                tree.Fill()
            tree.Write()
            root_file.Close()

            density_path = tmp / "density.npz"
            np.savez_compressed(
                density_path,
                bin_0=np.asarray([0, 1], dtype=np.int32),
                bin_1=np.asarray([2, 3], dtype=np.int32),
                invalid=np.asarray([4], dtype=np.int32),
            )
            uid = "dsf_00000000000000000001"
            sources = tmp / "sources.csv"
            pd.DataFrame([{
                "dataset_file_uid": uid, "source_uid": "src_1", "domain": "data",
                "fold": 1, "feature_path": str(feature_path), "hit3d_path": "unused",
                "density_index_path": str(density_path), "n_events": 5,
                "event_filter": "true", "valid_density_events": 4,
                "invalid_density_events": 1,
            }]).to_csv(sources, index=False)
            samples = tmp / "samples.csv"
            pd.DataFrame([
                {"dataset_file_uid": uid, "source_entry": 1, "domain": "data",
                 "density_bin": "bin_0", "multiplicity": 2},
                {"dataset_file_uid": uid, "source_entry": 3, "domain": "data",
                 "density_bin": "bin_1", "multiplicity": 1},
            ]).to_csv(samples, index=False)
            dataset_config = {
                "density_sampling": {
                    "variable": "density_sndsw_scifi",
                    "bins": [
                        {"name": "low", "low": 0, "high": 25},
                        {"name": "high", "low": 25, "high": None},
                    ],
                }
            }
            hist_config = {
                "defaults": {"fold_underflow": True, "fold_overflow": True},
                "definitions": {},
                "features": {
                    "x": {"dimension": 1, "expression": "x", "n_bins": 4,
                          "x_min": 0, "x_max": 40, "axis_title": "x", "logy": False},
                    "xy": {"dimension": 2, "x_expression": "x", "y_expression": "y",
                           "x_bins": 4, "x_min": 0, "x_max": 40,
                           "y_bins": 4, "y_min": 0, "y_max": 4,
                           "x_title": "x", "y_title": "y", "logz": False},
                },
            }
            region_config = {"derived_columns": {}, "diagnostic_features": {}}
            shard = tmp / "shard.root"
            result = build_shard(
                sources, samples, uid, dataset_config, hist_config,
                region_config, "sndData", shard,
            )
            self.assertEqual(result["natural_events"], 4)
            self.assertEqual(result["sampled_draws"], 3)

            sampling_summary = tmp / "sampling.json"
            sampling_summary.write_text(json.dumps({
                "version": "muondis_feature_sampling_v1", "cv_fold": 0,
                "epoch": 0, "seed": 42, "epoch_size": 21,
                "sampling_stage": "hierarchical_pre_empty_scifi_filter",
                "natural_valid_events": 4, "natural_invalid_events": 1,
                "natural_domain_events": {"data": 4}, "muondis_draws": 3,
                "muondis_unique_events": 2, "muondis_repeated_draws": 1,
                "domain_draws": {"data": 3}, "density_draws": {},
                "class_counts": {"muon_DIS": 3}, "source_files": 1,
            }))
            merged = tmp / "merged.root"
            audit = tmp / "audit.csv"
            summary = tmp / "summary.json"
            merge_shards(
                [shard], sources, sampling_summary, hist_config, region_config,
                merged, audit, summary,
            )
            merged_file = ROOT.TFile.Open(str(merged))
            self.assertEqual(int(merged_file.Get("inclusive/natural/x").Integral()), 4)
            self.assertEqual(int(merged_file.Get("inclusive/sampled/x").Integral()), 3)
            merged_file.Close()
            pdf = tmp / "report.pdf"
            self.assertEqual(
                plot_report(merged, summary, hist_config, region_config, pdf), 3
            )
            self.assertGreater(pdf.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
