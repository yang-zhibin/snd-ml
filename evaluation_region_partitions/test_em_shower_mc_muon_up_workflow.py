#!/usr/bin/env python3
"""Focused configuration tests for the EM-shower MC-muon-up extension."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))

from prepare_em_shower_mc_muon_up_metadata import prepare  # noqa: E402


class EmShowerMcMuonUpWorkflowTest(unittest.TestCase):
    def setUp(self):
        config_dir = REPO_ROOT / "snakemake/metadata/configs/em_shower"
        self.muon_path = config_dir / "mc_muon_up_v1.yaml"
        self.eval_path = config_dir / "evaluation_options_muon_up_v1.yaml"
        self.partition_path = config_dir / "partitions_v1.yaml"
        self.hist_path = config_dir / "hist_features_v1.yaml"
        self.muon = yaml.safe_load(self.muon_path.read_text())
        self.evaluation = yaml.safe_load(self.eval_path.read_text())
        self.partitions = yaml.safe_load(self.partition_path.read_text())
        self.hists = yaml.safe_load(self.hist_path.read_text())

    def test_comparison_contract(self):
        self.assertEqual(self.muon["expected_source_files"], 15)
        self.assertEqual(
            set(self.muon["productions"]),
            {"scoring_1p8", "scoring_2", "scoring_2p5", "scoring_at449m"},
        )
        self.assertEqual(
            self.evaluation["comparisons"]["particle_groups"],
            ["CC_nue", "CC_numu", "NC_nue", "NC_numu", "MC_muon_up"],
        )
        self.assertEqual(
            self.evaluation["plot_features"],
            ["density_sndsw_scifi", "scifi_cluster_qdc_per_hit"],
        )
        self.assertEqual(self.evaluation["plot_features"], self.muon["plot_features"])
        self.assertEqual(
            self.evaluation["histogram_input_version"],
            "em_shower_evaluation_scifi35_energy_v2",
        )

    def test_local_outputs_use_work_volume(self):
        output_root = "/afs/cern.ch/work/z/zhibin/snd-ml-outputs/em_shower_neutrino_v1"
        self.assertEqual(
            self.muon["outputs"]["prepared_metadata"],
            f"{output_root}/metadata/mc_muon_up/source_features.csv",
        )
        self.assertEqual(
            self.muon["outputs"]["partition_metadata"],
            f"{output_root}/metadata/mc_muon_up/partitions.csv",
        )
        self.assertEqual(
            self.muon["outputs"]["comparison_metadata"],
            f"{output_root}/metadata/partitions_with_mc_muon_up.csv",
        )
        self.assertEqual(
            self.evaluation["metadata"]["partition_csv"],
            f"{output_root}/metadata/partitions_with_mc_muon_up.csv",
        )
        self.assertEqual(
            self.evaluation["outputs"]["plot_dir"],
            f"{output_root}/plots_muon_up",
        )

    def test_muon_partition_rule(self):
        rule = self.partitions["particle_group_rules"]["MC_muon_up"]
        self.assertEqual(self.partitions["classification"]["labels"]["MC_muon_up"], 5)
        self.assertEqual(rule["source_data_type"], "MC_muon")
        self.assertEqual(rule["source_subfolders"], ["up"])
        self.assertEqual(rule["particle_family"], "muon")
        self.assertEqual(rule["event_filter"], "std::abs(pdgCode) == 13")

    def test_histogram_feature_sets(self):
        excluded_features = set(self.muon["histograms"]["excluded_features"])
        excluded_pairs = set(self.muon["histograms"]["excluded_pairs"])
        self.assertEqual(
            excluded_features, {"mc_incoming_energy", "mc_incoming_log10_energy"}
        )
        self.assertEqual(len(self.hists["features"]), 30)
        self.assertEqual(len(set(self.hists["features"]) - excluded_features), 28)
        self.assertEqual(len(self.hists["features2d"]), 16)
        self.assertEqual(len(set(self.hists["features2d"]) - excluded_pairs), 7)

    def test_metadata_normalization(self):
        productions = list(self.muon["productions"].values())
        rows = []
        for index in range(15):
            production = productions[index % len(productions)]
            rows.append(
                {
                    "data_type": "MC_muon",
                    "subfolder": "up",
                    "partition": production,
                    "digi_path": f"/input/{production}/digi_{index}.root",
                    "geo_path": f"/input/{production}/geo.root",
                    "output_base_path": f"/old/{production}",
                    "feature_path": f"old_feature_{index}.root",
                    "n_event": 10,
                }
            )
        with tempfile.TemporaryDirectory(prefix="em_shower_muon_metadata_") as directory:
            directory = Path(directory)
            source = directory / "source.csv"
            output = directory / "prepared.csv"
            config = directory / "config.yaml"
            pd.DataFrame(rows).to_csv(source, index=False)
            local_config = dict(self.muon)
            local_config["source_metadata"] = str(source)
            local_config["outputs"] = dict(self.muon["outputs"])
            local_config["outputs"]["feature_root"] = str(directory / "features")
            config.write_text(yaml.safe_dump(local_config), encoding="utf-8")
            result = prepare(config, output)

        self.assertEqual(len(result), 15)
        self.assertEqual(set(result["production"]), set(self.muon["productions"]))
        self.assertEqual(result["source_key"].nunique(), 15)
        self.assertEqual(result["feature_path"].nunique(), 15)
        self.assertTrue(result["feature_path"].str.contains("feature_scifi_cluster_v1").all())
        self.assertTrue((result["input_tree"] == "cbmsim").all())

    def test_rule_and_plotter_wiring(self):
        rules = (
            REPO_ROOT / "snakemake/rules/em_shower_neutrino_evaluation.smk"
        ).read_text(encoding="utf-8")
        for name in (
            "em_shower_prepare_mc_muon_up_metadata_v1",
            "em_shower_mc_muon_up_feature_v1",
            "em_shower_mc_muon_up_partition_v1",
            "em_shower_mc_muon_up_hist_v1",
            "em_shower_mc_muon_up_hist2d_v1",
            "em_shower_neutrino_muon_up_plot_v1",
            "em_shower_neutrino_muon_up_comparison_v1",
        ):
            self.assertIn(name, rules)
        self.assertIn("--selection-mode direct_input", rules)
        self.assertIn("EM_SHOWER_MUON_PLOT_FEATURES", rules)

        plotter = (
            REPO_ROOT / "evaluation_region_partitions/plot_comparison.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def histogram_input_version", plotter)
        self.assertGreaterEqual(plotter.count("histogram_input_version(eval_cfg)"), 4)


if __name__ == "__main__":
    unittest.main()
