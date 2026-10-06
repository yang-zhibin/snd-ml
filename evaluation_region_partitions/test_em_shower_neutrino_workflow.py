#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "convertData"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_partition_metadata import combine  # noqa: E402
from prepare_em_shower_neutrino_metadata import prepare_metadata  # noqa: E402
from build_region_partitions import extract_identifiers  # noqa: E402
from scifi_plane_clustering import scalar_feature_dtypes  # noqa: E402


class EmShowerNeutrinoWorkflowTest(unittest.TestCase):
    def setUp(self):
        config_dir = REPO_ROOT / "snakemake/metadata/configs/em_shower"
        self.production = yaml.safe_load((config_dir / "production_v1.yaml").read_text())
        self.regions = yaml.safe_load((config_dir / "regions_v1.yaml").read_text())
        self.partitions = yaml.safe_load((config_dir / "partitions_v1.yaml").read_text())
        self.hists = yaml.safe_load((config_dir / "hist_features_v1.yaml").read_text())
        self.evaluation = yaml.safe_load((config_dir / "evaluation_options_v1.yaml").read_text())

    def test_configs_reference_emitted_scalar_branches(self):
        branches = {name for name, _ in scalar_feature_dtypes()}
        branches.update(
            {
                "count_scifi",
                "qdc_scifi",
                "density_sndsw_scifi",
                "mc_incoming_energy",
                "scifi_qdc_per_hit_direct",
            }
        )
        self.assertTrue(
            set(self.production["classes"]).issubset(self.partitions["particle_group_rules"])
        )
        self.assertEqual(set(self.regions["regions"]), {"inclusive_selected", "scifi35", "signal_region"})
        self.assertEqual(self.evaluation["comparisons"]["regions"], ["scifi35"])
        allowed_functions = {"log10"}
        for feature, config in self.hists["features"].items():
            expression = str(config.get("expression", feature))
            self.assertTrue(extract_identifiers(expression) - allowed_functions <= branches)
        for pair in self.hists["features2d"].values():
            self.assertIn(pair["x_expression"], branches)
            self.assertIn(pair["y_expression"], branches)

    def test_local_outputs_use_work_volume(self):
        output_root = "/afs/cern.ch/work/z/zhibin/snd-ml-outputs/em_shower_neutrino_v1"
        self.assertEqual(self.production["outputs"]["workspace_root"], output_root)
        self.assertEqual(
            self.partitions["output"]["metadata_csv"],
            f"{output_root}/metadata/partitions.csv",
        )
        self.assertEqual(
            self.evaluation["metadata"]["partition_csv"],
            f"{output_root}/metadata/partitions.csv",
        )
        self.assertEqual(self.evaluation["outputs"]["plot_dir"], f"{output_root}/plots")

    def test_energy_histogram_configuration(self):
        self.assertEqual(self.hists["version"], "em_shower_hist_features_energy_v2")
        energy = self.hists["features"]["mc_incoming_energy"]
        log_energy = self.hists["features"]["mc_incoming_log10_energy"]
        self.assertEqual(energy["feature_cut"], "mc_incoming_energy > 0")
        self.assertEqual(log_energy["expression"], "log10(mc_incoming_energy)")
        energy_pairs = {
            name: config
            for name, config in self.hists["features2d"].items()
            if name.startswith("incoming_energy_vs_")
        }
        self.assertEqual(len(energy_pairs), 9)
        self.assertTrue(
            all(config["x_expression"] == "mc_incoming_energy" for config in energy_pairs.values())
        )
        self.assertTrue(
            all("mc_incoming_energy > 0" in config["feature_cut"] for config in energy_pairs.values())
        )

    def test_requested_cluster_plot_axis_ranges(self):
        features = self.hists["features"]
        for orientation in ("vertical", "horizontal"):
            leading_hits = features[f"scifi_{orientation}_cluster1_nhits"]
            self.assertTrue(leading_hits["logy"])
            self.assertEqual(leading_hits["y_max"], 1.0)
            for suffix in (
                "cluster1_qdc_per_hit",
                "cluster1_orientation_qdc_fraction",
                "other_cluster_qdc_fraction",
            ):
                self.assertEqual(features[f"scifi_{orientation}_{suffix}"]["y_max"], 0.3)
        self.assertEqual(features["scifi_other_clusters_qdc_fraction"]["y_max"], 0.3)

    def test_batch_rules_have_explicit_resources(self):
        snakefile = (
            REPO_ROOT / "snakemake/rules/em_shower_neutrino_evaluation.smk"
        ).read_text(encoding="utf-8")
        expected = {
            "em_shower_prepare_evaluation_output_dirs_v1": ("10 * 60", 500, 100),
            "em_shower_combine_partition_metadata_v1": ("30 * 60", 1000, 1000),
            "em_shower_neutrino_plot_v1": ("20 * 60", 2000, 1000),
            "em_shower_neutrino_hist2d_v1": ("45 * 60", 3000, 3000),
            "em_shower_neutrino_plot2d_v1": ("20 * 60", 2000, 1000),
        }
        for rule, (runtime, memory, disk) in expected.items():
            match = re.search(
                rf"^rule {rule}:\n(?P<body>.*?)(?=^rule |^def |\Z)",
                snakefile,
                flags=re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(match, rule)
            body = match.group("body")
            self.assertIn("resources:", body, rule)
            self.assertIn(f"runtime={runtime}", body, rule)
            self.assertIn(f"mem_mb={memory}", body, rule)
            self.assertIn(f"disk_mb={disk}", body, rule)
            self.assertIn("nvidia_gpu=0", body, rule)

    def test_output_directory_preparation_wiring(self):
        snakefile = (
            REPO_ROOT / "snakemake/rules/em_shower_neutrino_evaluation.smk"
        ).read_text(encoding="utf-8")
        preparation_rule = "em_shower_prepare_evaluation_output_dirs_v1"
        self.assertIn(f"localrules: {preparation_rule}", snakefile)
        self.assertIn(
            '"\\n".join(EM_SHOWER_OUTPUT_DIRECTORIES) + "\\n"',
            snakefile,
        )
        for rule in (
            "em_shower_neutrino_hist_v1",
            "em_shower_neutrino_plot_v1",
            "em_shower_neutrino_hist2d_v1",
            "em_shower_neutrino_plot2d_v1",
        ):
            match = re.search(
                rf"^rule {rule}:\n(?P<body>.*?)(?=^rule |^def |\Z)",
                snakefile,
                flags=re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(match, rule)
            self.assertIn(
                "output_dirs=EM_SHOWER_OUTPUT_DIR_MARKER",
                match.group("body"),
                rule,
            )
        expected_count = (
            len(self.hists["features"])
            * len(self.evaluation["comparisons"]["regions"])
            + len(self.hists["features"])
            + len(self.hists["features2d"])
            * len(self.evaluation["comparisons"]["regions"])
            + len(self.hists["features2d"])
            * len(self.evaluation["comparisons"]["regions"])
        )
        self.assertEqual(expected_count, 92)

    def test_metadata_mapping_preserves_absolute_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sources = []
            for subfolder, partition in (("2024_ve", 1), ("2024_vm", 2)):
                source = directory / f"{subfolder}.csv"
                pd.DataFrame(
                    [
                        {
                            "data_type": "MC_neutrino",
                            "subfolder": subfolder,
                            "partition": partition,
                            "output_base_path": f"/old/{subfolder}/{partition}",
                            "feature_path": "feature.root",
                            "hit3d_path": "hit3d.root",
                            "nueAnalysisFilter_path": "cutflow.root",
                        }
                    ]
                ).to_csv(source, index=False)
                sources.append(str(source))
            config = {
                "source_metadata": sources,
                "outputs": {"feature_root": "/new/features"},
                "feature": {"filename_template": "cluster_{subfolder}_{partition}.root"},
            }
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = directory / "mapped.csv"
            result = prepare_metadata(config_path, output)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.loc[0, "source_feature_path"], "/old/2024_ve/1/feature.root")
            self.assertEqual(result.loc[1, "source_hit3d_path"], "/old/2024_vm/2/hit3d.root")
            self.assertEqual(result.loc[0, "output_base_path"], "/new/features/2024_ve/1")
            self.assertEqual(result.loc[1, "feature_path"], "cluster_2024_vm_2.root")

    def test_partition_metadata_combination_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            row = pd.DataFrame([{"partition_id": "one", "n_events": 3}])
            first = directory / "first.csv"
            second = directory / "second.csv"
            row.to_csv(first, index=False)
            row.iloc[:0].to_csv(second, index=False)
            result = combine([str(first), str(second)], str(directory / "combined.csv"))
            self.assertEqual(result["partition_id"].tolist(), ["one"])
            row.to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate partition IDs"):
                combine([str(first), str(second)], str(directory / "duplicate.csv"))


if __name__ == "__main__":
    unittest.main()
