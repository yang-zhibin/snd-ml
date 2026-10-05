#!/usr/bin/env python3

from __future__ import annotations

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
from scifi_plane_clustering import scalar_feature_dtypes  # noqa: E402


class EmShowerNeutrinoWorkflowTest(unittest.TestCase):
    def setUp(self):
        config_dir = REPO_ROOT / "snakemake/metadata/configs/em_shower"
        self.production = yaml.safe_load((config_dir / "production_v1.yaml").read_text())
        self.regions = yaml.safe_load((config_dir / "regions_v1.yaml").read_text())
        self.partitions = yaml.safe_load((config_dir / "partitions_v1.yaml").read_text())
        self.hists = yaml.safe_load((config_dir / "hist_features_v1.yaml").read_text())

    def test_configs_reference_emitted_scalar_branches(self):
        branches = {name for name, _ in scalar_feature_dtypes()}
        branches.update({"count_scifi", "qdc_scifi"})
        self.assertEqual(
            set(self.production["classes"]),
            set(self.partitions["particle_group_rules"]),
        )
        self.assertEqual(set(self.regions["regions"]), {"inclusive_selected", "scifi35", "signal_region"})
        self.assertTrue(set(self.hists["features"]).issubset(branches))
        for pair in self.hists["features2d"].values():
            self.assertIn(pair["x_expression"], branches)
            self.assertIn(pair["y_expression"], branches)

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
