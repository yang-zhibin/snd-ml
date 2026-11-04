#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_feature_distributions import (  # noqa: E402
    _numeric_exposure,
    combine_selection,
    histogram_model,
    load_config,
    load_validated_rows,
    normalization_scale,
)


class UnifiedFeatureDistributionTests(unittest.TestCase):
    def setUp(self):
        self.config_path = (
            REPO_ROOT
            / "snakemake/metadata/configs/unified_shower_distributions_v1.yaml"
        )
        self.config = load_config(self.config_path)

    def test_frozen_features_and_stage1_groups(self):
        self.assertEqual(
            set(self.config["features"]),
            {
                "density_sndsw_scifi",
                "scifi_cluster_qdc_per_hit",
                "scifi_qdc_per_hit_direct",
            },
        )
        stage = self.config["stages"]["stage1"]
        self.assertEqual(
            stage["unit_area_groups"],
            ["CC_nue", "CC_numu", "NC_nue", "NC_numu", "MC_muon_up"],
        )
        self.assertNotIn("MC_muon_up", stage["common_exposure_groups"])
        self.assertEqual(self.config["groups"]["CC_nue"]["source_subfolders"], ["2024_ve"])
        self.assertEqual(self.config["groups"]["CC_numu"]["source_subfolders"], ["2024_vm"])

    def test_histogram_binning_is_integral(self):
        for feature, settings in self.config["features"].items():
            bins, low, high = histogram_model(feature, settings)
            self.assertGreater(bins, 0)
            self.assertGreater(high, low)
            self.assertTrue(
                math.isclose(
                    bins * float(settings["bin_width"]),
                    high - low,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )

    def test_unit_area_is_probability_density(self):
        scale = normalization_scale(
            "unit_area", 20.0, 0.5, 0.0, "event_count_only", self.config["normalizations"]
        )
        self.assertEqual(scale, 0.1)
        self.assertEqual(
            normalization_scale(
                "unit_area", 0.0, 0.5, 0.0, "event_count_only", self.config["normalizations"]
            ),
            0.0,
        )

    def test_common_exposure_accepts_only_validated_luminosity(self):
        scale = normalization_scale(
            "common_exposure_19fb",
            100.0,
            1.0,
            9.5,
            "simulated_luminosity_fb",
            self.config["normalizations"],
        )
        self.assertEqual(scale, 2.0)
        for kind in ("unvalidated_equivalent_luminosity", "event_count_only"):
            with self.assertRaisesRegex(ValueError, "not allowed"):
                normalization_scale(
                    "common_exposure_19fb",
                    100.0,
                    1.0,
                    9.5,
                    kind,
                    self.config["normalizations"],
                )

    def test_selection_combines_scifi_and_particle_cuts(self):
        self.assertEqual(
            combine_selection("count_scifi > 35", "pdgCode == 12 || pdgCode == -12"),
            "(count_scifi > 35) && (pdgCode == 12 || pdgCode == -12)",
        )
        self.assertEqual(combine_selection("count_scifi > 35", ""), "(count_scifi > 35)")
        self.assertEqual(
            combine_selection(
                "count_scifi > 35",
                "abs(source_pdg_code) == 11",
                "scifi_vertical_cluster1_exists == 1",
            ),
            "(count_scifi > 35) && (abs(source_pdg_code) == 11) && "
            "(scifi_vertical_cluster1_exists == 1)",
        )

    def test_validation_records_are_mandatory_and_bound_to_manifest(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            feature = directory / "feature.root"
            feature.touch()
            marker = directory / "feature.root.validated.json"
            manifest = directory / "manifest.csv"
            row = {
                "family": "mc_neutrino",
                "source_uid": "unified_src_0123456789abcdef0123",
                "feature_scifi_cluster_v1_path": str(feature),
                "validation_record": str(marker),
                "lumi_per_file": 1.0,
                "exposure_kind": "simulated_luminosity_fb",
            }
            pd.DataFrame([row]).to_csv(manifest, index=False)
            marker.write_text(
                json.dumps(
                    {
                        "validation_passed": True,
                        "source_uid": row["source_uid"],
                        "feature_path": str(feature),
                            "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                            "feature_size_bytes": feature.stat().st_size,
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_validated_rows(manifest, "mc_neutrino")
            self.assertEqual(len(loaded), 1)
            marker.write_text(
                json.dumps(
                    {
                        "validation_passed": True,
                        "source_uid": "wrong",
                        "feature_path": str(feature),
                            "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                            "feature_size_bytes": feature.stat().st_size,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "source UID mismatch"):
                load_validated_rows(manifest, "mc_neutrino")

    def test_shard_rows_require_unique_units_and_deduplicate_exposure(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            uid = "unified_src_0123456789abcdef0123"
            rows = []
            for index in range(2):
                feature = directory / f"part_{index}.root"
                marker = directory / f"part_{index}.validated.json"
                feature.touch()
                marker.write_text(
                    json.dumps(
                        {
                            "validation_passed": True,
                            "source_uid": uid,
                            "feature_path": str(feature),
                            "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                            "feature_size_bytes": feature.stat().st_size,
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "family": "mc_neutrino",
                        "source_uid": uid,
                        "feature_unit_uid": f"{uid}:part_{index:04d}",
                        "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                        "feature_size_bytes": feature.stat().st_size,
                        "validation_record_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
                        "selected_entry_start": index,
                        "feature_scifi_cluster_v1_path": str(feature),
                        "validation_record": str(marker),
                        "lumi_per_file": 4.5,
                        "exposure_kind": "simulated_luminosity_fb",
                    }
                )
            manifest = directory / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            loaded = load_validated_rows(manifest, "mc_neutrino")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded["source_uid"].nunique(), 1)
            self.assertEqual(_numeric_exposure(loaded), 4.5)

            rows[1]["feature_unit_uid"] = rows[0]["feature_unit_uid"]
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate feature unit UIDs"):
                load_validated_rows(manifest, "mc_neutrino")

    def test_snakemake_targets_and_runtime_guards_are_present(self):
        rules = (
            REPO_ROOT / "snakemake/rules/unified_shower_distributions.smk"
        ).read_text(encoding="utf-8")
        for target in (
            "unified_stage1_feature_audits_v1",
            "unified_stage1_raw_hists_v1",
            "unified_stage1_plots_unit_area_v1",
            "unified_stage1_plots_exposure_v1",
            "unified_stage1_distributions_v1",
        ):
            self.assertIn(f"rule {target}:", rules)
        self.assertIn("ready=lambda wc: _unified_distribution_audit_path", rules)
        self.assertNotIn("validated=_unified_distribution_validation_records", rules)
        self.assertIn("common_exposure_19fb", rules)


if __name__ == "__main__":
    unittest.main()
