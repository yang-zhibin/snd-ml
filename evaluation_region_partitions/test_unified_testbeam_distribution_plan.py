#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINED_SNAKEFILE = (
    REPO_ROOT / "snakemake/unified_testbeam_combined_plots.smk"
)
sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))
sys.path.insert(0, str(REPO_ROOT / "snakemake/metadata"))

from prepare_unified_testbeam_2023_manifest import (  # noqa: E402
    GEOMETRY_CHECKSUM,
    GEOMETRY_ID,
    balanced_allocations,
    build_manifest,
    classify_legacy_geometry,
    deterministic_entry_indices,
)
from unified_feature_distributions import (  # noqa: E402
    filter_source_rows,
    load_config,
)


class TestbeamDistributionPlanTests(unittest.TestCase):
    def setUp(self):
        self.config_path = (
            REPO_ROOT
            / "snakemake/metadata/configs/unified_testbeam_distributions_v3.yaml"
        )

    def test_balanced_allocation_and_sparse_indices_are_deterministic(self):
        self.assertEqual(balanced_allocations([10, 10, 10], 8), [3, 3, 2])
        self.assertEqual(deterministic_entry_indices(10, 4), [1, 3, 6, 8])
        self.assertEqual(
            deterministic_entry_indices(1_000_000, 3),
            deterministic_entry_indices(1_000_000, 3),
        )

    def test_legacy_geometry_classifier_is_fail_closed(self):
        classified = classify_legacy_geometry(
            "/geometry/geofile_sndlhc_H8_2023_3walls.root"
        )
        self.assertEqual(classified["geometry_setup_id"], GEOMETRY_ID)
        self.assertEqual(classified["geometry_checksum"], GEOMETRY_CHECKSUM)
        self.assertEqual(classified["wall_count"], 3)
        with self.assertRaisesRegex(ValueError, "unknown unaudited"):
            classify_legacy_geometry("/geometry/unseen.root")

    def test_real_2023_manifest_has_configured_three_wall_pion_cells(self):
        metadata = Path(
            "/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/"
            "real_data_testbeam_June2023_H8_metadata.csv"
        )
        with self.config_path.open("r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)["testbeam_2023"]
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            manifest, report = build_manifest(
                metadata,
                Path(directory) / "indices",
                Path(directory) / "features",
                selected_cells=settings["selected_cells"],
                target_events_per_cell=settings["target_events_per_cell"],
                manifest_version=settings["manifest_version"],
            )
        self.assertEqual(len(manifest), 385)
        self.assertEqual(set(manifest["source_energy_gev"]), {100, 140, 180, 300})
        self.assertEqual(set(manifest["source_pdg_code"]), {-211, 211})
        self.assertEqual(set(manifest["source_geometry_id"]), {GEOMETRY_ID})
        self.assertEqual(
            set(manifest["source_geometry_checksum"]), {GEOMETRY_CHECKSUM}
        )
        counts = manifest.groupby(
            ["source_pdg_code", "source_energy_gev"]
        )["source_uid"].count().to_dict()
        self.assertEqual(
            counts, {(-211, 300): 186, (211, 100): 56, (211, 140): 54, (211, 180): 89}
        )
        selected = (
            manifest.groupby(["source_pdg_code", "source_energy_gev"])[
                "selected_event_count"
            ].sum().to_dict()
        )
        self.assertEqual(
            selected,
            {(-211, 300): 50_000, (211, 100): 50_000, (211, 140): 50_000, (211, 180): 50_000},
        )
        self.assertEqual(report["manifest_version"], "unified_testbeam_2023_h8_v2")
        self.assertEqual(report["wall_count"], 3)
        self.assertEqual(report["absorber_material"], "iron")

    def test_plot_geometry_policies_and_comparison_balancing(self):
        config = load_config(self.config_path)
        self.assertEqual(len(config["groups"]), 30)
        cross_names = {
            "data_cross_geometry_100",
            "data_cross_geometry_140pi_150e",
            "data_cross_geometry_180pi_200e",
            "data_cross_geometry_300",
            "data_combined_particle_energy",
        }
        self.assertEqual(
            {
                name
                for name, plot in config["plot_sets"].items()
                if plot["geometry_policy"] == "cross_geometry_diagnostic"
            },
            cross_names,
        )
        for name, plot in config["plot_sets"].items():
            geometries = {
                config["groups"][name]["geometry_setup_id"]
                for name in plot["groups"]
            }
            self.assertEqual(len(geometries), 2 if name in cross_names else 1)
        for energy in (50, 100, 150, 200, 250, 300):
            pion = config["groups"][f"mc_pion_scan_{energy}"]
            electron = config["groups"][f"mc_electron_compare_{energy}"]
            self.assertEqual(pion["max_events"], electron["max_events"])
            self.assertEqual(pion["max_events"], 2520)
        data_combined = config["plot_sets"]["data_combined_particle_energy"]
        self.assertEqual(
            data_combined["groups"],
            [
                "data_pion_scan_100", "data_electron_scan_100",
                "data_pion_scan_140", "data_electron_scan_150",
                "data_pion_scan_180", "data_electron_scan_200",
                "data_pion_fe3_300_negative", "data_electron_scan_300",
            ],
        )
        mc_combined = config["plot_sets"]["mc_combined_particle_energy"]
        self.assertEqual(
            mc_combined["groups"],
            [
                "mc_pion_scan_50", "mc_electron_compare_50",
                "mc_pion_scan_100", "mc_electron_compare_100",
                "mc_pion_scan_150", "mc_electron_compare_150",
                "mc_pion_scan_200", "mc_electron_compare_200",
                "mc_pion_scan_250", "mc_electron_compare_250",
                "mc_pion_scan_300", "mc_electron_compare_300",
            ],
        )
        for plot in (data_combined, mc_combined):
            self.assertEqual(plot["style_policy"], "paired_particle_energy")
            self.assertEqual(plot["legend_columns"], 2)
            self.assertEqual(plot["draw_style"], "points_errors")
            self.assertEqual(plot["unit_area_mode"], "bin_fraction")
            self.assertTrue(plot["annotation_lines"])
            self.assertIn("bin fractions sum to 1", plot["annotation_lines"][-1])
        self.assertEqual(
            data_combined["legend_label_mode"],
            "configured",
        )
        self.assertEqual(
            mc_combined["legend_label_mode"], "compact_particle_energy"
        )
        self.assertIn("cross-geometry", data_combined["annotation_lines"][0])
        self.assertIn("W target", mc_combined["annotation_lines"][0])
        self.assertEqual(config["features"]["density_sndsw_scifi"]["bin_width"], 250)
        cluster_qdc = config["features"]["scifi_cluster_qdc_per_hit"]
        self.assertEqual(cluster_qdc["x_min"], 0)
        self.assertEqual(cluster_qdc["x_max"], 7)
        self.assertEqual(cluster_qdc["bin_width"], 0.1)
        self.assertFalse(cluster_qdc["fold_overflow_for_plot"])
        for feature in tuple(config["features"])[2:]:
            self.assertEqual(config["features"][feature]["bin_width"], 0.2)

    def test_source_filters_are_applied_before_validation(self):
        frame = pd.DataFrame([
            {"source_energy_gev": "100.0", "source_pdg_code": "211"},
            {"source_energy_gev": "180.0", "source_pdg_code": "211"},
            {"source_energy_gev": "180.0", "source_pdg_code": "11"},
        ])
        filtered = filter_source_rows(
            frame, {"source_energy_gev": 180, "source_pdg_code": 211}
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["source_pdg_code"], "211")

    def test_standalone_workflow_exposes_preparation_and_final_targets(self):
        workflow = (REPO_ROOT / "snakemake/unified_testbeam_distributions.smk").read_text()
        self.assertIn(
            "checkpoint unified_prepare_testbeam_2023_manifest_v1:", workflow
        )
        self.assertIn(
            "checkpoint unified_prepare_testbeam_feature_snapshots_v3:", workflow
        )
        for rule in (
            "unified_testbeam_2023_selection_indexes_v1",
            "unified_testbeam_2023_features_v1",
            "unified_testbeam_raw_hists_v3",
            "unified_testbeam_distributions_v3",
            "unified_testbeam_cross_geometry_distributions_v3",
            "unified_testbeam_combined_particle_energy_plots_v3",
        ):
            self.assertIn(f"rule {rule}:", workflow)
        self.assertIn("--source-pdg-code {params.pdg_code}", workflow)
        self.assertNotIn("--source-pdg-code 211", workflow)
        self.assertIn("--config {input.config}", workflow)
        self.assertIn("--geometry-policy {params.geometry_policy}", workflow)

    def test_combined_plot_workflow_cannot_schedule_feature_production(self):
        workflow = COMBINED_SNAKEFILE.read_text(encoding="utf-8")
        self.assertIn("rule unified_testbeam_combined_raw_hist_v3:", workflow)
        self.assertIn("rule unified_testbeam_combined_plot_v3:", workflow)
        self.assertIn(
            "rule unified_testbeam_combined_particle_energy_plots_v3:", workflow
        )
        self.assertIn(
            "rule unified_testbeam_combined_scifi_cluster_qdc_per_hit_plot_v3:", workflow
        )
        self.assertIn("rule unified_testbeam_combined_faceted_plot_v3:", workflow)
        self.assertIn(
            "rule unified_testbeam_combined_energy_summary_plot_v3:", workflow
        )
        self.assertIn("rule unified_testbeam_combined_readable_plots_v3:", workflow)
        self.assertNotIn("unified_testbeam_2023_feature_v1", workflow)
        self.assertNotIn("unified_testbeam_2023_selection_index_v1", workflow)
        self.assertNotIn("unified_prepare_testbeam_feature_snapshots_v3", workflow)


if __name__ == "__main__":
    unittest.main()
