#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_unified_testbeam_histogram_bundles import (  # noqa: E402
    energy_summary,
    faceted_1d,
    faceted_2d,
)
from unified_testbeam_histogram_bundles import (  # noqa: E402
    config_hash,
    load_bundle_config,
    required_branches,
)


class UnifiedTestbeamHistogramBundleTests(unittest.TestCase):
    def setUp(self):
        self.hist_config = (
            REPO_ROOT / "snakemake/metadata/configs/unified_testbeam_histograms_v4.yaml"
        )
        self.plot_config = (
            REPO_ROOT / "snakemake/metadata/configs/unified_testbeam_plots_v4.yaml"
        )

    def test_configuration_is_split_and_complete(self):
        histograms, base = load_bundle_config(self.hist_config)
        plots = yaml.safe_load(self.plot_config.read_text(encoding="utf-8"))
        self.assertEqual(len(histograms["features"]), 30)
        self.assertEqual(len(histograms["features2d"]), 7)
        self.assertEqual(set(plots["cohorts"]), {
            "data_combined_particle_energy", "mc_combined_particle_energy",
        })
        self.assertFalse(plots["write_individual_csv"])
        self.assertEqual(len(config_hash(histograms, base)), 64)
        branches = required_branches(histograms, base["groups"]["mc_pion_scan_100"])
        self.assertIn("count_scifi", branches)
        self.assertIn("scifi_vertical_cluster1_qdc_std", branches)

    def test_workflow_has_bundle_boundary_and_no_csv_outputs(self):
        workflow = (
            REPO_ROOT / "snakemake/unified_testbeam_distributions_v4.smk"
        ).read_text(encoding="utf-8")
        for name in (
            "unified_testbeam_histogram_bundle_v4",
            "unified_testbeam_validate_histogram_bundle_v4",
            "unified_testbeam_histogram_bundles_v4",
            "unified_testbeam_faceted_1d_plot_v4",
            "unified_testbeam_energy_summary_plot_v4",
            "unified_testbeam_faceted_2d_plot_v4",
            "unified_testbeam_all_plots_v4",
        ):
            self.assertIn(f"rule {name}:", workflow)
        self.assertNotIn("summary=", workflow)
        self.assertNotIn(".csv\"\n    params:", workflow)
        self.assertIn("plot_manifest.json", workflow)

    @staticmethod
    def _write_bundle(path: Path, group: str, particle: str, energy: int) -> None:
        import ROOT

        source = ROOT.TFile(str(path), "RECREATE")
        source.mkdir("features")
        source.cd("features")
        one = ROOT.TH1D("count_scifi", ";Selected SciFi hits", 10, 0, 100)
        for value in range(10, 90, 5):
            one.Fill(value + (3 if particle.startswith("e") else 0))
        one.Write()
        source.mkdir("features2d")
        source.cd("features2d")
        two = ROOT.TH2D("total_qdc_vs_hits", ";Selected SciFi hits;QDC", 10, 0, 100, 10, 0, 200)
        for value in range(10, 90, 5):
            two.Fill(value, value * 2)
        two.Write()
        source.mkdir("metadata")
        source.cd("metadata")
        metadata = {
            "group": group,
            "particle_charge": particle,
            "beam_energy_gev": energy,
            "base_selection": "count_scifi > 35",
        }
        ROOT.TObjString(json.dumps(metadata)).Write("bundle_json")
        source.Close()

    def test_pdf_renderers_read_only_bundles(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            groups = ["pion_100", "electron_100"]
            bundle_paths = []
            for group, particle in zip(groups, ("pi-", "e-")):
                path = directory / f"{group}.root"
                self._write_bundle(path, group, particle, 100)
                bundle_paths.append(f"{group}={path}")
            config = {
                "version": "test_v4",
                "normalization": "unit_area",
                "cohorts": {"test": {"groups": groups, "facet_columns": 1}},
                "styles": {
                    "pion": {"color": 4, "marker": 20},
                    "electron": {"color": 2, "marker": 24},
                    "th2_log_z": True,
                },
                "display_x_ranges": {},
            }
            config_path = directory / "plots.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            outputs = (
                (faceted_1d, "count_scifi", "one.pdf"),
                (energy_summary, "count_scifi", "summary.pdf"),
                (faceted_2d, "total_qdc_vs_hits", "two.pdf"),
            )
            for function, feature, name in outputs:
                output = directory / name
                function(argparse.Namespace(
                    config=str(config_path), cohort="test", feature=feature,
                    bundle=bundle_paths, output=str(output),
                ))
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)
            self.assertFalse(list(directory.glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
