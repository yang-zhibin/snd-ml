#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_unified_testbeam_histogram_bundles import (  # noqa: E402
    faceted_1d,
    faceted_2d,
    manifest as write_plot_manifest,
)
from unified_testbeam_histogram_bundles import load_bundle_config  # noqa: E402
from validate_unified_feature_snapshot_coverage import validate  # noqa: E402


class UnifiedRealDataHistogramBundleTests(unittest.TestCase):
    def setUp(self):
        self.hist_config = REPO_ROOT / "snakemake/metadata/configs/unified_real_data_histograms_v2.yaml"
        self.plot_config = REPO_ROOT / "snakemake/metadata/configs/unified_real_data_plots_v2.yaml"

    def test_configuration_reuses_complete_v4_histogram_definitions(self):
        histograms, base = load_bundle_config(self.hist_config)
        plots = yaml.safe_load(self.plot_config.read_text(encoding="utf-8"))
        self.assertEqual(len(histograms["features"]), 30)
        self.assertEqual(len(histograms["features2d"]), 7)
        self.assertEqual(set(base["groups"]), {
            "real_data_2024_no_veto", "real_data_2024_veto_10pct",
        })
        cohort = plots["cohorts"]["no_veto_vs_veto_10pct"]
        self.assertEqual(cohort["two_dimensional_columns"], 2)
        self.assertTrue(cohort["show_source_count"])
        self.assertEqual(plots["coverage"]["policy"], "partial")
        self.assertFalse(plots["write_individual_csv"])

    def test_workflow_is_downstream_only_and_has_expected_targets(self):
        workflow = (REPO_ROOT / "snakemake/unified_real_data_distributions_v2.smk").read_text(encoding="utf-8")
        for name in (
            "unified_real_data_feature_snapshot_coverage_v2",
            "unified_real_data_histogram_bundle_v2",
            "unified_real_data_validate_histogram_bundle_v2",
            "unified_real_data_histogram_bundles_v2",
            "unified_real_data_faceted_1d_plot_v2",
            "unified_real_data_faceted_2d_plot_v2",
            "unified_real_data_all_plots_v2",
        ):
            self.assertIn(f"rule {name}:", workflow)
        self.assertNotIn("include:", workflow)
        self.assertNotIn("--allow-missing-validated-sources", workflow)
        self.assertNotIn("energy_summary", workflow)
        self.assertNotIn(".csv\"\n    params:", workflow)
        localrules = workflow.split("rule unified_real_data_feature_snapshot_coverage_v2:", 1)[0]
        self.assertIn("localrules:", localrules)
        self.assertIn("unified_real_data_feature_snapshot_coverage_v2", localrules)
        self.assertIn("unified_real_data_plot_manifest_v2", localrules)
        self.assertIn("--policy {params.policy}", workflow)
        self.assertIn("--coverage-receipt {input.coverage}", workflow)
        producer = (REPO_ROOT / "snakemake/unified_real_data_distributions.smk").read_text(encoding="utf-8")
        self.assertIn("@lru_cache(maxsize=None)\ndef _manifest_frame", producer)
        self.assertIn("def _manifest_records", producer)

    def test_coverage_gate_rejects_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            snapshot = directory / "family.csv"
            snapshot.write_text("source_uid,family,feature_unit_uid,feature_layout,feature_scifi_cluster_v1_path,validation_record,shard_index,selected_entry_start,selected_entry_stop\nsource_a,family,source_a,canonical,a.root,a.json,-1,0,0\n", encoding="utf-8")
            source_manifest = directory / "source_family.csv"
            source_manifest.write_text("source_uid,family,feature_scifi_cluster_v1_path,validation_record\nsource_a,family,a.root,a.json\n", encoding="utf-8")
            coverage = directory / "coverage.json"
            record = {
                "coverage_schema_version": 2,
                "available_feature_file_count": 1,
                "source_statuses": {"source_a": "available"},
                "snapshot_manifest_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "family": "family", "available_source_count": 1,
                "total_source_count": 1, "unavailable_source_count": 0,
                "available_fraction": 1.0,
                "status_counts": {"available": 1},
                "snapshot_manifest": str(snapshot),
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
            }
            coverage.write_text(json.dumps({"families": [record]}), encoding="utf-8")
            self.assertTrue(validate(str(coverage), ["family"])["validation_passed"])
            with source_manifest.open("a") as stream:
                stream.write("source_b,family,b.root,b.json\n")
            record["source_manifest_sha256"] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
            record["source_statuses"]["source_b"] = "missing_feature"
            record["total_source_count"] = 2
            record["unavailable_source_count"] = 1
            record["available_fraction"] = 0.5
            record["status_counts"]["missing_feature"] = 1
            coverage.write_text(json.dumps({"families": [record]}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid feature snapshot coverage"):
                validate(str(coverage), ["family"])
            partial = validate(
                str(coverage), ["family"], policy="partial",
                allowed_unavailable_statuses=["missing_feature"],
            )
            self.assertTrue(partial["validation_passed"])
            self.assertEqual(partial["coverage_policy"], "partial")
            self.assertEqual(partial["families"][0]["available_fraction"], 0.5)

    @staticmethod
    def _write_bundle(path: Path, group: str, offset: int) -> None:
        import ROOT

        source = ROOT.TFile(str(path), "RECREATE")
        source.mkdir("features")
        source.cd("features")
        one = ROOT.TH1D("count_scifi", ";Selected SciFi hits", 10, 0, 100)
        for value in range(10, 90, 5):
            one.Fill(value + offset)
        one.Write()
        source.mkdir("features2d")
        source.cd("features2d")
        two = ROOT.TH2D("total_qdc_vs_hits", ";Selected SciFi hits;QDC", 10, 0, 100, 10, 0, 200)
        for value in range(10, 90, 5):
            two.Fill(value + offset, value * 2)
        two.Write()
        source.mkdir("metadata")
        source.cd("metadata")
        ROOT.TObjString(json.dumps({
            "group": group,
            "group_label": group,
            "particle_charge": "",
            "beam_energy_gev": "",
            "base_selection": "count_scifi > 35",
        })).Write("bundle_json")
        source.Close()

    def test_real_data_renderers_use_configured_labels_and_write_only_pdf(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            groups = ["no_veto", "veto_10pct"]
            bundles = []
            for group, offset in zip(groups, (0, 3)):
                path = directory / f"{group}.root"
                self._write_bundle(path, group, offset)
                bundles.append(f"{group}={path}")
            config = {
                "version": "real_test_v2", "normalization": "unit_area",
                "cohorts": {"test": {
                    "groups": groups, "facet_columns": 1, "facet_group_size": 2,
                    "two_dimensional_columns": 2, "sort_2d_by_particle": False,
                    "annotation": "test annotation",
                    "group_labels": {"no_veto": "No Veto hit", "veto_10pct": "Veto hit (10% sample)"},
                    "group_styles": {"no_veto": "no_veto", "veto_10pct": "veto_10pct"},
                }},
                "styles": {
                    "no_veto": {"color": 4, "marker": 20},
                    "veto_10pct": {"color": 2, "marker": 24},
                    "th2_log_z": True, "th2_z_floor_fraction": 1e-5,
                },
                "display_x_ranges": {},
            }
            config_path = directory / "plots.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            for function, feature, name in (
                (faceted_1d, "count_scifi", "one.pdf"),
                (faceted_2d, "total_qdc_vs_hits", "two.pdf"),
            ):
                output = directory / name
                function(argparse.Namespace(
                    config=str(config_path), cohort="test", feature=feature,
                    bundle=bundles, output=str(output),
                ))
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)
            self.assertFalse(list(directory.glob("*.csv")))

            coverage_path = directory / "coverage.validated.json"
            coverage_path.write_text(json.dumps({
                "validation_passed": True,
                "coverage_policy": "partial",
                "families": [{"family": "no_veto", "available_fraction": 0.1}],
            }), encoding="utf-8")
            manifest_path = directory / "plot_manifest.json"
            write_plot_manifest(argparse.Namespace(
                config=str(config_path), bundle=bundles,
                plot=[str(directory / "one.pdf"), str(directory / "two.pdf")],
                coverage_receipt=str(coverage_path), output=str(manifest_path),
            ))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["feature_snapshot_coverage"]["coverage_policy"], "partial"
            )


if __name__ == "__main__":
    unittest.main()
