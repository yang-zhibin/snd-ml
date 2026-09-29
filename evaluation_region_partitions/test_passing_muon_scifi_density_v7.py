#!/usr/bin/env python3
"""Synthetic end-to-end test for V7 SciFi-density category distributions."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

import test_passing_muon_binary_region_study_v5 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
DATA_DISPLAY_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_v7.yaml"
)
MC_DISPLAY_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_mc_v7.yaml"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/"
    "make_passing_muon_scifi_density_hist_v7.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/"
    "plot_passing_muon_scifi_density_study_v7.py"
)

sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))
from make_passing_muon_binary_region_hist_v5 import (  # noqa: E402
    configured_v5_features,
)
from passing_muon_v7_scifi_density import (  # noqa: E402
    category_expression,
    configured_density_categories,
    configured_density_selection,
)


def load(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def run(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


class PassingMuonSciFiDensityV7Test(unittest.TestCase):
    def test_configuration_uses_only_density_categories(self):
        study = load(REGION_CONFIG)
        data = load(DATA_DISPLAY_CONFIG)
        mc = load(MC_DISPLAY_CONFIG)
        categories = configured_density_categories(data, mc)
        self.assertEqual(list(categories), [
            "scifi_density_gt100",
            "scifi_density_gt500",
            "scifi_density_gt1000",
            "scifi_density_gt5000",
        ])
        self.assertNotIn("scifi_hit_count_gt100", categories)
        selection = configured_density_selection(study, data, mc)
        for category in categories.values():
            expression = category_expression(selection, category)
            self.assertIn("!", expression)
            self.assertIn("density_sndsw_scifi", expression)
            self.assertIn("count_veto >= 0", expression)
            self.assertIn("count_veto == 3", expression)
            self.assertIn("count_veto > 3", expression)

    def test_histogram_and_pdf_outputs(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(prefix="scifi_density_v7_") as directory:
            directory = Path(directory)
            real_root = directory / "real.root"
            mc_normal_root = directory / "mc_normal.root"
            mc_empty_root = directory / "mc_empty.root"
            data_metadata = directory / "data.csv"
            mc_metadata = directory / "mc.csv"
            study_path = directory / "study.yaml"
            output = directory / "density.root"

            base.write_tree(ROOT, real_root)
            base.write_tree(ROOT, mc_normal_root)
            base.write_tree(ROOT, mc_empty_root, zero_scifi=True)
            base.write_metadata(data_metadata, [base.standardized_row(
                real_root,
                "real_data_dq_run8329",
                "all_events__real_data_dq_run8329__part001",
            )])

            study = load(REGION_CONFIG)
            study["expected"] = {"real_data_dq": len(base.EVENTS)}
            study["mc_muon_up"]["expected_source_files"] = 2
            study["mc_muon_up"]["productions"] = {
                "scoring_1p8": study["mc_muon_up"]["productions"]["scoring_1p8"],
                "scoring_at449m": (
                    study["mc_muon_up"]["productions"]["scoring_at449m"]
                ),
            }
            study_path.write_text(
                yaml.safe_dump(study, sort_keys=False), encoding="utf-8"
            )
            base.write_metadata(mc_metadata, [
                base.standardized_row(
                    mc_normal_root,
                    study["mc_muon_up"]["productions"]["scoring_1p8"]["group"],
                    "all_events__mc_normal__part001",
                    "scoring_1p8",
                ),
                base.standardized_row(
                    mc_empty_root,
                    study["mc_muon_up"]["productions"]["scoring_at449m"]["group"],
                    "all_events__mc_empty__part001",
                    "scoring_at449m",
                ),
            ])

            run([
                sys.executable,
                str(HISTOGRAMMER),
                "--partition-metadata", str(data_metadata),
                "--mc-muon-up-metadata", str(mc_metadata),
                "--hist-config", str(base.HIST_CONFIG),
                "--region-config", str(study_path),
                "--event-display-config", str(DATA_DISPLAY_CONFIG),
                "--mc-event-display-config", str(MC_DISPLAY_CONFIG),
                "--output", str(output),
                "--threads", "1",
            ])

            source = ROOT.TFile.Open(str(output))
            self.assertTrue(source and not source.IsZombie())
            try:
                categories = configured_density_categories(
                    load(DATA_DISPLAY_CONFIG), load(MC_DISPLAY_CONFIG)
                )
                features = configured_v5_features(load(base.HIST_CONFIG), study)
                self.assertEqual(int(base.parameter(source, "feature_count")), 46)
                self.assertEqual(
                    str(base.parameter(source, "category_order")),
                    ",".join(categories),
                )
                counts = []
                for category in categories:
                    prefix = f"category__{category}"
                    data_count = int(base.parameter(source, f"{prefix}__data_count"))
                    counts.append(data_count)
                    veto_sum = sum(
                        int(base.parameter(
                            source, f"{prefix}__data__{veto}__count"
                        ))
                        for veto in ("veto_lt3", "veto_eq3", "veto_gt3")
                    )
                    self.assertEqual(veto_sum, data_count)
                    mc_count = int(base.parameter(source, f"{prefix}__mc_count"))
                    component_sum = sum(
                        int(base.parameter(
                            source, f"{prefix}__mc__{production}__count"
                        ))
                        for production in ("scoring_1p8", "scoring_at449m")
                    )
                    self.assertEqual(component_sum, mc_count)
                    for feature in features:
                        path = f"categories/{category}/features/{feature}"
                        self.assertTrue(source.Get(f"{path}/data"), path)
                        self.assertTrue(source.Get(f"{path}/mc"), path)
                self.assertEqual(counts, sorted(counts, reverse=True))
            finally:
                source.Close()

            pdf = directory / "scifi_density_gt100.pdf"
            run([
                sys.executable,
                str(PLOTTER),
                "--input", str(output),
                "--hist-config", str(base.HIST_CONFIG),
                "--region-config", str(study_path),
                "--event-display-config", str(DATA_DISPLAY_CONFIG),
                "--mc-event-display-config", str(MC_DISPLAY_CONFIG),
                "--category", "scifi_density_gt100",
                "--output-pdf", str(pdf),
            ])
            self.assertTrue(pdf.is_file() and pdf.stat().st_size > 0)
            if shutil.which("pdfinfo"):
                self.assertEqual(base.pdf_pages(pdf), 47)


if __name__ == "__main__":
    unittest.main()
