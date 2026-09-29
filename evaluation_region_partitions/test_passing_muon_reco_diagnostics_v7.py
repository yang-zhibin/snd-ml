#!/usr/bin/env python3
"""Synthetic end-to-end test for the V7 flat-v2 track diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_reco_diagnostics_v7.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_reco_diagnostics_v7.py"
)


def parameter(source, name):
    value = source.Get(name)
    assert value, name
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def write_config(path):
    config = yaml.safe_load(REGION_CONFIG.read_text(encoding="utf-8"))
    diagnostics = config["reco_track_diagnostics"]
    diagnostics["track_types"] = {
        "scifi_hough": diagnostics["track_types"]["scifi_hough"]
    }
    hough_group = diagnostics["plot_groups"]["hough"]
    hough_group["sections"] = {
        "scifi": {
            "label": "SciFi tracks",
            "track_types": ["scifi_hough"],
        }
    }
    hough_group["event_features"] = {
        "n_tracks": {
            "dimension": 1,
            "expression": "reco_scifi_hough_n_all",
            "x_min": -0.5,
            "x_max": 5.5,
            "n_bins": 6,
            "axis_title": "Hough reconstructed tracks per event",
            "logy": True,
        },
        "n_converged_tracks": {
            "dimension": 1,
            "expression": "reco_scifi_hough_n_converged",
            "x_min": -0.5,
            "x_max": 5.5,
            "n_bins": 6,
            "axis_title": "Converged Hough tracks per event",
            "logy": True,
        },
        "n_track_types": {
            "dimension": 1,
            "expression": "int(reco_scifi_hough_n_all > 0)",
            "x_min": -0.5,
            "x_max": 1.5,
            "n_bins": 2,
            "axis_title": "Hough track types present per event",
            "logy": True,
        },
    }
    diagnostics["plot_groups"] = {"hough": hough_group}
    diagnostics["event_features"] = {
        "n_tracks": diagnostics["event_features"]["n_tracks"]
    }
    diagnostics["track_features"] = {
        name: diagnostics["track_features"][name]
        for name in ("start_z", "start_xy", "n_measurements", "n_unique_planes")
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def detector_columns():
    columns = {
        "count_veto1": [1, 1, 3, 1],
        "count_veto2": [1, 1, 0, 1],
        "count_veto3": [1, 1, 0, 1],
        "scifi_n_active_stations_xy": [5, 5, 5, 3],
    }
    for station in range(1, 6):
        columns[f"count_scifi{station}"] = [2, 10, 2, 2]
        columns[f"count_us{station}"] = [1, 1, 1, 1]
    for station in range(1, 5):
        columns[f"count_ds{station}"] = [1, 1, 1, int(station < 4)]
    for detector, count in (("veto", 3), ("scifi", 5), ("us", 5), ("ds", 4)):
        columns[f"count_{detector}"] = np.sum(
            [columns[f"count_{detector}{station}"] for station in range(1, count + 1)],
            axis=0,
        )
    return columns


def write_input(ROOT, path):
    values = detector_columns()
    values.update({
        "runId": [8329] * 4,
        "eventId": [1, 2, 3, 4],
        "reco_track_n_all": [1, 2, 0, 1],
        "reco_track_n_converged": [1, 1, 0, 1],
        "reco_track_n_types_present": [1, 1, 0, 1],
        "reco_scifi_hough_has_any": [1, 1, 0, 1],
        "reco_scifi_hough_has_converged": [1, 1, 0, 1],
        "reco_scifi_hough_has_best": [1, 1, 0, 1],
        "reco_scifi_hough_n_all": [1, 2, 0, 1],
        "reco_scifi_hough_n_converged": [1, 1, 0, 1],
        "reco_scifi_hough_best_start_x": [-30.0, -31.0, np.nan, -33.0],
        "reco_scifi_hough_best_start_y": [20.0, 21.0, np.nan, 23.0],
        "reco_scifi_hough_best_start_z": [300.0, 301.0, np.nan, 303.0],
        "reco_scifi_hough_best_stop_x": [-29.0, -30.0, np.nan, -32.0],
        "reco_scifi_hough_best_stop_y": [21.0, 22.0, np.nan, 24.0],
        "reco_scifi_hough_best_stop_z": [500.0, 501.0, np.nan, 503.0],
        "reco_scifi_hough_best_chi2ndf": [1.0, 2.0, np.nan, 4.0],
        "reco_scifi_hough_best_ndf": [6, 7, -1, 8],
        "reco_scifi_hough_best_slope_xz": [0.01, 0.02, np.nan, 0.04],
        "reco_scifi_hough_best_slope_yz": [0.02, 0.03, np.nan, 0.05],
        "reco_scifi_hough_best_n_measurements": [10, 11, 0, 13],
        "reco_scifi_hough_best_n_unique_planes": [8, 9, 0, 10],
    })
    arrays = {}
    integer_names = {
        name for name in values
        if name.startswith("count_")
        or name in {
            "runId", "eventId", "scifi_n_active_stations_xy",
            "reco_track_n_all", "reco_track_n_converged",
            "reco_track_n_types_present",
        }
        or "_has_" in name
        or name.endswith((
            "_n_all", "_n_converged", "_best_ndf",
            "_best_n_measurements", "_best_n_unique_planes",
        ))
    }
    for name, value in values.items():
        arrays[name] = np.asarray(
            value, dtype=np.int32 if name in integer_names else np.float64
        )
    result = ROOT.RDF.FromNumpy(arrays).Snapshot("sndData", str(path))
    if hasattr(result, "GetValue"):
        result.GetValue()


def pdf_pages(path):
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return int(
        next(line for line in output.splitlines() if line.startswith("Pages:"))
        .split(":", 1)[1]
    )


class PassingMuonRecoDiagnosticsV7Test(unittest.TestCase):
    def test_histograms_metadata_and_pdf(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(prefix="passing_muon_v7_test_") as directory:
            directory = Path(directory)
            input_root = directory / "tracks.root"
            manifest = directory / "manifest.csv"
            config = directory / "config.yaml"
            histogram = directory / "diagnostics.root"
            pdf = directory / "coverage_loose.pdf"
            write_input(ROOT, input_root)
            write_config(config)
            pd.DataFrame([{
                "run": 8329,
                "source_file_index": 0,
                "particle_group": "passing_muon_dq_run8329_flat_v2",
                "feature_path": str(input_root),
                "tree_name": "sndData",
                "n_events": 4,
                "track_feature_schema": "flat_v2",
            }]).to_csv(manifest, index=False)

            subprocess.run([
                sys.executable, str(HISTOGRAMMER),
                "--track-manifest", str(manifest),
                "--region-config", str(config),
                "--output", str(histogram),
                "--threads", "1",
            ], cwd=REPO_ROOT, check=True)
            source = ROOT.TFile.Open(str(histogram))
            self.assertTrue(source and not source.IsZombie())
            try:
                self.assertEqual(str(parameter(source, "track_feature_schema")), "flat_v2")
                self.assertEqual(int(parameter(source, "track_input_event_count")), 4)
                expected = {"coverage_loose": (2, 2), "coverage_tight": (1, 3)}
                for selection, (candidate, complement) in expected.items():
                    self.assertEqual(
                        int(parameter(source, f"selection__{selection}__candidate__count")),
                        candidate,
                    )
                    self.assertEqual(
                        int(parameter(source, f"selection__{selection}__complement__count")),
                        complement,
                    )
                    for population in ("candidate", "complement"):
                        base = (
                            f"selections/{selection}/tracks/scifi_hough/"
                            "features/start_z/" + population
                        )
                        self.assertTrue(source.Get(base))
                        self.assertTrue(source.Get(
                            f"selections/{selection}/plot_groups/hough/"
                            f"event_features/n_tracks/{population}"
                        ))
                self.assertEqual(
                    int(parameter(
                        source,
                        "selection__coverage_loose__complement__scifi_hough__has_best_count",
                    )),
                    1,
                )
                self.assertEqual(
                    int(parameter(
                        source,
                        "selection__coverage_loose__complement__scifi_hough__start_z__valid_count",
                    )),
                    1,
                )
            finally:
                source.Close()

            subprocess.run([
                sys.executable, str(PLOTTER),
                "--input", str(histogram),
                "--region-config", str(config),
                "--selection", "coverage_loose",
                "--plot-group", "hough",
                "--output-pdf", str(pdf),
            ], cwd=REPO_ROOT, check=True)
            self.assertTrue(pdf.is_file() and pdf.stat().st_size > 0)
            if shutil.which("pdfinfo"):
                self.assertEqual(pdf_pages(pdf), 9)


if __name__ == "__main__":
    unittest.main()
