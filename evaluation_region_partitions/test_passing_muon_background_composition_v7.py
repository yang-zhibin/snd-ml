#!/usr/bin/env python3
"""Synthetic closure and PDF test for V7 background composition."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

import test_passing_muon_reco_diagnostics_v7 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
BUILDER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_background_composition_v7.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_background_composition_v7.py"
)


def parameter(source, name):
    value = source.Get(name)
    assert value, name
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def pages(path):
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return int(next(
        line.split(":", 1)[1] for line in output.splitlines()
        if line.startswith("Pages:")
    ))


def write_config(path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    block = config["background_composition"]
    block["track_definition"] = {
        "key": "scifi_hough",
        "valid_expression": "reco_scifi_hough_has_best == 1",
        "property_prefix": "reco_scifi_hough",
        "required_prefixes": ["reco_scifi_hough"],
        "label": "Synthetic SciFi Hough best track",
    }
    block["features"] = {
        name: block["features"][name]
        for name in ("count_veto", "track_n_measurements", "veto_excess_vs_scifi")
    }
    block["plot_sections"] = {
        "counts": {
            "label": "Synthetic counts",
            "features": ["count_veto", "track_n_measurements"],
        },
        "correlations": {
            "label": "Synthetic correlations",
            "features": ["veto_excess_vs_scifi"],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main():
    import ROOT

    ROOT.gROOT.SetBatch(True)
    with tempfile.TemporaryDirectory(prefix="composition_v7_test_") as directory:
        directory = Path(directory)
        input_root = directory / "input.root"
        manifest = directory / "manifest.csv"
        config = directory / "config.yaml"
        output = directory / "composition.root"
        pdf = directory / "coverage_loose.pdf"
        base.write_input(ROOT, input_root)
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
            sys.executable, str(BUILDER),
            "--track-manifest", str(manifest),
            "--region-config", str(config),
            "--output", str(output),
            "--threads", "1",
        ], cwd=REPO_ROOT, check=True)

        source = ROOT.TFile.Open(str(output))
        assert source and not source.IsZombie()
        try:
            assert str(parameter(source, "analysis_scope")) == "passing_muon_reco_only"
            expected = {
                "coverage_loose": (1, 2, 1),
                "coverage_tight": (1, 1, 2),
            }
            for selection, values in expected.items():
                assert int(parameter(source, f"selection__{selection}__total_count")) == 4
                for class_name, count in zip(
                    (
                        "no_valid_track",
                        "track_like_low_activity",
                        "track_plus_activity",
                    ),
                    values,
                ):
                    prefix = f"selection__{selection}__class__{class_name}"
                    assert int(parameter(source, f"{prefix}__count")) == count
                    histogram = source.Get(
                        f"selections/{selection}/features/count_veto/{class_name}"
                    )
                    assert histogram
                    assert int(round(histogram.Integral())) == count
                inclusive = source.Get(
                    f"selections/{selection}/features/count_veto/inclusive"
                )
                assert inclusive and int(round(inclusive.Integral())) == 4
                assert int(parameter(source, f"selection__{selection}__class_closure")) == 1
            assert int(parameter(
                source,
                "selection__coverage_loose__class__no_valid_track__feature__"
                "track_n_measurements__valid_count",
            )) == 0
        finally:
            source.Close()

        subprocess.run([
            sys.executable, str(PLOTTER),
            "--input", str(output),
            "--region-config", str(config),
            "--selection", "coverage_loose",
            "--output-pdf", str(pdf),
        ], cwd=REPO_ROOT, check=True)
        assert pdf.is_file() and pdf.stat().st_size > 0
        if shutil.which("pdfinfo"):
            assert pages(pdf) == 6
    print("passing_muon V7 background composition synthetic test: OK")


if __name__ == "__main__":
    main()
