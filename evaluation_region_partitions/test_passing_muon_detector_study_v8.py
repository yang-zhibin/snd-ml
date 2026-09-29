#!/usr/bin/env python3
"""Synthetic population-closure, metadata, and PDF test for V8."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import ROOT
import yaml

import test_passing_muon_binary_region_study_v5 as base
from passing_muon_v8 import configured_features


ROOT.gROOT.SetBatch(True)
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
V8_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detector_study_v8.yaml"
)
HIST_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_run8329_hist_v2.yaml"
)
BUILDER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_detector_study_v8.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_detector_study_v8.py"
)


def parameter(source, name):
    value = source.Get(name)
    assert value, name
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def pdf_pages(path):
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return int(next(
        line.split(":", 1)[1] for line in output.splitlines()
        if line.startswith("Pages:")
    ))


def write_configs(directory):
    base_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    base_config["expected"] = {"real_data_dq": 4}
    keep = ("scoring_1p8", "scoring_at449m")
    base_config["mc_muon_up"]["productions"] = {
        key: base_config["mc_muon_up"]["productions"][key] for key in keep
    }
    base_path = directory / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
    )
    overlay = yaml.safe_load(V8_CONFIG.read_text(encoding="utf-8"))
    overlay["base_config"] = str(base_path)
    overlay_path = directory / "v8.yaml"
    overlay_path.write_text(
        yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
    )
    return base_path, overlay_path, base_config


def main():
    original_events = base.EVENTS
    base.EVENTS = {
        1: original_events[1],
        2: original_events[3],
        3: ([0, 0, 0], [2, 2, 2, 2, 2], 5, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
        4: original_events[4],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="passing_muon_v8_test_") as temp:
            directory = Path(temp)
            data_root = directory / "data.root"
            mc_a_root = directory / "mc_a.root"
            mc_b_root = directory / "mc_b.root"
            for path in (data_root, mc_a_root, mc_b_root):
                base.write_tree(ROOT, path)

            data_metadata = directory / "data.csv"
            pd.DataFrame([{
                **base.standardized_row(
                    data_root, "real_data_dq_run8329", "data_part"
                ),
                "lumi_per_partition": 0.5,
            }]).to_csv(data_metadata, index=False)
            mc_metadata = directory / "mc.csv"
            pd.DataFrame([
                {
                    **base.standardized_row(
                        mc_a_root, "mc_muon_up_scoring_1p8", "mc_a", "scoring_1p8"
                    ),
                    "lumi_per_partition": 1.0e-6,
                },
                {
                    **base.standardized_row(
                        mc_b_root, "mc_muon_up_scoring_at449m", "mc_b", "scoring_at449m"
                    ),
                    "lumi_per_partition": 2.0e-6,
                },
            ]).to_csv(mc_metadata, index=False)
            _, config, base_config = write_configs(directory)
            histogram = directory / "v8.root"
            subprocess.run([
                sys.executable, str(BUILDER),
                "--partition-metadata", str(data_metadata),
                "--mc-muon-up-metadata", str(mc_metadata),
                "--hist-config", str(HIST_CONFIG),
                "--region-config", str(config),
                "--output", str(histogram),
                "--threads", "1",
            ], cwd=REPO_ROOT, check=True)

            source = ROOT.TFile.Open(str(histogram))
            assert source and not source.IsZombie()
            try:
                assert str(parameter(source, "study_version")) == "passing_muon_detector_study_v8"
                assert int(parameter(source, "data_year")) == 2024
                assert int(parameter(source, "data_event_count")) == 4
                assert abs(float(parameter(source, "data_luminosity_fb")) - 0.5) < 1e-12
                assert int(parameter(source, "population__passing_muon_candidate__data_count")) == 1
                assert int(parameter(source, "population__muon_dis_candidate__data_count")) == 2
                assert int(parameter(source, "population__signal_region__data_count")) == 1
                assert int(parameter(source, "population__passing_muon_candidate__mc_count")) == 2
                assert int(parameter(source, "population__muon_dis_candidate__mc_count")) == 4
                assert int(parameter(source, "population__signal_region__mc_count")) == 2
                assert abs(float(parameter(source, "mc_muon_up_luminosity_fb")) - 3e-6) < 1e-15
                assert source.Get(
                    "data/populations/signal_region/features/count_veto/histogram"
                )
            finally:
                source.Close()

            hist_config = yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))
            expected_pages = 1 + len(configured_features(hist_config, base_config))
            for mode in ("passing_muon_selection", "signal_region"):
                pdf = directory / f"{mode}.pdf"
                subprocess.run([
                    sys.executable, str(PLOTTER),
                    "--input", str(histogram),
                    "--hist-config", str(HIST_CONFIG),
                    "--region-config", str(config),
                    "--mode", mode,
                    "--output-pdf", str(pdf),
                ], cwd=REPO_ROOT, check=True)
                assert pdf.is_file() and pdf.stat().st_size > 0
                if shutil.which("pdfinfo"):
                    assert pdf_pages(pdf) == expected_pages
    finally:
        base.EVENTS = original_events
    print("passing_muon V8 synthetic test: OK")


if __name__ == "__main__":
    main()
