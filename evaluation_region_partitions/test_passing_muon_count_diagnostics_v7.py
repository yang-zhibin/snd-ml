#!/usr/bin/env python3
"""Synthetic end-to-end test for V7 raw-count and N-minus-one PDFs."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

import test_passing_muon_binary_region_study_v5 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_binary_region_hist_v7.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_binary_region_counts_v7.py"
)
COVERAGE_STACK_PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/"
    "plot_passing_muon_coverage_stacked_counts_v7.py"
)


def run(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_config(path):
    config = yaml.safe_load(REGION_CONFIG.read_text(encoding="utf-8"))
    config["expected"] = {"real_data_dq": len(base.EVENTS)}
    config["mc_muon_up"]["expected_source_files"] = 2
    config["mc_muon_up"]["productions"] = {
        "scoring_1p8": config["mc_muon_up"]["productions"]["scoring_1p8"],
        "scoring_at449m": config["mc_muon_up"]["productions"]["scoring_at449m"],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def histogram_integral(histogram):
    return int(round(histogram.Integral(1, histogram.GetNbinsX())))


def assert_count_output(ROOT, path):
    source = ROOT.TFile.Open(str(path))
    assert source and not source.IsZombie()
    try:
        assert int(base.parameter(source, "uses_count_diagnostics")) == 1
        expected_veto = {
            "coverage_loose": (5, 4, 1),
            "coverage_tight": (3, 1, 2),
        }
        for selection, (parent, passed, failed) in expected_veto.items():
            prefix = f"selection__{selection}__count_diagnostic__veto"
            assert int(base.parameter(source, f"{prefix}__inclusive_count")) == parent
            assert int(base.parameter(source, f"{prefix}__pass_count")) == passed
            assert int(base.parameter(source, f"{prefix}__fail_count")) == failed
            assert int(base.parameter(source, f"{prefix}__population_closure")) == 1
            for feature in (
                "count_veto1", "count_veto2", "count_veto3", "count_veto",
                "veto_n_active_planes", "veto_excess_measurements",
            ):
                root_path = (
                    f"count_diagnostics/{selection}/veto/features/{feature}"
                )
                inclusive = source.Get(f"{root_path}/inclusive")
                pass_hist = source.Get(f"{root_path}/pass")
                fail_hist = source.Get(f"{root_path}/fail")
                assert inclusive and pass_hist and fail_hist
                assert histogram_integral(inclusive) == parent
                assert histogram_integral(pass_hist) == passed
                assert histogram_integral(fail_hist) == failed
                combined = pass_hist.Clone(
                    f"combined_test_{selection}_{feature}"
                )
                combined.SetDirectory(0)
                combined.Add(fail_hist)
                for bin_index in range(0, combined.GetNbinsX() + 2):
                    assert combined.GetBinContent(bin_index) == inclusive.GetBinContent(
                        bin_index
                    )

        tight_candidate = source.Get(
            "selections/coverage_tight/features/count_veto/"
            "through_going_candidate"
        )
        assert histogram_integral(tight_candidate) == 1
        assert tight_candidate.GetBinContent(tight_candidate.FindBin(3.0)) == 1
        tight_parent = source.Get(
            "count_diagnostics/coverage_tight/veto/features/count_veto/inclusive"
        )
        assert histogram_integral(tight_parent) == 3
        assert tight_parent.GetBinContent(tight_parent.FindBin(3.0)) == 2
        assert tight_parent.GetBinContent(tight_parent.FindBin(4.0)) == 1
    finally:
        source.Close()


def main():
    import ROOT

    ROOT.gROOT.SetBatch(True)
    with tempfile.TemporaryDirectory(prefix="passing_muon_counts_v7_") as directory:
        directory = Path(directory)
        real_root = directory / "real_data.root"
        mc_normal_root = directory / "mc_scoring_1p8.root"
        mc_zero_scifi_root = directory / "mc_scoring_at449m.root"
        metadata = directory / "real_data.csv"
        source_metadata = directory / "mc_source.csv"
        mc_metadata = directory / "mc.csv"
        mc_validation = directory / "mc_validation.csv"
        config = directory / "v7.yaml"
        output = directory / "v7.root"

        base.write_tree(ROOT, real_root)
        base.write_tree(ROOT, mc_normal_root)
        base.write_tree(ROOT, mc_zero_scifi_root, zero_scifi=True)
        base.write_metadata(metadata, [base.standardized_row(
            real_root,
            "real_data_dq_run8329",
            "all_events__real_data_dq_run8329__part001",
        )])
        base.write_source_metadata(source_metadata, {
            "scoring_1.8": mc_normal_root,
            "scoring_at449m": mc_zero_scifi_root,
        })
        write_config(config)
        run([
            sys.executable,
            str(base.PREPARER),
            "--source-metadata", str(source_metadata),
            "--study-config", str(config),
            "--hist-config", str(base.HIST_CONFIG),
            "--output-metadata", str(mc_metadata),
            "--output-validation", str(mc_validation),
            "--threads", "1",
        ])
        assert len(pd.read_csv(mc_metadata)) == 2
        run([
            sys.executable,
            str(HISTOGRAMMER),
            "--partition-metadata", str(metadata),
            "--mc-muon-up-metadata", str(mc_metadata),
            "--hist-config", str(base.HIST_CONFIG),
            "--region-config", str(config),
            "--output", str(output),
            "--threads", "1",
        ])
        assert_count_output(ROOT, output)

        for selection in ("coverage_loose", "coverage_tight"):
            pdf = directory / f"{selection}_counts.pdf"
            run([
                sys.executable,
                str(PLOTTER),
                "--input", str(output),
                "--hist-config", str(base.HIST_CONFIG),
                "--region-config", str(config),
                "--selection", selection,
                "--output-pdf", str(pdf),
            ])
            assert pdf.is_file() and pdf.stat().st_size > 0
            if shutil.which("pdfinfo"):
                assert base.pdf_pages(pdf) == 30

            stacked_pdf = directory / f"{selection}_stacked_counts.pdf"
            run([
                sys.executable,
                str(COVERAGE_STACK_PLOTTER),
                "--input", str(output),
                "--hist-config", str(base.HIST_CONFIG),
                "--region-config", str(config),
                "--selection", selection,
                "--output-pdf", str(stacked_pdf),
            ])
            assert stacked_pdf.is_file() and stacked_pdf.stat().st_size > 0
            if shutil.which("pdfinfo"):
                assert base.pdf_pages(stacked_pdf) == 47
    print("passing_muon V7 count diagnostics synthetic test: OK")


if __name__ == "__main__":
    main()
