#!/usr/bin/env python3
"""Synthetic end-to-end test for the consolidated run-8329 comparison plots."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HIST_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_run8329_hist_v2.yaml"
)
EVAL_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_run8329_eval_v2.yaml"
)
BUILDER = REPO_ROOT / "evaluation_region_partitions/make_passing_muon_efficiency_hist.py"
PLOTTER = REPO_ROOT / "evaluation_region_partitions/plot_passing_muon_efficiency.py"


BRANCH_TYPES = {
    "runId": ("q", "L"),
    "eventId": ("q", "L"),
    "count_veto": ("i", "I"),
    "count_scifi": ("i", "I"),
    "count_us": ("i", "I"),
    "count_ds": ("i", "I"),
    "qdc_veto": ("d", "D"),
    "hitTime_veto_min_ns": ("d", "D"),
    "qdc_scifi": ("d", "D"),
    "density_sndsw_scifi": ("d", "D"),
    "avg_scifi_x": ("d", "D"),
    "avg_scifi_y": ("d", "D"),
    "qdcAvg_scifi_x": ("d", "D"),
    "qdcAvg_scifi_y": ("d", "D"),
    "qdc_us": ("d", "D"),
    "qdc_ds": ("d", "D"),
}


EVENTS = [
    {"eventId": 1, "count_veto": 0, "hitTime_veto_min_ns": -999.0, "count_scifi": 10},
    {"eventId": 2, "count_veto": 1, "hitTime_veto_min_ns": -0.5, "count_scifi": 40},
    {"eventId": 3, "count_veto": 2, "hitTime_veto_min_ns": 3.5, "count_scifi": 80},
    {"eventId": 4, "count_veto": 4, "hitTime_veto_min_ns": 8.0, "count_scifi": 120},
    {"eventId": 5, "count_veto": 45, "hitTime_veto_min_ns": 35.0, "count_scifi": 900},
    {"eventId": 6, "count_veto": 2, "hitTime_veto_min_ns": -999.0, "count_scifi": 50},
]


def complete_event(partial: dict) -> dict:
    event = {
        "runId": 8329,
        "eventId": 0,
        "count_veto": 0,
        "count_scifi": 0,
        "count_us": 0,
        "count_ds": 0,
        "qdc_veto": 0.0,
        "hitTime_veto_min_ns": -999.0,
        "qdc_scifi": 0.0,
        "density_sndsw_scifi": 0.0,
        "avg_scifi_x": -20.0,
        "avg_scifi_y": 30.0,
        "qdcAvg_scifi_x": -20.0,
        "qdcAvg_scifi_y": 30.0,
        "qdc_us": 0.0,
        "qdc_ds": 0.0,
    }
    event.update(partial)
    scale = float(event["eventId"])
    event["count_us"] = int(event["eventId"] % 3)
    event["count_ds"] = int(event["eventId"] % 4)
    event["qdc_veto"] = 10.0 * scale
    event["qdc_scifi"] = 100.0 * scale
    event["density_sndsw_scifi"] = 500.0 * scale
    event["avg_scifi_x"] = -30.0 + scale
    event["avg_scifi_y"] = 20.0 + scale
    event["qdcAvg_scifi_x"] = -29.5 + scale
    event["qdcAvg_scifi_y"] = 20.5 + scale
    event["qdc_us"] = 20.0 * scale
    event["qdc_ds"] = 30.0 * scale
    return event


def write_tree(ROOT, path: Path, events: list[dict]) -> None:
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic run-8329 comparison")
    storage = {}
    for name, (array_type, leaf_type) in BRANCH_TYPES.items():
        storage[name] = array(array_type, [0])
        tree.Branch(name, storage[name], f"{name}/{leaf_type}")
    for partial in events:
        event = complete_event(partial)
        for name, value in event.items():
            storage[name][0] = value
        tree.Fill()
    tree.Write()
    root_file.Close()


def write_metadata(path: Path, real_path: Path, passing_path: Path) -> None:
    rows = [
        {
            "particle_group": "real_data_dq_run8329",
            "partition_id": "all_events__real_data_dq_run8329__part001",
            "feature_partition_path": str(real_path),
            "lumi_per_partition": 0.5,
        },
        {
            "particle_group": "passing_muon_dq_run8329",
            "partition_id": "all_events__passing_muon_dq_run8329__part001",
            "feature_partition_path": str(passing_path),
            "lumi_per_partition": 0.5,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def pdf_pages(path: Path) -> int:
    if not shutil.which("pdfinfo"):
        raise RuntimeError("pdfinfo is required for the PDF-page assertion")
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    pages = [line for line in result.splitlines() if line.startswith("Pages:")]
    if len(pages) != 1:
        raise AssertionError(f"Could not determine page count for {path}")
    return int(pages[0].split(":", 1)[1])


def assert_root_contents(ROOT, path: Path, selection: str) -> None:
    config = yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))
    root_file = ROOT.TFile.Open(str(path))
    assert root_file and not root_file.IsZombie()
    assert root_file.Get("selection_key").GetTitle() == selection
    assert int(root_file.Get("real_data_selected").GetVal()) == (
        1 if selection == "no_veto" else 6
    )
    assert int(root_file.Get("passing_muon_selected").GetVal()) == (
        1 if selection == "no_veto" else 3
    )
    for feature_name, feature in config["features"].items():
        real_hist = root_file.Get(f"{feature_name}/real_data")
        passing_hist = root_file.Get(f"{feature_name}/passing_muon")
        assert real_hist and passing_hist, feature_name
        assert real_hist.GetDimension() == int(feature.get("dimension", 1))
        assert passing_hist.GetDimension() == int(feature.get("dimension", 1))
        if real_hist.GetDimension() == 1 and real_hist.Integral() > 0:
            assert ROOT.TEfficiency.CheckConsistency(passing_hist, real_hist)

    inclusive_time = root_file.Get("veto_earliest_time_inclusive/real_data")
    assert inclusive_time.GetBinContent(inclusive_time.FindBin(-1.0)) == 1.0
    if selection == "none":
        assert inclusive_time.Integral() == 5.0
        valid_time = root_file.Get("hitTime_veto_min_ns/real_data")
        assert valid_time.Integral() == 4.0
        count = root_file.Get("count_veto/real_data")
        assert count.GetBinContent(count.GetNbinsX()) == 1.0
        two_dimensional = root_file.Get("veto_time_vs_scifi_count/real_data")
        assert two_dimensional.Integral() == 5.0
    else:
        valid_time = root_file.Get("hitTime_veto_min_ns/real_data")
        assert valid_time.Integral() == 0.0
    root_file.Close()


def run_workflow_test(workdir: Path) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    real_path = workdir / "real.root"
    passing_path = workdir / "passing.root"
    metadata_path = workdir / "metadata.csv"
    write_tree(ROOT, real_path, EVENTS)
    write_tree(ROOT, passing_path, [EVENTS[index] for index in (0, 2, 4)])
    write_metadata(metadata_path, real_path, passing_path)
    expected_pages = len(
        yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"]
    )

    for selection in ("none", "no_veto"):
        histogram_path = workdir / "histograms" / f"{selection}.root"
        pdf_path = workdir / "comparison" / f"{selection}.pdf"
        run([
            sys.executable, str(BUILDER),
            "--metadata", str(metadata_path),
            "--hist-config", str(HIST_CONFIG),
            "--eval-config", str(EVAL_CONFIG),
            "--selection", selection,
            "--output", str(histogram_path),
            "--threads", "1",
        ])
        assert_root_contents(ROOT, histogram_path, selection)
        run([
            sys.executable, str(PLOTTER),
            "--input", str(histogram_path),
            "--hist-config", str(HIST_CONFIG),
            "--eval-config", str(EVAL_CONFIG),
            "--output-pdf", str(pdf_path),
        ])
        assert pdf_path.is_file() and pdf_path.stat().st_size > 0
        assert pdf_pages(pdf_path) == expected_pages

    # Comparison histogram/plot stages must not create efficiency CSV files.
    assert list(workdir.rglob("*.csv")) == [metadata_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.keep:
        workdir = Path(tempfile.mkdtemp(prefix="passing_muon_comparison_v2_test."))
        run_workflow_test(workdir)
        print(f"PASS; retained outputs at {workdir}")
        return
    with tempfile.TemporaryDirectory(
        prefix="passing_muon_comparison_v2_test."
    ) as value:
        run_workflow_test(Path(value))
    print("PASS")


if __name__ == "__main__":
    main()
