#!/usr/bin/env python3
"""Synthetic end-to-end test for the run-8329 region study v4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

import pandas as pd
import yaml

from passing_muon_v4_features import expanded_v4_hist_config


REPO_ROOT = Path(__file__).resolve().parents[1]
HIST_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_run8329_hist_v2.yaml"
)
REGION_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_region_study_v4.yaml"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_region_hist_v4.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_region_study_v4.py"
)


BRANCH_TYPES = {
    "source_file_index": ("i", "I"),
    "runId": ("q", "L"),
    "eventId": ("q", "L"),
    "eventIndex": ("q", "L"),
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
for detector, stations in (("scifi", range(1, 6)), ("ds", range(1, 5))):
    for station in stations:
        for coordinate in ("x", "y"):
            BRANCH_TYPES[f"avg_{detector}{station}_{coordinate}"] = ("d", "D")


# event_id: (veto count, earliest veto time, US count)
EVENT_REGIONS = {
    1: (0, -999.0, 0),
    2: (0, -999.0, 1),
    3: (1, 0.0, 4),       # exact lower boundary belongs to [0,2]
    4: (2, 2.0, 5),       # exact 2 belongs to [0,2]
    5: (3, 2.0001, 7),
    6: (4, 6.9999, 8),
    7: (1, 7.0, 12),      # exact 7 belongs to >=7
    8: (3, 12.0, 0),
    9: (5, 1.0, 6),
    10: (2, -999.0, 9),   # positive-hit invalid sentinel
    11: (4, math.nan, 2), # positive-hit non-finite time
    12: (1, -0.1, 5),     # valid negative-time audit
    13: (0, -999.0, 8),
    14: (3, 7.0, 4),
    15: (4, 0.0, 7),
    16: (2, 7.0, 8),
    17: (0, -999.0, 0),   # NoPassingMuonReco
    18: (4, 8.0, 10),     # NoPassingMuonReco
}


def event_values(event_id):
    veto_count, veto_time, us_count = EVENT_REGIONS[event_id]
    value = float(event_id)
    result = {
        "source_file_index": 1,
        "runId": 8329,
        "eventId": event_id,
        "eventIndex": event_id - 1,
        "count_veto": veto_count,
        "count_scifi": 20 + 10 * event_id,
        "count_us": us_count,
        "count_ds": event_id % 9,
        "qdc_veto": 10.0 * value,
        "hitTime_veto_min_ns": veto_time,
        "qdc_scifi": 50000.0 if event_id == 18 else 100.0 * value,
        "density_sndsw_scifi": 400.0 * value,
        "avg_scifi_x": -35.0 + value,
        "avg_scifi_y": 20.0 + value,
        "qdcAvg_scifi_x": -34.5 + value,
        "qdcAvg_scifi_y": 20.5 + value,
        "qdc_us": 25.0 * value,
        "qdc_ds": 30.0 * value,
    }
    for station in range(1, 6):
        result[f"avg_scifi{station}_x"] = -45.0 + value + 0.2 * station
        result[f"avg_scifi{station}_y"] = 10.0 + value + 0.2 * station
    for station in range(1, 5):
        result[f"avg_ds{station}_x"] = -50.0 + value + 0.2 * station
        result[f"avg_ds{station}_y"] = 5.0 + value + 0.2 * station
    # Verify that the generated feature cut rejects an incomplete x-y pair.
    if event_id == 18:
        result["avg_ds4_y"] = -999.0
    return result


def configured_test_features(config):
    hist_config = yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))
    return expanded_v4_hist_config(hist_config, config)["features"]


def write_tree(ROOT, path, event_ids):
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic v4 region study")
    storage = {}
    for name, (array_type, leaf_type) in BRANCH_TYPES.items():
        storage[name] = array(array_type, [0])
        tree.Branch(name, storage[name], f"{name}/{leaf_type}")
    for event_id in event_ids:
        for name, value in event_values(event_id).items():
            storage[name][0] = value
        tree.Fill()
    tree.Write()
    root_file.Close()


def write_metadata(path, group, root_path, count):
    pd.DataFrame(
        [
            {
                "region": "all_events",
                "particle_group": group,
                "partition_id": f"all_events__{group}__part001",
                "feature_partition_path": str(root_path),
                "feature_available": True,
                "n_events": count,
                "lumi_per_partition": 1.0,
                "source_file_indices": "1",
                "source_file_count": 1,
            }
        ]
    ).to_csv(path, index=False)


def write_test_config(path):
    config = yaml.safe_load(REGION_CONFIG.read_text(encoding="utf-8"))
    config["expected"] = {
        "real_data_dq": 18,
        "passing_muon_dq": 16,
        "no_passing_muon_reco": 2,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def pdf_pages(path):
    if not shutil.which("pdfinfo"):
        raise RuntimeError("pdfinfo is required for PDF-page assertions")
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    values = [line for line in result.splitlines() if line.startswith("Pages:")]
    if len(values) != 1:
        raise AssertionError(f"Cannot determine page count for {path}")
    return int(values[0].split(":", 1)[1])


def parameter(source, name):
    value = source.Get(name)
    assert value, name
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def assert_output(ROOT, path, config):
    source = ROOT.TFile.Open(str(path))
    assert source and not source.IsZombie()
    assert int(parameter(source, "passing_count")) == 16
    assert int(parameter(source, "missing_count")) == 2
    assert int(parameter(source, "real_data_dq_count")) == 18
    assert int(parameter(source, "joint_region_count")) == 40
    assert len(str(parameter(source, "joint_region_order")).split(",")) == 40
    features = configured_test_features(config)
    station_features = [
        *(f"avg_scifi{station}_x_vs_y" for station in range(1, 6)),
        *(f"avg_ds{station}_x_vs_y" for station in range(1, 5)),
    ]
    assert all(name in features for name in station_features)
    assert len(station_features) == 9
    assert str(parameter(source, "feature_order")).split(",") == list(features)

    expected_passing = {
        "veto_count": {
            "veto_n0": 3, "veto_n1_2": 6, "veto_n3": 3, "veto_nge4": 4,
        },
        "veto_time": {
            "veto_time_none": 3,
            "veto_time_0_2": 4,
            "veto_time_gt2_lt7": 2,
            "veto_time_ge7": 4,
            "veto_time_invalid": 2,
            "veto_time_negative": 1,
        },
        "us_count": {
            "us_n0": 2, "us_n1_4": 4, "us_n5_7": 5, "us_nge8": 5,
        },
    }
    for axis, categories in expected_passing.items():
        observed = 0
        for category, expected in categories.items():
            value = int(
                parameter(
                    source, f"axis__{axis}__{category}__passing__count"
                )
            )
            assert value == expected
            observed += value
        assert observed == 16
        assert int(parameter(source, f"axis__{axis}__passing__exclusive_closure")) == 1

    selected_expected = {
        "veto_n0__veto_time_none__us_n0": (1, 1),
        "veto_n0__veto_time_none__us_n1_4": (1, 0),
        "veto_n1_2__veto_time_0_2__us_n1_4": (1, 0),
        "veto_n3__veto_time_gt2_lt7__us_n5_7": (1, 0),
        "veto_nge4__veto_time_ge7__us_nge8": (0, 1),
    }
    for region, (passing, missing) in selected_expected.items():
        assert int(parameter(source, f"joint__{region}__passing__count")) == passing
        assert int(parameter(source, f"joint__{region}__missing__count")) == missing
        for feature_name in features:
            assert source.Get(
                f"joint_regions/{region}/{feature_name}/passing_muon_reco"
            )
            assert source.Get(
                f"joint_regions/{region}/{feature_name}/no_passing_muon_reco"
            )

    passing_joint = sum(
        int(parameter(source, f"joint__{region}__passing__count"))
        for region in str(parameter(source, "joint_region_order")).split(",")
    )
    missing_joint = sum(
        int(parameter(source, f"joint__{region}__missing__count"))
        for region in str(parameter(source, "joint_region_order")).split(",")
    )
    assert passing_joint == 13  # excludes two invalid and one negative time
    assert missing_joint == 2

    qdc = source.Get(
        "marginal/veto_count/veto_nge4/qdc_scifi/no_passing_muon_reco"
    )
    assert qdc.GetXaxis().GetXmax() == 30000.0
    assert qdc.GetBinContent(qdc.GetNbinsX()) == 1

    for feature_name in (
        "scifi_x_vs_y",
        "qdc_weighted_scifi_x_vs_y",
    ):
        spatial = source.Get(
            f"marginal/veto_count/veto_nge4/{feature_name}/"
            "no_passing_muon_reco"
        )
        assert spatial
        assert spatial.GetDimension() == 2
        assert spatial.GetXaxis().GetXmin() == -80.0
        assert spatial.GetXaxis().GetXmax() == 20.0
        assert spatial.GetYaxis().GetXmin() == 0.0
        assert spatial.GetYaxis().GetXmax() == 80.0
        assert spatial.Integral() == 1

    for feature_name in station_features:
        spatial = source.Get(
            f"marginal/veto_count/veto_nge4/{feature_name}/"
            "no_passing_muon_reco"
        )
        assert spatial
        assert spatial.GetDimension() == 2
        if feature_name.startswith("avg_scifi"):
            assert spatial.GetXaxis().GetXmin() == -80.0
            assert spatial.GetXaxis().GetXmax() == 20.0
            assert spatial.GetYaxis().GetXmin() == 0.0
            assert spatial.GetYaxis().GetXmax() == 80.0
            assert spatial.Integral() == 1
        else:
            assert spatial.GetXaxis().GetXmin() == -60.0
            assert spatial.GetXaxis().GetXmax() == 15.0
            assert spatial.GetYaxis().GetXmin() == -5.0
            assert spatial.GetYaxis().GetXmax() == 80.0
            expected = 0 if feature_name == "avg_ds4_x_vs_y" else 1
            assert spatial.Integral() == expected
    assert int(parameter(
        source,
        "selected__marginal__veto_count__veto_nge4__missing__avg_ds4_x_vs_y",
    )) == 0
    source.Close()


def run_workflow_test(workdir):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    passing_root = workdir / "passing.root"
    missing_root = workdir / "missing.root"
    paired_metadata = workdir / "partition_metadata.csv"
    missing_metadata = workdir / "missing_metadata.csv"
    test_config = workdir / "region_config.yaml"
    output_root = workdir / "regions.root"
    write_tree(ROOT, passing_root, range(1, 17))
    write_tree(ROOT, missing_root, range(17, 19))
    write_metadata(
        paired_metadata, "passing_muon_dq_run8329", passing_root, 16
    )
    write_metadata(
        missing_metadata, "no_passing_muon_reco_run8329", missing_root, 2
    )
    write_test_config(test_config)
    run(
        [
            sys.executable,
            str(HISTOGRAMMER),
            "--partition-metadata",
            str(paired_metadata),
            "--missing-metadata",
            str(missing_metadata),
            "--hist-config",
            str(HIST_CONFIG),
            "--region-config",
            str(test_config),
            "--output",
            str(output_root),
            "--threads",
            "1",
        ]
    )
    config = yaml.safe_load(test_config.read_text(encoding="utf-8"))
    assert_output(ROOT, output_root, config)

    comparisons = [f"{axis}_regions" for axis in config["axes"]]
    comparisons.append("joint_region_summary")
    comparisons.extend(
        f"region__{region}" for region in config["selected_joint_regions"]
    )
    feature_pages = 1 + len(configured_test_features(config))
    for comparison in comparisons:
        output_pdf = workdir / f"{comparison}.pdf"
        run(
            [
                sys.executable,
                str(PLOTTER),
                "--input",
                str(output_root),
                "--hist-config",
                str(HIST_CONFIG),
                "--region-config",
                str(test_config),
                "--comparison",
                comparison,
                "--output-pdf",
                str(output_pdf),
            ]
        )
        assert output_pdf.is_file() and output_pdf.stat().st_size > 0
        expected_pages = 5 if comparison == "joint_region_summary" else feature_pages
        assert pdf_pages(output_pdf) == expected_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.keep:
        workdir = Path(tempfile.mkdtemp(prefix="passing_muon_region_v4_test."))
        run_workflow_test(workdir)
        print(f"PASS; retained outputs at {workdir}")
        return
    with tempfile.TemporaryDirectory(
        prefix="passing_muon_region_v4_test."
    ) as value:
        run_workflow_test(Path(value))
    print("PASS")


if __name__ == "__main__":
    main()
