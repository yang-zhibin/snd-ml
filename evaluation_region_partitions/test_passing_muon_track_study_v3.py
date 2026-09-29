#!/usr/bin/env python3
"""Synthetic end-to-end test for the run-8329 track-type study v3."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
HIST_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_run8329_hist_v2.yaml"
)
STUDY_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_track_study_v3.yaml"
)
ANTI_JOIN = REPO_ROOT / "convertData/build_no_passing_muon_partition.py"
VALIDATOR = REPO_ROOT / "convertData/validate_passing_muon_track_study_v3.py"
MC_PREPARATOR = REPO_ROOT / (
    "snakemake/metadata/prepare_mc_muon_up_reference_v3.py"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_track_type_hist.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_track_type.py"
)


COMMON_BRANCH_TYPES = {
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

PREFIXES = (
    "reco_scifi_simple",
    "reco_ds_simple",
    "reco_scifi_hough",
    "reco_ds_hough",
    "reco_combined_hough",
)


def complete_event(event_id):
    value = float(event_id)
    return {
        "source_file_index": 1,
        "runId": 8329,
        "eventId": event_id,
        "eventIndex": event_id - 1,
        "count_veto": event_id % 5,
        "count_scifi": 20 + 10 * event_id,
        "count_us": event_id % 4,
        "count_ds": event_id % 6,
        "qdc_veto": 10.0 * value,
        # Events 5 and 10 have no veto hit and must appear at -1 ns in the
        # inclusive timing plots. Event 8 has veto hits but an invalid time
        # and must be excluded only from timing-dependent features.
        "hitTime_veto_min_ns": (
            -999.0 if event_id in (5, 8, 10) else 0.5 * value
        ),
        # Exercise explicit overflow folding at the configured 30k maximum.
        "qdc_scifi": 50000.0 if event_id == 10 else 100.0 * value,
        "density_sndsw_scifi": 400.0 * value,
        "avg_scifi_x": -35.0 + value,
        "avg_scifi_y": 20.0 + value,
        "qdcAvg_scifi_x": -34.5 + value,
        "qdcAvg_scifi_y": 20.5 + value,
        "qdc_us": 25.0 * value,
        "qdc_ds": 30.0 * value,
    }


TRACK_ASSIGNMENTS = {
    1: {"reco_scifi_simple": 5.0},
    2: {"reco_scifi_simple": 10.0},
    3: {"reco_ds_simple": 15.0},
    4: {"reco_scifi_hough": 25.0},
    5: {"reco_ds_hough": 45.0},
    6: {"reco_combined_hough": 60.0},
    7: {"reco_scifi_simple": 8.0, "reco_scifi_hough": 12.0},
    8: {"reco_ds_simple": None},
    9: {prefix: 1.0 + index for index, prefix in enumerate(PREFIXES)},
    10: {"reco_ds_simple": 7.0},
}


def write_tree(ROOT, path, event_ids, with_tracks):
    branch_types = dict(COMMON_BRANCH_TYPES)
    if with_tracks:
        branch_types["reco_track_n_types_present"] = ("i", "I")
        for prefix in PREFIXES:
            branch_types[f"{prefix}_has_converged"] = ("i", "I")
            branch_types[f"{prefix}_has_best"] = ("i", "I")
            branch_types[f"{prefix}_best_chi2ndf"] = ("d", "D")
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic track study")
    storage = {}
    for name, (array_type, leaf_type) in branch_types.items():
        storage[name] = array(array_type, [0])
        tree.Branch(name, storage[name], f"{name}/{leaf_type}")
    for event_id in event_ids:
        values = complete_event(event_id)
        if with_tracks:
            assigned = TRACK_ASSIGNMENTS[event_id]
            values["reco_track_n_types_present"] = len(assigned)
            for prefix in PREFIXES:
                score = assigned.get(prefix, "absent")
                values[f"{prefix}_has_converged"] = int(score != "absent")
                values[f"{prefix}_has_best"] = int(
                    score != "absent" and score is not None
                )
                values[f"{prefix}_best_chi2ndf"] = (
                    float(score)
                    if score != "absent" and score is not None
                    else math.nan
                )
        for name, value in values.items():
            storage[name][0] = value
        tree.Fill()
    if not with_tracks:
        # Production real-DQ partitions are RDataFrame snapshots made with
        # implicit multithreading and carry this bit. The missing-event writer
        # must work without attaching an unsafe entry-aligned friend tree.
        tree.SetBit(ROOT.TTree.kEntriesReshuffled)
    tree.Write()
    root_file.Close()


def write_mc_tree(ROOT, path):
    branch_types = dict(COMMON_BRANCH_TYPES)
    branch_types["pdgCode"] = ("i", "I")
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic track-free MC muon-up reference")
    storage = {}
    for name, (array_type, leaf_type) in branch_types.items():
        storage[name] = array(array_type, [0])
        tree.Branch(name, storage[name], f"{name}/{leaf_type}")
    detector_counts = {
        1: (13, 0, 0, 0, 0),       # muon, no detector activity
        2: (13, 1, 20, 5, 4),      # selected
        3: (-13, 2, 30, 6, 5),     # selected
        4: (13, 0, 40, 7, 6),      # missing veto
        5: (13, 1, 0, 8, 7),       # missing SciFi
        6: (13, 1, 50, 0, 8),      # missing US
        7: (13, 1, 60, 9, 0),      # missing DS
        8: (211, 1, 70, 10, 9),    # non-muon
    }
    for event_id, (pdg, veto, scifi, us, ds) in detector_counts.items():
        values = complete_event(event_id)
        values.update({
            "pdgCode": pdg,
            "count_veto": veto,
            "count_scifi": scifi,
            "count_us": us,
            "count_ds": ds,
        })
        if scifi == 0:
            values.update({
                "avg_scifi_x": -999.0,
                "avg_scifi_y": -999.0,
                "qdcAvg_scifi_x": -999.0,
                "qdcAvg_scifi_y": -999.0,
            })
        for name, value in values.items():
            storage[name][0] = value
        tree.Fill()
    tree.Write()
    root_file.Close()


def write_mc_source_metadata(path, root_path):
    pd.DataFrame([{
        "data_type": "MC_muon",
        "subfolder": "up",
        "partition": "synthetic",
        "n_event": 8,
        "output_base_path": str(root_path.parent),
        "feature_path": root_path.name,
        "lumi_per_file": 1.0,
    }]).to_csv(path, index=False)


def write_partition_metadata(path, real_path, passing_path):
    common = {
        "region": "all_events",
        "lumi_per_partition": 0.5,
        "source_file_indices": "1",
        "source_file_count": 1,
        "feature_available": True,
    }
    rows = [
        {
            **common,
            "particle_group": "real_data_dq_run8329",
            "partition_id": "all_events__real_data_dq_run8329__part001",
            "feature_partition_path": str(real_path),
            "n_events": 12,
        },
        {
            **common,
            "particle_group": "passing_muon_dq_run8329",
            "partition_id": "all_events__passing_muon_dq_run8329__part001",
            "feature_partition_path": str(passing_path),
            "n_events": 10,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_test_config(path):
    config = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    config["expected"] = {
        "n_partitions": 1,
        "real_data_dq": 12,
        "passing_muon_dq": 10,
        "no_passing_muon_reco": 2,
    }
    config["working_points"] = {
        "no_quality_cut": config["working_points"]["no_quality_cut"],
        "chi2ndf_lt_10": config["working_points"]["chi2ndf_lt_10"],
    }
    config["mc_muon_up"]["source"] = {
        "subfolder": "up",
        "partition": "synthetic",
        "feature_name": "mc_muon_up.root",
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
    line = [value for value in result.splitlines() if value.startswith("Pages:")]
    if len(line) != 1:
        raise AssertionError(f"Could not determine page count for {path}")
    return int(line[0].split(":", 1)[1])


def assert_missing(ROOT, missing_path, validation_path):
    arrays = ROOT.RDataFrame("sndData", str(missing_path)).AsNumpy(["eventId"])
    assert set(map(int, arrays["eventId"])) == {11, 12}
    validation = pd.read_csv(validation_path)
    overall = validation.loc[validation["scope"] == "overall"].iloc[0]
    assert int(overall["real_data_dq"]) == 12
    assert int(overall["passing_muon_dq"]) == 10
    assert int(overall["no_passing_muon_reco"]) == 2
    assert int(overall["count_closure"]) == 1


def assert_histograms(ROOT, path, working_point):
    source = ROOT.TFile.Open(str(path))
    assert source and not source.IsZombie()
    assert source.Get("working_point").GetTitle() == working_point
    assert int(source.Get("passing_count").GetVal()) == 10
    assert int(source.Get("missing_count").GetVal()) == 2
    assert int(source.Get("mc_muon_up_inclusive_count").GetVal()) == 7
    assert int(source.Get("mc_muon_up_reference_count").GetVal()) == 6
    assert "count_veto + count_scifi" in source.Get(
        "mc_muon_up_reference_expression"
    ).GetTitle()
    expected_target = 4 if working_point == "no_quality_cut" else 3
    assert int(source.Get("simple_scifi__has_target_count").GetVal()) == expected_target
    assert int(source.Get("simple_scifi__other_passing_count").GetVal()) == 10 - expected_target
    assert int(source.Get("simple_ds__unscorable_count").GetVal()) == 1
    expected_overlap = 2 if working_point == "no_quality_cut" else 1
    assert int(source.Get("overlap__simple_scifi__hough_scifi").GetVal()) == expected_overlap
    for category in ("simple_scifi_and_ds", "hough_scifi_and_ds"):
        assert int(source.Get(f"{category}__has_target_count").GetVal()) == 1
        assert int(source.Get(f"{category}__other_passing_count").GetVal()) == 9
        components = source.Get(f"{category}__components").GetTitle().split(",")
        assert len(components) == 2
        score_expression = source.Get(
            f"{category}__quality_score_expression"
        ).GetTitle()
        assert "std::max" in score_expression
    assert int(
        source.Get("overlap__simple_scifi__simple_scifi_and_ds").GetVal()
    ) == 1
    target = source.Get("targets/simple_scifi/count_veto/has_target")
    other = source.Get("targets/simple_scifi/count_veto/other_passing")
    missing = source.Get("missing/count_veto/no_passing_muon_reco")
    assert target and other and missing
    assert int(target.Integral()) == expected_target
    assert int(other.Integral()) == 10 - expected_target
    assert int(missing.Integral()) == 2
    quality = source.Get(
        "track_quality/simple_scifi_and_ds/veto_count/has_target"
    )
    assert quality and quality.GetDimension() == 2
    assert int(quality.Integral()) == 1
    for name in ("veto_count", "us_count", "ds_count", "scifi_count", "scifi_density"):
        assert source.Get(f"track_quality/hough_scifi_and_ds/{name}/has_target")

    expected_combinations = {
        "no_quality_cut": {
            "simple": {"sf0_ds0": 3, "sf1_ds0": 3, "sf0_ds1": 3, "sf1_ds1": 1},
            # Event 6 has only a type-15 Hough combined track, so it belongs
            # to sf0_ds0 for the separate type-11/type-13 bit comparison.
            "hough": {"sf0_ds0": 6, "sf1_ds0": 2, "sf0_ds1": 1, "sf1_ds1": 1},
        },
        "chi2ndf_lt_10": {
            "simple": {"sf0_ds0": 6, "sf1_ds0": 2, "sf0_ds1": 1, "sf1_ds1": 1},
            "hough": {"sf0_ds0": 9, "sf1_ds0": 0, "sf0_ds1": 0, "sf1_ds1": 1},
        },
    }[working_point]
    feature_names = list(
        yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"]
    )
    assert len(feature_names) == 27
    for feature_name in (
        "veto_count_vs_us_count",
        "veto_count_vs_ds_count",
        "veto_time_vs_us_count",
        "veto_time_vs_ds_count",
        "us_count_vs_ds_count",
    ):
        assert feature_name in feature_names
    for algorithm, expected in expected_combinations.items():
        observed = {}
        for combination, expected_count in expected.items():
            count = int(
                source.Get(
                    f"algorithm__{algorithm}__{combination}__count"
                ).GetVal()
            )
            observed[combination] = count
            assert count == expected_count
            for feature_name in feature_names:
                histogram = source.Get(
                    "algorithm_comparisons/"
                    f"{algorithm}/{combination}/{feature_name}/{combination}"
                )
                assert histogram
        assert sum(observed.values()) == 10
        assert bool(source.Get(f"algorithm__{algorithm}__four_way_closure").GetVal())

    # Event 8 is removed from timing-dependent features, while event 10 has
    # no veto hit and is retained at the dedicated -1 ns coordinate.
    expected_timed_ds_only = 2 if working_point == "no_quality_cut" else 1
    assert int(
        source.Get(
            "algorithm__simple__sf0_ds1__veto_time_vs_ds_count__selected"
        ).GetVal()
    ) == expected_timed_ds_only
    timed = source.Get(
        "algorithm_comparisons/simple/sf0_ds1/"
        "veto_time_vs_ds_count/sf0_ds1"
    )
    no_veto_x_bin = timed.GetXaxis().FindBin(-1.0)
    assert timed.Integral(no_veto_x_bin, no_veto_x_bin, 1, timed.GetNbinsY()) >= 1

    # Event 10 is above 30k and must be folded into the final SciFi-QDC bin.
    qdc = source.Get(
        "algorithm_comparisons/simple/sf0_ds1/qdc_scifi/sf0_ds1"
    )
    assert qdc.GetXaxis().GetXmax() == 30000.0
    assert qdc.GetBinContent(qdc.GetNbinsX()) >= 1
    for feature_name in feature_names:
        mc = source.Get(f"references/{feature_name}/mc_muon_up")
        real = source.Get(f"references/{feature_name}/real_data_dq")
        passing_reference = source.Get(
            f"references/{feature_name}/passing_muon_reco"
        )
        missing_reference = source.Get(
            f"references/{feature_name}/no_passing_muon_reco"
        )
        assert mc and real and passing_reference and missing_reference
        closure = passing_reference.Clone(f"closure_{feature_name}")
        closure.Add(missing_reference)
        closure.Add(real, -1.0)
        assert all(
            closure.GetBinContent(index) == 0
            for index in range(closure.GetNcells())
        )
    for feature_name in (
        "scifi_x_vs_y",
        "qdc_weighted_scifi_x_vs_y",
    ):
        spatial = source.Get(f"references/{feature_name}/mc_muon_up")
        assert spatial.GetDimension() == 2
        assert spatial.Integral() == 5
    source.Close()


def run_workflow_test(workdir):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    real_path = workdir / "real.root"
    passing_path = workdir / "passing.root"
    mc_path = workdir / "mc_muon_up.root"
    missing_path = workdir / "missing.root"
    audit_path = workdir / "audit.csv"
    partition_metadata = workdir / "partition_metadata.csv"
    missing_metadata = workdir / "missing_metadata.csv"
    validation = workdir / "validation.csv"
    mc_source_metadata = workdir / "mc_source_metadata.csv"
    mc_metadata = workdir / "mc_metadata.csv"
    mc_validation = workdir / "mc_validation.csv"
    test_config = workdir / "study.yaml"
    write_tree(ROOT, real_path, range(1, 13), with_tracks=False)
    write_tree(ROOT, passing_path, range(1, 11), with_tracks=True)
    write_mc_tree(ROOT, mc_path)
    write_mc_source_metadata(mc_source_metadata, mc_path)
    write_partition_metadata(partition_metadata, real_path, passing_path)
    write_test_config(test_config)

    run([
        sys.executable, str(MC_PREPARATOR),
        "--source-metadata", str(mc_source_metadata),
        "--study-config", str(test_config),
        "--hist-config", str(HIST_CONFIG),
        "--output-metadata", str(mc_metadata),
        "--output-validation", str(mc_validation),
    ])
    prepared_mc = pd.read_csv(mc_metadata)
    assert len(prepared_mc) == 1
    assert prepared_mc.iloc[0]["particle_group"] == "mc_muon_up"
    validated_mc = pd.read_csv(mc_validation)
    assert int(validated_mc.iloc[0]["tree_events"]) == 8
    assert int(validated_mc.iloc[0]["inclusive_muon_events"]) == 7
    assert int(validated_mc.iloc[0]["detector_activity_reference_events"]) == 6
    assert int(validated_mc.iloc[0]["reco_track_branches_required"]) == 0

    run([
        sys.executable, str(ANTI_JOIN),
        "--real-partition", str(real_path),
        "--passing-partition", str(passing_path),
        "--part-index", "1", "--run", "8329",
        "--output-root", str(missing_path),
        "--output-audit", str(audit_path),
    ])
    run([
        sys.executable, str(VALIDATOR),
        "--audits", str(audit_path),
        "--partition-metadata", str(partition_metadata),
        "--config", str(test_config),
        "--output-metadata", str(missing_metadata),
        "--output-validation", str(validation),
    ])
    assert_missing(ROOT, missing_path, validation)

    test_study_config = yaml.safe_load(test_config.read_text(encoding="utf-8"))
    target_expected_pages = (
        1
        + len(yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"])
        + len(test_study_config["track_quality_2d"]["observables"])
    )
    algorithm_expected_pages = (
        1 + len(yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"])
    )
    for working_point in ("no_quality_cut", "chi2ndf_lt_10"):
        histogram = workdir / f"{working_point}.root"
        run([
            sys.executable, str(HISTOGRAMMER),
            "--partition-metadata", str(partition_metadata),
            "--missing-metadata", str(missing_metadata),
            "--mc-muon-up-metadata", str(mc_metadata),
            "--hist-config", str(HIST_CONFIG),
            "--study-config", str(test_config),
            "--working-point", working_point,
            "--output", str(histogram), "--threads", "1",
        ])
        assert_histograms(ROOT, histogram, working_point)
        for comparison in (
            "simple_scifi",
            "simple_scifi_and_ds",
            "hough_scifi_and_ds",
            "all_track_types",
            "simple_track_combinations",
            "hough_track_combinations",
        ):
            output_pdf = workdir / working_point / f"{comparison}.pdf"
            run([
                sys.executable, str(PLOTTER),
                "--input", str(histogram),
                "--hist-config", str(HIST_CONFIG),
                "--study-config", str(test_config),
                "--comparison", comparison,
                "--output-pdf", str(output_pdf),
            ])
            assert output_pdf.is_file() and output_pdf.stat().st_size > 0
            expected_pages = (
                algorithm_expected_pages
                if comparison.endswith("_track_combinations")
                else target_expected_pages
            )
            assert pdf_pages(output_pdf) == expected_pages

        if working_point == "no_quality_cut":
            output_pdf = workdir / working_point / "mc_muon_up_reference.pdf"
            run([
                sys.executable, str(PLOTTER),
                "--input", str(histogram),
                "--hist-config", str(HIST_CONFIG),
                "--study-config", str(test_config),
                "--comparison", "mc_muon_up_reference",
                "--output-pdf", str(output_pdf),
            ])
            assert output_pdf.is_file() and output_pdf.stat().st_size > 0
            assert pdf_pages(output_pdf) == 1 + len(
                yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.keep:
        workdir = Path(tempfile.mkdtemp(prefix="passing_muon_track_study_v3_test."))
        run_workflow_test(workdir)
        print(f"PASS; retained outputs at {workdir}")
        return
    with tempfile.TemporaryDirectory(
        prefix="passing_muon_track_study_v3_test."
    ) as value:
        run_workflow_test(Path(value))
    print("PASS")


if __name__ == "__main__":
    main()
