#!/usr/bin/env python3
"""Synthetic end-to-end test for detector-only run-8329 study v5."""

from __future__ import annotations

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
REGION_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v5.yaml"
)
HISTOGRAMMER = REPO_ROOT / (
    "evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py"
)
PLOTTER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_binary_region_study_v5.py"
)
PREPARER = REPO_ROOT / (
    "snakemake/metadata/prepare_mc_muon_up_reference_v5.py"
)
IDENTITY_CUTFLOW = REPO_ROOT / "convertData/make_identity_cutflow.py"


INTEGER_BRANCHES = [
    "source_file_index", "runId", "eventId", "eventIndex", "pdgCode",
    "count_veto", "count_scifi", "count_us", "count_ds",
    *(f"count_veto{i}" for i in range(1, 4)),
    *(f"count_scifi{i}" for i in range(1, 6)),
    *(f"count_us{i}" for i in range(1, 6)),
    *(f"count_ds{i}" for i in range(1, 5)),
    "scifi_n_active_stations_xy",
]
FLOAT_BRANCHES = [
    "qdc_veto", "hitTime_veto_min_ns", "qdc_scifi",
    "density_sndsw_scifi", "avg_scifi_x", "avg_scifi_y",
    "qdcAvg_scifi_x", "qdcAvg_scifi_y", "qdc_us", "qdc_ds",
]


# Events 1, 2, 7, 8, 9, and 10 pass loose coverage; only event 1 passes
# tight low-multiplicity coverage. Events 3 and 4 demonstrate why total counts
# cannot replace station-level coverage. Events 8, 9, and 10 exercise the
# exclusive US, SciFi, and Veto per-station multiplicity boundaries.
EVENTS = {
    1: ([1, 1, 1], [2, 2, 2, 2, 2], 5, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
    2: ([2, 1, 1], [3, 3, 3, 3, 0], 4, [1, 1, 1, 1, 1], [1, 1, 1, 0]),
    3: ([3, 0, 0], [2, 2, 2, 2, 2], 5, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
    4: ([1, 1, 1], [2, 2, 2, 2, 2], 5, [2, 1, 1, 1, 0], [1, 1, 1, 1]),
    5: ([1, 1, 1], [2, 2, 2, 0, 0], 3, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
    6: ([1, 1, 1], [2, 2, 2, 2, 2], 5, [1, 1, 1, 1, 1], [1, 1, 0, 0]),
    7: ([4, 4, 4], [40, 40, 40, 40, 40], 5, [6, 6, 6, 6, 6], [5, 5, 5, 5]),
    8: ([1, 1, 1], [4, 4, 4, 4, 4], 5, [2, 1, 1, 1, 1], [2, 1, 1, 1]),
    9: ([1, 1, 1], [10, 10, 10, 10, 10], 5, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
    10: ([2, 1, 1], [2, 2, 2, 2, 2], 5, [1, 1, 1, 1, 1], [1, 1, 1, 1]),
}


def event_values(event_id):
    veto, scifi, active_scifi_xy, us, ds = EVENTS[event_id]
    value = float(event_id)
    result = {
        "source_file_index": 1,
        "runId": 8329,
        "eventId": event_id,
        "eventIndex": event_id - 1,
        "pdgCode": 13,
        "count_veto": sum(veto),
        "count_scifi": sum(scifi),
        "count_us": sum(us),
        "count_ds": sum(ds),
        "scifi_n_active_stations_xy": active_scifi_xy,
        "qdc_veto": 10.0 * value,
        "hitTime_veto_min_ns": value,
        "qdc_scifi": 100.0 * value,
        "density_sndsw_scifi": 25.0 * value,
        "avg_scifi_x": -30.0 + value,
        "avg_scifi_y": 15.0 + value,
        "qdcAvg_scifi_x": -29.5 + value,
        "qdcAvg_scifi_y": 15.5 + value,
        "qdc_us": 20.0 * value,
        "qdc_ds": 30.0 * value,
    }
    for prefix, values in (("veto", veto), ("scifi", scifi), ("us", us), ("ds", ds)):
        for index, count in enumerate(values, start=1):
            result[f"count_{prefix}{index}"] = count
    return result


def write_tree(ROOT, path, zero_scifi=False):
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic detector-only v5 study")
    storage = {}
    for name in INTEGER_BRANCHES:
        storage[name] = array("q", [0])
        tree.Branch(name, storage[name], f"{name}/L")
    for name in FLOAT_BRANCHES:
        storage[name] = array("d", [0.0])
        tree.Branch(name, storage[name], f"{name}/D")
    for event_id in EVENTS:
        values = event_values(event_id)
        if zero_scifi:
            for station in range(1, 6):
                values[f"count_scifi{station}"] = 0
            values.update({
                "count_scifi": 0,
                "scifi_n_active_stations_xy": 0,
                "qdc_scifi": 0.0,
                "density_sndsw_scifi": 0.0,
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


def write_input_event_tree(ROOT, path, entries):
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("cbmsim", "synthetic MC input")
    value = array("q", [0])
    tree.Branch("event", value, "event/L")
    for index in range(entries):
        value[0] = index
        tree.Fill()
    tree.Write()
    root_file.Close()


def standardized_row(root_path, particle_group, partition_id, production_key=""):
    return {
        "region": "all_events",
        "particle_group": particle_group,
        "partition_id": partition_id,
        "feature_partition_path": str(root_path),
        "feature_available": True,
        "n_events": len(EVENTS),
        "lumi_per_partition": 1.0,
        "source_file_indices": "1",
        "source_file_count": 1,
        "production_key": production_key,
    }


def write_metadata(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def write_source_metadata(path, roots):
    rows = []
    for index, (partition, root_path) in enumerate(roots.items()):
        rows.append({
            "data_type": "MC_muon",
            "subfolder": "up",
            "partition": partition,
            "n_event": len(EVENTS),
            "geo_path": (
                "/synthetic/scoring_2.5/geofile.root"
                if partition == "scoring_2" else f"/synthetic/{partition}/geofile.root"
            ),
            "output_base_path": str(root_path.parent),
            "feature_path": root_path.name,
            "lumi_per_file": 1.0 + index,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parameter(source, name):
    value = source.Get(name)
    assert value, name
    return value.GetVal() if hasattr(value, "GetVal") else value.GetTitle()


def pdf_pages(path):
    if not shutil.which("pdfinfo"):
        raise RuntimeError("pdfinfo is required for PDF-page assertions")
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return int(
        next(line for line in output.splitlines() if line.startswith("Pages:"))
        .split(":", 1)[1]
    )


def write_test_config(path):
    config = yaml.safe_load(REGION_CONFIG.read_text(encoding="utf-8"))
    config["expected"] = {"real_data_dq": len(EVENTS)}
    config["mc_muon_up"]["expected_source_files"] = 2
    config["mc_muon_up"]["productions"] = {
        "scoring_1p8": config["mc_muon_up"]["productions"]["scoring_1p8"],
        "scoring_at449m": config["mc_muon_up"]["productions"]["scoring_at449m"],
    }
    # Exercise the future config-only update path and N-minus-one machinery.
    config["selection_working_points"]["nominal"].update({
        "enabled": True,
        "veto_max": None,
        "scifi_max": 20,
        "us_max": 6,
        "ds_max": 5,
    })
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def merged_feature_names(config):
    common_features = yaml.safe_load(HIST_CONFIG.read_text(encoding="utf-8"))["features"]
    return [*common_features, *config["diagnostic_features"]]


def assert_output(ROOT, path, config):
    source = ROOT.TFile.Open(str(path))
    assert source and not source.IsZombie()
    assert int(parameter(source, "real_data_dq_count")) == 10
    assert int(parameter(source, "uses_reconstructed_tracks")) == 0
    assert int(parameter(source, "uses_mc_muon_up")) == 1
    assert int(parameter(source, "mc_muon_up_baseline_count")) == 20
    assert str(parameter(source, "mc_muon_up_production_order")) == (
        "scoring_1p8,scoring_at449m"
    )
    assert int(parameter(source, "mc_muon_up__scoring_1p8__baseline_count")) == 10
    assert int(parameter(source, "mc_muon_up__scoring_at449m__baseline_count")) == 10
    assert str(parameter(source, "selection_order")) == (
        "coverage_loose,coverage_tight,nominal"
    )
    expected = {
        "coverage_loose": (6, 4),
        "coverage_tight": (1, 9),
        "nominal": (1, 9),
    }
    features = merged_feature_names(config)
    for selection, (candidate, complement) in expected.items():
        prefix = f"selection__{selection}"
        assert int(parameter(source, f"{prefix}__candidate_count")) == candidate
        assert int(parameter(source, f"{prefix}__complement_count")) == complement
        assert int(parameter(source, f"{prefix}__population_closure")) == 1
        assert int(parameter(source, f"{prefix}__mc_candidate_count")) == candidate
        assert int(parameter(source, f"{prefix}__mc_complement_count")) == complement + 10
        assert int(parameter(source, f"{prefix}__mc_population_closure")) == 1
        assert int(parameter(
            source, f"{prefix}__mc__scoring_1p8__candidate_count"
        )) == candidate
        assert int(parameter(
            source, f"{prefix}__mc__scoring_at449m__candidate_count"
        )) == 0
        for feature in features:
            base = f"selections/{selection}/features/{feature}"
            assert source.Get(f"{base}/through_going_candidate")
            assert source.Get(f"{base}/shower_enriched_complement")
            assert source.Get(
                f"mc_muon_up/scoring_1p8/selections/{selection}/features/"
                f"{feature}/candidate"
            )
            assert source.Get(
                f"mc_muon_up/scoring_at449m/selections/{selection}/features/"
                f"{feature}/candidate"
            )
            combined_candidate = source.Get(
                f"mc_muon_up/all/selections/{selection}/features/"
                f"{feature}/candidate"
            )
            combined_complement = source.Get(
                f"mc_muon_up/all/selections/{selection}/features/"
                f"{feature}/complement"
            )
            assert combined_candidate and combined_complement
            component_candidates = [
                source.Get(
                    f"mc_muon_up/{production}/selections/{selection}/features/"
                    f"{feature}/candidate"
                )
                for production in ("scoring_1p8", "scoring_at449m")
            ]
            component_complements = [
                source.Get(
                    f"mc_muon_up/{production}/selections/{selection}/features/"
                    f"{feature}/complement"
                )
                for production in ("scoring_1p8", "scoring_at449m")
            ]
            assert combined_candidate.Integral() == sum(
                histogram.Integral() for histogram in component_candidates
            )
            assert combined_complement.Integral() == sum(
                histogram.Integral() for histogram in component_complements
            )

    # Exact topology boundaries and deceptive equal-total station patterns.
    assert int(parameter(source, "selection__coverage_loose__candidate_count")) == 6
    assert int(parameter(source, "selection__coverage_tight__candidate_count")) == 1
    tight_order = str(parameter(source, "selection__coverage_tight__cut_order"))
    for component in (
        "scifi_station_multiplicity",
        "veto_station_multiplicity",
        "us_station_multiplicity",
    ):
        assert component in tight_order
    assert "veto_max" not in str(parameter(source, "selection__nominal__cut_order"))
    assert int(parameter(source, "selection__nominal__candidate_count")) == 1

    # First scan values: three loose candidates have veto <=3, four have
    # SciFi <=20, four have US <=5, and none has DS <=1.
    expected_first_bins = {
        "veto_max": 3,
        "scifi_max": 4,
        "us_max": 4,
        "ds_max": 0,
    }
    for scan, expected_count in expected_first_bins.items():
        histogram = source.Get(f"diagnostics/threshold_scans/{scan}/counts")
        assert histogram
        assert int(histogram.GetBinContent(1)) == expected_count
        values = [histogram.GetBinContent(i) for i in range(1, histogram.GetNbinsX() + 1)]
        assert values == sorted(values)

    for field in ("scifi_max", "us_max", "ds_max"):
        base = f"diagnostics/nminus1/nominal/{field}"
        assert source.Get(f"{base}/passes_requirement")
        assert source.Get(f"{base}/fails_requirement")

    spatial = source.Get(
        "selections/coverage_loose/features/scifi_x_vs_y/through_going_candidate"
    )
    assert spatial and spatial.GetDimension() == 2
    assert spatial.Integral() == 6
    source.Close()


def run_workflow_test(workdir):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    identity_input = workdir / "identity_input.root"
    identity_output = workdir / "identity_cutflow.root"
    write_input_event_tree(ROOT, identity_input, 7)
    run([
        sys.executable, str(IDENTITY_CUTFLOW),
        "--input", str(identity_input),
        "--input-tree", "cbmsim",
        "--output", str(identity_output),
    ])
    identity_file = ROOT.TFile.Open(str(identity_output))
    identity_tree = identity_file.Get("cutFlowSummary")
    assert identity_tree and identity_tree.GetEntries() == 7
    identity_tree.GetEntry(6)
    assert int(identity_tree.entry) == 6
    identity_file.Close()
    real_root = workdir / "real_data.root"
    mc_scifi_root = workdir / "mc_scoring_1p8.root"
    mc_empty_scifi_root = workdir / "mc_scoring_at449m.root"
    metadata = workdir / "real_data.csv"
    source_metadata = workdir / "mc_muon_up_source.csv"
    mc_metadata = workdir / "mc_muon_up.csv"
    mc_validation = workdir / "mc_muon_up_validation.csv"
    config_path = workdir / "v5.yaml"
    output = workdir / "v5.root"
    write_tree(ROOT, real_root)
    write_tree(ROOT, mc_scifi_root)
    write_tree(ROOT, mc_empty_scifi_root, zero_scifi=True)
    write_metadata(metadata, [standardized_row(
        real_root, "real_data_dq_run8329",
        "all_events__real_data_dq_run8329__part001",
    )])
    write_source_metadata(source_metadata, {
        "scoring_1.8": mc_scifi_root,
        "scoring_at449m": mc_empty_scifi_root,
    })
    write_test_config(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run([
        sys.executable, str(PREPARER),
        "--source-metadata", str(source_metadata),
        "--study-config", str(config_path),
        "--hist-config", str(HIST_CONFIG),
        "--output-metadata", str(mc_metadata),
        "--output-validation", str(mc_validation),
        "--threads", "1",
    ])
    prepared = pd.read_csv(mc_metadata)
    validation = pd.read_csv(mc_validation)
    assert len(prepared) == 2
    assert set(prepared["production_key"]) == {"scoring_1p8", "scoring_at449m"}
    assert validation["event_count_match"].all()
    assert dict(zip(
        validation["production_key"], validation["scifi_positive_events"]
    )) == {"scoring_1p8": 10, "scoring_at449m": 0}
    run([
        sys.executable, str(HISTOGRAMMER),
        "--partition-metadata", str(metadata),
        "--mc-muon-up-metadata", str(mc_metadata),
        "--hist-config", str(HIST_CONFIG),
        "--region-config", str(config_path),
        "--output", str(output),
        "--threads", "1",
    ])
    assert_output(ROOT, output, config)

    features = merged_feature_names(config)
    comparisons = [
        "diagnostics", "mc_muon_up_summary",
        "coverage_loose", "coverage_tight", "nominal",
        "coverage_loose__mc__scoring_1p8",
        "coverage_loose__mc__scoring_at449m",
        "coverage_tight__mc__all",
    ]
    for comparison in comparisons:
        pdf = workdir / f"{comparison}.pdf"
        run([
            sys.executable, str(PLOTTER),
            "--input", str(output),
            "--hist-config", str(HIST_CONFIG),
            "--region-config", str(config_path),
            "--comparison", comparison,
            "--output-pdf", str(pdf),
        ])
        if comparison == "diagnostics":
            expected_pages = 1 + 3 + len(config["multiplicity_scans"])
        elif comparison == "mc_muon_up_summary":
            expected_pages = 1 + 3
        else:
            nminus_pages = 3 if comparison == "nominal" else 0
            expected_pages = 1 + len(features) + nminus_pages
        assert pdf_pages(pdf) == expected_pages


def main():
    with tempfile.TemporaryDirectory(prefix="passing_muon_v5_test_") as directory:
        run_workflow_test(Path(directory))
    print("passing_muon detector-only study v5 synthetic test: OK")


if __name__ == "__main__":
    main()
