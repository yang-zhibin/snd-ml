#!/usr/bin/env python3
"""Synthetic end-to-end test for the DS--SciFi rejection workflow."""

from __future__ import annotations

import copy
import csv
import subprocess
import tempfile
from array import array
from pathlib import Path

import ROOT
import yaml

from make_abcd_ds_scifi_rejection_report import cutflow_table_body


HERE = Path(__file__).resolve().parent


def feature_file(path, repetitions=1):
    target = ROOT.TFile.Open(str(path), "RECREATE"); tree = ROOT.TTree("sndData", "")
    names = ["scifi_n_active_stations_xy", "count_scifi", *(f"count_scifi{i}" for i in range(1, 6)), "count_ds", *(f"count_ds{i}" for i in range(1, 5)), "region_code"]
    values = {name: array("i", [0]) for name in names}; density = array("d", [0])
    for name, value in values.items(): tree.Branch(name, value, f"{name}/I")
    tree.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")
    patterns = [(True, 50.0), (False, 7000.0), (True, 12000.0), (False, 45000.0)]
    for _ in range(repetitions):
        for tagged, x in patterns:
            values["scifi_n_active_stations_xy"][0] = 5; values["count_scifi"][0] = 40; values["region_code"][0] = 0
            for station in range(1, 6): values[f"count_scifi{station}"][0] = 8
            for station in range(1, 5): values[f"count_ds{station}"][0] = 1 if tagged else (3 if station == 1 else 1)
            values["count_ds"][0] = 4 if tagged else 6; density[0] = x; tree.Fill()
    tree.Write(); target.Close()


def run(*args):
    subprocess.run(["python3", *map(str, args)], check=True, cwd=HERE.parent)


def main():
    ROOT.gROOT.SetBatch(True)
    probe = cutflow_table_body(
        [{"label": "Base"}, {"label": "A"}, {"label": "B"}],
        [100, 80, 20],
    )
    assert probe[0][4:] == ["--", "100%"]
    assert probe[1][3:] == ["20", "80%", "80%"]
    assert probe[2][3:] == ["60", "25%", "20%"]
    with tempfile.TemporaryDirectory(prefix="abcd_ds_scifi_test_") as directory:
        work = Path(directory); data_a = work / "data_a.root"; data_b = work / "data_b.root"; mc = work / "mc.root"
        feature_file(data_a); feature_file(data_b); feature_file(mc, repetitions=2)
        metadata = work / "metadata.csv"
        with open(metadata, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["output_base_path", "feature_path", "lumi_per_file"]); writer.writeheader()
            writer.writerow({"output_base_path": work, "feature_path": data_a.name, "lumi_per_file": 1.0}); writer.writerow({"output_base_path": work, "feature_path": data_b.name, "lumi_per_file": 1.0})
        region_name = "signal_no_veto_has_us_fiducial_inside"
        region_config = work / "regions.yaml"; region_config.write_text(yaml.safe_dump({"regions": {region_name: {"expression": "region_code == 0"}}, "expression_macros": {}}), encoding="utf-8")
        config = {
            "version": "synthetic_v1", "input": {"tree_name": "sndData", "density_feature": "density_sndsw_scifi", "partition_version": "partition_v1", "region_version": "region_v1", "data_particle_group": "real_data", "partition_catalogue": str(work / "catalogue.csv"), "data_shards": 2, "skip_missing_source_files": False, "expected_full_data_luminosity": 2.0},
            "selection": {"base_expression": "count_scifi > 35", "scifi": {"active_xy_stations_equal": 5, "max_hits_per_station_exclusive": 20}, "ds": {"require_all_stations": True, "max_hits_per_station_inclusive": 2, "max_hits_total_inclusive": 5}},
            "histograms": {"analysis": {"bins": 80, "min": 0, "max": 40000, "underflow": "preserve", "overflow": "fold_last"}, "original_plot": {"bins": 80, "min": 100, "max": 40000, "underflow": "preserve", "overflow": "preserve"}},
            "statistics": {"confidence_level": 0.682689492137}, "regions": {region_name: "Signal"},
            "mc_samples": {"CC_nue": {"role": "signal", "source_luminosity": 4.0}}, "region_exposures": {region_name: 2.0},
            "validation": {"luminosity_absolute_tolerance": 1e-9},
        }
        config_path = work / "config.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        roots, summaries = [], []
        for shard in range(2):
            root, summary = work / f"shard_{shard}.root", work / f"shard_{shard}.csv"; roots.append(root); summaries.append(summary)
            run(HERE / "make_abcd_ds_scifi_rejection_data_shard.py", "--metadata", metadata, "--config", config_path, "--region-config", region_config, "--shard-index", shard, "--shard-count", 2, "--output-root", root, "--output-summary", summary)
        merged_root, merged_csv = work / "data.root", work / "data.csv"
        command = [HERE / "merge_abcd_ds_scifi_rejection_data.py", "--config", config_path]
        for root in roots: command += ["--input-root", root]
        for summary in summaries: command += ["--input-summary", summary]
        command += ["--output-root", merged_root, "--output-summary", merged_csv]; run(*command)
        with open(merged_csv, encoding="utf-8") as stream: data_rows = list(csv.DictReader(stream))
        assert [(row["population"], int(row["n_before"]), int(row["n_tagged"]), int(row["n_retained"])) for row in data_rows] == [("total", 8, 4, 4), (region_name, 8, 4, 4)]
        cutflow_fields = [
            "cutflow_base", "cutflow_scifi_xy",
            "cutflow_scifi_station_hits", "cutflow_ds_all_active",
            "cutflow_ds_station_hits", "cutflow_ds_total_hits",
        ]
        for row in data_rows:
            assert [int(row[field]) for field in cutflow_fields] == [8, 8, 8, 8, 4, 4]
        catalogue = work / "catalogue.csv"
        with open(catalogue, "w", encoding="utf-8", newline="") as stream:
            fields = ["partition_version", "region_version", "region", "particle_group", "partition_id", "feature_available", "feature_partition_path", "n_events", "lumi_per_partition"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            common = {"partition_version": "partition_v1", "region_version": "region_v1", "region": region_name, "feature_available": True}
            writer.writerow(common | {"particle_group": "CC_nue", "partition_id": "mc_0", "feature_partition_path": mc, "n_events": 8, "lumi_per_partition": 4.0})
            writer.writerow(common | {"particle_group": "real_data", "partition_id": "data_0", "feature_partition_path": data_a, "n_events": 4, "lumi_per_partition": 1.0})
            writer.writerow(common | {"particle_group": "real_data", "partition_id": "data_1", "feature_partition_path": data_b, "n_events": 4, "lumi_per_partition": 1.0})
        mc_root, mc_csv = work / "mc_out.root", work / "mc_out.csv"
        run(HERE / "make_abcd_ds_scifi_rejection_mc.py", "--partition-catalogue", catalogue, "--config", config_path, "--region", region_name, "--sample", "CC_nue", "--output-root", mc_root, "--output-summary", mc_csv)
        with open(mc_csv, encoding="utf-8") as stream: mc_row = next(csv.DictReader(stream))
        assert (int(mc_row["n_before"]), int(mc_row["n_retained"]), float(mc_row["sumw_retained"])) == (8, 4, 2.0)
        assert [int(mc_row[field]) for field in cutflow_fields] == [8, 8, 8, 8, 4, 4]
        report_csv, report_root, report_pdf, report_md = work / "all.csv", work / "plots.root", work / "report.pdf", work / "report.md"
        run(HERE / "make_abcd_ds_scifi_rejection_report.py", "--config", config_path, "--data-root", merged_root, "--data-summary", merged_csv, "--mc-root", mc_root, "--mc-summary", mc_csv, "--output-summary", report_csv, "--output-root", report_root, "--output-pdf", report_pdf, "--output-markdown", report_md)
        assert all(path.exists() and path.stat().st_size > 0 for path in (report_csv, report_root, report_pdf, report_md))
        report_file = ROOT.TFile.Open(str(report_root), "READ")
        assert report_file and not report_file.IsZombie()
        assert report_file.Get("cutflow_data") and report_file.Get("cutflow_signal_mc")
        report_file.Close()
        pdf_info = subprocess.run(
            ["pdfinfo", str(report_pdf)], check=True, text=True, capture_output=True
        ).stdout
        assert any(
            line.startswith("Pages:") and line.split()[-1] == "6"
            for line in pdf_info.splitlines()
        )
        report_text = report_md.read_text(encoding="utf-8")
        assert "Full-2024 data sequential cutflow" in report_text
        assert "CC_nue signal-MC sequential cutflow" in report_text
        assert "Removal-tag step efficiency" in report_text
        pdf_text = subprocess.run(
            ["pdftotext", str(report_pdf), "-"], check=True, text=True,
            capture_output=True,
        ).stdout
        assert "Neutrino-MC acceptance after rejection" not in pdf_text
        assert "Clopper" not in pdf_text
        assert "Keep efficiency" not in pdf_text
        assert "before, tagged, and retained" not in pdf_text
        assert "Retained Full-2024 data by category" in pdf_text
        assert "Removed Full-2024 data by category" in pdf_text

        loose_config = copy.deepcopy(config)
        loose_config["version"] = "synthetic_loose_v1"
        loose_config["input"]["data_shards"] = 1
        loose_config["selection"]["ds"] = {"require_all_stations": True}
        loose_config_path = work / "loose_config.yaml"
        loose_config_path.write_text(
            yaml.safe_dump(loose_config, sort_keys=False), encoding="utf-8"
        )
        loose_region_root = work / "loose_region.root"
        loose_region_csv = work / "loose_region.csv"
        run(
            HERE / "make_abcd_ds_scifi_rejection_mc.py",
            "--partition-catalogue", catalogue, "--config", loose_config_path,
            "--region", region_name, "--sample", "real_data",
            "--output-root", loose_region_root,
            "--output-summary", loose_region_csv,
        )
        loose_data_root = work / "loose_data.root"
        loose_data_csv = work / "loose_data.csv"
        run(
            HERE / "merge_abcd_ds_scifi_rejection_data_partitions.py",
            "--config", loose_config_path, "--input-root", loose_region_root,
            "--input-summary", loose_region_csv, "--output-root", loose_data_root,
            "--output-summary", loose_data_csv,
        )
        with loose_data_csv.open(encoding="utf-8") as stream:
            loose_data_rows = list(csv.DictReader(stream))
        assert [row["population"] for row in loose_data_rows] == ["total", region_name]
        loose_fields = [
            "cutflow_base", "cutflow_scifi_xy",
            "cutflow_scifi_station_hits", "cutflow_ds_all_active",
        ]
        for row in loose_data_rows:
            assert [int(row[field]) for field in loose_fields] == [8, 8, 8, 8]
            assert int(row["n_tagged"]) == 8 and int(row["n_retained"]) == 0
            assert "cutflow_ds_station_hits" not in row
            assert "cutflow_ds_total_hits" not in row
        assert loose_data_rows[0]["exposure_fb"] == "mixed"
        assert float(loose_data_rows[1]["exposure_fb"]) == 2.0
        loose_mc_root = work / "loose_mc.root"
        loose_mc_csv = work / "loose_mc.csv"
        run(
            HERE / "make_abcd_ds_scifi_rejection_mc.py",
            "--partition-catalogue", catalogue, "--config", loose_config_path,
            "--region", region_name, "--sample", "CC_nue",
            "--output-root", loose_mc_root, "--output-summary", loose_mc_csv,
        )
        with loose_mc_csv.open(encoding="utf-8") as stream:
            loose_mc_row = next(csv.DictReader(stream))
        assert [int(loose_mc_row[field]) for field in loose_fields] == [8, 8, 8, 8]
        assert int(loose_mc_row["n_tagged"]) == 8
        loose_report_csv = work / "loose_all.csv"
        loose_report_root = work / "loose_plots.root"
        loose_report_pdf = work / "loose_report.pdf"
        loose_report_md = work / "loose_report.md"
        run(
            HERE / "make_abcd_ds_scifi_rejection_report.py",
            "--config", loose_config_path, "--data-root", loose_data_root,
            "--data-summary", loose_data_csv, "--mc-root", loose_mc_root,
            "--mc-summary", loose_mc_csv, "--output-summary", loose_report_csv,
            "--output-root", loose_report_root, "--output-pdf", loose_report_pdf,
            "--output-markdown", loose_report_md,
        )
        loose_text = loose_report_md.read_text(encoding="utf-8")
        assert "count_ds <=" not in loose_text
        assert all(f"count_ds{station} <=" not in loose_text for station in range(1, 5))
        loose_pdf_info = subprocess.run(
            ["pdfinfo", str(loose_report_pdf)], check=True, text=True,
            capture_output=True,
        ).stdout
        assert any(
            line.startswith("Pages:") and line.split()[-1] == "6"
            for line in loose_pdf_info.splitlines()
        )
    print("DS--SciFi rejection end-to-end test: OK")


if __name__ == "__main__": main()
