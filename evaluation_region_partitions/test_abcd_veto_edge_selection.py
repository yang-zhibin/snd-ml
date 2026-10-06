#!/usr/bin/env python3
"""Synthetic end-to-end test for the Veto edge-muon selection workflow."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from array import array
from pathlib import Path

import ROOT
import yaml

from abcd_ds_scifi_selection import configured_selection

HERE = Path(__file__).resolve().parent


def run(*args):
    subprocess.run(["python3", *map(str, args)], check=True, cwd=HERE.parent)


def feature_file(path):
    target = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "")
    integer_names = [
        "count_scifi", "count_veto1", "count_veto2", "count_veto3",
    ]
    integers = {name: array("i", [0]) for name in integer_names}
    doubles = {
        name: array("d", [0.0])
        for name in (
            "density_sndsw_scifi", "avg_veto1_y",
            "avg_veto2_y", "avg_veto3_x",
        )
    }
    for name, value in integers.items():
        tree.Branch(name, value, f"{name}/I")
    for name, value in doubles.items():
        tree.Branch(name, value, f"{name}/D")

    # (v1 y or None, v2 y or None, v3 x or None, expected edge selection)
    patterns = [
        (20.0, 20.0, -40.0, True),
        (50.0, 50.0, -10.0, True),
        (20.0, 20.0, -25.0, False),
        (20.0, 50.0, -40.0, False),
        (20.0, None, -40.0, True),
        (None, 50.0, -10.0, True),
        (None, None, -40.0, True),
        (20.0, None, None, True),
        (None, None, None, False),
        (None, 30.0, None, False),
    ]
    for index, (v1, v2, v3, _) in enumerate(patterns):
        integers["count_scifi"][0] = 40
        doubles["density_sndsw_scifi"][0] = 250.0 + 500.0 * index
        for plane, coordinate, value in (
            (1, "avg_veto1_y", v1),
            (2, "avg_veto2_y", v2),
            (3, "avg_veto3_x", v3),
        ):
            integers[f"count_veto{plane}"][0] = 0 if value is None else 1
            doubles[coordinate][0] = -999.0 if value is None else value
        tree.Fill()
    tree.Write()
    target.Close()
    return patterns


def main():
    ROOT.gROOT.SetBatch(True)
    with tempfile.TemporaryDirectory(prefix="abcd_veto_edge_test_") as directory:
        work = Path(directory)
        feature = work / "features.root"
        patterns = feature_file(feature)
        region = "signal_no_veto_has_us_fiducial_inside"
        catalogue = work / "catalogue.csv"
        fields = [
            "partition_version", "region_version", "region", "particle_group",
            "partition_id", "feature_available", "feature_partition_path",
            "n_events", "lumi_per_partition",
        ]
        with catalogue.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            common = {
                "partition_version": "partition_v1",
                "region_version": "region_v1",
                "region": region,
                "feature_available": True,
                "feature_partition_path": feature,
                "n_events": len(patterns),
            }
            writer.writerow(common | {
                "particle_group": "real_data", "partition_id": "data_0",
                "lumi_per_partition": 2.0,
            })
            writer.writerow(common | {
                "particle_group": "CC_nue", "partition_id": "mc_0",
                "lumi_per_partition": 4.0,
            })

        base_config = yaml.safe_load(
            (
                HERE.parent
                / "snakemake/metadata/configs/region_evaluation/"
                "abcd_veto_edge_muon_selection_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        base_config["version"] = "synthetic_veto_edge_v1"
        base_config["input"]["partition_catalogue"] = str(catalogue)
        base_config["input"]["expected_full_data_luminosity"] = 2.0
        base_config["regions"] = {region: "Signal"}
        base_config["region_exposures"] = {region: 2.0}
        base_config["mc_samples"]["CC_nue"]["source_luminosity"] = 4.0
        config = work / "config.yaml"
        config.write_text(
            yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
        )

        selection = configured_selection(base_config)
        assert all(
            name in selection["tag_expression"]
            for name in ("count_veto1", "count_veto2", "count_veto3")
        )
        assert selection["keep_expression"].startswith("!(")

        outputs = {}
        for sample in ("real_data", "CC_nue"):
            root = work / f"{sample}.root"
            summary = work / f"{sample}.csv"
            outputs[sample] = (root, summary)
            run(
                HERE / "make_abcd_ds_scifi_rejection_mc.py",
                "--partition-catalogue", catalogue,
                "--config", config, "--region", region, "--sample", sample,
                "--output-root", root, "--output-summary", summary,
            )
            with summary.open(encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            assert (int(row["n_before"]), int(row["n_tagged"]), int(row["n_retained"])) == (10, 6, 4)
            assert (
                int(row["n_selected_edge"]), int(row["n_nonedge"]),
                float(row["edge_efficiency"]),
            ) == (6, 4, 0.6)
            assert [
                int(row[f"category_{key}"])
                for key in ("both_orientations", "y_only", "x_only", "no_position")
            ] == [6, 2, 1, 1]

        data_root = work / "data_full.root"
        data_csv = work / "data_full.csv"
        run(
            HERE / "merge_abcd_ds_scifi_rejection_data_partitions.py",
            "--config", config,
            "--input-root", outputs["real_data"][0],
            "--input-summary", outputs["real_data"][1],
            "--output-root", data_root, "--output-summary", data_csv,
        )
        with data_csv.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["population"] for row in rows] == ["total", region]
        assert int(rows[0]["n_tagged"]) == 6
        assert int(rows[0]["category_no_position"]) == 1

        report_csv = work / "efficiencies.csv"
        report_root = work / "plots.root"
        report_pdf = work / "report.pdf"
        report_md = work / "report.md"
        run(
            HERE / "make_abcd_veto_edge_report.py",
            "--config", config,
            "--data-root", data_root, "--data-summary", data_csv,
            "--mc-root", outputs["CC_nue"][0],
            "--mc-summary", outputs["CC_nue"][1],
            "--output-summary", report_csv, "--output-root", report_root,
            "--output-pdf", report_pdf, "--output-markdown", report_md,
        )
        assert all(
            path.exists() and path.stat().st_size > 0
            for path in (report_csv, report_root, report_pdf, report_md)
        )
        markdown = report_md.read_text(encoding="utf-8")
        assert "No confidence interval is reported" in markdown
        assert "Clopper" not in markdown
        assert "selected/kept population is edge_muon" in markdown
        pdf_text = subprocess.run(
            ["pdftotext", str(report_pdf), "-"],
            check=True, text=True, capture_output=True,
        ).stdout
        assert "Clopper" not in pdf_text
        assert "Selected Veto edge-muon data by category" in pdf_text
        assert "Complementary non-edge data by category" in pdf_text
        pdf_info = subprocess.run(
            ["pdfinfo", str(report_pdf)],
            check=True, text=True, capture_output=True,
        ).stdout
        assert any(
            line.startswith("Pages:") and line.split()[-1] == "8"
            for line in pdf_info.splitlines()
        )
    print("Veto edge-muon selection end-to-end test: OK")


if __name__ == "__main__":
    main()
