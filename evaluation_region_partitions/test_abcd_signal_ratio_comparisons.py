#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import ROOT
import yaml

from plot_abcd_signal_ratio_comparisons import create_signal_ratio_comparisons


ROOT.gROOT.SetBatch(True)


def write_hist(path: Path, counts: list[float], lumi: float) -> None:
    output = ROOT.TFile(str(path), "RECREATE")
    hist = ROOT.TH1D("hist", "", len(counts), 100.0, 40000.0)
    for index, value in enumerate(counts, 1):
        hist.SetBinContent(index, value)
    hist.Write(); ROOT.TParameter("double")("lumi", lumi).Write(); output.Close()


def test_seven_page_report() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_signal_ratio_test_") as directory:
        workdir = Path(directory)
        roles = ("signal", "sb1", "sb2", "sb3", "sb4", "sb5", "sb6", "sb7")
        paths = {}
        for offset, role in enumerate(roles):
            paths[role] = workdir / f"{role}.root"
            write_hist(paths[role], [20.0 + offset, 0.0 if role == "signal" else 5.0 + offset], 20.0 if offset < 4 else 2.0)
        pairs = [
            ("S1", "signal", "sb2", "sb4", "sb6"),
            ("S6", "signal", "sb4", "sb2", "sb6"),
            ("S2", "signal", "sb2", "sb1", "sb3"),
            ("S4", "signal", "sb1", "sb2", "sb3"),
            ("S3", "signal", "sb1", "sb4", "sb5"),
            ("S5", "signal", "sb4", "sb1", "sb5"),
        ]
        config = {
            "source": {
                "histogram_name": "hist", "expected_n_bins": 2,
                "expected_x_min": 100.0, "expected_x_max": 40000.0,
                "flow_policy": "exclude_underflow_and_overflow",
                "roles": {role: role for role in roles},
            },
            "front_page": {
                "title": "Region comparison for Real data",
                "reference_lumi_fb": 20.0,
                "preselection_label": "Region preselection: SciFi count > 35",
                "x_axis_title": "density",
                "y_axis_title": "Events / bin",
                "y_min": 0.1,
                "y_max": 1000.0,
                "regions": [
                    {"role": role, "label": role, "color": "#0000aa", "marker": 20 + offset}
                    for offset, role in enumerate(roles)
                ],
            },
            "comparisons": [
                {
                    "id": identifier, "transfer_title": identifier,
                    "subtitle": identifier, "transfer_y_max": 100.0,
                    "left": {"numerator": a, "denominator": b, "label": f"{a}/{b}"},
                    "right": {"numerator": c, "denominator": d, "label": f"{c}/{d}"},
                }
                for identifier, a, b, c, d in pairs
            ],
            "calculation": {"error_bar": "symmetric_independent_poisson_sqrt_n_propagation"},
            "plot": {
                "top_log_dynamic_range_threshold": 100.0,
                "lower_log_value_threshold": 0.05,
                "lower_log_dynamic_range_threshold": 100.0,
                "log_floor_fraction": 0.3,
            },
        }
        config["comparisons"][0]["top_scale"] = "linear"
        config_path = workdir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        args = argparse.Namespace(
            config=str(config_path), **{role: str(path) for role, path in paths.items()},
            output_csv=str(workdir / "result.csv"),
            output_root=str(workdir / "result.root"),
            output_pdf=str(workdir / "result.pdf"),
        )
        create_signal_ratio_comparisons(args)
        rows = list(csv.DictReader(Path(args.output_csv).open(encoding="utf-8")))
        assert len(rows) == 12
        assert {row["comparison_id"] for row in rows} == {item[0] for item in pairs}
        assert all(row["left_status"] == "PASS" for row in rows)
        assert all(row["double_status"] == "UNDEFINED_NONPOSITIVE_COUNT" for row in rows[1::2])
        output = ROOT.TFile.Open(args.output_root)
        assert output and not output.IsZombie()
        distribution_canvas = output.Get("c_region_distributions")
        assert distribution_canvas
        assert distribution_canvas.GetListOfPrimitives().GetSize() >= 11
        assert abs(output.Get("distribution_signal").Integral() - 20.0) < 1.0e-12
        assert abs(output.Get("distribution_sb4").Integral() - 330.0) < 1.0e-12
        assert output.Get("distribution_reference_lumi_fb").GetTitle() == "20"
        for identifier, *_ in pairs:
            canvas = output.Get(f"c_{identifier}")
            assert canvas
            header = canvas.GetPrimitive(f"header_{identifier}")
            top = canvas.GetPrimitive(f"top_{identifier}")
            bottom = canvas.GetPrimitive(f"bottom_{identifier}")
            assert header and header.GetListOfPrimitives().GetSize() >= 3
            assert top and top.GetListOfPrimitives().GetSize() >= 3
            assert bottom and bottom.GetListOfPrimitives().GetSize() >= 3
            assert output.Get(f"g_{identifier}_double").GetN() == 1
        assert output.Get("S1_top_scale").GetTitle() == "linear"
        output.Close()
        assert Path(args.output_pdf).stat().st_size > 0


if __name__ == "__main__":
    test_seven_page_report()
