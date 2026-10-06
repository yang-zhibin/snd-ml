#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import ROOT
import yaml

from plot_abcd_original_binning_transfer_comparison import create_transfer_factor_comparison


ROOT.gROOT.SetBatch(True)


def write_hist(path: Path, counts: list[float], lumi: float) -> None:
    output = ROOT.TFile(str(path), "RECREATE")
    hist = ROOT.TH1D("hist", "", len(counts), 100.0, 40000.0)
    for index, value in enumerate(counts, 1):
        hist.SetBinContent(index, value)
        hist.SetBinError(index, value ** 0.5)
    hist.Write()
    ROOT.TParameter("double")("lumi", lumi).Write()
    output.Close()


def test_transfer_factor_plot() -> None:
    with tempfile.TemporaryDirectory(prefix="abcd_transfer_plot_test_") as directory:
        workdir = Path(directory)
        inputs = {
            "signal": ([20.0, 0.0], 20.0),
            "sb2": ([10.0, 5.0], 20.0),
            "sb4": ([40.0, 12.0], 2.0),
            "sb6": ([20.0, 6.0], 2.0),
        }
        paths = {}
        for role, (counts, lumi) in inputs.items():
            paths[role] = workdir / f"{role}.root"
            write_hist(paths[role], counts, lumi)
        config = {
            "source": {
                "histogram_name": "hist",
                "expected_n_bins": 2,
                "expected_x_min": 100.0,
                "expected_x_max": 40000.0,
                "flow_policy": "exclude_underflow_and_overflow",
            },
            "plot": {"transfer_y_max": 40.0},
        }
        config_path = workdir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        args = argparse.Namespace(
            config=str(config_path),
            **{role: str(path) for role, path in paths.items()},
            output_csv=str(workdir / "result.csv"),
            output_root=str(workdir / "result.root"),
            output_pdf=str(workdir / "result.pdf"),
            output_png=str(workdir / "result.png"),
        )
        create_transfer_factor_comparison(args)
        with open(args.output_csv, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert float(rows[0]["transfer_signal_over_sb2"]) == 2.0
        assert float(rows[0]["transfer_sb4_over_sb6"]) == 2.0
        assert float(rows[0]["double_ratio"]) == 1.0
        assert float(rows[0]["transfer_signal_over_sb2_interval_low"]) < 2.0
        assert float(rows[0]["transfer_signal_over_sb2_interval_high"]) > 2.0
        assert float(rows[0]["double_ratio_interval_low"]) < 1.0
        assert float(rows[0]["double_ratio_interval_high"]) > 1.0
        assert float(rows[1]["transfer_signal_over_sb2"]) == 0.0
        assert float(rows[1]["transfer_signal_over_sb2_interval_low"]) == 0.0
        assert float(rows[1]["transfer_signal_over_sb2_interval_high"]) > 0.0
        assert rows[1]["status_signal_over_sb2"] == "PASS"
        assert rows[1]["status_double_ratio"] == "UNDEFINED_NONPOSITIVE_COUNT"
        result = ROOT.TFile.Open(args.output_root)
        assert result and not result.IsZombie()
        assert result.Get("g_signal_over_sb2").GetN() == 2
        assert result.Get("g_sb4_over_sb6").GetN() == 2
        assert result.Get("g_double_ratio").GetN() == 1
        result.Close()
        assert Path(args.output_pdf).stat().st_size > 0
        assert Path(args.output_png).stat().st_size > 0


if __name__ == "__main__":
    test_transfer_factor_plot()
