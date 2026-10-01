#!/usr/bin/env python3
"""Focused synthetic tests for the blinded ABCD expected-inference path."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pyhf

from make_abcd_blinded_inputs import assert_blinded
from run_abcd_blinded_expected import aggregate, load_sanitized, run_analysis


def config() -> dict:
    return {
        "version": "test_blinded_v1",
        "source": {
            "observable": "density_scifi_sndsw",
            "roles": {"A": "signal", "B": "b", "C": "c", "D": "d"},
            "signal_sample": "CC_nue",
            "background_samples": ["CC_numu", "NC_nue", "NC_numu"],
        },
        "model": {
            "signal_parameter": "mu", "signal_initial": 1.0,
            "signal_bounds": [0.0, 20.0], "signal_injection": 1.0,
            "nonclosure": {"nominal": 1.0, "constraint": "none", "validation_status": "UNVALIDATED"},
            "background_mc_normalization": {"CC_numu": 0.3, "NC_nue": 0.3, "NC_numu": 0.3},
            "background_mc_uncertainty_status": "PLACEHOLDER",
        },
        "fit": {"maxiter": 100000, "tolerance": 0.1, "mu_scan_range": [0.0, 3.0], "mu_scan_points": 9},
        "analyses": {
            "cutcount": {"title": "test cut", "role": "test", "bin_edges": [5000, 40000], "output_directory": "cut"},
            "shape": {"title": "test shape", "role": "test", "bin_edges": [5000, 7000, 40000], "output_directory": "shape"},
        },
        "output": {
            "model_inputs": "model_inputs.csv", "control_workspace": "control_workspace.json",
            "expected_background_workspace": "expected_background_workspace.json",
            "expected_signal_workspace": "expected_signal_workspace.json", "expected_results": "expected_results.json",
            "mu_scan": "expected_mu_scan.csv", "report_markdown": "expected_report.md",
            "report_tex": "expected_report.tex", "plot_directory": "plots", "freeze_manifest": "freeze_manifest.json",
        },
        "blinding": {"status": "REBLINDED_DEVELOPMENT", "observed_inference_authorized": False},
    }


def rows(cfg: dict) -> list[dict]:
    output = []
    for role in ("A", "B", "C", "D"):
        for index, low in enumerate(range(0, 40000, 500), start=1):
            active = low >= 5000
            counts = {"B": 40, "C": 30, "D": 20}
            row = {
                "input_version": cfg["version"], "blinding_status": cfg["blinding"]["status"],
                "role": role, "region": cfg["source"]["roles"][role], "bin_index": index,
                "bin_low": low, "bin_high": low + 500,
                "exposure": 10.0, "observed_data": "" if role == "A" else (counts[role] if active else 1),
            }
            for sample in ("cc_nue", "cc_numu", "nc_nue", "nc_numu"):
                value = 1.0 if active and sample == "cc_nue" and role == "A" else (0.05 if active else 0.0)
                row[f"{sample}_sum_weights"] = value
                row[f"{sample}_sum_weights_squared"] = value * 0.01
            output.append(row)
    return output


def write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def main() -> None:
    pyhf.set_backend("numpy")
    cfg = config(); values = rows(cfg); assert_blinded(values)
    with tempfile.TemporaryDirectory(prefix="abcd_blinded_test.") as temporary:
        root = Path(temporary); table = root / "sanitized.csv"; write_csv(table, values)
        loaded = load_sanitized(table, cfg)
        merged = aggregate(loaded, [5000.0, 7000.0, 40000.0], cfg)
        assert len(merged) == 2 and merged[0]["observed_a"] == ""
        assert merged[0]["observed_b"] == 160
        result = run_analysis(cfg, loaded, "shape", root)
        assert result["z"] > 0.0
        payload = json.loads((root / "shape" / "expected_results.json").read_text())
        assert payload["signal_region_observation_used"] is False
        workspace = json.loads((root / "shape" / "expected_signal_workspace.json").read_text())
        assert pyhf.Workspace(workspace).model().config.nmaindata == 8
        assert "LOCKED" in (root / "shape" / "expected_report.tex").read_text()

        broken = [dict(row) for row in values]
        next(row for row in broken if row["role"] == "A")["observed_data"] = 123
        write_csv(root / "broken.csv", broken)
        try:
            load_sanitized(root / "broken.csv", cfg)
        except ValueError as error:
            assert "Observed signal-region data" in str(error)
        else:
            raise AssertionError("An A observation must be rejected")
    print("Blinded ABCD expected-inference synthetic checks: OK")


if __name__ == "__main__":
    main()
