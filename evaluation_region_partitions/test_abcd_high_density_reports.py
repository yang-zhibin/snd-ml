#!/usr/bin/env python3
"""Focused synthetic checks for the high-density simultaneous fit inputs."""

from __future__ import annotations

import math

import pyhf

from fit_abcd_high_density_reports import aggregate_inputs, build_workspace


def source_rows() -> dict[str, list[dict]]:
    rows = {role: [] for role in ("A", "B", "C", "D")}
    exposure = {"A": 10.0, "B": 10.0, "C": 2.0, "D": 2.0}
    low_counts = {"A": [7, 7, 7, 6], "B": [14, 13, 13, 13],
                  "C": [5, 4, 4, 4], "D": [3, 3, 2, 2]}
    tail_counts = {"A": 32, "B": 27, "C": 19, "D": 8}
    for role in rows:
        for low in range(0, 40000, 500):
            count = 0
            if 5000 <= low < 7000:
                count = low_counts[role][(low - 5000) // 500]
            elif low == 7000:
                count = tail_counts[role]
            rows[role].append(
                {
                    "bin_low": str(low),
                    "bin_high": str(low + 500),
                    "data_raw_events": str(count),
                    "cc_nue_sum_weights": "0.1",
                    "cc_nue_sum_weights_squared": "0.01",
                    "data_exposure": str(exposure[role]),
                }
            )
    return rows


def main() -> None:
    source = source_rows()
    inputs = aggregate_inputs(source, [5000.0, 7000.0, 40000.0])
    assert len(inputs) == 2
    assert inputs[0]["count_b"] == 53
    assert inputs[0]["count_c"] == 17
    assert inputs[0]["count_d"] == 10
    expected = 53 * 17 / 10
    assert math.isclose(inputs[0]["abcd_prediction_a"], expected)
    config = {
        "model": {
            "signal_parameter": "mu",
            "signal_initial": 1.0,
            "signal_bounds": [0.0, 100.0],
            "nonclosure_kappa": 1.0,
        }
    }
    spec = build_workspace(config, inputs)
    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    assert model.config.nmaindata == 8
    assert model.config.npars == 7
    assert set(model.config.channels) == {
        "A_signal", "B_control", "C_control", "D_control"
    }
    assert set(model.config.samples) == {"abcd_background", "cc_nue_signal"}
    assert model.config.poi_name == "mu"

    broken = source_rows()
    for row in broken["D"]:
        if float(row["bin_low"]) >= 5000:
            row["data_raw_events"] = "0"
    try:
        aggregate_inputs(broken, [5000.0, 40000.0])
    except ValueError as error:
        assert "positive B/C/D" in str(error)
    else:
        raise AssertionError("zero D count must fail")
    print("High-density simultaneous ABCD synthetic checks: OK")


if __name__ == "__main__":
    main()
