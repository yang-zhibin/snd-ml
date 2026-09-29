#!/usr/bin/env python3
"""Compare natural and uniform-bin muon-DIS sampling for the V7 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import configured_density_bins, load_yaml


def run_number(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def build_report(audits: pd.DataFrame, config: dict, data_run: str = "8329") -> pd.DataFrame:
    cohorts = {
        f"data_run{data_run}": audits.loc[
            audits["domain"].astype(str).eq("data")
            & audits["run"].map(run_number).eq(str(data_run))
        ],
        "mc_muon": audits.loc[audits["domain"].astype(str).eq("mc")],
    }
    configured_names = [item["name"] for item in configured_density_bins(config)]
    rows = []
    for cohort, selected in cohorts.items():
        if selected.empty:
            raise ValueError(f"V7 pilot cohort {cohort} has no audit rows")
        totals = {name: 0 for name in [*configured_names, "invalid"]}
        for value in selected.get("fail_density_counts", pd.Series(dtype=str)):
            for name, count in json.loads(str(value)).items():
                totals[name] = totals.get(name, 0) + int(count)
        total_events = sum(totals.values())
        populated = [name for name in configured_names if totals[name] > 0]
        if total_events <= 0 or not populated:
            raise ValueError(f"V7 pilot cohort {cohort} has no loose-failure events")
        for name in [*configured_names, "invalid"]:
            natural = totals[name] / total_events if total_events else 0.0
            uniform = 1.0 / len(populated) if name in populated else 0.0
            rows.append({
                "cohort": cohort,
                "source_files": int(len(selected)),
                "density_bin": name,
                "n_events": int(totals[name]),
                "natural_fraction": natural,
                "uniform_training_fraction": uniform,
                "relative_sampling_multiplier": uniform / natural if natural else 0.0,
            })
    return pd.DataFrame(rows)


def root_parameter(path: str, name: str) -> int:
    import ROOT

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open {path}")
    try:
        value = root_file.Get(name)
        if not value or not hasattr(value, "GetVal"):
            raise KeyError(f"Parameter {name!r} is missing from {path}")
        return int(value.GetVal())
    finally:
        root_file.Close()


def v7_density_counts(config: dict, density_hist: str, selection_hist: str) -> dict:
    """Derive approximate exclusive bins from V7 cumulative >threshold counts."""
    bins = configured_density_bins(config)
    boundaries = []
    for item in bins[:-1]:
        high = float(item["high"])
        if not high.is_integer():
            raise ValueError("V7 pilot histograms support integer density boundaries only")
        boundaries.append(int(high))
    result = {}
    for cohort, total_name, pass_name, sample in (
        ("data_run8329", "data_total_count", "selection__coverage_loose__candidate_count", "data"),
        ("mc_muon", "mc_total_count", "selection__coverage_loose__mc_candidate_count", "mc"),
    ):
        total = root_parameter(density_hist, total_name)
        passing = root_parameter(selection_hist, pass_name)
        failing = total - passing
        if failing <= 0:
            raise ValueError(f"V7 cohort {cohort} has no loose-failure events")
        cumulative = [
            root_parameter(
                density_hist,
                f"category__scifi_density_gt{boundary}__{sample}_count",
            )
            for boundary in boundaries
        ]
        if any(left < right for left, right in zip(cumulative, cumulative[1:])):
            raise ValueError(f"V7 cumulative density counts are not monotonic for {cohort}")
        if cumulative and cumulative[0] > failing:
            raise ValueError(f"V7 density subset exceeds loose-failure count for {cohort}")
        exclusive = [failing - cumulative[0]] if cumulative else [failing]
        exclusive.extend(
            left - right for left, right in zip(cumulative, cumulative[1:])
        )
        if cumulative:
            exclusive.append(cumulative[-1])
        if len(exclusive) != len(bins) or sum(exclusive) != failing:
            raise RuntimeError(f"V7 density-bin closure failed for {cohort}")
        result[cohort] = dict(zip((item["name"] for item in bins), exclusive))
    return result


def report_from_counts(counts_by_cohort: dict[str, dict[str, int]],
                       method: str) -> pd.DataFrame:
    rows = []
    for cohort, counts in counts_by_cohort.items():
        total = sum(counts.values())
        populated = [name for name, count in counts.items() if count > 0]
        if total <= 0 or not populated:
            raise ValueError(f"Cohort {cohort} has no events for a sampling report")
        for name, count in counts.items():
            natural = count / total
            uniform = 1.0 / len(populated) if count else 0.0
            rows.append({
                "cohort": cohort,
                "count_method": method,
                "density_bin": name,
                "n_events": int(count),
                "natural_fraction": natural,
                "uniform_training_fraction": uniform,
                "relative_sampling_multiplier": uniform / natural if natural else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--v7-density-hist", required=True)
    parser.add_argument("--v7-selection-hist", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = report_from_counts(
        v7_density_counts(config, args.v7_density_hist, args.v7_selection_hist),
        "V7 cumulative >threshold counts; equality is included in the lower bin",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"Wrote V7 muon-DIS sampling pilot report to {output}")


if __name__ == "__main__":
    main()
