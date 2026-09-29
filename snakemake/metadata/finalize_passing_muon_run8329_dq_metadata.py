#!/usr/bin/env python3
"""Finalize paired run-8329 metadata from per-source DQ-intersection audits."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def atomic_csv(frame: pd.DataFrame, destination: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assign_partitions(rows: pd.DataFrame, n_partitions: int) -> pd.Series:
    if n_partitions <= 0 or n_partitions > len(rows):
        raise ValueError("Invalid number of partitions")
    weights = pd.to_numeric(rows["real_dq_events"]).astype(int).tolist()
    total = sum(weights)
    assignments = []
    current = 1
    cumulative = 0
    for position, weight in enumerate(weights):
        assignments.append(current)
        cumulative += weight
        remaining_sources = len(weights) - position - 1
        remaining_parts = n_partitions - current
        threshold_reached = total == 0 or cumulative >= total * current / n_partitions
        if current < n_partitions and threshold_reached and remaining_sources >= remaining_parts:
            current += 1
    if sorted(set(assignments)) != list(range(1, n_partitions + 1)):
        raise RuntimeError("Failed to populate every paired partition")
    return pd.Series(assignments, index=rows.index, dtype=int)


def finalize(real: pd.DataFrame, pairs: pd.DataFrame, audits: pd.DataFrame, config: dict):
    required_audit = {
        "status", "run", "source_file_index", "filtered_passing_feature_path",
        "real_dq_events", "passing_before_dq", "passing_after_dq",
        "passing_rejected_by_dq", "duplicate_real_keys", "duplicate_passing_keys",
        "wrong_run_real", "wrong_run_passing",
    }
    missing = required_audit - set(audits.columns)
    if missing:
        raise ValueError(f"DQ audits lack columns: {sorted(missing)}")
    run = int(config["run"])
    audits = audits.loc[pd.to_numeric(audits["run"]) == run].copy()
    audits["source_file_index"] = pd.to_numeric(audits["source_file_index"]).astype(int)
    if audits["source_file_index"].duplicated().any():
        raise ValueError("DQ audits contain duplicate source indices")
    expected_indices = sorted(pd.to_numeric(real["source_file_index"]).astype(int).tolist())
    if sorted(audits["source_file_index"].tolist()) != expected_indices:
        raise ValueError("DQ audits do not cover exactly the paired source indices")
    fatal_columns = [
        "duplicate_real_keys", "duplicate_passing_keys", "wrong_run_real",
        "wrong_run_passing",
    ]
    if not (audits["status"].astype(str) == "pass").all():
        raise ValueError("At least one source DQ audit did not pass")
    if any(pd.to_numeric(audits[column]).sum() for column in fatal_columns):
        raise ValueError("DQ audits contain duplicate keys or wrong-run events")
    before = pd.to_numeric(audits["passing_before_dq"]).astype(int)
    after = pd.to_numeric(audits["passing_after_dq"]).astype(int)
    rejected = pd.to_numeric(audits["passing_rejected_by_dq"]).astype(int)
    denominator = pd.to_numeric(audits["real_dq_events"]).astype(int)
    if not ((after + rejected) == before).all():
        raise ValueError("Per-source passing DQ accounting does not close")
    if (after > denominator).any():
        raise ValueError("A source has more matched passing events than real DQ events")
    expected_passing = int(config["expected"]["passing_events"])
    if int(before.sum()) != expected_passing:
        raise ValueError(f"Passing-before-DQ total {before.sum()} != {expected_passing}")

    audit_columns = [
        "source_file_index", "filtered_passing_feature_path", "real_dq_events",
        "passing_before_dq", "passing_after_dq", "passing_rejected_by_dq",
        "passing_dq_acceptance", "passing_reco_efficiency_after_dq",
    ]
    real = real.drop(
        columns=[column for column in audit_columns[1:] if column in real],
        errors="ignore",
    )
    real = real.merge(audits[audit_columns], on="source_file_index", validate="one_to_one")
    real = real.sort_values("source_file_index").reset_index(drop=True)
    real["paired_part_index"] = assign_partitions(
        real, int(config["partitioning"]["n_partitions"])
    )
    real["selection_mode"] = "external_cutflow_data_quality"

    passing = real[[
        "run", "source_file_index", "source_path", "source_n_event",
        "geo_path", "lumi_per_file", "paired_part_index", "real_dq_events",
        "passing_before_dq", "passing_after_dq", "passing_rejected_by_dq",
        "filtered_passing_feature_path",
    ]].copy()
    passing["feature_full_path"] = passing["filtered_passing_feature_path"]
    passing["selection_mode"] = "passing_muon_reco_intersect_real_data_dq"

    pairs = pairs.drop(
        columns=[
            column for column in ["paired_part_index", *audit_columns[1:]]
            if column in pairs
        ],
        errors="ignore",
    )
    pairs = pairs.merge(
        real[["source_file_index", "paired_part_index", *audit_columns[1:]]],
        on="source_file_index", validate="one_to_one",
    )
    real_total = int(real["real_dq_events"].sum())
    passing_after = int(real["passing_after_dq"].sum())
    passing_before = int(real["passing_before_dq"].sum())
    rejected_total = int(real["passing_rejected_by_dq"].sum())
    luminosity = float(pd.to_numeric(real["lumi_per_file"]).sum())
    expected_luminosity = float(config["expected"]["luminosity_fb"])
    luminosity_tolerance = float(
        config["expected"].get("luminosity_tolerance_fb", 1e-12)
    )
    if abs(luminosity - expected_luminosity) > luminosity_tolerance:
        raise ValueError(
            f"Final real-data luminosity {luminosity:.15g} != "
            f"expected {expected_luminosity:.15g}"
        )
    validation = pd.DataFrame([{
        "status": "pass",
        "run": run,
        "source_files": len(real),
        "real_dq_events": real_total,
        "passing_before_dq": passing_before,
        "passing_after_dq": passing_after,
        "passing_rejected_by_dq": rejected_total,
        "raw_passing_muon_efficiency": (
            passing_before / int(config["expected"]["attempted_events"])
        ),
        "passing_dq_acceptance": passing_after / passing_before if passing_before else 0.0,
        "passing_reco_efficiency_after_dq": passing_after / real_total if real_total else 0.0,
        "luminosity_fb": luminosity,
    }])
    return real, passing, pairs, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-metadata", required=True)
    parser.add_argument("--source-pairs", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-real", required=True)
    parser.add_argument("--output-passing", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-validation", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    audits = pd.concat([pd.read_csv(path) for path in args.audits], ignore_index=True)
    tables = finalize(
        pd.read_csv(args.real_metadata), pd.read_csv(args.source_pairs), audits, config
    )
    for table, path in zip(tables, [
        args.output_real, args.output_passing, args.output_pairs, args.output_validation,
    ]):
        atomic_csv(table, path)
    result = tables[-1].iloc[0]
    print(
        f"DQ intersection: {int(result.passing_after_dq)}/"
        f"{int(result.real_dq_events)} = "
        f"{float(result.passing_reco_efficiency_after_dq):.6%}"
    )


if __name__ == "__main__":
    main()
