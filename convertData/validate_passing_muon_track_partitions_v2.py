#!/usr/bin/env python3
"""Validate paired run-8329 inclusive/passing partition closure and containment."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
import yaml


def atomic_csv(frame, destination):
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


def tree_info(path, tree_name):
    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError(f"Invalid ROOT partition: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Missing {tree_name} in {path}")
    entries = int(tree.GetEntries())
    schema = {branch.GetName(): branch.GetClassName() or branch.GetTitle() for branch in tree.GetListOfBranches()}
    root_file.Close()
    return entries, schema


def encoded_keys(path, tree_name):
    columns = ["source_file_index", "runId", "eventId"]
    arrays = ROOT.RDataFrame(tree_name, str(path)).AsNumpy(columns)
    return np.rec.fromarrays(
        [np.asarray(arrays[column], dtype=np.int64) for column in columns],
        names=columns,
    )


def validate_containment(inclusive_path, passing_path, tree_name):
    inclusive = np.sort(encoded_keys(inclusive_path, tree_name))
    passing = np.sort(encoded_keys(passing_path, tree_name))
    duplicate_inclusive = int(len(inclusive) - len(np.unique(inclusive)))
    duplicate_passing = int(len(passing) - len(np.unique(passing)))
    if duplicate_inclusive or duplicate_passing:
        raise RuntimeError(
            f"Duplicate event keys: inclusive={duplicate_inclusive}, passing={duplicate_passing}"
        )
    positions = np.searchsorted(inclusive, passing)
    valid = positions < len(inclusive)
    matched = np.zeros(len(passing), dtype=bool)
    matched[valid] = inclusive[positions[valid]] == passing[valid]
    unmatched = int((~matched).sum())
    if unmatched:
        raise RuntimeError(f"Passing partition contains {unmatched} unmatched event keys")
    return len(passing), unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--output-validation", required=True)
    parser.add_argument("--dq-validation", required=True)
    parser.add_argument("--event-containment", choices=("exact", "off"), default="exact")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metadata = pd.concat([pd.read_csv(path) for path in args.rows], ignore_index=True)
    expected_rows = int(config["n_partitions"]) * len(config["groups"])
    if len(metadata) != expected_rows or metadata["partition_id"].duplicated().any():
        raise RuntimeError(f"Expected {expected_rows} unique partition rows, found {len(metadata)}")

    tree_name = config.get("tree_name", "sndData")
    validation_rows = []
    schemas = {}
    for part_index in range(1, int(config["n_partitions"]) + 1):
        group_rows = {}
        for group in config["groups"]:
            selected = metadata.loc[
                (metadata["particle_group"].astype(str) == group)
                & metadata["partition_id"].astype(str).str.endswith(f"part{part_index:03d}")
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Expected one {group} row for part {part_index:03d}")
            row = selected.iloc[0]
            path = str(row["feature_partition_path"])
            entries, schema = tree_info(path, tree_name)
            if entries != int(row["n_events"]):
                raise RuntimeError(f"ROOT/metadata event mismatch for {path}")
            schemas.setdefault(group, schema)
            if schema != schemas[group]:
                raise RuntimeError(f"Schema changes across {group} partitions")
            group_rows[group] = row

        inclusive_group = "real_data_dq_run8329"
        passing_group = "passing_muon_dq_run8329"
        inclusive = group_rows[inclusive_group]
        passing = group_rows[passing_group]
        inclusive_sources = set(str(inclusive["source_file_indices"]).split(";"))
        passing_sources = set(str(passing["source_file_indices"]).split(";"))
        if inclusive_sources != passing_sources:
            raise RuntimeError(f"Source membership differs in part {part_index:03d}")
        if int(passing["n_events"]) > int(inclusive["n_events"]):
            raise RuntimeError(f"Passing count exceeds inclusive count in part {part_index:03d}")
        if abs(float(passing["lumi_per_partition"]) - float(inclusive["lumi_per_partition"])) > 1e-12:
            raise RuntimeError(f"Paired luminosities differ in part {part_index:03d}")
        unmatched = 0
        if args.event_containment == "exact":
            _, unmatched = validate_containment(
                inclusive["feature_partition_path"], passing["feature_partition_path"], tree_name
            )
        validation_rows.append({
            "scope": f"part{part_index:03d}", "status": "pass",
            "source_file_count": len(inclusive_sources),
            "real_dq_events": int(inclusive["n_events"]),
            "passing_after_dq": int(passing["n_events"]),
            "passing_reco_efficiency_after_dq": (
                int(passing["n_events"]) / int(inclusive["n_events"])
                if int(inclusive["n_events"]) else 0.0
            ),
            "luminosity_fb": float(inclusive["lumi_per_partition"]),
            "unmatched_passing_events": unmatched,
            "event_containment": args.event_containment,
        })

    common = set(schemas["real_data_dq_run8329"]) - {
        "particle_group", "particle_id"
    }
    passing_schema = set(schemas["passing_muon_dq_run8329"])
    missing_common = sorted(common - passing_schema)
    if missing_common:
        raise RuntimeError(f"Passing schema lacks inclusive common branches: {missing_common}")

    totals = metadata.groupby("particle_group", sort=False).agg(
        n_events=("n_events", "sum"), lumi=("lumi_per_partition", "sum")
    )
    expected = config["expected"]
    dq_validation = pd.read_csv(args.dq_validation)
    if len(dq_validation) != 1 or str(dq_validation.iloc[0]["status"]) != "pass":
        raise RuntimeError("Invalid DQ-intersection validation input")
    observed_inclusive = int(totals.loc["real_data_dq_run8329", "n_events"])
    observed_passing = int(totals.loc["passing_muon_dq_run8329", "n_events"])
    if observed_inclusive != int(dq_validation.iloc[0]["real_dq_events"]):
        raise RuntimeError("Real-data DQ event-total closure failed")
    if observed_passing != int(dq_validation.iloc[0]["passing_after_dq"]):
        raise RuntimeError("Passing-after-DQ event-total closure failed")
    tolerance = float(expected.get("luminosity_tolerance_fb", 1e-12))
    for group in config["groups"]:
        if abs(float(totals.loc[group, "lumi"]) - float(expected["luminosity_fb"])) > tolerance:
            raise RuntimeError(f"Luminosity closure failed for {group}")
    validation_rows.append({
        "scope": "overall", "status": "pass",
        "source_file_count": sum(row["source_file_count"] for row in validation_rows),
        "real_dq_events": observed_inclusive, "passing_after_dq": observed_passing,
        "passing_reco_efficiency_after_dq": (
            observed_passing / observed_inclusive if observed_inclusive else 0.0
        ),
        "luminosity_fb": float(expected["luminosity_fb"]),
        "unmatched_passing_events": sum(row["unmatched_passing_events"] for row in validation_rows),
        "event_containment": args.event_containment,
    })
    atomic_csv(metadata.sort_values(["particle_group", "partition_id"]), args.output_metadata)
    atomic_csv(pd.DataFrame(validation_rows), args.output_validation)
    efficiency = observed_passing / observed_inclusive if observed_inclusive else 0.0
    print(f"Validated paired partitions; DQ-conditional efficiency={efficiency:.6%}")


if __name__ == "__main__":
    main()
