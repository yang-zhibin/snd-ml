#!/usr/bin/env python3
"""Validate run-8329 NoPassingMuonReco partitions and write their metadata."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
import yaml


KEY_COLUMNS = ("source_file_index", "runId", "eventId")


def atomic_csv(frame, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def root_input(path):
    path = str(path)
    return f"root://eospublic.cern.ch/{path}" if path.startswith("/eos/") else path


def tree_info(path, tree_name):
    root_file = ROOT.TFile.Open(root_input(path))
    if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError(f"Invalid ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Missing {tree_name} in {path}")
    entries = int(tree.GetEntries())
    schema = {
        branch.GetName(): branch.GetClassName() or branch.GetTitle()
        for branch in tree.GetListOfBranches()
    }
    root_file.Close()
    return entries, schema


def event_keys(path, tree_name):
    arrays = ROOT.RDataFrame(tree_name, root_input(path)).AsNumpy(list(KEY_COLUMNS))
    return np.rec.fromarrays(
        [np.asarray(arrays[name], dtype=np.int64) for name in KEY_COLUMNS],
        names=KEY_COLUMNS,
    )


def metadata_row(metadata, group, part_index):
    selected = metadata.loc[
        (metadata["particle_group"].astype(str) == group)
        & metadata["partition_id"].astype(str).str.endswith(f"part{part_index:03d}")
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {group} metadata row for part {part_index:03d}")
    return selected.iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--output-validation", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    audits = pd.concat([pd.read_csv(path) for path in args.audits], ignore_index=True)
    paired = pd.read_csv(args.partition_metadata, keep_default_na=False)
    expected = config["expected"]
    n_partitions = int(expected["n_partitions"])
    if len(audits) != n_partitions or audits["part_index"].duplicated().any():
        raise RuntimeError(
            f"Expected {n_partitions} unique audit rows, found {len(audits)}"
        )
    if not (audits["status"].astype(str) == "pass").all():
        raise RuntimeError("At least one missing-partition audit did not pass")

    tree_name = str(config.get("tree_name", "sndData"))
    real_group = str(config["groups"]["real_data"])
    passing_group = str(config["groups"]["passing"])
    missing_group = str(config["groups"]["missing"])
    output_rows = []
    validation_rows = []
    reference_schema = None

    for part_index in range(1, n_partitions + 1):
        selected = audits.loc[
            pd.to_numeric(audits["part_index"]).astype(int) == part_index
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing audit for part {part_index:03d}")
        audit = selected.iloc[0]
        real_row = metadata_row(paired, real_group, part_index)
        passing_row = metadata_row(paired, passing_group, part_index)
        real_path = str(real_row["feature_partition_path"])
        passing_path = str(passing_row["feature_partition_path"])
        missing_path = str(audit["missing_partition_path"])
        if str(audit["real_partition_path"]) != real_path:
            raise RuntimeError(f"Real path disagreement for part {part_index:03d}")
        if str(audit["passing_partition_path"]) != passing_path:
            raise RuntimeError(f"Passing path disagreement for part {part_index:03d}")

        real_entries, real_schema = tree_info(real_path, tree_name)
        passing_entries, _ = tree_info(passing_path, tree_name)
        missing_entries, missing_schema = tree_info(missing_path, tree_name)
        if missing_schema != real_schema:
            raise RuntimeError(f"Missing/real schema mismatch in part {part_index:03d}")
        if reference_schema is None:
            reference_schema = missing_schema
        elif missing_schema != reference_schema:
            raise RuntimeError("NoPassingMuonReco schema changes across partitions")
        if (
            real_entries != int(audit["real_dq_events"])
            or passing_entries != int(audit["passing_muon_dq_events"])
            or missing_entries != int(audit["no_passing_muon_reco_events"])
        ):
            raise RuntimeError(f"ROOT/audit count mismatch in part {part_index:03d}")
        if real_entries != passing_entries + missing_entries:
            raise RuntimeError(f"Count closure failed in part {part_index:03d}")

        real_keys = event_keys(real_path, tree_name)
        passing_keys = event_keys(passing_path, tree_name)
        missing_keys = event_keys(missing_path, tree_name)
        if len(np.unique(missing_keys)) != len(missing_keys):
            raise RuntimeError(f"Duplicate missing keys in part {part_index:03d}")
        if np.isin(passing_keys, missing_keys, assume_unique=True).any():
            raise RuntimeError(f"Passing/missing overlap in part {part_index:03d}")
        combined = np.concatenate([passing_keys, missing_keys])
        if not np.array_equal(np.sort(combined), np.sort(real_keys)):
            raise RuntimeError(f"Passing plus missing does not equal real DQ in part {part_index:03d}")

        lumi = float(real_row["lumi_per_partition"])
        if abs(lumi - float(passing_row["lumi_per_partition"])) > 1e-12:
            raise RuntimeError(f"Paired luminosity mismatch in part {part_index:03d}")
        output_rows.append(
            {
                "study_version": config["version"],
                "region": "all_events",
                "particle_group": missing_group,
                "particle_family": "real_data",
                # This is a real-data complement selected by event identity,
                # not a new simulated particle species.
                "particle_id": 0,
                "partition_id": f"all_events__{missing_group}__part{part_index:03d}",
                "feature_partition_path": missing_path,
                "feature_available": True,
                "n_events": missing_entries,
                "lumi_per_partition": lumi,
                "source_file_indices": real_row["source_file_indices"],
                "source_file_count": real_row["source_file_count"],
                "selection_expression": "RealDataDQ AND NOT PassingMuonReco",
            }
        )
        validation_rows.append(
            {
                "scope": f"part{part_index:03d}",
                "status": "pass",
                "real_data_dq": real_entries,
                "passing_muon_dq": passing_entries,
                "no_passing_muon_reco": missing_entries,
                "passing_missing_overlap": 0,
                "count_closure": 1,
            }
        )

    totals = {
        "real_data_dq": int(audits["real_dq_events"].sum()),
        "passing_muon_dq": int(audits["passing_muon_dq_events"].sum()),
        "no_passing_muon_reco": int(audits["no_passing_muon_reco_events"].sum()),
    }
    for name, value in totals.items():
        if value != int(expected[name]):
            raise RuntimeError(
                f"Overall {name} mismatch: observed={value}, expected={expected[name]}"
            )
    if totals["passing_muon_dq"] + totals["no_passing_muon_reco"] != totals["real_data_dq"]:
        raise RuntimeError("Overall RealDataDQ partition closure failed")
    validation_rows.append(
        {
            "scope": "overall",
            "status": "pass",
            **totals,
            "passing_missing_overlap": 0,
            "count_closure": 1,
        }
    )
    atomic_csv(pd.DataFrame(output_rows), args.output_metadata)
    atomic_csv(pd.DataFrame(validation_rows), args.output_validation)
    print(
        "Validated NoPassingMuonReco: "
        f"{totals['no_passing_muon_reco']}/{totals['real_data_dq']} events"
    )


if __name__ == "__main__":
    main()
