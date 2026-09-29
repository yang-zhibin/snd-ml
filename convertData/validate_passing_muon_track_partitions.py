#!/usr/bin/env python3
"""Combine and validate full-schema passing-muon track partitions."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import ROOT
import yaml


def atomic_csv(frame, output):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def branch_names(tree):
    return [branch.GetName() for branch in tree.GetListOfBranches()]


def open_tree(path):
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot open partition ROOT file: {path}")
    if root_file.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError(f"Partition ROOT file has recovered bit set: {path}")
    tree = root_file.Get("sndData")
    if not tree:
        raise RuntimeError(f"Partition lacks sndData: {path}")
    return root_file, tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--partition-config", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--output-validation", required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--expected-lumi", type=float, required=True)
    args = parser.parse_args()

    with open(args.partition_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    fieldnames = config["metadata_output_columns"]
    frames = [pd.read_csv(path, keep_default_na=False) for path in args.rows]
    rows = pd.concat(frames, ignore_index=True)
    if len(rows) != len(args.rows):
        raise RuntimeError(
            f"Expected one metadata row per partition: {len(args.rows)} files, "
            f"{len(rows)} rows"
        )
    if rows["partition_id"].duplicated().any():
        raise RuntimeError("Duplicate partition_id in partition metadata")

    source = pd.read_csv(args.source_metadata)
    source_paths = (
        source["output_base_path"].astype(str).str.rstrip("/")
        + "/"
        + source["feature_path"].astype(str).str.lstrip("/")
    ).tolist()
    if len(source_paths) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} source files, found {len(source_paths)}"
        )
    expected_source_set = set(source_paths)
    if len(expected_source_set) != len(source_paths):
        raise RuntimeError("Source metadata contains duplicate feature paths")

    first_source_file, first_source_tree = open_tree(source_paths[0])
    source_schema = branch_names(first_source_tree)
    first_source_file.Close()
    expected_output_schema = set(source_schema) | set(
        config["output"].get("added_event_branches", [])
    )

    observed_sources = []
    total_entries = 0
    partition_schema = None
    for _, row in rows.sort_values("partition_id").iterrows():
        path = str(row["feature_partition_path"])
        root_file, tree = open_tree(path)
        schema = branch_names(tree)
        if partition_schema is None:
            partition_schema = schema
        elif schema != partition_schema:
            raise RuntimeError(f"Partition branch schema/order differs: {path}")
        if not set(source_schema).issubset(schema):
            missing = sorted(set(source_schema) - set(schema))
            raise RuntimeError(f"Partition dropped input branches {missing}: {path}")
        if set(schema) != expected_output_schema:
            unexpected = sorted(set(schema) - expected_output_schema)
            missing = sorted(expected_output_schema - set(schema))
            raise RuntimeError(
                f"Unexpected partition schema for {path}; missing={missing}, "
                f"unexpected={unexpected}"
            )

        entries = int(tree.GetEntries())
        if entries != int(row["n_events"]):
            raise RuntimeError(
                f"Entry-count mismatch for {path}: ROOT={entries}, "
                f"metadata={row['n_events']}"
            )
        source_list = [
            value for value in str(row["source_feature_paths"]).split(";") if value
        ]
        indices = [
            int(value)
            for value in str(row["source_file_indices"]).split(";")
            if value != ""
        ]
        if len(indices) != len(source_list) or len(indices) != len(set(indices)):
            raise RuntimeError(f"Invalid source_file_indices mapping in {path}")
        observed_sources.extend(source_list)

        for index in sorted({0, entries // 2, entries - 1}) if entries else []:
            tree.GetEntry(index)
            source_index = int(tree.source_file_index)
            if source_index not in set(indices):
                raise RuntimeError(
                    f"source_file_index out of range in {path}, entry {index}"
                )
            if int(tree.original_entry) != int(tree.eventIndex):
                raise RuntimeError(
                    f"original_entry/eventIndex mismatch in {path}, entry {index}"
                )
            if not str(tree.event_uid):
                raise RuntimeError(f"Empty event_uid in {path}, entry {index}")
        total_entries += entries
        root_file.Close()

    if len(observed_sources) != len(set(observed_sources)):
        raise RuntimeError("A source feature file appears in multiple partitions")
    observed_source_set = set(observed_sources)
    if observed_source_set != expected_source_set:
        missing = sorted(expected_source_set - observed_source_set)
        unexpected = sorted(observed_source_set - expected_source_set)
        raise RuntimeError(
            f"Source-file partition closure failed; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    if total_entries != args.expected_events:
        raise RuntimeError(
            f"Expected {args.expected_events} partition events, found {total_entries}"
        )
    luminosity = float(pd.to_numeric(rows["lumi_per_partition"]).sum())
    if abs(luminosity - args.expected_lumi) > 1e-12:
        raise RuntimeError(
            f"Expected luminosity {args.expected_lumi:.15g}, "
            f"found {luminosity:.15g}"
        )

    atomic_csv(rows[fieldnames], args.output_metadata)
    validation = pd.DataFrame(
        [
            {
                "status": "pass",
                "partition_count": len(rows),
                "source_file_count": len(observed_sources),
                "event_count": total_entries,
                "luminosity_fb": luminosity,
                "input_branch_count": len(source_schema),
                "partition_branch_count": len(partition_schema or []),
                "track_feature_schema": "flat_v1",
            }
        ]
    )
    atomic_csv(validation, args.output_validation)
    print(
        f"Validated {len(rows)} partitions, {len(observed_sources)} files, "
        f"{total_entries} events, {luminosity:.15g} fb^-1"
    )


if __name__ == "__main__":
    main()
