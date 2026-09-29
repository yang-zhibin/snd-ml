#!/usr/bin/env python3
"""Build one source-aligned run-8329 inclusive or passing ROOT partition."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import ROOT
import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def selected_paths(metadata: pd.DataFrame, kind: str, part_index: int):
    required = {"source_file_index", "paired_part_index", "lumi_per_file"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks columns: {sorted(missing)}")
    selected = metadata.loc[
        pd.to_numeric(metadata["paired_part_index"]).astype(int) == int(part_index)
    ].copy()
    if selected.empty:
        raise ValueError(f"No {kind} rows assigned to part {part_index:03d}")
    if kind in {"real_dq", "passing_dq"}:
        if "feature_full_path" not in selected:
            raise ValueError(f"{kind} metadata lacks feature_full_path")
        paths = selected["feature_full_path"].astype(str).tolist()
    else:
        raise ValueError(f"Unknown metadata kind: {kind}")
    return selected, paths


def partition_luminosity(rows: pd.DataFrame, kind: str) -> float:
    if kind not in {"real_dq", "passing_dq"}:
        raise ValueError(f"Unknown metadata kind: {kind}")
    # Both finalized tables contain one row per original source.  Exposure is
    # defined by the real-data source, never by reconstructed chunks/events.
    values = rows.drop_duplicates("source_file_index")["lumi_per_file"]
    return float(pd.to_numeric(values).sum())


def input_path(path):
    return f"root://eospublic.cern.ch/{path}" if str(path).startswith("/eos/") else str(path)


def build_dataframe(rows, paths, tree_name):
    spec = ROOT.RDF.Experimental.RDatasetSpec()
    for sample_number, ((_, row), path) in enumerate(zip(rows.iterrows(), paths)):
        metadata = ROOT.RDF.Experimental.RMetaData()
        metadata.Add("source_file_index", int(row["source_file_index"]))
        sample = ROOT.RDF.Experimental.RSample(
            f"sample_{sample_number:06d}", tree_name, input_path(path), metadata
        )
        spec.AddSample(sample)
    return ROOT.RDataFrame(spec).DefinePerSample(
        "source_file_index", 'rdfsampleinfo_.GetI("source_file_index")'
    )


def vector_string(values):
    result = ROOT.std.vector("string")()
    for value in values:
        result.push_back(str(value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--part-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-row", required=True)
    parser.add_argument("--published-root")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.group not in config["groups"]:
        raise ValueError(f"Unknown group {args.group!r}")
    if not 1 <= args.part_index <= int(config["n_partitions"]):
        raise ValueError("part-index is outside configured range")
    group = config["groups"][args.group]
    rows, paths = selected_paths(pd.read_csv(args.metadata), group["metadata_kind"], args.part_index)
    missing = [
        path for path in paths
        if path.startswith("/") and not path.startswith("/eos/") and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(f"Missing input feature file: {missing[0]}")

    threads = args.threads if args.threads is not None else int(config["output"].get("rdf_threads", 0))
    if threads > 0:
        ROOT.EnableImplicitMT(threads)
    rdf = build_dataframe(rows, paths, config.get("tree_name", "sndData"))
    columns = {str(value) for value in rdf.GetColumnNames()}
    required = {"runId", "eventId", "eventIndex"}
    if required - columns:
        raise RuntimeError(f"Input lacks identity branches: {sorted(required - columns)}")
    group_literal = args.group.replace('"', '\\"')
    if "original_entry" not in columns:
        rdf = rdf.Define("original_entry", "(long long)eventIndex")
    if "particle_group" not in columns:
        rdf = rdf.Define("particle_group", f'std::string("{group_literal}")')
    if "particle_id" not in columns:
        rdf = rdf.Define("particle_id", str(int(group["particle_id"])))
    if "event_uid" not in columns:
        rdf = rdf.Define(
            "event_uid",
            'std::string("source_") + std::to_string(source_file_index) + '
            'std::string(":run_") + std::to_string(runId) + '
            'std::string(":event_") + std::to_string(eventId)',
        )
    output_columns = [str(value) for value in rdf.GetColumnNames()]
    for added in ("source_file_index", "original_entry", "particle_group", "particle_id", "event_uid"):
        if added not in output_columns:
            output_columns.append(added)
    count = rdf.Count()
    Path(args.output_root).parent.mkdir(parents=True, exist_ok=True)
    options = ROOT.RDF.RSnapshotOptions()
    options.fCompressionLevel = int(config["output"].get("compression_level", 1))
    snapshot = rdf.Snapshot(
        config.get("tree_name", "sndData"),
        args.output_root,
        vector_string(output_columns),
        options,
    )
    if hasattr(snapshot, "GetValue"):
        snapshot.GetValue()
    n_events = int(count.GetValue())
    source_indices = sorted(pd.to_numeric(rows["source_file_index"]).astype(int).unique())
    lumi = partition_luminosity(rows, group["metadata_kind"])
    published = args.published_root or args.output_root
    output_row = Path(args.output_row)
    output_row.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "partition_version", "region_version", "region", "particle_group",
        "particle_family", "particle_id", "partition_id", "feature_partition_path",
        "feature_available", "n_events", "lumi_per_partition", "source_file_indices",
        "source_file_count", "source_input_count", "selection_expression",
    ]
    with open(output_row, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "partition_version": config["partition_version"],
            "region_version": config["region_version"],
            "region": config.get("region", "all_events"),
            "particle_group": args.group,
            "particle_family": group["particle_family"],
            "particle_id": group["particle_id"],
            "partition_id": f"all_events__{args.group}__part{args.part_index:03d}",
            "feature_partition_path": published,
            "feature_available": True,
            "n_events": n_events,
            "lumi_per_partition": lumi,
            "source_file_indices": ";".join(map(str, source_indices)),
            "source_file_count": len(source_indices),
            "source_input_count": len(paths),
            "selection_expression": "true",
        })
    print(f"Wrote {args.group} part {args.part_index:03d}: {n_events} events, {lumi:.12g} fb^-1")


if __name__ == "__main__":
    main()
