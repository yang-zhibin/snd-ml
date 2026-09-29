#!/usr/bin/env python3
"""Merge per-source Muon-DIS natural/sampled feature histogram shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation_region_partitions"))

from make_passing_muon_binary_region_hist_v5 import configured_v5_features
from make_passing_muon_track_type_hist import mkdir_path, write_parameter
from plot_passing_muon_region_study_v4 import histogram_integral, object_value


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_shards(inputs, source_manifest, sampling_summary, hist_config,
                 region_config, output_root, output_audit, output_summary):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    sources = pd.read_csv(source_manifest, keep_default_na=False)
    if sources["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Source manifest contains duplicate dataset_file_uid values")
    expected = set(sources["dataset_file_uid"].astype(str))
    source_lookup = sources.set_index("dataset_file_uid")
    with open(sampling_summary, encoding="utf-8") as stream:
        summary = json.load(stream)
    features = configured_v5_features(hist_config, region_config)
    merged = {}
    valid_counts = {}
    seen = set()
    natural_total = sampled_total = unique_total = 0
    domain_totals = {}

    for path in inputs:
        source = ROOT.TFile.Open(str(path))
        if not source or source.IsZombie():
            raise OSError(f"Could not open histogram shard {path}")
        try:
            uid = str(object_value(source, "dataset_file_uid"))
            domain = str(object_value(source, "domain"))
            if uid in seen or uid not in expected:
                raise ValueError(f"Unexpected or duplicate histogram shard {uid}")
            seen.add(uid)
            if domain != str(source_lookup.loc[uid, "domain"]):
                raise ValueError(f"Domain mismatch in histogram shard {uid}")
            shard_natural = int(object_value(source, "natural_events"))
            shard_sampled = int(object_value(source, "sampled_draws"))
            shard_unique = int(object_value(source, "sampled_unique_events"))
            if shard_natural != int(source_lookup.loc[uid, "valid_density_events"]):
                raise ValueError(f"Natural count mismatch in shard {uid}")
            natural_total += shard_natural
            sampled_total += shard_sampled
            unique_total += shard_unique
            values = domain_totals.setdefault(domain, {"natural": 0, "sampled": 0})
            values["natural"] += shard_natural
            values["sampled"] += shard_sampled
            for population in ("natural", "sampled"):
                for feature_name in features:
                    histogram = source.Get(f"{population}/{feature_name}")
                    if not histogram:
                        raise RuntimeError(
                            f"Shard {uid} lacks {population}/{feature_name}"
                        )
                    valid = int(object_value(
                        source, f"valid__{population}__{feature_name}"
                    ))
                    for scope in ("inclusive", f"domain/{domain}"):
                        key = (scope, population, feature_name)
                        valid_counts[key] = valid_counts.get(key, 0) + valid
                        if key not in merged:
                            merged[key] = histogram.Clone(
                                f"{scope.replace('/', '_')}__{population}__{feature_name}"
                            )
                            merged[key].SetDirectory(0)
                        else:
                            merged[key].Add(histogram)
        finally:
            source.Close()

    if seen != expected:
        raise ValueError(f"Histogram shard closure failed: missing={sorted(expected - seen)}")
    if natural_total != int(summary["natural_valid_events"]):
        raise RuntimeError(
            f"Natural merge closure failed: {natural_total} != "
            f"{summary['natural_valid_events']}"
        )
    if sampled_total != int(summary["muondis_draws"]):
        raise RuntimeError(
            f"Sample merge closure failed: {sampled_total} != {summary['muondis_draws']}"
        )
    if unique_total != int(summary["muondis_unique_events"]):
        raise RuntimeError(
            f"Unique-event closure failed: {unique_total} != "
            f"{summary['muondis_unique_events']}"
        )

    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_root.name}.", suffix=".root", dir=output_root.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    target = ROOT.TFile.Open(temporary, "RECREATE")
    if not target or target.IsZombie():
        raise OSError(f"Could not create {temporary}")
    try:
        write_parameter(target, "version", "muondis_feature_sampling_v1")
        write_parameter(target, "cv_fold", int(summary["cv_fold"]))
        write_parameter(target, "epoch", int(summary["epoch"]))
        write_parameter(target, "natural_events", natural_total)
        write_parameter(target, "sampled_draws", sampled_total)
        write_parameter(target, "sampled_unique_events", unique_total)
        for (scope, population, feature_name), histogram in merged.items():
            directory = mkdir_path(target, f"{scope}/{population}")
            directory.cd()
            histogram.Write(feature_name)
        target.Write()
    finally:
        target.Close()
    os.replace(temporary, output_root)

    audit_rows = []
    for key in sorted(merged):
        scope, population, feature_name = key
        histogram = merged[key]
        audit_rows.append({
            "scope": scope,
            "population": population,
            "feature": feature_name,
            "dimension": int(features[feature_name]["dimension"]),
            "valid_weight": int(valid_counts[key]),
            "visible_integral": float(histogram_integral(histogram)),
        })
    merged_summary = dict(summary)
    merged_summary.update({
        "merged_source_files": len(seen),
        "merged_natural_events": natural_total,
        "merged_sampled_draws": sampled_total,
        "merged_sampled_unique_events": unique_total,
        "merged_domain_totals": domain_totals,
        "feature_valid_counts": {
            population: {
                feature_name: int(valid_counts[("inclusive", population, feature_name)])
                for feature_name in features
            }
            for population in ("natural", "sampled")
        },
    })
    atomic_csv(pd.DataFrame(audit_rows), output_audit)
    atomic_json(merged_summary, output_summary)
    return merged_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--sampling-summary", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    result = merge_shards(
        args.inputs, args.source_manifest, args.sampling_summary,
        load_yaml(args.hist_config), load_yaml(args.region_config),
        args.output_root, args.output_audit, args.output_summary,
    )
    print(
        f"Merged {result['merged_source_files']} sources: "
        f"{result['merged_natural_events']} natural events, "
        f"{result['merged_sampled_draws']} sampled draws"
    )


if __name__ == "__main__":
    main()
