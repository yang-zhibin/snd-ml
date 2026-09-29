#!/usr/bin/env python3
"""Fill natural and sampled raw-count feature histograms for one Muon-DIS file."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation_region_partitions"))

from make_passing_muon_binary_region_hist_v5 import configured_v5_features
from make_passing_muon_track_type_hist import (
    clone_and_fold, define_shared_columns, feature_frame_and_columns,
    mkdir_path, root_input, write_parameter,
)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def one_row(path, dataset_file_uid):
    rows = pd.read_csv(path, keep_default_na=False)
    rows = rows.loc[rows["dataset_file_uid"].astype(str).eq(dataset_file_uid)]
    if len(rows) != 1:
        raise ValueError(f"Expected one source row for {dataset_file_uid}, found {len(rows)}")
    return rows.iloc[0]


def sampled_multiplicities(path, dataset_file_uid):
    rows = pd.read_csv(path, keep_default_na=False)
    rows = rows.loc[rows["dataset_file_uid"].astype(str).eq(dataset_file_uid)].copy()
    result = {}
    density = {}
    for row in rows.itertuples(index=False):
        entry = int(row.source_entry)
        multiplicity = int(row.multiplicity)
        if entry in result or multiplicity <= 0:
            raise ValueError(f"Invalid duplicate/multiplicity for {dataset_file_uid}:{entry}")
        result[entry] = multiplicity
        density[entry] = str(row.density_bin)
    return result, density


def validate_density_index(path, n_events, sampled_density):
    valid = invalid = 0
    observed = {}
    sampled_entries = set(sampled_density)
    with np.load(path, allow_pickle=False) as payload:
        for name in sorted(payload.files):
            entries = np.asarray(payload[name])
            if entries.dtype.kind not in "iu":
                raise TypeError(f"Non-integral density index {path}:{name}")
            if len(entries) and (
                int(entries.min()) < 0 or int(entries.max()) >= int(n_events)
            ):
                raise ValueError(f"Out-of-range density index {path}:{name}")
            if name.startswith("bin_"):
                valid += len(entries)
            elif name == "invalid":
                invalid += len(entries)
            else:
                raise ValueError(f"Unknown density index array {path}:{name}")
            for entry in entries:
                value = int(entry)
                if value in sampled_entries:
                    if value in observed:
                        raise ValueError(f"Density index overlap for entry {value}")
                    observed[value] = name
    if valid + invalid != int(n_events):
        raise ValueError(f"Density closure failed: {valid} + {invalid} != {n_events}")
    if set(observed) != sampled_entries:
        raise ValueError("One or more sampled entries are absent from density indices")
    for entry, expected in sampled_density.items():
        if observed[entry] != expected or expected == "invalid":
            raise ValueError(
                f"Sampled density mismatch at entry {entry}: {observed[entry]} != {expected}"
            )
    return int(valid), int(invalid)


def weighted_histogram(frame, name, feature, columns, weight):
    if int(feature["dimension"]) == 1:
        model = (
            name, "", int(feature["n_bins"]),
            float(feature["x_min"]), float(feature["x_max"]),
        )
        return frame.Histo1D(model, columns[0], weight)
    model = (
        name, "", int(feature["x_bins"]),
        float(feature["x_min"]), float(feature["x_max"]),
        int(feature["y_bins"]), float(feature["y_min"]),
        float(feature["y_max"]),
    )
    return frame.Histo2D(model, columns[0], columns[1], weight)


def declare_sample_weights(ROOT, dataset_file_uid, n_events, multiplicities):
    namespace = "muondis_weights_" + re.sub(r"[^A-Za-z0-9_]", "_", dataset_file_uid)
    assignments = "\n".join(
        f"values[{entry}] = {multiplicity}u;"
        for entry, multiplicity in sorted(multiplicities.items())
    )
    ROOT.gInterpreter.Declare(f"""
        #include <vector>
        namespace {namespace} {{
        const std::vector<unsigned int> sampled = []() {{
            std::vector<unsigned int> values({int(n_events)}, 0u);
            {assignments}
            return values;
        }}();
        }}
    """)
    return namespace


def build_shard(source_manifest, sample_manifest, dataset_file_uid,
                dataset_config, hist_config, region_config, feature_tree,
                output, threads=1):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()
    ROOT.TH1.AddDirectory(False)
    if int(threads) > 1:
        ROOT.EnableImplicitMT(int(threads))

    row = one_row(source_manifest, dataset_file_uid)
    n_events = int(row["n_events"])
    multiplicities, sampled_density = sampled_multiplicities(
        sample_manifest, dataset_file_uid
    )
    valid_events, invalid_events = validate_density_index(
        str(row["density_index_path"]), n_events, sampled_density
    )
    if int(row["valid_density_events"]) != valid_events:
        raise ValueError("Source-manifest valid density count disagrees with index")
    if int(row["invalid_density_events"]) != invalid_events:
        raise ValueError("Source-manifest invalid density count disagrees with index")
    if any(entry < 0 or entry >= n_events for entry in multiplicities):
        raise ValueError("Sample manifest contains an out-of-range source entry")

    source = ROOT.TFile.Open(root_input(str(row["feature_path"])))
    if not source or source.IsZombie():
        raise OSError(f"Could not open feature file {row['feature_path']}")
    try:
        tree = source.Get(str(feature_tree))
        if not tree:
            raise KeyError(f"Tree {feature_tree!r} is missing from {row['feature_path']}")
        if int(tree.GetEntries()) != n_events:
            raise ValueError(
                f"Feature/index entry mismatch for {dataset_file_uid}: "
                f"{tree.GetEntries()} != {n_events}"
            )
    finally:
        source.Close()

    features = configured_v5_features(hist_config, region_config)
    chain = ROOT.TChain(str(feature_tree))
    if chain.Add(root_input(str(row["feature_path"]))) <= 0:
        raise RuntimeError(f"Could not add feature file {row['feature_path']}")
    frame = ROOT.RDataFrame(chain)
    frame = define_shared_columns(frame, hist_config.get("definitions", {}))
    frame = define_shared_columns(frame, region_config.get("derived_columns", {}))
    density_variable = str(dataset_config["density_sampling"]["variable"])
    density_low = float(dataset_config["density_sampling"]["bins"][0]["low"])
    columns = {str(value) for value in frame.GetColumnNames()}
    if density_variable not in columns:
        raise KeyError(f"Feature tree lacks density variable {density_variable!r}")
    namespace = declare_sample_weights(
        ROOT, dataset_file_uid, n_events, multiplicities
    )
    frame = frame.Define(
        "__natural_weight",
        f"(std::isfinite({density_variable}) && {density_variable} >= "
        f"{density_low:.17g}) ? 1.0 : 0.0",
    )
    frame = frame.Define(
        "__sampled_weight", f"double({namespace}::sampled[rdfentry_])"
    )

    total_actions = {
        "natural": frame.Sum("__natural_weight"),
        "sampled": frame.Sum("__sampled_weight"),
    }
    counts, histograms = {}, {}
    for population, weight in (
        ("natural", "__natural_weight"),
        ("sampled", "__sampled_weight"),
    ):
        selected = frame.Filter(f"{weight} > 0", f"{population}_positive_weight")
        for feature_name, feature in features.items():
            feature_frame, feature_columns = feature_frame_and_columns(
                selected, feature_name, feature,
                f"{population}_{dataset_file_uid}",
            )
            key = (population, feature_name)
            counts[key] = feature_frame.Sum(weight)
            histograms[key] = weighted_histogram(
                feature_frame, f"{population}__{feature_name}", feature,
                feature_columns, weight,
            )
    actions = [*total_actions.values(), *counts.values(), *histograms.values()]
    if hasattr(ROOT.RDF, "RunGraphs"):
        ROOT.RDF.RunGraphs(actions)
    else:  # pragma: no cover - old ROOT fallback
        actions[0].GetValue()

    natural_total = int(round(float(total_actions["natural"].GetValue())))
    sampled_total = int(round(float(total_actions["sampled"].GetValue())))
    expected_sampled = int(sum(multiplicities.values()))
    if natural_total != valid_events:
        raise RuntimeError(f"Natural feature closure failed: {natural_total} != {valid_events}")
    if sampled_total != expected_sampled:
        raise RuntimeError(f"Sampled feature closure failed: {sampled_total} != {expected_sampled}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".root", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    target = ROOT.TFile.Open(temporary, "RECREATE")
    if not target or target.IsZombie():
        raise OSError(f"Could not create {temporary}")
    try:
        write_parameter(target, "version", "muondis_feature_sampling_v1")
        write_parameter(target, "dataset_file_uid", dataset_file_uid)
        write_parameter(target, "domain", str(row["domain"]))
        write_parameter(target, "n_events", n_events)
        write_parameter(target, "natural_events", natural_total)
        write_parameter(target, "invalid_density_events", invalid_events)
        write_parameter(target, "sampled_draws", sampled_total)
        write_parameter(target, "sampled_unique_events", len(multiplicities))
        for population in ("natural", "sampled"):
            directory = mkdir_path(target, population)
            for feature_name, feature in features.items():
                key = (population, feature_name)
                valid = int(round(float(counts[key].GetValue())))
                write_parameter(
                    target, f"valid__{population}__{feature_name}", valid
                )
                histogram = clone_and_fold(
                    histograms[key], feature_name, feature
                )
                directory.cd()
                histogram.Write(feature_name)
        target.Write()
    finally:
        target.Close()
    os.replace(temporary, output)
    return {
        "natural_events": natural_total,
        "sampled_draws": sampled_total,
        "sampled_unique_events": len(multiplicities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--dataset-file-uid", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--feature-tree", default="sndData")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = build_shard(
        args.source_manifest, args.sample_manifest, args.dataset_file_uid,
        load_yaml(args.dataset_config), load_yaml(args.hist_config),
        load_yaml(args.region_config), args.feature_tree, args.output,
        args.threads,
    )
    print(
        f"Wrote {args.dataset_file_uid}: {result['natural_events']} natural, "
        f"{result['sampled_draws']} sampled draws"
    )


if __name__ == "__main__":
    main()
