#!/usr/bin/env python3
"""Build Run-8329 detector-selection diagnostics from flat-v2 reco tracks."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import ROOT

from make_passing_muon_binary_region_hist_v5 import (
    configured_selections,
    validate_region_config,
)
from make_passing_muon_track_type_hist import (
    book_histogram,
    clone_and_fold,
    define_shared_columns,
    feature_frame_and_columns,
    load_yaml,
    mkdir_path,
    write_parameter,
)
from passing_muon_v7_reco import (
    configured_reco_diagnostics,
    required_reco_branches,
)


POPULATIONS = OrderedDict((
    ("candidate", "candidate"),
    ("complement", "complement"),
))


def root_input(path):
    path = str(path)
    return f"root://eospublic.cern.ch/{path}" if path.startswith("/eos/") else path


def dataframe_from_manifest(path, diagnostics, tree_name, configured_run):
    metadata = pd.read_csv(path, keep_default_na=False)
    required = {
        "run",
        "source_file_index",
        "particle_group",
        "feature_path",
        "tree_name",
        "n_events",
        "track_feature_schema",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"V7 track manifest lacks columns: {sorted(missing)}")
    rows = metadata.loc[
        metadata["particle_group"].astype(str).eq(diagnostics["input_group"])
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No V7 track inputs for group {diagnostics['input_group']!r}"
        )
    schemas = set(rows["track_feature_schema"].astype(str))
    if schemas != {diagnostics["required_schema"]}:
        raise RuntimeError(f"Unexpected V7 track schemas: {sorted(schemas)}")
    if set(pd.to_numeric(rows["run"]).astype(int)) != {int(configured_run)}:
        raise RuntimeError("V7 track manifest contains an unexpected run")
    if set(rows["tree_name"].astype(str)) != {tree_name}:
        raise RuntimeError("V7 track manifest contains an unexpected tree name")
    if rows["source_file_index"].duplicated().any():
        raise RuntimeError("V7 track manifest repeats source_file_index")
    chain = ROOT.TChain(tree_name)
    nonempty_rows = rows.loc[pd.to_numeric(rows["n_events"]) > 0].copy()
    if nonempty_rows.empty:
        raise RuntimeError("V7 track manifest contains no events")
    for source in nonempty_rows.sort_values("source_file_index")[
        "feature_path"
    ].astype(str):
        if chain.Add(root_input(source)) <= 0:
            raise RuntimeError(f"Could not add V7 track input {source}")
    return ROOT.RDataFrame(chain), rows, chain


def book_features(frame, features, tag):
    counts, histograms = {}, {}
    for feature_name, feature in features.items():
        feature_frame, columns = feature_frame_and_columns(
            frame, feature_name, feature, tag
        )
        counts[feature_name] = feature_frame.Count()
        histograms[feature_name] = book_histogram(
            feature_frame, f"{tag}__{feature_name}", feature, columns
        )
    return counts, histograms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-manifest", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    config = load_yaml(args.region_config)
    validate_region_config(config)
    diagnostics = configured_reco_diagnostics(config)
    selections = configured_selections(config)
    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    tree_name = str(config.get("tree_name", "sndData"))
    frame, rows, input_owner = dataframe_from_manifest(
        args.track_manifest, diagnostics, tree_name, int(config["run"])
    )
    columns = {str(value) for value in frame.GetColumnNames()}
    required = required_reco_branches(diagnostics)
    required.update({
        "count_veto",
        "count_scifi",
        "count_us",
        "count_ds",
        "scifi_n_active_stations_xy",
        *(f"count_veto{i}" for i in range(1, 4)),
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
    })
    missing = required - columns
    if missing:
        raise RuntimeError(f"V7 track inputs lack branches: {sorted(missing)}")
    frame = define_shared_columns(frame, config["derived_columns"])

    actions = {("total",): frame.Count()}
    feature_counts = {}
    histogram_actions = {}
    group_event_feature_counts = {}
    group_event_histogram_actions = {}
    availability_actions = {}
    for selection_name, selection in selections.items():
        expression = selection["expression"]
        population_frames = {
            "candidate": frame.Filter(expression, f"{selection_name}_candidate"),
            "complement": frame.Filter(
                f"!({expression})", f"{selection_name}_complement"
            ),
        }
        for population, population_frame in population_frames.items():
            actions[("selection", selection_name, population)] = (
                population_frame.Count()
            )
            counts, histograms = book_features(
                population_frame,
                diagnostics["event_features"],
                f"{selection_name}_{population}_event",
            )
            for feature_name in diagnostics["event_features"]:
                key = (selection_name, population, "event", feature_name)
                feature_counts[key] = counts[feature_name]
                histogram_actions[key] = histograms[feature_name]

            for group_key, plot_group in diagnostics["plot_groups"].items():
                counts, histograms = book_features(
                    population_frame,
                    plot_group["event_features"],
                    f"{selection_name}_{population}_{group_key}_event",
                )
                for feature_name in plot_group["event_features"]:
                    key = (selection_name, population, group_key, feature_name)
                    group_event_feature_counts[key] = counts[feature_name]
                    group_event_histogram_actions[key] = histograms[feature_name]

            for track_key, track in diagnostics["track_types"].items():
                prefix = str(track["prefix"])
                for state in ("has_any", "has_converged", "has_best"):
                    availability_actions[(
                        selection_name, population, track_key, state
                    )] = population_frame.Filter(
                        f"{prefix}_{state} == 1",
                        f"{selection_name}_{population}_{track_key}_{state}",
                    ).Count()
                counts, histograms = book_features(
                    population_frame,
                    diagnostics["track_features"][track_key],
                    f"{selection_name}_{population}_{track_key}",
                )
                for feature_name in diagnostics["track_features"][track_key]:
                    key = (selection_name, population, track_key, feature_name)
                    feature_counts[key] = counts[feature_name]
                    histogram_actions[key] = histograms[feature_name]

    ROOT.RDF.RunGraphs(
        list(actions.values())
        + list(feature_counts.values())
        + list(histogram_actions.values())
        + list(group_event_feature_counts.values())
        + list(group_event_histogram_actions.values())
        + list(availability_actions.values())
    )
    counts = {key: int(action.GetValue()) for key, action in actions.items()}
    total = counts[("total",)]
    declared = int(pd.to_numeric(rows["n_events"]).sum())
    if total != declared:
        raise RuntimeError(f"V7 manifest/tree count mismatch: {declared} != {total}")
    for selection_name in selections:
        candidate = counts[("selection", selection_name, "candidate")]
        complement = counts[("selection", selection_name, "complement")]
        if candidate + complement != total:
            raise RuntimeError(
                f"V7 selection closure failed for {selection_name}: "
                f"{candidate} + {complement} != {total}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        output = ROOT.TFile.Open(temporary, "RECREATE")
        if not output or output.IsZombie():
            raise OSError(f"Could not create {temporary}")
        try:
            for key, result in histogram_actions.items():
                selection_name, population, owner, feature_name = key
                features = (
                    diagnostics["event_features"]
                    if owner == "event"
                    else diagnostics["track_features"][owner]
                )
                histogram = clone_and_fold(
                    result, POPULATIONS[population], features[feature_name]
                )
                if owner == "event":
                    path = (
                        f"selections/{selection_name}/event_features/{feature_name}"
                    )
                else:
                    path = (
                        f"selections/{selection_name}/tracks/{owner}/"
                        f"features/{feature_name}"
                    )
                directory = mkdir_path(output, path)
                directory.cd()
                histogram.Write(POPULATIONS[population])

            for key, result in group_event_histogram_actions.items():
                selection_name, population, group_key, feature_name = key
                feature = diagnostics["plot_groups"][group_key][
                    "event_features"
                ][feature_name]
                histogram = clone_and_fold(
                    result, POPULATIONS[population], feature
                )
                directory = mkdir_path(
                    output,
                    f"selections/{selection_name}/plot_groups/{group_key}/"
                    f"event_features/{feature_name}",
                )
                directory.cd()
                histogram.Write(POPULATIONS[population])

            write_parameter(output, "study_version", config["version"])
            write_parameter(output, "run", int(config["run"]))
            write_parameter(output, "uses_reconstructed_tracks", True)
            write_parameter(output, "track_feature_schema", "flat_v2")
            write_parameter(output, "track_input_group", diagnostics["input_group"])
            write_parameter(output, "track_input_file_count", len(rows))
            write_parameter(output, "track_input_event_count", total)
            write_parameter(output, "track_manifest", str(args.track_manifest))
            write_parameter(output, "selection_order", ",".join(selections))
            write_parameter(
                output, "track_type_order", ",".join(diagnostics["track_types"])
            )
            write_parameter(
                output,
                "event_feature_order",
                ",".join(diagnostics["event_features"]),
            )
            write_parameter(
                output, "plot_group_order", ",".join(diagnostics["plot_groups"])
            )
            for group_key, plot_group in diagnostics["plot_groups"].items():
                write_parameter(
                    output,
                    f"plot_group__{group_key}__event_feature_order",
                    ",".join(plot_group["event_features"]),
                )
            write_parameter(
                output,
                "track_feature_order",
                ",".join(next(iter(diagnostics["track_features"].values()))),
            )
            for selection_name in selections:
                for population in POPULATIONS:
                    population_count = counts[(
                        "selection", selection_name, population
                    )]
                    prefix = f"selection__{selection_name}__{population}"
                    write_parameter(output, f"{prefix}__count", population_count)
                    for feature_name in diagnostics["event_features"]:
                        write_parameter(
                            output,
                            f"{prefix}__event__{feature_name}__valid_count",
                            int(feature_counts[(
                                selection_name, population, "event", feature_name
                            )].GetValue()),
                        )
                    for group_key, plot_group in diagnostics["plot_groups"].items():
                        for feature_name in plot_group["event_features"]:
                            write_parameter(
                                output,
                                f"{prefix}__plot_group__{group_key}__event__"
                                f"{feature_name}__valid_count",
                                int(group_event_feature_counts[(
                                    selection_name,
                                    population,
                                    group_key,
                                    feature_name,
                                )].GetValue()),
                            )
                    for track_key in diagnostics["track_types"]:
                        track_prefix = f"{prefix}__{track_key}"
                        for state in ("has_any", "has_converged", "has_best"):
                            value = int(availability_actions[(
                                selection_name, population, track_key, state
                            )].GetValue())
                            write_parameter(
                                output, f"{track_prefix}__{state}_count", value
                            )
                            write_parameter(
                                output,
                                f"{track_prefix}__{state}_fraction",
                                float(value) / population_count
                                if population_count else 0.0,
                            )
                        for feature_name in diagnostics["track_features"][track_key]:
                            write_parameter(
                                output,
                                f"{track_prefix}__{feature_name}__valid_count",
                                int(feature_counts[(
                                    selection_name,
                                    population,
                                    track_key,
                                    feature_name,
                                )].GetValue()),
                            )
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    del input_owner
    print(
        f"Wrote V7 flat-v2 diagnostics for {total} PassingMuonReco DQ events, "
        f"{len(selections)} selections and {len(diagnostics['track_types'])} track types"
    )


if __name__ == "__main__":
    main()
