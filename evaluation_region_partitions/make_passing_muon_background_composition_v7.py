#!/usr/bin/env python3
"""Build raw-count background-composition histograms for PassingMuonReco V7."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import ROOT

from make_passing_muon_binary_region_hist_v5 import (
    configured_selections,
    validate_region_config,
)
from make_passing_muon_reco_diagnostics_v7 import (
    book_features,
    dataframe_from_manifest,
)
from make_passing_muon_track_type_hist import (
    clone_and_fold,
    define_shared_columns,
    load_yaml,
    mkdir_path,
    write_parameter,
)
from passing_muon_background_composition_v7 import (
    class_expressions,
    configured_background_composition,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-manifest", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    config = load_yaml(args.region_config)
    validate_region_config(config)
    composition = configured_background_composition(config)
    selections = configured_selections(config)
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)
    tree_name = str(config.get("tree_name", "sndData"))
    frame, rows, input_owner = dataframe_from_manifest(
        args.track_manifest,
        {
            "input_group": composition["input_group"],
            "required_schema": composition["required_schema"],
        },
        tree_name,
        int(config["run"]),
    )
    columns = {str(value) for value in frame.GetColumnNames()}
    prefix = composition["track_prefix"]
    required = {
        "reco_track_n_all",
        f"{prefix}_best_start_z",
        f"{prefix}_best_chi2ndf",
        f"{prefix}_best_n_measurements",
        f"{prefix}_best_n_unique_planes",
    }
    required.update(
        f"{track_prefix}_has_best"
        for track_prefix in composition["required_track_prefixes"]
    )
    required.update({
        "count_veto", "count_scifi", "count_us", "count_ds",
        "scifi_n_active_stations_xy",
        *(f"count_veto{i}" for i in range(1, 4)),
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
    })
    missing = required - columns
    if missing:
        raise RuntimeError(f"Composition inputs lack branches: {sorted(missing)}")
    frame = define_shared_columns(frame, config["derived_columns"])

    actions = {("total",): frame.Count()}
    expressions = {}
    feature_counts = {}
    histogram_actions = {}
    for selection_name, selection in selections.items():
        inclusive_counts, inclusive_histograms = book_features(
            frame,
            composition["features"],
            f"composition_{selection_name}_inclusive",
        )
        for feature_name in composition["features"]:
            key = (selection_name, "inclusive", feature_name)
            feature_counts[key] = inclusive_counts[feature_name]
            histogram_actions[key] = inclusive_histograms[feature_name]
        configured_classes = class_expressions(selection["expression"], composition)
        expressions[selection_name] = configured_classes
        for class_name, expression in configured_classes.items():
            class_frame = frame.Filter(
                expression, f"{selection_name}_{class_name}"
            )
            actions[(selection_name, class_name)] = class_frame.Count()
            counts, histograms = book_features(
                class_frame,
                composition["features"],
                f"composition_{selection_name}_{class_name}",
            )
            for feature_name in composition["features"]:
                key = (selection_name, class_name, feature_name)
                feature_counts[key] = counts[feature_name]
                histogram_actions[key] = histograms[feature_name]

    ROOT.RDF.RunGraphs(
        list(actions.values())
        + list(feature_counts.values())
        + list(histogram_actions.values())
    )
    counts = {key: int(action.GetValue()) for key, action in actions.items()}
    total = counts[("total",)]
    declared = int(pd.to_numeric(rows["n_events"]).sum())
    if total != declared:
        raise RuntimeError(f"Composition manifest/tree mismatch: {declared} != {total}")
    for selection_name in selections:
        class_sum = sum(
            counts[(selection_name, class_name)]
            for class_name in composition["class_order"]
        )
        if class_sum != total:
            raise RuntimeError(
                f"Composition closure failed for {selection_name}: {class_sum} != {total}"
            )
        for feature_name in composition["features"]:
            inclusive_valid = int(feature_counts[(
                selection_name, "inclusive", feature_name
            )].GetValue())
            class_valid = sum(
                int(feature_counts[(
                    selection_name, class_name, feature_name
                )].GetValue())
                for class_name in composition["class_order"]
            )
            if class_valid != inclusive_valid:
                raise RuntimeError(
                    f"Feature closure failed for {selection_name}/{feature_name}: "
                    f"{class_valid} != {inclusive_valid}"
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
                selection_name, class_name, feature_name = key
                histogram = clone_and_fold(
                    result, class_name, composition["features"][feature_name]
                )
                directory = mkdir_path(
                    output,
                    f"selections/{selection_name}/features/{feature_name}",
                )
                directory.cd()
                histogram.Write(class_name)

            write_parameter(output, "study_version", config["version"])
            write_parameter(output, "run", int(config["run"]))
            write_parameter(output, "analysis_scope", composition["scope"])
            write_parameter(output, "track_feature_schema", "flat_v2")
            write_parameter(output, "track_input_group", composition["input_group"])
            write_parameter(output, "track_definition_key", composition["track_key"])
            write_parameter(output, "track_definition_prefix", composition["track_prefix"])
            write_parameter(output, "track_definition_valid_expression", composition["track_valid_expression"])
            write_parameter(output, "track_definition_label", composition["track_label"])
            write_parameter(output, "input_event_count", total)
            write_parameter(output, "input_file_count", len(rows))
            write_parameter(output, "selection_order", ",".join(selections))
            write_parameter(output, "class_order", ",".join(composition["class_order"]))
            write_parameter(output, "feature_order", ",".join(composition["features"]))
            write_parameter(output, "section_order", ",".join(composition["sections"]))
            for class_name, label in composition["class_labels"].items():
                write_parameter(output, f"class__{class_name}__label", label)
            for section_name, section in composition["sections"].items():
                write_parameter(output, f"section__{section_name}__label", section["label"])
                write_parameter(
                    output, f"section__{section_name}__feature_order",
                    ",".join(section["features"]),
                )
            for selection_name in selections:
                write_parameter(
                    output, f"selection__{selection_name}__total_count", total
                )
                for feature_name in composition["features"]:
                    write_parameter(
                        output,
                        f"selection__{selection_name}__inclusive__feature__"
                        f"{feature_name}__valid_count",
                        int(feature_counts[(
                            selection_name, "inclusive", feature_name
                        )].GetValue()),
                    )
                for class_name in composition["class_order"]:
                    value = counts[(selection_name, class_name)]
                    prefix = f"selection__{selection_name}__class__{class_name}"
                    write_parameter(output, f"{prefix}__expression", expressions[selection_name][class_name])
                    write_parameter(output, f"{prefix}__count", value)
                    write_parameter(output, f"{prefix}__fraction", value / total if total else 0.0)
                    for feature_name in composition["features"]:
                        write_parameter(
                            output,
                            f"{prefix}__feature__{feature_name}__valid_count",
                            int(feature_counts[(selection_name, class_name, feature_name)].GetValue()),
                        )
                write_parameter(output, f"selection__{selection_name}__class_closure", 1)
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    del input_owner
    print(
        f"Wrote PassingMuonReco-only composition for {total} events, "
        f"{len(selections)} coverage definitions and "
        f"{len(composition['features'])} features"
    )


if __name__ == "__main__":
    main()
