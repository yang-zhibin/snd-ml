#!/usr/bin/env python3
"""Build V8 Run-8329 passing-muon and no-Veto signal-region histograms."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import ROOT

from make_passing_muon_binary_region_hist_v5 import book_feature_set
from make_passing_muon_track_type_hist import (
    chain_for_group,
    clone_and_fold,
    define_shared_columns,
    load_yaml,
    mkdir_path,
    validate_hist_config,
    write_parameter,
)
from passing_muon_v8 import configured_features, load_v8_config, population_definitions


def metadata_luminosity(rows):
    field = "lumi_per_partition"
    if field not in rows:
        raise ValueError(f"Metadata lacks required luminosity column {field!r}")
    return float(pd.to_numeric(rows[field], errors="raise").sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--mc-muon-up-metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()

    overlay, base_config, base_selection = load_v8_config(args.region_config)
    hist_config = load_yaml(args.hist_config)
    validate_hist_config(hist_config)
    features = configured_features(hist_config, base_config)
    populations = population_definitions(overlay, base_selection)
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)

    tree_name = str(base_config.get("tree_name", "sndData"))
    metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    data_chain, data_rows = chain_for_group(
        metadata, str(base_config["groups"]["real_data"]), tree_name
    )
    data_frame = ROOT.RDataFrame(data_chain)
    data_frame = define_shared_columns(data_frame, hist_config.get("definitions", {}))
    data_frame = define_shared_columns(data_frame, base_config["derived_columns"])

    mc_metadata = pd.read_csv(args.mc_muon_up_metadata, keep_default_na=False)
    mc_samples = OrderedDict()
    for key, definition in base_config["mc_muon_up"]["productions"].items():
        chain, rows = chain_for_group(mc_metadata, str(definition["group"]), tree_name)
        frame = ROOT.RDataFrame(chain)
        frame = define_shared_columns(frame, hist_config.get("definitions", {}))
        frame = define_shared_columns(frame, base_config["derived_columns"])
        frame = frame.Filter(
            str(base_config["mc_muon_up"]["baseline_expression"]),
            f"v8_{key}_baseline",
        )
        mc_samples[key] = {
            "definition": definition,
            "chain": chain,
            "rows": rows,
            "frame": frame,
        }

    total_actions = {"data": data_frame.Count()}
    population_actions = {}
    feature_count_actions = {}
    histogram_actions = {}

    def book(sample, frame):
        for population, definition in populations.items():
            selected = frame.Filter(
                definition["expression"], f"v8_{sample}_{population}"
            )
            population_actions[(sample, population)] = selected.Count()
            counts, histograms = book_feature_set(
                selected, features, f"v8_{sample}_{population}"
            )
            for feature in features:
                feature_count_actions[(sample, population, feature)] = counts[feature]
                histogram_actions[(sample, population, feature)] = histograms[feature]

    book("data", data_frame)
    for sample, values in mc_samples.items():
        total_actions[sample] = values["frame"].Count()
        book(sample, values["frame"])

    ROOT.RDF.RunGraphs([
        *total_actions.values(),
        *population_actions.values(),
        *feature_count_actions.values(),
        *histogram_actions.values(),
    ])
    totals = {key: int(action.GetValue()) for key, action in total_actions.items()}
    counts = {key: int(action.GetValue()) for key, action in population_actions.items()}
    for sample, total in totals.items():
        observed = sum(counts[(sample, population)] for population in populations)
        if observed != total:
            raise RuntimeError(
                f"V8 population closure failed for {sample}: {observed} != {total}"
            )
    expected = int(base_config.get("expected", {}).get("real_data_dq", totals["data"]))
    if totals["data"] != expected:
        raise RuntimeError(f"Expected {expected} data events, observed {totals['data']}")

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
            raise OSError(f"Cannot create {temporary}")
        try:
            for (sample, population, feature), result in histogram_actions.items():
                histogram = clone_and_fold(result, population, features[feature])
                root = "data" if sample == "data" else f"mc_muon_up/{sample}"
                directory = mkdir_path(
                    output, f"{root}/populations/{population}/features/{feature}"
                )
                directory.cd()
                histogram.Write("histogram")

            aggregate_key = str(base_config["mc_muon_up"]["aggregate"]["key"])
            for population in populations:
                for feature in features:
                    combined = None
                    for sample in mc_samples:
                        histogram = clone_and_fold(
                            histogram_actions[(sample, population, feature)],
                            f"{aggregate_key}_{population}_{feature}",
                            features[feature],
                        )
                        if combined is None:
                            combined = histogram
                            combined.SetDirectory(0)
                        else:
                            combined.Add(histogram)
                    directory = mkdir_path(
                        output,
                        f"mc_muon_up/{aggregate_key}/populations/{population}/"
                        f"features/{feature}",
                    )
                    directory.cd()
                    combined.Write("histogram")

            write_parameter(output, "study_version", overlay["version"])
            write_parameter(output, "run", int(overlay["run"]))
            write_parameter(output, "data_year", int(overlay["data"]["year"]))
            write_parameter(output, "data_label", overlay["data"]["label"])
            write_parameter(output, "data_event_count", totals["data"])
            write_parameter(output, "data_luminosity_fb", metadata_luminosity(data_rows))
            write_parameter(output, "data_partition_count", len(data_rows))
            write_parameter(output, "selection_key", overlay["selection"]["key"])
            write_parameter(output, "selection_label", overlay["selection"]["label"])
            write_parameter(output, "selection_expression", base_selection["expression"])
            write_parameter(output, "population_order", ",".join(populations))
            write_parameter(output, "feature_order", ",".join(features))
            for population, definition in populations.items():
                prefix = f"population__{population}"
                write_parameter(output, f"{prefix}__label", definition["label"])
                write_parameter(output, f"{prefix}__expression", definition["expression"])
                write_parameter(output, f"{prefix}__data_count", counts[("data", population)])
                write_parameter(
                    output,
                    f"{prefix}__mc_count",
                    sum(counts[(sample, population)] for sample in mc_samples),
                )
                for feature in features:
                    write_parameter(
                        output,
                        f"{prefix}__data__{feature}__valid_count",
                        int(feature_count_actions[("data", population, feature)].GetValue()),
                    )
            write_parameter(output, "population_closure", True)
            write_parameter(output, "mc_muon_up_production_order", ",".join(mc_samples))
            write_parameter(output, "mc_muon_up_aggregate_key", aggregate_key)
            write_parameter(
                output, "mc_muon_up_event_count", sum(totals[key] for key in mc_samples)
            )
            write_parameter(
                output,
                "mc_muon_up_declared_event_count",
                int(sum(pd.to_numeric(s["rows"]["n_events"]).sum() for s in mc_samples.values())),
            )
            write_parameter(
                output,
                "mc_muon_up_luminosity_fb",
                sum(metadata_luminosity(s["rows"]) for s in mc_samples.values()),
            )
            for key, sample in mc_samples.items():
                prefix = f"mc_muon_up__{key}"
                definition = sample["definition"]
                write_parameter(output, f"{prefix}__label", definition["label"])
                write_parameter(output, f"{prefix}__catalog", definition["source_partition"])
                write_parameter(output, f"{prefix}__file_count", len(sample["rows"]))
                write_parameter(
                    output,
                    f"{prefix}__declared_event_count",
                    int(pd.to_numeric(sample["rows"]["n_events"]).sum()),
                )
                write_parameter(output, f"{prefix}__baseline_event_count", totals[key])
                write_parameter(
                    output, f"{prefix}__luminosity_fb", metadata_luminosity(sample["rows"])
                )
                for population in populations:
                    write_parameter(
                        output,
                        f"{prefix}__{population}__count",
                        counts[(key, population)],
                    )
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"Wrote V8 detector study with {len(features)} features, "
        f"{totals['data']} data events, and "
        f"{sum(totals[key] for key in mc_samples)} MC baseline events"
    )


if __name__ == "__main__":
    main()
