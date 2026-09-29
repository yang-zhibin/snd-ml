#!/usr/bin/env python3
"""Build full-population V7 loose-failure SciFi-density histograms."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import ROOT

from make_passing_muon_binary_region_hist_v5 import (
    book_feature_set,
    configured_v5_features,
    validate_region_config,
)
from make_passing_muon_track_type_hist import (
    chain_for_group,
    clone_and_fold,
    define_shared_columns,
    load_yaml,
    mkdir_path,
    validate_hist_config,
    write_parameter,
)
from passing_muon_v7_scifi_density import (
    category_expression,
    configured_density_categories,
    configured_density_selection,
)


ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


def prepared_frame(chain, hist_config, study_config):
    frame = ROOT.RDataFrame(chain)
    frame = define_shared_columns(frame, hist_config.get("definitions", {}))
    return define_shared_columns(frame, study_config["derived_columns"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--mc-muon-up-metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--event-display-config", required=True)
    parser.add_argument("--mc-event-display-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()

    hist_config = load_yaml(args.hist_config)
    study = load_yaml(args.region_config)
    data_display = load_yaml(args.event_display_config)
    mc_display = load_yaml(args.mc_event_display_config)
    validate_hist_config(hist_config)
    validate_region_config(study)
    features = configured_v5_features(hist_config, study)
    categories = configured_density_categories(data_display, mc_display)
    selection = configured_density_selection(study, data_display, mc_display)
    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)

    metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    tree_name = str(study.get("tree_name", "sndData"))
    data_chain, _ = chain_for_group(
        metadata, str(study["groups"]["real_data"]), tree_name
    )
    data_frame = prepared_frame(data_chain, hist_config, study)

    mc_metadata = pd.read_csv(args.mc_muon_up_metadata, keep_default_na=False)
    # Keep every TChain alive for as long as its RDataFrame graph exists.
    # PyROOT's RDataFrame does not take Python ownership of the input TChain;
    # retaining only the frame leaves a dangling C++ tree pointer after the
    # loop variable is replaced and garbage-collected.
    mc_samples = OrderedDict()
    for production_key, production in study["mc_muon_up"]["productions"].items():
        chain, rows = chain_for_group(
            mc_metadata, str(production["group"]), tree_name
        )
        frame = prepared_frame(chain, hist_config, study).Filter(
            str(study["mc_muon_up"]["baseline_expression"]),
            f"density_{production_key}_baseline",
        )
        mc_samples[str(production_key)] = {
            "chain": chain,
            "rows": rows,
            "frame": frame,
        }

    detector_counts = [
        "count_veto", "count_scifi", "count_us", "count_ds",
        *(f"count_veto{i}" for i in range(1, 4)),
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
    ]
    negative_count_expression = " || ".join(
        f"{name} < 0" for name in detector_counts
    )
    event_actions = {
        ("data_total",): data_frame.Count(),
        ("data_invalid_count",): data_frame.Filter(
            negative_count_expression, "density_data_negative_detector_count"
        ).Count(),
    }
    for production, sample in mc_samples.items():
        frame = sample["frame"]
        event_actions[("mc_total", production)] = frame.Count()
        event_actions[("mc_invalid_count", production)] = frame.Filter(
            negative_count_expression,
            f"density_{production}_negative_detector_count",
        ).Count()
    category_actions = {}
    veto_actions = {}
    feature_count_actions = {}
    histogram_actions = {}

    def book_sample(sample, frame, category_key, expression, production=""):
        tag_parts = ["density", sample]
        if production:
            tag_parts.append(production)
        tag_parts.append(category_key)
        tag = "_".join(tag_parts)
        selected = frame.Filter(expression, f"{tag}_selected")
        sample_key = (sample, production, category_key)
        category_actions[sample_key] = selected.Count()
        for veto_key, veto_expression in selection["veto_categories"].items():
            veto_actions[(*sample_key, veto_key)] = selected.Filter(
                veto_expression, f"{tag}_{veto_key}"
            ).Count()
        counts, histograms = book_feature_set(selected, features, tag)
        for feature_name in features:
            feature_key = (*sample_key, feature_name)
            feature_count_actions[feature_key] = counts[feature_name]
            histogram_actions[feature_key] = histograms[feature_name]

    for category_key, category in categories.items():
        expression = category_expression(selection, category)
        book_sample("data", data_frame, category_key, expression)
        for production, sample in mc_samples.items():
            book_sample(
                "mc", sample["frame"], category_key, expression, production
            )

    ROOT.RDF.RunGraphs([
        *event_actions.values(),
        *category_actions.values(),
        *veto_actions.values(),
        *feature_count_actions.values(),
        *histogram_actions.values(),
    ])
    event_counts = {key: int(action.GetValue()) for key, action in event_actions.items()}
    category_counts = {
        key: int(action.GetValue()) for key, action in category_actions.items()
    }
    veto_counts = {key: int(action.GetValue()) for key, action in veto_actions.items()}
    feature_counts = {
        key: int(action.GetValue())
        for key, action in feature_count_actions.items()
    }

    data_total = event_counts[("data_total",)]
    if event_counts[("data_invalid_count",)]:
        raise RuntimeError(
            f"Found {event_counts[('data_invalid_count',)]} Run-8329 events "
            "with negative detector counts"
        )
    expected = study.get("expected", {})
    if expected and data_total != int(expected["real_data_dq"]):
        raise RuntimeError(
            f"Expected real_data_dq={expected['real_data_dq']}, "
            f"observed {data_total}"
        )
    for production in mc_samples:
        invalid = event_counts[("mc_invalid_count", production)]
        total = event_counts[("mc_total", production)]
        if invalid:
            raise RuntimeError(
                f"Found {invalid} negative-count MC events in {production}"
            )
        if total <= 0:
            raise RuntimeError(f"MC baseline is empty for {production}")

    for sample_key, count in category_counts.items():
        subgroup_sum = sum(
            veto_counts[(*sample_key, veto_key)]
            for veto_key in selection["veto_categories"]
        )
        if subgroup_sum != count:
            raise RuntimeError(
                f"Veto-union closure failed for {sample_key}: "
                f"subgroups={subgroup_sum}, category={count}"
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
            for category_key in categories:
                for feature_name, feature in features.items():
                    data_key = ("data", "", category_key, feature_name)
                    data_histogram = clone_and_fold(
                        histogram_actions[data_key], "data", feature
                    )
                    directory = mkdir_path(
                        output,
                        f"categories/{category_key}/features/{feature_name}",
                    )
                    directory.cd()
                    data_histogram.Write("data")

                    combined = None
                    for production in mc_samples:
                        key = ("mc", production, category_key, feature_name)
                        histogram = clone_and_fold(
                            histogram_actions[key], production, feature
                        )
                        production_directory = mkdir_path(
                            output,
                            f"mc_muon_up/{production}/categories/{category_key}/"
                            f"features/{feature_name}",
                        )
                        production_directory.cd()
                        histogram.Write("mc")
                        if combined is None:
                            combined = histogram.Clone(
                                f"mc_all_{category_key}_{feature_name}"
                            )
                            combined.SetDirectory(0)
                        else:
                            combined.Add(histogram)
                    directory.cd()
                    combined.Write("mc")

            write_parameter(output, "schema_version", "scifi_density_categories_v1")
            write_parameter(output, "working_point", selection["working_point"])
            write_parameter(
                output, "coverage_expression", selection["coverage_expression"]
            )
            write_parameter(
                output, "veto_union_expression", selection["veto_union_expression"]
            )
            write_parameter(output, "category_order", ",".join(categories))
            write_parameter(output, "feature_order", ",".join(features))
            write_parameter(output, "feature_count", len(features))
            write_parameter(output, "thresholds_are_cumulative", True)
            write_parameter(output, "uses_reconstructed_tracks", False)
            write_parameter(output, "data_total_count", data_total)
            write_parameter(
                output,
                "mc_total_count",
                sum(
                    event_counts[("mc_total", production)]
                    for production in mc_samples
                ),
            )
            write_parameter(output, "mc_production_order", ",".join(mc_samples))
            for production in mc_samples:
                write_parameter(
                    output,
                    f"mc__{production}__baseline_count",
                    event_counts[("mc_total", production)],
                )
            for category_key, category in categories.items():
                prefix = f"category__{category_key}"
                expression = category_expression(selection, category)
                data_key = ("data", "", category_key)
                write_parameter(output, f"{prefix}__label", category["label"])
                write_parameter(output, f"{prefix}__expression", expression)
                write_parameter(
                    output, f"{prefix}__data_count", category_counts[data_key]
                )
                mc_count = sum(
                    category_counts[("mc", production, category_key)]
                    for production in mc_samples
                )
                write_parameter(output, f"{prefix}__mc_count", mc_count)
                for veto_key in selection["veto_categories"]:
                    write_parameter(
                        output,
                        f"{prefix}__data__{veto_key}__count",
                        veto_counts[(*data_key, veto_key)],
                    )
                for production in mc_samples:
                    mc_key = ("mc", production, category_key)
                    write_parameter(
                        output,
                        f"{prefix}__mc__{production}__count",
                        category_counts[mc_key],
                    )
                    for veto_key in selection["veto_categories"]:
                        write_parameter(
                            output,
                            f"{prefix}__mc__{production}__{veto_key}__count",
                            veto_counts[(*mc_key, veto_key)],
                        )
                for feature_name in features:
                    write_parameter(
                        output,
                        f"{prefix}__data__{feature_name}__valid_count",
                        feature_counts[(*data_key, feature_name)],
                    )
                    write_parameter(
                        output,
                        f"{prefix}__mc__{feature_name}__valid_count",
                        sum(
                            feature_counts[(
                                "mc", production, category_key, feature_name
                            )]
                            for production in mc_samples
                        ),
                    )
            output.Write()
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(
        f"Wrote {len(categories)} cumulative SciFi-density categories and "
        f"{len(features)} features to {output_path}"
    )


if __name__ == "__main__":
    main()
