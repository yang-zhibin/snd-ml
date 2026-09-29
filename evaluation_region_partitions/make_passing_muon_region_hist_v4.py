#!/usr/bin/env python3
"""Build run-8329 v4 marginal and joint-region comparison histograms."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import ROOT

from make_passing_muon_track_type_hist import (
    book_histogram,
    chain_for_group,
    clone_and_fold,
    configured_features,
    define_shared_columns,
    feature_frame_and_columns,
    load_yaml,
    mkdir_path,
    validate_hist_config,
    write_parameter,
)
from passing_muon_v4_features import expanded_v4_hist_config


POPULATIONS = {
    "passing": "passing_muon_reco",
    "missing": "no_passing_muon_reco",
}


def validate_region_config(config):
    axes = config.get("axes")
    if not isinstance(axes, dict) or tuple(axes) != (
        "veto_count", "veto_time", "us_count"
    ):
        raise ValueError(
            "Region config axes must be ordered as veto_count, veto_time, us_count"
        )
    for axis_name, axis in axes.items():
        categories = axis.get("categories")
        if not isinstance(categories, dict) or len(categories) != 4:
            raise ValueError(f"Axis {axis_name!r} requires exactly four categories")
        if axis.get("reference") not in categories:
            raise ValueError(f"Axis {axis_name!r} has an unknown reference")
        for category_name, category in categories.items():
            if not str(category.get("label", "")).strip():
                raise ValueError(f"Category {category_name!r} lacks a label")
            if not str(category.get("expression", "")).strip():
                raise ValueError(f"Category {category_name!r} lacks an expression")
    count_states = {
        str(category.get("state", ""))
        for category in axes["veto_count"]["categories"].values()
    }
    time_states = {
        str(category.get("state", ""))
        for category in axes["veto_time"]["categories"].values()
    }
    if count_states != {"none", "positive"} or time_states != {
        "none", "positive"
    }:
        raise ValueError("Veto count/time categories require none/positive states")
    audits = axes["veto_time"].get("audit_categories")
    if not isinstance(audits, dict) or set(audits) != {
        "veto_time_invalid", "veto_time_negative"
    }:
        raise ValueError("Veto-time invalid and negative audits are required")
    for name, category in audits.items():
        if not str(category.get("expression", "")).strip():
            raise ValueError(f"Audit category {name!r} lacks an expression")
    if list(config.get("joint_axes", [])) != list(axes):
        raise ValueError("joint_axes must match the configured axis order")


def category_expression(category):
    return str(category["expression"]).strip()


def joined(parts):
    return " && ".join(f"({part})" for part in parts)


def assignment_expression(categories):
    return " + ".join(
        f"(({category_expression(category)}) ? 1 : 0)"
        for category in categories.values()
    )


def joint_regions(config):
    axes = config["axes"]
    result = {}
    for count_name, count_category in axes["veto_count"]["categories"].items():
        count_state = str(count_category["state"])
        for time_name, time_category in axes["veto_time"]["categories"].items():
            if str(time_category["state"]) != count_state:
                continue
            for us_name, us_category in axes["us_count"]["categories"].items():
                key = f"{count_name}__{time_name}__{us_name}"
                result[key] = joined(
                    [
                        category_expression(count_category),
                        category_expression(time_category),
                        category_expression(us_category),
                    ]
                )
    if len(result) != 40:
        raise ValueError(f"Expected 40 physical joint regions, found {len(result)}")
    selected = list(config.get("selected_joint_regions", []))
    if len(selected) != len(set(selected)):
        raise ValueError("selected_joint_regions contains duplicates")
    unknown = set(selected) - set(result)
    if unknown:
        raise ValueError(f"Unknown selected joint regions: {sorted(unknown)}")
    return result


def book_feature_set(frame, features, tag):
    counts = {}
    histograms = {}
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
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--missing-metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()

    hist_config = load_yaml(args.hist_config)
    config = load_yaml(args.region_config)
    hist_config = expanded_v4_hist_config(hist_config, config)
    validate_hist_config(hist_config)
    validate_region_config(config)
    regions = joint_regions(config)
    features = configured_features(hist_config)
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)

    paired_metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    missing_metadata = pd.read_csv(args.missing_metadata, keep_default_na=False)
    tree_name = str(config.get("tree_name", "sndData"))
    passing_chain, passing_rows = chain_for_group(
        paired_metadata, str(config["groups"]["passing"]), tree_name
    )
    missing_chain, missing_rows = chain_for_group(
        missing_metadata, str(config["groups"]["missing"]), tree_name
    )
    frames = {
        "passing": define_shared_columns(
            ROOT.RDataFrame(passing_chain), hist_config.get("definitions", {})
        ),
        "missing": define_shared_columns(
            ROOT.RDataFrame(missing_chain), hist_config.get("definitions", {})
        ),
    }

    actions = {}
    marginal_feature_counts = {}
    marginal_histograms = {}
    joint_feature_counts = {}
    joint_histograms = {}
    axes = config["axes"]
    selected_regions = list(config.get("selected_joint_regions", []))

    for population, frame in frames.items():
        actions[("total", population)] = frame.Count()
        for axis_name, axis in axes.items():
            closure_categories = dict(axis["categories"])
            closure_categories.update(axis.get("audit_categories", {}))
            assignment = assignment_expression(closure_categories)
            actions[("bad_assignment", axis_name, population)] = frame.Filter(
                f"({assignment}) != 1",
                f"{axis_name}_{population}_nonexclusive_or_unassigned",
            ).Count()
            for category_name, category in closure_categories.items():
                category_frame = frame.Filter(
                    category_expression(category),
                    f"{axis_name}_{category_name}_{population}",
                )
                actions[("axis", axis_name, category_name, population)] = (
                    category_frame.Count()
                )
                if category_name not in axis["categories"]:
                    continue
                feature_counts, histograms = book_feature_set(
                    category_frame,
                    features,
                    f"marginal_{axis_name}_{category_name}_{population}",
                )
                for feature_name in features:
                    key = (axis_name, category_name, population, feature_name)
                    marginal_feature_counts[key] = feature_counts[feature_name]
                    marginal_histograms[key] = histograms[feature_name]

        for region_name, expression in regions.items():
            region_frame = frame.Filter(
                expression, f"joint_{region_name}_{population}"
            )
            actions[("joint", region_name, population)] = region_frame.Count()
            if region_name not in selected_regions:
                continue
            feature_counts, histograms = book_feature_set(
                region_frame, features, f"joint_{region_name}_{population}"
            )
            for feature_name in features:
                key = (region_name, population, feature_name)
                joint_feature_counts[key] = feature_counts[feature_name]
                joint_histograms[key] = histograms[feature_name]

    ROOT.RDF.RunGraphs(
        list(actions.values())
        + list(marginal_feature_counts.values())
        + list(marginal_histograms.values())
        + list(joint_feature_counts.values())
        + list(joint_histograms.values())
    )
    counts = {key: int(action.GetValue()) for key, action in actions.items()}

    expected = config.get("expected", {})
    passing_total = counts[("total", "passing")]
    missing_total = counts[("total", "missing")]
    real_total = passing_total + missing_total
    if expected:
        observed = {
            "passing_muon_dq": passing_total,
            "no_passing_muon_reco": missing_total,
            "real_data_dq": real_total,
        }
        for name, value in observed.items():
            if value != int(expected[name]):
                raise RuntimeError(
                    f"Expected {name}={expected[name]}, observed {value}"
                )

    for population in POPULATIONS:
        for axis_name in axes:
            bad = counts[("bad_assignment", axis_name, population)]
            if bad:
                raise RuntimeError(
                    f"Axis {axis_name} has {bad} non-exclusive/unassigned "
                    f"{population} events"
                )
        eligible = sum(
            counts[("axis", "veto_time", category, population)]
            for category in axes["veto_time"]["categories"]
        )
        joint_total = sum(
            counts[("joint", region, population)] for region in regions
        )
        if joint_total != eligible:
            raise RuntimeError(
                f"Joint-region closure failed for {population}: "
                f"{joint_total} != {eligible}"
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
            for key, result in marginal_histograms.items():
                axis_name, category_name, population, feature_name = key
                histogram = clone_and_fold(
                    result, POPULATIONS[population], features[feature_name]
                )
                directory = mkdir_path(
                    output,
                    f"marginal/{axis_name}/{category_name}/{feature_name}",
                )
                directory.cd()
                histogram.Write(POPULATIONS[population])
            for key, result in joint_histograms.items():
                region_name, population, feature_name = key
                histogram = clone_and_fold(
                    result, POPULATIONS[population], features[feature_name]
                )
                directory = mkdir_path(
                    output, f"joint_regions/{region_name}/{feature_name}"
                )
                directory.cd()
                histogram.Write(POPULATIONS[population])

            write_parameter(output, "study_version", config["version"])
            write_parameter(output, "run", int(config["run"]))
            write_parameter(output, "passing_count", passing_total)
            write_parameter(output, "missing_count", missing_total)
            write_parameter(output, "real_data_dq_count", real_total)
            write_parameter(output, "population_closure", True)
            write_parameter(output, "feature_order", ",".join(features))
            write_parameter(output, "axis_order", ",".join(axes))
            write_parameter(output, "joint_region_order", ",".join(regions))
            write_parameter(
                output, "selected_joint_region_order", ",".join(selected_regions)
            )
            write_parameter(output, "joint_region_count", len(regions))
            write_parameter(output, "passing_partition_count", len(passing_rows))
            write_parameter(output, "missing_partition_count", len(missing_rows))
            write_parameter(
                output,
                "missing_event_semantics",
                "RealDataDQ minus PassingMuonReco",
            )
            for axis_name, axis in axes.items():
                write_parameter(output, f"axis__{axis_name}__label", axis["label"])
                write_parameter(
                    output,
                    f"axis__{axis_name}__category_order",
                    ",".join(axis["categories"]),
                )
                write_parameter(
                    output,
                    f"axis__{axis_name}__reference",
                    axis["reference"],
                )
                closure_categories = dict(axis["categories"])
                closure_categories.update(axis.get("audit_categories", {}))
                write_parameter(
                    output,
                    f"axis__{axis_name}__closure_category_order",
                    ",".join(closure_categories),
                )
                for category_name, category in closure_categories.items():
                    write_parameter(
                        output,
                        f"axis__{axis_name}__{category_name}__label",
                        category["label"],
                    )
                    write_parameter(
                        output,
                        f"axis__{axis_name}__{category_name}__expression",
                        category_expression(category),
                    )
                    for population in POPULATIONS:
                        write_parameter(
                            output,
                            f"axis__{axis_name}__{category_name}__"
                            f"{population}__count",
                            counts[("axis", axis_name, category_name, population)],
                        )
                for population in POPULATIONS:
                    write_parameter(
                        output,
                        f"axis__{axis_name}__{population}__exclusive_closure",
                        counts[("bad_assignment", axis_name, population)] == 0,
                    )
            for region_name, expression in regions.items():
                write_parameter(
                    output, f"joint__{region_name}__expression", expression
                )
                for population in POPULATIONS:
                    write_parameter(
                        output,
                        f"joint__{region_name}__{population}__count",
                        counts[("joint", region_name, population)],
                    )
            for key, action in marginal_feature_counts.items():
                axis_name, category_name, population, feature_name = key
                write_parameter(
                    output,
                    f"selected__marginal__{axis_name}__{category_name}__"
                    f"{population}__{feature_name}",
                    int(action.GetValue()),
                )
            for key, action in joint_feature_counts.items():
                region_name, population, feature_name = key
                write_parameter(
                    output,
                    f"selected__joint__{region_name}__{population}__{feature_name}",
                    int(action.GetValue()),
                )
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    passing_eligible = sum(
        counts[("axis", "veto_time", category, "passing")]
        for category in axes["veto_time"]["categories"]
    )
    missing_eligible = sum(
        counts[("axis", "veto_time", category, "missing")]
        for category in axes["veto_time"]["categories"]
    )
    print(
        f"Wrote {len(features)} features, 12 marginal categories and "
        f"{len(regions)} joint-region counts; DQ={real_total}, "
        f"joint-eligible={passing_eligible + missing_eligible}"
    )


if __name__ == "__main__":
    main()
