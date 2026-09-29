#!/usr/bin/env python3
"""Build run-8329 detector-only through-going-muon study histograms."""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from collections import OrderedDict
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


POPULATIONS = OrderedDict((
    ("candidate", "through_going_candidate"),
    ("complement", "shower_enriched_complement"),
))
MC_POPULATIONS = OrderedDict((
    ("candidate", "candidate"),
    ("complement", "complement"),
))
MAXIMUM_FIELDS = OrderedDict((
    ("veto_max", "count_veto"),
    ("scifi_max", "count_scifi"),
    ("us_max", "count_us"),
    ("ds_max", "count_ds"),
))


def joined(parts):
    return " && ".join(f"({part})" for part in parts)


def coverage_components(working_point):
    components = OrderedDict((
        ("all_veto_stations", "has_all_veto"),
        (
            "scifi_xy_station_coverage",
            "scifi_n_active_stations_xy >= "
            f"{int(working_point['scifi_min_active_xy_stations'])}",
        ),
        ("all_us_stations", "has_all_us"),
        (
            "ds_station_coverage",
            f"n_active_ds >= {int(working_point['ds_min_active_stations'])}",
        ),
    ))
    per_station_limits = (
        ("scifi_station_multiplicity", "scifi", 5),
        ("veto_station_multiplicity", "veto", 3),
        ("us_station_multiplicity", "us", 5),
    )
    for component, detector, station_count in per_station_limits:
        exclusive_field = f"{detector}_max_hits_per_station_exclusive"
        inclusive_field = f"{detector}_max_hits_per_station_inclusive"
        exclusive = working_point.get(exclusive_field)
        inclusive = working_point.get(inclusive_field)
        if exclusive is not None and inclusive is not None:
            raise ValueError(
                f"Configure only one of {exclusive_field} and {inclusive_field}"
            )
        if exclusive is not None:
            components[component] = joined(
                f"count_{detector}{station} < {int(exclusive)}"
                for station in range(1, station_count + 1)
            )
        elif inclusive is not None:
            components[component] = joined(
                f"count_{detector}{station} <= {int(inclusive)}"
                for station in range(1, station_count + 1)
            )
        total = working_point.get(f"{detector}_max_hits_total_inclusive")
        if total is not None:
            components[f"{detector}_total_multiplicity"] = (
                f"count_{detector} <= {int(total)}"
            )
    return components


def configured_selections(config):
    """Return enabled topology and final detector-only working points."""
    result = OrderedDict()
    coverage = config["coverage_working_points"]
    for name, definition in coverage.items():
        components = coverage_components(definition)
        result[name] = {
            "kind": "coverage",
            "label": str(definition["label"]),
            "candidate_label": "Through-going topology candidate",
            "complement_label": "Topology-failure complement",
            "components": components,
            "expression": joined(components.values()),
        }
    for name, definition in config.get("selection_working_points", {}).items():
        if not bool(definition.get("enabled", False)):
            continue
        coverage_name = str(definition["coverage"])
        components = coverage_components(coverage[coverage_name])
        for field, variable in MAXIMUM_FIELDS.items():
            threshold = definition.get(field)
            if threshold is not None:
                components[field] = f"{variable} <= {float(threshold):.17g}"
        result[name] = {
            "kind": "selection",
            "label": str(definition["label"]),
            "candidate_label": "Clean through-going muon candidate",
            "complement_label": "Shower-enriched complement",
            "components": components,
            "expression": joined(components.values()),
        }
    return result


def configured_v5_features(hist_config, study_config):
    raw = {
        "defaults": dict(hist_config.get("defaults", {})),
        "features": OrderedDict(hist_config["features"]),
    }
    for name, feature in study_config.get("diagnostic_features", {}).items():
        if name in raw["features"]:
            raise ValueError(f"Duplicate common/v5 feature {name!r}")
        raw["features"][name] = feature
    validate_hist_config(raw)
    return configured_features(raw)


def configured_count_diagnostics(config, features, selections):
    """Validate optional data-only detector-family count diagnostics."""
    block = config.get("count_diagnostics")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("count_diagnostics must be a mapping")
    count_features = OrderedDict(features)
    raw_extra = block.get("extra_features", {})
    if not isinstance(raw_extra, dict):
        raise ValueError("count_diagnostics.extra_features must be a mapping")
    for name, definition in raw_extra.items():
        name = str(name)
        if name in count_features:
            raise ValueError(f"Duplicate count-diagnostic feature {name!r}")
        count_features[name] = dict(definition)
    validate_hist_config({"features": count_features})
    count_features = configured_features({"features": count_features})

    overview = [str(value) for value in block.get("overview_features", [])]
    if not overview or len(overview) != len(set(overview)):
        raise ValueError(
            "count_diagnostics.overview_features must be non-empty and unique"
        )
    missing = set(overview) - set(count_features)
    if missing:
        raise ValueError(f"Unknown count overview features: {sorted(missing)}")

    raw_families = block.get("detector_families")
    if not isinstance(raw_families, dict) or not raw_families:
        raise ValueError("count_diagnostics.detector_families must be non-empty")
    families = OrderedDict()
    for family_name, raw_family in raw_families.items():
        family_name = str(family_name)
        if not re.fullmatch(r"[A-Za-z0-9_]+", family_name):
            raise ValueError(f"Unsafe count-diagnostic family {family_name!r}")
        if not isinstance(raw_family, dict):
            raise ValueError(f"Count family {family_name!r} must be a mapping")
        label = str(raw_family.get("label", "")).strip()
        family_features = [str(value) for value in raw_family.get("features", [])]
        if not label or not family_features or len(family_features) != len(
            set(family_features)
        ):
            raise ValueError(
                f"Count family {family_name!r} requires a label and unique features"
            )
        unknown_features = set(family_features) - set(count_features)
        if unknown_features:
            raise ValueError(
                f"Count family {family_name!r} has unknown features: "
                f"{sorted(unknown_features)}"
            )
        for feature_name in family_features:
            if int(count_features[feature_name].get("dimension", 1)) != 1:
                raise ValueError(
                    f"Count feature {feature_name!r} must be one-dimensional"
                )
        raw_remove = raw_family.get("remove_components")
        if not isinstance(raw_remove, dict) or set(raw_remove) != set(selections):
            raise ValueError(
                f"Count family {family_name!r} must define remove_components "
                "for every configured selection"
            )
        remove_components = OrderedDict()
        for selection_name, selection in selections.items():
            removed = [str(value) for value in raw_remove[selection_name]]
            if not removed or len(removed) != len(set(removed)):
                raise ValueError(
                    f"{family_name}.{selection_name} requires unique removed components"
                )
            unknown_components = set(removed) - set(selection["components"])
            if unknown_components:
                raise ValueError(
                    f"{family_name}.{selection_name} references unknown components: "
                    f"{sorted(unknown_components)}"
                )
            if len(removed) == len(selection["components"]):
                raise ValueError(
                    f"{family_name}.{selection_name} removes every selection component"
                )
            remove_components[selection_name] = removed
        families[family_name] = {
            "label": label,
            "features": family_features,
            "remove_components": remove_components,
        }
    return {
        "description": str(block.get("description", "")).strip(),
        "overview_features": overview,
        "features": count_features,
        "families": families,
    }


def configured_mc_aggregate(config):
    aggregate = config["mc_muon_up"].get("aggregate", {}) or {}
    if not aggregate:
        return None
    if str(aggregate.get("mode", "")) != "unweighted_event_sum":
        raise ValueError("Only MC aggregate mode 'unweighted_event_sum' is supported")
    key = str(aggregate.get("key", "")).strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_]+", key):
        raise ValueError(f"Unsafe MC aggregate key {key!r}")
    if key in config["mc_muon_up"]["productions"]:
        raise ValueError(f"MC aggregate key {key!r} duplicates a production key")
    return aggregate


def validate_region_config(config):
    if not str(config.get("groups", {}).get("real_data", "")).strip():
        raise ValueError("V5 config requires groups.real_data")
    mc_config = config.get("mc_muon_up")
    if not isinstance(mc_config, dict):
        raise ValueError("V5 config requires mc_muon_up")
    for field in (
        "source_metadata", "subfolder", "metadata_csv", "validation_csv",
        "baseline_expression", "productions",
    ):
        if not str(mc_config.get(field, "")).strip():
            raise ValueError(f"V5 mc_muon_up requires {field}")
    productions = mc_config["productions"]
    if not isinstance(productions, dict) or not productions:
        raise ValueError("V5 mc_muon_up.productions must be a non-empty mapping")
    groups, source_partitions = set(), set()
    for key, definition in productions.items():
        if not re.match(r"^[A-Za-z0-9_]+$", str(key)):
            raise ValueError(f"Unsafe MC production key {key!r}")
        for field in ("source_partition", "group", "label"):
            if not str(definition.get(field, "")).strip():
                raise ValueError(f"MC production {key!r} requires {field}")
        groups.add(str(definition["group"]))
        source_partitions.add(str(definition["source_partition"]))
    if len(groups) != len(productions) or len(source_partitions) != len(productions):
        raise ValueError("MC production groups and source partitions must be unique")
    derived = config.get("derived_columns")
    if not isinstance(derived, dict) or set(derived) != {
        "has_all_veto", "has_all_us", "n_active_ds"
    }:
        raise ValueError(
            "derived_columns must define has_all_veto, has_all_us, n_active_ds"
        )
    serialized = " ".join(str(value).lower() for value in derived.values())
    if "reco" in serialized or "track" in serialized:
        raise ValueError("Detector-only derived columns cannot use track information")
    coverage = config.get("coverage_working_points")
    if not isinstance(coverage, dict) or not coverage:
        raise ValueError("At least one coverage_working_point is required")
    for name, definition in coverage.items():
        for field, upper in (
            ("scifi_min_active_xy_stations", 5),
            ("ds_min_active_stations", 4),
        ):
            value = int(definition[field])
            if value < 1 or value > upper:
                raise ValueError(f"{name}.{field}={value} is outside 1..{upper}")
        if not str(definition.get("label", "")).strip():
            raise ValueError(f"Coverage point {name!r} lacks a label")
        for detector in ("scifi", "veto", "us"):
            fields = (
                f"{detector}_max_hits_per_station_exclusive",
                f"{detector}_max_hits_per_station_inclusive",
                f"{detector}_max_hits_total_inclusive",
            )
            if (definition.get(fields[0]) is not None
                    and definition.get(fields[1]) is not None):
                raise ValueError(
                    f"{name} cannot configure both {fields[0]} and {fields[1]}"
                )
            for field in fields:
                value = definition.get(field)
                if value is not None and int(value) <= 0:
                    raise ValueError(
                        f"{name}.{field} must be a positive integer"
                    )
    for name, definition in config.get("selection_working_points", {}).items():
        if str(definition.get("coverage")) not in coverage:
            raise ValueError(f"Selection {name!r} references unknown coverage")
        for field in MAXIMUM_FIELDS:
            value = definition.get(field)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise ValueError(f"Selection {name!r} has invalid {field}")
        if bool(definition.get("enabled", False)) and all(
            definition.get(field) is None for field in MAXIMUM_FIELDS
        ):
            raise ValueError(
                f"Enabled final selection {name!r} has no multiplicity maximum"
            )
    scans = config.get("multiplicity_scans")
    if not isinstance(scans, dict):
        raise ValueError("multiplicity_scans must be a mapping")
    for name, scan in scans.items():
        if str(scan.get("base_coverage")) not in coverage:
            raise ValueError(f"Scan {name!r} references unknown base coverage")
        thresholds = [float(value) for value in scan.get("thresholds", [])]
        if not thresholds or thresholds != sorted(set(thresholds)):
            raise ValueError(f"Scan {name!r} thresholds must be sorted and unique")
        if not str(scan.get("variable", "")).strip():
            raise ValueError(f"Scan {name!r} lacks a variable")


def book_feature_set(frame, features, tag):
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


def make_labeled_histogram(name, title, labels, values):
    histogram = ROOT.TH1D(name, title, len(labels), 0.5, len(labels) + 0.5)
    histogram.SetDirectory(0)
    for index, (label, value) in enumerate(zip(labels, values), start=1):
        histogram.GetXaxis().SetBinLabel(index, str(label))
        histogram.SetBinContent(index, float(value))
    return histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--mc-muon-up-metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()

    hist_config = load_yaml(args.hist_config)
    config = load_yaml(args.region_config)
    validate_hist_config(hist_config)
    validate_region_config(config)
    features = configured_v5_features(hist_config, config)
    selections = configured_selections(config)
    count_diagnostics = configured_count_diagnostics(config, features, selections)
    mc_aggregate = configured_mc_aggregate(config)
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)

    metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    tree_name = str(config.get("tree_name", "sndData"))
    chain, rows = chain_for_group(
        metadata, str(config["groups"]["real_data"]), tree_name
    )
    frame = ROOT.RDataFrame(chain)
    frame = define_shared_columns(frame, hist_config.get("definitions", {}))
    frame = define_shared_columns(frame, config["derived_columns"])

    mc_metadata = pd.read_csv(args.mc_muon_up_metadata, keep_default_na=False)
    mc_samples = OrderedDict()
    for production_key, production in config["mc_muon_up"]["productions"].items():
        mc_chain, mc_rows = chain_for_group(
            mc_metadata, str(production["group"]), tree_name
        )
        mc_frame = ROOT.RDataFrame(mc_chain)
        mc_frame = define_shared_columns(
            mc_frame, hist_config.get("definitions", {})
        )
        mc_frame = define_shared_columns(mc_frame, config["derived_columns"])
        mc_frame = mc_frame.Filter(
            str(config["mc_muon_up"]["baseline_expression"]),
            f"mc_muon_up_{production_key}_baseline",
        )
        mc_samples[production_key] = {
            "definition": production,
            "chain": mc_chain,
            "rows": mc_rows,
            "frame": mc_frame,
        }

    actions = {("total",): frame.Count()}
    detector_counts = [
        "count_veto", "count_scifi", "count_us", "count_ds",
        *(f"count_veto{i}" for i in range(1, 4)),
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
    ]
    actions[("invalid_count",)] = frame.Filter(
        " || ".join(f"{name} < 0" for name in detector_counts),
        "negative_detector_count",
    ).Count()
    for production_key, sample in mc_samples.items():
        actions[("mc_total", production_key)] = sample["frame"].Count()
        actions[("mc_invalid_count", production_key)] = sample["frame"].Filter(
            " || ".join(f"{name} < 0" for name in detector_counts),
            f"mc_{production_key}_negative_detector_count",
        ).Count()

    feature_counts, histogram_actions = {}, {}
    mc_feature_counts, mc_histogram_actions = {}, {}
    cutflow_actions = {}
    nminus_counts, nminus_histogram_actions = {}, {}
    count_diag_population_actions = {}
    count_diag_feature_counts, count_diag_histogram_actions = {}, {}
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
            counts, histograms = book_feature_set(
                population_frame, features, f"{selection_name}_{population}"
            )
            for feature_name in features:
                key = (selection_name, population, feature_name)
                feature_counts[key] = counts[feature_name]
                histogram_actions[key] = histograms[feature_name]
        for production_key, sample in mc_samples.items():
            mc_population_frames = {
                "candidate": sample["frame"].Filter(
                    expression, f"{selection_name}_{production_key}_mc_candidate"
                ),
                "complement": sample["frame"].Filter(
                    f"!({expression})",
                    f"{selection_name}_{production_key}_mc_complement",
                ),
            }
            for population, population_frame in mc_population_frames.items():
                actions[(
                    "mc_selection", production_key, selection_name, population
                )] = population_frame.Count()
                counts, histograms = book_feature_set(
                    population_frame,
                    features,
                    f"{selection_name}_{production_key}_mc_{population}",
                )
                for feature_name in features:
                    key = (
                        production_key, selection_name, population, feature_name
                    )
                    mc_feature_counts[key] = counts[feature_name]
                    mc_histogram_actions[key] = histograms[feature_name]
        cumulative = []
        for component_name, component_expression in selection["components"].items():
            cumulative.append(component_expression)
            cutflow_actions[(selection_name, component_name)] = frame.Filter(
                joined(cumulative), f"{selection_name}_cutflow_{component_name}"
            ).Count()
        if selection["kind"] == "selection":
            for field, variable in MAXIMUM_FIELDS.items():
                if field not in selection["components"]:
                    continue
                other_components = [
                    expression
                    for component, expression in selection["components"].items()
                    if component != field
                ]
                nminus_frame = frame.Filter(
                    joined(other_components), f"{selection_name}_nminus1_{field}"
                )
                cut_expression = selection["components"][field]
                for population, expression in (
                    ("passes_requirement", cut_expression),
                    ("fails_requirement", f"!({cut_expression})"),
                ):
                    selected_frame = nminus_frame.Filter(
                        expression, f"{selection_name}_{field}_{population}"
                    )
                    tag = f"{selection_name}_nminus1_{field}_{population}"
                    feature_frame, columns = feature_frame_and_columns(
                        selected_frame, variable, features[variable], tag
                    )
                    key = (selection_name, field, population)
                    nminus_counts[key] = feature_frame.Count()
                    nminus_histogram_actions[key] = book_histogram(
                        feature_frame, tag, features[variable], columns
                    )
        if count_diagnostics:
            for family_name, family in count_diagnostics["families"].items():
                removed = set(family["remove_components"][selection_name])
                kept_expressions = [
                    component_expression
                    for component_name, component_expression
                    in selection["components"].items()
                    if component_name not in removed
                ]
                removed_expressions = [
                    component_expression
                    for component_name, component_expression
                    in selection["components"].items()
                    if component_name in removed
                ]
                family_expression = joined(removed_expressions)
                parent = frame.Filter(
                    joined(kept_expressions),
                    f"{selection_name}_{family_name}_nminus1_parent",
                )
                population_frames = OrderedDict((
                    ("inclusive", parent),
                    (
                        "pass",
                        parent.Filter(
                            family_expression,
                            f"{selection_name}_{family_name}_nminus1_pass",
                        ),
                    ),
                    (
                        "fail",
                        parent.Filter(
                            f"!({family_expression})",
                            f"{selection_name}_{family_name}_nminus1_fail",
                        ),
                    ),
                ))
                family_features = OrderedDict(
                    (name, count_diagnostics["features"][name])
                    for name in family["features"]
                )
                for population, population_frame in population_frames.items():
                    population_key = (selection_name, family_name, population)
                    count_diag_population_actions[population_key] = (
                        population_frame.Count()
                    )
                    feature_counts_for_frame, histograms = book_feature_set(
                        population_frame,
                        family_features,
                        f"{selection_name}_{family_name}_nminus1_{population}",
                    )
                    for feature_name in family["features"]:
                        key = (
                            selection_name,
                            family_name,
                            population,
                            feature_name,
                        )
                        count_diag_feature_counts[key] = (
                            feature_counts_for_frame[feature_name]
                        )
                        count_diag_histogram_actions[key] = histograms[feature_name]

    if {"coverage_loose", "coverage_tight"}.issubset(selections):
        loose_expression = selections["coverage_loose"]["expression"]
        tight_expression = selections["coverage_tight"]["expression"]
        actions[("tight_not_loose", "real_data")] = frame.Filter(
            f"({tight_expression}) && !({loose_expression})",
            "real_data_tight_not_loose",
        ).Count()
        for production_key, sample in mc_samples.items():
            actions[("tight_not_loose", production_key)] = sample["frame"].Filter(
                f"({tight_expression}) && !({loose_expression})",
                f"{production_key}_tight_not_loose",
            ).Count()

    scan_actions = {}
    coverage = config["coverage_working_points"]
    for scan_name, scan in config["multiplicity_scans"].items():
        base_expression = joined(
            coverage_components(coverage[str(scan["base_coverage"])]).values()
        )
        base_frame = frame.Filter(base_expression, f"{scan_name}_base_coverage")
        scan_actions[(scan_name, "base")] = base_frame.Count()
        for index, threshold in enumerate(scan["thresholds"]):
            scan_actions[(scan_name, index)] = base_frame.Filter(
                f"{scan['variable']} <= {float(threshold):.17g}",
                f"{scan_name}_threshold_{index}",
            ).Count()

    all_actions = (
        list(actions.values()) + list(feature_counts.values())
        + list(histogram_actions.values()) + list(cutflow_actions.values())
        + list(scan_actions.values()) + list(nminus_counts.values())
        + list(nminus_histogram_actions.values())
        + list(mc_feature_counts.values()) + list(mc_histogram_actions.values())
        + list(count_diag_population_actions.values())
        + list(count_diag_feature_counts.values())
        + list(count_diag_histogram_actions.values())
    )
    ROOT.RDF.RunGraphs(all_actions)
    counts = {key: int(action.GetValue()) for key, action in actions.items()}
    total = counts[("total",)]
    if counts[("invalid_count",)]:
        raise RuntimeError(
            f"Found {counts[('invalid_count',)]} events with negative detector counts"
        )
    mc_totals = {}
    for production_key in mc_samples:
        invalid = counts[("mc_invalid_count", production_key)]
        if invalid:
            raise RuntimeError(
                f"Found {invalid} negative-count MC events in {production_key}"
            )
        mc_totals[production_key] = counts[("mc_total", production_key)]
        if mc_totals[production_key] <= 0:
            raise RuntimeError(
                f"MC-muon-up baseline selection is empty for {production_key}"
            )
    expected = config.get("expected", {})
    if expected and total != int(expected["real_data_dq"]):
        raise RuntimeError(
            f"Expected real_data_dq={expected['real_data_dq']}, observed {total}"
        )
    for selection_name in selections:
        candidate = counts[("selection", selection_name, "candidate")]
        complement = counts[("selection", selection_name, "complement")]
        if candidate + complement != total:
            raise RuntimeError(
                f"Selection {selection_name} closure failed: "
                f"{candidate} + {complement} != {total}"
            )
        for production_key, mc_total in mc_totals.items():
            mc_candidate = counts[(
                "mc_selection", production_key, selection_name, "candidate"
            )]
            mc_complement = counts[(
                "mc_selection", production_key, selection_name, "complement"
            )]
            if mc_candidate + mc_complement != mc_total:
                raise RuntimeError(
                    f"MC {production_key} selection {selection_name} closure failed: "
                    f"{mc_candidate} + {mc_complement} != {mc_total}"
                )
    for key, value in counts.items():
        if key[0] == "tight_not_loose" and value:
            raise RuntimeError(
                f"Tight-to-loose nesting failed for {key[1]}: {value} events"
            )
    count_diag_population_counts = {
        key: int(action.GetValue())
        for key, action in count_diag_population_actions.items()
    }
    if count_diagnostics:
        for selection_name in selections:
            for family_name in count_diagnostics["families"]:
                parent = count_diag_population_counts[(
                    selection_name, family_name, "inclusive"
                )]
                passed = count_diag_population_counts[(
                    selection_name, family_name, "pass"
                )]
                failed = count_diag_population_counts[(
                    selection_name, family_name, "fail"
                )]
                if passed + failed != parent:
                    raise RuntimeError(
                        f"Count-diagnostic closure failed for {selection_name}/"
                        f"{family_name}: {passed} + {failed} != {parent}"
                    )
                candidate = counts[("selection", selection_name, "candidate")]
                if passed != candidate:
                    raise RuntimeError(
                        f"Count-diagnostic pass population disagrees with the full "
                        f"selection for {selection_name}/{family_name}: "
                        f"{passed} != {candidate}"
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
                selection_name, population, feature_name = key
                histogram = clone_and_fold(
                    result, POPULATIONS[population], features[feature_name]
                )
                directory = mkdir_path(
                    output,
                    f"selections/{selection_name}/features/{feature_name}",
                )
                directory.cd()
                histogram.Write(POPULATIONS[population])

            for key, result in mc_histogram_actions.items():
                production_key, selection_name, population, feature_name = key
                histogram = clone_and_fold(
                    result, MC_POPULATIONS[population], features[feature_name]
                )
                directory = mkdir_path(
                    output,
                    f"mc_muon_up/{production_key}/selections/{selection_name}/"
                    f"features/{feature_name}",
                )
                directory.cd()
                histogram.Write(MC_POPULATIONS[population])

            if mc_aggregate:
                aggregate_key = str(mc_aggregate["key"])
                for selection_name in selections:
                    for population in MC_POPULATIONS:
                        for feature_name in features:
                            combined = None
                            for production_key in mc_samples:
                                result = mc_histogram_actions[(
                                    production_key, selection_name, population,
                                    feature_name,
                                )]
                                histogram = clone_and_fold(
                                    result,
                                    f"{aggregate_key}_{population}_{feature_name}",
                                    features[feature_name],
                                )
                                if combined is None:
                                    combined = histogram
                                    combined.SetDirectory(0)
                                else:
                                    combined.Add(histogram)
                            directory = mkdir_path(
                                output,
                                f"mc_muon_up/{aggregate_key}/selections/"
                                f"{selection_name}/features/{feature_name}",
                            )
                            directory.cd()
                            combined.Write(MC_POPULATIONS[population])

            for key, result in nminus_histogram_actions.items():
                selection_name, field, population = key
                variable = MAXIMUM_FIELDS[field]
                histogram = clone_and_fold(
                    result, population, features[variable]
                )
                directory = mkdir_path(
                    output, f"diagnostics/nminus1/{selection_name}/{field}"
                )
                directory.cd()
                histogram.Write(population)

            for key, result in count_diag_histogram_actions.items():
                selection_name, family_name, population, feature_name = key
                feature = count_diagnostics["features"][feature_name]
                histogram = clone_and_fold(result, population, feature)
                directory = mkdir_path(
                    output,
                    f"count_diagnostics/{selection_name}/{family_name}/"
                    f"features/{feature_name}",
                )
                directory.cd()
                histogram.Write(population)

            for selection_name, selection in selections.items():
                labels = ["RealDataDQ", *selection["components"]]
                values = [total]
                values.extend(
                    int(cutflow_actions[(selection_name, component)].GetValue())
                    for component in selection["components"]
                )
                count_hist = make_labeled_histogram(
                    "counts", ";Cumulative requirement;Events", labels, values
                )
                fraction_hist = make_labeled_histogram(
                    "real_data_fraction_pct",
                    ";Cumulative requirement;Fraction of RealDataDQ [%]",
                    labels,
                    [100.0 * value / total if total else 0.0 for value in values],
                )
                directory = mkdir_path(output, f"diagnostics/cutflow/{selection_name}")
                directory.cd()
                count_hist.Write()
                fraction_hist.Write()

            for scan_name, scan in config["multiplicity_scans"].items():
                labels = [str(value) for value in scan["thresholds"]]
                values = [
                    int(scan_actions[(scan_name, index)].GetValue())
                    for index in range(len(labels))
                ]
                base = int(scan_actions[(scan_name, "base")].GetValue())
                histograms = (
                    make_labeled_histogram(
                        "counts", ";Maximum hit count;Selected events", labels, values
                    ),
                    make_labeled_histogram(
                        "real_data_fraction_pct",
                        ";Maximum hit count;Fraction of RealDataDQ [%]",
                        labels,
                        [100.0 * value / total if total else 0.0 for value in values],
                    ),
                    make_labeled_histogram(
                        "base_coverage_fraction_pct",
                        ";Maximum hit count;Fraction of topology candidates [%]",
                        labels,
                        [100.0 * value / base if base else 0.0 for value in values],
                    ),
                )
                directory = mkdir_path(output, f"diagnostics/threshold_scans/{scan_name}")
                directory.cd()
                for histogram in histograms:
                    histogram.Write()

            write_parameter(output, "study_version", config["version"])
            write_parameter(output, "run", int(config["run"]))
            write_parameter(output, "real_data_dq_count", total)
            aggregate_mc_total = sum(mc_totals.values())
            write_parameter(output, "mc_muon_up_baseline_count", aggregate_mc_total)
            write_parameter(
                output, "mc_muon_up_label", "MC muon up (productions separate)"
            )
            write_parameter(
                output, "mc_muon_up_baseline_expression",
                config["mc_muon_up"]["baseline_expression"],
            )
            write_parameter(
                output, "mc_muon_up_partition_count",
                sum(len(sample["rows"]) for sample in mc_samples.values()),
            )
            write_parameter(
                output, "mc_muon_up_production_order", ",".join(mc_samples)
            )
            write_parameter(
                output, "mc_muon_up_note", config["mc_muon_up"].get("note", "")
            )
            if mc_aggregate:
                aggregate_key = str(mc_aggregate["key"])
                aggregate_prefix = f"mc_muon_up__{aggregate_key}"
                write_parameter(
                    output, f"{aggregate_prefix}__label", mc_aggregate["label"]
                )
                write_parameter(
                    output, f"{aggregate_prefix}__aggregation_mode",
                    mc_aggregate["mode"],
                )
                write_parameter(
                    output, f"{aggregate_prefix}__baseline_count",
                    aggregate_mc_total,
                )
            for production_key, sample in mc_samples.items():
                production = sample["definition"]
                prefix = f"mc_muon_up__{production_key}"
                write_parameter(output, f"{prefix}__group", production["group"])
                write_parameter(output, f"{prefix}__label", production["label"])
                write_parameter(
                    output, f"{prefix}__source_partition",
                    production["source_partition"],
                )
                write_parameter(
                    output, f"{prefix}__partition_count", len(sample["rows"])
                )
                write_parameter(
                    output, f"{prefix}__declared_event_count",
                    int(sample["rows"]["n_events"].astype(int).sum()),
                )
                write_parameter(
                    output, f"{prefix}__lumi", float(
                        sample["rows"]["lumi_per_partition"].astype(float).sum()
                    ),
                )
                write_parameter(
                    output, f"{prefix}__baseline_count", mc_totals[production_key]
                )
            write_parameter(output, "partition_count", len(rows))
            write_parameter(output, "feature_order", ",".join(features))
            write_parameter(output, "selection_order", ",".join(selections))
            write_parameter(
                output, "threshold_scan_order", ",".join(config["multiplicity_scans"])
            )
            write_parameter(output, "uses_reconstructed_tracks", False)
            write_parameter(output, "uses_mc_muon_up", True)
            write_parameter(
                output, "uses_count_diagnostics", bool(count_diagnostics)
            )
            if count_diagnostics:
                write_parameter(
                    output,
                    "count_diagnostic_family_order",
                    ",".join(count_diagnostics["families"]),
                )
                write_parameter(
                    output,
                    "count_diagnostic_overview_feature_order",
                    ",".join(count_diagnostics["overview_features"]),
                )
                for family_name, family in count_diagnostics["families"].items():
                    family_prefix = f"count_diagnostic__{family_name}"
                    write_parameter(
                        output, f"{family_prefix}__label", family["label"]
                    )
                    write_parameter(
                        output,
                        f"{family_prefix}__feature_order",
                        ",".join(family["features"]),
                    )
            for selection_name, selection in selections.items():
                candidate = counts[("selection", selection_name, "candidate")]
                complement = counts[("selection", selection_name, "complement")]
                prefix = f"selection__{selection_name}"
                write_parameter(output, f"{prefix}__kind", selection["kind"])
                write_parameter(output, f"{prefix}__label", selection["label"])
                write_parameter(output, f"{prefix}__expression", selection["expression"])
                write_parameter(
                    output, f"{prefix}__candidate_label", selection["candidate_label"]
                )
                write_parameter(
                    output, f"{prefix}__complement_label", selection["complement_label"]
                )
                write_parameter(output, f"{prefix}__candidate_count", candidate)
                write_parameter(output, f"{prefix}__complement_count", complement)
                write_parameter(
                    output, f"{prefix}__population_closure", candidate + complement == total
                )
                aggregate_candidate = 0
                aggregate_complement = 0
                for production_key, mc_total in mc_totals.items():
                    mc_candidate = counts[(
                        "mc_selection", production_key, selection_name, "candidate"
                    )]
                    mc_complement = counts[(
                        "mc_selection", production_key, selection_name, "complement"
                    )]
                    aggregate_candidate += mc_candidate
                    aggregate_complement += mc_complement
                    mc_prefix = f"{prefix}__mc__{production_key}"
                    write_parameter(
                        output, f"{mc_prefix}__candidate_count", mc_candidate
                    )
                    write_parameter(
                        output, f"{mc_prefix}__complement_count", mc_complement
                    )
                    write_parameter(
                        output, f"{mc_prefix}__population_closure",
                        mc_candidate + mc_complement == mc_total,
                    )
                    write_parameter(
                        output, f"{mc_prefix}__candidate_efficiency",
                        float(mc_candidate) / float(mc_total) if mc_total else 0.0,
                    )
                    for population in MC_POPULATIONS:
                        for feature_name in features:
                            write_parameter(
                                output,
                                f"{mc_prefix}__{population}__{feature_name}__valid_count",
                                int(mc_feature_counts[(
                                    production_key, selection_name, population,
                                    feature_name,
                                )].GetValue()),
                            )
                # Compatibility totals are counts only; no combined-production
                # histogram is written because no physics weighting is defined.
                write_parameter(
                    output, f"{prefix}__mc_candidate_count", aggregate_candidate
                )
                write_parameter(
                    output, f"{prefix}__mc_complement_count", aggregate_complement
                )
                write_parameter(
                    output, f"{prefix}__mc_population_closure",
                    aggregate_candidate + aggregate_complement == aggregate_mc_total,
                )
                write_parameter(
                    output, f"{prefix}__mc_candidate_efficiency",
                    float(aggregate_candidate) / float(aggregate_mc_total)
                    if aggregate_mc_total else 0.0,
                )
                if mc_aggregate:
                    aggregate_selection_prefix = (
                        f"{prefix}__mc__{mc_aggregate['key']}"
                    )
                    write_parameter(
                        output,
                        f"{aggregate_selection_prefix}__candidate_count",
                        aggregate_candidate,
                    )
                    write_parameter(
                        output,
                        f"{aggregate_selection_prefix}__complement_count",
                        aggregate_complement,
                    )
                    write_parameter(
                        output,
                        f"{aggregate_selection_prefix}__population_closure",
                        aggregate_candidate + aggregate_complement
                        == aggregate_mc_total,
                    )
                    write_parameter(
                        output,
                        f"{aggregate_selection_prefix}__candidate_efficiency",
                        float(aggregate_candidate) / float(aggregate_mc_total)
                        if aggregate_mc_total else 0.0,
                    )
                write_parameter(
                    output, f"{prefix}__cut_order", ",".join(selection["components"])
                )
                for component, expression in selection["components"].items():
                    write_parameter(
                        output, f"{prefix}__cut__{component}__expression", expression
                    )
                    write_parameter(
                        output, f"{prefix}__cut__{component}__count",
                        int(cutflow_actions[(selection_name, component)].GetValue()),
                    )
                for population in POPULATIONS:
                    for feature_name in features:
                        write_parameter(
                            output,
                            f"{prefix}__{population}__{feature_name}__valid_count",
                            int(feature_counts[(selection_name, population, feature_name)].GetValue()),
                        )
                if count_diagnostics:
                    for family_name, family in count_diagnostics["families"].items():
                        count_prefix = f"{prefix}__count_diagnostic__{family_name}"
                        removed = family["remove_components"][selection_name]
                        write_parameter(
                            output,
                            f"{count_prefix}__removed_components",
                            ",".join(removed),
                        )
                        for population in ("inclusive", "pass", "fail"):
                            write_parameter(
                                output,
                                f"{count_prefix}__{population}_count",
                                count_diag_population_counts[(
                                    selection_name, family_name, population
                                )],
                            )
                            for feature_name in family["features"]:
                                write_parameter(
                                    output,
                                    f"{count_prefix}__{population}__"
                                    f"{feature_name}__valid_count",
                                    int(count_diag_feature_counts[(
                                        selection_name,
                                        family_name,
                                        population,
                                        feature_name,
                                    )].GetValue()),
                                )
                        parent = count_diag_population_counts[(
                            selection_name, family_name, "inclusive"
                        )]
                        passed = count_diag_population_counts[(
                            selection_name, family_name, "pass"
                        )]
                        failed = count_diag_population_counts[(
                            selection_name, family_name, "fail"
                        )]
                        write_parameter(
                            output,
                            f"{count_prefix}__population_closure",
                            passed + failed == parent,
                        )
                for field in MAXIMUM_FIELDS:
                    if field not in selection["components"]:
                        continue
                    for population in ("passes_requirement", "fails_requirement"):
                        write_parameter(
                            output,
                            f"{prefix}__nminus1__{field}__{population}__count",
                            int(nminus_counts[(selection_name, field, population)].GetValue()),
                        )
            for scan_name, scan in config["multiplicity_scans"].items():
                prefix = f"scan__{scan_name}"
                write_parameter(output, f"{prefix}__variable", scan["variable"])
                write_parameter(output, f"{prefix}__base_coverage", scan["base_coverage"])
                write_parameter(output, f"{prefix}__base_count", int(scan_actions[(scan_name, 'base')].GetValue()))
                write_parameter(
                    output, f"{prefix}__thresholds",
                    ",".join(str(value) for value in scan["thresholds"]),
                )
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(
        f"Wrote {len(features)} features for {len(selections)} detector-only "
        f"working points from {total} RealDataDQ events and "
        f"{sum(mc_totals.values())} MC-muon-up baseline events across "
        f"{len(mc_totals)} separate productions"
    )


if __name__ == "__main__":
    main()
