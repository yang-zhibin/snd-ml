#!/usr/bin/env python3
"""Build run-8329 track-type and NoPassingMuonReco comparison histograms."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import ROOT
import yaml


ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


ALGORITHM_COMBINATION_ORDER = (
    "sf0_ds0",
    "sf1_ds0",
    "sf0_ds1",
    "sf1_ds1",
)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def root_input(path):
    path = str(path)
    return f"root://eospublic.cern.ch/{path}" if path.startswith("/eos/") else path


def chain_for_group(metadata, group, tree_name):
    rows = metadata.loc[metadata["particle_group"].astype(str) == group].copy()
    if rows.empty:
        raise ValueError(f"No partition rows for {group}")
    rows = rows.sort_values("partition_id")
    chain = ROOT.TChain(tree_name)
    for path in rows["feature_partition_path"].astype(str):
        if chain.Add(root_input(path)) <= 0:
            raise RuntimeError(f"Could not add {path}")
    return chain, rows


def validate_hist_config(config):
    features = config.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("Histogram config requires a non-empty features mapping")
    for name, feature in features.items():
        dimension = int(feature.get("dimension", 1))
        if dimension == 1:
            required = {"expression", "n_bins", "x_min", "x_max", "axis_title"}
        elif dimension == 2:
            required = {
                "x_expression", "y_expression", "x_bins", "x_min", "x_max",
                "y_bins", "y_min", "y_max", "x_title", "y_title",
            }
        else:
            raise ValueError(f"Feature {name!r} has unsupported dimension {dimension}")
        missing = required - set(feature)
        if missing:
            raise ValueError(f"Feature {name!r} lacks fields: {sorted(missing)}")


def validate_study_config(config):
    if not config.get("track_types"):
        raise ValueError("Study config requires track_types")
    if not config.get("working_points"):
        raise ValueError("Study config requires working_points")
    for name, track in config["track_types"].items():
        components = track.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"Track category {name!r} requires non-empty components")
        components = [str(component).strip() for component in components]
        if any(not component for component in components):
            raise ValueError(f"Track category {name!r} contains an empty component")
        if len(components) != len(set(components)):
            raise ValueError(f"Track category {name!r} repeats a component")
    algorithms = config.get("algorithm_comparisons")
    if not isinstance(algorithms, dict) or not algorithms:
        raise ValueError("Study config requires algorithm_comparisons")
    for name, algorithm in algorithms.items():
        required = {"label", "scifi_component", "ds_component"}
        missing = required - set(algorithm)
        if missing:
            raise ValueError(
                f"Algorithm comparison {name!r} lacks fields: {sorted(missing)}"
            )
        scifi = str(algorithm["scifi_component"]).strip()
        ds = str(algorithm["ds_component"]).strip()
        if not scifi or not ds or scifi == ds:
            raise ValueError(
                f"Algorithm comparison {name!r} requires distinct components"
            )
    finite_thresholds = []
    for name, working_point in config["working_points"].items():
        threshold = working_point.get("chi2ndf_max")
        if threshold is None:
            continue
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Working point {name!r} has invalid chi2ndf_max")
        finite_thresholds.append(threshold)
    if len(finite_thresholds) != len(set(finite_thresholds)):
        raise ValueError("Finite chi2/NDF working-point thresholds must be unique")

    quality = config.get("track_quality_2d")
    if not isinstance(quality, dict):
        raise ValueError("Study config requires track_quality_2d")
    score = quality.get("score")
    if not isinstance(score, dict):
        raise ValueError("track_quality_2d requires score binning")
    for field in ("n_bins", "min", "max"):
        if field not in score:
            raise ValueError(f"track_quality_2d.score lacks {field!r}")
    if int(score["n_bins"]) <= 0 or float(score["max"]) <= float(score["min"]):
        raise ValueError("track_quality_2d.score has invalid binning")
    observables = quality.get("observables")
    if not isinstance(observables, dict) or not observables:
        raise ValueError("track_quality_2d requires observables")
    required = {"expression", "n_bins", "min", "max", "axis_title"}
    for name, observable in observables.items():
        missing = required - set(observable)
        if missing:
            raise ValueError(
                f"Track-quality observable {name!r} lacks fields: {sorted(missing)}"
            )
        if int(observable["n_bins"]) <= 0:
            raise ValueError(f"Track-quality observable {name!r} has no bins")
        if float(observable["max"]) <= float(observable["min"]):
            raise ValueError(f"Track-quality observable {name!r} has invalid range")

    mc_config = config.get("mc_muon_up")
    if not isinstance(mc_config, dict):
        raise ValueError("Study config requires mc_muon_up")
    for field in (
        "group", "label", "source", "inclusive_expression",
        "reference_expression",
    ):
        if not mc_config.get(field):
            raise ValueError(f"mc_muon_up lacks {field!r}")


def configured_features(config):
    defaults = config.get("defaults", {})
    result = {}
    for name, value in config["features"].items():
        feature = dict(defaults)
        feature.update(value)
        feature["dimension"] = int(feature.get("dimension", 1))
        result[name] = feature
    return result


def define_shared_columns(dataframe, definitions):
    columns = {str(column) for column in dataframe.GetColumnNames()}
    for name, expression in definitions.items():
        if name in columns:
            raise ValueError(f"Definition {name!r} already exists in input")
        dataframe = dataframe.Define(str(name), str(expression))
        columns.add(str(name))
    return dataframe


def track_components(track):
    return [str(component) for component in track["components"]]


def joined_expression(parts):
    return " && ".join(f"({part})" for part in parts)


def converged_expression(track):
    return joined_expression(
        [f"{prefix}_has_converged == 1" for prefix in track_components(track)]
    )


def best_expression(track):
    return joined_expression(
        [f"{prefix}_has_best == 1" for prefix in track_components(track)]
    )


def quality_score_expression(track):
    scores = [
        f"{prefix}_best_chi2ndf" for prefix in track_components(track)
    ]
    expression = scores[0]
    for score in scores[1:]:
        expression = f"std::max({expression}, {score})"
    return expression


def target_expression(track, threshold):
    if threshold is None:
        return converged_expression(track)
    threshold_expression = [best_expression(track)]
    threshold_expression.extend(
        f"{prefix}_best_chi2ndf < {float(threshold):.17g}"
        for prefix in track_components(track)
    )
    return joined_expression(threshold_expression)


def component_expression(prefix, threshold):
    return target_expression({"components": [str(prefix)]}, threshold)


def algorithm_combination_expressions(scifi_expression, ds_expression):
    return {
        "sf0_ds0": f"!({scifi_expression}) && !({ds_expression})",
        "sf1_ds0": f"({scifi_expression}) && !({ds_expression})",
        "sf0_ds1": f"!({scifi_expression}) && ({ds_expression})",
        "sf1_ds1": f"({scifi_expression}) && ({ds_expression})",
    }


def configured_track_quality_features(config):
    quality = config["track_quality_2d"]
    score = quality["score"]
    result = {}
    for name, observable in quality["observables"].items():
        result[name] = {
            "dimension": 2,
            "x_bins": int(score["n_bins"]),
            "x_min": float(score["min"]),
            "x_max": float(score["max"]),
            "y_bins": int(observable["n_bins"]),
            "y_min": float(observable["min"]),
            "y_max": float(observable["max"]),
            "x_title": "Best track #chi^{2}/NDF",
            "y_title": str(observable["axis_title"]),
            "y_expression": str(observable["expression"]),
            "fold_x_underflow": bool(score.get("fold_underflow", False)),
            "fold_x_overflow": bool(score.get("fold_overflow", False)),
            "fold_y_underflow": bool(observable.get("fold_underflow", False)),
            "fold_y_overflow": bool(observable.get("fold_overflow", False)),
            "logz": bool(observable.get("logz", True)),
        }
    return result


def feature_frame_and_columns(dataframe, feature_name, feature, tag):
    frame = dataframe
    feature_cut = str(feature.get("feature_cut", "")).strip()
    if feature_cut:
        frame = frame.Filter(feature_cut, f"{tag}_{feature_name}_valid")
    if feature["dimension"] == 1:
        column = f"__{tag}_{feature_name}"
        return frame.Define(column, str(feature["expression"])), (column,)
    x_column = f"__{tag}_{feature_name}_x"
    y_column = f"__{tag}_{feature_name}_y"
    frame = frame.Define(x_column, str(feature["x_expression"]))
    frame = frame.Define(y_column, str(feature["y_expression"]))
    return frame, (x_column, y_column)


def book_histogram(dataframe, name, feature, columns):
    if feature["dimension"] == 1:
        model = (
            name, "", int(feature["n_bins"]),
            float(feature["x_min"]), float(feature["x_max"]),
        )
        return dataframe.Histo1D(model, columns[0])
    model = (
        name, "", int(feature["x_bins"]),
        float(feature["x_min"]), float(feature["x_max"]),
        int(feature["y_bins"]), float(feature["y_min"]), float(feature["y_max"]),
    )
    return dataframe.Histo2D(model, columns[0], columns[1])


def _add_bin(histogram, target, source):
    histogram.SetBinContent(
        target, histogram.GetBinContent(target) + histogram.GetBinContent(source)
    )
    histogram.SetBinError(
        target,
        math.hypot(histogram.GetBinError(target), histogram.GetBinError(source)),
    )
    histogram.SetBinContent(source, 0.0)
    histogram.SetBinError(source, 0.0)


def fold_1d(histogram, underflow, overflow):
    if underflow:
        _add_bin(histogram, 1, 0)
    if overflow:
        last = histogram.GetNbinsX()
        _add_bin(histogram, last, last + 1)


def fold_2d_axis(histogram, axis, underflow, overflow):
    nx, ny = histogram.GetNbinsX(), histogram.GetNbinsY()
    if axis == "x":
        for iy in range(0, ny + 2):
            if underflow:
                _add_bin(histogram, histogram.GetBin(1, iy), histogram.GetBin(0, iy))
            if overflow:
                _add_bin(
                    histogram, histogram.GetBin(nx, iy), histogram.GetBin(nx + 1, iy)
                )
        return
    if axis == "y":
        for ix in range(1, nx + 1):
            if underflow:
                _add_bin(histogram, histogram.GetBin(ix, 1), histogram.GetBin(ix, 0))
            if overflow:
                _add_bin(
                    histogram, histogram.GetBin(ix, ny), histogram.GetBin(ix, ny + 1)
                )
        return
    raise ValueError(f"Unsupported axis {axis!r}")


def clone_and_fold(result, name, feature):
    histogram = result.GetValue().Clone(name)
    histogram.SetDirectory(0)
    if feature["dimension"] == 1:
        fold_1d(
            histogram,
            bool(feature.get("fold_underflow", False)),
            bool(feature.get("fold_overflow", False)),
        )
        histogram.GetXaxis().SetTitle(str(feature["axis_title"]))
        histogram.GetYaxis().SetTitle("Events")
    else:
        fold_2d_axis(
            histogram, "x",
            bool(feature.get("fold_x_underflow", False)),
            bool(feature.get("fold_x_overflow", False)),
        )
        fold_2d_axis(
            histogram, "y",
            bool(feature.get("fold_y_underflow", False)),
            bool(feature.get("fold_y_overflow", False)),
        )
        histogram.GetXaxis().SetTitle(str(feature["x_title"]))
        histogram.GetYaxis().SetTitle(str(feature["y_title"]))
        histogram.GetZaxis().SetTitle("Events")
    return histogram


def write_parameter(output, name, value):
    output.cd()
    if isinstance(value, bool):
        ROOT.TParameter("int")(name, int(value)).Write()
    elif isinstance(value, int):
        ROOT.TParameter("Long64_t")(name, value).Write()
    elif isinstance(value, float):
        ROOT.TParameter("double")(name, value).Write()
    else:
        ROOT.TNamed(name, str(value)).Write()


def mkdir_path(output, path):
    directory = output
    for component in path.split("/"):
        child = directory.GetDirectory(component)
        directory = child if child else directory.mkdir(component)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-metadata", required=True)
    parser.add_argument("--missing-metadata", required=True)
    parser.add_argument("--mc-muon-up-metadata", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--working-point", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()

    hist_config = load_yaml(args.hist_config)
    study_config = load_yaml(args.study_config)
    validate_hist_config(hist_config)
    validate_study_config(study_config)
    if args.working_point not in study_config["working_points"]:
        raise KeyError(f"Unknown working point {args.working_point!r}")
    if args.threads > 0:
        ROOT.EnableImplicitMT(args.threads)

    paired_metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    missing_metadata = pd.read_csv(args.missing_metadata, keep_default_na=False)
    mc_metadata = pd.read_csv(args.mc_muon_up_metadata, keep_default_na=False)
    tree_name = str(study_config.get("tree_name", "sndData"))
    passing_group = str(study_config["groups"]["passing"])
    missing_group = str(study_config["groups"]["missing"])
    passing_chain, passing_rows = chain_for_group(
        paired_metadata, passing_group, tree_name
    )
    missing_chain, missing_rows = chain_for_group(
        missing_metadata, missing_group, tree_name
    )
    mc_config = study_config["mc_muon_up"]
    mc_chain, mc_rows = chain_for_group(
        mc_metadata, str(mc_config["group"]), tree_name
    )
    passing = define_shared_columns(
        ROOT.RDataFrame(passing_chain), hist_config.get("definitions", {})
    )
    missing = define_shared_columns(
        ROOT.RDataFrame(missing_chain), hist_config.get("definitions", {})
    )
    mc_inclusive = define_shared_columns(
        ROOT.RDataFrame(mc_chain), hist_config.get("definitions", {})
    ).Filter(
        str(mc_config["inclusive_expression"]), "mc_muon_up_inclusive"
    )
    mc_reference = mc_inclusive.Filter(
        str(mc_config["reference_expression"]),
        "mc_muon_up_detector_activity",
    )

    track_types = study_config["track_types"]
    algorithm_comparisons = study_config["algorithm_comparisons"]
    working_point = study_config["working_points"][args.working_point]
    threshold = working_point.get("chi2ndf_max")
    threshold = None if threshold is None else float(threshold)
    features = configured_features(hist_config)
    quality_features = configured_track_quality_features(study_config)
    passing_columns = {str(column) for column in passing.GetColumnNames()}
    required_track_columns = {"reco_track_n_types_present"}
    for track in track_types.values():
        for prefix in track_components(track):
            required_track_columns.update(
                {
                    f"{prefix}_has_converged",
                    f"{prefix}_has_best",
                    f"{prefix}_best_chi2ndf",
                }
            )
    for algorithm in algorithm_comparisons.values():
        for prefix in (
            str(algorithm["scifi_component"]),
            str(algorithm["ds_component"]),
        ):
            required_track_columns.update(
                {
                    f"{prefix}_has_converged",
                    f"{prefix}_has_best",
                    f"{prefix}_best_chi2ndf",
                }
            )
    required_quality_columns = {
        str(observable["y_expression"]) for observable in quality_features.values()
    }
    required_track_columns.update(required_quality_columns)
    absent = required_track_columns - passing_columns
    if absent:
        raise RuntimeError(f"Passing partitions lack track branches: {sorted(absent)}")
    mc_columns = {str(column) for column in mc_reference.GetColumnNames()}
    required_mc_columns = {
        "pdgCode", "count_veto", "count_scifi", "count_us", "count_ds",
    }
    absent_mc = required_mc_columns - mc_columns
    if absent_mc:
        raise RuntimeError(
            f"MC-muon-up features lack detector branches: {sorted(absent_mc)}"
        )

    actions = {
        "passing_count": passing.Count(),
        "missing_count": missing.Count(),
        "mc_muon_up_inclusive_count": mc_inclusive.Count(),
        "mc_muon_up_reference_count": mc_reference.Count(),
        "multiple_types_count": passing.Filter(
            "reco_track_n_types_present > 1", "multiple_track_types"
        ).Count(),
    }
    histogram_actions = {}
    missing_feature_counts = {}
    passing_reference_feature_counts = {}
    mc_reference_feature_counts = {}
    for feature_name, feature in features.items():
        frame, columns = feature_frame_and_columns(
            missing, feature_name, feature, "missing"
        )
        missing_feature_counts[feature_name] = frame.Count()
        histogram_actions[("missing", feature_name)] = book_histogram(
            frame, f"missing__{feature_name}", feature, columns
        )
        passing_frame, passing_columns_for_feature = feature_frame_and_columns(
            passing, feature_name, feature, "passing_reference"
        )
        passing_reference_feature_counts[feature_name] = passing_frame.Count()
        histogram_actions[("passing_reference", feature_name)] = book_histogram(
            passing_frame,
            f"passing_reference__{feature_name}",
            feature,
            passing_columns_for_feature,
        )
        mc_frame, mc_columns_for_feature = feature_frame_and_columns(
            mc_reference, feature_name, feature, "mc_muon_up_reference"
        )
        mc_reference_feature_counts[feature_name] = mc_frame.Count()
        histogram_actions[("mc_muon_up_reference", feature_name)] = book_histogram(
            mc_frame,
            f"mc_muon_up_reference__{feature_name}",
            feature,
            mc_columns_for_feature,
        )

    expressions = {}
    target_frames = {}
    other_frames = {}
    target_feature_counts = {}
    other_feature_counts = {}
    quality_histogram_actions = {}
    quality_feature_counts = {}
    converged_expressions = {}
    best_expressions = {}
    score_expressions = {}
    for key, track in track_types.items():
        converged = converged_expression(track)
        best = best_expression(track)
        score = quality_score_expression(track)
        expression = target_expression(track, threshold)
        converged_expressions[key] = converged
        best_expressions[key] = best
        score_expressions[key] = score
        expressions[key] = expression
        target_frames[key] = passing.Filter(expression, f"{key}_has_target")
        other_frames[key] = passing.Filter(f"!({expression})", f"{key}_other_passing")
        actions[f"{key}__has_target_count"] = target_frames[key].Count()
        actions[f"{key}__other_passing_count"] = other_frames[key].Count()
        actions[f"{key}__has_converged_count"] = passing.Filter(
            converged, f"{key}_has_converged"
        ).Count()
        actions[f"{key}__has_best_count"] = passing.Filter(
            best, f"{key}_has_best"
        ).Count()
        actions[f"{key}__unscorable_count"] = passing.Filter(
            f"({converged}) && !({best})",
            f"{key}_unscorable",
        ).Count()
        for feature_name, feature in features.items():
            target_frame, target_columns = feature_frame_and_columns(
                target_frames[key], feature_name, feature, f"{key}_target"
            )
            other_frame, other_columns = feature_frame_and_columns(
                other_frames[key], feature_name, feature, f"{key}_other"
            )
            target_feature_counts[(key, feature_name)] = target_frame.Count()
            other_feature_counts[(key, feature_name)] = other_frame.Count()
            histogram_actions[(key, "has_target", feature_name)] = book_histogram(
                target_frame, f"{key}__has_target__{feature_name}", feature, target_columns
            )
            histogram_actions[(key, "other_passing", feature_name)] = book_histogram(
                other_frame, f"{key}__other_passing__{feature_name}", feature, other_columns
            )

        score_column = f"__{key}_quality_score"
        quality_base = (
            target_frames[key]
            .Filter(best, f"{key}_quality_has_best")
            .Define(score_column, score)
            .Filter(
                f"std::isfinite({score_column}) && {score_column} >= 0",
                f"{key}_quality_score_valid",
            )
        )
        for feature_name, feature in quality_features.items():
            value_column = f"__{key}_quality_{feature_name}"
            quality_frame = quality_base.Define(
                value_column, feature["y_expression"]
            )
            quality_feature_counts[(key, feature_name)] = quality_frame.Count()
            quality_histogram_actions[(key, feature_name)] = book_histogram(
                quality_frame,
                f"{key}__quality__{feature_name}",
                feature,
                (score_column, value_column),
            )

    algorithm_component_expressions = {}
    algorithm_expressions = {}
    algorithm_feature_counts = {}
    algorithm_histogram_actions = {}
    for algorithm_key, algorithm in algorithm_comparisons.items():
        scifi_expression = component_expression(
            algorithm["scifi_component"], threshold
        )
        ds_expression = component_expression(algorithm["ds_component"], threshold)
        algorithm_component_expressions[algorithm_key] = {
            "scifi": scifi_expression,
            "ds": ds_expression,
        }
        combinations = algorithm_combination_expressions(
            scifi_expression, ds_expression
        )
        algorithm_expressions[algorithm_key] = combinations
        for combination in ALGORITHM_COMBINATION_ORDER:
            expression = combinations[combination]
            frame = passing.Filter(
                expression, f"{algorithm_key}_{combination}_selection"
            )
            actions[f"algorithm__{algorithm_key}__{combination}__count"] = (
                frame.Count()
            )
            for feature_name, feature in features.items():
                feature_frame, columns = feature_frame_and_columns(
                    frame,
                    feature_name,
                    feature,
                    f"algorithm_{algorithm_key}_{combination}",
                )
                algorithm_feature_counts[
                    (algorithm_key, combination, feature_name)
                ] = feature_frame.Count()
                algorithm_histogram_actions[
                    (algorithm_key, combination, feature_name)
                ] = book_histogram(
                    feature_frame,
                    f"algorithm__{algorithm_key}__{combination}__{feature_name}",
                    feature,
                    columns,
                )

    track_keys = list(track_types)
    for left_index, left in enumerate(track_keys):
        for right_index in range(left_index, len(track_keys)):
            right = track_keys[right_index]
            actions[f"overlap__{left}__{right}"] = passing.Filter(
                f"({expressions[left]}) && ({expressions[right]})",
                f"overlap_{left}_{right}",
            ).Count()

    ROOT.RDF.RunGraphs(
        list(actions.values())
        + list(missing_feature_counts.values())
        + list(passing_reference_feature_counts.values())
        + list(mc_reference_feature_counts.values())
        + list(target_feature_counts.values())
        + list(other_feature_counts.values())
        + list(histogram_actions.values())
        + list(quality_feature_counts.values())
        + list(quality_histogram_actions.values())
        + list(algorithm_feature_counts.values())
        + list(algorithm_histogram_actions.values())
    )
    counts = {name: int(action.GetValue()) for name, action in actions.items()}
    passing_count = counts["passing_count"]
    missing_count = counts["missing_count"]
    expected = study_config.get("expected", {})
    if expected:
        if passing_count != int(expected["passing_muon_dq"]):
            raise RuntimeError("Passing-muon total disagrees with study configuration")
        if missing_count != int(expected["no_passing_muon_reco"]):
            raise RuntimeError("NoPassingMuonReco total disagrees with study configuration")
        if passing_count + missing_count != int(expected["real_data_dq"]):
            raise RuntimeError("RealDataDQ total closure failed")
    for key in track_keys:
        if counts[f"{key}__has_target_count"] + counts[f"{key}__other_passing_count"] != passing_count:
            raise RuntimeError(f"Has/Other closure failed for {key}")
    for algorithm_key in algorithm_comparisons:
        combination_total = sum(
            counts[f"algorithm__{algorithm_key}__{combination}__count"]
            for combination in ALGORITHM_COMBINATION_ORDER
        )
        if combination_total != passing_count:
            raise RuntimeError(
                f"Four-combination closure failed for {algorithm_key}: "
                f"{combination_total} != {passing_count}"
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
            for feature_name, feature in features.items():
                missing_histogram = clone_and_fold(
                    histogram_actions[("missing", feature_name)],
                    "no_passing_muon_reco", feature,
                )
                directory = mkdir_path(output, f"missing/{feature_name}")
                directory.cd()
                missing_histogram.Write("no_passing_muon_reco")

                passing_reference_histogram = clone_and_fold(
                    histogram_actions[("passing_reference", feature_name)],
                    "passing_muon_reco", feature,
                )
                missing_reference_histogram = clone_and_fold(
                    histogram_actions[("missing", feature_name)],
                    "no_passing_muon_reco", feature,
                )
                real_data_histogram = passing_reference_histogram.Clone(
                    "real_data_dq"
                )
                real_data_histogram.SetDirectory(0)
                real_data_histogram.Add(missing_reference_histogram)
                mc_histogram = clone_and_fold(
                    histogram_actions[("mc_muon_up_reference", feature_name)],
                    "mc_muon_up", feature,
                )
                directory = mkdir_path(output, f"references/{feature_name}")
                directory.cd()
                mc_histogram.Write("mc_muon_up")
                real_data_histogram.Write("real_data_dq")
                passing_reference_histogram.Write("passing_muon_reco")
                missing_reference_histogram.Write("no_passing_muon_reco")
                for key in track_keys:
                    directory = mkdir_path(output, f"targets/{key}/{feature_name}")
                    directory.cd()
                    for population in ("has_target", "other_passing"):
                        histogram = clone_and_fold(
                            histogram_actions[(key, population, feature_name)],
                            population, feature,
                        )
                        histogram.Write(population)
            for key, track in track_types.items():
                for feature_name, raw_feature in quality_features.items():
                    feature = dict(raw_feature)
                    if len(track_components(track)) > 1:
                        feature["x_title"] = "Maximum component best #chi^{2}/NDF"
                    histogram = clone_and_fold(
                        quality_histogram_actions[(key, feature_name)],
                        "has_target",
                        feature,
                    )
                    directory = mkdir_path(
                        output, f"track_quality/{key}/{feature_name}"
                    )
                    directory.cd()
                    histogram.Write("has_target")
            for algorithm_key in algorithm_comparisons:
                for combination in ALGORITHM_COMBINATION_ORDER:
                    for feature_name, feature in features.items():
                        histogram = clone_and_fold(
                            algorithm_histogram_actions[
                                (algorithm_key, combination, feature_name)
                            ],
                            combination,
                            feature,
                        )
                        directory = mkdir_path(
                            output,
                            "algorithm_comparisons/"
                            f"{algorithm_key}/{combination}/{feature_name}",
                        )
                        directory.cd()
                        histogram.Write(combination)
            write_parameter(output, "study_version", study_config["version"])
            write_parameter(output, "histogram_version", hist_config["version"])
            write_parameter(output, "working_point", args.working_point)
            write_parameter(output, "working_point_label", working_point["label"])
            write_parameter(output, "chi2ndf_max", "none" if threshold is None else threshold)
            write_parameter(output, "passing_count", passing_count)
            write_parameter(output, "missing_count", missing_count)
            write_parameter(output, "real_data_dq_count", passing_count + missing_count)
            write_parameter(
                output, "mc_muon_up_inclusive_count",
                counts["mc_muon_up_inclusive_count"],
            )
            write_parameter(
                output, "mc_muon_up_reference_count",
                counts["mc_muon_up_reference_count"],
            )
            write_parameter(output, "mc_muon_up_label", mc_config["label"])
            write_parameter(
                output, "mc_muon_up_inclusive_expression",
                mc_config["inclusive_expression"],
            )
            write_parameter(
                output, "mc_muon_up_reference_expression",
                mc_config["reference_expression"],
            )
            write_parameter(output, "mc_muon_up_partition_count", len(mc_rows))
            write_parameter(
                output,
                "mc_muon_up_source_files",
                ",".join(mc_rows["feature_partition_path"].astype(str)),
            )
            write_parameter(output, "multiple_types_count", counts["multiple_types_count"])
            write_parameter(output, "track_type_order", ",".join(track_keys))
            write_parameter(output, "feature_order", ",".join(features))
            write_parameter(
                output, "track_quality_feature_order", ",".join(quality_features)
            )
            write_parameter(
                output,
                "algorithm_comparison_order",
                ",".join(algorithm_comparisons),
            )
            write_parameter(
                output,
                "algorithm_combination_order",
                ",".join(ALGORITHM_COMBINATION_ORDER),
            )
            write_parameter(output, "three_way_closure", True)
            write_parameter(output, "missing_event_semantics", "RealDataDQ minus PassingMuonReco")
            write_parameter(output, "passing_partition_count", len(passing_rows))
            write_parameter(output, "missing_partition_count", len(missing_rows))
            for feature_name in features:
                write_parameter(
                    output, f"missing__{feature_name}__selected",
                    int(missing_feature_counts[feature_name].GetValue()),
                )
                write_parameter(
                    output, f"passing_reference__{feature_name}__selected",
                    int(passing_reference_feature_counts[feature_name].GetValue()),
                )
                write_parameter(
                    output, f"mc_muon_up__{feature_name}__selected",
                    int(mc_reference_feature_counts[feature_name].GetValue()),
                )
            for key, track in track_types.items():
                write_parameter(output, f"{key}__label", track["label"])
                write_parameter(
                    output, f"{key}__components", ",".join(track_components(track))
                )
                write_parameter(
                    output, f"{key}__converged_expression", converged_expressions[key]
                )
                write_parameter(output, f"{key}__best_expression", best_expressions[key])
                write_parameter(
                    output, f"{key}__quality_score_expression", score_expressions[key]
                )
                write_parameter(output, f"{key}__target_expression", expressions[key])
                for suffix in (
                    "has_target_count", "other_passing_count", "has_converged_count",
                    "has_best_count", "unscorable_count",
                ):
                    write_parameter(output, f"{key}__{suffix}", counts[f"{key}__{suffix}"])
                for feature_name in features:
                    write_parameter(
                        output, f"{key}__{feature_name}__has_target_selected",
                        int(target_feature_counts[(key, feature_name)].GetValue()),
                    )
                    write_parameter(
                        output, f"{key}__{feature_name}__other_passing_selected",
                        int(other_feature_counts[(key, feature_name)].GetValue()),
                    )
                for feature_name in quality_features:
                    write_parameter(
                        output,
                        f"{key}__quality__{feature_name}__selected",
                        int(quality_feature_counts[(key, feature_name)].GetValue()),
                    )
            for algorithm_key, algorithm in algorithm_comparisons.items():
                write_parameter(
                    output, f"algorithm__{algorithm_key}__label", algorithm["label"]
                )
                write_parameter(
                    output,
                    f"algorithm__{algorithm_key}__scifi_component",
                    algorithm["scifi_component"],
                )
                write_parameter(
                    output,
                    f"algorithm__{algorithm_key}__ds_component",
                    algorithm["ds_component"],
                )
                for component in ("scifi", "ds"):
                    write_parameter(
                        output,
                        f"algorithm__{algorithm_key}__{component}_expression",
                        algorithm_component_expressions[algorithm_key][component],
                    )
                write_parameter(
                    output, f"algorithm__{algorithm_key}__four_way_closure", True
                )
                for combination in ALGORITHM_COMBINATION_ORDER:
                    write_parameter(
                        output,
                        f"algorithm__{algorithm_key}__{combination}__expression",
                        algorithm_expressions[algorithm_key][combination],
                    )
                    write_parameter(
                        output,
                        f"algorithm__{algorithm_key}__{combination}__count",
                        counts[
                            f"algorithm__{algorithm_key}__{combination}__count"
                        ],
                    )
                    for feature_name in features:
                        write_parameter(
                            output,
                            "algorithm__"
                            f"{algorithm_key}__{combination}__{feature_name}__selected",
                            int(
                                algorithm_feature_counts[
                                    (algorithm_key, combination, feature_name)
                                ].GetValue()
                            ),
                        )
            for left_index, left in enumerate(track_keys):
                for right_index in range(left_index, len(track_keys)):
                    right = track_keys[right_index]
                    value = counts[f"overlap__{left}__{right}"]
                    write_parameter(output, f"overlap__{left}__{right}", value)
                    if left != right:
                        write_parameter(output, f"overlap__{right}__{left}", value)
        finally:
            output.Close()
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(
        f"{args.working_point}: wrote {len(features)} features for "
        f"{len(track_keys)} track targets; passing={passing_count}, "
        f"missing={missing_count}, "
        f"mc_muon_up={counts['mc_muon_up_reference_count']}"
    )


if __name__ == "__main__":
    main()
