#!/usr/bin/env python3
"""Build matched veto-tagged transform and signal-region histograms for one source."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
import tempfile
from pathlib import Path

import ROOT
import yaml

from make_passing_muon_binary_region_hist_v5 import configured_v5_features
from make_passing_muon_track_type_hist import fold_1d, fold_2d_axis


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.TH1.AddDirectory(False)

VETO_TAGGED_POPULATIONS = ("passing_muon", "muon_dis")

COUNTER_LABELS = (
    "all RealDataDQ",
    "no-Veto signal region",
    "passing-muon candidate",
    "passing-muon eligible",
    "passing-muon matched",
    "passing-muon ineligible",
    "passing-muon eligible missing",
    "Muon-DIS candidate",
    "Muon-DIS eligible",
    "Muon-DIS matched",
    "Muon-DIS ineligible",
    "Muon-DIS eligible missing",
)

COUNTER_BINS = {label: index for index, label in enumerate(COUNTER_LABELS, 1)}


def classify_population(count_veto: int, passes_loose_v7: bool) -> str:
    """Return the original-event population used throughout the comparison."""
    count_veto = int(count_veto)
    if count_veto < 0:
        raise ValueError(f"Negative veto-hit count {count_veto}")
    if count_veto == 0:
        return "signal_region"
    return "passing_muon" if bool(passes_loose_v7) else "muon_dis"


def counter_label(population: str, suffix: str = "candidate") -> str:
    prefix = "passing-muon" if population == "passing_muon" else "Muon-DIS"
    return f"{prefix} {suffix}"


def validate_counter_closure(counter) -> None:
    values = {
        label: int(round(counter.GetBinContent(COUNTER_BINS[label])))
        for label in COUNTER_LABELS
    }
    classified = (
        values["no-Veto signal region"]
        + values["passing-muon candidate"]
        + values["Muon-DIS candidate"]
    )
    if values["all RealDataDQ"] != classified:
        raise RuntimeError("RealDataDQ population-count closure failed")
    for population in VETO_TAGGED_POPULATIONS:
        candidate = values[counter_label(population)]
        eligible = values[counter_label(population, "eligible")]
        ineligible = values[counter_label(population, "ineligible")]
        matched = values[counter_label(population, "matched")]
        missing = values[counter_label(population, "eligible missing")]
        if candidate != eligible + ineligible:
            raise RuntimeError(f"{population} eligibility-count closure failed")
        if eligible != matched + missing:
            raise RuntimeError(f"{population} match-count closure failed")


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def expand_expression(expression: str, definitions: dict[str, str]) -> str:
    result = str(expression)
    for _ in range(len(definitions) + 1):
        previous = result
        for name, value in sorted(definitions.items(), key=lambda item: -len(item[0])):
            result = re.sub(rf"\b{re.escape(name)}\b", f"({value})", result)
        if result == previous:
            return result
    raise RuntimeError(f"Recursive expression definitions in {expression!r}")


def make_formula(tree, name: str, expression: str):
    result = ROOT.TTreeFormula(name, expression, tree)
    if result.GetNdim() <= 0:
        raise RuntimeError(f"Invalid TTreeFormula {name}: {expression}")
    return result


def book_histogram(name: str, feature: dict):
    if int(feature.get("dimension", 1)) == 1:
        return ROOT.TH1D(
            name, "", int(feature["n_bins"]),
            float(feature["x_min"]), float(feature["x_max"]),
        )
    return ROOT.TH2D(
        name, "", int(feature["x_bins"]), float(feature["x_min"]),
        float(feature["x_max"]), int(feature["y_bins"]),
        float(feature["y_min"]), float(feature["y_max"]),
    )


def fold_histogram(histogram, feature: dict) -> None:
    if int(feature.get("dimension", 1)) == 1:
        fold_1d(
            histogram,
            bool(feature.get("fold_underflow", False)),
            bool(feature.get("fold_overflow", False)),
        )
        return
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


def mkdir(root, path: str):
    current = root
    for component in path.split("/"):
        current = current.GetDirectory(component) or current.mkdir(component)
    return current


def read_manifest(path: str) -> dict[tuple[int, int], dict]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        result = {}
        for row in csv.DictReader(stream):
            identity = (int(row["runId"]), int(row["eventId"]))
            if identity in result:
                raise RuntimeError(f"Duplicate manifest identity {identity} in {path}")
            result[identity] = row
    return result


def open_tree(path: str, tree_name: str):
    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {path}")
    tree = source.Get(tree_name)
    if not tree:
        raise RuntimeError(f"No {tree_name} tree in {path}")
    if tree.GetEntries() > 0 and tree.BuildIndex("runId", "eventId") < 0:
        raise RuntimeError(f"Cannot index {path}")
    return source, tree


def feature_expressions(feature: dict) -> list[str]:
    if int(feature.get("dimension", 1)) == 1:
        return [str(feature["expression"])]
    return [str(feature["x_expression"]), str(feature["y_expression"])]


def evaluate(formulas) -> list[float] | None:
    values = [float(item.EvalInstance()) for item in formulas]
    return values if all(math.isfinite(value) for value in values) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--simple", required=True)
    parser.add_argument("--reco-scifi", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    v7_config = load_yaml(config["inputs"]["v7_config"])
    histogram_config = load_yaml(config["inputs"]["feature_config"])
    features = configured_v5_features(histogram_config, v7_config)
    definitions = {
        **{str(k): str(v) for k, v in histogram_config.get("definitions", {}).items()},
        **{str(k): str(v) for k, v in v7_config.get("derived_columns", {}).items()},
    }
    manifest = read_manifest(args.manifest)
    tree_name = str(config.get("tree_name", "sndData"))
    original_file, original = open_tree(args.original, tree_name)
    simple_file, simple = open_tree(args.simple, tree_name)
    reco_file, reco = open_tree(args.reco_scifi, tree_name)
    transformed = {"simple": simple, "reco_scifi": reco}

    histograms = {}
    signal_histograms = {}
    formulas = {}
    signal_formulas = {}
    counters = {}
    for feature_name, feature in features.items():
        signal_histograms[feature_name] = book_histogram(
            f"signal_region_{feature_name}", feature
        )
        expressions = [
            expand_expression(value, definitions)
            for value in feature_expressions(feature)
        ]
        cut = expand_expression(feature.get("feature_cut", "1"), definitions)
        signal_formulas[feature_name] = {
            "values": [
                make_formula(original, f"s_{feature_name}_{i}", expression)
                for i, expression in enumerate(expressions)
            ],
            "cut": make_formula(original, f"sc_{feature_name}", cut),
        }
    for method in config["methods"]:
        counters[method] = ROOT.TH1D(
            f"counts_{method}", "", len(COUNTER_LABELS), 0.5, len(COUNTER_LABELS) + 0.5
        )
        for index, label in enumerate(COUNTER_LABELS, 1):
            counters[method].GetXaxis().SetBinLabel(index, label)
        target = transformed[method]
        for population in VETO_TAGGED_POPULATIONS:
            for feature_name, feature in features.items():
                key = (method, population, feature_name)
                histograms[key] = {
                    version: book_histogram(
                        f"{population}_{version}_{method}_{feature_name}", feature
                    )
                    for version in ("original", "transformed")
                }
                expressions = [
                    expand_expression(value, definitions)
                    for value in feature_expressions(feature)
                ]
                cut = expand_expression(feature.get("feature_cut", "1"), definitions)
                formulas[key] = {
                    "original_values": [
                        make_formula(
                            original,
                            f"o_{method}_{population}_{feature_name}_{i}",
                            expression,
                        )
                        for i, expression in enumerate(expressions)
                    ],
                    "transformed_values": [
                        make_formula(
                            target,
                            f"t_{method}_{population}_{feature_name}_{i}",
                            expression,
                        )
                        for i, expression in enumerate(expressions)
                    ],
                    "original_cut": make_formula(
                        original, f"oc_{method}_{population}_{feature_name}", cut
                    ),
                    "transformed_cut": make_formula(
                        target, f"tc_{method}_{population}_{feature_name}", cut
                    ),
                }

    for original_entry in range(int(original.GetEntries())):
        if original.GetEntry(original_entry) <= 0:
            raise RuntimeError(f"Cannot read original entry {original_entry}")
        identity = (int(original.runId), int(original.eventId))
        row = manifest.get(identity)
        if row is None:
            raise RuntimeError(f"Original event {identity} is absent from the manifest")
        population = classify_population(
            int(row["original_count_veto"]),
            int(row["original_passes_loose_v7"]) == 1,
        )

        if population == "signal_region":
            for feature_name in features:
                values = signal_formulas[feature_name]
                if not bool(values["cut"].EvalInstance()):
                    continue
                coordinates = evaluate(values["values"])
                if coordinates is not None:
                    signal_histograms[feature_name].Fill(*coordinates)

        for method, method_config in config["methods"].items():
            counter = counters[method]
            counter.Fill(COUNTER_BINS["all RealDataDQ"])
            if population == "signal_region":
                counter.Fill(COUNTER_BINS["no-Veto signal region"])
                continue

            counter.Fill(COUNTER_BINS[counter_label(population)])
            eligible = int(row[method_config["eligibility_field"]]) == 1
            counter.Fill(
                COUNTER_BINS[counter_label(population, "eligible")], int(eligible)
            )
            counter.Fill(
                COUNTER_BINS[counter_label(population, "ineligible")], int(not eligible)
            )
            target = transformed[method]
            target_entry = int(target.GetEntryNumberWithIndex(*identity))
            if target_entry < 0:
                if eligible:
                    counter.Fill(
                        COUNTER_BINS[counter_label(population, "eligible missing")]
                    )
                continue
            if not eligible:
                raise RuntimeError(
                    f"Ineligible {method} event {identity} has a transformed partner"
                )
            if target.GetEntry(target_entry) <= 0:
                raise RuntimeError(f"Cannot read transformed event {identity} for {method}")
            counter.Fill(COUNTER_BINS[counter_label(population, "matched")])
            for feature_name in features:
                key = (method, population, feature_name)
                values = formulas[key]
                if not (
                    bool(values["original_cut"].EvalInstance())
                    and bool(values["transformed_cut"].EvalInstance())
                ):
                    continue
                original_values = evaluate(values["original_values"])
                transformed_values = evaluate(values["transformed_values"])
                if original_values is None or transformed_values is None:
                    continue
                histograms[key]["original"].Fill(*original_values)
                histograms[key]["transformed"].Fill(*transformed_values)

    for counter in counters.values():
        validate_counter_closure(counter)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".root", dir=output.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        destination = ROOT.TFile.Open(temporary, "RECREATE")
        if not destination or destination.IsZombie():
            raise OSError(f"Cannot create {temporary}")
        for method in config["methods"]:
            directory = mkdir(destination, f"methods/{method}")
            directory.cd()
            counters[method].Write("counts")
            for population in VETO_TAGGED_POPULATIONS:
                for feature_name, feature in features.items():
                    directory = mkdir(
                        destination,
                        f"methods/{method}/populations/{population}/features/{feature_name}",
                    )
                    directory.cd()
                    for version, histogram in histograms[
                        (method, population, feature_name)
                    ].items():
                        fold_histogram(histogram, feature)
                        histogram.Write(version)
        for feature_name, feature in features.items():
            directory = mkdir(
                destination, f"reference/signal_region/features/{feature_name}"
            )
            directory.cd()
            fold_histogram(signal_histograms[feature_name], feature)
            signal_histograms[feature_name].Write("original")
        destination.cd()
        ROOT.TNamed("study_version", str(config["version"])).Write()
        ROOT.TNamed("feature_order", ",".join(features)).Write()
        destination.Close()
        os.replace(temporary, output)
    finally:
        original_file.Close()
        simple_file.Close()
        reco_file.Close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote Muon-DIS transform comparison shard to {output}")


if __name__ == "__main__":
    main()
