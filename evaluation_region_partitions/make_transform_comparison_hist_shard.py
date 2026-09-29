#!/usr/bin/env python3
"""Build event-matched original/transformed feature histograms for one source."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import ROOT
import yaml

from make_passing_muon_binary_region_hist_v5 import configured_v5_features


ROOT.gROOT.SetBatch(True)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def expand_expression(expression: str, definitions: dict[str, str]) -> str:
    result = str(expression)
    for _ in range(len(definitions) + 1):
        previous = result
        for name, value in sorted(definitions.items(), key=lambda item: -len(item[0])):
            result = re.sub(rf"\b{re.escape(name)}\b", f"({value})", result)
        if result == previous:
            return result
    raise RuntimeError(f"Recursive expression definitions in {expression!r}")


def formula(tree, name: str, expression: str):
    value = ROOT.TTreeFormula(name, expression, tree)
    if value.GetNdim() <= 0:
        raise RuntimeError(f"Invalid TTreeFormula {name}: {expression}")
    return value


def book(name: str, definition: dict):
    if int(definition.get("dimension", 1)) == 1:
        return ROOT.TH1D(
            name, "", int(definition["n_bins"]),
            float(definition["x_min"]), float(definition["x_max"]),
        )
    return ROOT.TH2D(
        name, "", int(definition["x_bins"]), float(definition["x_min"]),
        float(definition["x_max"]), int(definition["y_bins"]),
        float(definition["y_min"]), float(definition["y_max"]),
    )


def mkdir(root, path: str):
    current = root
    for component in path.split("/"):
        found = current.GetDirectory(component)
        current = found or current.mkdir(component)
    return current


def manifest_flags(path: str) -> dict[tuple[int, int], dict]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return {
            (int(row["runId"]), int(row["eventId"])): row
            for row in csv.DictReader(stream)
        }


def open_tree(path: str):
    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {path}")
    tree = source.Get("sndData")
    if not tree:
        raise RuntimeError(f"No sndData tree in {path}")
    if tree.BuildIndex("runId", "eventId") < 0:
        raise RuntimeError(f"Cannot index {path}")
    return source, tree


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
    v7 = load_yaml(config["inputs"]["v7_config"])
    histogram_config = load_yaml(config["inputs"]["feature_config"])
    features = configured_v5_features(histogram_config, v7)
    definitions = {
        **{str(k): str(v) for k, v in histogram_config.get("definitions", {}).items()},
        **{str(k): str(v) for k, v in v7.get("derived_columns", {}).items()},
    }
    flags = manifest_flags(args.manifest)
    original_file, original = open_tree(args.original)
    simple_file, simple = open_tree(args.simple)
    reco_file, reco = open_tree(args.reco_scifi)
    transformed = {"simple": simple, "reco_scifi": reco}
    transformed_files = {"simple": simple_file, "reco_scifi": reco_file}

    comparisons = OrderedDict(config["comparisons"])
    histograms = {}
    formulas = {}
    counters = {}
    for comparison, comparison_config in comparisons.items():
        method = comparison_config["method"]
        target = transformed[method]
        counters[comparison] = ROOT.TH1D(f"counts_{comparison}", "", 4, 0.5, 4.5)
        counters[comparison].GetXaxis().SetBinLabel(1, "original population")
        counters[comparison].GetXaxis().SetBinLabel(2, "matched transform")
        counters[comparison].GetXaxis().SetBinLabel(3, "feature-valid pairs")
        counters[comparison].GetXaxis().SetBinLabel(4, "missing transform")
        for feature, definition in features.items():
            key = (comparison, feature)
            histograms[key] = {
                "original": book(f"original_{comparison}_{feature}", definition),
                "transformed": book(f"transformed_{comparison}_{feature}", definition),
            }
            dimension = int(definition.get("dimension", 1))
            expressions = (
                [definition["expression"]] if dimension == 1 else
                [definition["x_expression"], definition["y_expression"]]
            )
            formulas[key] = {
                "original_values": [
                    formula(original, f"o_{comparison}_{feature}_{i}", expand_expression(value, definitions))
                    for i, value in enumerate(expressions)
                ],
                "target_values": [
                    formula(target, f"t_{comparison}_{feature}_{i}", expand_expression(value, definitions))
                    for i, value in enumerate(expressions)
                ],
                "original_cut": formula(
                    original, f"oc_{comparison}_{feature}",
                    expand_expression(definition.get("feature_cut", "1"), definitions),
                ),
                "target_cut": formula(
                    target, f"tc_{comparison}_{feature}",
                    expand_expression(definition.get("feature_cut", "1"), definitions),
                ),
            }

    for original_entry in range(int(original.GetEntries())):
        original.GetEntry(original_entry)
        identity = (int(original.runId), int(original.eventId))
        manifest = flags.get(identity)
        if manifest is None:
            raise RuntimeError(f"Original feature event {identity} is absent from manifest")
        for comparison, comparison_config in comparisons.items():
            selection = comparison_config["original_selection"]
            selected = (
                (selection == "reco_scifi_eligible" and int(manifest["reco_scifi_eligible"]) == 1)
                or (selection == "count_veto == 0" and int(manifest["original_count_veto"]) == 0)
                or (selection == "original_passes_loose_v7" and int(manifest["original_passes_loose_v7"]) == 1)
            )
            if not selected:
                continue
            counters[comparison].Fill(1)
            target = transformed[comparison_config["method"]]
            target_entry = int(target.GetEntryNumberWithIndex(*identity))
            if target_entry < 0:
                counters[comparison].Fill(4)
                continue
            target.GetEntry(target_entry)
            counters[comparison].Fill(2)
            any_valid = False
            for feature, definition in features.items():
                values = formulas[(comparison, feature)]
                if not bool(values["original_cut"].EvalInstance()) or not bool(
                    values["target_cut"].EvalInstance()
                ):
                    continue
                original_values = [float(value.EvalInstance()) for value in values["original_values"]]
                target_values = [float(value.EvalInstance()) for value in values["target_values"]]
                if not all(math.isfinite(value) for value in original_values + target_values):
                    continue
                histograms[(comparison, feature)]["original"].Fill(*original_values)
                histograms[(comparison, feature)]["transformed"].Fill(*target_values)
                any_valid = True
            counters[comparison].Fill(3, int(any_valid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".root", dir=output.parent)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        destination = ROOT.TFile.Open(temporary, "RECREATE")
        for comparison in comparisons:
            directory = mkdir(destination, f"comparisons/{comparison}")
            directory.cd()
            counters[comparison].Write("counts")
            for feature in features:
                directory = mkdir(destination, f"comparisons/{comparison}/features/{feature}")
                directory.cd()
                histograms[(comparison, feature)]["original"].Write("original")
                histograms[(comparison, feature)]["transformed"].Write("transformed")
        destination.cd()
        ROOT.TNamed("study_version", config["version"]).Write()
        ROOT.TNamed("feature_order", ",".join(features)).Write()
        destination.Close()
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
        original_file.Close()
        for source in transformed_files.values():
            source.Close()
    print(f"Wrote matched comparison histograms to {output}")


if __name__ == "__main__":
    main()
