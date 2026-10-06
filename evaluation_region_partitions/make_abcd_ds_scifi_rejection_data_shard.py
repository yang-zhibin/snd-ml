#!/usr/bin/env python3
"""Materialize one full-2024-data shard for the DS--SciFi rejection study."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path

import ROOT

from abcd_ds_scifi_selection import (
    configured_cutflow,
    configured_region_expressions,
    configured_selection,
    cutflow_field,
    load_yaml,
)


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.TH1.AddDirectory(False)

STATES = ("before", "tagged", "retained")
COUNT_BRANCHES = [
    "count_scifi", *(f"count_scifi{i}" for i in range(1, 6)),
    "count_ds", *(f"count_ds{i}" for i in range(1, 5)),
]
BASE_SUMMARY_FIELDS = [
    "version", "fingerprint", "shard_index", "shard_count", "population",
    "label", "source_metadata_rows", "source_files_added", "source_files_skipped",
    "source_tree_entries", "source_luminosity", "n_before", "n_tagged",
    "n_retained",
]
VALIDATION_SUMMARY_FIELDS = [
    "negative_count_entries", "nonfinite_before", "nonfinite_tagged",
    "nonfinite_retained", "count_closure", "histogram_closure", "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def resolve_path(base: str, value: str) -> str:
    value = str(value)
    if value.startswith("root://") or value.startswith("/"):
        return value
    return f"{str(base).rstrip('/')}/{value.lstrip('/')}"


def source_rows(path: str, shard_index: int, shard_count: int, skip_missing: bool):
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"output_base_path", "feature_path", "lumi_per_file"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Source metadata lacks {sorted(required)}")
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    added, skipped = [], []
    for row in selected:
        feature_path = resolve_path(row["output_base_path"], row["feature_path"])
        if feature_path.startswith("/") and not os.path.exists(feature_path):
            if skip_missing:
                skipped.append(feature_path)
                continue
            raise FileNotFoundError(feature_path)
        added.append((feature_path, float(row.get("lumi_per_file") or 0.0)))
    if not added:
        raise ValueError(f"Shard {shard_index} has no available feature files")
    return len(selected), added, skipped


def identifiers(expression: str) -> set[str]:
    excluded = {"true", "false", "int", "float", "double", "std", "isfinite"}
    return {
        name for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression)
        if name not in excluded
    }


def atomic_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=f".{output.name}.",
        suffix=".tmp", dir=output.parent, delete=False,
    )
    temporary = handle.name
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def histogram_model(name: str, cfg: dict) -> ROOT.RDF.TH1DModel:
    return ROOT.RDF.TH1DModel(
        name, "", int(cfg["bins"]), float(cfg["min"]), float(cfg["max"])
    )


def fold_overflow(hist: ROOT.TH1) -> None:
    last = hist.GetNbinsX()
    hist.SetBinContent(last, hist.GetBinContent(last) + hist.GetBinContent(last + 1))
    hist.SetBinError(last, math.hypot(hist.GetBinError(last), hist.GetBinError(last + 1)))
    hist.SetBinContent(last + 1, 0.0); hist.SetBinError(last + 1, 0.0)


def integral_with_flow(hist: ROOT.TH1) -> float:
    return float(hist.Integral(0, hist.GetNbinsX() + 1))


def write_root(path: str, histograms: dict, metadata: dict) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".root", dir=output.parent)
    os.close(descriptor); os.unlink(temporary)
    try:
        root_file = ROOT.TFile(temporary, "RECREATE")
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not create {temporary}")
        for hist in histograms.values():
            hist.Write()
        for key, value in metadata.items():
            if isinstance(value, bool):
                ROOT.TParameter("int")(key, int(value)).Write()
            elif isinstance(value, int):
                ROOT.TParameter("long")(key, value).Write()
            elif isinstance(value, float):
                ROOT.TParameter("double")(key, value).Write()
            else:
                ROOT.TNamed(key, str(value)).Write()
        root_file.Close(); os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    region_config = load_yaml(args.region_config)
    selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    cutflow_fields = [cutflow_field(stage) for stage in cutflow]
    summary_fields = BASE_SUMMARY_FIELDS + cutflow_fields + VALIDATION_SUMMARY_FIELDS
    region_expressions = configured_region_expressions(config, region_config)
    if args.shard_count != int(config["input"]["data_shards"]):
        raise ValueError("CLI shard count differs from frozen configuration")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("Shard index is outside the configured range")
    selected_rows, files, skipped = source_rows(
        args.metadata, args.shard_index, args.shard_count,
        truthy(config["input"].get("skip_missing_source_files", False)),
    )
    chain = ROOT.TChain(str(config["input"]["tree_name"]))
    for path, _ in files:
        if chain.Add(path) != 1:
            raise OSError(f"Could not add source feature file: {path}")
    columns = {str(branch.GetName()) for branch in chain.GetListOfBranches()}
    required = set(selection["required_branches"])
    for expression in region_expressions.values():
        required.update(identifiers(expression))
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"Shard input lacks required branches: {missing}")

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    frame = ROOT.RDataFrame(chain)
    density = str(config["input"]["density_feature"])
    finite_expression = f"std::isfinite({density})"
    negative_expression = " || ".join(f"{branch} < 0" for branch in COUNT_BRANCHES)
    populations = {"total": selection["base_expression"]} | region_expressions
    actions, frames = {}, {}
    for population, population_expression in populations.items():
        before = frame.Filter(population_expression, f"{population}_before")
        cumulative = before
        actions[(population, "cutflow", cutflow[0]["key"])] = cumulative.Count()
        for stage in cutflow[1:]:
            cumulative = cumulative.Filter(
                stage["expression"], f"{population}_{stage['key']}"
            )
            actions[(population, "cutflow", stage["key"])] = cumulative.Count()
        tagged = cumulative
        retained = before.Filter(selection["keep_expression"], f"{population}_retained")
        frames[(population, "before")] = before
        frames[(population, "tagged")] = tagged
        frames[(population, "retained")] = retained
        actions[(population, "negative")] = before.Filter(negative_expression).Count()
        for state in STATES:
            selected_frame = frames[(population, state)]
            actions[(population, state, "count")] = selected_frame.Count()
            finite = selected_frame.Filter(finite_expression)
            actions[(population, state, "finite")] = finite.Count()
            for binning, hist_cfg in config["histograms"].items():
                name = f"h_{binning}_{population}_{state}"
                actions[(population, state, binning)] = finite.Histo1D(
                    histogram_model(name, hist_cfg), density
                )
    ROOT.RDF.RunGraphs(list(actions.values()))

    histograms, rows = {}, []
    count_values = {}
    histogram_ok = {}
    for population in populations:
        counts = {state: int(actions[(population, state, "count")].GetValue()) for state in STATES}
        finite = {state: int(actions[(population, state, "finite")].GetValue()) for state in STATES}
        cut_counts = [
            int(actions[(population, "cutflow", stage["key"])].GetValue())
            for stage in cutflow
        ]
        count_values[population] = counts
        negative = int(actions[(population, "negative")].GetValue())
        count_closure = counts["before"] == counts["tagged"] + counts["retained"]
        population_hist_ok = True
        for state in STATES:
            for binning, hist_cfg in config["histograms"].items():
                hist = actions[(population, state, binning)].GetValue().Clone(
                    f"h_{binning}_{population}_{state}"
                )
                hist.SetDirectory(0)
                if str(hist_cfg["overflow"]) == "fold_last":
                    fold_overflow(hist)
                elif str(hist_cfg["overflow"]) != "preserve":
                    raise ValueError(f"Unsupported overflow policy for {binning}")
                if str(hist_cfg["underflow"]) != "preserve":
                    raise ValueError(f"Unsupported underflow policy for {binning}")
                population_hist_ok &= math.isclose(
                    integral_with_flow(hist), float(finite[state]), rel_tol=0.0, abs_tol=1e-9
                )
                histograms[hist.GetName()] = hist
        histogram_ok[population] = population_hist_ok
        cutflow_ok = (
            cut_counts[0] == counts["before"]
            and cut_counts[-1] == counts["tagged"]
            and all(left >= right for left, right in zip(cut_counts, cut_counts[1:]))
        )
        status = (
            negative == 0 and count_closure and population_hist_ok
            and cutflow_ok and all(finite[state] == counts[state] for state in STATES)
        )
        rows.append({
            "version": config["version"], "fingerprint": selection["fingerprint"],
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "population": population,
            "label": "Total" if population == "total" else config["regions"][population],
            "source_metadata_rows": selected_rows, "source_files_added": len(files),
            "source_files_skipped": len(skipped), "source_tree_entries": int(chain.GetEntries()),
            "source_luminosity": f"{sum(value for _, value in files):.12g}",
            "n_before": counts["before"], "n_tagged": counts["tagged"],
            "n_retained": counts["retained"],
            **{
                cutflow_field(stage): value
                for stage, value in zip(cutflow, cut_counts)
            },
            "negative_count_entries": negative,
            "nonfinite_before": counts["before"] - finite["before"],
            "nonfinite_tagged": counts["tagged"] - finite["tagged"],
            "nonfinite_retained": counts["retained"] - finite["retained"],
            "count_closure": count_closure, "histogram_closure": population_hist_ok,
            "status": "PASS" if status else "FAIL",
        })
        if not status:
            raise ValueError(f"Shard validation failed for {population}")

    for state in STATES:
        regional = sum(count_values[region][state] for region in region_expressions)
        if regional != count_values["total"][state]:
            raise ValueError(
                f"Region-total closure failed for {state}: regions={regional}, "
                f"total={count_values['total'][state]}"
            )
    metadata = {
        "version": config["version"], "selection_fingerprint": selection["fingerprint"],
        "base_expression": selection["base_expression"],
        "scifi_expression": selection["scifi_expression"],
        "ds_expression": selection["ds_expression"],
        "tag_expression": selection["tag_expression"],
        "keep_expression": selection["keep_expression"],
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "source_metadata": str(Path(args.metadata).resolve()),
        "source_files_added": len(files), "source_files_skipped": len(skipped),
        "source_luminosity": sum(value for _, value in files),
        "region_total_closure": True,
    }
    total_row = next(row for row in rows if row["population"] == "total")
    metadata.update({field: int(total_row[field]) for field in cutflow_fields})
    write_root(args.output_root, histograms, metadata)
    atomic_csv(args.output_summary, rows, summary_fields)
    print(f"Wrote {args.output_root}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    build(parse_args())
