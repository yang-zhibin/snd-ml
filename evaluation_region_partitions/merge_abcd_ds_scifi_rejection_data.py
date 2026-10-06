#!/usr/bin/env python3
"""Merge and validate full-data DS--SciFi rejection shards."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import ROOT

from abcd_ds_scifi_selection import (
    configured_cutflow,
    configured_selection,
    cutflow_field,
    load_yaml,
)

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--input-summary", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def cp_interval(passed, total, level):
    if total <= 0:
        return float("nan"), float("nan")
    return (
        float(ROOT.TEfficiency.ClopperPearson(total, passed, level, False)),
        float(ROOT.TEfficiency.ClopperPearson(total, passed, level, True)),
    )


def write_csv(path, rows):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False)
    temporary = handle.name
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def write_root(path, histograms, metadata):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".root")
    os.close(fd); os.unlink(temporary)
    try:
        root_file = ROOT.TFile.Open(temporary, "RECREATE")
        if not root_file or root_file.IsZombie(): raise OSError(temporary)
        for hist in histograms.values(): hist.Write()
        for key, value in metadata.items():
            ROOT.TNamed(str(key), str(value)).Write()
        root_file.Close(); os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def merge(args):
    config = load_yaml(args.config)
    selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    cutflow_fields = [cutflow_field(stage) for stage in cutflow]
    expected_shards = int(config["input"]["data_shards"])
    if len(args.input_root) != expected_shards or len(args.input_summary) != expected_shards:
        raise ValueError(f"Expected exactly {expected_shards} ROOT and CSV shards")
    rows = read_rows(args.input_summary)
    populations = ["total", *config["regions"]]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["population"]].append(row)
    for population in populations:
        selected = grouped[population]
        indices = sorted(int(row["shard_index"]) for row in selected)
        if indices != list(range(expected_shards)):
            raise ValueError(f"Incomplete/duplicate shards for {population}: {indices}")
        if any(row["fingerprint"] != selection["fingerprint"] or row["status"] != "PASS" for row in selected):
            raise ValueError(f"Configuration/status mismatch in {population}")

    histograms = {}
    expected_names = [
        f"h_{binning}_{population}_{state}"
        for binning in config["histograms"] for population in populations
        for state in ("before", "tagged", "retained")
    ]
    for shard_path in args.input_root:
        source = ROOT.TFile.Open(shard_path, "READ")
        if not source or source.IsZombie(): raise OSError(shard_path)
        for name in expected_names:
            obj = source.Get(name)
            if not obj: raise KeyError(f"{name} missing from {shard_path}")
            if name not in histograms:
                histograms[name] = obj.Clone(name); histograms[name].SetDirectory(0)
            else:
                histograms[name].Add(obj)
        source.Close()

    level = float(config["statistics"]["confidence_level"])
    merged_rows = []
    totals = {}
    total_lumi = None
    for population in populations:
        selected = grouped[population]
        summed = lambda key: sum(int(row[key]) for row in selected)
        before, tagged, retained = (summed("n_before"), summed("n_tagged"), summed("n_retained"))
        if before != tagged + retained: raise ValueError(f"Count closure failed for {population}")
        cut_counts = [summed(field) for field in cutflow_fields]
        if (
            cut_counts[0] != before
            or cut_counts[-1] != tagged
            or any(left < right for left, right in zip(cut_counts, cut_counts[1:]))
        ):
            raise ValueError(f"Cutflow closure failed for {population}: {cut_counts}")
        if summed("negative_count_entries") or summed("nonfinite_before") or summed("nonfinite_tagged") or summed("nonfinite_retained"):
            raise ValueError(f"Invalid detector/density values in {population}")
        for state, count in (("before", before), ("tagged", tagged), ("retained", retained)):
            for binning in config["histograms"]:
                hist = histograms[f"h_{binning}_{population}_{state}"]
                if not math.isclose(hist.Integral(0, hist.GetNbinsX() + 1), count, abs_tol=1e-8):
                    raise ValueError(f"Histogram closure failed: {population} {state} {binning}")
        remove_lo, remove_hi = cp_interval(tagged, before, level)
        keep_lo, keep_hi = cp_interval(retained, before, level)
        luminosity = sum(float(row["source_luminosity"]) for row in selected)
        if total_lumi is None: total_lumi = luminosity
        elif not math.isclose(luminosity, total_lumi, abs_tol=1e-9):
            raise ValueError("Population luminosities differ in full-source evaluation")
        totals[population] = (before, tagged, retained)
        merged_rows.append({
            "dataset": "data_full_source", "population": population,
            "label": "Total" if population == "total" else config["regions"][population],
            "role": "data", "exposure_fb": f"{luminosity:.12g}",
            "n_before": before, "n_tagged": tagged, "n_retained": retained,
            **{
                field: value for field, value in zip(cutflow_fields, cut_counts)
            },
            "removed_fraction": f"{tagged / before:.12g}" if before else "nan",
            "removed_low": f"{remove_lo:.12g}", "removed_high": f"{remove_hi:.12g}",
            "keep_efficiency": f"{retained / before:.12g}" if before else "nan",
            "keep_low": f"{keep_lo:.12g}", "keep_high": f"{keep_hi:.12g}",
            "sumw_before": f"{before:.12g}", "sumw_tagged": f"{tagged:.12g}",
            "sumw_retained": f"{retained:.12g}",
            "sumw2_before": f"{before:.12g}", "sumw2_tagged": f"{tagged:.12g}",
            "sumw2_retained": f"{retained:.12g}",
            "selection_fingerprint": selection["fingerprint"], "status": "PASS",
        })
    for state, index in (("before", 0), ("tagged", 1), ("retained", 2)):
        regional = sum(totals[population][index] for population in config["regions"])
        if regional != totals["total"][index]: raise ValueError(f"Region-total closure failed for {state}")
    expected_lumi = float(config["input"]["expected_full_data_luminosity"])
    tolerance = float(config["validation"]["luminosity_absolute_tolerance"])
    if not math.isclose(total_lumi, expected_lumi, abs_tol=tolerance):
        raise ValueError(f"Full luminosity mismatch: {total_lumi} != {expected_lumi}")
    metadata = {
        "version": config["version"], "selection_fingerprint": selection["fingerprint"],
        "source_shards": expected_shards, "source_luminosity_fb": total_lumi,
        "base_expression": selection["base_expression"], "tag_expression": selection["tag_expression"],
        "keep_expression": selection["keep_expression"], "validation_status": "PASS",
    }
    total_row = next(row for row in merged_rows if row["population"] == "total")
    metadata.update({field: int(total_row[field]) for field in cutflow_fields})
    write_root(args.output_root, histograms, metadata)
    write_csv(args.output_summary, merged_rows)
    print(f"Merged {expected_shards} shards; luminosity={total_lumi:.12g} fb^-1; status=PASS")


if __name__ == "__main__":
    merge(parse_args())
