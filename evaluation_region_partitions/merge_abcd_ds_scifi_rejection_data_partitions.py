#!/usr/bin/env python3
"""Merge validated per-region real-data DS--SciFi partition products."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
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


def read_one(path):
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def cp_interval(passed, total, level):
    if total <= 0:
        return float("nan"), float("nan")
    return (
        float(ROOT.TEfficiency.ClopperPearson(total, passed, level, False)),
        float(ROOT.TEfficiency.ClopperPearson(total, passed, level, True)),
    )


def atomic_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output.parent,
        prefix=f".{output.name}.", suffix=".tmp", delete=False,
    )
    temporary = handle.name
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_root(path, histograms, metadata):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".root"
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        target = ROOT.TFile.Open(temporary, "RECREATE")
        if not target or target.IsZombie():
            raise OSError(temporary)
        for histogram in histograms.values():
            histogram.Write()
        for key, value in metadata.items():
            ROOT.TNamed(str(key), str(value)).Write()
        target.Close()
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge(args):
    config = load_yaml(args.config)
    selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    fields = [cutflow_field(stage) for stage in cutflow]
    regions = list(config["regions"])
    data_sample = str(config["input"].get("data_particle_group", "real_data"))
    if len(args.input_root) != len(regions) or len(args.input_summary) != len(regions):
        raise ValueError(f"Expected exactly {len(regions)} ROOT and CSV inputs")

    paired = []
    for root_path, summary_path in zip(args.input_root, args.input_summary):
        row = read_one(summary_path)
        if row["dataset"] != data_sample or row["role"] != "data":
            raise ValueError(f"Unexpected data identity in {summary_path}")
        if row["status"] != "PASS" or row["selection_fingerprint"] != selection["fingerprint"]:
            raise ValueError(f"Validation/fingerprint failure in {summary_path}")
        paired.append((row["population"], root_path, row))
    if {population for population, _, _ in paired} != set(regions):
        raise ValueError("Incomplete or duplicate real-data region coverage")
    by_region = {population: (root_path, row) for population, root_path, row in paired}

    histograms = {}
    total_histograms = {}
    for region in regions:
        root_path, row = by_region[region]
        expected_exposure = float(config["region_exposures"][region])
        tolerance = float(config["validation"]["luminosity_absolute_tolerance"])
        if not math.isclose(float(row["exposure_fb"]), expected_exposure, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"Exposure mismatch for {region}")
        source = ROOT.TFile.Open(root_path, "READ")
        if not source or source.IsZombie():
            raise OSError(root_path)
        for binning in config["histograms"]:
            for state in ("before", "tagged", "retained"):
                source_name = f"h_{binning}_{state}_raw"
                obj = source.Get(source_name)
                if not obj:
                    raise KeyError(f"{source_name} missing from {root_path}")
                name = f"h_{binning}_{region}_{state}"
                hist = obj.Clone(name)
                hist.SetDirectory(0)
                expected = int(row[f"n_{state}"])
                if not math.isclose(hist.Integral(0, hist.GetNbinsX() + 1), expected, abs_tol=1e-8):
                    raise ValueError(f"Histogram closure failed: {region} {state} {binning}")
                histograms[name] = hist
                total_name = f"h_{binning}_total_{state}"
                if total_name not in total_histograms:
                    total_histograms[total_name] = hist.Clone(total_name)
                    total_histograms[total_name].SetDirectory(0)
                else:
                    total_histograms[total_name].Add(hist)
        source.Close()
    histograms.update(total_histograms)

    ordered_rows = [by_region[region][1] for region in regions]
    summed = lambda key: sum(int(row[key]) for row in ordered_rows)
    before, tagged, retained = summed("n_before"), summed("n_tagged"), summed("n_retained")
    if before != tagged + retained:
        raise ValueError("Aggregate partition count closure failed")
    cut_counts = [summed(field) for field in fields]
    if cut_counts[0] != before or cut_counts[-1] != tagged or any(
        left < right for left, right in zip(cut_counts, cut_counts[1:])
    ):
        raise ValueError(f"Aggregate cutflow closure failed: {cut_counts}")
    category_fields = [
        f"category_{key}"
        for key in selection.get("diagnostic_categories", {})
    ]
    category_counts = {field: summed(field) for field in category_fields}
    exclusive = [
        "category_both_orientations", "category_y_only",
        "category_x_only", "category_no_position",
    ]
    if all(field in category_counts for field in exclusive) and sum(
        category_counts[field] for field in exclusive
    ) != before:
        raise ValueError("Aggregate Veto-availability categories do not close")
    level = float(config["statistics"]["confidence_level"])
    remove_lo, remove_hi = cp_interval(tagged, before, level)
    keep_lo, keep_hi = cp_interval(retained, before, level)
    total_row = {
        "dataset": data_sample,
        "population": "total",
        "label": "All stored regions (mixed exposure)",
        "role": "data",
        "exposure_fb": "mixed",
        "n_before": before,
        "n_tagged": tagged,
        "n_retained": retained,
        **{field: value for field, value in zip(fields, cut_counts)},
        **category_counts,
        "removed_fraction": f"{tagged / before:.12g}" if before else "nan",
        "removed_low": f"{remove_lo:.12g}",
        "removed_high": f"{remove_hi:.12g}",
        "keep_efficiency": f"{retained / before:.12g}" if before else "nan",
        "keep_low": f"{keep_lo:.12g}",
        "keep_high": f"{keep_hi:.12g}",
        "sumw_before": f"{before:.12g}",
        "sumw_tagged": f"{tagged:.12g}",
        "sumw_retained": f"{retained:.12g}",
        "sumw2_before": f"{before:.12g}",
        "sumw2_tagged": f"{tagged:.12g}",
        "sumw2_retained": f"{retained:.12g}",
        "selection_fingerprint": selection["fingerprint"],
        "status": "PASS",
    }
    if str(config["selection"].get("mode", "ds_scifi")) == "veto_edge_muon":
        total_row.update({
            "n_selected_edge": tagged,
            "n_nonedge": retained,
            "edge_efficiency": f"{tagged / before:.12g}" if before else "nan",
            "sumw_selected_edge": f"{tagged:.12g}",
            "sumw2_selected_edge": f"{tagged:.12g}",
        })
    if (
        str(config["selection"].get("mode", "ds_scifi"))
        == "veto_central_muon_rejection"
    ):
        total_row.update({
            "n_central_muon": tagged,
            "n_kept": retained,
            "central_muon_efficiency": (
                f"{tagged / before:.12g}" if before else "nan"
            ),
            "keep_efficiency_central_rejection": (
                f"{retained / before:.12g}" if before else "nan"
            ),
            "sumw_central_muon": f"{tagged:.12g}",
            "sumw_kept": f"{retained:.12g}",
        })
    for state, count in (("before", before), ("tagged", tagged), ("retained", retained)):
        for binning in config["histograms"]:
            hist = histograms[f"h_{binning}_total_{state}"]
            if not math.isclose(hist.Integral(0, hist.GetNbinsX() + 1), count, abs_tol=1e-8):
                raise ValueError(f"Aggregate histogram closure failed: {state} {binning}")

    metadata = {
        "version": config["version"],
        "selection_fingerprint": selection["fingerprint"],
        "partition_catalogue": str(Path(config["input"]["partition_catalogue"]).resolve()),
        "source_regions": len(regions),
        "exposure_semantics": "per-region configured exposure; aggregate is mixed exposure",
        "base_expression": selection["base_expression"],
        "tag_expression": selection["tag_expression"],
        "keep_expression": selection["keep_expression"],
        "validation_status": "PASS",
        **{field: value for field, value in zip(fields, cut_counts)},
        **category_counts,
    }
    write_rows = [total_row, *ordered_rows]
    atomic_root(args.output_root, histograms, metadata)
    atomic_csv(args.output_summary, write_rows)
    print(f"Merged {len(regions)} real-data regions; events={before}; status=PASS")


if __name__ == "__main__":
    merge(parse_args())
