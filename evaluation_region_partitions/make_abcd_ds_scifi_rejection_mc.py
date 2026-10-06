#!/usr/bin/env python3
"""Measure DS--SciFi retention in one Region V1 partition population."""

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
from make_abcd_ds_scifi_rejection_data_shard import fold_overflow, histogram_model

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.TH1.AddDirectory(False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-catalogue", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def selected_paths(path, config, region, sample):
    with open(path, encoding="utf-8", newline="") as stream:
        catalogue = list(csv.DictReader(stream))
    required = {
        "partition_version", "region_version", "region", "particle_group",
        "partition_id", "feature_available", "feature_partition_path",
        "n_events", "lumi_per_partition",
    }
    if not catalogue or not required.issubset(catalogue[0]):
        raise ValueError(f"Partition catalogue lacks {sorted(required)}")
    rows = [
        row for row in catalogue
        if row["partition_version"] == str(config["input"]["partition_version"])
        and row["region_version"] == str(config["input"]["region_version"])
        and row["region"] == region
        and row["particle_group"] == sample
    ]
    if not rows:
        raise ValueError(f"No catalogue rows for {region}/{sample}")
    partition_ids = [row["partition_id"] for row in rows]
    if len(partition_ids) != len(set(partition_ids)):
        raise ValueError(f"Duplicate partition IDs for {region}/{sample}")
    paths = []
    for row in rows:
        if str(row["feature_available"]).lower() not in {"true", "1"}:
            raise ValueError("Unavailable feature partition")
        feature = row["feature_partition_path"]
        if not Path(feature).is_file():
            raise FileNotFoundError(feature)
        paths.append(feature)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate feature paths for {region}/{sample}")
    return rows, paths


def cp(passed, total, level, upper):
    return float(ROOT.TEfficiency.ClopperPearson(total, passed, level, upper)) if total else float("nan")


def atomic_csv(path, row):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False)
    temporary = handle.name
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_root(path, histograms, metadata):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".root")
    os.close(fd); os.unlink(temporary)
    try:
        target = ROOT.TFile.Open(temporary, "RECREATE")
        if not target or target.IsZombie(): raise OSError(temporary)
        for histogram in histograms.values(): histogram.Write()
        for key, value in metadata.items(): ROOT.TNamed(str(key), str(value)).Write()
        target.Close(); os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def build(args):
    config = load_yaml(args.config)
    data_sample = str(config["input"].get("data_particle_group", "real_data"))
    is_data = args.sample == data_sample
    if args.region not in config["regions"] or (
        not is_data and args.sample not in config["mc_samples"]
    ):
        raise KeyError("Unconfigured region/sample")
    selection = configured_selection(config)
    cutflow = configured_cutflow(config)
    catalogue_rows, paths = selected_paths(
        args.partition_catalogue, config, args.region, args.sample
    )
    chain = ROOT.TChain(str(config["input"]["tree_name"]))
    for path in paths:
        if chain.Add(path) != 1: raise OSError(path)
    expected_entries = sum(int(row["n_events"]) for row in catalogue_rows)
    if int(chain.GetEntries()) != expected_entries:
        raise ValueError(
            f"Catalogue/tree count mismatch: {expected_entries} != {chain.GetEntries()}"
        )
    branches = {str(branch.GetName()) for branch in chain.GetListOfBranches()}
    missing = sorted(set(selection["required_branches"]) - branches)
    if missing: raise KeyError(f"Missing branches: {missing}")
    if args.threads > 1: ROOT.EnableImplicitMT(args.threads)
    frame = ROOT.RDataFrame(chain)
    density = str(config["input"]["density_feature"])
    finite = f"std::isfinite({density})"
    count_branches = selection.get(
        "count_branches",
        [
            "count_scifi", *(f"count_scifi{i}" for i in range(1, 6)),
            "count_ds", *(f"count_ds{i}" for i in range(1, 5)),
        ],
    )
    negative = " || ".join(f"{name} < 0" for name in count_branches)
    cutflow_frames = {}
    cumulative = frame.Filter(cutflow[0]["expression"], cutflow[0]["key"])
    cutflow_frames[cutflow[0]["key"]] = cumulative
    for stage in cutflow[1:]:
        cumulative = cumulative.Filter(stage["expression"], stage["key"])
        cutflow_frames[stage["key"]] = cumulative
    frames = {
        "before": frame,
        "tagged": cumulative,
        "retained": frame.Filter(selection["keep_expression"]),
    }
    actions = {"negative": frame.Filter(negative).Count()}
    for key, expression in selection.get("diagnostic_categories", {}).items():
        actions[("category", key)] = frame.Filter(expression, key).Count()
    for stage in cutflow:
        actions[("cutflow", stage["key"])] = cutflow_frames[stage["key"]].Count()
    for state, selected in frames.items():
        actions[(state, "count")] = selected.Count()
        finite_frame = selected.Filter(finite)
        actions[(state, "finite")] = finite_frame.Count()
        for binning, hist_cfg in config["histograms"].items():
            actions[(state, binning)] = finite_frame.Histo1D(histogram_model(f"h_{binning}_{state}_raw", hist_cfg), density)
    ROOT.RDF.RunGraphs(list(actions.values()))
    counts = {state: int(actions[(state, "count")].GetValue()) for state in frames}
    finite_counts = {state: int(actions[(state, "finite")].GetValue()) for state in frames}
    cut_counts = [
        int(actions[("cutflow", stage["key"])].GetValue()) for stage in cutflow
    ]
    category_counts = {
        key: int(actions[("category", key)].GetValue())
        for key in selection.get("diagnostic_categories", {})
    }
    if int(actions["negative"].GetValue()): raise ValueError("Negative detector count found")
    if counts["before"] != counts["tagged"] + counts["retained"]: raise ValueError("MC count closure failed")
    if (
        cut_counts[0] != counts["before"]
        or cut_counts[-1] != counts["tagged"]
        or any(left < right for left, right in zip(cut_counts, cut_counts[1:]))
    ): raise ValueError(f"MC cutflow closure failed: {cut_counts}")
    if counts != finite_counts: raise ValueError("Non-finite density value found")
    target_lumi = float(config["region_exposures"][args.region])
    if is_data:
        source_lumi = math.fsum(
            float(row["lumi_per_partition"]) for row in catalogue_rows
        )
        tolerance = float(config["validation"]["luminosity_absolute_tolerance"])
        if not math.isclose(source_lumi, target_lumi, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Data catalogue exposure mismatch: {source_lumi} != {target_lumi}"
            )
        weight = 1.0
        role = "data"
    else:
        source_lumi = float(config["mc_samples"][args.sample]["source_luminosity"])
        weight = target_lumi / source_lumi
        role = str(config["mc_samples"][args.sample]["role"])
    histograms = {}
    for state in frames:
        for binning, hist_cfg in config["histograms"].items():
            raw = actions[(state, binning)].GetValue().Clone(f"h_{binning}_{state}_raw")
            raw.SetDirectory(0)
            if hist_cfg["overflow"] == "fold_last": fold_overflow(raw)
            if not math.isclose(raw.Integral(0, raw.GetNbinsX() + 1), counts[state], abs_tol=1e-9): raise ValueError("MC histogram closure failed")
            weighted = raw.Clone(f"h_{binning}_{state}_weighted"); weighted.Scale(weight); weighted.SetDirectory(0)
            sumw2 = raw.Clone(f"h_{binning}_{state}_sumw2"); sumw2.Scale(weight * weight); sumw2.SetDirectory(0)
            histograms[raw.GetName()] = raw; histograms[weighted.GetName()] = weighted; histograms[sumw2.GetName()] = sumw2
    level = float(config["statistics"]["confidence_level"])
    before, tagged, retained = counts["before"], counts["tagged"], counts["retained"]
    row = {
        "dataset": args.sample, "population": args.region,
        "label": config["regions"][args.region],
        "role": role, "exposure_fb": f"{target_lumi:.12g}",
        "n_before": before, "n_tagged": tagged, "n_retained": retained,
        **{
            cutflow_field(stage): value
            for stage, value in zip(cutflow, cut_counts)
        },
        **{f"category_{key}": value for key, value in category_counts.items()},
        "removed_fraction": f"{tagged / before:.12g}" if before else "nan",
        "removed_low": f"{cp(tagged, before, level, False):.12g}", "removed_high": f"{cp(tagged, before, level, True):.12g}",
        "keep_efficiency": f"{retained / before:.12g}" if before else "nan",
        "keep_low": f"{cp(retained, before, level, False):.12g}", "keep_high": f"{cp(retained, before, level, True):.12g}",
        "sumw_before": f"{before * weight:.12g}", "sumw_tagged": f"{tagged * weight:.12g}", "sumw_retained": f"{retained * weight:.12g}",
        "sumw2_before": f"{before * weight * weight:.12g}", "sumw2_tagged": f"{tagged * weight * weight:.12g}", "sumw2_retained": f"{retained * weight * weight:.12g}",
        "selection_fingerprint": selection["fingerprint"], "status": "PASS",
    }
    if str(config["selection"].get("mode", "ds_scifi")) == "veto_edge_muon":
        row.update({
            "n_selected_edge": tagged,
            "n_nonedge": retained,
            "edge_efficiency": f"{tagged / before:.12g}" if before else "nan",
            "sumw_selected_edge": f"{tagged * weight:.12g}",
            "sumw2_selected_edge": f"{tagged * weight * weight:.12g}",
        })
    if (
        str(config["selection"].get("mode", "ds_scifi"))
        == "veto_central_muon_rejection"
    ):
        row.update({
            "n_central_muon": tagged,
            "n_kept": retained,
            "central_muon_efficiency": (
                f"{tagged / before:.12g}" if before else "nan"
            ),
            "keep_efficiency_central_rejection": (
                f"{retained / before:.12g}" if before else "nan"
            ),
            "sumw_central_muon": f"{tagged * weight:.12g}",
            "sumw_kept": f"{retained * weight:.12g}",
        })
    metadata = {
        "version": config["version"], "selection_fingerprint": selection["fingerprint"],
        "partition_catalogue": str(Path(args.partition_catalogue).resolve()), "region": args.region,
        "sample": args.sample, "catalogue_rows": len(catalogue_rows), "source_luminosity_fb": source_lumi,
        "target_luminosity_fb": target_lumi, "normalization_weight": weight,
        "tag_expression": selection["tag_expression"], "keep_expression": selection["keep_expression"],
        "validation_status": "PASS",
    }
    metadata.update({
        cutflow_field(stage): value for stage, value in zip(cutflow, cut_counts)
    })
    metadata.update({
        f"category_{key}": value for key, value in category_counts.items()
    })
    atomic_root(args.output_root, histograms, metadata)
    atomic_csv(args.output_summary, row)
    if str(config["selection"].get("mode", "ds_scifi")) == "veto_edge_muon":
        print(
            f"{args.region}/{args.sample}: {before} -> "
            f"{tagged} selected edge muons, "
            f"efficiency={tagged / before if before else float('nan'):.6g}"
        )
    elif (
        str(config["selection"].get("mode", "ds_scifi"))
        == "veto_central_muon_rejection"
    ):
        print(
            f"{args.region}/{args.sample}: {before} -> "
            f"{tagged} central muons rejected, {retained} kept, "
            f"central efficiency={tagged / before if before else float('nan'):.6g}"
        )
    else:
        print(f"{args.region}/{args.sample}: {before} -> {retained}, efficiency={retained / before if before else float('nan'):.6g}")


if __name__ == "__main__":
    build(parse_args())
