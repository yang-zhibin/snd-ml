#!/usr/bin/env python3

import os
import re
import csv
import math
import glob
import argparse
from collections import defaultdict

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


KAON_BINS = [
    "MC_kaon_5-10GeV",
    "MC_kaon_10-20GeV",
    "MC_kaon_20-30GeV",
    "MC_kaon_30-40GeV",
    "MC_kaon_50-60GeV",
    "MC_kaon_60-70GeV",
    "MC_kaon_70-80GeV",
    "MC_kaon_80-90GeV",
    "MC_kaon_90-100GeV",
    "MC_kaon_100-150GeV",
]

NEUTRON_BINS = [
    "MC_neutron_5-10GeV",
    "MC_neutron_10-20GeV",
    "MC_neutron_20-30GeV",
    "MC_neutron_30-40GeV",
    "MC_neutron_50-60GeV",
    "MC_neutron_60-70GeV",
    "MC_neutron_70-80GeV",
    "MC_neutron_80-90GeV",
    "MC_neutron_90-100GeV",
    "MC_neutron_100-150GeV",
]

NEUTRINO_CATS = [
    "MC_NC_nue",
    "MC_NC_numu",
    "MC_CC_nue",
    "MC_CC_numu",
]


def get_total_lumi(root_file):
    obj = root_file.Get("total_lumi")
    if not obj:
        raise RuntimeError(f"Could not find 'total_lumi' in {root_file.GetName()}")
    return float(obj.GetVal())


def partition_from_filename(filepath):
    base = os.path.basename(filepath)
    m = re.match(r"^eff_(.+)\.root$", base)
    return m.group(1) if m else None


def classify_partition(partition):
    if partition.startswith("real_data"):
        return "real_data"

    if partition in set(NEUTRINO_CATS):
        return partition

    m = re.match(r"^(MC_(?:kaon|neutron)_\d+-\d+GeV)(?:_Max\d+-\d+)?$", partition)
    if m:
        return m.group(1)

    return None


def is_valid_number(x):
    return math.isfinite(float(x))


def clone_hist(hist, new_name):
    if hist is None:
        return None
    out = hist.Clone(new_name)
    out.SetDirectory(0)
    return out


def zero_hist(hist):
    if hist is None:
        return
    hist.Reset("ICES")
    hist.SetDirectory(0)


def collect_files_and_lumi(input_dir):
    filepaths = sorted(glob.glob(os.path.join(input_dir, "eff_*.root")))
    if not filepaths:
        raise RuntimeError(f"No files matching eff_*.root found in {input_dir}")

    category_files = defaultdict(list)
    grouped_lumi = defaultdict(float)
    open_files = []

    for filepath in filepaths:
        partition = partition_from_filename(filepath)
        if partition is None:
            continue

        category = classify_partition(partition)
        if category is None:
            print(f"[skip] unrecognized partition: {partition}")
            continue

        root_file = ROOT.TFile.Open(filepath)
        if not root_file or root_file.IsZombie():
            print(f"[skip] could not open {filepath}")
            continue

        open_files.append(root_file)

        try:
            lumi = get_total_lumi(root_file)
        except Exception as e:
            print(f"[skip] failed to read lumi from {filepath}: {e}")
            continue

        category_files[category].append(filepath)
        grouped_lumi[category] += lumi

        print(f"[ok] {partition:35s} -> {category:25s} lumi={lumi}")

    if "real_data" not in category_files:
        raise RuntimeError("No real_data files found.")

    return category_files, grouped_lumi, open_files


def read_category_count_hists(category_files):
    grouped_counts = {}

    for category, files in category_files.items():
        summed = None

        for i, filepath in enumerate(files):
            root_file = ROOT.TFile.Open(filepath)
            if not root_file or root_file.IsZombie():
                print(f"[skip] could not open {filepath}")
                continue

            hist = root_file.Get("cutflow_counts")
            if hist is None:
                print(f"[skip] missing cutflow_counts in {filepath}")
                root_file.Close()
                continue

            hist = clone_hist(hist, f"tmp_{category}_{i}")

            if summed is None:
                summed = clone_hist(hist, f"count_{category}")
            else:
                summed.Add(hist)

            root_file.Close()

        if summed is None:
            print(f"[warn] no cutflow_counts found for {category}")
            continue

        grouped_counts[category] = summed
        print(f"[count] {category:25s} integral={summed.Integral(0, summed.GetNbinsX()+1):.6g}")

    return grouped_counts


def scale_mc_counts_to_data(grouped_counts, grouped_lumi, data_category="real_data"):
    data_lumi = grouped_lumi[data_category]
    if not is_valid_number(data_lumi) or data_lumi <= 0:
        raise RuntimeError(f"Invalid data lumi: {data_lumi}")

    print(f"\nTotal data lumi = {data_lumi:.6g}")

    for category, hist in grouped_counts.items():
        if category == data_category:
            continue

        mc_lumi = grouped_lumi.get(category, float("nan"))
        if not is_valid_number(mc_lumi) or mc_lumi <= 0:
            zero_hist(hist)
            print(f"[scale] {category:25s}: set to 0 (invalid lumi = {mc_lumi})")
            continue

        scale = data_lumi / mc_lumi
        if not is_valid_number(scale):
            zero_hist(hist)
            print(f"[scale] {category:25s}: set to 0 (invalid scale = {scale})")
            continue

        hist.Scale(scale)
        print(f"[scale] {category:25s}: x {scale:.6g}")

    return data_lumi


def combine_categories(grouped_counts, categories, output_name):
    combined = None
    for cat in categories:
        hist = grouped_counts.get(cat)
        if hist is None:
            continue

        if combined is None:
            combined = clone_hist(hist, output_name)
        else:
            combined.Add(hist)

    return combined


def build_final_count_hists(grouped_counts):
    final_counts = {}

    if "real_data" not in grouped_counts:
        raise RuntimeError("Missing real_data in grouped_counts")

    final_counts["data"] = clone_hist(grouped_counts["real_data"], "count_data")

    kaon = combine_categories(grouped_counts, KAON_BINS, "count_MC_kaon")
    neutron = combine_categories(grouped_counts, NEUTRON_BINS, "count_MC_neutron")

    if kaon is not None:
        final_counts["MC_kaon"] = kaon
    if neutron is not None:
        final_counts["MC_neutron"] = neutron

    for cat in NEUTRINO_CATS:
        if cat in grouped_counts:
            final_counts[cat] = clone_hist(grouped_counts[cat], f"count_{cat}")

    return final_counts


def compute_efficiencies_from_counts(count_hist, prefix):
    cumeff = clone_hist(count_hist, f"{prefix}_cumeff")
    releff = clone_hist(count_hist, f"{prefix}_releff")

    nb = count_hist.GetNbinsX()
    first = count_hist.GetBinContent(1)

    for i in range(1, nb + 1):
        current = count_hist.GetBinContent(i)
        prev = count_hist.GetBinContent(i - 1) if i > 1 else first

        cum = current / first if first > 0 else 0.0
        rel = current / prev if prev > 0 else 0.0

        cumeff.SetBinContent(i, cum)
        cumeff.SetBinError(i, 0.0)

        releff.SetBinContent(i, rel)
        releff.SetBinError(i, 0.0)

    return cumeff, releff


def hist_to_rows(count_hist, cumeff_hist, releff_hist):
    rows = []
    nb = count_hist.GetNbinsX()

    for i in range(1, nb + 1):
        label = count_hist.GetXaxis().GetBinLabel(i)
        if not label:
            label = f"bin_{i}"

        rows.append({
            "cut_index": i,
            "cut_name": label,
            "count": count_hist.GetBinContent(i),
            "cumeff": cumeff_hist.GetBinContent(i),
            "releff": releff_hist.GetBinContent(i),
        })

    return rows


def write_category_csv(outdir, category, rows):
    path = os.path.join(outdir, f"cutflow_{category}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cut_index", "cut_name", "count", "cumeff", "releff"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def write_summary_csv(outdir, category_rows, ordered_categories):
    if not ordered_categories:
        return

    first_cat = ordered_categories[0]
    nrows = len(category_rows[first_cat])

    path = os.path.join(outdir, "cutflow_summary.csv")
    with open(path, "w", newline="") as f:
        fieldnames = ["cut_index", "cut_name"]
        for cat in ordered_categories:
            fieldnames += [f"{cat}_count", f"{cat}_cumeff", f"{cat}_releff"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(nrows):
            row = {
                "cut_index": category_rows[first_cat][i]["cut_index"],
                "cut_name": category_rows[first_cat][i]["cut_name"],
            }

            for cat in ordered_categories:
                row[f"{cat}_count"] = category_rows[cat][i]["count"]
                row[f"{cat}_cumeff"] = category_rows[cat][i]["cumeff"]
                row[f"{cat}_releff"] = category_rows[cat][i]["releff"]

            writer.writerow(row)

    print(f"[saved] {path}")

def fmt(x):
    if x == 0:
        return "0"

    x = float(x)

    if abs(x) < 1e-3:
        return f"{x:.3e}"   # scientific notation
    else:
        return f"{x:.6f}".rstrip("0").rstrip(".")

def print_cut_table(category, rows):
    print(f"\n=== {category} ===")
    print(f"{'#':>2s}  {'cut':<24s}  {'count':>14s}  {'cumeff(%)':>12s}  {'releff(%)':>12s}")

    for row in rows:
        print(
            f"{row['cut_index']:2d}  "
            f"{row['cut_name']:<24.24s}  "
            f"{fmt(row['count']):>14s}  "
            f"{fmt(row['cumeff']*100):>12s}  "
            f"{fmt(row['releff']*100):>12s}"
        )


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    category_files, grouped_lumi, open_files = collect_files_and_lumi(args.input_dir)

    grouped_counts = read_category_count_hists(category_files)

    scale_mc_counts_to_data(grouped_counts, grouped_lumi, data_category="real_data")

    final_counts = build_final_count_hists(grouped_counts)

    ordered_categories = [
        "data",
        "MC_kaon",
        "MC_neutron",
        "MC_CC_nue",
        "MC_CC_numu",
        "MC_NC_nue",
        "MC_NC_numu",
    ]

    category_rows = {}

    for category in ordered_categories:
        hist = final_counts.get(category)
        if hist is None:
            continue

        cumeff, releff = compute_efficiencies_from_counts(hist, category)
        rows = hist_to_rows(hist, cumeff, releff)
        category_rows[category] = rows

        write_category_csv(args.outdir, category, rows)
        print_cut_table(category, rows)
        
        

    existing_categories = [cat for cat in ordered_categories if cat in category_rows]
    write_summary_csv(args.outdir, category_rows, existing_categories)

    _ = open_files




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cutflow table builder from eff_{partition}.root files.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/user/z/zhibin/nueAnalysis",
        help="Directory containing eff_{partition}.root files",
    )
    parser.add_argument(
        "--outdir",
        default="eff_plots",
        help="Output directory",
    )

    args = parser.parse_args()
    main(args)