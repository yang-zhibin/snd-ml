#!/usr/bin/env python3

import os
import math
import argparse
from collections import Counter

import ROOT
import pandas as pd
import re

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()


# ============================================================
# HIST CONFIG
# ============================================================
hist_info = {
    "primary_xy": {
        "dim": 2,
        "title": "Primary start X vs Y",
        "xexpr": "primary_startX",
        "yexpr": "primary_startY",
        "xbins": (200, -200.0, 200.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": False,
    },
    
    "avg_scifi": {
        "dim": 2,
        "title": "Average Scifi Position",
        "xexpr": "avg_scifi_ver_x",
        "yexpr": "avg_scifi_hor_y",
        "xbins": (200, -80.0, 20.0),
        "ybins": (200, -20.0, 80.0),
        "log": False,
        "selected": False,
    },
    
    "avg_scifi_sel": {
        "dim": 2,
        "title": "Average Scifi Position",
        "xexpr": "avg_scifi_ver_x",
        "yexpr": "avg_scifi_hor_y",
        "xbins": (200, -80.0, 20.0),
        "ybins": (200, -20.0, 80.0),
        "log": False,
        "selected": True,
    },
    
    "density_sndsw_scifi": {
        "dim": 1,
        "title": "Sum of SciFi Density Weight (SNDSW)",
        "xexpr": "density_sndsw_scifi",
        "xbins": (200, 0, 6000),
        "log": True,
        "selected": True,
    },

    "count_scifi": {
        "dim": 1,
        "title": "SciFi Hit Total Count",
        "xexpr": "count_scifi",
        "xbins": (100, 0, 800),
        "log": True,
        "selected": True,
    },

    "count_scifi1": {
        "dim": 1,
        "title": "Plane1 SciFi Hit Total Count",
        "xexpr": "count_scifi1",
        "xbins": (50, 0, 50),
        "log": True,
        "selected": True,
    },

    "count_scifi2": {
        "dim": 1,
        "title": "Plane2 SciFi Hit Total Count",
        "xexpr": "count_scifi2",
        "xbins": (50, 0, 50),
        "log": True,
        "selected": True,
    },

    "count_scifi3": {
        "dim": 1,
        "title": "Plane3 SciFi Hit Total Count",
        "xexpr": "count_scifi3",
        "xbins": (50, 0, 500),
        "log": True,
        "selected": True,
    },

    "count_scifi4": {
        "dim": 1,
        "title": "Plane4 SciFi Hit Total Count",
        "xexpr": "count_scifi4",
        "xbins": (50, 0, 500),
        "log": True,
        "selected": True,
    },

    "count_scifi5": {
        "dim": 1,
        "title": "Plane5 SciFi Hit Total Count",
        "xexpr": "count_scifi5",
        "xbins": (50, 0, 500),
        "log": True,
        "selected": True,
    },

    "count_us": {
        "dim": 1,
        "title": "US Hit Count",
        "xexpr": "count_us",
        "xbins": (52, 0, 52),
        "log": True,
        "selected": True,
    },

    "count_us1": {
        "dim": 1,
        "title": "US1 Hit Count",
        "xexpr": "count_us1",
        "xbins": (12, 0, 12),
        "log": True,
        "selected": True,
    },

    "count_us2": {
        "dim": 1,
        "title": "US2 Hit Count",
        "xexpr": "count_us2",
        "xbins": (12, 0, 12),
        "log": True,
        "selected": True,
    },

    "count_us3": {
        "dim": 1,
        "title": "US3 Hit Count",
        "xexpr": "count_us3",
        "xbins": (12, 0, 12),
        "log": True,
        "selected": True,
    },

    "count_us4": {
        "dim": 1,
        "title": "US4 Hit Count",
        "xexpr": "count_us4",
        "xbins": (12, 0, 12),
        "log": True,
        "selected": True,
    },

    "count_us5": {
        "dim": 1,
        "title": "US5 Hit Count",
        "xexpr": "count_us5",
        "xbins": (12, 0, 12),
        "log": True,
        "selected": True,
    },

    "count_ds": {
        "dim": 1,
        "title": "DS Hit Count",
        "xexpr": "count_ds",
        "xbins": (12, 0, 40),
        "log": True,
        "selected": True,
    },

    "count_ds1": {
        "dim": 1,
        "title": "DS1 Hit Count",
        "xexpr": "count_ds1",
        "xbins": (40, 0, 40),
        "log": True,
        "selected": True,
    },

    "count_ds2": {
        "dim": 1,
        "title": "DS2 Hit Count",
        "xexpr": "count_ds2",
        "xbins": (40, 0, 40),
        "log": True,
        "selected": True,
    },

    "count_ds3": {
        "dim": 1,
        "title": "DS3 Hit Count",
        "xexpr": "count_ds3",
        "xbins": (1, 0, 40),
        "log": True,
        "selected": True,
    },

    "count_ds4": {
        "dim": 1,
        "title": "DS4 Hit Count",
        "xexpr": "count_ds4",
        "xbins": (40, 0, 40),
        "log": True,
        "selected": True,
    },


    "primary_z": {
        "dim": 1,
        "title": "Primary start Z",
        "xexpr": "primary_startZ",
        "xbins": (200, -300.0, 600.0),
        "log": False,
        "selected": False,
    },

    "secondary_pdg": {
        "dim": 1,
        "title": "Secondary Particle Type",
        "xexpr": "secondary_pdg_bin",
        "xbins": None,  # fixed below from PDG_CATEGORY_LABELS
        "log": True,
        "selected": False,
        "is_pdg": True,
        "density": True,
    },

    "secondary_pdg_sel": {
        "dim": 1,
        "title": "Secondary Particle Type  ",
        "xexpr": "secondary_pdg_bin",
        "xbins": None,
        "log": True,
        "selected": True,
        "is_pdg": True,
        "density": True,
    },

    "secondary_energy": {
        "dim": 1,
        "title": "Secondary energy",
        "xexpr": "secondary_energy",
        "xbins": (200, 0.0, 5000.0),
        "log": True,
        "selected": False,
    },

    "secondary_energy_sel": {
        "dim": 1,
        "title": "Secondary energy  ",
        "xexpr": "secondary_energy",
        "xbins": (200, 0.0, 5000.0),
        "log": True,
        "selected": True,
    },

    "secondary_energy_sel_density": {
        "dim": 1,
        "title": "Secondary energy  ",
        "xexpr": "secondary_energy",
        "xbins": (200, 0.0, 5000.0),
        "log": True,
        "selected": True,
        "density": True,
    },

    "secondary_xy_all": {
        "dim": 2,
        "title": "Secondary start X vs Y all",
        "xexpr": "secondary_startX",
        "yexpr": "secondary_startY",
        "xbins": (200, -200.0, 200.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": False,
    },

    "secondary_xz_all": {
        "dim": 2,
        "title": "Secondary start X vs Z all",
        "xexpr": "secondary_startZ",
        "yexpr": "secondary_startX",
        "xbins": (200, -300.0, 600.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": False,
    },

    "secondary_yz_all": {
        "dim": 2,
        "title": "Secondary start Y vs Z all",
        "xexpr": "secondary_startZ",
        "yexpr": "secondary_startY",
        "xbins": (200, -300.0, 600.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": False,
    },

    "secondary_xy_sel": {
        "dim": 2,
        "title": "Secondary start X vs Y  ",
        "xexpr": "secondary_startX",
        "yexpr": "secondary_startY",
        "xbins": (200, -200.0, 200.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": True,
    },

    "secondary_xz_sel": {
        "dim": 2,
        "title": "Secondary start X vs Z  ",
        "xexpr": "secondary_startZ",
        "yexpr": "secondary_startX",
        "xbins": (200, -300.0, 600.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": True,
    },

    "secondary_yz_sel": {
        "dim": 2,
        "title": "Secondary start Y vs Z  ",
        "xexpr": "secondary_startZ",
        "yexpr": "secondary_startY",
        "xbins": (200, -300.0, 600.0),
        "ybins": (200, -200.0, 200.0),
        "log": False,
        "selected": True,
    },

    "theta_x_all": {
        "dim": 1,
        "title": "Secondary #theta_{x} all",
        "xexpr": "theta_x",
        "xbins": (200, -1.5, 1.5),
        "log": False,
        "selected": False,
    },

    "theta_x_sel": {
        "dim": 1,
        "title": "Secondary #theta_{x}  ",
        "xexpr": "theta_x",
        "xbins": (200, -1.5, 1.5),
        "log": False,
        "selected": True,
    },

    "theta_y_all": {
        "dim": 1,
        "title": "Secondary #theta_{y} all",
        "xexpr": "theta_y",
        "xbins": (200, -1.5, 1.5),
        "log": False,
        "selected": False,
    },

    "theta_y_sel": {
        "dim": 1,
        "title": "Secondary #theta_{y}  ",
        "xexpr": "theta_y",
        "xbins": (200, -1.5, 1.5),
        "log": False,
        "selected": True,
    },
}


def sanitize_expr(expr):
    s = expr.strip()

    replacements = {
        ">=": "ge",
        "<=": "le",
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        "<": "lt",
        "&&": "_and_",
        "||": "_or_",
        "!": "not_",
    }

    for k, v in replacements.items():
        s = s.replace(k, v)

    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")

    return s

# ============================================================
# FIXED PDG CATEGORY MAP FOR RDF / MT-SAFE PROCESSING
# ============================================================


# ============================================================
# DATASET CONFIG
# ============================================================
def get_datasets(which):
    metadata_path_cris = "/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_muonDIS_cvilela_metadata.csv"
    metadata_path_simona = "/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_muonDIS_sii_metadata.csv"

    if which == "cris":
        return {"cris": metadata_path_cris}
    elif which == "simona":
        return {"simona": metadata_path_simona}
    elif which == "both":
        return {
            "cris": metadata_path_cris,
            "simona": metadata_path_simona,
        }
    else:
        raise ValueError(f"Unsupported dataset choice: {which}")


# ============================================================
# FILE LIST BUILDING
# ============================================================
def build_file_list(
    metadata_csv,
    path_col="muonDISFeature_path",
    base_col="output_base_path",
    max_files=None,
):
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    if path_col not in df.columns:
        raise KeyError(f"Column '{path_col}' not found in {metadata_csv}")
    if base_col not in df.columns:
        raise KeyError(f"Column '{base_col}' not found in {metadata_csv}")

    files = []
    for _, row in df.iterrows():
        rel_path = str(row[path_col]).strip()
        base_path = str(row[base_col]).strip()

        if not rel_path or rel_path == "nan":
            continue
        if not base_path or base_path == "nan":
            continue

        full_path = os.path.join(base_path, rel_path.lstrip("/"))
        if not os.path.exists(full_path):
            print(f"[WARN] Missing ROOT file, skip: {full_path}")
            continue

        files.append(full_path)

        if max_files is not None and len(files) >= max_files:
            break

    print(f"[INFO] Built file list from {metadata_csv}")
    print(f"[INFO] Files added: {len(files)}")
    return files


def to_root_string_vector(py_list):
    out = ROOT.std.vector("string")()
    for item in py_list:
        out.push_back(item)
    return out


# ============================================================
# RDF HELPERS
# ============================================================
def declare_rdf_helpers():
    ROOT.gInterpreter.Declare(r"""
        #include <cmath>
        #include <cstdint>
        #include "ROOT/RVec.hxx"

        using ROOT::VecOps::RVec;
        using ROOT::VecOps::Construct;

        bool secondary_sizes_ok(const RVec<int> &secondary_pdg,
                                const RVec<float> &secondary_energy,
                                const RVec<float> &secondary_startX,
                                const RVec<float> &secondary_startY,
                                const RVec<float> &secondary_startZ,
                                const RVec<float> &secondary_px,
                                const RVec<float> &secondary_py,
                                const RVec<float> &secondary_pz) {
            const auto n = secondary_pdg.size();
            return secondary_energy.size() == n &&
                   secondary_startX.size() == n &&
                   secondary_startY.size() == n &&
                   secondary_startZ.size() == n &&
                   secondary_px.size() == n &&
                   secondary_py.size() == n &&
                   secondary_pz.size() == n;
        }

        RVec<double> make_theta_x(const RVec<float> &px, const RVec<float> &pz) {
            RVec<double> out;
            out.reserve(px.size());
            for (size_t i = 0; i < px.size(); ++i) {
                if (pz[i] == 0.f) continue;
                out.push_back(std::atan(double(px[i]) / double(pz[i])));
            }
            return out;
        }

        RVec<double> make_theta_y(const RVec<float> &py, const RVec<float> &pz) {
            RVec<double> out;
            out.reserve(py.size());
            for (size_t i = 0; i < py.size(); ++i) {
                if (pz[i] == 0.f) continue;
                out.push_back(std::atan(double(py[i]) / double(pz[i])));
            }
            return out;
        }
    """)


def pdg_code_to_name(pdg_code):
    db = ROOT.TDatabasePDG.Instance()
    p = db.GetParticle(int(pdg_code))

    if p:
        return str(p.GetName())

    return f"unknown_{int(pdg_code)}"


def make_pdg_table(df, selected=False):
    node = df.Filter("secondary_sizes_consistent")

    if selected:
        node = node.Filter("passed_sel")

    arr = node.Take["ROOT::RVec<int>"]("secondary_pdg").GetValue()

    counter = Counter()
    total = 0

    for event_pdgs in arr:
        for pdg in event_pdgs:
            counter[int(pdg)] += 1
            total += 1

    rows = []
    for pdg_code, count in counter.items():
        frac = count / total if total > 0 else 0.0
        rows.append((pdg_code, pdg_code_to_name(pdg_code), count, frac))

    rows.sort(key=lambda x: x[3], reverse=True)

    return rows


def save_pdg_table(rows, outpath):
    with open(outpath, "w") as f:
        f.write(f"{'pdgcode':>12}  {'pdg_name':<24}  {'count':>12}  {'fraction':>14}\n")
        f.write("-" * 70 + "\n")

        for pdg_code, name, count, frac in rows:
            f.write(f"{pdg_code:12d}  {name:<24}  {count:12d}  {frac:14.8e}\n")

    print(f"[INFO] Saved PDG table: {outpath}")


# ============================================================
# FEATURE SELECTION
# ============================================================
def resolve_requested_features(requested):
    if "all" in requested:
        return list(hist_info.keys())

    selected_direct = []
    for feat in requested:
        if feat in hist_info:
            selected_direct.append(feat)
        else:
            raise ValueError(f"Unknown feature: {feat}")

    return selected_direct


# ============================================================
# RDF GRAPH BUILDING
# ============================================================
def make_base_rdf(files, max_events, passed_sel):
    files_vec = to_root_string_vector(files)
    df = ROOT.RDataFrame("sndData", files_vec)

    use_range = (max_events is not None and max_events >= 0)

    if use_range:
        if ROOT.IsImplicitMTEnabled():
            print("[WARN] --max-events is not compatible with ROOT implicit MT; "
                  "ignoring max-events and processing all entries.")
        else:
            df = df.Range(max_events)

    df = (
        df.Define("passed_sel", passed_sel)
          .Define(
              "secondary_sizes_consistent",
              "secondary_sizes_ok(secondary_pdg, secondary_energy, secondary_startX, secondary_startY, secondary_startZ, secondary_px, secondary_py, secondary_pz)"
          )
          .Define("theta_x", "make_theta_x(secondary_px, secondary_pz)")
          .Define("theta_y", "make_theta_y(secondary_py, secondary_pz)")
    )

    return df

def make_hist_title(base_title, sample, selected, passed_sel):
    title = f"{base_title}"
    if selected:
        title += f" [{passed_sel}]"
    return title

def hist_model_1d(feature_name, sample, passed_sel):
    cfg = hist_info[feature_name]

    title = make_hist_title(
        cfg["title"],
        sample,
        cfg["selected"],
        passed_sel
    )

    nbx, xmin, xmax = cfg["xbins"]
    xlabel = cfg["xexpr"]
    ylabel = "Probability density" if cfg.get("density", False) else "Entries"

    return ROOT.RDF.TH1DModel(
        f"{feature_name}__Dataset-{sample}",
        f'{title};{xlabel};{ylabel}',
        nbx, xmin, xmax
    )


def hist_model_2d(feature_name, sample, passed_sel):
    cfg = hist_info[feature_name]
    nbx, xmin, xmax = cfg["xbins"]
    nby, ymin, ymax = cfg["ybins"]

    title = make_hist_title(
        cfg["title"],
        sample,
        cfg["selected"],
        passed_sel
    )
    
    return ROOT.RDF.TH2DModel(
        f"{feature_name}__Dataset-{sample}",
        f'{title};{cfg["xexpr"]};{cfg["yexpr"]}',
        nbx, xmin, xmax,
        nby, ymin, ymax
    )


def materialize_hist(result_ptr, feature_name):
    hist = result_ptr.GetValue().Clone()
    hist.SetDirectory(0)

    cfg = hist_info[feature_name]

    if cfg["dim"] == 1 and cfg.get("density", False):
        integral = hist.Integral("width")
        if integral > 0:
            hist.Scale(1.0 / integral)

    return hist


def is_secondary_feature(feature_name):
    return feature_name.startswith("secondary_") or feature_name.startswith("theta_")

def is_event_feature(feature_name):
    return not is_secondary_feature(feature_name)

def book_histograms(df, selected_features, sample, passed_sel):
    booked = {}

    df_all = df
    df_sel = df.Filter("passed_sel")
    df_sec_all = df.Filter("secondary_sizes_consistent")
    df_sec_sel = df.Filter("passed_sel && secondary_sizes_consistent")

    for feat in selected_features:
        cfg = hist_info[feat]

        if cfg.get("is_pdg", False):
            continue

        if is_secondary_feature(feat):
            node = df_sec_sel if cfg["selected"] else df_sec_all
        else:
            node = df_sel if cfg["selected"] else df_all


        if cfg["dim"] == 1:
            model = hist_model_1d(feat, sample, passed_sel)
            booked[feat] = node.Histo1D(model, cfg["xexpr"])
        elif cfg["dim"] == 2:
            model = hist_model_2d(feat, sample, passed_sel)
            booked[feat] = node.Histo2D(model, cfg["xexpr"], cfg["yexpr"])
        else:
            raise ValueError(f"Unsupported histogram dim for {feat}")

    return booked


def run_histograms(booked, selected_features):
    out = {}
    for feat in selected_features:
        if hist_info[feat].get("is_pdg", False):
            continue
        out[feat] = materialize_hist(booked[feat], feat)
    return out


# ============================================================
# DRAW HELPERS
# ============================================================
def clone_norm(hist):
    h = hist.Clone(hist.GetName() + "_norm")
    h.SetDirectory(0)
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)
    return h


def set_overlay_style(hist, color, width=2):
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetLineWidth(width)


def save_single_hist_pdf(hist, outpath, log=False):
    c = ROOT.TCanvas(f"c_{hist.GetName()}", "", 900, 700)
    if log:
        c.SetLogy()
    hist.Draw("hist")
    c.SaveAs(outpath)
    c.Close()


def save_single_2d_pdf(hist, outpath, log=False):
    c = ROOT.TCanvas(f"c_{hist.GetName()}", "", 950, 800)
    if log:
        c.SetLogz()
    hist.Draw("COLZ")
    c.SaveAs(outpath)
    c.Close()


def save_overlay_pdf(hist_a, label_a, hist_b, label_b, outpath, normalize=False, log=False):
    h1 = clone_norm(hist_a) if normalize else hist_a.Clone(hist_a.GetName() + "_cpy")
    h2 = clone_norm(hist_b) if normalize else hist_b.Clone(hist_b.GetName() + "_cpy")
    h1.SetDirectory(0)
    h2.SetDirectory(0)

    c = ROOT.TCanvas(f"c_{os.path.basename(outpath).replace('.pdf', '')}", "", 900, 700)
    if log:
        c.SetLogy()

    set_overlay_style(h1, ROOT.kRed + 1)
    set_overlay_style(h2, ROOT.kBlue + 1)

    ymax = max(h1.GetMaximum(), h2.GetMaximum())
    if ymax > 0:
        h1.SetMaximum(ymax * 1.25)

    if normalize:
        h1.GetYaxis().SetTitle("Normalized entries")
        h2.GetYaxis().SetTitle("Normalized entries")

    h1.Draw("hist")
    h2.Draw("hist same")

    leg = ROOT.TLegend(0.68, 0.80, 0.88, 0.90)
    leg.AddEntry(h1, label_a, "l")
    leg.AddEntry(h2, label_b, "l")
    leg.Draw()

    c.SaveAs(outpath)
    c.Close()


# ============================================================
# OUTPUT
# ============================================================
def save_sample_outputs(sample, hists, selected_features, tag, outdir):
    os.makedirs(outdir, exist_ok=True)

    for feat in selected_features:
        cfg = hist_info[feat]

        if cfg.get("is_pdg", False):
            continue

        suffix = f"_{tag}" if tag else ""
        outpath = os.path.join(outdir, f"{feat}{suffix}.pdf")

        if cfg["dim"] == 1:
            save_single_hist_pdf(hists[feat], outpath, log=cfg["log"])
        elif cfg["dim"] == 2:
            save_single_2d_pdf(hists[feat], outpath, log=cfg["log"])


def save_compare_outputs(results_hists, selected_features, tag, outdir, normalize=False):
    if "cris" not in results_hists or "simona" not in results_hists:
        return

    os.makedirs(outdir, exist_ok=True)

    for feat in selected_features:
        cfg = hist_info[feat]

        if cfg.get("is_pdg", False):
            continue

        if cfg["dim"] != 1:
            continue

        suffix = f"_{tag}" if tag else ""
        outpath = os.path.join(outdir, f"{feat}_compare{suffix}.pdf")

        do_normalize = normalize and not cfg.get("density", False)

        save_overlay_pdf(
            results_hists["cris"][feat], "cris",
            results_hists["simona"][feat], "simona",
            outpath,
            normalize=do_normalize,
            log=cfg["log"]
        )


def save_root(results_hists, dataset_label, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    outpath = os.path.join(outdir, f"muondis_{dataset_label}{suffix}.root")

    fout = ROOT.TFile(outpath, "RECREATE")

    for sample, hdict in results_hists.items():
        fout.mkdir(sample)
        fout.cd(sample)

        for hist in hdict.values():
            hist.Write()

        fout.cd()

    fout.Close()
    print(f"[INFO] Saved ROOT output: {outpath}")


# ============================================================
# MAIN
# ============================================================
def main(args):
    sel_tag = sanitize_expr(args.passed_sel)
    args.outdir = f"{args.outdir}_{sel_tag}"
    os.makedirs(args.outdir, exist_ok=True)

    declare_rdf_helpers()

    if args.threads is not None and args.threads > 0:
        if args.max_events is not None and args.max_events >= 0:
            print("[WARN] Implicit MT + --max-events is unsupported in RDataFrame.")
            print("[WARN] Processing all entries from the selected files.")
        ROOT.EnableImplicitMT(args.threads)
        print(f"[INFO] Enabled ROOT implicit MT with {args.threads} threads")
    else:
        ROOT.EnableImplicitMT()
        if args.max_events is not None and args.max_events >= 0:
            print("[WARN] Implicit MT + --max-events is unsupported in RDataFrame.")
            print("[WARN] Processing all entries from the selected files.")
        print("[INFO] Enabled ROOT implicit MT with automatic thread count")

    datasets = get_datasets(args.dataset)
    selected_features = resolve_requested_features(args.features)

    print(f"[INFO] Dataset choice: {args.dataset}")
    print(f"[INFO] Direct features: {selected_features}")

    results_hists = {}

    for sample, metadata_csv in datasets.items():
        print(f"\n[INFO] ===== Processing dataset: {sample} =====")
        print(f"[INFO] Metadata: {metadata_csv}")

        files = build_file_list(
            metadata_csv,
            path_col="muonDISFeature_path",
            base_col="output_base_path",
            max_files=args.max_files,
        )

        if len(files) == 0:
            print(f"[WARN] No valid files for {sample}, skip")
            continue

        df = make_base_rdf(files, args.max_events, args.passed_sel)

        sample_outdir = os.path.join(args.outdir, sample)
        os.makedirs(sample_outdir, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""

        if "secondary_pdg" in selected_features:
            pdg_table_all = make_pdg_table(df, selected=False)
            save_pdg_table(
                pdg_table_all,
                os.path.join(sample_outdir, f"secondary_pdg{suffix}.txt")
            )

        if "secondary_pdg_sel" in selected_features:
            pdg_table_sel = make_pdg_table(df, selected=True)
            save_pdg_table(
                pdg_table_sel,
                os.path.join(sample_outdir, f"secondary_pdg_sel{suffix}.txt")
            )

        selected_features_for_plots = [
            feat for feat in selected_features
            if not hist_info[feat].get("is_pdg", False)
        ]

        booked = book_histograms(df, selected_features_for_plots, sample, args.passed_sel)
        hists = run_histograms(booked, selected_features_for_plots)

        save_sample_outputs(
            sample=sample,
            hists=hists,
            selected_features=selected_features_for_plots,
            tag=args.tag,
            outdir=os.path.join(args.outdir, sample),
        )

        results_hists[sample] = hists

    if not results_hists:
        print("[ERROR] No dataset processed successfully.")
        return

    selected_features_for_plots = [
        feat for feat in selected_features
        if not hist_info[feat].get("is_pdg", False)
    ]

    if args.dataset == "both":
        save_compare_outputs(
            results_hists=results_hists,
            selected_features=selected_features_for_plots,
            tag=args.tag,
            outdir=args.outdir,
            normalize=args.normalize,
        )

    save_root(
        results_hists=results_hists,
        dataset_label=args.dataset,
        tag=args.tag,
        outdir=args.outdir,
    )

    print("[INFO] Done.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze muonDIS sndData feature trees with RDataFrame and save selected plots as PDF"
    )

    parser.add_argument(
        "--dataset",
        choices=["cris", "simona", "both"],
        required=True,
        help="Which dataset to analyze"
    )

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        choices=list(hist_info.keys()) + ["all"],
        help="Which plots to make. Use 'all' for all direct plots."
    )

    parser.add_argument(
        "--tag",
        default="",
        help="Identifier appended to output file names"
    )

    parser.add_argument(
        "--outdir",
        default="plots_muondis",
        help="Output directory for PDFs and ROOT file"
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events to process per dataset (-1 means all)"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=60,
        help="Maximum number of ROOT files to read per dataset (default: all)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of ROOT implicit-MT threads (0 means ROOT decides automatically)"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize 1D comparison plots to unit area"
    )
    
    parser.add_argument(
        "--passed-sel",
        default="count_scifi > 1 && count_veto == 0",
        help="Selection used for selected histograms"
    )
    # && avg_scifi_ver_x >= -40.9989 && avg_scifi_ver_x <= -15.5431 && avg_scifi_hor_y >= 21.9541 && avg_scifi_hor_y <= 48.6972
    args = parser.parse_args()
    main(args)