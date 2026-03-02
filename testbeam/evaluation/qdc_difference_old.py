#!/usr/bin/env python3
import argparse
import os
import re
import ROOT


def _sorted_energy_dirs(tdir):
    """
    Return list of (E_GeV:int, dirname:str) sorted by energy,
    with 180 GeV explicitly removed.
    """
    out = []
    for k in tdir.GetListOfKeys():
        name = k.GetName()  # e.g. "100GeV"
        m = re.match(r"^(\d+)\s*GeV$", name)
        if not m:
            continue

        E = int(m.group(1))

        # remove 180 GeV
        if E == 180:
            continue

        out.append((E, name))

    out.sort(key=lambda x: x[0])
    return out


def _find_hist_for_particle(tdir, particle="e-"):
    keys = list(tdir.GetListOfKeys())

    def score(name):
        s = 0
        for token in ["qdc", "QDC", "scifi", "SciFi", "charge", "Charge"]:
            if token in name:
                s += 1
        obj = tdir.Get(name)
        if obj and obj.InheritsFrom("TH1"):
            s += 5
        return s

    preferred = [k.GetName() for k in keys if particle in k.GetName()]
    preferred.sort(key=score, reverse=True)
    for name in preferred:
        obj = tdir.Get(name)
        if obj and obj.InheritsFrom("TH1"):
            return obj

    th1_names = []
    for k in keys:
        obj = tdir.Get(k.GetName())
        if obj and obj.InheritsFrom("TH1"):
            th1_names.append(k.GetName())

    if not th1_names:
        return None

    th1_names.sort(key=score, reverse=True)
    return tdir.Get(th1_names[0])


def _hist_mean_sigma(h):
    if not h or h.GetEntries() <= 0:
        return None, None
    return float(h.GetMean()), float(h.GetStdDev())


def _hist_mean_sigma_with_cut(h, xmin):
    """
    Compute mean/std of a TH1 using only bins with center >= xmin.
    Implemented by zeroing bins below xmin on a clone.
    """
    if not h or h.GetEntries() <= 0:
        return None, None

    htmp = h.Clone(f"{h.GetName()}_cut")
    htmp.SetDirectory(0)

    for ib in range(1, htmp.GetNbinsX() + 1):
        if htmp.GetBinCenter(ib) < xmin:
            htmp.SetBinContent(ib, 0.0)
            htmp.SetBinError(ib, 0.0)

    if htmp.GetEntries() <= 0:
        return None, None

    return float(htmp.GetMean()), float(htmp.GetStdDev())


def process(args):
    # Output dir = <input dir>/ratio
    in_dir = os.path.dirname(os.path.abspath(args.file))
    outdir = os.path.join(in_dir, "ratio")
    os.makedirs(outdir, exist_ok=True)

    ROOT.gROOT.SetBatch(True)

    # Extract feature name from file name: testbeam_2024_GravNet_v2_<feature>.root
    base = os.path.basename(args.file)
    m = re.match(r"testbeam_2024_GravNet_v2_(.+)\.root$", base)
    feature_name = m.group(1) if m else "unknown"

    f = ROOT.TFile.Open(args.file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open: {args.file}")

    mc_top = f.Get("MC_full")
    data_top = f.Get("Data_full")
    if not mc_top or not data_top:
        raise RuntimeError("Missing MC_full or Data_full directories")

    energies = _sorted_energy_dirs(mc_top)
    if not energies:
        raise RuntimeError("No valid energy folders found")

    particle = "e-"

    stats = {"MC": {}, "Data": {}}

    # -----------------------
    # Loop over energies
    # -----------------------
    for E, dname in energies:
        mc_dir = mc_top.Get(dname)
        da_dir = data_top.Get(dname)

        if not mc_dir or not da_dir:
            continue

        h_mc = _find_hist_for_particle(mc_dir, particle)
        h_da = _find_hist_for_particle(da_dir, particle)

        mc_mean, mc_sig = _hist_mean_sigma(h_mc)
        # Data unchanged
        da_mean, da_sig = _hist_mean_sigma(h_da) if h_da else (None, None)

        if mc_mean is not None:
            stats["MC"][E] = mc_mean
        if da_mean is not None:
            stats["Data"][E] = da_mean

        extra = ""


        print(f"{E:>4} GeV | MC mean={mc_mean} sigma={mc_sig} | Data mean={da_mean} sigma={da_sig}{extra}")

    common_E = sorted(set(stats["MC"]) & set(stats["Data"]))
    if not common_E:
        raise RuntimeError("No common energies between MC and Data")

    # -----------------------
    # Build graphs
    # -----------------------
    x_vals = [float(E) for E in common_E]
    y_mc = [float(stats["MC"][E]) for E in common_E]
    y_da = [float(stats["Data"][E]) for E in common_E]
    y_rt = [float(stats["Data"][E] / stats["MC"][E]) if stats["MC"][E] else 0.0 for E in common_E]

    g_mc = ROOT.TGraph()
    g_da = ROOT.TGraph()
    g_rt = ROOT.TGraph()

    for i, E in enumerate(common_E):
        g_mc.SetPoint(i, float(E), y_mc[i])
        g_da.SetPoint(i, float(E), y_da[i])
        g_rt.SetPoint(i, float(E), y_rt[i])

    # Styles
    g_mc.SetMarkerStyle(20)
    g_mc.SetMarkerColor(ROOT.kBlue + 1)
    g_mc.SetLineColor(ROOT.kBlue + 1)
    g_mc.SetLineWidth(2)

    g_da.SetMarkerStyle(21)
    g_da.SetMarkerColor(ROOT.kRed + 1)
    g_da.SetLineColor(ROOT.kRed + 1)
    g_da.SetLineWidth(2)

    g_rt.SetMarkerStyle(20)
    g_rt.SetMarkerColor(ROOT.kBlack)
    g_rt.SetLineColor(ROOT.kBlack)
    g_rt.SetLineWidth(2)

    # Titles/axes (add feature name)
    g_mc.SetTitle(f"Mean(QDC) vs Energy ({particle}) [{feature_name}]")
    g_mc.GetXaxis().SetTitle("Energy [GeV]")
    g_mc.GetYaxis().SetTitle("Mean(QDC)")

    g_rt.SetTitle(f"Data/MC Mean Ratio vs Energy ({particle}) [{feature_name}]")
    g_rt.GetXaxis().SetTitle("Energy [GeV]")
    g_rt.GetYaxis().SetTitle("Mean(Data) / Mean(MC)")

    # -----------------------
    # Canvas 1: fix y-title margin + widen y upper limit + save PDF only
    # -----------------------
    c1 = ROOT.TCanvas("c1", "means", 900, 700)
    c1.SetLeftMargin(0.14)
    c1.SetBottomMargin(0.12)

    ymax = max(max(y_mc), max(y_da))
    g_mc.GetYaxis().SetRangeUser(0.0, 1.25 * ymax)
    g_mc.GetYaxis().CenterTitle(True)

    g_mc.Draw("APL")
    g_da.Draw("PL SAME")

    leg = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
    leg.AddEntry(g_mc, "MC", "lp")
    leg.AddEntry(g_da, "Data", "lp")
    leg.Draw()

    c1.SaveAs(os.path.join(outdir, "mean_vs_energy.pdf"))

    # -----------------------
    # Canvas 2: ratio (PDF only)
    # -----------------------
    c2 = ROOT.TCanvas("c2", "ratio", 900, 700)
    c2.SetLeftMargin(0.14)
    c2.SetBottomMargin(0.12)

    g_rt.Draw("APL")
    c2.SaveAs(os.path.join(outdir, "ratio_vs_energy.pdf"))

    f.Close()
    print(f"\n[OK] Output written to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        dest="file",
        default="/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/evaluation/plots/scifi_gt_50-150ns_previous_cut/qdc_scifi/testbeam_2024_GravNet_v4_qdc_scifi.root",
        help="input root file"
    )
    # kept for CLI compatibility, but ignored by design (output is always <input dir>/ratio)
    parser.add_argument(
        "-o", "--outdir",
        dest="outdir",
        default="",
        help="(ignored) output is always <input dir>/ratio"
    )
    args = parser.parse_args()
    process(args)
