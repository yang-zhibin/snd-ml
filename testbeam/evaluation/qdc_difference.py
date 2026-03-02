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
        for token in ["qdc_scifi"]:
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


def process_one_file(infile):
    # Output dir = <input dir>/ratio
    in_dir = os.path.dirname(os.path.abspath(infile))
    outdir = os.path.join(in_dir, "ratio")
    os.makedirs(outdir, exist_ok=True)

    ROOT.gROOT.SetBatch(True)

    # Extract feature name from file name: testbeam_2024_GravNet_vX_<feature>.root
    base = os.path.basename(infile)
    m = re.match(r"testbeam_2024_GravNet_v\d+_(.+)\.root$", base)
    feature_name = m.group(1) if m else "unknown"

    f = ROOT.TFile.Open(infile)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open: {infile}")

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
        da_mean, da_sig = _hist_mean_sigma(h_da) if h_da else (None, None)

        if mc_mean is not None:
            stats["MC"][E] = mc_mean
        if da_mean is not None:
            stats["Data"][E] = da_mean

        print(f"[{feature_name}] {E:>4} GeV | MC mean={mc_mean} sigma={mc_sig} | Data mean={da_mean} sigma={da_sig}")

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

    g_rt.SetTitle("")  # bottom pad doesn't need a main title
    g_rt.GetXaxis().SetTitle("Energy [GeV]")
    g_rt.GetYaxis().SetTitle("Data / MC")

    # -----------------------
    # Single canvas with 2 pads: top = means, bottom = ratio
    # -----------------------
    c = ROOT.TCanvas(f"c_{feature_name}", "mean_and_ratio", 900, 900)

    pad_top = ROOT.TPad("pad_top", "top", 0.0, 0.30, 1.0, 1.0)
    pad_bot = ROOT.TPad("pad_bot", "bottom", 0.0, 0.00, 1.0, 0.30)

    pad_top.SetLeftMargin(0.14)
    pad_top.SetRightMargin(0.04)
    pad_top.SetTopMargin(0.06)
    pad_top.SetBottomMargin(0.02)

    pad_bot.SetLeftMargin(0.18)   # prevent y-title clipping in short pad
    pad_bot.SetRightMargin(0.04)
    pad_bot.SetTopMargin(0.02)
    pad_bot.SetBottomMargin(0.35)

    pad_top.Draw()
    pad_bot.Draw()

    # ---- Dynamic y-range (top)
    ymin = min(min(y_mc), min(y_da))
    ymax = max(max(y_mc), max(y_da))
    yspan = ymax - ymin
    ymargin = 0.15 * yspan if yspan > 0 else 0.1 * (abs(ymax) if ymax != 0 else 1.0)

    # ---- Dynamic y-range (ratio)
    rmin = min(y_rt)
    rmax = max(y_rt)
    rspan = rmax - rmin
    rmargin = 0.15 * rspan if rspan > 0 else 0.1 * (abs(rmax) if rmax != 0 else 1.0)

    # ---- Top pad
    pad_top.cd()

    g_mc.GetYaxis().SetRangeUser(max(0.0, ymin - ymargin), ymax + ymargin)

    # Hide x axis labels/titles on top pad
    g_mc.GetXaxis().SetLabelSize(0)
    g_mc.GetXaxis().SetTitleSize(0)

    g_mc.GetYaxis().CenterTitle(True)
    g_mc.GetYaxis().SetTitleOffset(1.4)
    g_mc.GetYaxis().SetTitleSize(0.05)
    g_mc.GetYaxis().SetLabelSize(0.045)

    g_mc.Draw("APL")
    g_da.Draw("PL SAME")

    leg = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_mc, "MC", "lp")
    leg.AddEntry(g_da, "Data", "lp")
    leg.Draw()

    # ---- Bottom pad
    pad_bot.cd()

    g_rt.GetYaxis().SetTitleSize(0.11)
    g_rt.GetYaxis().SetLabelSize(0.095)
    g_rt.GetYaxis().SetTitleOffset(0.55)  # smaller moves title into frame
    g_rt.GetYaxis().SetNdivisions(505)

    g_rt.GetXaxis().SetTitleSize(0.12)
    g_rt.GetXaxis().SetLabelSize(0.10)
    g_rt.GetXaxis().SetTitleOffset(1.0)

    g_rt.GetYaxis().SetRangeUser(rmin - rmargin, rmax + rmargin)

    g_rt.Draw("APL")

    # Reference line at 1
    line = ROOT.TLine(min(x_vals), 1.0, max(x_vals), 1.0)
    line.SetLineStyle(2)
    line.Draw("SAME")

    # Save PDF (unique per file)
    out_pdf = os.path.join(outdir, f"mean_and_ratio_vs_energy_{feature_name}.pdf")
    c.SaveAs(out_pdf)

    f.Close()
    print(f"\n[OK] Output written to {out_pdf}\n")


def process(args):
    # Hardcode your files here (edit as needed)
    files = [
        "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/evaluation/plots/scifi_gt_50-150ns_previous_cut/testbeam_2024_GravNet_v6/qdc_scifi/testbeam_2024_GravNet_v6_qdc_scifi.root",
    ]

    for infile in files:
        process_one_file(infile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # keep argparse (minimal change), but not required anymore
    args = parser.parse_args()
    process(args)
