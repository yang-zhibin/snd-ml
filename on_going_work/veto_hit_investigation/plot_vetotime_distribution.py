import ROOT
import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import math
import argparse
import matplotlib.pyplot as plt
import csv

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)



particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}
class_2_particle = {v: k for k, v in particle_2_class.items()}


def read_metadata(directory="/afs/cern.ch/work/z/zhibin/snd-ml/on_going_work/compare_pred_MC/processed_metadata_GravNet_v2"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict                  


# --- Helper function ---
def tree_entries_and_branch(path, treename, required_branch=None):
    if not path or not os.path.exists(path):
        return (0, False)
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        return (0, False)
    t = f.Get(treename)
    if not t:
        f.Close()
        return (0, False)
    n = int(t.GetEntries())
    has_req = True
    if required_branch:
        brs = t.GetListOfBranches()
        has_req = bool(brs and any(b.GetName() == required_branch for b in brs))
    f.Close()
    return (n, has_req)

def read_rdf(args, metadata_df, MC_muon=False):
    preSelect_chain = ROOT.TChain("sndData")
    
    int_lumi = 0
    # Defaultdict of dicts
    
    for _, row in metadata_df.iterrows():
        sub = row['subfolder']


        preSelect_path = row['preSelect_path']
   

        n_preSelect, _ = tree_entries_and_branch(preSelect_path, "sndData")
        
        if n_preSelect > 0:
            preSelect_chain.Add(preSelect_path)
            
            if pd.notna(row['lumi_per_file']):
                int_lumi += row['lumi_per_file']
        else:
            if n_preSelect == 0:
                print(f"[Skip] features empty/missing: {preSelect_path}")
                if pd.notna(row['lumi_per_file']):
                    int_lumi += row['lumi_per_file']
            
    print(f"Added {preSelect_chain.GetNtrees()} preSelect files")
    if (int_lumi==0):
        return None, None, 0
    # Print total events per subfolder
   
    rdf = ROOT.RDataFrame(preSelect_chain)
    if args.cut =='withVetoHit':
        rdf = rdf.Filter("count_veto>0")
    
    return rdf, preSelect_chain, int_lumi

    
    
def process_hist(args):
    hist_name=args.hist_name
    n_bins, x_min, x_max, axis_title, logy = hist_info[hist_name]
    
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    #reading real data
    data_rdf, data_chain, data_int_lumi = read_rdf(args,real_data[:10])
    print(f'data_int_lumi:{data_int_lumi}')
    pred_classes = [ "kaon", "neutron", "muon"]
    

    data_hist_proxies = []
    data_hist = data_rdf.Histo1D(
            (f"h_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
    data_hist.SetDirectory(0)
    data_hist.GetXaxis().SetTitle(axis_title)
    data_hist.GetYaxis().SetTitle("Events")
    
    ##reading MC
    normalise_lumi = data_int_lumi
    
    ## reading neutrino
    neutrino_rdf, neutrino_chain, neutrino_int_lumi = read_rdf(args, neutrino_df[:10])
    
    scale_factor = normalise_lumi/ neutrino_int_lumi  if neutrino_int_lumi else 1.0
    
    neutrino_classes = ["ve", "vm", "vt", "NC"]
    MC_neutrino_true_hists = {}
    neutrino_hist_proxies = []
    for cls in neutrino_classes:
        class_id = particle_2_class[cls]
        rdf_true = neutrino_rdf.Filter(f"ParticleClass == {class_id}")
        h_proxy_true = rdf_true.Histo1D(
            (f"h_{cls}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        h_ture = h_proxy_true.GetValue()
        neutrino_hist_proxies.append(h_proxy_true)
        h_ture.Scale(scale_factor)  
        h_ture.SetDirectory(0)
        h_ture.GetXaxis().SetTitle(axis_title)
        h_ture.GetYaxis().SetTitle("Expected Events")
        MC_neutrino_true_hists[cls] = h_ture
        
    # --- reading muon ---
    beam_types = sorted(muon_df['subfolder'].unique(), key=lambda x: (str(type(x)), x))

    muon_true_hists = {}
    # add another hist for background pred hist
    muon_hist_proxies = []

    mu_class_id = particle_2_class['muon']
    kaon_class_id = particle_2_class['kaon']
    neutron_class_id = particle_2_class['neutron']

    for beam_type in beam_types:
        sub_df = muon_df[muon_df['subfolder'] == beam_type]

        # build RDF and lumi for this slice
        rdf, _, int_lumi = read_rdf(args, sub_df[:10], MC_muon=True)
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {mu_class_id} && count_scifi > 200") 
        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_muon_true_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_muon_pred_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        h_proxy_pred_bkg = rdf_pred_bkg.Histo1D(
            (f"h_muon_pred_bkg_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        muon_hist_proxies.extend([h_proxy_true, h_proxy_pred, h_proxy_pred_bkg])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        h_pred_bkg = h_proxy_pred_bkg.GetValue().Clone()
        for h in (h_true, h_pred, h_pred_bkg):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by veto Ineff = 1e-8 factor for muon
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi) * 1e-8
            h_true.Scale(scale)
            h_pred.Scale(scale)
            h_pred_bkg.Scale(scale)
        
        muon_true_hists[beam_type] = h_true
        muon_pred_hists[beam_type] = h_pred
        muon_pred_bkg_hists[beam_type] = h_pred_bkg
        
    # --- reading kaon ---
    if 'energy_range' not in kaon_df.columns:
        raise KeyError("MC_kaon metadata requires an 'energy_range' column")

    ranges = sorted(kaon_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))

    kaon_true_hists = {}
    kaon_pred_hists = {}
    kaon_pred_bkg_hists = {}
    kaon_hist_proxies = []

    for erange in ranges:
        sub_df = kaon_df[kaon_df['energy_range'] == erange]

        # build RDF and lumi for this slice
        rdf, _, int_lumi = read_rdf(args, sub_df[:10])
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {kaon_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {kaon_class_id}")
        rdf_pred_bkg = rdf.Filter(f"pred_class_first == {mu_class_id} || pred_class_first == {kaon_class_id} || pred_class_first == {neutron_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_kaon_true_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_kaon_pred_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred_bkg = rdf_pred_bkg.Histo1D(
            (f"h_kaon_pred_bkg_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        kaon_hist_proxies.extend([h_proxy_true, h_proxy_pred, h_proxy_pred_bkg])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        h_pred_bkg = h_proxy_pred_bkg.GetValue().Clone()
        for h in (h_true, h_pred, h_proxy_pred_bkg):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by lumi (no extra 1e-8 here unless you need it for consistency)
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi)
            h_true.Scale(scale)
            h_pred.Scale(scale)
            h_pred_bkg.Scale(scale)

        kaon_true_hists[erange] = h_true
        kaon_pred_hists[erange] = h_pred
        kaon_pred_bkg_hists[erange] = h_pred_bkg
        
    # --- reading neutron ---
    if 'energy_range' not in neutron_df.columns:
        raise KeyError("MC_neutron metadata requires an 'energy_range' column")

    ranges = sorted(kaon_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))

    neutron_true_hists = {}
    neutron_pred_hists = {}
    neutron_pred_bkg_hists = {}
    neutron_hist_proxies = []

    

    for erange in ranges:
        sub_df = neutron_df[neutron_df['energy_range'] == erange]

        # build RDF and lumi for this slice
        
        rdf, _, int_lumi = read_rdf(args, sub_df[:10])
        if not int_lumi:
            continue


        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {neutron_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {neutron_class_id}")
        rdf_pred_bkg = rdf.Filter(f"pred_class_first == {mu_class_id} || pred_class_first == {kaon_class_id} || pred_class_first == {neutron_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_neutron_true_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_neutron_pred_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred_bkg = rdf_pred_bkg.Histo1D(
            (f"h_neutron_pred_bkg_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        
        
        neutron_hist_proxies.extend([h_proxy_true, h_proxy_pred,h_proxy_pred_bkg])


        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        h_pred_bkg = h_proxy_pred_bkg.GetValue().Clone()
        for h in (h_true, h_pred, h_pred_bkg):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by lumi (no extra 1e-8 here unless you need it for consistency)
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi)
            h_true.Scale(scale)
            h_pred.Scale(scale)
            h_pred_bkg.Scale(scale)
        
        neutron_true_hists[erange] = h_true
        neutron_pred_hists[erange] = h_pred
        neutron_pred_bkg_hists[erange] = h_pred_bkg
        

        
    if (hist_name == "avg_us5_y"):
        summary_yield(data_pred_hists, 
                    MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                    muon_true_hists, muon_pred_hists, 
                    kaon_true_hists, kaon_pred_hists, 
                    neutron_true_hists, neutron_pred_hists, 
                    args, hist_name
                    )
    plot_MC_pred_VS_data_pred_combined(MC_neutrino_pred_hists, data_pred_hists, muon_pred_bkg_hists, kaon_pred_bkg_hists, neutron_pred_bkg_hists, hist_name, data_int_lumi, logy)
    plot_MC_pred_VS_data_pred_seperated(data_pred_hists, muon_pred_hists, kaon_pred_hists, neutron_pred_hists, hist_name, data_int_lumi, logy)
    plot_data_pred_bkg(data_pred_hists, hist_name, data_int_lumi, logy)
    plot_MC(MC_neutrino_true_hists, muon_true_hists, kaon_true_hists,neutron_true_hists, hist_name, data_int_lumi, logy)
    plot_MC_VS_MC_pred(data_pred_hists,
                       MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                       muon_true_hists, muon_pred_hists, 
                       kaon_true_hists, kaon_pred_hists, 
                       neutron_true_hists, neutron_pred_hists, 
                       hist_name, data_int_lumi, logy)
    plot_MC_vs_data(data_hist, muon_true_hists, kaon_true_hists, neutron_true_hists, hist_name, data_int_lumi, logy)
    
    # plot_2d_hist(data_pred_hists,
    #                    MC_neutrino_true_hists, MC_neutrino_pred_hists, 
    #                    muon_true_hists, muon_pred_hists, 
    #                    kaon_true_hists, kaon_pred_hists, 
    #                    neutron_true_hists, neutron_pred_hists, 
    #                    hist_name, data_int_lumi)



def summary_yield(data_pred_hists, 
                  MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                  muon_true_hists, muon_pred_hists, 
                  kaon_true_hists, kaon_pred_hists, 
                  neutron_true_hists, neutron_pred_hists, 
                  args, hist_name):
    """
    Summarize yields before/after GNN and data prediction, and plot a table with matplotlib.

    data_pred_hists:
        list or dict of data-pred hists (with GNN), containing muon / kaon / neutron.
        - If list:   [muon_hist, kaon_hist, neutron_hist]
        - If dict:   keys "muon", "kaon", "neutron"

    MC_neutrino_true_hists / MC_neutrino_pred_hists:
        list or dict of neutrino MC histograms, order/keys: 'vm','ve','vt','NC'
        - If list:   [vm, ve, vt, NC]
        - If dict:   keys "vm", "ve", "vt", "NC"

    *_true_hists / *_pred_hists:
        iterables of TH1 histograms for muon / kaon / neutron backgrounds
        (will be summed with sum_hists).

    args.cut:
        string that describes the final selection cut (after no-veto + SciFi>200).

    args.hist_name or args.hist (optional):
        tag for output directory under ./plots/.
    """
    cut_name = args.cut

    # ---------- helpers to access list/dict containers ----------
    def get_nu_hist(container, idx, key):
        """Return hist from container by index (list) or key (dict)."""
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(key, None)
        # assume list-like
        if idx < len(container):
            return container[idx]
        return None

    def get_data_hist(container, idx, key):
        """Return muon/kaon/neutron from container list/dict."""
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(key, None)
        if idx < len(container):
            return container[idx]
        return None

    def integral(h):
        return h.Integral() if h else 0.0

    # ---------- sum background components (true & pred) ----------
    h_mu_tot_true = sum_hists(muon_true_hists,    "mu_MC_total_true")
    h_ka_tot_true = sum_hists(kaon_true_hists,    "ka_MC_total_true")
    h_ne_tot_true = sum_hists(neutron_true_hists, "ne_MC_total_true")

    h_mu_tot_pred = sum_hists(muon_pred_hists,    "mu_MC_total_pred")
    h_ka_tot_pred = sum_hists(kaon_pred_hists,    "ka_MC_total_pred")
    h_ne_tot_pred = sum_hists(neutron_pred_hists, "ne_MC_total_pred")

    mu_true_y = integral(h_mu_tot_true)
    ka_true_y = integral(h_ka_tot_true)
    ne_true_y = integral(h_ne_tot_true)

    mu_pred_y = integral(h_mu_tot_pred)
    ka_pred_y = integral(h_ka_tot_pred)
    ne_pred_y = integral(h_ne_tot_pred)

    # ---------- neutrino components (true & pred) ----------
    nu_keys  = ["vm", "ve", "vt", "NC"]
    nu_names = [r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"$\nu_{\tau}$ CC", "NC"]

    nu_true_yields = []
    nu_pred_yields = []

    for i, key in enumerate(nu_keys):
        h_true = get_nu_hist(MC_neutrino_true_hists, i, key)
        h_pred = get_nu_hist(MC_neutrino_pred_hists, i, key)
        nu_true_yields.append(integral(h_true))
        nu_pred_yields.append(integral(h_pred))

    # ---------- data-pred yields (with GNN) ----------
    # data_pred_hists is list or dict with muon / kaon / neutron
    data_mu = get_data_hist(data_pred_hists, 0, "muon")
    data_ka = get_data_hist(data_pred_hists, 1, "kaon")
    data_ne = get_data_hist(data_pred_hists, 2, "neutron")

    data_mu_y = integral(data_mu)
    data_ka_y = integral(data_ka)
    data_ne_y = integral(data_ne)

    # ---------- totals ----------
    mc_true_total    = sum(nu_true_yields) + mu_true_y + ka_true_y + ne_true_y
    mc_pred_total    = sum(nu_pred_yields) + mu_pred_y + ka_pred_y + ne_pred_y
    data_pred_total  = data_mu_y + data_ka_y + data_ne_y


    # ============================================================
    #              BUILD TABLE CONTENT
    # ============================================================
    header = ["Component", "MC (no GNN)", "MC (with GNN)", "Data-pred (with GNN)"]

    rows = []

    # Neutrino rows
    for name, y_true, y_pred in zip(nu_names, nu_true_yields, nu_pred_yields):
        rows.append([name, y_true, y_pred, ""])

    # Backgrounds
    rows.append(["Muon bkg",    mu_true_y, mu_pred_y, data_mu_y])
    rows.append(["Kaon bkg",    ka_true_y, ka_pred_y, data_ka_y])
    rows.append(["Neutron bkg", ne_true_y, ne_pred_y, data_ne_y])

    # Totals
    rows.append(["Total MC",       mc_true_total, mc_pred_total, ""])
    rows.append(["Total data-pred", "", "", data_pred_total])

    # helper for formatting
    def fmt(x):
        if isinstance(x, str):
            return "-" if x == "" else x
        return f"{x:.1f}"

    # ============================================================
    #              PRINT TABLE TO STDOUT
    # ============================================================
    print("\n=== Yield summary ===")
    print(f"cuts: 1) no veto hit, 2) SciFi > 200, 3) {cut_name}")

    col_widths = [20, 18, 18, 22]
    def pad(text, w): return str(text).ljust(w)

    line = "  ".join(pad(h, w) for h, w in zip(header, col_widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [fmt(v) for v in r]
        print("  ".join(pad(c, w) for c, w in zip(cells, col_widths)))

    
    # ============================================================
    #                   SAVE TABLE AS CSV
    # ============================================================
    csv_path = os.path.join(out_dir, f"yield_summary_{cut_name}.csv")


    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            # write raw values (not fmt()), to preserve numeric data
            out_row = [r[0]]  # component name
            for val in r[1:]:
                if isinstance(val, str) and val == "":
                    out_row.append("")     # empty cell
                else:
                    out_row.append(val)    # numeric or string
            writer.writerow(out_row)

    print(f"Wrote CSV summary to {csv_path}")
    
    # ============================================================
    #              MATPLOTLIB TABLE PLOT
    # ============================================================
    # Output directory
    out_dir = os.path.join("plots", hist_name)
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"yield_summary_{cut_name}.pdf")

    # Prepare data for matplotlib.table
    row_labels = [r[0] for r in rows]
    cell_data  = [[fmt(v) for v in r[1:]] for r in rows]  # drop first col (name) into table cells

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Title / cut description
    title_str = f"Yields summary\nCuts: 1) no veto hit, 2) SciFi > 200, 3) {cut_name}"
    ax.set_title(title_str, fontsize=12, pad=20)

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=header[1:],  # skip "Component" which becomes rowLabels
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)  # (x, y) cell scaling; tweak y if you have many rows

    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"\nWrote yield table plot (matplotlib) to {out_pdf}")
def plot_MC_vs_data(data_hist, muon_true_hists, kaon_true_hists, neutron_true_hists, hist_name, data_int_lumi, logy):
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)
    
    # ---------------- build totals ----------------
    h_mu_tot = sum_hists(muon_true_hists,        "mu_MC_total")
    h_ka_tot = sum_hists(kaon_true_hists,        "ka_MC_total")
    h_ne_tot = sum_hists(neutron_true_hists,     "ne_MC_total")

    # Put totals in a dict for stacking
    combined_mc = {
        "kaon":     h_ka_tot,
        "neutron":  h_ne_tot,
        "muon":     h_mu_tot,
    }

    # Remove None/empty totals so stack doesn’t choke
    def integral_with_oflow(h):
        return h.Integral(0, h.GetNbinsX() + 1)

    combined_mc = {k: h for k, h in combined_mc.items()
                   if h is not None and integral_with_oflow(h) > 0}

    if not combined_mc:
        print("[WARN] plot_MC_vs_data: no non-empty MC totals to draw.")
        return

    # ---------------- make stack ----------------
    stack_tot = ROOT.THStack("MC_tot_stack_true", "")

    # Choose colors for the 4 categories
    colors = {
        "muon":     ROOT.kRed+1,
        "kaon":     ROOT.kBlue,
        "neutron":  ROOT.kViolet+2
    }

    # THStack uses the hist objects you add; clone+style for safety
    comps_tot = []
    for k, h in combined_mc.items():
        hh = h.Clone(f"{k}_stack_clone_true")
        hh.SetDirectory(0)
        hh.SetFillColor(colors.get(k, ROOT.kGray+1))
        hh.SetLineColor(colors.get(k, ROOT.kGray+1))
        stack_tot.Add(hh)
        comps_tot.append((k, hh))

    # ---------------- draw vs data ----------------
    _draw_mc_stack_with_data(
        neutron_true_hists,
        stack_tot,
        comps_tot,
        data_hist,  # already a hist; overlay as points
        particle="No veto hit",
        out_pdf=os.path.join(outdir, f"BeforeGNN_MC_vs_data_Total_True_{hist_name}.pdf"),
        int_lumi=data_int_lumi,
        logy=logy
    )
    # sum true muon, kaon, neutron seperately and the stack them toghter
    # data_hist is already a hist
    # use _draw_mc_stack_with_pred to plot

def plot_MC_VS_MC_pred(data_pred_hists,
                       MC_neutrino_true_hists, MC_neutrino_pred_hists,
                       muon_true_hists, muon_pred_hists,
                       kaon_true_hists, kaon_pred_hists,
                       neutron_true_hists, neutron_pred_hists, 
                       hist_name,int_lumi, logy=False,
                       ):
    
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)

    # Sum to totals
    nu_true  = _sum_th1_dict(MC_neutrino_true_hists, "nu_true")
    nu_pred  = _sum_th1_dict(MC_neutrino_pred_hists, "nu_pred")
    mu_true  = _sum_th1_dict(muon_true_hists,        "mu_true")
    mu_pred  = _sum_th1_dict(muon_pred_hists,        "mu_pred")
    ka_true  = _sum_th1_dict(kaon_true_hists,        "ka_true")
    ka_pred  = _sum_th1_dict(kaon_pred_hists,        "ka_pred")
    neutron_true  = _sum_th1_dict(neutron_true_hists,        "neutron_true")
    neutron_pred  = _sum_th1_dict(neutron_pred_hists,        "neutron_pred")
    
    data_muon = data_pred_hists["muon"]
    data_kaon = data_pred_hists["kaon"]
    data_neutron = data_pred_hists["neutron"]
    
    

    out_nu = os.path.join(outdir, f"MC_and_data_shape_neutrino_{hist_name}.pdf")
    out_mu = os.path.join(outdir, f"MC_and_data_shape_muon_{hist_name}.pdf")
    out_ka = os.path.join(outdir, f"MC_and_data_shape_kaon_{hist_name}.pdf")
    out_neutron = os.path.join(outdir, f"MC_and_data_shape_neutron_{hist_name}.pdf")
    out_ve = os.path.join(outdir, f"MC_and_data_shape_neutrino_ve_{hist_name}.pdf")
    out_vm = os.path.join(outdir, f"MC_and_data_shape_neutrino_vm_{hist_name}.pdf")
    out_vt = os.path.join(outdir, f"MC_and_data_shape_neutrino_vt_{hist_name}.pdf")
    out_NC = os.path.join(outdir, f"MC_and_data_shape_neutrino_NC_{hist_name}.pdf")
    
    
    
    _draw_pair(nu_true, nu_pred, f"Neutrino: MC True vs Pred ({hist_name})", out_nu, "Neutrino",int_lumi, logy=logy)
    _draw_pair(mu_true, mu_pred, f"Muon: MC True vs Pred ({hist_name})",     out_mu, "Muon", int_lumi,h_data=data_muon, logy=logy)
    _draw_pair(ka_true, ka_pred, f"Kaon: MC True vs Pred ({hist_name})",     out_ka,"Kaon", int_lumi,h_data=data_kaon, logy=logy)
    _draw_pair(neutron_true, neutron_pred, f"Neutron: MC True vs Pred ({hist_name})", out_neutron,"Neutron", int_lumi,h_data=data_neutron, logy=logy)
    
    _draw_pair(
        MC_neutrino_true_hists['ve'], MC_neutrino_pred_hists['ve'],
        f"#nu_{{e}} CC: MC True vs Pred ({hist_name})", out_ve, "#nu_{e} CC", int_lumi, logy=logy
    )
    _draw_pair(
        MC_neutrino_true_hists['vm'], MC_neutrino_pred_hists['vm'],
        f"#nu_{{#mu}} CC: MC True vs Pred ({hist_name})", out_vm, "#nu_{#mu} CC", int_lumi, logy=logy
    )
    _draw_pair(
        MC_neutrino_true_hists['vt'], MC_neutrino_pred_hists['vt'],
        f"#nu_{{#tau}} CC: MC True vs Pred ({hist_name})", out_vt, "#nu_{#tau} CC", int_lumi, logy=logy
    )
    _draw_pair(MC_neutrino_true_hists['NC'], MC_neutrino_pred_hists['NC'], f"NC: MC True vs Pred ({hist_name})", out_NC, "NC", int_lumi, logy=logy)

    
def _draw_pair(h_true, h_pred, title, out_pdf, particle, int_lumi, h_data=None, logy=False, as_density=True):

    # Clone so we don't touch inputs
    t = h_true.Clone(f"{h_true.GetName()}_draw"); t.SetDirectory(0)
    p = h_pred.Clone(f"{h_pred.GetName()}_draw"); p.SetDirectory(0)
    
    
    d = None
    if h_data is not None:
        d = h_data.Clone(f"{h_data.GetName()}_draw"); d.SetDirectory(0)


     # ---- Normalize to probability density (unit area) if requested ----
    def _to_density(h):
        if not h:
            return
        integral = h.Integral()
        if integral != 0:
            h.Scale(1.0 / integral)

    if as_density:
        _to_density(t)
        _to_density(p)
        _to_density(d)
        
    # Style
    t.SetLineColorAlpha(ROOT.kBlue, 0.5); t.SetLineWidth(3)
    p.SetLineColorAlpha(ROOT.kRed,  0.5); p.SetLineWidth(3)
    if d:
        d.SetMarkerStyle(20); d.SetMarkerSize(1.0)
        d.SetLineColor(ROOT.kBlack); d.SetMarkerColor(ROOT.kBlack)

    # Canvas
    c = ROOT.TCanvas(f"c_{title}", title, 900, 750)
    if logy: c.SetLogy()

    # Pick a histogram to own the axes (first non-empty among t, p, d)
    axis_hist = None
    for h in (t, p, d):
        if h is not None and h.Integral() != 0:
            axis_hist = h
            break
    if axis_hist is None:
        axis_hist = t  # fall back to t even if empty, to avoid crashes

    # Axis setup (include data if present)
    hists = [h for h in (t, p, d) if h is not None]
    ymax = max(h.GetMaximum() for h in hists) if hists else 1.0

    axis_hist.SetTitle("")
    axis_hist.GetYaxis().SetTitle("Probability density" if as_density else "Events")
    axis_hist.SetMaximum(ymax * (10.0 if logy else 1.35))

    if logy:
        positive_bins = [
            h.GetBinContent(i)
            for h in hists
            for i in range(1, h.GetNbinsX() + 1)
            if h.GetBinContent(i) > 0
        ]
        ymin = min(positive_bins) if positive_bins else 1e-9
        axis_hist.SetMinimum(ymin * 0.1)
    else:
        mins = [h.GetMinimum() for h in hists] if hists else [0.0]
        axis_hist.SetMinimum(min(mins) * 0.5)

    # Draw (make sure the axis_hist is drawn first)
    def draw_hist(h, primary_opt, same_opt=None, also_err=False):
        if h is None or h.Integral() == 0: return
        h.Draw(primary_opt)
        if same_opt:
            h.Draw(same_opt)
        if also_err:
            h.Draw("E1 SAME")

    # Primary draw for axis owner
    if axis_hist is t:
        draw_hist(t, "HIST")
    elif axis_hist is p:
        draw_hist(p, "HIST"); p.Draw("E1 SAME")
    elif axis_hist is d:
        draw_hist(d, "E1")

    # Draw the others
    if axis_hist is not t: draw_hist(t, "HIST SAME")
    if axis_hist is not p:
        draw_hist(p, "HIST SAME")
        if p is not None and p.Integral() != 0:
            p.Draw("E1 SAME")
    if d and axis_hist is not d:
        draw_hist(d, "E1 SAME")

    # Legend
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    leg.AddEntry(t, f"MC {particle}", "l")
    leg.AddEntry(p, f"MC {particle} (GNN Selected)", "l")
    if d:
        lbl = f"{particle}-like (Data)"
        leg.AddEntry(d, lbl, "lep")
    leg.Draw()

    # Title placeholder (kept as in original)
    pave = ROOT.TPaveText(0.12, 0.92, 0.88, 0.99, "NDC")
    pave.SetFillStyle(0); pave.SetBorderSize(0); pave.AddText(""); pave.Draw()

    c.Print(out_pdf)
    print(f"[OK] wrote {out_pdf}")

def _sum_th1_dict(hdict, name, normalize=False):
    """Sum all TH1 in a dict into a single TH1 clone."""
    if not hdict:
        raise ValueError(f"{name}: empty histogram dict")
    it = iter(hdict.values())
    hsum = next(it).Clone(f"{name}_sum")
    hsum.SetDirectory(0)
    for h in it:
        hsum.Add(h)
        
    if normalize:
        integral = hsum.Integral()
        if integral != 0:
            hsum.Scale(1.0 / integral)
        
    return hsum

def _draw_totals_overlay(hsum_dict, title, out_pdf,int_lumi, logy=False):
    # Style map
    colmap = {
        "ve":      ROOT.kMagenta+1,   # electron neutrino
        "vm":      ROOT.kBlue+1,      # muon neutrino
        "vt":      ROOT.kOrange+1,    # tau neutrino
        "NC":      ROOT.kTeal+2,     # neutral current
        "muon":    ROOT.kRed+1,
        "kaon":    ROOT.kGreen+2,
        "neutron": ROOT.kViolet+2,
    }
    labelmap = {
    "ve": "#nu_{e} CC",
    "vm": "#nu_{#mu} CC",
    "vt": "#nu_{#tau} CC",
    "NC": "NC",
    "muon": "Muon",
    "kaon": "Kaon",
    "neutron": "Neutron"
    }
    markmap = {
    "ve": 22, "vm": 22, "vt": 22, "NC": 22,  
        "muon": 21, "kaon": 20, "neutron": 20   
    }

    # Prepare canvas
    c = ROOT.TCanvas(f"c_{title}", title, 900, 750)
    if logy: c.SetLogy()

    # clone & style
    styled = {}
    for k, h in hsum_dict.items():
        hh = h.Clone(f"{k}_total_clone"); hh.SetDirectory(0)
        col = colmap.get(k, ROOT.kBlack)
        mk  = markmap.get(k, 24)
        hh.SetLineColor(col); hh.SetMarkerColor(col)
        hh.SetMarkerStyle(mk); hh.SetMarkerSize(1.0); hh.SetLineWidth(2)
        styled[k] = hh

    # axis & range
    non_empty = [h for h in styled.values() if h.Integral() != 0]

    first = non_empty[0]
    ymax = max(h.GetMaximum() for h in styled.values())
    ymin = min(
        (h.GetBinContent(i)
        for h in styled.values()
        for i in range(1, h.GetNbinsX() + 1)
        if h.GetBinContent(i) > 0),
        default=1e-9
    )
    
    print(f"(ymin, ymax): ({ymin}, {ymax})")
    first.SetMaximum(ymax * (10.0 if logy else 1.35))
    first.SetMinimum(ymin * (0.1 if logy else 0.5))

    first.SetTitle("")
    first.GetYaxis().SetTitle("Expected Events")
    ya = first.GetYaxis()
    ya.SetNoExponent(False)     
    #ya.SetMaxDigits(1)  
    #ROOT.TGaxis.SetExponentOffset(-0.07, 0.01, "y")
    
    first.Draw("E1")

    for k, h in styled.items():
        if h is not first and (h.Integral() != 0):
            h.Draw("E1 SAME")

    leg = ROOT.TLegend(0.60, 0.65, 0.88, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    for k, h in styled.items():
        leg.AddEntry(h, f"MC {labelmap.get(k, k.capitalize())}", "lep")
    leg.Draw()
    
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)  # smaller than 0.045
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.3f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")


    c.Print(out_pdf)
    print(f"[OK] wrote {out_pdf}")

def plot_MC(MC_neutrino_true_hists, muon_true_hists, kaon_true_hists,neutron_true_hists, hist_name, int_lumi, logy=True):
    #hist of neutrino, muon, kaon. sum of their hists 
    
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)

    # Build stacks (and totals) using your helper
    # neutrino: no special sort
    #sum_nu = _sum_th1_dict(MC_neutrino_true_hists, "MC_neutrino")
    # muon: order beam types
    sum_mu = _sum_th1_dict(muon_true_hists, "MC_muon")
    # kaon: sort numeric ranges
    sum_ka = _sum_th1_dict(kaon_true_hists, "MC_kaon")
    
    sum_neutron = _sum_th1_dict(neutron_true_hists, "MC_neutron")
    
    _draw_totals_overlay(
        {"ve": MC_neutrino_true_hists["ve"], "vm": MC_neutrino_true_hists["vm"], "vt": MC_neutrino_true_hists["vt"], "NC": MC_neutrino_true_hists["NC"],
         "muon": sum_mu, "kaon": sum_ka, "neutron": sum_neutron},
        title=f"MC Signal vs Backgrounds — {hist_name}",
        out_pdf=os.path.join(outdir, f"All_MC_shape_{hist_name}.pdf"),
        int_lumi=int_lumi,
        logy=logy)
     
def plot_data_pred_bkg(data_pred_hists, hist_name, int_lumi, logy=True):
    """
    Plot kaon, neutron, muon prediction histograms on one canvas.
    data_pred_hists : dict with keys 'kaon', 'neutron', 'muon' and TH1 values
    hist_name       : used in output filename
    logy            : if True, draw with log y axis
    """
    required = ["kaon", "neutron", "muon"]
    for k in required:
        if k not in data_pred_hists:
            raise KeyError(f"Missing key '{k}' in data_pred_hists")

    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)


    # Clone & style
    colors = {
        "muon":     ROOT.kBlue+1,
        "kaon":     ROOT.kRed+1,
        "neutron":ROOT.kGreen+1,
    }
    markers = {
        "kaon": 20,
        "neutron": 21,
        "muon": 22
    }

    hists = {}
    for sp in required:
        h = data_pred_hists[sp].Clone(f"{sp}_clone")
        h.SetDirectory(0)
        h.SetLineColor(colors[sp])
        h.SetMarkerColor(colors[sp])
        h.SetMarkerStyle(markers[sp])
        h.SetMarkerSize(1.0)
        h.SetLineWidth(2)
        hists[sp] = h

    # Determine max for y-axis
    ymax = max(h.GetMaximum() for h in hists.values())
    ymin = min(
        (h.GetBinContent(i)
        for h in hists.values()
        for i in range(1, h.GetNbinsX() + 1)
        if h.GetBinContent(i) > 0),
        default=1e-9
    )
    for h in hists.values():
        h.SetMaximum(ymax * (1.35 if not logy else 5.0))
        h.SetMinimum(ymin * 0.5)

    # Canvas
    c = ROOT.TCanvas(f"c_{hist_name}", hist_name, 900, 750)
    if logy: c.SetLogy()

    # Draw
    first_drawn = True
    for sp in required:
        h = hists[sp]
        # remove the nested print; cast entries to int if you prefer
        print(f"{sp}: Entries = {int(h.GetEntries())}, Integral = {h.Integral()}")

        if h.Integral() != 0:
            print(f"drawing {sp}")
            draw_opt = "E1" if first_drawn else "E1 SAME"
            h.Draw(draw_opt)
            first_drawn = False

    # Legend
    leg = ROOT.TLegend(0.60, 0.65, 0.88, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    for sp in required:
        leg.AddEntry(hists[sp], f"{sp.capitalize()}-like (Data)", "lep")
    leg.Draw()
    
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)  # smaller than 0.045
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.3f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")

    outpath = os.path.join(outdir, f"Data_GNN_selected_{hist_name}.pdf")
    c.Print(outpath)
    print(f"[OK] Wrote {outpath}")
    


def _parse_range_key(k):
    """Parse '(lo, hi)' strings to a float tuple for sorting; fallback to original."""
    m = re.match(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)", str(k))
    return (float(m.group(1)), float(m.group(2))) if m else k

def _stack_hists(hdict, name, sort_fn=None, color_seq=None):
    """Return (stack, components_list) from dict[str, TH1]."""
    keys = list(hdict.keys())
    if sort_fn:
        try:
            keys = sorted(keys, key=sort_fn)
        except Exception:
            pass
    stack = ROOT.THStack(f"stack_{name}", "")
    comps = []
    colors = color_seq or [ROOT.kRed+1, ROOT.kRed+2, ROOT.kBlue+1,
                           ROOT.kBlue+2, ROOT.kGreen+2, ROOT.kGreen+3,
                           ROOT.kCyan+1, ROOT.kCyan+2, ROOT.kMagenta+1]
    
    for i, k in enumerate(keys):
        h = hdict[k].Clone(f"{name}_{i}"); h.SetDirectory(0)
        col = colors[i % len(colors)]
        h.SetFillColor(col); h.SetLineColor(col); h.SetLineWidth(1)
        #print(f"{k}: entries = {h.GetEntries()}, integral = {h.Integral()}")
        stack.Add(h)
        comps.append((str(k), h))

    return stack, comps

def min_positive_bin_content_stack(stack):
    min_val = float("inf")
    n_bins = stack.GetHistogram().GetNbinsX()
    for ib in range(1, n_bins + 1):
        # sum contents of all hists in the stack for this bin
        bin_sum = 0.0
        for h in stack.GetHists():
            bin_sum += h.GetBinContent(ib)
        if bin_sum > 0 and bin_sum < min_val:
            min_val = bin_sum
    return min_val if min_val != float("inf") else 0.0



def _draw_mc_stack_with_data(MC_neutrino_hists, stack, comps, h_pred, particle, out_pdf,int_lumi,
                             normalize_mc_to_pred=False, logy=False):
    
    title = f"MC vs Pred Data ({particle})"
    c = ROOT.TCanvas(f"c_{title}", title, 900, 750)

    # --- Pads: top = main plot, bottom = ratio ---
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30)
    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.30)
    pad1.Draw()
    pad2.Draw()

    if logy:
        pad1.SetLogy()

    # --- Clone pred to keep input pristine ---
    h_pred_d = h_pred.Clone(f"pred_{title}")
    h_pred_d.SetDirectory(0)
    h_pred_d.SetMarkerStyle(20)
    h_pred_d.SetMarkerSize(1.0)
    h_pred_d.SetLineColor(ROOT.kBlack)

    # --- Special nu_e component ---
    h_ve = MC_neutrino_hists.get("ve", None)
    print(h_ve)
    if h_ve:
        h_ve = h_ve.Clone(f"nu_e_{title}")
        h_ve.SetDirectory(0)
        h_ve.SetMarkerStyle(24)
        h_ve.SetMarkerSize(1.0)
        h_ve.SetLineColor(ROOT.kPink)
        h_ve.SetMarkerColor(h_ve.GetLineColor())

    # --- Check stack content safely ---
    hlist = stack.GetHists()
    has_content = bool(hlist) and any(h.Integral() != 0 for h in hlist)

    # --- Optional normalization of MC to pred ---
    if has_content and normalize_mc_to_pred and h_pred_d.Integral() > 0:
        total_mc = stack.GetStack().Last().Integral()
        if total_mc > 0:
            sf = h_pred_d.Integral() / total_mc
            for h in hlist:
                h.Scale(sf)
            if h_ve:
                h_ve.Scale(sf)

    # ========================================================
    #              TOP PAD: STACK + DATA + nu_e
    # ========================================================
    pad1.cd()

    if has_content:
        stack.Draw("HIST")

        stack.GetYaxis().SetTitle("Events")
        stack.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())

        # Hide X labels on top pad (ratio pad will handle them)
        stack.GetXaxis().SetLabelSize(0.)
        stack.GetXaxis().SetTitleSize(0.)

        ymax = max(stack.GetMaximum(), h_pred_d.GetMaximum())
        ymin = min_positive_bin_content_stack(stack)

        print(f"in _draw_mc_stack_with_data, (ymin, ymax): ({ymin}, {ymax})")

        stack.SetMaximum(ymax * (1.35 if not logy else 5.0))
        if ymin > 0:
            stack.SetMinimum(ymin * 0.5)
        elif logy:
            stack.SetMinimum(1e-3)
    else:
        # Fallback: draw pred alone if no MC
        h_pred_d.Draw("E1")
        h_pred_d.GetYaxis().SetTitle("Events")
        h_pred_d.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
        h_pred_d.GetXaxis().SetLabelSize(0.)
        h_pred_d.GetXaxis().SetTitleSize(0.)

    # Overlay pred and nu_e
    if h_pred_d.Integral() != 0:
        h_pred_d.Draw("E1 SAME")
    if h_ve and h_ve.Integral() != 0:
        h_ve.Draw("E1 SAME")

    # Legend
    leg = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    for label, h in comps:
        leg.AddEntry(h, f"MC {label}", "f")

    if particle == "No veto hit":
        leg.AddEntry(h_pred_d, "Data (without veto hit)", "lep")
    else:
        leg.AddEntry(h_pred_d, f"{particle}-like (data)", "lep")

    if h_ve:
        leg.AddEntry(h_ve, "#nu_{e} CC", "lep")

    leg.Draw()

    # Title / text at top
    pave = ROOT.TPaveText(0.12, 0.92, 0.88, 0.99, "NDC")
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    pave.AddText("")
    pave.Draw()

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)

    # First line (lumi)
    label.SetTextSize(0.040)
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.3f} fb^{{-1}}")

    # Second line (preselection)
    label.SetTextSize(0.030)
    label.SetTextAlign(11)
    #label.DrawLatex(0.20, 0.94, "Preselection: SciFi Hits > 200")

    # ========================================================
    #              BOTTOM PAD: RATIO (Data / MC)
    # ========================================================
    pad2.cd()

    if has_content:
        # Total MC histogram from stack
        mc_tot = stack.GetStack().Last().Clone(f"mc_tot_{title}")
        mc_tot.SetDirectory(0)

        if mc_tot.Integral() > 0:
            h_ratio = h_pred_d.Clone(f"ratio_{title}")
            h_ratio.SetDirectory(0)
            h_ratio.Divide(mc_tot)

            h_ratio.SetMarkerStyle(20)
            h_ratio.SetMarkerSize(0.8)
            h_ratio.SetLineColor(ROOT.kBlack)

            h_ratio.GetYaxis().SetTitle("Data/MC")
            h_ratio.GetYaxis().SetNdivisions(505)
            h_ratio.GetYaxis().SetTitleSize(0.10)
            h_ratio.GetYaxis().SetLabelSize(0.08)
            h_ratio.GetYaxis().SetTitleOffset(0.5)

            h_ratio.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
            h_ratio.GetXaxis().SetTitleSize(0.12)
            h_ratio.GetXaxis().SetLabelSize(0.10)

            # Ratio range – can tune if you want
            ratio_vals = [
            h_ratio.GetBinContent(i)
                for i in range(1, h_ratio.GetNbinsX() + 1)
                if h_ratio.GetBinContent(i) > 0 and h_ratio.GetBinError(i) < 1e5
            ]

            if len(ratio_vals) == 0:
                # Edge case: no valid ratio bins
                h_ratio.SetMinimum(0.5)
                h_ratio.SetMaximum(1.5)
            else:
                rmin = min(ratio_vals)
                rmax = max(ratio_vals)

                # Add 20% padding
                padding = 0.20
                ymin = max(0.0, rmin * (1 - padding))
                ymax = rmax * (1 + padding)

                # Protect against collapse if rmin ≈ rmax
                if abs(ymax - ymin) < 0.05:
                    ymin = rmin - 0.1
                    ymax = rmax + 0.1

                # Final bounds
                h_ratio.SetMinimum(ymin)
                h_ratio.SetMaximum(ymax)

            h_ratio.Draw("E1")

            # Line at ratio = 1
            x_min = h_ratio.GetXaxis().GetXmin()
            x_max = h_ratio.GetXaxis().GetXmax()
            line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
            line.SetLineStyle(2)
            line.Draw("SAME")

            # Keep a reference so Python doesn't garbage-collect it
            c._ratio_line = line
        else:
            # Degenerate case: no MC content
            frame = pad2.DrawFrame(h_pred.GetXaxis().GetXmin(), 0.0,
                                   h_pred.GetXaxis().GetXmax(), 2.0)
            frame.GetYaxis().SetTitle("Data/MC")
            frame.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
    else:
        # No MC: just draw empty ratio frame
        frame = pad2.DrawFrame(h_pred.GetXaxis().GetXmin(), 0.0,
                               h_pred.GetXaxis().GetXmax(), 2.0)
        frame.GetYaxis().SetTitle("Data/MC")
        frame.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())

    c.Print(out_pdf)
    print(f"Wrote {out_pdf}")

def _draw_mc_stack_with_pred(stack, comps, h_pred, particle, out_pdf, int_lumi,
                             normalize_mc_to_pred=False, logy=False):
    """
    Draw MC stack vs prediction with a Data/MC ratio panel.

    stack   : THStack with MC components
    comps   : list of (label, hist) tuples (same hists as in stack, for legend)
    h_pred  : "data-like" prediction histogram
    particle: string label ("Muon", etc.)
    out_pdf : output file name
    int_lumi: integrated luminosity in fb^-1
    """

    title = f"MC vs Pred Data ({particle})"
    c = ROOT.TCanvas(f"c_{title}", title, 900, 750)

    # --- Two pads: top = main plot, bottom = ratio ---
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30)
    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.30)
    pad1.Draw()
    pad2.Draw()

    if logy:
        pad1.SetLogy()

    # --- Clone pred to keep input pristine ---
    h_pred_d = h_pred.Clone(f"pred_{title}")
    h_pred_d.SetDirectory(0)
    h_pred_d.SetMarkerStyle(20)
    h_pred_d.SetMarkerSize(1.0)
    h_pred_d.SetLineColor(ROOT.kBlack)

    # --- Collect MC histograms from stack ---
    hlist_raw = stack.GetHists()
    hlist = [h for h in hlist_raw] if hlist_raw else []
    has_content = bool(hlist) and any(h.Integral() != 0 for h in hlist)

    # --- Build total MC histogram (sum) for normalization & ratio ---
    hsum = None
    if has_content:
        hsum = hlist[0].Clone(f"hsum_{title}")
        hsum.SetDirectory(0)
        hsum.Reset()
        for h in hlist:
            hsum.Add(h)

    # --- Optional normalization: scale MC to pred yield ---
    if normalize_mc_to_pred and has_content:
        i_pred = h_pred_d.Integral()
        i_mc   = hsum.Integral()
        if i_pred > 0 and i_mc > 0:
            scale = i_pred / i_mc
            # scale components (for legend consistency)
            for _, h in comps:
                h.Scale(scale)
            # scale the stack contents and hsum
            for h in hlist:
                h.Scale(scale)
            hsum.Scale(scale)

    # ========================================================
    #                TOP PAD: STACK + DATA
    # ========================================================
    pad1.cd()

    if has_content:
        stack.Draw("HIST")

        stack.GetYaxis().SetTitle("Events")
        stack.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())

        # Hide X labels on top pad – ratio pad will show them
        stack.GetXaxis().SetLabelSize(0.)
        stack.GetXaxis().SetTitleSize(0.)

        ymax = max(stack.GetMaximum(), h_pred_d.GetMaximum())
        ymin = min_positive_bin_content_stack(stack)

        print(f"in _draw_mc_stack_with_pred, (ymin, ymax): ({ymin}, {ymax})")

        stack.SetMaximum(ymax * (1.35 if not logy else 5.0))
        if ymin > 0:
            stack.SetMinimum(ymin * 0.5)
        elif logy:
            stack.SetMinimum(1e-3)
    else:
        # Fallback: no MC, only pred
        h_pred_d.Draw("E1")
        h_pred_d.GetYaxis().SetTitle("Events")
        h_pred_d.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
        h_pred_d.GetXaxis().SetLabelSize(0.)
        h_pred_d.GetXaxis().SetTitleSize(0.)

    # Overlay pred
    if h_pred_d.Integral() != 0:
        h_pred_d.Draw("E1 SAME")

    # --- Legend ---
    leg = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    if particle == "Muon":
        for label, h in comps:
            leg.AddEntry(h, f"MC {particle} ({label})", "f")
    else:
        for label, h in comps:
            leg.AddEntry(h, f"MC {particle} ({label.strip('()').replace(',', '-')}GeV)", "f")

    leg.AddEntry(h_pred_d, f"{particle}-like (data)", "lep")
    leg.Draw()

    # --- Title / lumi text ---
    pave = ROOT.TPaveText(0.12, 0.92, 0.88, 0.99, "NDC")
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    pave.AddText("")
    pave.Draw()

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.3f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")

    # ========================================================
    #                BOTTOM PAD: RATIO (Data/MC)
    # ========================================================
    pad2.cd()

    if has_content and hsum.Integral() > 0:
        h_ratio = h_pred_d.Clone(f"ratio_{title}")
        h_ratio.SetDirectory(0)
        h_ratio.Divide(hsum)

        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.8)
        h_ratio.SetLineColor(ROOT.kBlack)

        h_ratio.GetYaxis().SetTitle("Data/MC")
        h_ratio.GetYaxis().SetNdivisions(505)
        h_ratio.GetYaxis().SetTitleSize(0.10)
        h_ratio.GetYaxis().SetLabelSize(0.08)
        h_ratio.GetYaxis().SetTitleOffset(0.5)

        h_ratio.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
        h_ratio.GetXaxis().SetTitleSize(0.12)
        h_ratio.GetXaxis().SetLabelSize(0.10)

        # --- Auto-range for ratio ---
        ratio_vals = [
            h_ratio.GetBinContent(i)
            for i in range(1, h_ratio.GetNbinsX() + 1)
            if h_ratio.GetBinContent(i) > 0 and h_ratio.GetBinError(i) < 1e5
        ]

        if len(ratio_vals) == 0:
            ymin, ymax = 0.5, 1.5
        else:
            rmin = min(ratio_vals)
            rmax = max(ratio_vals)

            # Make sure 1 is inside the range
            rmin = min(rmin, 1.0)
            rmax = max(rmax, 1.0)

            pad = 0.20
            ymin = rmin * (1 - pad)
            ymax = rmax * (1 + pad)

            if abs(ymax - ymin) < 0.05:
                ymin = rmin - 0.1
                ymax = rmax + 0.1

        h_ratio.SetMinimum(ymin)
        h_ratio.SetMaximum(ymax)

        h_ratio.Draw("E1")

        # Line at ratio = 1
        x_min = h_ratio.GetXaxis().GetXmin()
        x_max = h_ratio.GetXaxis().GetXmax()
        line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
        line.SetLineStyle(2)
        line.Draw("SAME")

        # keep reference alive
        c._ratio_line = line
    else:
        # No MC or zero MC: empty frame
        frame = pad2.DrawFrame(h_pred.GetXaxis().GetXmin(), 0.0,
                               h_pred.GetXaxis().GetXmax(), 2.0)
        frame.GetYaxis().SetTitle("Data/MC")
        frame.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
        frame.GetYaxis().SetTitleSize(0.10)
        frame.GetYaxis().SetLabelSize(0.08)
        frame.GetYaxis().SetTitleOffset(0.5)
        frame.GetXaxis().SetTitleSize(0.12)
        frame.GetXaxis().SetLabelSize(0.10)

    c.Print(out_pdf)
    print(f"Wrote {out_pdf}")
    
def _beam_order(k):
    order = {"up": 0, "down": 1, "horizontal": 2}
    return order.get(str(k), 99)
    
def sum_hists(hdict, name):
    """Return TH1 sum of all hists in dict."""
    hsums = None
    for _, h in hdict.items():
        if hsums is None:
            hsums = h.Clone(name)
            hsums.SetDirectory(0)
            hsums.Reset("ICES")
        hsums.Add(h)
    return hsums

def plot_MC_pred_VS_data_pred_seperated(data_pred_hists, muon_pred_hists, kaon_pred_hists,neutron_pred_hists, hist_name,int_lumi, logy):
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)

    if "muon" not in data_pred_hists or "kaon" not in data_pred_hists or "neutron" not in data_pred_hists:
        raise KeyError("data_pred_hists must contain keys 'muon', 'neutron' and 'kaon'.")

    # ---- Muon: stack by beam type (use natural order up/down/horizontal if present) ----
    stack_mu, comps_mu= _stack_hists(
        muon_pred_hists, "MC_muon", sort_fn=_beam_order,
        color_seq=[ROOT.kAzure+1, ROOT.kOrange-3, ROOT.kGreen+2]
    )
    
    _draw_mc_stack_with_pred(
        stack_mu, comps_mu, data_pred_hists["muon"],
        particle="Muon",
        out_pdf=os.path.join(outdir, f"Stack_MC_and_data_Muon_{hist_name}.pdf"),
        int_lumi = int_lumi,
        logy=logy
    )

    # ---- Kaon: stack by energy ranges '(lo, hi)' sorted numerically ----
    stack_ka, comps_ka = _stack_hists(
        kaon_pred_hists, "MC_kaon", sort_fn=_parse_range_key
    )
    _draw_mc_stack_with_pred(
        stack_ka, comps_ka, data_pred_hists["kaon"],
        particle="Kaon",
        out_pdf=os.path.join(outdir, f"Stack_MC_and_data_Kaon_{hist_name}.pdf"),
        int_lumi = int_lumi,
        logy=logy
    )
    
    
    stack_neutron, comps_neutron = _stack_hists(
        neutron_pred_hists, "MC_neutron", sort_fn=_parse_range_key
    )
    _draw_mc_stack_with_pred(
        stack_neutron, comps_neutron, data_pred_hists["neutron"],
        particle="Neutron",
        out_pdf=os.path.join(outdir, f"Stack_MC_and_data_Neutron_{hist_name}.pdf"),
        int_lumi = int_lumi,
        logy=logy
    )
    
    
    
    

def plot_MC_pred_VS_data_pred_combined(MC_neutrino_pred_hists, data_pred_hists, muon_pred_hists, kaon_pred_hists,neutron_pred_hists, hist_name,int_lumi, logy):
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)

    if "muon" not in data_pred_hists or "kaon" not in data_pred_hists or "neutron" not in data_pred_hists:
        raise KeyError("data_pred_hists must contain keys 'muon', 'neutron' and 'kaon'.")
    
    
    # sum pred muon, kaon, neutron seperately and the stack them toghter
    # stack  muon, kaon, neutron data_pred_hists
    # use _draw_mc_stack_with_pred to plot
    
    # ---- MC total per flavour ----
    h_mu_tot = sum_hists(muon_pred_hists,    "muon_MC_total")
    h_ka_tot = sum_hists(kaon_pred_hists,    "kaon_MC_total")
    h_ne_tot = sum_hists(neutron_pred_hists, "neutron_MC_total")

    # ---- Data predicted total per flavour ----
    h_mu_dt = data_pred_hists["muon"]
    h_ka_dt = data_pred_hists["kaon"]
    h_ne_dt = data_pred_hists["neutron"]

    # ---- build stack for the combined MC ----
    combined_mc_dict = {
        "kaon":    h_ka_tot,
        "neutron": h_ne_tot,
        "muon":    h_mu_tot,
    }

    # Each entry becomes one layer of the THStack
    stack_tot = ROOT.THStack("MC_tot_stack", "")
    comps_tot = []


    colors = {
        "muon": ROOT.kRed+1,
        "kaon": ROOT.kBlue,
        "neutron": ROOT.kViolet+2
    }

    for k, h in combined_mc_dict.items():
        hh = h.Clone(f"{k}_stack_clone"); hh.SetDirectory(0)
        hh.SetFillColor(colors[k])
        hh.SetLineColor(colors[k])
        stack_tot.Add(hh)
        comps_tot.append((k, hh))

    # ---- total data predicted (sum) ----
    h_data_total = h_mu_dt.Clone("data_total")
    h_data_total.Add(h_ka_dt)
    h_data_total.Add(h_ne_dt)
    h_data_total.SetDirectory(0)

    # ---- plot stack_tot vs data_total ----
    _draw_mc_stack_with_data(
        MC_neutrino_pred_hists,
        stack_tot, comps_tot, h_data_total,
        particle="(Muon/Kaon/Neutron)",
        out_pdf=os.path.join(outdir, f"AfterGNN_MC_vs_pred_Total_{hist_name}.pdf"),
        int_lumi=int_lumi,
        logy=logy
    )

METADATA_dict = read_metadata()
vetoTagged = False
model_name = 'GravNet_v4' #'baseline_muon'

#control_region_columns = ['count_scifi', 'sum_hit_density', 'centroid_slope_x', 'centroid_slope_y']
hist_info = {
    "vetoHitTime_earlist": (50, 0, 25, 'veto_hit_earlist_time', True),
    "vetoHitTime_lastest": (50, 0, 25, 'veto_hit_latest_time', True),
    
    "vetoHitTime_earlist_veto1": (50, 0, 25, 'vetoHitTime_earlist_veto1', True),
    "vetoHitTime_earlist_veto2": (50, 0, 25, 'vetoHitTime_earlist_veto2', True),
    "vetoHitTime_earlist_veto3": (50, 0, 25, 'vetoHitTime_earlist_veto3', True),
    "vetoHitTime_lastest_veto1": (50, 0, 25, 'veto_hit_latest_time_veto1', True),
    "vetoHitTime_lastest_veto2": (50, 0, 25, 'veto_hit_latest_time_veto2', True),
    "vetoHitTime_lastest_veto3": (50, 0, 25, 'veto_hit_latest_time_veto3', True),
    
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--hist_name", dest="hist_name", help="hist name", default="count_scifi")
    parser.add_argument("-c", "--cut", dest="cut", help="apply cut", default="nocut")
    args = parser.parse_args()
    
    print(f"processing hist of {args.hist_name}")
    
    print(f"applying cut: {args.cut}")
    
    process_hist(args)
    
    # plot options
    # control region (scifi hits, density, shower direction)
    
    #read metadata
    


