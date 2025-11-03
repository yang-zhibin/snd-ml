import ROOT
import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import math
import argparse

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


def read_metadata(directory="/afs/cern.ch/work/z/zhibin/snd-ml/evaluation/compare_pred_MC/processed_metadata_baseline_muon"):
    
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
    feature_chain = ROOT.TChain("sndData")
    prediction_chain = ROOT.TChain("sndData")
    
    int_lumi = 0
    # Defaultdict of dicts
    events_per_subfolder = defaultdict(lambda: {"vetoFree": 0, "vetoTagged": 0})

    
    for _, row in metadata_df.iterrows():
        sub = row['subfolder']

        # ---- vetoFree ----
        vf_feat = row['vetoFree_feature_path']
        vf_pred = row[f'vetoFree_prediction_{model_name}_output_path']

        n_feat, _ = tree_entries_and_branch(vf_feat, "sndData")
        n_pred, has_branch = tree_entries_and_branch(vf_pred, "sndData", required_branch="pred_class_first")

        if n_feat > 0 and n_pred > 0 and has_branch and n_feat == n_pred:
            feature_chain.Add(vf_feat)
            prediction_chain.Add(vf_pred)
            events_per_subfolder[sub]["vetoFree"] += row['n_event']
            if pd.notna(row['lumi_per_file']):
                int_lumi += row['lumi_per_file']
        else:
            if n_feat == 0:
                print(f"[Skip] features empty/missing: {vf_feat}")
            if n_pred == 0:
                print(f"[Skip] predictions empty/missing: {vf_pred}")
            if n_feat != 0 and n_pred != 0 and n_feat != n_pred:
                print(f"[Skip] entry mismatch (features={n_feat}, predictions={n_pred}):\n  {vf_feat}\n  {vf_pred}")
            if n_pred > 0 and not has_branch:
                print(f"[Skip] predictions missing branch 'pred_class_first': {vf_pred}")

        # ---- vetoTagged (optional) ----
        if vetoTagged or MC_muon:
            vt_feat = row.get('vetoTagged_feature_path')
            vt_pred = row.get(f'vetoTagged_prediction_{model_name}_output_path')

            n_feat, _ = tree_entries_and_branch(vt_feat, "sndData") if vt_feat else (0, False)
            n_pred, has_branch = tree_entries_and_branch(vt_pred, "sndData", required_branch="pred_class_first") if vt_pred else (0, False)

            if n_feat > 0 and n_pred > 0 and has_branch and n_feat == n_pred:
                feature_chain.Add(vt_feat)
                prediction_chain.Add(vt_pred)
                events_per_subfolder[sub]["vetoTagged"] += row['n_event']
            else:
                if vt_feat and n_feat == 0:
                    print(f"[Skip] features empty/missing: {vt_feat}")
                if vt_pred and n_pred == 0:
                    print(f"[Skip] predictions empty/missing: {vt_pred}")
                if vt_feat and vt_pred and n_feat != 0 and n_pred != 0 and n_feat != n_pred:
                    print(f"[Skip] entry mismatch (features={n_feat}, predictions={n_pred}):\n  {vt_feat}\n  {vt_pred}")
                if vt_pred and n_pred > 0 and not has_branch:
                    print(f"[Skip] predictions missing branch 'pred_class_first': {vt_pred}")

    print(f"Added {feature_chain.GetNtrees()} feature files and {prediction_chain.GetNtrees()} prediction files.")
    if (int_lumi==0):
        return None, None, 0
    # Print total events per subfolder
    print("\nEvents read per subfolder:")
    for subfolder, counts in events_per_subfolder.items():
        print(f"  {subfolder}: vetoFree={counts['vetoFree']}, vetoTagged={counts['vetoTagged']}")

    feature_chain.AddFriend(prediction_chain, 'prediction')
    rdf = ROOT.RDataFrame(feature_chain)
    
    rdf= rdf.Define("sum_hit_density", "density_scifi1 + density_scifi2 + density_scifi3 + density_scifi4 + density_scifi5")
    
    #if (args.cut):
    #    rdf = rdf.Filter("count_scifi > 200")
        
    return rdf, feature_chain, int_lumi

    
    
def process_hist(args):
    hist_name=args.hist_name
    n_bins, x_min, x_max, axis_title, logy = hist_info[hist_name]
    
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    #reading real data
    # data_rdf, data_chain, data_int_lumi = read_rdf(args,real_data[0:20])
    # print(f'data_int_lumi:{data_int_lumi}')
    # pred_classes = [ "kaon", "neutron", "muon"]
    
    # data_pred_hists = {}
    # data_hist_proxies = []
    # for cls in pred_classes:
    #     class_id = particle_2_class[cls]
    #     rdf_pred = data_rdf.Filter(f"pred_class_first == {class_id}")
    #     h_proxy_pred = rdf_pred.Histo1D(
    #         (f"h_{cls}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
    #         hist_name
    #     )
        
    #     h_pred = h_proxy_pred.GetValue()
    #     data_hist_proxies.append(h_proxy_pred)
    #     h_pred.SetDirectory(0)
    #     h_pred.GetXaxis().SetTitle(axis_title)
    #     h_pred.GetYaxis().SetTitle("Events")
    #     data_pred_hists[cls] = h_pred
        
    ##reading MC
    normalise_lumi = 1 #data_int_lumi
    
    ## reading neutrino
    neutrino_rdf, neutrino_chain, neutrino_int_lumi = read_rdf(args, neutrino_df[:10])
    
    scale_factor = normalise_lumi/ neutrino_int_lumi  if neutrino_int_lumi else 1.0
    
    neutrino_classes = ["ve", "vm", "vt", "NC"]
    MC_neutrino_true_hists = {}
    MC_neutrino_pred_hists = {}
    neutrino_hist_proxies = []
    for cls in neutrino_classes:
        class_id = particle_2_class[cls]
        rdf_true = neutrino_rdf.Filter(f"ParticleClass == {class_id}")
        rdf_pred = neutrino_rdf.Filter(f"pred_class_first == {class_id}")
        h_proxy_true = rdf_true.Histo1D(
            (f"h_{cls}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
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
        
        h_pred = h_proxy_pred.GetValue()
        neutrino_hist_proxies.append(h_proxy_pred)
        h_pred.Scale(scale_factor)  # apply lumi scaling
        h_pred.SetDirectory(0)
        h_pred.GetXaxis().SetTitle(axis_title)
        h_pred.GetYaxis().SetTitle("Expected Events")
        MC_neutrino_pred_hists[cls] = h_pred
        
    # --- reading muon ---
    beam_types = sorted(muon_df['subfolder'].unique(), key=lambda x: (str(type(x)), x))

    muon_true_hists = {}
    muon_pred_hists = {}
    muon_hist_proxies = []

    mu_class_id = particle_2_class['muon']

    for beam_type in beam_types:
        sub_df = muon_df[muon_df['subfolder'] == beam_type]

        # build RDF and lumi for this slice
        rdf, _, int_lumi = read_rdf(args, sub_df, MC_muon=True)
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {mu_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {mu_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_muon_true_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_muon_pred_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        muon_hist_proxies.extend([h_proxy_true, h_proxy_pred])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by veto Ineff = 1e-8 factor for muon
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi) * 1e-8
            h_true.Scale(scale)
            h_pred.Scale(scale)
        
        muon_true_hists[beam_type] = h_true
        muon_pred_hists[beam_type] = h_pred
        
    # --- reading kaon ---
    if 'energy_range' not in kaon_df.columns:
        raise KeyError("MC_kaon metadata requires an 'energy_range' column")

    ranges = sorted(kaon_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))

    kaon_true_hists = {}
    kaon_pred_hists = {}
    kaon_hist_proxies = []

    kaon_class_id = particle_2_class['kaon']

    for erange in ranges:
        sub_df = kaon_df[kaon_df['energy_range'] == erange]

        # build RDF and lumi for this slice
        rdf, _, int_lumi = read_rdf(args, sub_df[:15])
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {kaon_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {kaon_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_kaon_true_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_kaon_pred_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        kaon_hist_proxies.extend([h_proxy_true, h_proxy_pred])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by lumi (no extra 1e-8 here unless you need it for consistency)
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi)
            h_true.Scale(scale)
            h_pred.Scale(scale)

        kaon_true_hists[erange] = h_true
        kaon_pred_hists[erange] = h_pred
        
    # --- reading neutron ---
    if 'energy_range' not in neutron_df.columns:
        raise KeyError("MC_neutron metadata requires an 'energy_range' column")

    ranges = sorted(kaon_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))

    neutron_true_hists = {}
    neutron_pred_hists = {}
    neutron_hist_proxies = []

    neutron_class_id = particle_2_class['neutron']

    for erange in ranges:
        sub_df = neutron_df[neutron_df['energy_range'] == erange]

        # build RDF and lumi for this slice
        
        rdf, _, int_lumi = read_rdf(args, sub_df[:15])
        if not int_lumi:
            continue


        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {neutron_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {neutron_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo1D(
            (f"h_neutron_true_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_neutron_pred_{erange}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        
        neutron_hist_proxies.extend([h_proxy_true, h_proxy_pred])


        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("Expected Events")

        # scale by lumi (no extra 1e-8 here unless you need it for consistency)
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi)
            h_true.Scale(scale)
            h_pred.Scale(scale)
        
        neutron_true_hists[erange] = h_true
        neutron_pred_hists[erange] = h_pred
        

        

    
    plot_MC_pred_VS_data_pred(data_pred_hists, muon_pred_hists, kaon_pred_hists, neutron_pred_hists, hist_name, data_int_lumi, logy)
    plot_data_pred_bkg(data_pred_hists, hist_name, data_int_lumi, logy)
    plot_MC(MC_neutrino_true_hists, muon_true_hists, kaon_true_hists,neutron_true_hists, hist_name, data_int_lumi, logy)
    plot_MC_VS_MC_pred(data_pred_hists,
                       MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                       muon_true_hists, muon_pred_hists, 
                       kaon_true_hists, kaon_pred_hists, 
                       neutron_true_hists, neutron_pred_hists, 
                       hist_name, data_int_lumi, logy)
    
    plot_2d_hist(data_pred_hists,
                       MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                       muon_true_hists, muon_pred_hists, 
                       kaon_true_hists, kaon_pred_hists, 
                       neutron_true_hists, neutron_pred_hists, 
                       hist_name, data_int_lumi)

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
    
    

    out_nu = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutrino_{hist_name}.pdf")
    out_mu = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_muon_{hist_name}.pdf")
    out_ka = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_kaon_{hist_name}.pdf")
    out_neutron = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutron_{hist_name}.pdf")
    out_ve = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutrino_ve_{hist_name}.pdf")
    out_vm = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutrino_vm_{hist_name}.pdf")
    out_vt = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutrino_vt_{hist_name}.pdf")
    out_NC = os.path.join(outdir, f"MC_bkg_vs_MC_pred_bkg_neutrino_NC_{hist_name}.pdf")
    
    
    
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
        out_pdf=os.path.join(outdir, f"MC_signal_vs_bkg_{hist_name}.pdf"),
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

    outpath = os.path.join(outdir, f"Pred_bkgs_{hist_name}.pdf")
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

def _draw_mc_stack_with_pred(stack, comps, h_pred, particle, out_pdf,int_lumi,
                             normalize_mc_to_pred=False, logy=False):
    
    title = f"MC vs Pred Data ({particle})"
    c = ROOT.TCanvas(f"c_{title}", title, 900, 750)
    if logy: c.SetLogy()

    # Clone pred to keep input pristine
    h_pred_d = h_pred.Clone(f"pred_{title}"); h_pred_d.SetDirectory(0)
    h_pred_d.SetMarkerStyle(20); h_pred_d.SetMarkerSize(1.0)
    h_pred_d.SetLineColor(ROOT.kBlack)

    # Optional normalization: scale total MC (and all stack components) to pred yield
    if normalize_mc_to_pred:
        i_pred = h_pred_d.Integral()
        i_mc   = hsum.Integral()
        if i_pred > 0 and i_mc > 0:
            scale = i_pred / i_mc
            # scale all stack hists consistently
            for _, h in comps:
                h.Scale(scale)
            hsum.Scale(scale)

    # Draw stack
    has_content = any(h.Integral() != 0 for h in stack.GetHists())
    if has_content:
        stack.Draw("HIST")

        stack.GetYaxis().SetTitle("Events")
        stack.GetXaxis().SetTitle(h_pred.GetXaxis().GetTitle())
        ymax = max(stack.GetMaximum(), h_pred_d.GetMaximum())
        ymin = min_positive_bin_content_stack(stack)
        
        print(f'in _draw_mc_stack_with_pred, (ymin, ymax): ({ymin}, {ymax})')
        
        stack.SetMaximum(ymax * (1.35 if not logy else 5.0))
        stack.SetMinimum(ymin * 0.5)

    # Overlay pred as points
    if h_pred_d.Integral() != 0:
        h_pred_d.Draw("E1 SAME")

    # Legend
    leg = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    if particle =='Muon':
        for label, h in comps:
            leg.AddEntry(h, f"MC {particle} ({label})", "f")
    else:
        for label, h in comps:
            leg.AddEntry(h, f"MC {particle} ({label.strip('()').replace(',', '-')}GeV)", "f")
    leg.AddEntry(h_pred_d, f"{particle}-like (data)", "lep")
    leg.Draw()

    # Title
    pave = ROOT.TPaveText(0.12, 0.92, 0.88, 0.99, "NDC")
    pave.SetFillStyle(0); pave.SetBorderSize(0); pave.AddText(""); pave.Draw()
    
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)  # smaller than 0.045
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.3f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")
    

    c.Print(out_pdf)
    print(f"Wrote {out_pdf}")
    
def _beam_order(k):
    order = {"up": 0, "down": 1, "horizontal": 2}
    return order.get(str(k), 99)
    
def plot_MC_pred_VS_data_pred(data_pred_hists, muon_pred_hists, kaon_pred_hists,neutron_pred_hists, hist_name,int_lumi, logy):
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
        out_pdf=os.path.join(outdir, f"MC_vs_pred_Muon_{hist_name}.pdf"),
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
        out_pdf=os.path.join(outdir, f"MC_vs_pred_Kaon_{hist_name}.pdf"),
        int_lumi = int_lumi,
        logy=logy
    )
    
    
    stack_neutron, comps_neutron = _stack_hists(
        neutron_pred_hists, "MC_neutron", sort_fn=_parse_range_key
    )
    _draw_mc_stack_with_pred(
        stack_neutron, comps_neutron, data_pred_hists["neutron"],
        particle="Neutron",
        out_pdf=os.path.join(outdir, f"MC_vs_pred_Neutron_{hist_name}.pdf"),
        int_lumi = int_lumi,
        logy=logy
    )
    
    

METADATA_dict = read_metadata()
vetoTagged = False
model_name = 'baseline_muon'

#control_region_columns = ['count_scifi', 'sum_hit_density', 'centroid_slope_x', 'centroid_slope_y']

hist_info = {
    "signed_slope_x": (60, -3, 3, 'Shower Direction X', True),
    "signed_slope_y": (60, -3, 3, 'Shower Direction Y', True),
    "sum_hit_density": (70, 0, 7e4, 'Sum of Density Weight',True),
    'count_scifi':  (100, 0, 1000, 'SciFi Hit Total Count', True),
    'count_us1':  (13, 0, 13, 'US1 Hit Count', True),
    "start_centroid_x": (90, -70, 20, 'Start Centroid X', True),
    "start_centroid_y": (80, 0, 80, 'Start Centroid Y', True),
    "centroid_slope_y": (60, -3, 3, 'Centroid Slope y', True),
    "centroid_slope_x": (60, -3, 3, 'Centroid Slope x', True),
    "start_avgPos_x": (90, -70, 20, 'Start AvgPos X', True),
    "start_avgPos_y": (80, 0, 80, 'Start AvgPos Y', True),
    "avgPos_slope_x": (60,-3, 3, 'Centroid Slope X', True),
    "avgPos_slope_y": (60,-3, 3, 'Centroid Slope Y', True),
    "avg_scifi1_y": (70, 0, 70, 'Scifi1 AvgPos Y', True),
    "avg_scifi2_y": (70, 0, 70, 'Scifi2 AvgPos Y', False),
    "avg_scifi3_y": (70, 0, 70, 'Scifi3 AvgPos Y', False),
    "avg_scifi4_y": (70, 0, 70, 'Scifi4 AvgPos Y', False),
    "avg_scifi5_y": (70, 0, 70, 'Scifi5 AvgPos Y', False),

}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--hist_name", dest="hist_name", help="hist name", default="avg_scifi1_y")
    parser.add_argument("-c", "--cut", action="store_true", help="apply cut")
    args = parser.parse_args()
    
    print(f"processing hist of {args.hist_name}")
    if args.cut:
        print("→ applying cut")
    
    process_hist(args)
    
    # plot options
    # control region (scifi hits, density, shower direction)
    
    #read metadata
    


