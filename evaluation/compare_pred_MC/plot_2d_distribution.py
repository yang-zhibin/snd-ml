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


def read_metadata(directory="/afs/cern.ch/work/z/zhibin/snd-ml/evaluation/compare_pred_MC/processed_metadata"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict

def read_rdf(args, metadata_df, MC_muon=False):
    feature_chain = ROOT.TChain("sndData")
    eval_chain = ROOT.TChain("snddata")
    
    int_lumi = 0
    # Defaultdict of dicts
    events_per_subfolder = defaultdict(lambda: {"vetoFree": 0, "vetoTagged": 0})

    for _, row in metadata_df.iterrows():
        # Always check vetoFree files first
        vetoFree_feature_path = row['vetoFree_feature_path']
        vetoFree_eval_path = row[f'vetoFree_eval_{model_name}_output_path']
        
        # print("vetoFree_feature_path",vetoFree_feature_path)
        # print("vetoFree_eval_path",vetoFree_eval_path)

        if os.path.exists(vetoFree_feature_path) and os.path.exists(vetoFree_eval_path):
            feature_chain.Add(vetoFree_feature_path)
            eval_chain.Add(vetoFree_eval_path)
            events_per_subfolder[row['subfolder']]["vetoFree"] += row['n_event']

            int_lumi += 0 if math.isnan(row['lumi_per_file']) else row['lumi_per_file']

        # Optionally also add vetoTagged files
        if vetoTagged or MC_muon:
            vetoTagged_feature_path = row['vetoTagged_feature_path']
            vetoTagged_eval_path = row[f'vetoTagged_eval_{model_name}_output_path']

            if os.path.exists(vetoTagged_feature_path) and os.path.exists(vetoTagged_eval_path):
                feature_chain.Add(vetoTagged_feature_path)
                eval_chain.Add(vetoTagged_eval_path)
                events_per_subfolder[row['subfolder']]["vetoTagged"] += row['n_event']
    
    if (int_lumi==0):
        return None, None, 0
    # Print total events per subfolder
    print("\nEvents read per subfolder:")
    for subfolder, counts in events_per_subfolder.items():
        print(f"  {subfolder}: vetoFree={counts['vetoFree']}, vetoTagged={counts['vetoTagged']}")

    feature_chain.AddFriend(eval_chain, 'eval')
    rdf = ROOT.RDataFrame(feature_chain)
    rdf= rdf.Define("sum_hit_density", "density_scifi1 + density_scifi2 + density_scifi3 + density_scifi4 + density_scifi5")
    rdf = (
        rdf.Define("start_avgPos_x",
            "showerStartStation == 1 ? avg_scifi1_x : "
            "showerStartStation == 2 ? avg_scifi2_x : "
            "showerStartStation == 3 ? avg_scifi3_x : "
            "showerStartStation == 4 ? avg_scifi4_x : "
            "avg_scifi5_x")
        .Define("start_avgPos_y",
            "showerStartStation == 1 ? avg_scifi1_y : "
            "showerStartStation == 2 ? avg_scifi2_y : "
            "showerStartStation == 3 ? avg_scifi3_y : "
            "showerStartStation == 4 ? avg_scifi4_y : "
            "avg_scifi5_y")
    )
    
    
    mid_x_val = (-6.9 + -45.9) / 2.0   # -26.4
    mid_y_val = (18.8 + 57.8) / 2.0    # 38.3

    expr_start_centroid_x = r"""
    showerStartStation == 1 ? centroid_scifi1_x :
    showerStartStation == 2 ? centroid_scifi2_x :
    showerStartStation == 3 ? centroid_scifi3_x :
    showerStartStation == 4 ? centroid_scifi4_x :
    centroid_scifi5_x
    """

    expr_start_centroid_y = r"""
    showerStartStation == 1 ? centroid_scifi1_y :
    showerStartStation == 2 ? centroid_scifi2_y :
    showerStartStation == 3 ? centroid_scifi3_y :
    showerStartStation == 4 ? centroid_scifi4_y :
    centroid_scifi5_y
    """

    expr_start_z = r"""
    showerStartStation == 1 ? 300 :
    showerStartStation == 2 ? 313 :
    showerStartStation == 3 ? 326 :
    showerStartStation == 4 ? 339 :
    352
    """

    
    # Compute where the line through (start_centroid_x, start_z) 
    # with slope centroid_slope_x intersects the plane x = mid_x
    # Return signed slope depending on whether intercept_z is before or after start_z
    expr_signed_slope_x = f"""
    const double mid_x = {mid_x_val};
    double intercept_z = fabs(centroid_slope_x) > 1e-12
        ? (mid_x - start_centroid_x) / centroid_slope_x + start_z
        : start_z;
    return fabs(centroid_slope_x) < 1e-12 ? 0.0
        : (intercept_z < start_z ? fabs(centroid_slope_x) : -fabs(centroid_slope_x));
    
    """

    expr_signed_slope_y = f"""
    const double mid_y = {mid_y_val};
    double intercept_z = fabs(centroid_slope_y) > 1e-12
        ? (mid_y - start_centroid_y) / centroid_slope_y + start_z
        : start_z;
    return fabs(centroid_slope_y) < 1e-12 ? 0.0
        : (intercept_z < start_z ? fabs(centroid_slope_y) : -fabs(centroid_slope_y));
    """

    # Use them in RDataFrame
    rdf = (
        rdf.Define("start_centroid_x", expr_start_centroid_x)
        .Define("start_centroid_y", expr_start_centroid_y)
        .Define("start_z",          expr_start_z)
        .Define("signed_slope_x",   expr_signed_slope_x)
        .Define("signed_slope_y",   expr_signed_slope_y)
    )
    
        
    rdf = (
        rdf
        # --- SciFi centroids (5 planes) ---
        .Define("centroid_scifi_x",
                "(centroid_scifi1_x + centroid_scifi2_x + centroid_scifi3_x + centroid_scifi4_x + centroid_scifi5_x)/5.0")
        .Define("centroid_scifi_y",
                "(centroid_scifi1_y + centroid_scifi2_y + centroid_scifi3_y + centroid_scifi4_y + centroid_scifi5_y)/5.0")

        # --- DS centroids (4 planes) ---
        .Define("centroid_ds_x",
                "(centroid_ds1_x + centroid_ds2_x + centroid_ds3_x + centroid_ds4_x)/4.0")
        .Define("centroid_ds_y",
                "(centroid_ds1_y + centroid_ds2_y + centroid_ds3_y + centroid_ds4_y)/4.0")

        # --- SciFi averages (5 planes) ---
        .Define("avg_scifi_x",
                "(avg_scifi1_x + avg_scifi2_x + avg_scifi3_x + avg_scifi4_x + avg_scifi5_x)/5.0")
        .Define("avg_scifi_y",
                "(avg_scifi1_y + avg_scifi2_y + avg_scifi3_y + avg_scifi4_y + avg_scifi5_y)/5.0")

        # --- DS averages (4 planes) ---
        .Define("avg_ds_x",
                "(avg_ds1_x + avg_ds2_x + avg_ds3_x + avg_ds4_x)/4.0")
        .Define("avg_ds_y",
                "(avg_ds1_y + avg_ds2_y + avg_ds3_y + avg_ds4_y)/4.0")
        )

    
    
    if (args.cut):
        rdf = rdf.Filter("count_scifi > 200")

    return rdf, feature_chain, int_lumi

    
def process_hist(args):
    hist_name = args.hist_name
    nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logz, bin_width_x, bin_width_y = hist_info[hist_name]
    
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    #reading real data
    data_rdf, data_chain, data_int_lumi = read_rdf(args, real_data)
    print(f'data_int_lumi:{data_int_lumi}')
    pred_classes = [ "kaon", "neutron", "muon"]
    
    data_pred_hists = {}
    data_hist_proxies = []
    for cls in pred_classes:
        class_id = particle_2_class[cls]
        rdf_pred = data_rdf.Filter(f"pred_class_first == {class_id}")
        h_proxy_pred = rdf_pred.Histo2D(
            (f"h_{cls}_{hist_name}","", 
             int(nbins_x), float(x_min), float(x_max), 
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        
        h_pred = h_proxy_pred.GetValue()
        data_hist_proxies.append(h_proxy_pred)
        h_pred.SetDirectory(0)
        # 2D: set X/Y to variable titles (reuse axis_title) and Z to counts
        h_pred.GetXaxis().SetTitle(f"{axis_title} X")
        h_pred.GetYaxis().SetTitle(f"{axis_title} Y")
        h_pred.GetZaxis().SetTitle("Events")
        data_pred_hists[cls] = h_pred
        
    ##reading MC
    normalise_lumi = data_int_lumi
    
    ## reading neutrino
    neutrino_rdf, neutrino_chain, neutrino_int_lumi = read_rdf(args, neutrino_df)
    
    scale_factor = normalise_lumi/ neutrino_int_lumi  if neutrino_int_lumi else 1.0
    
    neutrino_classes = ["ve", "vm", "vt", "NC"]
    MC_neutrino_true_hists = {}
    MC_neutrino_pred_hists = {}
    neutrino_hist_proxies = []
    for cls in neutrino_classes:
        class_id = particle_2_class[cls]
        rdf_true = neutrino_rdf.Filter(f"ParticleClass == {class_id}")
        rdf_pred = neutrino_rdf.Filter(f"pred_class_first == {class_id}")
        h_proxy_true = rdf_true.Histo2D(
            (f"h_{cls}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        h_proxy_pred = rdf_pred.Histo2D(
            (f"h_{cls}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        
        h_ture = h_proxy_true.GetValue()
        neutrino_hist_proxies.append(h_proxy_true)
        h_ture.Scale(scale_factor)  
        h_ture.SetDirectory(0)
        h_ture.GetXaxis().SetTitle(f"{axis_title} X")
        h_ture.GetYaxis().SetTitle(f"{axis_title} Y")
        h_ture.GetZaxis().SetTitle("Expected Events")
        MC_neutrino_true_hists[cls] = h_ture
        
        h_pred = h_proxy_pred.GetValue()
        neutrino_hist_proxies.append(h_proxy_pred)
        h_pred.Scale(scale_factor)  # apply lumi scaling
        h_pred.SetDirectory(0)
        h_pred.GetXaxis().SetTitle(f"{axis_title} X")
        h_pred.GetYaxis().SetTitle(f"{axis_title} Y")
        h_pred.GetZaxis().SetTitle("Expected Events")
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
        h_proxy_true = rdf_true.Histo2D(
            (f"h_muon_true_{beam_type}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        h_proxy_pred = rdf_pred.Histo2D(
            (f"h_muon_pred_{beam_type}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        muon_hist_proxies.extend([h_proxy_true, h_proxy_pred])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(f"{axis_title} X")
            h.GetYaxis().SetTitle(f"{axis_title} Y")
            h.GetZaxis().SetTitle("Expected Events")

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
        rdf, _, int_lumi = read_rdf(args, sub_df)
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {kaon_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {kaon_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo2D(
            (f"h_kaon_true_{erange}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        h_proxy_pred = rdf_pred.Histo2D(
            (f"h_kaon_pred_{erange}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        kaon_hist_proxies.extend([h_proxy_true, h_proxy_pred])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(f"{axis_title} X")
            h.GetYaxis().SetTitle(f"{axis_title} Y")
            h.GetZaxis().SetTitle("Expected Events")

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
        
        rdf, _, int_lumi = read_rdf(args, sub_df)
        if not int_lumi:
            continue

        # filters for true/pred
        rdf_true = rdf.Filter(f"ParticleClass == {neutron_class_id}")
        rdf_pred = rdf.Filter(f"pred_class_first == {neutron_class_id}")

        # histogram proxies (keep them alive!)
        h_proxy_true = rdf_true.Histo2D(
            (f"h_neutron_true_{erange}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        h_proxy_pred = rdf_pred.Histo2D(
            (f"h_neutron_pred_{erange}_{hist_name}", "", 
             int(nbins_x), float(x_min), float(x_max),
             int(nbins_y), float(y_min), float(y_max)),
            f"{hist_name}_x", f"{hist_name}_y"
        )
        
        neutron_hist_proxies.extend([h_proxy_true, h_proxy_pred])

        # materialize
        h_true = h_proxy_true.GetValue().Clone()
        h_pred = h_proxy_pred.GetValue().Clone()
        for h in (h_true, h_pred):
            h.SetDirectory(0)
            h.GetXaxis().SetTitle(f"{axis_title} X")
            h.GetYaxis().SetTitle(f"{axis_title} Y")
            h.GetZaxis().SetTitle("Expected Events")

        # scale by lumi (no extra 1e-8 here unless you need it for consistency)
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi)
            h_true.Scale(scale)
            h_pred.Scale(scale)
        
        neutron_true_hists[erange] = h_true
        neutron_pred_hists[erange] = h_pred

    plot_2d_hist(data_pred_hists,
                MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                muon_true_hists, muon_pred_hists, 
                kaon_true_hists, kaon_pred_hists, 
                neutron_true_hists, neutron_pred_hists, 
                hist_name, data_int_lumi)

    

def _sum_th2_dict(hdict, name):
    """Sum all TH2 in a dict into a single TH2 clone."""
    if not hdict:
        raise ValueError(f"{name}: empty histogram dict")
    it = iter(hdict.values())
    hsum = next(it).Clone(f"{name}_sum")
    hsum.SetDirectory(0)
    for h in it:
        hsum.Add(h)
    return hsum

def plot_2d(hist2d, 
            hist_name, 
            output_path,
            top_left_label="", 
            top_right_label=""):
    
    ROOT.gStyle.SetOptStat("emr")
    
    nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logz, bin_width_x, bin_width_y = hist_info[hist_name]
    nbins_x = int((x_max - x_min) / bin_width_x)
    nbins_y = int((y_max - y_min) / bin_width_y)
    
    
    # Canvas setup
    canvas = ROOT.TCanvas(hist_name, "", 800, 600)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.12)
    canvas.SetBottomMargin(0.15)

        
    hist2d.SetTitle("")  # Disable default title
    hist2d.Draw("COLZ")
    
    canvas.Update()
    stat = hist2d.GetListOfFunctions().FindObject("stats")
    if stat:
        stat.SetX1NDC(0.75)  # Left edge of box
        stat.SetX2NDC(0.85)  # Right edge (avoid overlap with color bar)
        stat.SetY1NDC(0.77)  # Bottom edge
        stat.SetY2NDC(0.87) 
        stat.SetName("")  # remove title from stat box 
        canvas.Update()
        
        
    if logz:
        canvas.SetLogz()

    # Draw top-left label
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.05)
    label.DrawLatex(0.12, 0.92, top_left_label)
    
     # Draw top right title
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)  # smaller than 0.045
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {top_right_label:.2f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")

    # # Draw bottom title
    # bottom = ROOT.TLatex()
    # bottom.SetNDC()
    # bottom.SetTextAlign(21)  # Centered
    # bottom.SetTextFont(42)
    # bottom.SetTextSize(0.05)
    # bottom.DrawLatex(0.5, 0.04, axis_title)
    
    canvas.SaveAs(output_path)
    print(f"[OK] wrote {output_path}")
    canvas.Close()

def plot_2d_hist(data_pred_hists,
                MC_neutrino_true_hists, MC_neutrino_pred_hists, 
                muon_true_hists, muon_pred_hists, 
                kaon_true_hists, kaon_pred_hists, 
                neutron_true_hists, neutron_pred_hists, 
                hist_name, data_int_lumi):
    
    outdir = f"./2d_plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)

    # Sum to totals
    neutrino_true  = _sum_th2_dict(MC_neutrino_true_hists, "nutrino_true")
    ve_true = MC_neutrino_true_hists['ve']
    vm_true = MC_neutrino_true_hists['vm']
    vt_true = MC_neutrino_true_hists['vt']
    NC_true = MC_neutrino_true_hists['NC']
    
    
    
    neutrino_pred  = _sum_th2_dict(MC_neutrino_pred_hists, "nutrino_pred")
    ve_pred = MC_neutrino_pred_hists['ve']
    vm_pred = MC_neutrino_pred_hists['vm']
    vt_pred = MC_neutrino_pred_hists['vt']
    NC_pred = MC_neutrino_pred_hists['NC']
    
    muon_true_sum     = _sum_th2_dict(muon_true_hists,    "muon_true")
    muon_pred_sum     = _sum_th2_dict(muon_pred_hists,    "muon_pred")
    kaon_true_sum     = _sum_th2_dict(kaon_true_hists,    "kaon_true")
    kaon_pred_sum     = _sum_th2_dict(kaon_pred_hists,    "kaon_pred")
    neutron_true_sum  = _sum_th2_dict(neutron_true_hists, "neutron_true")
    neutron_pred_sum  = _sum_th2_dict(neutron_pred_hists, "neutron_pred")
    
    data_muon = data_pred_hists["muon"]
    data_kaon = data_pred_hists["kaon"]
    data_neutron = data_pred_hists["neutron"]
    
    
    created = []
    def _p(hist2d, filename, top_left_label):
        path = os.path.join(outdir, filename)
        plot_2d(
            hist2d=hist2d,
            hist_name=hist_name,
            output_path=path,
            top_left_label=top_left_label,
            top_right_label=data_int_lumi
        )
        created.append(path)

    # --- MC neutrino totals + components ---
    _p(neutrino_true, f"{hist_name}_MC_neutrino_true_2d.pdf",       "Neutrino [MC, true]")
    _p(neutrino_pred, f"{hist_name}_MC_neutrino_pred_2d.pdf",       "Neutrino [MC, GNN select]")
    _p(ve_true,       f"{hist_name}_MC_ve_true_2d.pdf",             "Ve [MC, true]")
    _p(vm_true,       f"{hist_name}_MC_vm_true_2d.pdf",             "Vm [MC, true]")
    _p(vt_true,       f"{hist_name}_MC_vt_true_2d.pdf",             "Vt [MC, true]")
    _p(NC_true,       f"{hist_name}_MC_NC_true_2d.pdf",             "NC [MC, true]")
    _p(ve_pred,       f"{hist_name}_MC_ve_pred_2d.pdf",             "Ve [MC, GNN select]")
    _p(vm_pred,       f"{hist_name}_MC_vm_pred_2d.pdf",             "Vm [MC, GNN select]")
    _p(vt_pred,       f"{hist_name}_MC_vt_pred_2d.pdf",             "Vt [MC, GNN select]")
    _p(NC_pred,       f"{hist_name}_MC_NC_pred_2d.pdf",             "NC [MC, GNN select]")

    # --- MC particles (muon/kaon/neutron) totals ---
    _p(muon_true_sum,    f"{hist_name}_MC_muon_true_2d.pdf",        "Muon [MC, true]")
    _p(muon_pred_sum,    f"{hist_name}_MC_muon_pred_2d.pdf",        "Muon [MC, GNN select]")
    _p(kaon_true_sum,    f"{hist_name}_MC_kaon_true_2d.pdf",        "Kaon [MC, true]")
    _p(kaon_pred_sum,    f"{hist_name}_MC_kaon_pred_2d.pdf",        "Kaon [MC, GNN select]")
    _p(neutron_true_sum, f"{hist_name}_MC_neutron_true_2d.pdf",     "Neutron [MC, true]")
    _p(neutron_pred_sum, f"{hist_name}_MC_neutron_pred_2d.pdf",     "Neutron [MC, GNN select]")

    # --- Data categories (predicted) ---
    _p(data_muon,    f"{hist_name}_Data_muon_2d.pdf",               "Muon-like [Data]")
    _p(data_kaon,    f"{hist_name}_Data_kaon_2d.pdf",               "Kaon-like [Data]")
    _p(data_neutron, f"{hist_name}_Data_neutron_2d.pdf",            "Neutron-like [Data]")

    return 0

METADATA_dict = read_metadata()
vetoTagged = False
model_name = 'baseline_muon'


#nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logz, bin_width_x, bin_width_y
hist_info = {
    "start_avgPos": (90, -70, 20, 80, 0, 80, 'Start AvgPos', False, 0.5, 0.5),
    "avg_scifi": (90, -70, 20, 80, 0, 80, 'SciFi AvgPos', False, 0.5, 0.5),
    "avg_scifi1": (90, -70, 20, 80, 0, 80, 'SciFi 1 AvgPos', False, 0.5, 0.5),
    "avg_scifi2": (90, -70, 20, 80, 0, 80, 'SciFi 2 AvgPos', False, 0.5, 0.5),
    "avg_scifi3": (90, -70, 20, 80, 0, 80, 'SciFi 3 AvgPos', False, 0.5, 0.5),
    "avg_scifi4": (90, -70, 20, 80, 0, 80, 'SciFi 4 AvgPos', False, 0.5, 0.5),
    "avg_scifi5": (90, -70, 20, 80, 0, 80, 'SciFi 5 AvgPos', False, 0.5, 0.5),
    
    "avg_ds": (90, -70, 20, 80, 0, 80, 'DS  AvgPos', False, 1.5, 1.5),
    "avg_ds1": (90, -70, 20, 80, 0, 80, 'DS 1 AvgPos', False, 1.5, 1.5),
    "avg_ds2": (90, -70, 20, 80, 0, 80, 'DS 2 AvgPos', False, 1.5, 1.5),
    "avg_ds3": (90, -70, 20, 80, 0, 80, 'DS 3 AvgPos', False, 1.5, 1.5),
    
    "start_centroid": (90, -70, 20, 80, 0, 80, 'Start Centroid', False, 0.5, 0.5),
    "centroid_scifi": (90, -70, 20, 80, 0, 80, 'SciFi Centroid', False, 0.5, 0.5),
    "centroid_scifi1": (90, -70, 20, 80, 0, 80, 'SciFi 1 Centroid', False, 0.5, 0.5),
    "centroid_scifi2": (90, -70, 20, 80, 0, 80, 'SciFi 2 Centroid', False, 0.5, 0.5),
    "centroid_scifi3": (90, -70, 20, 80, 0, 80, 'SciFi 3 Centroid', False, 0.5, 0.5),
    "centroid_scifi4": (90, -70, 20, 80, 0, 80, 'SciFi 4 Centroid', False, 0.5, 0.5),
    "centroid_scifi5": (90, -70, 20, 80, 0, 80, 'SciFi 5 Centroid', False, 0.5, 0.5),
    
    "centroid_ds": (90, -70, 20, 80, 0, 80, 'DS Centroid', False, 1.5, 1.5),
    "centroid_ds1": (90, -70, 20, 80, 0, 80, 'DS 1 Centroid', False, 1.5, 1.5),
    "centroid_ds2": (90, -70, 20, 80, 0, 80, 'DS 2 Centroid', False, 1.5, 1.5),
    "centroid_ds3": (90, -70, 20, 80, 0, 80, 'DS 3 Centroid', False, 1.5, 1.5),
    
    "centroid_slope": (100, -5, 5,100, -5, 5, 'DS 3 Centroid', False, 0.01, 0.01),
    "avgPos_slope": (100, -5, 5,100, -5, 5, 'DS 3 Centroid', False, 0.01, 0.01),
    "signed_slope": (100, -5, 5,100, -5, 5, 'DS 3 Centroid', False, 0.01, 0.01),
    
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--hist_name", dest="hist_name", help="hist name", default="count_scifi")
    parser.add_argument("-c", "--cut", action="store_true", help="apply cut")
    args = parser.parse_args()
    
    print(f"processing hist of {args.hist_name}")
    if args.cut:
        print("→ applying cut")
    
    process_hist(args)
    
    # plot options
    # control region (scifi hits, density, shower direction)
    
    #read metadata
    


