import ROOT
import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import math

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


def read_metadata(directory="./processed_metadata"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict

def read_rdf(metadata_df, MC_muon=False):
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
        rdf.Define("start_centroid_x",
            "showerStartStation == 1 ? centroid_scifi1_x : "
            "showerStartStation == 2 ? centroid_scifi2_x : "
            "showerStartStation == 3 ? centroid_scifi3_x : "
            "showerStartStation == 4 ? centroid_scifi4_x : "
            "centroid_scifi5_x")
        .Define("start_centroid_y",
            "showerStartStation == 1 ? centroid_scifi1_y : "
            "showerStartStation == 2 ? centroid_scifi2_y : "
            "showerStartStation == 3 ? centroid_scifi3_y : "
            "showerStartStation == 4 ? centroid_scifi4_y : "
            "centroid_scifi5_y")
    )
    
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
    
    # apply filter
    

        
    return rdf, feature_chain, int_lumi    
    
def process_table():
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    #reading real data
    data_rdf, data_chain, data_int_lumi = read_rdf(real_data)
    print(f'data_int_lumi:{data_int_lumi}')
    pred_classes = [ "kaon", "neutron", "muon"]
    

    for cls in pred_classes:
        class_id = particle_2_class[cls]
        rdf_pred = data_rdf.Filter(f"pred_class_first == {class_id}")
        h_proxy_pred = rdf_pred.Histo1D(
            (f"h_{cls}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name
        )
        
        h_pred = h_proxy_pred.GetValue()
        data_hist_proxies.append(h_proxy_pred)
        h_pred.SetDirectory(0)
        h_pred.GetXaxis().SetTitle(axis_title)
        h_pred.GetYaxis().SetTitle("Events")
        data_pred_hists[cls] = h_pred
        
    ##reading MC
    normalise_lumi = data_int_lumi
    
    ## reading neutrino
    neutrino_rdf, neutrino_chain, neutrino_int_lumi = read_rdf(neutrino_df[:10])
    
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
        rdf, _, int_lumi = read_rdf(sub_df[:10], MC_muon=True)
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

        # scale by veto Ineff = 1e-7 factor for muon
        if normalise_lumi:
            scale = float(normalise_lumi) / float(int_lumi) * 1e-7
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
        rdf, _, int_lumi = read_rdf(sub_df[:100])
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
        rdf, _, int_lumi = read_rdf(sub_df[:100])
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
        




    
    

METADATA_dict = read_metadata()
vetoTagged = False
model_name = 'baseline_muon'

if __name__ == "__main__":
    process_table()



