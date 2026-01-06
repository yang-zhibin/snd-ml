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
from pathlib import Path

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)




particle_2_class = {
    'e': 0,
    'pion': 1,
}
class_2_particle = {v: k for k, v in particle_2_class.items()}


def read_metadata(directory, split_name):
    """Load all processed metadata CSVs into a dictionary, excluding rows in split."""
    metadata_dict = {}
    
    # Construct the path to the split CSV file
    split_csv_path = f'{directory}/../training/{split_name}.csv'
    
    # Read the split CSV if it exists
    split_digi_paths = set()
    if os.path.exists(split_csv_path):
        split_df = pd.read_csv(split_csv_path)
        # Assuming the column name is 'digi_path' - adjust if different
        if 'digi_path' in split_df.columns:
            split_digi_paths = set(split_df['digi_path'])
        else:
            print(f"Warning: 'digi_path' column not found in {split_csv_path}")
    else:
        print(f"Warning: Split file not found at {split_csv_path}")
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(directory, file))
            
            # Filter the dataframe if split CSV exists and has digi_path column
            if split_digi_paths and 'digi_path' in df.columns:
                # Keep only rows where digi_path is NOT in the split CSV
                df = df[~df['digi_path'].isin(split_digi_paths)]
                print(f"Filtered {key}: kept {len(df)} rows (excluded {len(split_digi_paths)} split entries)")
            elif 'digi_path' not in df.columns:
                print(f"Warning: 'digi_path' column not found in {file}")
            
            metadata_dict[key] = df
            
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

def read_rdf(args, metadata_df, max_file=1e5):
    model_name = args.model_name
    feature_chain = ROOT.TChain("sndData")

    
    n_read_files = 0
    for _, row in metadata_df.iterrows():
        sub = row['subfolder']

        # ---- vetoFree ----
        feat = row['feature_path']

        n_feat, _ = tree_entries_and_branch(feat, "sndData")

        if n_feat > 0:
            feature_chain.Add(feat)
            n_read_files+=1
        else:
            if n_feat == 0:
                #print(f"[Skip] features empty/missing: {feat}")
                pass
        if n_read_files>=max_file:
            break
    print(f"Added {feature_chain.GetNtrees()} feature files ")
    rdf = ROOT.RDataFrame(feature_chain)
    if args.cut == 'nocut':
        # no selection
        pass
    else:
        raise ValueError(f'Unknown cut: {args.cut}')
    
    return rdf, feature_chain, n_read_files
    
def process_hist(args):
    model_name = args.model_name 
    hist_name = args.hist_name
    n_bins, x_min, x_max, axis_title, logy = hist_info[args.hist_name]
    METADATA_dict = read_metadata(args.metadata_dir, args.split)
    
    print(n_bins, x_min, x_max, axis_title, logy)
    
    #print(METADATA_dict)
    # get MC and realData metadata
    MC_df = METADATA_dict['MC_data_testbeam2024_metadata']
    Data_df = METADATA_dict['real_data_testbeam_24_metadata']
    
    # nested dicts: hists[energy][ptype] -> TH1
    data_full_hists = defaultdict(dict)
    MC_full_hists   = defaultdict(dict)

    # keep RDFs and RResultPtrs alive
    hist_proxies = []
    energies = sorted(set(MC_df["beam_energy"]) | set(Data_df["beam_energy"]) - {"no energy"})
    print(energies)
    for E in energies:
        mc_slice   = MC_df[MC_df["beam_energy"] == E]
        data_slice = Data_df[Data_df["beam_energy"] == E]
        print(f"\n=== Energy: {E} ===")

        particles = sorted(set(mc_slice["beam_type"]) | set(data_slice["beam_type"]))
        for ptype in particles:
            mc_part   = mc_slice[mc_slice["beam_type"] == ptype]
            data_part = data_slice[data_slice["beam_type"] == ptype]

            has_mc   = not mc_part.empty
            has_data = not data_part.empty

            # if both are empty, really nothing to do
            if not has_mc and not has_data:
                print(f"  -> particle: {ptype} (no MC and no Data, skip)")
                continue

            print(f"  -> particle: {ptype} (MC: {has_mc}, Data: {has_data})")

            # --- read RDFs only where needed ---
            mc_rdf = mc_rdf_chain = None
            data_rdf = data_rdf_chain = None
            mc_rdf_nfiles = data_rdf_nfiles = 0

            if has_mc:
                mc_rdf, mc_rdf_chain, mc_rdf_nfiles = read_rdf(args, mc_part)
                if mc_rdf_nfiles < 1:
                    has_mc = False

            if has_data:
                data_rdf, data_rdf_chain, data_rdf_nfiles = read_rdf(args, data_part)
                if data_rdf_nfiles < 1:
                    has_data = False

            if not has_mc and not has_data:
                print("    No files after read_rdf, skip")
                continue

            

            # if neither MC nor Data got a pred proxy (e.g. only MC but has_mc was killed above), skip
            if not has_mc and not has_data:
                continue

            # --- full (unselected) histograms ---
            if has_mc:
                MC_full_proxy = mc_rdf.Histo1D(
                    (f"MC_full_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                    hist_name
                )
            if has_data:
                Data_full_proxy = data_rdf.Histo1D(
                    (f"Data_full_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                    hist_name
                )

            # materialize TH1s and detach from any file
            if has_mc:
                MC_full_hist = MC_full_proxy.GetValue()
                MC_full_hist.SetDirectory(0)
                MC_full_hist.GetXaxis().SetTitle(axis_title)
                MC_full_hists[E][ptype] = MC_full_hist


            if has_data:
                Data_full_hist = Data_full_proxy.GetValue()
                Data_full_hist.SetDirectory(0)
                Data_full_hist.GetXaxis().SetTitle(axis_title)
                data_full_hists[E][ptype] = Data_full_hist


            # keep proxies & RDFs alive (None is fine for missing ones)
            hist_proxies.append(
                (mc_rdf, data_rdf,
                MC_full_proxy, Data_full_proxy)
            )


    print(MC_full_hists, data_full_hists)
    return MC_full_hists, data_full_hists, hist_proxies
            
 


#control_region_columns = ['count_scifi', 'sum_hit_density', 'centroid_slope_x', 'centroid_slope_y']

hist_info = {
    "Prediction": (100, 0, 1, 'GNN Prediction Score', True),
    "z": (206, 319, 370, 'Start Z Position', True),
    
    
    "density_scifi": (70, 0, 7e4, 'Sum of Density Weight',True),
    'density_scifi1':  (70, 0, 7e4, 'SciFi1 Sum of Density Weight', True),
    'density_scifi2':  (70, 0, 7e4, 'SciFi2 Sum of Density Weight', True),
    'density_scifi3':  (70, 0, 7e4, 'SciFi3 Sum of Density Weight', True),
    'density_scifi4':  (70, 0, 7e4, 'SciFi4 Sum of Density Weight', True),
    
    'count_scifi':  (100, 0, 3000, 'SciFi Hit Total Count', True),
    'count_scifi1':  (100, 0, 1000, 'SciFi1 Hit Total Count', True),
    'count_scifi2':  (100, 0, 1000, 'SciFi2 Hit Total Count', True),
    'count_scifi3':  (100, 0, 1000, 'SciFi3 Hit Total Count', True),
    'count_scifi4':  (100, 0, 1000, 'SciFi4 Hit Total Count', True),

    "avg_scifi_y": (28, 37, 51, 'Scifi AvgPos Y', False),
    "avg_scifi1_y": (28, 37,51, 'Scifi1 AvgPos Y', False),
    "avg_scifi2_y": (28, 37,51, 'Scifi2 AvgPos Y', True),
    "avg_scifi3_y": (28, 37,51, 'Scifi3 AvgPos Y', True),
    "avg_scifi4_y": (28, 37,51, 'Scifi4 AvgPos Y', True),

    
    "avg_scifi_x": (30, -30, -45, 'Scifi AvgPos X', True),
    "avg_scifi1_x": (50, -30, -45, 'Scifi1 AvgPos X', True),
    "avg_scifi2_x": (30, -30, -45, 'Scifi2 AvgPos X', True),
    "avg_scifi3_x": (30, -30, -45, 'Scifi3 AvgPos X', True),
    "avg_scifi4_x": (30, -30, -45, 'Scifi4 AvgPos X', True),
}


    
def save_hists_to_root(filename, MC_full_hists, data_full_hists, hist_name):
    """
    MC_full_hists[E][ptype]  -> TH1
    data_full_hists[E][ptype] -> TH1
    """
    outfile = ROOT.TFile(str(filename), "RECREATE")
    if outfile.IsZombie():
        raise RuntimeError(f"Could not create ROOT file: {filename}")

    # Helper to write one nested dict under a top-level directory
    def _write_group(topdir_name, nested_dict):
        if not nested_dict:
            return
        outfile.cd()
        topdir = outfile.mkdir(topdir_name)
        for E, pmap in nested_dict.items():
            topdir.cd()
            # make a subdirectory per energy
            edir = topdir.mkdir(str(E))
            edir.cd()
            for ptype, hist in pmap.items():
                # Optionally rename to something simple / consistent
                # current hist already has a name like "MC_full_{ptype}_{E}_{hist_name}"
                # but you can override if you like:
                # hist.SetName(f"{ptype}_{hist_name}")
                hist.Write()  # writes with current name

    _write_group("MC", MC_full_hists)
    _write_group("Data", data_full_hists)

    outfile.Close()
    print(f"Wrote histograms to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--hist_name", dest="hist_name", help="hist name", default="Prediction")
    parser.add_argument("-m", "--model_name", dest="model_name", help="model name", default="testbeam_2024_GravNet_v2")
    parser.add_argument("-f", "--metadata_dir", dest="metadata_dir", help="metadata diretory", default="/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/")
    parser.add_argument("-c", "--cut", dest="cut", help="apply cut", default="nocut")
    parser.add_argument("-s", "--split", dest="split", help="split name")
    args = parser.parse_args()
    
    print(f"processing hist of {args.hist_name}")
    
    print(f"applying cut: {args.cut}")
    
    MC_full_hists, data_full_hists, proxies = process_hist(args)
    n_bins, x_min, x_max, axis_title, logy = hist_info[args.hist_name]
    
    # choose an output file name; adapt to your arg names
    outdir = Path(f"./plots_tmp/{args.cut}/{args.hist_name}/")
    outdir.mkdir(parents=True, exist_ok=True)
    out_root = outdir / f"{args.hist_name}_hists.root"

    save_hists_to_root(out_root, MC_full_hists, data_full_hists, args.hist_name)
    

    # for E in sorted(MC_full_hists.keys()):
    #     plot_energy_distributions(
    #         E,
    #         MC_full_hists,
    #         data_full_hists,
    #         hist_name=args.hist_name,
    #         axis_title=axis_title,
    #         logy=logy,
    #         normalize=True,
    #         save_dir=f"./plots_tmp/{args.cut}/{args.hist_name}/"
    #     )
    
    # plot options
    # control region (scifi hits, density, shower direction)
    
    #read metadata
    

