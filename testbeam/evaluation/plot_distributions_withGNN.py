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
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
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
    Prediction_chain = ROOT.TChain("sndData")
    
    n_read_files = 0
    for _, row in metadata_df.iterrows():
        sub = row['subfolder']

        # ---- vetoFree ----
        feat = row['feature_path']

        pred = row[f'prediction_{model_name}_output_path']

        n_feat, _ = tree_entries_and_branch(feat, "sndData")
        n_pred, _ = tree_entries_and_branch(pred, "sndData")

        if n_feat > 0 and n_pred > 0 and n_feat == n_pred:
            feature_chain.Add(feat)
            Prediction_chain.Add(pred)
            n_read_files+=1
        else:
            if n_feat == 0:
                #print(f"[Skip] features empty/missing: {feat}")
                pass
            if n_pred == 0:
                #print(f"[Skip] Predictions empty/missing: {pred}")
                pass
            if n_feat != 0 and n_pred != 0 and n_feat != n_pred:
                #print(f"[Skip] entry mismatch (features={n_feat}, Predictions={n_pred}):\n  {feat}\n  {pred}")
                pass
        if n_read_files>=max_file:
            break
    print(f"Added {feature_chain.GetNtrees()} feature files and {Prediction_chain.GetNtrees()} Prediction files.")
    feature_chain.AddFriend(Prediction_chain, 'GnnPrediction')
    rdf = ROOT.RDataFrame(feature_chain)
    if args.cut == 'nocut':
        # no selection
        pass
    elif 'scifi_gt_50' in args.cut:
        rdf = rdf.Filter("count_scifi>50")
    else:
        raise ValueError(f'Unknown cut: {args.cut}')
    
    return rdf, feature_chain, n_read_files
    

def cal_matrix(args, outdir):
    model_name = args.model_name
    hist_name = args.hist_name

    METADATA_dict = read_metadata(args.metadata_dir, args.split)
    MC_df   = METADATA_dict['MC_data_testbeam2024_metadata']
    Data_df = METADATA_dict['real_data_testbeam_24_metadata']

    # energies present in either, excluding placeholders
    energies = sorted((set(MC_df["beam_energy"]) | set(Data_df["beam_energy"])) - {"no energy"})

    rows = []

    def _count_pred(rdf, cut):
        """Return event count after applying cut to rdf."""
        return int(rdf.Filter(cut).Count().GetValue())

    def _process_one(label, df_slice_e, df_slice_p, energy):
        """Build one row for MC or Data at a given energy."""
        # Define your threshold convention once
        cut_e = "Prediction < 0.5"
        cut_p = "Prediction >= 0.5"
        if (len(df_slice_e) > 0):
            e_rdf, e_chain, e_nfiles = read_rdf(args, df_slice_e)
            e_true_pred_e = _count_pred(e_rdf, cut_e)
            e_true_pred_p = _count_pred(e_rdf, cut_p)
        else:
            e_true_pred_e = 0
            e_true_pred_p = 0
        if (len(df_slice_p) > 0):
            p_rdf, p_chain, p_nfiles = read_rdf(args, df_slice_p)
            p_true_pred_e = _count_pred(p_rdf, cut_e)
            p_true_pred_p = _count_pred(p_rdf, cut_p)
        else:
            p_true_pred_e = 0
            p_true_pred_p = 0

        return {
            "sample": label,
            "beam_energy": energy,
            "e_true_pred_e": e_true_pred_e,
            "e_true_pred_p": e_true_pred_p,
            "p_true_pred_e": p_true_pred_e,
            "p_true_pred_p": p_true_pred_p,
        }

    for E in energies:
        print(f"\n=== Energy: {E} ===")

        # correct pandas boolean masks
        mc_electron_slice   = MC_df[(MC_df["beam_energy"] == E) & (MC_df["beam_type"] == "e-")]
        data_electron_slice = Data_df[(Data_df["beam_energy"] == E) & (Data_df["beam_type"] == "e-")]

        mc_pion_slice       = MC_df[(MC_df["beam_energy"] == E) & (MC_df["beam_type"] == "pi+")]
        data_pion_slice     = Data_df[(Data_df["beam_energy"] == E) & (Data_df["beam_type"] == "pi+")]

        mc_row   = _process_one("MC",   mc_electron_slice,   mc_pion_slice,   E)
        data_row = _process_one("Data", data_electron_slice, data_pion_slice, E)

        # quick printout for sanity
        print("MC  :", mc_row)
        print("Data:", data_row)

        rows.append(mc_row)
        rows.append(data_row)

    matrix_df = pd.DataFrame(rows).sort_values(["beam_energy", "sample"]).reset_index(drop=True)

    # Optional save
    out_csv = f'{outdir}/matrix.csv'
    matrix_df.to_csv(out_csv, index=False)

    print(matrix_df)
    return matrix_df
    

def plot_matrix(df, outdir):
    """
    Simplified confusion matrix plots for electron vs pion
    """
    # Create PDF
    with PdfPages(f'{outdir}/confusion_matrices.pdf') as pdf:
        
        # 1. Plot individual confusion matrices
        for _, row in df.iterrows():
            energy = row['beam_energy']
            sample_type = row['sample']
            
            # Create confusion matrix
            cm = np.array([
                [row['e_true_pred_e'], row['e_true_pred_p']],  # True electrons
                [row['p_true_pred_e'], row['p_true_pred_p']]   # True pions
            ])
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Absolute counts
            sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                       xticklabels=['Pred e', 'Pred π'],
                       yticklabels=['True e', 'True π'],
                       ax=ax1, cbar_kws={'label': 'Counts'})
            ax1.set_title(f'{energy} - {sample_type}\nAbsolute Counts')
            
            # Normalized by true class
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm / row_sums if row_sums.all() > 0 else cm * 0
            
            sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                       vmin=0, vmax=1, ax=ax2,
                       xticklabels=['Pred e', 'Pred π'],
                       yticklabels=['True e', 'True π'],
                       cbar_kws={'label': 'Fraction'})
            ax2.set_title(f'{energy} - {sample_type}\nNormalized by True Class')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # 2. Plot sum of all energies
        for sample_type in ['Data', 'MC']:
            # Sum values for this sample type
            sample_df = df[df['sample'] == sample_type]
            
            sum_e_e = sample_df['e_true_pred_e'].sum()
            sum_e_pi = sample_df['e_true_pred_p'].sum()
            sum_pi_e = sample_df['p_true_pred_e'].sum()
            sum_pi_pi = sample_df['p_true_pred_p'].sum()
            
            cm_sum = np.array([
                [sum_e_e, sum_e_pi],
                [sum_pi_e, sum_pi_pi]
            ])
            
            # Create plot for summed matrix
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Absolute counts
            sns.heatmap(cm_sum, annot=True, fmt=',d', cmap='Blues',
                       xticklabels=['Pred e', 'Pred π'],
                       yticklabels=['True e', 'True π'],
                       ax=ax1, cbar_kws={'label': 'Counts'})
            ax1.set_title(f'All Energies - {sample_type}\nAbsolute Counts')
            
            # Normalized
            row_sums = cm_sum.sum(axis=1, keepdims=True)
            cm_sum_norm = cm_sum / row_sums if row_sums.all() > 0 else cm_sum * 0
            
            sns.heatmap(cm_sum_norm, annot=True, fmt='.1%', cmap='Blues',
                       vmin=0, vmax=1, ax=ax2,
                       xticklabels=['Pred e', 'Pred π'],
                       yticklabels=['True e', 'True π'],
                       cbar_kws={'label': 'Fraction'})
            ax2.set_title(f'All Energies - {sample_type}\nNormalized by True Class')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print("PDF saved: confusion_matrices.pdf")
    
    
    
def process_hist(args):
    model_name = args.model_name
    hist_name = args.hist_name
    n_bins, x_min, x_max, axis_title, logy = hist_info[args.hist_name]
    METADATA_dict = read_metadata(args.metadata_dir, args.split)

    MC_df = METADATA_dict['MC_data_testbeam2024_metadata']
    Data_df = METADATA_dict['real_data_testbeam_24_metadata']

    data_full_hists = defaultdict(dict)
    data_pred_hists = defaultdict(dict)
    MC_full_hists   = defaultdict(dict)
    MC_pred_hists   = defaultdict(dict)

    hist_proxies = []

    # FIX: precedence bug
    energies = sorted((set(MC_df["beam_energy"]) | set(Data_df["beam_energy"])) - {"no energy"})

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

            if not has_mc and not has_data:
                print(f"  -> particle: {ptype} (no MC and no Data, skip)")
                continue

            print(f"  -> particle: {ptype} (MC: {has_mc}, Data: {has_data})")

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
            

            # FIX: clamp histogram variable into a stable column so all partial hists merge
            # (works even if some files have crazy sentinel values)
            safe_col = f"{hist_name}__clamped"
            clamp_expr = f"std::min(std::max(static_cast<double>({hist_name}), {float(x_min)}), {float(x_max)} - 1e-9)"
            if has_mc:
                mc_rdf = mc_rdf.Define(safe_col, clamp_expr)
            if has_data:
                data_rdf = data_rdf.Define(safe_col, clamp_expr)

            MC_pred_proxy = Data_pred_proxy = None
            MC_full_proxy = Data_full_proxy = None

            if "e" in ptype:
                if has_mc:
                    MC_pred_proxy = mc_rdf.Filter("Prediction < 0.5").Histo1D(
                        (f"MC_pred_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                        safe_col
                    )
                if has_data:
                    Data_pred_proxy = data_rdf.Filter("Prediction < 0.5").Histo1D(
                        (f"Data_pred_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                        safe_col
                    )

            elif "pi" in ptype:
                if has_mc:
                    MC_pred_proxy = mc_rdf.Filter("Prediction >= 0.5").Histo1D(
                        (f"MC_pred_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                        safe_col
                    )
                if has_data:
                    Data_pred_proxy = data_rdf.Filter("Prediction >= 0.5").Histo1D(
                        (f"Data_pred_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                        safe_col
                    )

            else:
                print(f"    Particle {ptype} not in training, skip")
                continue

            if has_mc:
                MC_full_proxy = mc_rdf.Histo1D(
                    (f"MC_full_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                    safe_col
                )
            if has_data:
                Data_full_proxy = data_rdf.Histo1D(
                    (f"Data_full_{ptype}_{E}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
                    safe_col
                )

            if has_mc:
                MC_pred_hist = MC_pred_proxy.GetValue()
                MC_full_hist = MC_full_proxy.GetValue()
                MC_pred_hist.SetDirectory(0)
                MC_full_hist.SetDirectory(0)
                MC_pred_hist.GetXaxis().SetTitle(axis_title)
                MC_full_hist.GetXaxis().SetTitle(axis_title)
                MC_full_hists[E][ptype] = MC_full_hist
                MC_pred_hists[E][ptype] = MC_pred_hist

            if has_data:
                Data_pred_hist = Data_pred_proxy.GetValue()
                Data_full_hist = Data_full_proxy.GetValue()
                Data_pred_hist.SetDirectory(0)
                Data_full_hist.SetDirectory(0)
                Data_pred_hist.GetXaxis().SetTitle(axis_title)
                Data_full_hist.GetXaxis().SetTitle(axis_title)
                data_full_hists[E][ptype] = Data_full_hist
                data_pred_hists[E][ptype] = Data_pred_hist

            hist_proxies.append(
                (mc_rdf, data_rdf,
                 MC_full_proxy, Data_full_proxy,
                 MC_pred_proxy, Data_pred_proxy)
            )

    print(MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists)
    return MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists, hist_proxies
  

def process_2d_hist(args):
    model_name = args.model_name
    hist_name  = args.hist_name

    nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logz = hist_info[hist_name]
    METADATA_dict = read_metadata(args.metadata_dir, args.split)

    MC_df   = METADATA_dict['MC_data_testbeam2024_metadata']
    Data_df = METADATA_dict['real_data_testbeam_24_metadata']

    # nested dicts: hists[energy][ptype] -> TH2
    data_full_hists = defaultdict(dict)
    data_pred_hists = defaultdict(dict)
    MC_full_hists   = defaultdict(dict)
    MC_pred_hists   = defaultdict(dict)

    # keep RDFs and RResultPtrs alive
    hist_proxies = []

    # decide which branches to plot on X/Y
    # avg_scifi, avg_scifi1, ... -> <hist_name>_x, <hist_name>_y
    if "avg_scifi" in hist_name:
        base = hist_name.removeprefix("2d_")  # Python ≥3.9
        x_var = f"{base}_x"
        y_var = f"{base}_y"
        x_title = axis_title + "X"
        y_title = axis_title + "Y"
    elif hist_name == "2d_xy_start_position":
        x_var = "x"
        y_var = "y"
        x_title = "Start X Position"
        y_title = "Start Y Position"
    else:
        raise ValueError(f"process_2d_hist: don't know x/y variables for hist_name='{hist_name}'")

    energies = sorted(set(MC_df["beam_energy"]) | set(Data_df["beam_energy"]) - {"no energy"})
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

            if not has_mc and not has_data:
                print(f"  -> particle: {ptype} (no MC and no Data, skip)")
                continue

            print(f"  -> particle: {ptype} (MC: {has_mc}, Data: {has_data})")

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

            # --- Prediction-selected histograms ---
            MC_pred_proxy = Data_pred_proxy = None
            MC_full_proxy = Data_full_proxy = None

            if "e" in ptype:
                cut = "Prediction < 0.5"
            elif "pi" in ptype:
                cut = "Prediction >= 0.5"
            else:
                print(f"    Particle {ptype} not in training, skip")
                continue

            if has_mc:
                MC_pred_proxy = mc_rdf.Filter(cut).Histo2D(
                    (f"MC_pred_{ptype}_{E}_{hist_name}", "",
                     int(nbins_x), float(x_min), float(x_max),
                     int(nbins_y), float(y_min), float(y_max)),
                    x_var, y_var
                )
            if has_data:
                Data_pred_proxy = data_rdf.Filter(cut).Histo2D(
                    (f"Data_pred_{ptype}_{E}_{hist_name}", "",
                     int(nbins_x), float(x_min), float(x_max),
                     int(nbins_y), float(y_min), float(y_max)),
                    x_var, y_var
                )

            # --- full (unselected) histograms ---
            if has_mc:
                MC_full_proxy = mc_rdf.Histo2D(
                    (f"MC_full_{ptype}_{E}_{hist_name}", "",
                     int(nbins_x), float(x_min), float(x_max),
                     int(nbins_y), float(y_min), float(y_max)),
                    x_var, y_var
                )
            if has_data:
                Data_full_proxy = data_rdf.Histo2D(
                    (f"Data_full_{ptype}_{E}_{hist_name}", "",
                     int(nbins_x), float(x_min), float(x_max),
                     int(nbins_y), float(y_min), float(y_max)),
                    x_var, y_var
                )

            # --- materialize TH2s and detach ---
            if has_mc:
                MC_pred_hist = MC_pred_proxy.GetValue()
                MC_full_hist = MC_full_proxy.GetValue()
                MC_pred_hist.SetDirectory(0)
                MC_full_hist.SetDirectory(0)

                MC_pred_hist.GetXaxis().SetTitle(x_title)
                MC_pred_hist.GetYaxis().SetTitle(y_title)
                MC_full_hist.GetXaxis().SetTitle(x_title)
                MC_full_hist.GetYaxis().SetTitle(y_title)

                MC_full_hists[E][ptype] = MC_full_hist
                MC_pred_hists[E][ptype] = MC_pred_hist

            if has_data:
                Data_pred_hist = Data_pred_proxy.GetValue()
                Data_full_hist = Data_full_proxy.GetValue()
                Data_pred_hist.SetDirectory(0)
                Data_full_hist.SetDirectory(0)

                Data_pred_hist.GetXaxis().SetTitle(x_title)
                Data_pred_hist.GetYaxis().SetTitle(y_title)
                Data_full_hist.GetXaxis().SetTitle(x_title)
                Data_full_hist.GetYaxis().SetTitle(y_title)

                data_full_hists[E][ptype] = Data_full_hist
                data_pred_hists[E][ptype] = Data_pred_hist

            # keep proxies & RDFs alive
            hist_proxies.append(
                (mc_rdf, data_rdf,
                 MC_full_proxy, Data_full_proxy,
                 MC_pred_proxy, Data_pred_proxy)
            )

    print(MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists)
    return MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists, hist_proxies
  
def plot_1d_distributions(
    E,
    MC_full_hists,
    MC_pred_hists,
    data_full_hists,
    data_pred_hists,
    hist_name,
    axis_title,
    logy=False,
    normalize=True,      
    save_dir=None,
):
    """
    Dictionary structure:
        MC_full_hists[E]["pion"/"electron"]   -> TH1   (before GNN)
        MC_pred_hists[E]["pion"/"electron"]   -> TH1   (after GNN)
        data_full_hists[E]["pion"/"electron"] -> TH1   (before GNN)
        data_pred_hists[E]["pion"/"electron"] -> TH1   (after GNN)

    Produces plots even if one histogram is missing.
    """
    # -------------------------
    # safe-get + clone
    # -------------------------
    def _get(hdict, E, key, name):
        try:
            h = hdict[E][key]
            return h.Clone(name) if h else None
        except Exception:
            return None
    # -------------------------
    # get hist (explicit)
    # -------------------------
    # "full" = before GNN
    h_mc_pion_full       = _get(MC_full_hists,   E, "pi+",      f"h_mc_pion_beforeGNN_{E}")
    h_mc_electron_full   = _get(MC_full_hists,   E, "e-",       f"h_mc_electron_beforeGNN_{E}")
    h_data_pion_full     = _get(data_full_hists, E, "pi+",      f"h_data_pion_beforeGNN_{E}")
    h_data_electron_full = _get(data_full_hists, E, "e-",       f"h_data_electron_beforeGNN_{E}")

    # "pred" = after GNN
    h_mc_pion_pred       = _get(MC_pred_hists,   E, "pi+",     f"h_mc_pion_afterGNN_{E}")
    h_mc_electron_pred   = _get(MC_pred_hists,   E, "e-",      f"h_mc_electron_afterGNN_{E}")
    h_data_pion_pred     = _get(data_pred_hists, E, "pi+",     f"h_data_pion_afterGNN_{E}")
    h_data_electron_pred = _get(data_pred_hists, E, "e-",      f"h_data_electron_afterGNN_{E}")

    all_hists = [
        h_mc_pion_full, h_mc_electron_full, h_data_pion_full, h_data_electron_full,
        h_mc_pion_pred, h_mc_electron_pred, h_data_pion_pred, h_data_electron_pred
    ]

    # -------------------------
    # normalize + axis titles
    # -------------------------
    for h in all_hists:
        if not h:
            continue
        if normalize:
            integ = h.Integral()
            if integ > 0:
                h.Scale(1.0 / integ)
        h.GetXaxis().SetTitle(axis_title)
        h.GetYaxis().SetTitle("a.u." if normalize else "Entries")
    # -------------------------
    # styling (only if exists)
    # -------------------------

    def style_mc(h, color, linestyle):
        h.SetLineColor(color)
        h.SetLineWidth(3)
        h.SetLineStyle(linestyle)
        h.SetMarkerStyle(0)   # ensure no markers sneak in

    def style_data(h, color, marker, open_marker=False):
        h.SetLineColor(color)          # error bars same color
        h.SetMarkerColor(color)
        h.SetMarkerStyle(marker)
        h.SetMarkerSize(1.1)
        # open vs filled
        if open_marker:
            h.SetMarkerStyle(marker + 4)  # ROOT convention: 20->24, 21->25, 22->26, 23->27 etc.
        # (optional) remove connecting line if you want pure markers:
        h.SetLineWidth(1)

    # choose distinct colors per particle
    COL_PION = ROOT.kRed + 1
    COL_ELE  = ROOT.kGreen + 2   # or ROOT.kAzure+2 if you prefer blue-ish electrons

    # MC: lines (before solid, after dashed)
    if h_mc_pion_full:     style_mc(h_mc_pion_full,     COL_PION, 1)
    if h_mc_pion_pred:     style_mc(h_mc_pion_pred,     COL_PION, 2)
    if h_mc_electron_full: style_mc(h_mc_electron_full, COL_ELE,  1)
    if h_mc_electron_pred: style_mc(h_mc_electron_pred, COL_ELE,  2)

    # Data: markers (before filled, after open) + particle-distinct marker shapes
    # pion: circle, electron: square
    if h_data_pion_full:     style_data(h_data_pion_full,     COL_PION, 20, open_marker=False)
    if h_data_pion_pred:     style_data(h_data_pion_pred,     COL_PION, 20, open_marker=True)

    if h_data_electron_full: style_data(h_data_electron_full, COL_ELE,  21, open_marker=False)
    if h_data_electron_pred: style_data(h_data_electron_pred, COL_ELE,  21, open_marker=True)
    # -------------------------
    # output dir
    # -------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    canvases = {}
    # =========================================================
    # 1) MC vs Data (separate pion, electron) [BEFORE GNN]
    # =========================================================
    # pion
    c1 = ROOT.TCanvas(f"c_{hist_name}_mc_vs_data_pion_beforeGNN_E{E}", "", 800, 600)
    if logy: c1.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False
    if h_mc_pion_full:
        h_mc_pion_full.SetTitle(f"{hist_name}: MC vs Data (pion, before GNN) | E={E}")
        h_mc_pion_full.Draw("HIST")
        leg.AddEntry(h_mc_pion_full, "MC pion (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC pion (before GNN) (missing)", "")
    if h_data_pion_full:
        h_data_pion_full.SetTitle(f"{hist_name}: MC vs Data (pion, before GNN) | E={E}")
        h_data_pion_full.Draw("E SAME" if drawn else "E")
        leg.AddEntry(h_data_pion_full, "Data pion (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data pion (before GNN) (missing)", "")
    leg.Draw(); c1.Update()
    if save_dir: c1.SaveAs(f"{save_dir}/{hist_name}_mc_vs_data_pion_beforeGNN_E{E}.pdf")
    canvases["mc_vs_data_pion_beforeGNN"] = c1
    # electron
    c2 = ROOT.TCanvas(f"c_{hist_name}_mc_vs_data_electron_beforeGNN_E{E}", "", 800, 600)
    if logy: c2.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False
    if h_mc_electron_full:
        h_mc_electron_full.SetTitle(f"{hist_name}: MC vs Data (electron, before GNN) | E={E}")
        h_mc_electron_full.Draw("HIST")
        leg.AddEntry(h_mc_electron_full, "MC electron (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC electron (before GNN) (missing)", "")
    if h_data_electron_full:
        h_data_electron_full.SetTitle(f"{hist_name}: MC vs Data (electron, before GNN) | E={E}")
        h_data_electron_full.Draw("E SAME" if drawn else "E")
        leg.AddEntry(h_data_electron_full, "Data electron (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data electron (before GNN) (missing)", "")
    leg.Draw(); c2.Update()
    if save_dir: c2.SaveAs(f"{save_dir}/{hist_name}_mc_vs_data_electron_beforeGNN_E{E}.pdf")
    canvases["mc_vs_data_electron_beforeGNN"] = c2
    # =========================================================
    # 2) MC before GNN vs MC after GNN (separate pion, electron)
    # =========================================================
    # pion
    c3 = ROOT.TCanvas(f"c_{hist_name}_mc_before_vs_after_pion_E{E}", "", 800, 600)
    if logy: c3.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_mc_pion_full:
        h_mc_pion_full.SetTitle(f"{hist_name}: MC before GNN vs after GNN (pion) | E={E}")
        h_mc_pion_full.Draw("HIST")
        leg.AddEntry(h_mc_pion_full, "MC pion (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC pion (before GNN) (missing)", "")

    if h_mc_pion_pred:
        h_mc_pion_pred.SetTitle(f"{hist_name}: MC before GNN vs after GNN (pion) | E={E}")
        h_mc_pion_pred.Draw("HIST SAME" if drawn else "HIST")
        leg.AddEntry(h_mc_pion_pred, "MC pion (after GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC pion (after GNN) (missing)", "")

    leg.Draw(); c3.Update()
    if save_dir: c3.SaveAs(f"{save_dir}/{hist_name}_mc_beforeGNN_vs_afterGNN_pion_E{E}.pdf")
    canvases["mc_before_vs_after_pion"] = c3

    # electron
    c4 = ROOT.TCanvas(f"c_{hist_name}_mc_before_vs_after_electron_E{E}", "", 800, 600)
    if logy: c4.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_mc_electron_full:
        h_mc_electron_full.SetTitle(f"{hist_name}: MC before GNN vs after GNN (electron) | E={E}")
        h_mc_electron_full.Draw("HIST")
        leg.AddEntry(h_mc_electron_full, "MC electron (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC electron (before GNN) (missing)", "")

    if h_mc_electron_pred:
        h_mc_electron_pred.SetTitle(f"{hist_name}: MC before GNN vs after GNN (electron) | E={E}")
        h_mc_electron_pred.Draw("HIST SAME" if drawn else "HIST")
        leg.AddEntry(h_mc_electron_pred, "MC electron (after GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC electron (after GNN) (missing)", "")

    leg.Draw(); c4.Update()
    if save_dir: c4.SaveAs(f"{save_dir}/{hist_name}_mc_beforeGNN_vs_afterGNN_electron_E{E}.pdf")
    canvases["mc_before_vs_after_electron"] = c4

    # =========================================================
    # 3) Data before GNN vs Data after GNN (separate pion, electron)
    # =========================================================
    # pion
    c5 = ROOT.TCanvas(f"c_{hist_name}_data_before_vs_after_pion_E{E}", "", 800, 600)
    if logy: c5.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_data_pion_full:
        h_data_pion_full.SetTitle(f"{hist_name}: Data before GNN vs after GNN (pion) | E={E}")
        h_data_pion_full.Draw("E")
        leg.AddEntry(h_data_pion_full, "Data pion (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data pion (before GNN) (missing)", "")

    if h_data_pion_pred:
        h_data_pion_pred.SetTitle(f"{hist_name}: Data before GNN vs after GNN (pion) | E={E}")
        h_data_pion_pred.Draw("E SAME" if drawn else "E")
        leg.AddEntry(h_data_pion_pred, "Data pion (after GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data pion (after GNN) (missing)", "")

    leg.Draw(); c5.Update()
    if save_dir: c5.SaveAs(f"{save_dir}/{hist_name}_data_beforeGNN_vs_afterGNN_pion_E{E}.pdf")
    canvases["data_before_vs_after_pion"] = c5

    # electron
    c6 = ROOT.TCanvas(f"c_{hist_name}_data_before_vs_after_electron_E{E}", "", 800, 600)
    if logy: c6.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_data_electron_full:
        h_data_electron_full.SetTitle(f"{hist_name}: Data before GNN vs after GNN (electron) | E={E}")
        h_data_electron_full.Draw("E")
        leg.AddEntry(h_data_electron_full, "Data electron (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data electron (before GNN) (missing)", "")

    if h_data_electron_pred:
        h_data_electron_pred.SetTitle(f"{hist_name}: Data before GNN vs after GNN (electron) | E={E}")
        h_data_electron_pred.Draw("E SAME" if drawn else "E")
        leg.AddEntry(h_data_electron_pred, "Data electron (after GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data electron (after GNN) (missing)", "")

    leg.Draw(); c6.Update()
    if save_dir: c6.SaveAs(f"{save_dir}/{hist_name}_data_beforeGNN_vs_afterGNN_electron_E{E}.pdf")
    canvases["data_before_vs_after_electron"] = c6

    # =========================================================
    # 4) Pion vs electron (separate MC, data) [BEFORE GNN]
    # =========================================================
    # MC
    c7 = ROOT.TCanvas(f"c_{hist_name}_pion_vs_electron_mc_beforeGNN_E{E}", "", 800, 600)
    if logy: c7.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_mc_pion_full:
        h_mc_pion_full.SetTitle(f"{hist_name}: pion vs electron (MC, before GNN) | E={E}")
        h_mc_pion_full.Draw("HIST")
        leg.AddEntry(h_mc_pion_full, "MC pion (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC pion (before GNN) (missing)", "")

    if h_mc_electron_full:
        h_mc_electron_full.SetTitle(f"{hist_name}: pion vs electron (MC, before GNN) | E={E}")
        h_mc_electron_full.Draw("HIST SAME" if drawn else "HIST")
        leg.AddEntry(h_mc_electron_full, "MC electron (before GNN)", "l")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "MC electron (before GNN) (missing)", "")

    leg.Draw(); c7.Update()
    if save_dir: c7.SaveAs(f"{save_dir}/{hist_name}_pion_vs_electron_mc_beforeGNN_E{E}.pdf")
    canvases["pion_vs_electron_mc_beforeGNN"] = c7

    # Data
    c8 = ROOT.TCanvas(f"c_{hist_name}_pion_vs_electron_data_beforeGNN_E{E}", "", 800, 600)
    if logy: c8.SetLogy()
    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    drawn = False

    if h_data_pion_full:
        h_data_pion_full.SetTitle(f"{hist_name}: pion vs electron (Data, before GNN) | E={E}")
        h_data_pion_full.Draw("E")
        leg.AddEntry(h_data_pion_full, "Data pion (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data pion (before GNN) (missing)", "")

    if h_data_electron_full:
        h_data_electron_full.SetTitle(f"{hist_name}: pion vs electron (Data, before GNN) | E={E}")
        h_data_electron_full.Draw("E SAME" if drawn else "E")
        leg.AddEntry(h_data_electron_full, "Data electron (before GNN)", "pe")
        drawn = True
    else:
        tmp = 0
        leg.AddEntry(tmp, "Data electron (before GNN) (missing)", "")

    leg.Draw(); c8.Update()
    if save_dir: c8.SaveAs(f"{save_dir}/{hist_name}_pion_vs_electron_data_beforeGNN_E{E}.pdf")
    canvases["pion_vs_electron_data_beforeGNN"] = c8

    
    
    # plot sum hist during plotting one of the energy
    if "300" in str(E):
        # ---------------------------------------------------------
        # helper: sum over all energies in a dict for a given key
        # ---------------------------------------------------------
        def _sum_hists(hdict, key, name):
            """Sum TH1 over all energies for a given particle key. Returns a CLONE (owned by caller)."""
            if not hdict:
                return None
            hsum = None
            # iterate deterministically (energies may be str/int)
            for Ek in sorted(list(hdict.keys()), key=lambda x: str(x)):
                try:
                    h = hdict[Ek].get(key, None)
                except Exception:
                    h = None
                if not h:
                    continue
                if not hsum:
                    hsum = h.Clone(name)
                    hsum.SetDirectory(0)
                else:
                    hsum.Add(h)
            return hsum

        # ---------------------------------------------------------
        # build summed (BEFORE GNN = full) histograms
        # ---------------------------------------------------------
        h_mc_pion_sum       = _sum_hists(MC_full_hists,   "pi+", f"h_mc_pion_sum_beforeGNN")
        h_mc_electron_sum   = _sum_hists(MC_full_hists,   "e-",  f"h_mc_electron_sum_beforeGNN")
        h_data_pion_sum     = _sum_hists(data_full_hists, "pi+", f"h_data_pion_sum_beforeGNN")
        h_data_electron_sum = _sum_hists(data_full_hists, "e-",  f"h_data_electron_sum_beforeGNN")

        sum_hists = [h_mc_pion_sum, h_mc_electron_sum, h_data_pion_sum, h_data_electron_sum]

        # normalize + axis titles (same policy as above)
        for h in sum_hists:
            if not h:
                continue
            if normalize:
                integ = h.Integral()
                if integ > 0:
                    h.Scale(1.0 / integ)
            h.GetXaxis().SetTitle(axis_title)
            h.GetYaxis().SetTitle("a.u." if normalize else "Entries")

        # style (sum plots are "before GNN" styling)
        if h_mc_pion_sum:       style_mc(h_mc_pion_sum,       COL_PION, 1)
        if h_mc_electron_sum:   style_mc(h_mc_electron_sum,   COL_ELE,  1)
        if h_data_pion_sum:     style_data(h_data_pion_sum,   COL_PION, 20, open_marker=False)
        if h_data_electron_sum: style_data(h_data_electron_sum, COL_ELE, 21, open_marker=False)

        # =========================================================
        # SUM PLOTS (full/before GNN): make the 4 requested combos
        # =========================================================

        # 1) plot MC pion vs MC electron
        c9 = ROOT.TCanvas(f"c_{hist_name}_SUM_mc_pion_vs_electron_beforeGNN", "", 800, 600)
        if logy: c9.SetLogy()
        leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
        drawn = False

        if h_mc_pion_sum:
            h_mc_pion_sum.SetTitle(f"{hist_name}: MC pion vs electron (SUM over E, before GNN)")
            h_mc_pion_sum.Draw("HIST")
            leg.AddEntry(h_mc_pion_sum, "MC pion (SUM, before GNN)", "l")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "MC pion (SUM, before GNN) (missing)", "")

        if h_mc_electron_sum:
            h_mc_electron_sum.SetTitle(f"{hist_name}: MC pion vs electron (SUM over E, before GNN)")
            h_mc_electron_sum.Draw("HIST SAME" if drawn else "HIST")
            leg.AddEntry(h_mc_electron_sum, "MC electron (SUM, before GNN)", "l")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "MC electron (SUM, before GNN) (missing)", "")

        leg.Draw(); c9.Update()
        if save_dir: c9.SaveAs(f"{save_dir}/{hist_name}_SUM_mc_pion_vs_electron_beforeGNN.pdf")
        canvases["SUM_mc_pion_vs_electron_beforeGNN"] = c9

        # 2) plot Data pion vs Data electron
        c10 = ROOT.TCanvas(f"c_{hist_name}_SUM_data_pion_vs_electron_beforeGNN", "", 800, 600)
        if logy: c10.SetLogy()
        leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
        drawn = False

        if h_data_pion_sum:
            h_data_pion_sum.SetTitle(f"{hist_name}: Data pion vs electron (SUM over E, before GNN)")
            h_data_pion_sum.Draw("E")
            leg.AddEntry(h_data_pion_sum, "Data pion (SUM, before GNN)", "pe")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "Data pion (SUM, before GNN) (missing)", "")

        if h_data_electron_sum:
            h_data_electron_sum.SetTitle(f"{hist_name}: Data pion vs electron (SUM over E, before GNN)")
            h_data_electron_sum.Draw("E SAME" if drawn else "E")
            leg.AddEntry(h_data_electron_sum, "Data electron (SUM, before GNN)", "pe")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "Data electron (SUM, before GNN) (missing)", "")

        leg.Draw(); c10.Update()
        if save_dir: c10.SaveAs(f"{save_dir}/{hist_name}_SUM_data_pion_vs_electron_beforeGNN.pdf")
        canvases["SUM_data_pion_vs_electron_beforeGNN"] = c10

        # 3) plot MC pion vs Data pion
        c11 = ROOT.TCanvas(f"c_{hist_name}_SUM_mc_vs_data_pion_beforeGNN", "", 800, 600)
        if logy: c11.SetLogy()
        leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
        drawn = False

        if h_mc_pion_sum:
            h_mc_pion_sum.SetTitle(f"{hist_name}: MC vs Data (pion, SUM over E, before GNN)")
            h_mc_pion_sum.Draw("HIST")
            leg.AddEntry(h_mc_pion_sum, "MC pion (SUM, before GNN)", "l")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "MC pion (SUM, before GNN) (missing)", "")

        if h_data_pion_sum:
            h_data_pion_sum.SetTitle(f"{hist_name}: MC vs Data (pion, SUM over E, before GNN)")
            h_data_pion_sum.Draw("E SAME" if drawn else "E")
            leg.AddEntry(h_data_pion_sum, "Data pion (SUM, before GNN)", "pe")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "Data pion (SUM, before GNN) (missing)", "")

        leg.Draw(); c11.Update()
        if save_dir: c11.SaveAs(f"{save_dir}/{hist_name}_SUM_mc_vs_data_pion_beforeGNN.pdf")
        canvases["SUM_mc_vs_data_pion_beforeGNN"] = c11

        # 4) plot MC electron vs Data electron
        c12 = ROOT.TCanvas(f"c_{hist_name}_SUM_mc_vs_data_electron_beforeGNN", "", 800, 600)
        if logy: c12.SetLogy()
        leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
        drawn = False

        if h_mc_electron_sum:
            h_mc_electron_sum.SetTitle(f"{hist_name}: MC vs Data (electron, SUM over E, before GNN)")
            h_mc_electron_sum.Draw("HIST")
            leg.AddEntry(h_mc_electron_sum, "MC electron (SUM, before GNN)", "l")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "MC electron (SUM, before GNN) (missing)", "")

        if h_data_electron_sum:
            h_data_electron_sum.SetTitle(f"{hist_name}: MC vs Data (electron, SUM over E, before GNN)")
            h_data_electron_sum.Draw("E SAME" if drawn else "E")
            leg.AddEntry(h_data_electron_sum, "Data electron (SUM, before GNN)", "pe")
            drawn = True
        else:
            tmp = 0
            leg.AddEntry(tmp, "Data electron (SUM, before GNN) (missing)", "")

        leg.Draw(); c12.Update()
        if save_dir: c12.SaveAs(f"{save_dir}/{hist_name}_SUM_mc_vs_data_electron_beforeGNN.pdf")
        canvases["SUM_mc_vs_data_electron_beforeGNN"] = c12
        
        
        

    
    return canvases


def plot_2d_distributions(
    E,
    MC_full_hists,
    MC_pred_hists,
    data_full_hists,
    data_pred_hists,
    hist_name,
    axis_title,          # can be str or (x_title, y_title)
    logy=False,          # kept for API compatibility; used as logZ for TH2
    normalize=True,      # probability density on z-axis
    save_dir=None,
):
    # -------------------------
    # unpack limits from hist_info
    # -------------------------
    if hist_name not in hist_info:
        raise KeyError(f"hist_name='{hist_name}' not found in hist_info")

    nbx, xmin, xmax, nby, ymin, ymax, axis_from_info, logz_from_info = hist_info[hist_name]

    # auto-fix swapped ranges
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    # choose axis title + logz
    axis_title = axis_from_info if axis_title is None else axis_title
    use_logz = logz_from_info if logy is None else bool(logy)

    # If you want separate X/Y titles, pass axis_title=("X title","Y title")
    if isinstance(axis_title, (tuple, list)) and len(axis_title) >= 2:
        x_title, y_title = axis_title[0], axis_title[1]
    else:
        x_title, y_title = axis_title, ""

    # -------------------------
    # safe-get + clone (detach from directory)
    # -------------------------
    def _get(hdict, E, key, name):
        try:
            h = hdict[E][key]
            if not h:
                return None
            hc = h.Clone(name)
            hc.SetDirectory(0)
            return hc
        except Exception:
            return None

    # get hists
    h_mc_pion_full       = _get(MC_full_hists,   E, "pi+", f"h_mc_pi_before_{E}")
    h_mc_electron_full   = _get(MC_full_hists,   E, "e-",  f"h_mc_e_before_{E}")
    h_data_pion_full     = _get(data_full_hists, E, "pi+", f"h_data_pi_before_{E}")
    h_data_electron_full = _get(data_full_hists, E, "e-",  f"h_data_e_before_{E}")

    h_mc_pion_pred       = _get(MC_pred_hists,   E, "pi+", f"h_mc_pi_after_{E}")
    h_mc_electron_pred   = _get(MC_pred_hists,   E, "e-",  f"h_mc_e_after_{E}")
    h_data_pion_pred     = _get(data_pred_hists, E, "pi+", f"h_data_pi_after_{E}")
    h_data_electron_pred = _get(data_pred_hists, E, "e-",  f"h_data_e_after_{E}")

    items = [
        (h_mc_pion_full,       "MC_pi+_beforeGNN"),
        (h_mc_electron_full,   "MC_e-_beforeGNN"),
        (h_data_pion_full,     "Data_pi+_beforeGNN"),
        (h_data_electron_full, "Data_e-_beforeGNN"),
        (h_mc_pion_pred,       "MC_pi+_afterGNN"),
        (h_mc_electron_pred,   "MC_e-_afterGNN"),
        (h_data_pion_pred,     "Data_pi+_afterGNN"),
        (h_data_electron_pred, "Data_e-_afterGNN"),
    ]

    # normalize + set axis limits/titles
    for h, _ in items:
        if not h:
            continue

        # enforce display range from hist_info
        h.GetXaxis().SetLimits(xmin, xmax)
        h.GetYaxis().SetLimits(ymin, ymax)
        # also restrict displayed bins if histogram has wider booking:
        h.GetXaxis().SetRangeUser(xmin, xmax)
        h.GetYaxis().SetRangeUser(ymin, ymax)

        if normalize:
            integ = h.Integral()  # total contents
            if integ > 0:
                h.Scale(1.0 / integ)

        h.GetXaxis().SetTitle(x_title)
        h.GetYaxis().SetTitle(y_title)
        h.GetZaxis().SetTitle("a.u." if normalize else "Entries")
        h.SetStats(0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    canvases = {}
    for h, tag in items:
        if not h:
            continue

        c = ROOT.TCanvas(f"c_{hist_name}_{tag}_E{E}", "", 900, 750)
        c.SetRightMargin(0.16)   # <-- THIS fixes the color bar
        c.SetLeftMargin(0.12)
        c.SetBottomMargin(0.12)
        c.SetTopMargin(0.08)
        if use_logz:
            c.SetLogz()

        h.SetTitle(f"{hist_name} | {tag.replace('_', ' ')} | E={E}")
        h.Draw("COLZ")
        c.Update()

        if save_dir:
            c.SaveAs(f"{save_dir}/{hist_name}_{tag}_E{E}.pdf")

        canvases[tag] = c

    return canvases


def plot_energy_distributions(
    E,
    MC_full_hists,
    MC_pred_hists,
    data_full_hists,
    data_pred_hists,
    hist_name,
    axis_title,
    logy=False,
    normalize=True,      # probability density on y-axis
    save_dir=None,
):
    """
    For a given energy E, plot:
      - all available particle types at this energy
      - MC and/or Data (whatever exists)
      - full and pred (whatever exists)
    all overlaid in a single canvas.

    Dictionary structure:
        MC_full_hists[E][ptype]   -> TH1
        MC_pred_hists[E][ptype]   -> TH1
        data_full_hists[E][ptype] -> TH1
        data_pred_hists[E][ptype] -> TH1

    If normalize=True, histograms are scaled to probability density:
        ∫ f(x) dx = 1
    """

    #Each energy:
        # MC_vs_Data_pion,  MC_vs_Data_electron
        # pion_vs_eletron_MC, pion_vs_eletron_Data
        #
        
        


    # Check if *anything* exists for this energy
    has_any_E = any(
        E in d for d in (
            MC_full_hists,
            MC_pred_hists,
            data_full_hists,
            data_pred_hists,
        )
    )
    if not has_any_E:
        print(f"[plot_energy_distributions] No histograms at all for energy {E}, skip.")
        return None

    # Collect all particle types that appear for this energy in any dict
    ptypes = set()
    if E in MC_full_hists:
        ptypes |= set(MC_full_hists[E].keys())
    if E in MC_pred_hists:
        ptypes |= set(MC_pred_hists[E].keys())
    if E in data_full_hists:
        ptypes |= set(data_full_hists[E].keys())
    if E in data_pred_hists:
        ptypes |= set(data_pred_hists[E].keys())

    ptypes = sorted(ptypes)
    if not ptypes:
        print(f"[plot_energy_distributions] No particle types for energy {E}, skip.")
        return None

    # Canvas
    cname = f"c_{hist_name}_{E}"
    canvas = ROOT.TCanvas(cname, cname, 900, 700)
    if logy:
        canvas.SetLogy()

    # Base color per particle (hue)
    particle_colors = {
        "e": ROOT.kAzure + 1,
        "electron": ROOT.kAzure + 1,
        "pion": ROOT.kRed + 1,
        "pi": ROOT.kRed + 1,
    }
    default_colors = [ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 1]
    default_color_idx = 0

    def get_color(ptype: str) -> int:
        nonlocal default_color_idx
        for key, col in particle_colors.items():
            if key in ptype:
                return col
        col = default_colors[default_color_idx % len(default_colors)]
        default_color_idx += 1
        return col

    legend = ROOT.TLegend(0.55, 0.60, 0.89, 0.89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    all_hists = []

    for ptype in ptypes:
        # Fetch histograms (may be None)
        h_MC_full   = MC_full_hists.get(E, {}).get(ptype)
        h_MC_pred   = MC_pred_hists.get(E, {}).get(ptype)
        h_Data_full = data_full_hists.get(E, {}).get(ptype)
        h_Data_pred = data_pred_hists.get(E, {}).get(ptype)

        if not any([h_MC_full, h_MC_pred, h_Data_full, h_Data_pred]):
            print(f"  [E={E}] No histograms for {ptype}, skip.")
            continue

        # Base hue for this particle
        base_color = get_color(ptype)

        # Derive four related colors:
        #   MC_full  : darkest
        #   MC_pred  : base
        #   Data_full: brighter
        #   Data_pred: even brighter
        color_MC_full   = ROOT.TColor.GetColorDark(base_color)
        color_MC_pred   = base_color
        color_Data_full = ROOT.TColor.GetColorBright(base_color)
        color_Data_pred = ROOT.TColor.GetColorBright(color_Data_full)

        # Normalize to probability density, if requested
        for h in (h_MC_full, h_MC_pred, h_Data_full, h_Data_pred):
            if h is None:
                continue
            if normalize:
                integral = h.Integral()
                if integral > 0:
                    h.Scale(1.0 / integral)


        # ---------------- MC styling ----------------
        # MC full: solid line, darker color
        if h_MC_full:
            h_MC_full.SetLineColor(color_MC_full)
            h_MC_full.SetLineWidth(3)
            h_MC_full.SetLineStyle(1)
            h_MC_full.GetXaxis().SetTitle(axis_title)
            all_hists.append(h_MC_full)
            legend.AddEntry(h_MC_full, f"MC full {ptype}, {E} GeV", "l")

        # MC pred: dashed line, slightly lighter
        if h_MC_pred:
            h_MC_pred.SetLineColor(color_MC_pred)
            h_MC_pred.SetLineWidth(3)
            h_MC_pred.SetLineStyle(2)
            if not h_MC_full:
                h_MC_pred.GetXaxis().SetTitle(axis_title)
            all_hists.append(h_MC_pred)
            legend.AddEntry(h_MC_pred, f"MC pred {ptype}, {E} GeV", "l")

        # ---------------- Data styling ----------------
        # Data full: filled markers, bright color
        if h_Data_full:
            h_Data_full.SetLineColor(color_Data_full)
            h_Data_full.SetMarkerColor(color_Data_full)
            h_Data_full.SetMarkerStyle(20)   # filled circles
            h_Data_full.SetMarkerSize(0.9)
            if not (h_MC_full or h_MC_pred):
                h_Data_full.GetXaxis().SetTitle(axis_title)
            all_hists.append(h_Data_full)
            legend.AddEntry(h_Data_full, f"Data full {ptype}, {E} GeV", "p")

        # Data pred: open markers, even brighter
        if h_Data_pred:
            h_Data_pred.SetLineColor(color_Data_pred)
            h_Data_pred.SetMarkerColor(color_Data_pred)
            h_Data_pred.SetMarkerStyle(24)   # open triangles
            h_Data_pred.SetMarkerSize(1.1)
            if not (h_MC_full or h_MC_pred or h_Data_full):
                h_Data_pred.GetXaxis().SetTitle(axis_title)
            all_hists.append(h_Data_pred)
            legend.AddEntry(h_Data_pred, f"Data pred {ptype}, {E} GeV", "p")

    if not all_hists:
        print(f"[plot_energy_distributions] No valid histograms to draw for E={E}.")
        return None

    # Determine global y-range
    max_y = max(h.GetMaximum() for h in all_hists)

    # Positive minimum for log-scale
    min_pos_vals = []
    for h in all_hists:
        # search over bins for positive minima
        nb = h.GetNbinsX()
        vals = [h.GetBinContent(i) for i in range(1, nb + 1) if h.GetBinContent(i) > 0]
        if vals:
            min_pos_vals.append(min(vals))
    min_y_pos = min(min_pos_vals) if min_pos_vals else 1e-6

    # Frame histogram for axis & title
    href = all_hists[0].Clone(f"{all_hists[0].GetName()}_frame")
    href.Reset("ICE")
    if normalize:
        href.SetTitle(f"{hist_name} @ {E} GeV;{axis_title};Probability density")
    else:
        href.SetTitle(f"{hist_name} @ {E} GeV;{axis_title};Entries")

    if logy:
        href.SetMinimum(min_y_pos * 0.5)
        href.SetMaximum(max_y * 10.0)
    else:
        href.SetMinimum(0.0)
        href.SetMaximum(max_y * 1.3)

    href.Draw("AXIS")

    # Draw all histograms
    for h in all_hists:
        name = h.GetName()
        if name.startswith("MC_") or "_MC_" in name:
            h.Draw("HIST SAME")
        else:
            h.Draw("E SAME")

    legend.Draw()
    canvas.RedrawAxis()

    # Save to disk if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_pdf = os.path.join(save_dir, f"{hist_name}_E{E}.pdf")
        canvas.SaveAs(out_pdf)
        print(f"[plot_energy_distributions] Saved {out_pdf}")

    return canvas
            

def _write_group_to_root(outfile, group_name, nested_dict):
    """
    nested_dict[E][ptype] -> TH1
    Writes them under: /group_name/E/ in the file.
    """
    if not nested_dict:
        return

    outfile.cd()
    topdir = outfile.mkdir(group_name)

    for E, pmap in nested_dict.items():
        topdir.cd()
        edir = topdir.mkdir(str(E))
        edir.cd()
        for ptype, hist in pmap.items():
            # Make sure each hist has a unique, sensible name
            # If they already do, you can comment this out.
            if not hist.GetName():
                hist.SetName(f"{group_name}_{ptype}_{E}")
            hist.Write()  # uses current name


def save_all_hists_to_root(filename,
                           MC_full_hists,
                           MC_pred_hists,
                           data_full_hists,
                           data_pred_hists):
    """
    Save all 4 nested dicts to one ROOT file.
    """
    outfile = ROOT.TFile(str(filename), "RECREATE")
    if outfile.IsZombie():
        raise RuntimeError(f"Could not create ROOT file: {filename}")

    _write_group_to_root(outfile, "MC_full",   MC_full_hists)
    _write_group_to_root(outfile, "MC_pred",   MC_pred_hists)
    _write_group_to_root(outfile, "Data_full", data_full_hists)
    _write_group_to_root(outfile, "Data_pred", data_pred_hists)

    outfile.Close()
    print(f"[INFO] Wrote histograms to {filename}")

#control_region_columns = ['count_scifi', 'sum_hit_density', 'centroid_slope_x', 'centroid_slope_y']

hist_info = {
    # n_bins, x_min, x_max, axis_title, logy
    "Prediction": (100, 0, 1, 'GNN Prediction Score', False),
    "z": (206, 319, 370, 'Start Z Position', True),
    "density_scifi": (100, 0, 2e5, 'Sum of Density Weight',True),
    'density_scifi1':  (100, 0, 1e5, 'SciFi1 Sum of Density Weight', True),
    'density_scifi2':  (100, 0, 1e5, 'SciFi2 Sum of Density Weight', True),
    'density_scifi3':  (100, 0, 1e5, 'SciFi3 Sum of Density Weight', True),
    'density_scifi4':  (100, 0, 1e5, 'SciFi4 Sum of Density Weight', True),
    
    'count_scifi':  (300, 0, 3000, 'SciFi Hit Total Count', True),
    'count_scifi1':  (100, 0, 1000, 'SciFi1 Hit Total Count', True),
    'count_scifi2':  (100, 0, 1000, 'SciFi2 Hit Total Count', True),
    'count_scifi3':  (100, 0, 1000, 'SciFi3 Hit Total Count', True),
    'count_scifi4':  (100, 0, 1000, 'SciFi4 Hit Total Count', True),
    
    "avg_scifi_y": (28, 37, 51, 'Scifi AvgPos Y', False),
    "avg_scifi1_y": (28, 37,51, 'Scifi1 AvgPos Y', False),
    "avg_scifi2_y": (28, 37,51, 'Scifi2 AvgPos Y', False),
    "avg_scifi3_y": (28, 37,51, 'Scifi3 AvgPos Y', False),
    "avg_scifi4_y": (28, 37,51, 'Scifi4 AvgPos Y', False),

    
    "avg_scifi_x": (30, -45, -30, 'Scifi AvgPos X', False),
    "avg_scifi1_x": (50, -45, -30, 'Scifi1 AvgPos X', False),
    "avg_scifi2_x": (30, -45, -30, 'Scifi2 AvgPos X', False),
    "avg_scifi3_x": (30, -45, -30, 'Scifi3 AvgPos X', False),
    "avg_scifi4_x": (30, -45, -30, 'Scifi4 AvgPos X', False),

    #nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logz, bin_width_x, bin_width_y
    "2d_avg_scifi": (30, -45,-30, 28, 37, 51, 'Scifi Average Position ', False),
    "2d_avg_scifi1": (50, -45,-30, 28, 37,51, 'Scifi1 Average Position ', False),
    "2d_avg_scifi2": (30, -45,-30, 28, 37,51, 'Scifi2 Average Position ', False),
    "2d_avg_scifi3": (30, -45,-30, 28, 37,51, 'Scifi3 Average Position ', False),
    "2d_avg_scifi4": (30, -45,-30, 28, 37,51, 'Scifi4 Average Position ', False),
    "2d_xy_start_position": (30, -45,-30, 28, 37,51, 'XY Start Position ', False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--hist_name", dest="hist_name", help="hist name", default="Prediction")
    parser.add_argument("-m", "--model_name", dest="model_name", help="model name", default="testbeam_2024_GravNet_v2")
    parser.add_argument("-f", "--metadata_dir", dest="metadata_dir", help="metadata diretory", default="/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/")
    parser.add_argument("-c", "--cut", dest="cut", help="apply cut", default="nocut")
    parser.add_argument("-s", "--split", dest="split", help="split name")
    parser.add_argument("-t", "--threshold", dest="threshold", help="gnn score threshold", default=0.5)
    args = parser.parse_args()
    
    print(f"processing hist of {args.hist_name}")
    
    print(f"applying cut: {args.cut}")
    
    outdir = Path(f"./plots_tmp/{args.cut}/{args.hist_name}/")
    outdir.mkdir(parents=True, exist_ok=True)
    out_root = outdir / f"{args.model_name}_{args.hist_name}.root"
    
    if ("matrix" in  args.hist_name):
        matrix_df = cal_matrix(args, outdir)
        #matrix_df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/evaluation/plots/scifi_gt_50/confusion_matrix/matrix.csv")
        plot_matrix(matrix_df, outdir)
        return 0
    elif ("2d" in args.hist_name):
        MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists, proxies = process_2d_hist(args)
        nbins_x, x_min, x_max, nbins_y, y_min, y_max, axis_title, logy = hist_info[args.hist_name]
    else:
        MC_full_hists, MC_pred_hists, data_full_hists, data_pred_hists, proxies = process_hist(args)
        n_bins, x_min, x_max, axis_title, logy = hist_info[args.hist_name]

    save_all_hists_to_root(out_root,
                        MC_full_hists,
                        MC_pred_hists,
                        data_full_hists,
                        data_pred_hists)
    


    if ("2d" in args.hist_name):
        plot_func = plot_2d_distributions
    else:
        plot_func = plot_1d_distributions

    for E in sorted(MC_full_hists):
        plot_func(
            E,
            MC_full_hists,
            MC_pred_hists,
            data_full_hists,
            data_pred_hists,
            hist_name=args.hist_name,
            axis_title=axis_title,
            logy=logy,
            normalize=True,
            save_dir=outdir,
        )
    
    
    # plot options
    # control region (scifi hits, density, shower direction)
    
    #read metadata
    
if __name__ == "__main__":
    main()
