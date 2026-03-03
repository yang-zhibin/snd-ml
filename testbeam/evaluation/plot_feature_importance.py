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


from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)



particle_2_class = {
    'e': 0,
    'pion': 1,
}
class_2_particle = {v: k for k, v in particle_2_class.items()}

def rdf_safe_name(name: str) -> str:
    s = re.sub(r'[^0-9A-Za-z_]', '_', name)
    if re.match(r'^\d', s):
        s = '_' + s
    s = re.sub(r'__+', '_', s)
    return s

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

def read_rdf(args, metadata_df, max_file=1):
    model_name = args.model_name
    feature_chain = ROOT.TChain("sndData")
    Prediction_chain = ROOT.TChain("sndData")
    
    n_read_files = 0
    for _, row in metadata_df.iterrows():
        sub = row['subfolder']

        # ---- vetoFree ----
        feat = row['feature_path']

        pred = row[f'prediction_{model_name}_output_path']

        n_feat, has_req = tree_entries_and_branch(feat, "sndData", "qdc_avg")
        n_pred, _ = tree_entries_and_branch(pred, "sndData")

        if (n_feat > 0 and n_pred > 0 and n_feat == n_pred) and has_req:
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
    print(f"Added {feature_chain.GetNtrees()} feature files and {Prediction_chain.GetNtrees()} Prediction files. n_read_files: {n_read_files}")
    if n_read_files == 0:
        return None, None, n_read_files
    feature_chain.AddFriend(Prediction_chain, 'GnnPrediction')
    rdf = ROOT.RDataFrame(feature_chain)
    if args.cut == 'nocut':
        # no selection
        pass
    elif 'scifi_gt_50' in args.cut:
        rdf = rdf.Filter("count_scifi>50")
    elif 'scifi_gt_50-150ns_previous_cut' in args.cut:
        rdf = rdf.Filter("count_scifi>50 && previous_event_time_gap > 150")
    else:
        raise ValueError(f'Unknown cut: {args.cut}')
    
    
    
    feature_chain.GetEntry(408008)
    tree_number = feature_chain.GetTreeNumber()
    file_name = feature_chain.GetFile().GetName()

    print("File:", file_name)
    print("Tree number:", tree_number)
    return rdf, feature_chain, n_read_files
    

def rdf_to_xy(rdf, feature_cols, pred_col="Prediction", max_rows=2e5, seed=0):
    """
    Extract X (DataFrame) and y (1D numpy) from an RDataFrame.
    Optional subsampling for speed.
    """
    cols = list(feature_cols) + [pred_col]
    d = rdf.AsNumpy(columns=cols)  # dict of np arrays
    df = pd.DataFrame(d)

    # Keep numeric only (just in case)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN/inf anywhere in X or y
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if df.empty:
        return None, None

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    y = df[pred_col].to_numpy().astype(float)
    X = df.drop(columns=[pred_col])

    # Some MI routines hate constant columns; drop them
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep]

    if X.shape[1] == 0:
        return None, None

    return X, y


def spearman_importance(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Returns |Spearman rho| for each feature.
    """
    scores = {}
    for col in X.columns:
        rho, _ = spearmanr(X[col].to_numpy(), y)
        if np.isnan(rho):
            rho = 0.0
        scores[col] = abs(rho)
    return pd.Series(scores).sort_values(ascending=False)


def mi_importance(X: pd.DataFrame, y: np.ndarray, seed=0) -> pd.Series:
    """
    Mutual information importance I(x_j; y). Nonnegative.
    """
    # mutual_info_regression expects numpy array
    mi = mutual_info_regression(
        X.to_numpy(),
        y,
        random_state=seed,
        # n_neighbors default=3; you can tune if needed
    )
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def plot_importance(series: pd.Series, outpath: str, title: str, xlabel: str, topk=30):
    """
    Horizontal bar plot for a sorted importance series.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    s = series.iloc[:topk][::-1]  # reverse for barh (largest at top)
    plt.figure(figsize=(10, max(4, 0.28 * len(s))))
    plt.barh(s.index, s.values)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    

def run_feature_importance(rdf, feature_cols, tag, outdir):
    X, y = rdf_to_xy(rdf, feature_cols, seed=0)
    if X is None:
        print(f"    [{tag}] No valid rows/features after cleaning; skip importance")
        return

    # Spearman
    sp = spearman_importance(X, y)
    out_sp = os.path.join(outdir, "feature_importance", f"{tag}_spearman.png")
    plot_importance(
        sp,
        out_sp,
        title=f"Spearman |rho| importance ({tag})",
        xlabel="|Spearman rho(Prediction, feature)|",
        topk=30
    )

    # Mutual Information
    mi = mi_importance(X, y, seed=0)
    out_mi = os.path.join(outdir, "feature_importance", f"{tag}_mutual_info.png")
    plot_importance(
        mi,
        out_mi,
        title=f"Mutual information importance ({tag})",
        xlabel="Mutual information I(feature; Prediction)",
        topk=30
    )

    # also save numbers for later comparisons
    out_csv = os.path.join(outdir, "feature_importance", f"{tag}_importance.csv")
    pd.DataFrame({"spearman_abs_rho": sp, "mutual_info": mi}).to_csv(out_csv)
    print(f"    [{tag}] wrote:\n      {out_sp}\n      {out_mi}\n      {out_csv}")


    
def process_feature_importance(args, outdir):
    model_name = args.model_name
    METADATA_dict = read_metadata(args.metadata_dir, args.split)

    MC_df = METADATA_dict['MC_data_testbeam2024_metadata']
    Data_df = METADATA_dict['real_data_testbeam_24_metadata']

    data_full_hists = defaultdict(dict)
    data_pred_hists = defaultdict(dict)
    MC_full_hists   = defaultdict(dict)
    MC_pred_hists   = defaultdict(dict)

    hist_proxies = []

    energies = sorted((set(MC_df["beam_energy"]) | set(Data_df["beam_energy"])) - {"no energy"})

    feature_cols = HIST_NAMES  # your list of feature branch names

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
            
            tag_base = f"{model_name}_E{E}_{ptype}".replace(" ", "_").replace("/", "_")
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
            
            if has_data:
                run_feature_importance(data_rdf, feature_cols, tag=f"{tag_base}_DATA", outdir=outdir)

            if has_mc:
                run_feature_importance(mc_rdf, feature_cols,  tag=f"{tag_base}_MC", outdir=outdir)
            # inside the data_rdf, there are feature columns(save in HIST_NAMEs list) and predit columns (name "Prediction", where the predcition score saved)
            
            # I want to compute the Correlation-Based Importance and Mutual Information plots for the features
            

                
            

HIST_NAMES = [
    "qdc_scifi",
    # "qdc_scifi1",
    # "qdc_scifi2",
    # "qdc_scifi3",
    # "qdc_scifi4",
    "density_scifi", 
    # "density_scifi1",
    # "density_scifi2",
    # "density_scifi3",
    # "density_scifi4",

    "count_scifi",
    # "count_scifi1",
    # "count_scifi2",
    # "count_scifi3",
    # "count_scifi4",

    # "avg_scifi_x",
    # "avg_scifi1_x",
    # "avg_scifi2_x",
    # "avg_scifi3_x",
    # "avg_scifi4_x",

    # "avg_scifi_y",
    # "avg_scifi1_y",
    # "avg_scifi2_y",
    # "avg_scifi3_y",
    # "avg_scifi4_y",
]
            
  



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", dest="model_name", help="model name", default="testbeam_2024_GravNet_v2")
    parser.add_argument("-f", "--metadata_dir", dest="metadata_dir", help="metadata diretory", default="/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/")
    parser.add_argument("-c", "--cut", dest="cut", help="apply cut", default="nocut")
    parser.add_argument("-s", "--split", dest="split", help="split name")
    parser.add_argument("-t", "--threshold", dest="threshold", help="gnn score threshold", default=0.5)
    args = parser.parse_args()
    
    
    print(f"applying cut: {args.cut}")
    
    outdir = Path(f"./plots_FI_tmp/{args.cut}/{args.model_name}/")
    outdir.mkdir(parents=True, exist_ok=True)
    
    process_feature_importance(args, outdir)


    
    
if __name__ == "__main__":
    main()
