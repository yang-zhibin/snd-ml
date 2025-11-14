import os
import pandas as pd
import ROOT
from collections import defaultdict
from tqdm import tqdm
import numpy as np

ROOT.gROOT.SetBatch(True)

PDG_LABELS = {
    12:  "ve",    112:  "ve-NC",
    14:  "vm",    114:  "vm-NC",
    16:  "vt",    116:  "vt-NC",
    -12: "anti-ve",  -112: "anti-ve-NC",
    -14: "anti-vm",  -114: "anti-vm-NC",
    -16: "anti-vt",  -116: "anti-vt-NC",
}
PDG_LABELS_INV = {v: k for k, v in PDG_LABELS.items()}

TARGET_LUMI_FB = 150.0  # fb⁻¹


def _code_to_neu_cc(code: int):
    """
    Map your 12/14/16 and 112/114/116 (and negatives) codes to (neu, cc),
    where cc == 1 for CC and cc == 0 for NC.
    """
    # CC cases
    if code in (12, 14, 16, -12, -14, -16):
        return code, 1
    # NC cases: 112 -> 12 (cc=0), etc.
    nc_map = {
        112: 12, 114: 14, 116: 16,
        -112: -12, -114: -14, -116: -16,
    }
    if code in nc_map:
        return nc_map[code], 0
    raise ValueError(f"Unexpected code {code}; expected 12/14/16 (±) or 112/114/116 (±).")


def _detect_pdg_branch(rdf):
    """Return which PDG code branch exists: pdgCode or pdgcode."""
    cols = set(rdf.GetColumnNames())
    if "pdgCode" in cols:
        return "pdgCode"
    if "pdgcode" in cols:
        return "pdgcode"
    raise RuntimeError("No 'pdgCode' or 'pdgcode' branch found in tree.")


def _counts_for_file(root_path, tree_name, codes):
    """Count number of entries with given pdgCode for each code."""
    if not os.path.exists(root_path):
        return {c: 0 for c in codes}

    rdf = ROOT.RDataFrame(tree_name, root_path)
    pdg_branch = _detect_pdg_branch(rdf)

    results = {}
    for code in codes:
        n = int(rdf.Filter(f"{pdg_branch} == {code}").Count().GetValue())
        results[code] = n
    return results


def _aggregate_dataset(csv_path, tree_name, codes):
    """Aggregate normalized yields directly looping over pandas DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    #print(df.columns)

    path_col = next((c for c in df.columns if "preselect_path" in c), None)
    lumi_col = next((c for c in df.columns if "lumi_per_file" in c), None)

    if path_col is None or lumi_col is None:
        raise ValueError(f"Could not find path/lumi columns in {csv_path}")


    raw_total = defaultdict(int)
    total_sim_lumi_fb = 0.0

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        path = row[path_col]
        #n_event = row['n_events']
        lumi_fb = row[lumi_col]
        try:
            lumi_fb = float(lumi_fb)
        except Exception:
            continue
        if lumi_fb <= 0 or not isinstance(path, str) or not path.endswith(".root"):
            continue

        counts = _counts_for_file(path, tree_name, codes)
        # print(path)
        # print(counts)
        for code, n in counts.items():
            raw_total[code] += n
        total_sim_lumi_fb += lumi_fb
        
        if i>=100:
            break

    yields_150 = {c: raw_total[c] /total_sim_lumi_fb * TARGET_LUMI_FB  for c in codes}

    return {
        "yields_150fb": yields_150,
        "raw_total": dict(raw_total),
        "total_sim_lumi_fb": total_sim_lumi_fb,
    }


def _pretty_with_labels(d):
    """Attach readable PDG labels."""
    return {PDG_LABELS.get(k, str(k)): v for k, v in d.items()}


def _print_summary(title, results):
    """Print yields nicely using pandas DataFrames."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"Total luminosity of used files: {results['total_sim_lumi_fb']:.2f} fb⁻¹\n")

    df = pd.DataFrame({
        #"Raw Counts": _pretty_with_labels(results["raw_total"]),
        f"Yields @ {TARGET_LUMI_FB} fb⁻¹": _pretty_with_labels(results["yields_150fb"]),
    })
    print(df.to_string(float_format="%.3f"))


def _counts_for_gst_file(root_path: str, codes):
    """
    Count events per 'code' using gst tree with columns neu, cc.
    For each code, apply Filter(f"neu==X && cc==Y").Count().
    """
    rdf = ROOT.RDataFrame("gst", root_path)
    # quick branch sanity check (optional; will throw if missing anyway)
    cols = set(rdf.GetColumnNames())
    if "neu" not in cols or "cc" not in cols:
        raise RuntimeError(f"'neu' and/or 'cc' branches not found in {root_path}")

    results = {}
    for code in codes:
        neu_val, cc_val = _code_to_neu_cc(code)
        n = int(rdf.Filter(f"neu == {neu_val} && cc == {cc_val}").Count().GetValue())
        results[code] = n
    return results


def _aggregate_dataset_2022_up(codes):
    """
    Scan the Genie UP directory for .gst.root files, treat each file as lumi_per_file=100,
    count per flavour via (neu,cc) logic, and return per_fb / 150 fb^-1-normalized yields.
    """
    root_dir = '/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_up_volTarget_100fb-1_SNDG18_02a_01_000/'
    lumi_per_file_fb = 100.0  # as specified

    # Collect all .gst.root files recursively
    gst_files = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".gst.root"):
                gst_files.append(os.path.join(r, f))

    if not gst_files:
        raise FileNotFoundError(f"No .gst.root files found under: {root_dir}")

    raw_total = defaultdict(int)
    total_sim_lumi_fb = 0.0

    for fpath in tqdm(gst_files, desc="Processing 2022 UP .gst.root", unit="file"):
        # per-file counts
        counts = _counts_for_gst_file(fpath, codes)
        for code, n in counts.items():
            raw_total[code] += n
        total_sim_lumi_fb += lumi_per_file_fb

        # if total_sim_lumi_fb > 1:
        #     break
        
    yields_150 = {c: raw_total[c] /total_sim_lumi_fb * TARGET_LUMI_FB for c in codes}

    return {
        "yields_150fb": yields_150,
        "raw_total": dict(raw_total),
        "total_sim_lumi_fb": total_sim_lumi_fb,
    }

    

def get_neutrino_yields_from_zhibin():
    neutrino_MC_2022_down_csv = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    
    ve_MC_2024_csv       = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_2024_ve_metadata.csv'
    vm_MC_2024_csv       = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_2024_vm_metadata.csv'

    tree_name = "sndData"

    # 0) 2022 up
    codes_2022 = [14, -14, 12, -12, 16, -16, 114, -114, 112, -112, 116, -116]
    res_2022_up = _aggregate_dataset_2022_up(codes_2022)
    _print_summary("2022 Neutrino up", res_2022_up)

    
    # 1) 2022 down
    codes_2022 = [14, -14, 12, -12, 16, -16, 114, -114, 112, -112, 116, -116]
    res_2022_down = _aggregate_dataset(neutrino_MC_2022_down_csv, tree_name, codes_2022)
    _print_summary("2022 Neutrino Down", res_2022_down)
    
    
    # 2) 2024 νe dataset
    codes_ve_2024 = [12, -12, 112, -112]
    res_ve_2024 = _aggregate_dataset(ve_MC_2024_csv, tree_name, codes_ve_2024)
    _print_summary("2024 νe Dataset", res_ve_2024)

    # 3) 2024 νμ dataset
    codes_vm_2024 = [14, -14, 114, -114]
    res_vm_2024 = _aggregate_dataset(vm_MC_2024_csv, tree_name, codes_vm_2024)
    _print_summary("2024 νμ Dataset", res_vm_2024)

    return {
        "res_2022_up": res_2022_up,
        "res_2022_down": res_2022_down,
        "2024_ve": res_ve_2024,
        "2024_vm": res_vm_2024,
    }
    

def save_yield_table(results_dict, name, outdir="./yields_table"):
    """Save per-dataset yield summary to CSV."""
    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame({
        "Raw Counts": results_dict["raw_total"],
        "Yields @150 fb⁻¹": results_dict["yields_150fb"],
    }).fillna(0)

    if "PDG_LABELS" in globals():
        df.index = [PDG_LABELS.get(k, str(k)) for k in df.index]

    csv_path = os.path.join(outdir, f"{name}_yields.csv")
    df.to_csv(csv_path, float_format="%.6f")
    print(f"✅ Saved {csv_path}")



def _read_yields_series(csv_path):
    """
    Read one yields CSV and return a Series indexed by PDG code (ints),
    values = yields at 150 fb^-1.
    Handles indices saved as flavour labels or numeric codes.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, index_col=0)

    # Find the "Yields" column (robust to exact wording)
    ycol = next((c for c in df.columns if "Yields" in c), None)
    if ycol is None:
        raise ValueError(f"No 'Yields' column found in {csv_path}. Columns: {list(df.columns)}")

    s = df[ycol]

    # Map index (labels) back to integer PDG codes if needed
    def idx_to_code(x):
        if isinstance(x, (int, float)) and not pd.isna(x):
            return int(x)
        xs = str(x)
        if xs in PDG_LABELS_INV:
            return PDG_LABELS_INV[xs]
        try:
            return int(xs)
        except Exception:
            # Fallback: keep as string (shouldn't happen if your CSVs used PDG_LABELS)
            return xs

    s.index = [idx_to_code(ix) for ix in s.index]
    return s


def load_results_from_csvs(outdir="./yields_table"):
    """
    Load the four per-dataset CSVs and return a results-like dict containing only 'yields_150fb'.
    """
    up_s   = _read_yields_series(os.path.join(outdir, "2022_up_yields.csv"))
    down_s = _read_yields_series(os.path.join(outdir, "2022_down_yields.csv"))
    ve_s   = _read_yields_series(os.path.join(outdir, "2024_ve_yields.csv"))
    vm_s   = _read_yields_series(os.path.join(outdir, "2024_vm_yields.csv"))

    results = {
        "res_2022_up":   {"yields_150fb": up_s.to_dict()},
        "res_2022_down": {"yields_150fb": down_s.to_dict()},
        "2024_ve":       {"yields_150fb": ve_s.to_dict()},
        "2024_vm":       {"yields_150fb": vm_s.to_dict()},
    }
    return results


def combine_neutrino_yields(results, outdir="./yields_table"):
    """Combine yields with NaNs preserved, plus ratios:
       (2022 up / 2022 down) and (2024 up / 2022 down)."""
    # allow either key casing for "down"
    down_dict = (results.get("res_2022_down") or results.get("res_2022_Down"))["yields_150fb"]
    up_dict   = results["res_2022_up"]["yields_150fb"]
    ve_dict   = results["2024_ve"]["yields_150fb"]
    vm_dict   = results["2024_vm"]["yields_150fb"]
    
    print(ve_dict)
    print(vm_dict)
    print(up_dict)

    # 2024 up = ve + vm (aligned to up's keys; others may be NaN)
    codes = list(up_dict.keys())
    combined_2024 = {c: ve_dict.get(c, 0) + vm_dict.get(c, 0) for c in codes}

    print(combined_2024)
    # Build dataframe in requested column order; keep NaN (no fillna)
    df = pd.DataFrame({
        "2022 down": pd.Series(down_dict),
        "2022 up":   pd.Series(up_dict),
        "2024 up":   pd.Series(combined_2024),
    })

    # Reindex rows to follow the 2022 up keys order explicitly
    df = df.reindex(codes)

    # Ratios (keep NaN if denominator is NaN or 0)
    denom = df["2022 down"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio (2022 up / 2022 down)"] = df["2022 up"] / denom
        df["ratio (2024 up / 2022 down)"] = df["2024 up"] / denom

    # Replace inf with NaN (division by zero case)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Optional: map PDG codes to labels
    if "PDG_LABELS" in globals():
        df.index = [PDG_LABELS.get(k, str(k)) for k in df.index]

    os.makedirs(outdir, exist_ok=True)
    combined_csv_path = os.path.join(outdir, "combined_yields.csv")
    df.to_csv(combined_csv_path, float_format="%.6f")

    print(f"\n✅ Combined yields saved to: {combined_csv_path}\n")
    print(df.to_string(float_format="%.3f"))
    return df



def main():
    # # 1️⃣ Compute yields
    # print("=== Step 1: Reading neutrino yields ===")
    # results = get_neutrino_yields_from_zhibin()

    # # 2️⃣ Save individual CSV tables
    # print("\n=== Step 2: Saving individual yield tables ===")
    # save_yield_table(results["res_2022_up"], "2022_up")
    # save_yield_table(results["res_2022_down"], "2022_down")
    # save_yield_table(results["2024_ve"], "2024_ve")
    # save_yield_table(results["2024_vm"], "2024_vm")
    
    
    results = load_results_from_csvs()

    # 3️⃣ Combine yields and print summary
    print("\n=== Step 3: Combining yields ===")
    combine_neutrino_yields(results)


# get my owen table
# get LOI table
# get Technical proposal table
# get Edurad calcualtion
# compare tables

if __name__ == "__main__":
    main()