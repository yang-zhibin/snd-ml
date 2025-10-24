
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import ROOT as r
import os


def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data



def check_root_file(path: Path, tree_name="sndData", branch_name="pred_class_first"):
    """
    Returns a dict with booleans and a status string describing what was found.
    """
    result = {
        "exists": False,
        "root_opened": False,
        "has_tree": False,
        "has_branch": False,
        "entries": None,
        "status": "",
    }

    if not path.exists():
        result["status"] = "missing_file"
        return result

    result["exists"] = True

    f = r.TFile.Open(str(path))
    if not f or f.IsZombie():
        result["status"] = "file_not_openable_or_zombie"
        # ensure proper cleanup if ROOT gave us a handle
        try:
            if f:
                f.Close()
        except Exception:
            pass
        return result

    result["root_opened"] = True

    tree = f.Get(tree_name)
    if not tree:
        result["status"] = "missing_tree"
        f.Close()
        return result

    result["has_tree"] = True

    # Check branch
    has_branch = bool(tree.GetBranch(branch_name))
    if not has_branch:
        print(path)
    result["has_branch"] = has_branch
    try:
        result["entries"] = int(tree.GetEntries())
    except Exception:
        result["entries"] = None

    result["status"] = "ok" if has_branch else "missing_branch"
    f.Close()
    return result

def main():
    mc_files = [
        "MC_kaon_FTFP_BERT_metadata_subset.csv",
        "MC_neutron_FTFP_BERT_metadata_subset.csv",
        "MC_muon_down_metadata.csv",
        "MC_muon_horizontal_metadata.csv",
        "MC_muon_up_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
        "real_data_2024_skim_runs_metadata.csv",
    ]
    
    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)

    # Read CSV into a DataFrame
    df = metadata_vars['real_data_2024_skim_runs']
    model_name = "baseline_muon"
    
    col = f"vetoFree_prediction_{model_name}_output_path"
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")
    
    results = []
    # Loop through rows with tqdm
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        if i>1000:
           break
        path_str = row[col]
        p = Path(path_str)
        info = check_root_file(p, tree_name="sndData", branch_name="pred_class_first")

        results.append({
            "path": str(p),
            "exists": info["exists"],
            "root_opened": info["root_opened"],
            "has_tree": info["has_tree"],
            "has_branch": info["has_branch"],
            "entries": info["entries"],
            "status": info["status"],
        })

    res_df = pd.DataFrame(results)

    # High-level statistics
    stats = {
        "total": len(res_df),
        "missing_file": (res_df["status"] == "missing_file").sum(),
        "file_not_openable_or_zombie": (res_df["status"] == "file_not_openable_or_zombie").sum(),
        "missing_tree": (res_df["status"] == "missing_tree").sum(),
        "missing_branch": (res_df["status"] == "missing_branch").sum(),
        "ok": (res_df["status"] == "ok").sum(),
        "entries_eq_0": ((res_df["entries"] == 0) & (res_df["status"] == "ok")).sum(),
    }

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"{k:28s}: {v}")

    # Optional: show list of zero-entry files
    zero_entries = res_df[(res_df["entries"] == 0) & (res_df["status"] == "ok")]
    if not zero_entries.empty:
        print("\nFiles with 0 entries:")
        for p in zero_entries["path"]:
            print("  ", p)
            
    # Optional: save detailed report
    res_df.to_csv("vetoFree_predictions_check_report.csv", index=False)
    print("Saved: vetoFree_predictions_check_report.csv")
        
        
# check tree
# check branch
# 


if __name__ == "__main__":
    main()