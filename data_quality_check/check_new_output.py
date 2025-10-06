import os
import math
import pandas as pd
import ROOT
from tqdm import tqdm

# --- config ---
DEFAULT_METADATA_DIR = "/afs/cern.ch/work/z/zhibin/snd-ml/evaluation/compare_pred_MC/processed_metadata"
MODEL_NAME = "baseline_muon"  
TREE_NAME = "sndData"
BRANCH_NAME = "pred_class_first"
PATH_COL_TEMPLATE = f"vetoTagged_prediction_{MODEL_NAME}_output_path"

def read_metadata(directory: str = DEFAULT_METADATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all processed metadata CSVs into a dict keyed by filename (without .csv)."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file[:-4]
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict

def has_branch(root_path: str, tree_name: str, branch_name: str) -> bool:
    """
    Return True if the ROOT file at root_path has a TTree `tree_name` that contains `branch_name`.
    """
    if not os.path.exists(root_path):
        return False
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        if f: f.Close()
        return False
    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return False
    # Check via branches/leaves/explicit GetBranch (covers aliases in many cases)
    exists = bool(
        tree.GetBranch(branch_name) or
        (tree.GetListOfBranches() and tree.GetListOfBranches().FindObject(branch_name)) or
        (tree.GetListOfLeaves() and tree.GetListOfLeaves().FindObject(branch_name))
    )
    f.Close()
    return exists

def compute_int_lumi(metadata_df: pd.DataFrame,
                     model_name: str = MODEL_NAME,
                     tree_name: str = TREE_NAME,
                     branch_name: str = BRANCH_NAME,
                     path_col_template: str = PATH_COL_TEMPLATE) -> dict:
    """
    Iterate rows; if the referenced ROOT file has the branch, add lumi_per_file to the sum.
    Returns a small report dict.
    """
    col = path_col_template.format(model=model_name)
    if col not in metadata_df.columns:
        raise KeyError(f"Column '{col}' not found in metadata_df.")

    total_lumi = 0.0
    n_rows = len(metadata_df)
    n_checked = 0
    n_with_branch = 0
    n_missing_files = 0
    n_bad_root = 0

    for _, row in tqdm(metadata_df.iterrows(), total=n_rows, desc="Checking ROOT files"):
        root_path = row.get(col, None)
        if not isinstance(root_path, str) or not root_path:
            n_missing_files += 1
            continue

        n_checked += 1
        try:
            if has_branch(root_path, tree_name, branch_name):
                n_with_branch += 1
                lumi = row.get("lumi_per_file", float("nan"))
                # Accept both numeric and string; skip NaN or non-finite
                try:
                    lumi_val = float(lumi)
                except Exception:
                    lumi_val = float("nan")
                if not (math.isnan(lumi_val) or math.isinf(lumi_val)):
                    total_lumi += lumi_val
            # else: file exists but branch missing -> just skip adding lumi
        except Exception:
            # Consider this a bad ROOT file / unreadable entry
            n_bad_root += 1

    return {
        "model_name": model_name,
        "tree": tree_name,
        "branch": branch_name,
        "rows_in_metadata": n_rows,
        "rows_checked": n_checked,
        "files_with_branch": n_with_branch,
        "missing_or_empty_path": n_missing_files,
        "unreadable_root": n_bad_root,
        "integrated_lumi_sum": total_lumi,
    }

def main():
    metadata_dict = read_metadata()
    # pick your table — you used 'real_data_2024' in your snippet
    metadata_df = metadata_dict["real_data_2024"]

    report = compute_int_lumi(
        metadata_df=metadata_df,
        model_name=MODEL_NAME,
        tree_name=TREE_NAME,
        branch_name=BRANCH_NAME,
        path_col_template=PATH_COL_TEMPLATE,
    )

    # Print a concise report
    print("=== Integrated Luminosity Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()