import pandas as pd
import ROOT
import os
from tqdm import tqdm

def check_and_delete_missing_scifi(metadata_csv,
                                   path_column="vetoFree_feature_path",
                                   branch="count_scifi",
                                   delete=True):
    """
    Check each ROOT file for the given branch and delete file if missing.
    Uses tqdm for progress bar and prints detailed info for each file.
    """

    df = pd.read_csv(metadata_csv)

    total = len(df)
    missing = 0
    deleted_files = []

    for idx, row in tqdm(df.iterrows(), total=total, desc="Checking files"):
        path = row[path_column]

        tqdm.write(f"\n[{idx+1}/{total}] Checking: {path}")

        # Check existence
        if not os.path.isfile(path):
            tqdm.write("  ❌ File not found.")
            missing += 1
            deleted_files.append((path, "FILE_NOT_FOUND"))
            continue

        # Try opening ROOT file
        f = ROOT.TFile.Open(path, "READ")
        if not f or f.IsZombie():
            tqdm.write("  ❌ Zombie or unreadable ROOT file.")
            missing += 1
            deleted_files.append((path, "ZOMBIE_FILE"))
            if delete:
                tqdm.write("  → Deleting file.")
                os.remove(path)
            continue

        # Check tree
        tree = f.Get("sndData")
        if not tree:
            tqdm.write("  ❌ Tree 'features' not found.")
            missing += 1
            deleted_files.append((path, "NO_FEATURES_TREE"))
            f.Close()
            if delete:
                tqdm.write("  → Deleting file.")
                os.remove(path)
            continue

        # Check branch
        if not tree.GetListOfBranches().FindObject(branch):
            tqdm.write(f"  ❌ Branch '{branch}' missing.")
            missing += 1
            deleted_files.append((path, "BRANCH_MISSING"))
            f.Close()
            if delete:
                tqdm.write("  → Deleting file.")
                os.remove(path)
            continue

        f.Close()
        tqdm.write("  ✅ OK — branch exists.")
        
        # if idx >1000:
        #     break
        

    # Summary
    print("\n=== SciFi Branch Check Summary ===")
    print(f"Total files:        {total}")
    print(f"Missing/deleted:    {missing}")
    print(f"OK:                 {total - missing}")

    return {
        "total_files": total,
        "missing_or_deleted": missing,
        "ok": total - missing,
        "deleted_files": deleted_files,
    }


if __name__ == "__main__":
    metadata_csv = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv'
    check_and_delete_missing_scifi(metadata_csv)
    