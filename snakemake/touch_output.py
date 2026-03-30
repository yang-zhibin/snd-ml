import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ROOT

from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
raw_metadata_csv_list = [
    # "MC_muon_down_metadata.csv",
    # "MC_muon_horizontal_metadata.csv",
    # "MC_muon_up_metadata.csv",  # 2024 crossing angle

    # "MC_neutrino_volMuFilter_20fb-1_metadata.csv",
    # "MC_neutrino_volTarget_100fb-1_metadata.csv",

    # "MC_neutrino_2024_vm_metadata.csv",
    # "MC_neutrino_2024_ve_metadata.csv",

    # "MC_kaon_FTFP_BERT_metadata.csv",
    # "MC_neutron_FTFP_BERT_metadata.csv",

    # "MC_kaon_QGSP_BERT_HP_PEN_metadata.csv",
    # "MC_neutron_QGSP_BERT_HP_PEN_metadata.csv",

    # "real_data_2022_metadata.csv",
    # "real_data_2023_metadata.csv",
    # "real_data_2024_metadata.csv",
    # "real_data_2025_metadata.csv",
]

subset_metadata_csv_list = [
    # "MC_kaon_FTFP_BERT_metadata_subset.csv",
    # "MC_neutron_FTFP_BERT_metadata_subset.csv",

    "real_data_2024_skim_runs_metadata.csv",
    # "real_data_2024_skim_runs_metadata_subset.csv",
]

# -----------------------------------------------------------------------------
# User environment / switches
# -----------------------------------------------------------------------------
PERSONAL_WORK_SPACE = "/afs/cern.ch/work/z/zhibin/snd-ml"
# Set these as needed in your workflow
process_veto = False
process_model = "default_model"

# -----------------------------------------------------------------------------
# Metadata loading
# -----------------------------------------------------------------------------
def load_metadata():
    metadata_rootpath = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    metadata_files = [
        os.path.join(metadata_rootpath, file_name)
        for file_name in (raw_metadata_csv_list + subset_metadata_csv_list)
    ]
    metadata_files = [file for file in metadata_files if os.path.isfile(file)]

    if not metadata_files:
        print("Warning: No metadata files found.")
        return None
    elif len(metadata_files) == 1:
        return pd.read_csv(metadata_files[0])
    else:
        df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)
        return df


METADATA = load_metadata()

# -----------------------------------------------------------------------------
# Collect target paths from metadata
# -----------------------------------------------------------------------------
def collect_metadata_targets(target_name, after_model=False):
    if METADATA is None:
        return []

    # -------------------------
    # MODEL-RELATED TARGETS
    # -------------------------
    if after_model:
        process_model_set = {process_model} if isinstance(process_model, str) else set(process_model)
        target_paths = []

        for m in process_model_set:
            if target_name == "model_output":
                col = f"prediction_{m}_output_path"
            else:
                raise ValueError(f"Unknown model target: {target_name}")

            if col in METADATA.columns:
                paths = (
                    METADATA["output_base_path"].astype(str).str.rstrip("/") + "/" +
                    METADATA[col].astype(str).str.lstrip("/")
                )
                target_paths.extend(paths.tolist())

        return set([p for p in target_paths if str(p).lower() != "nan"])

    # -------------------------
    # NORMAL TARGETS
    # -------------------------
    def combine_paths(base, rel):
        valid = base.notna() & rel.notna()
        return (
            base[valid].astype(str).str.rstrip("/") + "/" +
            rel[valid].astype(str).str.lstrip("/")
        ).tolist()

    target_paths = []

    if target_name in METADATA.columns:
        target_paths.extend(combine_paths(METADATA["output_base_path"], METADATA[target_name]))

    if process_veto:
        col = f"veto_{target_name}"
        if col in METADATA.columns:
            target_paths.extend(combine_paths(METADATA["output_base_path"], METADATA[col]))

    return set(target_paths)

# -----------------------------------------------------------------------------
# Touch existing files only
# -----------------------------------------------------------------------------
def touch_existing_paths(paths, verbose=True, show_progress=True):
    touched = []
    skipped = []

    iterable = sorted(paths)
    if show_progress:
        iterable = tqdm(iterable, desc="Touching files", unit="file")

    for path_str in iterable:
        if not path_str or str(path_str).lower() == "nan":
            continue

        path = Path(path_str)

        if path.exists():
            path.touch()
            touched.append(str(path))
            if verbose and not show_progress:
                print(f"Touched: {path}")
        else:
            skipped.append(str(path))
            if verbose and not show_progress:
                print(f"Not found, skipped: {path}")

    return touched, skipped
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def touch_files():
    nueAnalysisFilter_path = collect_metadata_targets("nueAnalysisFilter_path", after_model=False)
    feature_path = collect_metadata_targets("feature_path", after_model=False)

    all_paths = set(nueAnalysisFilter_path) | set(feature_path)
    
    # print(all_paths)

    print(f"Found {len(nueAnalysisFilter_path)} nueAnalysisFilter_path entries")
    print(f"Found {len(feature_path)} feature_path entries")
    print(f"Total unique paths: {len(all_paths)}")

    touched, skipped = touch_existing_paths(
        all_paths,
        verbose=False,
        show_progress=True,
    )

    print("\nSummary")
    print(f"Touched files: {len(touched)}")
    print(f"Skipped missing files: {len(skipped)}")

def check_one_file(path, delete_bad_files=False):
    f = None
    try:
        if not os.path.exists(path):
            return ("missing_paths", path)

        f = ROOT.TFile.Open(path, "READ")
        if not f or f.IsZombie():
            if f:
                f.Close()
            return ("zombie_files", path)

        tree = f.Get("cutFlowSummary")

        if not tree:
            f.Close()
            f = None
            if delete_bad_files:
                try:
                    os.remove(path)
                    return ("deleted_missing_cutflow_tree", path)
                except Exception as e:
                    return ("failed_delete", (path, str(e)))
            return ("missing_cutflow_tree", path)

        if not tree.InheritsFrom("TTree"):
            f.Close()
            f = None
            if delete_bad_files:
                try:
                    os.remove(path)
                    return ("deleted_missing_cutflow_tree", path)
                except Exception as e:
                    return ("failed_delete", (path, str(e)))
            return ("missing_cutflow_tree", path)

        if not tree.GetBranch("species"):
            f.Close()
            f = None
            if delete_bad_files:
                try:
                    os.remove(path)
                    return ("deleted_missing_species_branch", path)
                except Exception as e:
                    return ("failed_delete", (path, str(e)))
            return ("missing_species_branch", path)

        f.Close()
        f = None
        return ("valid_paths", path)

    except Exception as e:
        return ("failed", (path, repr(e)))

    finally:
        if f:
            try:
                f.Close()
            except Exception:
                pass

def check_files(max_files=500, n_workers=1, use_parallel=True, delete_bad_files=True):
    # nueAnalysisFilter_path = collect_metadata_targets(
    #     "nueAnalysisFilter_path", after_model=False
    # )
    feature_path = collect_metadata_targets(
        "feature_path", after_model=False
    )

    #all_paths = list(set(nueAnalysisFilter_path) | set(feature_path))
    all_paths = list(set(feature_path))

    if max_files is not None:
        all_paths = list(islice(all_paths, max_files))

    valid_paths = []
    missing_paths = []
    missing_species_branch = []
    deleted_missing_species_branch = []
    deleted_missing_cutflow_tree = []
    missing_cutflow_tree = []
    zombie_files = []
    failed = []
    failed_delete = []

    results_map = {
        "valid_paths": valid_paths,
        "missing_paths": missing_paths,
        "missing_species_branch": missing_species_branch,
        "deleted_missing_species_branch": deleted_missing_species_branch,
        "deleted_missing_cutflow_tree": deleted_missing_cutflow_tree,
        "missing_cutflow_tree": missing_cutflow_tree,
        "zombie_files": zombie_files,
        "failed": failed,
        "failed_delete": failed_delete,
    }

    if use_parallel and len(all_paths) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(check_one_file, path, delete_bad_files)
                for path in all_paths
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Checking ROOT files",
            ):
                category, result = future.result()
                results_map[category].append(result)
    else:
        for path in tqdm(all_paths, desc="Checking ROOT files"):
            category, result = check_one_file(path, delete_bad_files)
            results_map[category].append(result)

    print(f"Checked files: {len(all_paths)}")
    print(f"Valid files: {len(valid_paths)}")
    print(f"Missing files: {len(missing_paths)}")
    print(f"Zombie/unreadable files: {len(zombie_files)}")
    print(f"Missing 'cutFlowSummary' tree: {len(missing_cutflow_tree)}")
    print(f"Deleted missing cutflow tree: {len(deleted_missing_cutflow_tree)}")
    print(f"Missing 'species' branch: {len(missing_species_branch)}")
    print(f"Deleted missing 'species' branch files: {len(deleted_missing_species_branch)}")
    print(f"Failed checks: {len(failed)}")
    print(f"Failed deletes: {len(failed_delete)}")

    
    
    return {
        "valid_paths": valid_paths,
        "missing_paths": missing_paths,
        "missing_species_branch": missing_species_branch,
        "deleted_missing_species_branch": deleted_missing_species_branch,
        "deleted_missing_cutflow_tree": deleted_missing_cutflow_tree,
        "missing_cutflow_tree": missing_cutflow_tree,
        "zombie_files": zombie_files,
        "failed": failed,
        "failed_delete": failed_delete,
    }


if __name__ == "__main__":
    # touch_files()
    
    check_files(
        max_files=None,
        n_workers=8,
        use_parallel=True,
        delete_bad_files=True,
    )