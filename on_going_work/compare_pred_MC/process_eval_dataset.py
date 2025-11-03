
import pandas as pd
import os


def drop_unwanted_files(
    df: pd.DataFrame,
    model_name: str,
    check_missing_column_name: str ,          # Column containing full paths to check existence
    train_set_column: str = 'digi_path',     # Column used to match against train metadata
    metadata_name: str = "",
    drop_muon_bkg: bool = True,
    drop_neutral_bkg: bool = True,
    drop_neutrino: bool = True,
) -> pd.DataFrame:
    """
    Drop rows that:
    1. Belong to one or more training sets
    2. Refer to missing files in `column_name`
    3. If vetoTagged is True, also check for corresponding vetoTagged files

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        model_name (str): Model name.
        column_name (str): Column with file paths to check existence.
        train_set_column (str): Column used to match entries in training metadata.
        metadata_name (str): Optional label for logging.
        drop_muon_bkg (bool): Whether to drop muon background files.
        drop_neutral_bkg (bool): Whether to drop neutral background files.
        drop_neutrino (bool): Whether to drop neutrino files.
        vetoTagged (bool): If True, replace 'vetoFree' with 'vetoTagged' in file paths and check both.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    
    if 'real_data' in metadata_name:
        return df
        
    # --- Step 1: Drop files used in training ---
    #print("--- Step 1: Drop files used in training ---")
    def load_train_files(metadata_name, model_name):
            
        if model_name == "baseline_muon":
            return '/afs/cern.ch/user/z/zhibin/eos/sndData/converted/combined_train.csv'
        else:
            if "neutrino" in metadata_name:
                name = 'neutrino'
            elif ("kaon" in metadata_name) or ("neutron" in metadata_name):
                name ='neutral_bkg'
            elif ("muon" in metadata_name):
                name = 'muon_bkg'
                
            return f"/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/training/{model_name}_{name}_split.csv"

    dropped_ids = set()

    path = load_train_files(metadata_name, model_name)
    if os.path.exists(path):
        train_df = pd.read_csv(path)
        if model_name=='baseline_muon':
            train_df["partition_number"] = train_df["file"].str.extract(r"Neutrinos/([^/]+)/").astype("Int64")
            current_ids = set(train_df["partition_number"].dropna().unique())
            train_set_column = 'partition'
            #print(f"Using column '{train_set_column}' for  {model_name}")
        else:
            current_ids = set(train_df[train_set_column].dropna().unique())
        dropped_ids.update(current_ids)
        print(f"{metadata_name}: Found {len(current_ids)} entries in training set")
    else:
        print(f"{metadata_name}: Training metadata  not found: {path}")
        
    if  (
            model_name != 'baseline_muon' or "neutrino" in metadata_name
        ):
        initial_len = len(df)
        df = df[~df[train_set_column].isin(dropped_ids)].copy()
        print(f"{metadata_name}: Dropped {initial_len - len(df)} entries from training sets")

    return df

def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data

def process_metadata():
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

    
    
    
    
    # options
    # metadata options
    model_name = 'baseline_muon' #'GravNet_v2' , 'baseline_muon'
    output_dir = f"./processed_metadata_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    check_missing_column_name = f'prediction_{model_name}_output_path'
    
    #print(metadata_vars)
    
    datasets_to_process = [
        ("MC_neutrino", metadata_vars["MC_neutrino_volTarget_100fb_1"]),
        ("MC_muon", pd.concat([
            metadata_vars["MC_muon_up"]
        ], ignore_index=True)),
        ("MC_kaon", metadata_vars["MC_kaon_FTFP_BERT_metadata_subset_csv"]),
        ("MC_neutron", metadata_vars["MC_neutron_FTFP_BERT_metadata_subset_csv"]),
        ("real_data_2024", metadata_vars["real_data_2024_skim_runs"]),
    ]

    results = {}
    for metadata_name, df in datasets_to_process:
        processed_df = drop_unwanted_files(
            df=df,
            metadata_name=metadata_name,
            model_name=model_name,
            check_missing_column_name=check_missing_column_name,
        )

        # Store results in dict
        results[metadata_name] = processed_df 
        # Save CSVs
        processed_df.to_csv(os.path.join(output_dir, f"{metadata_name}.csv"), index=False)

if __name__ == "__main__":
    process_metadata()