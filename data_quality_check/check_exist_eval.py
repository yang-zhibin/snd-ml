import pandas as pd
import os

def check_file_exist(path):
    return int(os.path.exists(path))

def main():
    # Load metadata
    metadata = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv")
    #metadata['lumi_per_file'] = 100 for MC neutrion
    # Initialize DataFrame to collect existence data
    exist_records = []

    for idx, row in metadata.iterrows():
        model_path = row["model_baseline_muon_output_path"]
        eval_path = row["eval_baseline_muon_output_path"]
        file_lumi = row["lumi_per_file"]

        model_exists = check_file_exist(model_path)
        eval_exists = check_file_exist(eval_path)

        exist_records.append({
            "model_path_exist": model_exists,
            "eval_path_exist": eval_exists,
            "file_lumi": file_lumi
        })

    # Convert to DataFrame
    exist_df = pd.DataFrame(exist_records)

    # Calculate total lumi and percentage for model path
    
    total_lumi = exist_df["file_lumi"].sum()

    model_exist_lumi = exist_df.loc[exist_df["model_path_exist"] == 1, "file_lumi"].sum()
    eval_exist_lumi = exist_df.loc[exist_df["eval_path_exist"] == 1, "file_lumi"].sum()

    model_exist_percent = 100 * model_exist_lumi / total_lumi
    eval_exist_percent = 100 * eval_exist_lumi / total_lumi

    print(f"Total Lumi: {total_lumi}")
    print(f"Model Path Exist Lumi: {model_exist_lumi} ({model_exist_percent:.2f}%)")
    print(f"Eval Path Exist Lumi: {eval_exist_lumi} ({eval_exist_percent:.2f}%)")


if __name__ == "__main__": 
    main()
