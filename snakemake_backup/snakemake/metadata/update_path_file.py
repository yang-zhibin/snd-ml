import os
import csv
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser

def generate_full_path(row, path_name, suffix, csv_input, eos_root_path):
    
    base = f"{path_name}_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}"
    
    if 'muon' in csv_input:
        filename = f"{base}_{row['n_event']}{suffix}"
    elif 'real_data' in csv_input:
        unique_file_id = os.path.splitext(os.path.basename(row['digi_path']))[0]
        filename = f"{base}_{unique_file_id}{suffix}"
    else:
        filename = f"{base}{suffix}"
    
    full_path = f"{eos_root_path}/{row['data_type']}/{row['subfolder']}/{row['partition']}/{filename}"
    return full_path


def add_new_path(path_name, path_type, df, csv_input, eos_root_path, force_rerun=True):
    # Determine column name
    column_name = f"{path_name}_path"
    
    # Check if the column already exists
    if (column_name in df.columns) and (not force_rerun):
        print(f"The column '{column_name}' already exists in the CSV file.")
        return df
    
    # Ensure 'subfolder' column is treated as string
    df['subfolder'] = df['subfolder'].astype(str)
    
    # Determine suffix based on path_type
    suffix = ".pt.gz" if path_type == "pt" else ".root"
    
    df[column_name] = df.apply(lambda row: generate_full_path(row, path_name, suffix, csv_input, eos_root_path),axis=1)
    
    print(f"The column '{column_name}' has been added")
    return df


def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder


def drop_0_event_row(csv_file):

    # Read the CSV file
    data = pd.read_csv(csv_file)
    # Drop rows where 'n_event' is 0
    filtered_data = data[data['n_event'] != 0]
    # Save the updated CSV back
    filtered_data.to_csv(csv_file, index=False)
    rm_rows = len(data) - len(filtered_data)
    print(f" {rm_rows} rows with n_event == 0 have been removed from {csv_file}.")

def drop_run_without_lumi(df, lumi_file):
    lumi_df = pd.read_csv(lumi_file)
    df['run'] = df['partition'].str.extract(r'run_(\d+)').astype(int)

    df = df[df["run"].isin(lumi_df["run"])]

    # Create a mapping from run_id to lumi
    lumi_map = dict(zip(lumi_df['run'], lumi_df['lumi']))

    # Compute lumi per file
    lumi_per_file_list = []
    for run_id, run_df in df.groupby('run'):
        total_events = run_df['n_event'].sum()
        run_lumi = lumi_map.get(run_id, 0)
        lumi_per_file = run_df['n_event'] * run_lumi / total_events
        lumi_per_file_list.append(lumi_per_file)

    # Flatten the list and assign back
    df['lumi_per_file'] = pd.concat(lumi_per_file_list)

    return df
    

def update_csv_file(args, data_type, root_path, subfolder, csv_output, csv_input, eos_root_path, models,lumi_file):
    
    df = pd.read_csv(csv_input)
    if (data_type=='real_data'):
        df = drop_run_without_lumi(df, lumi_file)
    

    # hit path
    df = add_new_path("hit","root", df, csv_input,  eos_root_path)
    # feature path
    df = add_new_path("feature","root", df, csv_input, eos_root_path)
    # pt hit path
    df = add_new_path("pt_hit","pt", df, csv_input, eos_root_path)

    # 
    model_names = [list(model.keys())[0] for model in models]
    for model in model_names:
        df = add_new_path(f"model_{model}_output", "root", df, csv_input, eos_root_path)
        df = add_new_path(f"eval_{model}", "root", df, csv_input, eos_root_path)
    

    df.to_csv(csv_output, index=False)


def main(args):
    
    config_path =args.config
    model_config =  args.model
    eos_root_path = args.root_path
    lumi_file = args.lumi
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    with open(model_config, "r") as file:
        models = yaml.safe_load(file)

    csv_input = args.csv_input
    csv_output = args.csv_output
    csv_name = os.path.basename(csv_output)
    
    data_type, particle_subfolder = extract_info(csv_name)
    print(data_type, particle_subfolder)

    data_paths = config.get(data_type, {})
    for data_path in data_paths:
        root_path = data_path['root_path']
        subfolder = data_path['subfolder']
        if (particle_subfolder != subfolder):
            print(f"skip {subfolder} ")
            continue
        print(data_path)
        update_csv_file(args, data_type, root_path, subfolder, csv_output, csv_input, eos_root_path, models,lumi_file)
        drop_0_event_row(csv_output)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--csv_input", dest="csv_input", help="csv input", required=True)
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    parser.add_argument("-r", "--root_path", dest="root_path", help="root path of the output file of the workflow", default='/eos/experiment/sndlhc/users/zhibin')
    parser.add_argument("-c", "--config", dest="config", help="metadata config file path", default='/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/metadata_config.yaml')
    parser.add_argument("-m", "--model", dest="model", help="model config file path", default='/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/model_config.yaml')
    parser.add_argument("-l", "--lumi", dest="lumi", help="lumi record file path", default='/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/SND_lumi.csv')
    args = parser.parse_args()
    main(args)

