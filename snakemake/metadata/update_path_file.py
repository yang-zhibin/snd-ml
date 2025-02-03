import os
import csv
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser

def add_new_path(path_name, path_type, csv_file, eos_root_path, force_rerun=True):
    """
    Adds a new column to the CSV file based on the path_name and path_type.

    Args:
        path_name (str): The name of the new path or prefix column to add.
        path_type (str): Either "path" or "prefix" to determine the suffix.
        csv_file (str): The path to the CSV file to be updated.
        eos_root_path (str): The root directory where the files are stored.
        force_rerun (bool): If True, the column will be overwritten if it exists.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Determine column name
    column_name = f"{path_name}_{path_type}"
    
    # Check if the column already exists
    if (column_name in df.columns) and (not force_rerun):
        print(f"The column '{column_name}' already exists in the CSV file.")
        #return
    
    # Ensure 'subfolder' column is treated as string
    df['subfolder'] = df['subfolder'].astype(str)
    
    # Determine suffix based on path_type
    suffix = "_chunk" if path_type == "prefix" else ".root"
    
    # Generate paths or prefixes
    if 'muon' in csv_file:
        df[column_name] = df.apply(
            lambda row: f"{path_name}_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}_{row['n_event']}{suffix}", axis=1
        )
    else:
        df[column_name] = df.apply(
            lambda row: f"{path_name}_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}{suffix}", axis=1
        )
    
    df[column_name] = df.apply(
        lambda row: f"{eos_root_path}/{row['data_type']}/{row['subfolder']}/{row['partition']}/{row[column_name]}", axis=1
    )
    
    df.to_csv(csv_file, index=False)
    print(f"The column '{column_name}' has been added and the file '{csv_file}' updated successfully.")


def update_csv_file(args, data_type, root_path, subfolder, csv_file):
    eos_root_path = '/eos/user/z/zhibin/sndData'
    pass
    # #hit path
    add_new_path("hit","path", csv_file, eos_root_path)
    # #feature path
    add_new_path("feature","path", csv_file, eos_root_path)
    # #pt prefix
    add_new_path("pt","prefix", csv_file, eos_root_path)

def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder



def main(args):
    
    config_path = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/metadata_config.yaml"
    metadata_rootoath = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    csv_file = args.csv_output
    csv_name = os.path.basename(csv_file)
    
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
        csv_file = f'/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/{data_type}_{subfolder}_metadata.csv'
        file_exists = os.path.isfile(csv_file)
        if (not file_exists) and (not args.force_rerun):
            print(f'{csv_file} does not exist, please generate from raw data first')
        else:
            print(f'{csv_file} exist, updating csv file')
            update_csv_file(args, data_type, root_path, subfolder, csv_file)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun", dest="force_rerun", help="force rerun", default=False)
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    args = parser.parse_args()
    main(args)

