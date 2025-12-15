import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib


def get_geofile(subfolder,output_subfolder_name):
    run_number = int(subfolder.replace("run_", ""))
    if "24" in output_subfolder_name:
        if run_number in range(100956, 100976):
            return "geofile_sndlhc_H4_2024_Fe_1wall.root"
        elif run_number in range(100881, 100954):
            return "geofile_sndlhc_H4_2024_W_2walls.root"
    elif "23" in output_subfolder_name:
        if run_number in range(100651, 100664):
            return "geofile_sndlhc_H8_2023_1wall.root"
        elif run_number in range(100649, 100651) or run_number in range(100664, 100670):
            return "geofile_sndlhc_H8_2023_2walls.root"
        elif run_number in range(100630, 100649) or run_number in range(100670, 100680):
            return "geofile_sndlhc_H8_2023_3walls.root"
    return None


def process_real_data_subfolders(root_path, output_subfolder_name, data_type, geo_file=None):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in tqdm(os.listdir(root_path), desc=f"Processing run folder"):
        
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue
        
        if "README" in subfolder:
            print(f"Skipping README: {subfolder_path}")
            continue

        if "recalib" in subfolder:
            print(f"Skipping recalib: {subfolder_path}")
            continue

        current_geo_file = get_geofile(subfolder,output_subfolder_name)
        if not current_geo_file:
            print(f"No geo file found for {subfolder}. Skipping.")
            continue
        current_geo_file = os.path.join(root_path, current_geo_file)

        digi_file = ''
        partition = subfolder
        n_event = 0

        for file in tqdm(os.listdir(subfolder_path), desc=f"Files in {subfolder}", leave=False):
            file_path = os.path.join(subfolder_path, file)
            if file.endswith(".root"):
                
                try:
                    digi_file = file_path
                    root_file = ROOT.TFile(file_path)
                    if not root_file or root_file.IsZombie():
                        raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
                    tree = root_file.Get(tree_name)
                    n_event = tree.GetEntries() if tree else 0
                    root_file.Close()
                    
                except Exception as e:
                    print(f"Error processing ROOT file {file_path}: {e}")

            one_file_data = {
            'data_type': data_type,
            'subfolder': output_subfolder_name,
            'partition': partition,
            'n_event': n_event,
            'digi_path': digi_file,
            'geo_path': current_geo_file,
            }
            metadata.append(one_file_data)
        
            #print('digi_file: ', digi_file)

    metadata.sort(key=lambda x: x['partition'])
    return metadata


def find_best_matching_geo(root_filename, geo_files):
    if not geo_files:
        return None
    best_match = max(geo_files, key=lambda geo: difflib.SequenceMatcher(
        None, os.path.basename(root_filename), os.path.basename(geo)
    ).ratio())
    return best_match


def process_MC_data_since2024(root_path, subfolder, data_type):
    metadata = []
    for subfolder in tqdm(os.listdir(root_path), desc=f"Processing subfolders"):
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue

        particle_type = None
        if subfolder.endswith("11"):
            particle_type = "electron"
        elif subfolder.endswith("211"):
            particle_type = "pi+"
        elif subfolder.endswith("-211"):
            particle_type = "pi-"
        
        subfolder_metadata = process_MC_subfolders(subfolder_path, subfolder, data_type, particle_type)

        metadata.extend(subfolder_metadata)

    return metadata

            
def process_MC_subfolders(root_path, output_subfolder_name, data_type, particle_type=None):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)
        #print(subfolder_path)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue
        
        digi_file = None
        geo_file = None
        partition = subfolder
        n_event = 0
        
        if particle_type == None:
            if subfolder.endswith("11"):
                particle_type = "electron"
            elif subfolder.endswith("211"):
                particle_type = "pi+"
            elif subfolder.endswith("-211"):
                particle_type = "pi-"
        
        if subfolder == "X_neg38_Y_45_Z_315":
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                #print(subfolder_path)
                if not os.path.isdir(subsubfolder_path):
                    print(f"Skipping non-directory: {subsubfolder_path}")
                    continue
                
                for file in os.listdir(subsubfolder_path):
                    file_path = os.path.join(subsubfolder_path, file)
                    if file.endswith("digCPP.root"):
                        try:
                            digi_file = file_path
                            root_file = ROOT.TFile(file_path)
                            if not root_file or root_file.IsZombie():
                                raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
                            tree = root_file.Get(tree_name)
                            n_event = tree.GetEntries() if tree else 0
                            root_file.Close()
                            
                        except Exception as e:
                            print(f"Error processing ROOT file {file_path}: {e}")

                    elif file.startswith("geo"):
                        geo_file = file_path
                        
                if geo_file == None:
                    geo_file = os.path.join(root_path, "geofile_full.PG_211-TGeant4.root")

                one_file_data = {
                    'data_type': data_type,
                    # 'particle_type': particle_type,
                    'subfolder': output_subfolder_name,
                    'partition': partition + "/" + subsubfolder,
                    'n_event': n_event,
                    'digi_path': digi_file,
                    'geo_path': geo_file,
                }
                metadata.append(one_file_data)
        else:
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file.endswith("digCPP.root"):
                    try:
                        digi_file = file_path
                        root_file = ROOT.TFile(file_path)
                        if not root_file or root_file.IsZombie():
                            raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
                        tree = root_file.Get(tree_name)
                        n_event = tree.GetEntries() if tree else 0
                        root_file.Close()
                        
                    except Exception as e:
                        print(f"Error processing ROOT file {file_path}: {e}")

                elif file.startswith("geo"):
                    geo_file = file_path
                    
            if geo_file == None:
                geo_file = os.path.join(root_path, "geofile_full.PG_211-TGeant4.root")

            one_file_data = {
                'data_type': data_type,
                # 'particle_type': particle_type,
                'subfolder': output_subfolder_name,
                'partition': partition,
                'n_event': n_event,
                'digi_path': digi_file,
                'geo_path': geo_file,
            }
            metadata.append(one_file_data)
            #print('add: ', one_file_data)

    metadata.sort(key=lambda x: x['partition'])
    return metadata

def save_metadata_to_csv(metadata, csv_name):
    column_names = ['data_type', 'subfolder', 'partition', 'n_event', 'digi_path', 'geo_path']
    
    # Sort metadata as per the specified rules
    data_type = metadata[0].get('data_type')
    
    metadata.sort(key=lambda x: (x['subfolder'], x['partition']))
    
    try:
        file_exists = os.path.isfile(csv_name)
        with open(csv_name, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=column_names)
            
            # Write header only if the file doesn't already exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(metadata)
        
        print(f"Metadata updated to {csv_name}")
    except Exception as e:
        print(f"Error updating metadata to CSV: {e}")


def generate_real_data_path(data_type, root_path, subfolder, csv_file):    
    metadata = process_real_data_subfolders(root_path, subfolder, data_type)
    save_metadata_to_csv(metadata, csv_file)


def generate_MC_data_path(data_type, root_path, subfolder, csv_file):
    if "2024" in subfolder:
        metadata = process_MC_data_since2024(root_path, subfolder, data_type)
    else:
        metadata = process_MC_subfolders(root_path, subfolder, data_type)
    save_metadata_to_csv(metadata, csv_file)


def process_root_file(file_path, tree_name):
    """Process a ROOT file and retrieve the number of entries in the specified tree."""
    try:
        root_file = ROOT.TFile(file_path)
        if not root_file or root_file.IsZombie():
            raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
        
        tree = root_file.Get(tree_name)
        n_event = tree.GetEntries() if tree else 0
        root_file.Close()
        return n_event
    except Exception as e:
        print(f"Error processing ROOT file {file_path}: {e}")
        return 0

def simple_update_csv_file(csv_file, updated_data):
    """Save the updated DataFrame back to the CSV file."""
    updated_data.to_csv(csv_file, index=False)
    print(f"CSV file updated: {csv_file}")

    
def process(data_type, root_path, subfolder, csv_file):
    """
    Calls appropriate helper functions based on the data type.
    """
    if data_type == "real_data":
        generate_real_data_path(data_type, root_path, subfolder, csv_file)
    elif data_type == "MC_data":
        generate_MC_data_path(data_type, root_path, subfolder, csv_file)
    else:
        print(f"Unknown data type: {data_type}")

def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder



def main(args):
    
    config_path = f"{args.work_path}/snd-ml/testbeam/metadata/metadata_testbeam_config.yaml"
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
        file_exists = os.path.isfile(csv_file)
        if (not file_exists):
            print(f'{csv_file} does not exist, generating from raw data')
            process(data_type, root_path, subfolder, csv_file)
        elif (args.force_rerun):
            print(f'{csv_file} force rerun, generating from raw data')
            process(data_type, root_path, subfolder, csv_file)
        else:
            print(f'{csv_file} exist, skip generating from raw data')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    parser.add_argument("-w", "--work_path", dest="work_path", help="work path",required=True)
    args = parser.parse_args()
    main(args)

