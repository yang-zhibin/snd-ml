import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib


def process_real_data_since2024(root_path, subfolder, data_type):
    metadata = []
    for subfolder in tqdm(os.listdir(root_path), desc=f"Processing subfolders"):
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue
        

        subfolder_metadata = process_real_data_subfolders(subfolder_path, subfolder, data_type)

        metadata.extend(subfolder_metadata)

    return metadata

def process_real_data_subfolders(root_path, output_subfolder_name, data_type, geo_file=None):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in tqdm(os.listdir(root_path), desc=f"Processing run folder"):
        
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue

        current_geo_file = geo_file or get_geo_file(subfolder_path)
        if not current_geo_file:
            print(f"No geo file found for {subfolder}. Skipping.")
            continue

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
                    n_event = tree.GetEntriesFast() if tree else 0
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


def process_MC_subfolders(root_path, output_subfolder_name, data_type):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in tqdm(os.listdir(root_path), desc="Processing subfolders"):
        subfolder_path = os.path.join(root_path, subfolder)
        #print(subfolder_path)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue

        digi_file = None
        fallback_digi_file = None
        geo_file = None
        partition = subfolder
        n_event = 0

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if file.endswith("20240126_digCPP.root"):
                try:
                    digi_file = file_path
                    root_file = ROOT.TFile(file_path)
                    if not root_file or root_file.IsZombie():
                        raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
                    tree = root_file.Get(tree_name)
                    n_event = tree.GetEntriesFast() if tree else 0
                    root_file.Close()
                    
                except Exception as e:
                    print(f"Error processing ROOT file {file_path}: {e}")
            
            elif (file.endswith("digCPP.root")  or file.endswith("_dig.root")) and fallback_digi_file is None:
                
                try:
                    fallback_digi_file = file_path
                    root_file = ROOT.TFile(file_path)
                    if not root_file or root_file.IsZombie():
                        raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
                    tree = root_file.Get(tree_name)
                    n_event = tree.GetEntriesFast() if tree else 0
                    root_file.Close()
                    
                except Exception as e:
                    print(f"Error processing ROOT file {file_path}: {e}")
            elif file.endswith("TGeant4.root") and file.startswith("sndLHC"):
                raw_file = file_path
            elif file.startswith("geo"):
                geo_file = file_path
                
            
        if digi_file is None:
            digi_file = fallback_digi_file

        one_file_data = {
            'data_type': data_type,
            'subfolder': output_subfolder_name,
            'partition': partition,
            'n_event': n_event,
            'digi_path': digi_file,
            'geo_path': geo_file,
            'raw_path': raw_file
        }
        metadata.append(one_file_data)
        #print('add: ', one_file_data)

    metadata.sort(key=lambda x: x['partition'])
    return metadata

def save_metadata_to_csv(metadata, csv_name):
    column_names = ['data_type', 'subfolder', 'partition', 'n_event', 'digi_path', 'geo_path','raw_path']
    
    #print(metadata)
    # Sort metadata as per the specified rules
    data_type = metadata[0].get('data_type')
    if "MC" in data_type and "muon" not in data_type:
        metadata.sort(key=lambda x: (x['subfolder'], int(x['partition'])))
    else:
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

geo_file_map = [
    (4361, 5422, 'eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root'),
    (5482, 7356, 'eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root'),
    (7357, 10422, 'eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root'),
    (10919, 12792, 'eos/experiment/sndlhc/convertedData/physics/2025/geofile_sndlhc_TI18_V8_2025.root'),
]
def get_geo_file(partition):
    try:
        run_number = int(partition.split('_')[-1])
    except ValueError:
        raise ValueError(f"Could not extract run number from partition: {partition}")

    for min_run, max_run, geo_file in geo_file_map:
        if min_run <= run_number <= max_run:
            return geo_file

    raise ValueError(f"No geometry file found for run {run_number}")

def process_MC_NC(root_path, subfolder,data_type):

    NC_ve_root = f'{root_path}/nu12/volume_volTarget/'
    NC_vm_root = f'{root_path}/nu14/volume_volTarget/'
    
    NC_ve_metadata = process_MC_subfolders(NC_ve_root, subfolder,data_type)
    NC_vm_metadata = process_MC_subfolders(NC_vm_root, subfolder,data_type)
    
    NC_metadata = NC_ve_metadata + NC_vm_metadata
    
    return NC_metadata
    
def generate_neutrino_path(data_type, root_path, subfolder,csv_file):
    
    if "2024_NC" in subfolder:
        metadata = process_MC_NC(root_path, subfolder,data_type)
    else:
        metadata = process_MC_subfolders(root_path, subfolder,data_type)
    save_metadata_to_csv(metadata, csv_file)

def generate_real_data_path(data_type, root_path, subfolder, csv_file):
    file_name = os.path.basename(csv_file)
    year = int(file_name.split("_")[2])
    if year<2024:
        metadata = process_real_data_subfolders(root_path, subfolder, data_type)
        save_metadata_to_csv(metadata, csv_file)
    else:
        metadata = process_real_data_since2024(root_path, subfolder, data_type)
        save_metadata_to_csv(metadata, csv_file)


def generate_neutral_hadron_path(data_type, root_path, output_subfolder_name, csv_file):

    metadata = []
    for subfolder in tqdm(os.listdir(root_path), desc="Processing subfolders"):
        subfolder_path = os.path.join(root_path, subfolder)
        for second_subfolder in os.listdir(subfolder_path):
            second_subfolder_path = os.path.join(subfolder_path, second_subfolder)
            second_subfolder_name = f'{output_subfolder_name}/{subfolder}'
            print(second_subfolder_path)
            tmp_metadata = process_MC_subfolders(second_subfolder_path, second_subfolder_name, data_type)
            metadata.extend(tmp_metadata)
    
    save_metadata_to_csv(metadata, csv_file)

def generate_neutron_QGSP_path(data_type, root_path, output_subfolder_name, csv_file):
    #metadata = []
    for subfolder in tqdm(os.listdir(root_path), desc="Processing subfolders"):

        subfolder_path = os.path.join(root_path, subfolder)

        subfolder_contents = os.listdir(subfolder_path)

        if any("tgtarea" in folder for folder in subfolder_contents):
            # Process the special subfolder and navigate to its Ntuples directory
            special_folder = next(
                folder for folder in subfolder_contents if "tgtarea" in folder
            )
            second_subfolder_path = os.path.join(subfolder_path, special_folder, "Ntuples")
        else:
            second_subfolder_path = os.path.join(subfolder_path, "Ntuples")
        second_subfolder_name = f'{output_subfolder_name}/{subfolder}'
        tmp_metadata = process_MC_subfolders(second_subfolder_path, second_subfolder_name, data_type)
        save_metadata_to_csv(tmp_metadata, csv_file)



def generate_MC_muon(data_type, root_path, output_subfolder_name, csv_file):

    extensions = {"Trks.root","dig.root", "digCPP.root"}  # Use a set for faster lookups
    metadata = []
    tree_name = 'cbmsim'

    # helper: get common prefix before the digi-kind suffix
    def _prefix_from_filename(name: str) -> str:
        for suf in ("_digCPP_Trks.root", "_dig_Trks.root", "_dig.root", "_digCPP.root"):
            if name.endswith(suf):
                return name[:-len(suf)]
        # fallback: strip .root if none matched (shouldn't happen with our filter)
        return os.path.splitext(name)[0]

    # helper: classify which kind this file is
    def _kind_from_filename(name: str) -> str:
        if name.endswith("_Trks.root"):
            return "Trks"
        if name.endswith("_dig.root"):
            return "dig"
        if name.endswith("_digCPP.root"):
            return "digCPP"
        return "unknown"

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if not any(filename.endswith(ext) for ext in extensions):
                continue

            file_path = os.path.join(dirpath, filename)
            try:
                # Validate the ROOT file
                root_file = ROOT.TFile.Open(file_path)
                if not root_file or root_file.IsZombie():
                    raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")

                tree = root_file.Get(tree_name)
                n_event = tree.GetEntriesFast() if tree else 0

            except Exception as e:
                print(f"Error processing ROOT file {file_path}: {e}")
                n_event = 0
            finally:
                try:
                    if root_file:
                        root_file.Close()
                except Exception:
                    pass

            # get the latest geo file (geo file will be updated correctly in the next step)
            geo_files = glob.glob(os.path.join(dirpath, "geo*"))
            geo_file = max(geo_files, key=os.path.getmtime) if geo_files else None

            # ToDo: a universal way to get partition
            partition = os.path.basename(dirpath)
            if partition in ('muons_down', 'muons_up'):
                partition = 'scoring_2'
            elif partition == '7016245':
                partition = 'scoring_2.5'

            one_file_data = {
                'data_type': data_type,
                'subfolder': output_subfolder_name,
                'partition': partition,
                'n_event': n_event,
                'digi_path': file_path,
                'geo_path': geo_file,
                'prefix': _prefix_from_filename(file_path),
                'kind': _kind_from_filename(filename),  # Trks / dig / digCPP
            }
            metadata.append(one_file_data)

    # --- Drop duplicate digi files rule ---
    # If a prefix has both Trks and any non-Trks (dig or digCPP), drop the Trks entries.
    by_prefix = {}
    for rec in metadata:
        p = rec['prefix']
        by_prefix.setdefault(p, []).append(rec)

    #print(by_prefix)
    filtered = []
    for p, recs in by_prefix.items():
        has_non_trks = any(r['kind'] in ('dig', 'digCPP') for r in recs)
        if has_non_trks:
            # keep only non-Trks for this prefix
            filtered.extend(r for r in recs if r['kind'] in ('dig', 'digCPP'))
        else:
            # keep whatever exists (only Trks available)
            filtered.extend(recs)

    # strip helper fields before saving
    for r in filtered:
        r.pop('prefix', None)
        r.pop('kind', None)
        
        
    save_metadata_to_csv(filtered, csv_file)



def process_root_file(file_path, tree_name):
    """Process a ROOT file and retrieve the number of entries in the specified tree."""
    try:
        root_file = ROOT.TFile(file_path)
        if not root_file or root_file.IsZombie():
            raise ValueError(f"Invalid or corrupted ROOT file: {file_path}")
        
        tree = root_file.Get(tree_name)
        n_event = tree.GetEntriesFast() if tree else 0
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
    elif data_type == "MC_neutrino":
        generate_neutrino_path(data_type, root_path, subfolder, csv_file)
    elif data_type == "MC_kaon" or data_type == "MC_neutron":
        if "QGSP" in subfolder:
            generate_neutron_QGSP_path(data_type, root_path, subfolder, csv_file)
        else:
            generate_neutral_hadron_path(data_type, root_path, subfolder, csv_file)
    elif data_type == "MC_muon":
        generate_MC_muon(data_type, root_path, subfolder, csv_file)
    else:
        print(f"Unknown data type: {data_type}")

def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder



def main(args):
    
    config_path = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/metadata_config.yaml"
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
    args = parser.parse_args()
    main(args)

