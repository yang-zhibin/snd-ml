import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser




def process_real_data_subfolders(root_path, output_subfolder_name, data_type):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue

        geo_file = get_geo_file(subfolder)
        if not geo_file:
            print(f"No geo file found for {subfolder}. Skipping.")
            continue

        digi_file = ''
        partition = subfolder
        n_event = 0

        for file in os.listdir(subfolder_path):
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
            'geo_path': geo_file,
            }
            metadata.append(one_file_data)
        
            print('digi_file: ', digi_file)

    metadata.sort(key=lambda x: x['partition'])
    return metadata


def process_MC_subfolders(root_path, output_subfolder_name, data_type):

    tree_name = "cbmsim"
    metadata = []

    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)
        #print(subfolder_path)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-directory: {subfolder_path}")
            continue

        digi_file = ''
        geo_file = ''
        partition = subfolder
        n_event = 0

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            #print(file_path)
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

        one_file_data = {
            'data_type': data_type,
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

def get_geo_file(partition):
    try:
        run_number = int(partition.split('_')[-1])  
    except ValueError:
        print(f"Could not extract run number from subfolder: {partition}")
        

    geo_file_map = {
        range(0, 4575): '/afs/cern.ch/user/z/zhibin/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V2_12July2022.root',
        range(4575, 4855): '/afs/cern.ch/user/z/zhibin/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V5_14August2022.root',
        range(4855, 5172): '/afs/cern.ch/user/z/zhibin/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V6_08October2022.root',
        range(5172, 5422): '/afs/cern.ch/user/z/zhibin/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V7_22November2022.root',
        range(5482, 7347): '/afs/cern.ch/user/z/zhibin/sndlhc/convertedData/physics/2023_reprocess/geofile_sndlhc_TI18_V4_2023.root',
    }
    for run_range, geo_file in geo_file_map.items():
        if run_number in run_range:
            return geo_file
    return None

def generate_neutrino_path(data_type, root_path, subfolder,csv_file):
    metadata = process_MC_subfolders(root_path, subfolder,data_type)
    save_metadata_to_csv(metadata, csv_file)

def generate_real_data_path(data_type, root_path, subfolder, csv_file):
    metadata = process_real_data_subfolders(root_path, subfolder, data_type)
    save_metadata_to_csv(metadata, csv_file)

def generate_neutral_hadron_path(data_type, root_path, output_subfolder_name, csv_file):

    metadata = []
    for subfolder in os.listdir(root_path):
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
    for subfolder in os.listdir(root_path):

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

    extensions = {"dig.root", "digCPP.root"}  # Use a set for faster lookups
    metadata = []
    tree_name = 'cbmsim'

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
                n_event = tree.GetEntries() if tree else 0
                root_file.Close()

                # Generate metadata
                geo_files = glob.glob(os.path.join(dirpath, "geo*"))
                geo_file = geo_files[0] if geo_files else None  # Take the first match, or None if no match

                #ToDo: a universal way to get partition
                partition = os.path.basename(dirpath)
                if partition == 'muons_down':
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
                }
                metadata.append(one_file_data)
                #print('Added:', one_file_data)

            except Exception as e:
                print(f"Error processing ROOT file {file_path}: {e}")

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

def check_meta_data(csv_file):
    # Constants
    TREE_NAME = 'cbmsim'
    DIGI_FILE_SUFFIX = "digCPP.root"
    GEO_FILE_PREFIX = "geo"

    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Filter rows where n_event == 0
        filtered_data = data[data['n_event'] == 0]


        
        # Process each row
        for index, row in filtered_data.iterrows():
            digi_file_path = row.get('digi_path', '')
            if not digi_file_path:
                print(f"Missing 'digi_path' in row {index}.")
                continue
            
            folder_path = os.path.dirname(digi_file_path)
            
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            
            n_event = 0
            geo_path = ''
            
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                
                # Check for digitized ROOT files
                if file.endswith(DIGI_FILE_SUFFIX):
                    n_event = process_root_file(file_path, TREE_NAME)
                
                # Check for geometry files
                elif file.startswith(GEO_FILE_PREFIX):
                    geo_path = file_path
            
            # Update the row if `n_event` is not 0
            if n_event != 0:
                print(f"Updating row {index}: n_event={n_event}, digi_path={digi_file_path}, geo_path={geo_path}")
                data.at[index, 'n_event'] = n_event
                data.at[index, 'digi_path'] = digi_file_path
                data.at[index, 'geo_path'] = geo_path
            elif n_event == 0:
                print(f"row {index}: n_event={n_event}, digi_path={digi_file_path}, geo_path={geo_path}")
        
        # Save the updated CSV file
        simple_update_csv_file(csv_file, data)
    
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
    except KeyError as e:
        print(f"Missing expected column: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def drop_0_event_row(csv_file):

    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Drop rows where 'n_event' is 0
    filtered_data = data[data['n_event'] != 0]
    
    # Save the updated CSV back
    filtered_data.to_csv(csv_file, index=False)

    rm_rows = len(data) - len(filtered_data)
    print(f" {rm_rows} rows with n_event == 0 have been removed from {csv_file}.")

def check_error(data_type):
    folder_path = './'
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if (file.endswith('.csv') and file.startswith(data_type))]

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        check_meta_data(csv_file)
        drop_0_event_row(csv_file)

    
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
        file_exists = os.path.isfile(csv_file)
        if (not file_exists) and (not args.force_rerun):
            print(f'{csv_file} does not exist, generating from raw data')
            process(data_type, root_path, subfolder, csv_file)
        else:
            print(f'{csv_file} exist, skip generating from raw data')
    check_error(data_type)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun", dest="force_rerun", help="force rerun", default=False)
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    args = parser.parse_args()
    main(args)

