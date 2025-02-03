import os
import csv
import ROOT
import pandas as pd
import glob



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
        geo_file = ''
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
        
            print('add: ', one_file_data)

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
            print('add: ', one_file_data)

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

def generate_neutrino_path(data_type, root_path, subfolder):
    csv_name = f'{data_type}_{subfolder}_metadata.csv'

    metadata = process_MC_subfolders(root_path, subfolder,data_type)
    save_metadata_to_csv(metadata, csv_name)

def generate_real_data_path(data_type, root_path, subfolder):
    metadata = process_real_data_subfolders(root_path, subfolder, data_type)

    csv_name = f'{data_type}_{subfolder}_metadata.csv'
    save_metadata_to_csv(metadata, csv_name)

def generate_neutral_hadron_path(data_type, root_path, output_subfolder_name):

    metadata = []
    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)
        for second_subfolder in os.listdir(subfolder_path):
            second_subfolder_path = os.path.join(subfolder_path, second_subfolder)
            second_subfolder_name = f'{output_subfolder_name}/{subfolder}'
            print(second_subfolder_path)
            tmp_metadata = process_MC_subfolders(second_subfolder_path, second_subfolder_name, data_type)
            metadata.extend(tmp_metadata)
    
    csv_name = f'{data_type}_{output_subfolder_name}_metadata.csv'
    save_metadata_to_csv(metadata, csv_name)

def generate_neutron_QGSP_path(data_type, root_path, output_subfolder_name):
    #metadata = []
    csv_name = f'{data_type}_{output_subfolder_name}_metadata.csv'
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
        save_metadata_to_csv(tmp_metadata, csv_name)

    


def generate_MC_muon(data_type, root_path, output_subfolder_name):

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
                print('Added:', one_file_data)

            except Exception as e:
                print(f"Error processing ROOT file {file_path}: {e}")

    csv_name = f'{data_type}_{output_subfolder_name}_metadata.csv'
    save_metadata_to_csv(metadata, csv_name)



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

def update_csv_file(csv_file, updated_data):
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
        update_csv_file(csv_file, data)
    
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
    print(f"Rows with n_event == 0 have been removed from {csv_file}.")


def main():
    data_type = 'real_data'
    root_path = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/"
    output_subfolder_name = '2023_reprocess'
    generate_real_data_path(data_type, root_path, output_subfolder_name)

    data_type = 'real_data'
    root_path = "/eos/experiment/sndlhc/convertedData/physics/2022/"
    output_subfolder_name = '2022'
    generate_real_data_path(data_type, root_path, output_subfolder_name)

    data_type = "MC_neutrino"
    root_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volMuFilter_20fb-1_SNDG18_02a_01_000"
    subfolder = 'volMuFilter_20fb-1'
    generate_neutrino_path(data_type, root_path, subfolder)

    data_type = "MC_neutrino"
    root_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"
    subfolder = 'volTarget_100fb-1'
    generate_neutrino_path(data_type, root_path, subfolder)


    root_path = "/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/FTFP_BERT/kaons/"
    subfolder = 'FTFP_BERT'
    data_type = 'MC_kaon'
    generate_neutral_hadron_path(data_type, root_path, subfolder)

    root_path = "/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/FTFP_BERT/neutrons/"
    subfolder = 'FTFP_BERT'
    data_type = 'MC_neutron'
    generate_neutral_hadron_path(data_type, root_path, subfolder)

    root_path = "/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/QGSP_BERT_HP_PEN/kaons/"
    subfolder = 'QGSP_BERT_HP_PEN'
    data_type = 'MC_kaon'
    generate_neutral_hadron_path(data_type, root_path, subfolder)

    root_path = "/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/QGSP_BERT_HP_PEN/neutrons/"
    subfolder = 'QGSP_BERT_HP_PEN'
    data_type = 'MC_neutron'
    generate_neutron_QGSP_path(data_type, root_path, subfolder)

    root_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/MuonBackground/muons_down'
    subfolder = 'down'
    data_type = 'MC_muon'
    generate_MC_muon(data_type,root_path, subfolder)

    root_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/MuonBackground/muons_up'
    subfolder = 'up'
    data_type = 'MC_muon'
    generate_MC_muon(data_type,root_path, subfolder)

    root_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/MuonBackground/muons_horizontal'
    subfolder = 'horizontal'
    data_type = 'MC_muon'
    generate_MC_muon(data_type,root_path, subfolder)

    print('finish main function')

def check_error():
    folder_path = './'
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        check_meta_data(csv_file)
        drop_0_event_row(csv_file)

if __name__ == "__main__":
    main()
    check_error()
    
