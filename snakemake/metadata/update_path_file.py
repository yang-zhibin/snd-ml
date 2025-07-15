import os
import csv
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
import re

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
    if path_type == "pt":
        suffix = ".pt.gz"
    elif path_type == "pkl":
        suffix = ".pkl.gz"
    elif path_type == "csv":
        suffix = ".csv"
    else:
        suffix = ".root"
    
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

def drop_run_without_lumi_and_add_ineff(df, lumi_file):
    lumi_df = pd.read_csv(lumi_file)
    df['run'] = df['partition'].str.extract(r'run_(\d+)').astype(int)

    df = df[df["run"].isin(lumi_df["run"])]

    # Create a mapping from run_id to lumi
    lumi_map = dict(zip(lumi_df['run'], lumi_df['lumi']))
    ineff_map = dict(zip(lumi_df['run'], lumi_df['ineff']))

    # Compute lumi per file
    lumi_per_file_list = []
    ineff_per_file_list = []
    for run_id, run_df in df.groupby('run'):
        total_events = run_df['n_event'].sum()
        run_lumi = lumi_map.get(run_id, 0)
        run_ineff = ineff_map.get(run_id, 0)

        lumi_per_file = run_df['n_event'] * run_lumi / total_events
        lumi_per_file_list.append(lumi_per_file)

        ineff_per_file = pd.Series(run_ineff, index=run_df.index)
        ineff_per_file_list.append(ineff_per_file)

    # Flatten the list and assign back
    df['lumi_per_file'] = pd.concat(lumi_per_file_list).sort_index()
    df['veto_ineff'] = pd.concat(ineff_per_file_list).sort_index()
    return df


def get_int_rate():
    # from paper CERN-SND@LHC-NOTE-2023-002
    # Neutron yield data
    neutron_yield_FTFP_BERT = {
        "Energy [GeV]": [
            "(5,10)", "(10,20)", "(20,30)", "(30,40)", "(40,50)", "(50,60)", "(60,70)",
            "(70,80)", "(80,90)", "(90,100)", "(100,150)", "(150,200)"
        ],
        "Int. Rate": [
            "4.62e+04", "7.59e+03", "1.18e+03", "5.30e+02", "4.66e+02", "2.60e+01",
            "1.80e+01", "8.48", "8.48", "0", "0", "0"
        ]
    }
    neutron_yield_FTFP_BERT_2022 = pd.DataFrame(neutron_yield_FTFP_BERT)

    # Kaon yield data
    kaon_yield_FTFP_BERT = {
        "Energy [GeV]": [
            "(5,10)", "(10,20)", "(20,30)", "(30,40)", "(40,50)", "(50,60)", "(60,70)",
            "(70,80)", "(80,90)", "(90,100)", "(100,150)", "(150,200)"
        ],
        "Int. Rate": [
            "2.51e+04", "5.72e+03", "8.53e+02", "1.10e+02", "9.38e+01", "6.48e+01",
            "9.90e+00", "2.32e+01", "1.15e+01", "1.15e+01", "0", "0"
        ]
    }
    kaon_yield_FTFP_BERT_2022 = pd.DataFrame(kaon_yield_FTFP_BERT)

    # Luminosity in fb^-1
    lumi_2022 = 36.77

    # Convert to float
    neutron_yield_FTFP_BERT_2022["Int. Rate"] = neutron_yield_FTFP_BERT_2022["Int. Rate"].astype(float)
    kaon_yield_FTFP_BERT_2022["Int. Rate"] = kaon_yield_FTFP_BERT_2022["Int. Rate"].astype(float)

    # Create normalized DataFrames
    neutron_yield_FTFP_BERT_per_fb = pd.DataFrame({
        "Energy [GeV]": neutron_yield_FTFP_BERT_2022["Energy [GeV]"],
        "Rate per fb^-1": neutron_yield_FTFP_BERT_2022["Int. Rate"] / lumi_2022
    })

    kaon_yield_FTFP_BERT_per_fb = pd.DataFrame({
        "Energy [GeV]": kaon_yield_FTFP_BERT_2022["Energy [GeV]"],
        "Rate per fb^-1": kaon_yield_FTFP_BERT_2022["Int. Rate"] / lumi_2022
    })

    return neutron_yield_FTFP_BERT_per_fb, kaon_yield_FTFP_BERT_per_fb


def extract_energy(subfolder):
    match = re.search(r'[_/]([a-zA-Z]+)_(\d+)_?(\d+)?', subfolder)
    if match:
        low = int(match.group(2))
        high = int(match.group(3)) if match.group(3) else None
        return (low, high)
    return None

def cal_lumi_for_neutral_bkg(int_rate_df, metadata_df):
    metadata_df["energy_range"] = metadata_df["subfolder"].apply(extract_energy)


    rate_dict = dict(zip(int_rate_df["Energy [GeV]"], int_rate_df["Rate per fb^-1"]))
    
    def compute_lumi(row):
        energy_range = row["energy_range"]
        if energy_range is None:
            return None
        energy_str = f"({energy_range[0]},{energy_range[1]})"
        rate = rate_dict.get(energy_str)
        n_event = row.get("n_event")
        #print("energy_range",energy_str,"rate",rate, "n_event", n_event)
        if rate == 0 or rate is None:
            return None
        return n_event / rate

    metadata_df["lumi_per_file"] = metadata_df.apply(compute_lumi, axis=1)

    #print(metadata_df)
    return metadata_df
    



def update_csv_file(args, data_type, root_path, subfolder, csv_output, csv_input, eos_root_path, models,lumi_file):
    
    df = pd.read_csv(csv_input)
    if (data_type=='real_data'):
        df = drop_run_without_lumi_and_add_ineff(df, lumi_file)
    elif (data_type=='MC_kaon' or data_type=='MC_neutron'):
        neutron_rates, kaon_rates = get_int_rate()
        if ("FTFP_BERT" in subfolder) and (data_type=='MC_neutron'):
            df = cal_lumi_for_neutral_bkg(neutron_rates, df)
        elif ("FTFP_BERT" in subfolder) and (data_type=='MC_kaon'):
            df = cal_lumi_for_neutral_bkg(neutron_rates, df)        
    elif (data_type=='MC_neutrino'):
        if '100fb-1' in csv_input:
            df['lumi_per_file'] =100
        elif '20fb-1' in csv_input:
            df['lumi_per_file'] =20

    # hit path
    df = add_new_path("hit","root", df, csv_input,  eos_root_path)
    # feature path
    df = add_new_path("feature","root", df, csv_input, eos_root_path)
    # pt hit path
    df = add_new_path("pt_hit","pt", df, csv_input, eos_root_path)
    # add 3d hit path
    df = add_new_path("pkl_hit","pkl", df, csv_input, eos_root_path)

    # 
    model_names = [list(model.keys())[0] for model in models]
    for model in model_names:
        
        if "3d" in model :
            df = add_new_path(f"eval_{model}_output", "pkl", df, csv_input, eos_root_path)
            df = add_new_path(f"model_{model}_output", "pkl", df, csv_input, eos_root_path)
        else:
            df = add_new_path(f"eval_{model}_output", "root", df, csv_input, eos_root_path)
            df = add_new_path(f"model_{model}_output", "root", df, csv_input, eos_root_path)
        
        df = add_new_path(f"matrix_{model}_output", "csv", df, csv_input, eos_root_path)

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
    parser.add_argument("-l", "--lumi", dest="lumi", help="lumi record file path", default='/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/SND_lumi_with_ineff.csv')
    args = parser.parse_args()
    main(args)

