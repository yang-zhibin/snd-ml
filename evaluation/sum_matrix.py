import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}

target_columns = ['ve', 'vm', 'vt', 'NC', 'kaon', 'neutron', 'muon']

def check_signal_region(full_matrix):
    realdata_matrix, realdata_lumi = full_matrix["real_data_2024"]

    all_matrix = pd.concat([
        realdata_matrix,
    ], ignore_index=False)

    all_matrix.index.name = 'true_class'
    all_matrix = all_matrix.reset_index()


    cut_groups = {
        "Prediction score cuts": [
            "no_cut",
            'Prediction_0 > 0.95 && fiducial_tl_1 && fiducial_br_1',
            'Prediction_0 > 0.95 && fiducial_tl_1 && fiducial_br_1 && scifi_gt_100',
        ],
        "Scifi hit with fiducial cuts": [
            "no_cut",
            'scifi_gt_100 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_300 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_500 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_700 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_900 && fiducial_tl_1 && fiducial_br_1',
        ],
        "Scifi hit cuts": [
            "no_cut",
            'scifi_gt_100',
            'scifi_gt_300',
            'scifi_gt_500',
            'scifi_gt_700',
            'scifi_gt_900',
        ],
    }

    # Group all_matrix by 'cut'
    grouped = dict(tuple(all_matrix.groupby('cut')))

    bkg_defs = {
        #"veto_inverted": lambda g: g[g["true_class"] == "veto_inverted"]["ve"].values[0],
        "pred_kaon": lambda g: g[g["true_class"] == "signal_region"]["kaon"].values[0],
        "pred_neutron ": lambda g: g[g["true_class"] == "signal_region"]["neutron"].values[0],
        "pred_muon ": lambda g: g[g["true_class"] == "signal_region"]["muon"].values[0],
        "pred_total_bkg ": lambda g: g[g["true_class"] == "signal_region"]["muon"].values[0] + g[g["true_class"] == "signal_region"]["neutron"].values[0] + g[g["true_class"] == "signal_region"]["kaon"].values[0],
        # "vm": lambda g: g[g["true_class"] == "vm"]["ve"].values[0],
        # "NC": lambda g: g[g["true_class"] == "NC"]["ve"].values[0],
        # "vt": lambda g: g[g["true_class"] == "vt"]["ve"].values[0],
        # "total_bkg": lambda g: (
        #     g[g["true_class"] == "veto_inverted"]["ve"].values[0] +
        #     g[g["true_class"] == "kaon"]["ve"].values[0] +
        #     g[g["true_class"] == "neutron"]["ve"].values[0] +
        #     g[g["true_class"] == "vm"]["ve"].values[0] +
        #     g[g["true_class"] == "NC"]["ve"].values[0] +
        #     g[g["true_class"] == "vt"]["ve"].values[0]
        # ),
    }

    # Loop over each cut group
    for title, cuts in cut_groups.items():
        cut_labels = ["no_cut" if c is None else c for c in cuts]
        fig_width = len(cuts) * 2
        plt.figure(figsize=(fig_width, 6))

        for bkg_label, bkg_func in bkg_defs.items():
            bkg_values = []
            for cut in cuts:
                group = grouped.get(cut)
                if group is None:
                    bkg_values.append(np.nan)
                    continue

                try:
                    
                    bkg = bkg_func(group)
                except (IndexError, KeyError):
                    bkg = np.nan

                bkg_values.append(bkg)

            plt.plot(cut_labels, bkg_values, marker='o', label=bkg_label)

        plt.title(f"{title} - Signal Region Predicted Backgrounds")
        plt.xlabel("Cut")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"plot/sig/pred_bkg_in_signal_region_{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        print(f'saved {filename}')
        plt.close()


def plot_full_matrix(full_matrix):
    # Unpack each (matrix, lumi) tuple
    neutrino_matrix, neutrino_lumi = full_matrix["neutrino"]
    realdata_matrix, realdata_lumi = full_matrix["real_data_2024"]
    kaon_matrix, kaon_lumi = full_matrix["kaon"]
    neutron_matrix, neutron_lumi = full_matrix["neutron"]

    # Normalize simulated data to real data lumi
    def normalize(df, lumi):
        df_norm = df.copy()
        scale = realdata_lumi / lumi
        df_norm.loc[:, target_columns] = df[target_columns] * scale
        return df_norm

    normalized_neutrino = normalize(neutrino_matrix, neutrino_lumi)
    normalized_kaon = normalize(kaon_matrix, kaon_lumi)
    normalized_neutron = normalize(neutron_matrix, neutron_lumi)

    # Combine all matrices
    all_matrix = pd.concat([
        normalized_neutrino,
        realdata_matrix,
        normalized_kaon,
        normalized_neutron
    ], ignore_index=False)

    all_matrix.index.name = 'true_class'
    all_matrix = all_matrix.reset_index()

    print(f"real data lumi: {realdata_lumi}")
    print(all_matrix)

    for cut_name, group in all_matrix.groupby('cut'):
        print(f"\nGroup: {cut_name}")
        print(group.reset_index(drop=True).to_string(index=False))

    cut_groups = {
        "Prediction score cuts": [
            "no_cut",
            'Prediction_0 > 0.95 && fiducial_tl_1 && fiducial_br_1',
            'Prediction_0 > 0.95 && fiducial_tl_1 && fiducial_br_1 && scifi_gt_100',
        ],
        "Scifi hit with fiducial cuts": [
            "no_cut",
            'scifi_gt_100 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_300 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_500 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_700 && fiducial_tl_1 && fiducial_br_1',
            'scifi_gt_900 && fiducial_tl_1 && fiducial_br_1',
        ],
        "Scifi hit cuts": [
            "no_cut",
            'scifi_gt_100',
            'scifi_gt_300',
            'scifi_gt_500',
            'scifi_gt_700',
            'scifi_gt_900',
        ],
    }

    # Background definitions
    bkg_defs = {
        #"veto_inverted": lambda g: g[g["true_class"] == "veto_inverted"]["ve"].values[0],
        #"neutral_bkg": lambda g: g[g["true_class"] == "kaon"]["ve"].values[0] + g[g["true_class"] == "neutron"]["ve"].values[0],
        #"neutral_bkg_and_veto_inverted ": lambda g: g[g["true_class"] == "kaon"]["ve"].values[0] + g[g["true_class"] == "neutron"]["ve"].values[0] + g[g["true_class"] == "veto_inverted"]["ve"].values[0],
        "vm": lambda g: g[g["true_class"] == "vm"]["ve"].values[0],
        "NC": lambda g: g[g["true_class"] == "NC"]["ve"].values[0],
        "vt": lambda g: g[g["true_class"] == "vt"]["ve"].values[0],
        "total_bkg": lambda g: (
            g[g["true_class"] == "veto_inverted"]["ve"].values[0] +
            g[g["true_class"] == "kaon"]["ve"].values[0] +
            g[g["true_class"] == "neutron"]["ve"].values[0] +
            g[g["true_class"] == "vm"]["ve"].values[0] +
            g[g["true_class"] == "NC"]["ve"].values[0] +
            g[g["true_class"] == "vt"]["ve"].values[0]
        ),
    }

    # Group all_matrix by 'cut'
    grouped = dict(tuple(all_matrix.groupby('cut')))

    # Loop over each cut group
    for title, cuts in cut_groups.items():
        cut_labels = ["no_cut" if c is None else c for c in cuts]
        fig_width = len(cuts) * 2
        plt.figure(figsize=(fig_width, 6))

        for bkg_label, bkg_func in bkg_defs.items():
            sig_values = []
            for cut in cuts:
                group = grouped.get(cut)
                if group is None:
                    sig_values.append(np.nan)
                    continue

                try:
                    signal = group[group["true_class"] == "ve"]["ve"].values[0]
                    bkg = bkg_func(group)
                    significance = signal / np.sqrt(bkg) if bkg > 0 else np.nan
                except (IndexError, KeyError):
                    significance = np.nan

                sig_values.append(significance)

            plt.plot(cut_labels, sig_values, marker='o', label=bkg_label)

        plt.title(f"{title} - Signal Significance vs Different Backgrounds")
        plt.xlabel("Cut")
        plt.ylabel("Significance")
        plt.xticks(rotation=45)
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"plot/sig/signal_significance_{title.replace(' ', '_').lower()}_neutral_bkg.png"
        plt.savefig(filename)
        print(f'saved {filename}')
        plt.close()

def read_metadata():
    metadata_paths = {
        "neutrino": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv',
        "kaon": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_kaon_FTFP_BERT_metadata.csv',
        "neutron": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_neutron_FTFP_BERT_metadata.csv',
        "real_data_2024": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/real_data_2022_metadata.csv'
    }

    cleaned_metadata = {}
    path_column = "matrix_baseline_muon_output_path" 

    for key, file_path in metadata_paths.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path) # debug
            #print(df)
            if path_column in df.columns:
                df = df[df[path_column].apply(os.path.exists)]
                cleaned_metadata[key] = df
            else:
                print(f"Column '{path_column}' not found in {file_path}")
        else:
            print(f"Metadata file not found: {file_path}")
    
    return cleaned_metadata

def process_row(row):
    matrix_path = row['matrix_baseline_muon_output_path']
    lumi_per_file = np.nan_to_num(row['lumi_per_file'], nan=0.0)
    data_type = row['data_type']
    
    if "real_data" in data_type:
        veto_ineff = row['veto_ineff']
        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix.loc['veto_inverted', target_columns] *= veto_ineff
        
    else:
        matrix = pd.read_csv(matrix_path, index_col=0)

    return lumi_per_file, matrix



def process_exist_metadata(exist_metadata):
    full_matrix = {} # name -> (matrix, lumi)

    for name, df in exist_metadata.items():
        print(f"Dataset: {name}, Entries: {len(df)}")
        particle_matrix = None
        particle_lumi = 0

        count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            
            lumi, matrix = process_row(row)
            #print(matrix)
            # add lumi to particle_matrix
            particle_lumi += lumi

                        # add matrix to particle_matrix
            if particle_matrix is None:
                particle_matrix = matrix.copy()
            else:
                # Add values for summable keys only
                particle_matrix[target_columns] = particle_matrix[target_columns].add(matrix[target_columns], fill_value=0)
            
            #print(matrix)
            if (name == 'kaon' or name == 'neutron'):
                break
            #if count>500:
            #    break
            count+=1
        full_matrix[name] = (particle_matrix, particle_lumi)
        #print(f"name {name}, particle_lumi {particle_lumi}")
        #print(particle_matrix)
    #print(full_matrix)
    
    plot_full_matrix(full_matrix)
    check_signal_region(full_matrix)


def main():
    exist_metadata = read_metadata()
    process_exist_metadata(exist_metadata)



if __name__ == "__main__": 
    main()
