import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}
class_2_particle = {v: k for k, v in particle_2_class.items()}
target_columns = ['ve', 'vm', 'vt', 'NC', 'kaon', 'neutron', 'muon']
preCut_target_columns=['count']



def read_metadata(directory="./processed_metadata"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict



def load_matrix_from_csv(df, max_file=1e6, int_lumi_real_data=1e6, process_real_data=False):
    #print(df)
    preCut_matrix = None
    
    vetoFree_matrix = None
    vetoTagged_matrix = None
    
    vetoFree_lumi = 0
    vetoTagged_lumi = 0
    

    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        preCutEff_path =  row[f'preCutEff_path']
        vetoFree_matrix_path = row[f'vetoFree_matrix_{model_name}_output_path']
        vetoFree_lumi_per_file = np.nan_to_num(row['lumi_per_file'], nan=0.0)
        
        if os.path.exists(preCutEff_path):
            pc_matrix = pd.read_csv(preCutEff_path, index_col=0)
            #vf_matrix = pd.read_csv(vetoFree_matrix_path, index_col=0)
            vetoFree_lumi += vetoFree_lumi_per_file
            
            
            
            # if ((vf_matrix.index == 'data_vetoFree') & (vf_matrix['cut'] == 'no_cut')).any():
            #     continue
            if preCut_matrix is None:
                #vetoFree_matrix = vf_matrix.copy()
                preCut_matrix = pc_matrix.copy()
            else:
                #vetoFree_matrix[target_columns] = vetoFree_matrix[target_columns].add(vf_matrix[target_columns], fill_value=0)
                preCut_matrix[preCut_target_columns] = preCut_matrix[preCut_target_columns].add(pc_matrix[preCut_target_columns], fill_value=0)
            
            if process_real_data and (vetoFree_lumi>int_lumi_real_data):
                break

            count+=1
        
        vetoTagged_matrix_path = row[f'vetoTagged_matrix_{model_name}_output_path']
        vetoTagged_lumi_per_file = np.nan_to_num(row['lumi_per_file'], nan=0.0)
        vetoTagged_veto_ineff = row['veto_ineff'] if 'veto_ineff' in row else 1e-8
        
        
        
        if os.path.exists(vetoTagged_matrix_path):
            
            vt_matrix = pd.read_csv(vetoTagged_matrix_path, index_col=0)
            numeric_cols = vt_matrix.select_dtypes(include='number').columns
            vt_matrix.loc[:, numeric_cols] = vt_matrix.loc[:, numeric_cols] * vetoTagged_veto_ineff
            vetoTagged_lumi += vetoTagged_lumi_per_file
        
        
            if vetoTagged_matrix is None:
                vetoTagged_matrix = vt_matrix.copy()
            else:
                vetoTagged_matrix[target_columns] = vetoTagged_matrix[target_columns].add(vt_matrix[target_columns], fill_value=0)
        
        
        if count > max_file:
            break
    #print(particle_matrix)


    #num_cols = vetoFree_matrix.select_dtypes(include="number").columns
    #vetoFree_matrix["count"] = vetoFree_matrix[num_cols].sum(axis=1)
    
    
    
    return vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi, preCut_matrix

def load_neutral_bkg_matrix(df):
    results = []

    # Ensure 'energy_range' exists in the DataFrame
    if 'energy_range' not in df.columns:
        raise ValueError("DataFrame must contain an 'energy_range' column.")

    # Group by energy_range
    grouped = df.groupby('energy_range')

    for energy_range, group in grouped:
        # Load matrix and lumi for this energy range
        vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi,  preCut_matrix= load_matrix_from_csv(group)

        # print('-----')
        # print(vetoFree_matrix)
        # print(preCut_matrix)
        results.append((energy_range, (vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi, preCut_matrix)))

    
    # print("------")
    # print(results["vetoFree_matrix"])
    # print(results["preCut_matrix"])
    return results

def sum_normalized_neutral_bkg_matrix(matrix_list, factor=1.0):
    total_matrix = None
    total_preCut_matrix = None
    # print(matrix_list)

    for energy_range, (vetoFree_matrix, vetoFree_lumi, _, _, preCut_matrix) in matrix_list:
        if vetoFree_lumi == 0:
            continue  # Skip to avoid division by zero

        normalized_matrix = normalize_matrix(vetoFree_matrix, vetoFree_lumi, factor)
        normalized_preCut_matrix = normalize_matrix(preCut_matrix, vetoFree_lumi, factor)
        


        if total_matrix is None:
            #total_matrix = normalized_matrix.copy()
            total_preCut_matrix = normalized_preCut_matrix.copy()
        else:
            #total_matrix[target_columns + ["count"]] = total_matrix[target_columns + ["count"]].add(normalized_matrix[target_columns + ["count"]], fill_value=0)
            total_preCut_matrix[preCut_target_columns] = total_preCut_matrix[preCut_target_columns].add(normalized_preCut_matrix[preCut_target_columns], fill_value=0)

    
    # print('-----total--')
    # print(total_matrix, total_preCut_matrix)
    
    return total_matrix, total_preCut_matrix

def normalize_matrix(matrix, lumi, factor):
    if matrix is None:
        return matrix
    if lumi == 0:
        raise ValueError("Cannot normalize matrix with lumi=0")
    
    numeric_cols = matrix.select_dtypes(include='number').columns
    print()
    matrix = matrix.copy()
    matrix[numeric_cols] = matrix[numeric_cols] / lumi * factor
    return matrix


def process_matrix(realdata_metadata, MC_neutrino_metadata, MC_kaon_metadata, MC_neutron_metadata):
    
    realdata_vetoFree_matrix, realdata_vetoFree_lumi, realdata_vetoTagged_matrix, realdata_vetoTagged_lumi, realdata_preCut_matrix = load_matrix_from_csv(realdata_metadata, max_file=1450)
    
    MC_neutrino_vetoFree_matrix, MC_neutrino_vetoFree_lumi, _, _, MC_neutrino_preCut_matrix = load_matrix_from_csv(MC_neutrino_metadata)
    
    MC_kaon_matrix_list= load_neutral_bkg_matrix(MC_kaon_metadata)
    MC_neutron_matrix_list = load_neutral_bkg_matrix(MC_neutron_metadata)
    
    #normalized_MC_neutrino_vetoFree_matrix = normalize_matrix(MC_neutrino_vetoFree_matrix, MC_neutrino_vetoFree_lumi, realdata_vetoFree_lumi)
    #normalized_realdata_vetoTagged_matrix = normalize_matrix(realdata_vetoTagged_matrix, realdata_vetoTagged_lumi, realdata_vetoFree_lumi)
    
    normalized_MC_neutrino_preCut_matrix = normalize_matrix(MC_neutrino_preCut_matrix, MC_neutrino_vetoFree_lumi, realdata_vetoFree_lumi)
    
    
    
    normalized_MC_kaon_vetoFree_matrix, normalized_MC_kaon_preCut_matrix = sum_normalized_neutral_bkg_matrix(MC_kaon_matrix_list, realdata_vetoFree_lumi)
    
    normalized_MC_neutron_vetoFree_matrix, normalized_MC_neutron_preCut_matrix = sum_normalized_neutral_bkg_matrix(MC_neutron_matrix_list, realdata_vetoFree_lumi)

    
    # print("\n--- Normalized Real Data PreCut Matrix ---")
    # print(realdata_preCut_matrix.to_string())
    
    # print("\n--- Normalized Real Data Veto-Free Matrix ---")
    # print(realdata_vetoFree_matrix.to_string())
    
    # print("\n--- Normalized Real Data Veto-Tagged Matrix ---")
    # print(normalized_realdata_vetoTagged_matrix.to_string())
    
    # print("\n--- Normalized MC Neutrino Veto-Free Matrix ---")
    # print(normalized_MC_neutrino_vetoFree_matrix.to_string())



    # print("\n--- Normalized MC Neutrino Pre-Cut Matrix ---")
    # print(normalized_MC_neutrino_preCut_matrix.to_string())

    # print("\n--- Normalized MC Kaon Veto-Free Matrix ---")
    # print(normalized_MC_kaon_vetoFree_matrix.to_string())

    # print("\n--- Normalized MC Kaon Pre-Cut Matrix ---")
    # print(normalized_MC_kaon_preCut_matrix.to_string())

    # print("\n--- Normalized MC Neutron Veto-Free Matrix ---")
    # print(normalized_MC_neutron_vetoFree_matrix.to_string())

    # print("\n--- Normalized MC Neutron Pre-Cut Matrix ---")
    # print(normalized_MC_neutron_preCut_matrix.to_string())
    
    realdata_cutFlow = pd.concat([
        realdata_preCut_matrix[['cut', 'count']],
        #realdata_vetoFree_matrix[['cut', 'count']]
    ])
    realdata_cutFlow.index = realdata_cutFlow.index.str.replace('data_vetoFree', 'data')
    realdata_cutFlow =realdata_cutFlow.set_index("cut")

    neutrino_cutFlow = pd.concat([
        normalized_MC_neutrino_preCut_matrix[['cut', 'count']],
        #normalized_MC_neutrino_vetoFree_matrix[['cut', 'count']]
    ])
    # print(neutrino_cutFlow.head())
    # print(neutrino_cutFlow.columns)
    neutrino_cutFlow =neutrino_cutFlow.reset_index().pivot(index="cut", columns="index", values="count")

    kaon_cutFlow = pd.concat([
        normalized_MC_kaon_preCut_matrix[['cut', 'count']],
        #normalized_MC_kaon_vetoFree_matrix[['cut', 'count']]
    ])
    kaon_cutFlow =kaon_cutFlow.set_index("cut")

    neutron_cutFlow = pd.concat([
        normalized_MC_neutron_preCut_matrix[['cut', 'count']],
        #normalized_MC_neutron_vetoFree_matrix[['cut', 'count']]
    ])
    neutron_cutFlow =neutron_cutFlow.set_index("cut")
    


    # Print them
    # print("\n--- Real Data Cut Flow ---")
    # print(realdata_cutFlow.to_string())

    # print("\n--- Neutrino Cut Flow ---")
    # print(neutrino_cutFlow.to_string())

    # print("\n--- Kaon Cut Flow ---")
    # print(kaon_cutFlow.to_string())

    # print("\n--- Neutron Cut Flow ---")
    # print(neutron_cutFlow.to_string())
    
    
    # Ensure all DataFrames have 'cut' as index
    real = realdata_cutFlow.rename(columns={'count': 'RealData'})
    kaon = kaon_cutFlow.rename(columns={'count': 'Kaon'})
    neutron = neutron_cutFlow.rename(columns={'count': 'Neutron'})
    neutrino = neutrino_cutFlow.copy()  # Already has multiple columns (NC, ve, vm, vt)

    # Combine all into a single DataFrame
    combined = pd.concat([real, neutrino, kaon, neutron], axis=1)

    # Display the combined DataFrame
    print("\n--- Combined Cut Flow ---")
    print(combined.to_string())
    
    
    
    stepwise_efficiency = combined.copy()

    # Apply row-wise division per column
    stepwise_efficiency = stepwise_efficiency / stepwise_efficiency.shift(1)
    
    # Set first row to 1 instead of NaN
    stepwise_efficiency.loc['a_raw'] = 1.0

    # Optional: Round for readability
    stepwise_efficiency = stepwise_efficiency.round(6)

    print("\n--- Efficiency Table (each cut vs previous) ---")
    print(stepwise_efficiency.to_string())
    
    
    
    cut_name_map = {
    "a_raw": "Raw file total events",
    "b_non_veto": "At least one non-veto hits",
    "1_scifi>200": "SciFi hits > 200",
    "3_veto0": "Veto hits = 0",
}
    renamed_combined = combined.rename(index=cut_name_map)
    renamed_stepwise = stepwise_efficiency.rename(index=cut_name_map)

    formatted_combined = renamed_combined.applymap(lambda x: format_to_sigfigs(x, sigfigs=2))
    formatted_stepwise = renamed_stepwise.applymap(lambda x: format_to_sigfigs(x, sigfigs=2))
    
    #formatted_combined = formatted_combined.drop("Raw file (0)")
    #formatted_stepwise = formatted_stepwise.drop("Raw file (0)")

    # Define row colors: first 2 rows white, next 4 green, rest blue
    row_colors = ['white'] * 2 + ['#ccffcc'] * 4 + ['#cce5ff'] * (len(formatted_combined) - 6)

    
        
    plot_table(
        df=formatted_combined,
        title="Combined Cut Flow Table",
        save_path="./cutFlow_table/new_combined_cut_flow_table.pdf",
        int_lumi=realdata_vetoFree_lumi,
        row_colors=row_colors
    )

    plot_table(
        df=formatted_stepwise,
        title="Stepwise Efficiency Table ",
        save_path="./cutFlow_table/new_stepwise_efficiency_table.pdf",
        int_lumi=realdata_vetoFree_lumi,
        row_colors=row_colors
    )
    
    
    #plot_table(all_matrix, realdata_vetoFree_lumi)



def format_to_sigfigs(value, sigfigs=2, sci_threshold=4):
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"

    abs_val = abs(value)
    order = math.floor(math.log10(abs_val))
    rounded = round(value, -order + (sigfigs - 1))

    if order >= sci_threshold or order < -sci_threshold:
        mantissa = rounded / (10 ** order)
        return f"{mantissa:.{sigfigs - 1}f}e{order}"
    else:
        decimal_places = max(sigfigs - 1 - order, 0)
        return f"{rounded:.{decimal_places}f}"

def plot_table(
    df, 
    title, 
    save_path, 
    int_lumi,
    figsize=(14, 6), 
    font_size=10, 
    col_scale=1.2, 
    row_scale=2.0,  # ← increased row height
    row_colors=None
):
    """
    Plots a pandas DataFrame as a matplotlib table and saves it as an image.

    Parameters:
        df (pd.DataFrame): The DataFrame to plot.
        title (str): Title of the table.
        save_path (str): Full path to save the image (e.g. './output/table.png').
        figsize (tuple): Size of the figure in inches.
        font_size (int): Font size used in the table.
        col_scale (float): Scaling factor for column width.
        row_scale (float): Scaling factor for row height.
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.axis('tight')
    
    # Prepare table data
    data = df.reset_index()
    col_labels = data.columns
    cell_text = data.values

    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(col_scale, row_scale)
    
    first_col_width_scale = 2.5  # Adjust as needed
    n_rows = len(cell_text)
    n_cols = len(col_labels)
    
    for row_idx in range(n_rows + 1):  # include header
        cell = table[(row_idx, 0)]
        cell.set_width(cell.get_width() * first_col_width_scale)
    
     # Apply row colors (first row is header)
    if row_colors:
        for row_idx in range(1, len(cell_text) + 1):  # +1 because header is row 0
            color = row_colors[row_idx - 1]
            for col_idx in range(len(col_labels)):
                table[(row_idx, col_idx)].set_facecolor(color)
    
    legend_elements = [
        Patch(facecolor='#ccffcc', edgecolor='black', label='PreSelect cut'),
        Patch(facecolor='#cce5ff', edgecolor='black', label='AfterSelect cut')
        ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)


    fig.text(
        0.75, 0.90,  # x, y in normalized figure coordinates
        f"$\\int\\!\\mathcal{{L}}\\,dt = {int_lumi:.2f}\\ \\mathrm{{fb}}^{{-1}}$",  # LaTeX-style string
        ha="right",  # horizontal alignment to match TextAlign(31)
        va="top",
        fontsize=12,  
    )
    
    ax.set_title(title, fontsize=14, pad=20)  
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

METADATA_dict = read_metadata()
model_name = 'baseline_muon'

def main():

    
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    process_matrix(real_data, neutrino_df, kaon_df, neutron_df)
    
    
    


def read_metadata(directory="./processed_metadata"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict
    


if __name__ == "__main__": 
    main()
