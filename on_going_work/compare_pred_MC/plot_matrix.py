import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

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



def read_metadata(directory="./processed_metadata_GravNet_v2"):
    
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict

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

def format_value_with_uncertainty(value, uncertainty, sci_threshold=4):
    if uncertainty == 0:
        return f"{value}(0)"
    
    abs_unc = abs(uncertainty)
    unc_order = math.floor(math.log10(abs_unc))
    
    # Round uncertainty to 1 significant digit
    rounded_unc = round(abs_unc, -unc_order)
    
    # Determine decimal places for value based on uncertainty
    decimal_places = max(-unc_order, 0)
    rounded_val = round(value, decimal_places)

    # Use scientific notation if value or uncertainty is too large/small
    if abs(rounded_val) < 10**-sci_threshold or abs(rounded_val) >= 10**sci_threshold:
        if rounded_val == 0:
            # Special case: value is 0 but uncertainty isn't
            exp = unc_order
        else:
            exp = math.floor(math.log10(abs(rounded_val)))
        scale = 10 ** exp
        val_scaled = rounded_val / scale
        unc_scaled = rounded_unc / scale
        
        # Round scaled uncertainty to 1 sig fig again (after scaling)
        unc_scaled_order = math.floor(math.log10(unc_scaled))
        unc_scaled_rounded = round(unc_scaled, -unc_scaled_order)
        digits = int(unc_scaled_rounded * 10**max(-unc_scaled_order, 0))
        val_scaled_rounded = round(val_scaled, max(-unc_scaled_order, 0))
        
        return f"{val_scaled_rounded:.{max(-unc_scaled_order, 0)}f}({digits})e{exp}"
    else:
        # Normal formatting
        digits = int(rounded_unc * (10 ** decimal_places) + 0.5)  # avoid rounding down to 0
        return f"{rounded_val:.{decimal_places}f}({digits})"

def plot_confusion_matrix_with_uncertainty(values_df, uncertainty_df,int_lumi, title="Confusion Matrix", filename="conf_matrix_table.png"):
    
    matrix = values_df.values
    uncertainty_values = uncertainty_df.values

    pred_labels = values_df.columns.astype(str)
    true_labels = values_df.index.astype(str)

    true_labels = true_labels.str.replace('data_vetoTagged', 'data (veto tagged)')
    true_labels = true_labels.str.replace('data_vetoFree', 'data (veto free)')

    # Create annotation text with uncertainties
    annotations = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            unc = uncertainty_values[i, j]
            if pd.isna(count) or pd.isna(unc):
                annotations[i, j] = "-"
            elif count == 0:
                annotations[i, j] = '0'
            else:
                annotations[i, j] = format_to_sigfigs(count)

    
    # Replace NaNs with 0 for summation
    safe_matrix = np.nan_to_num(matrix, nan=0.0)

    row_sums = safe_matrix.sum(axis=1)
    col_sums = safe_matrix.sum(axis=0)
    grand_total = safe_matrix.sum()

    # Extend matrix with row totals
    matrix_with_row = np.hstack([matrix, row_sums.reshape(-1, 1)])
    col_sums_with_total = np.append(col_sums, grand_total)
    full_matrix = np.vstack([matrix_with_row, col_sums_with_total.reshape(1, -1)])

    # Create extended annotations
    extended_annotations = np.empty_like(full_matrix, dtype=object)
    extended_annotations[:-1, :-1] = annotations
    for i in range(len(row_sums)):
        extended_annotations[i, -1] = format_to_sigfigs(row_sums[i])
    for j in range(len(col_sums)):
        extended_annotations[-1, j] = format_to_sigfigs(col_sums[j])
    extended_annotations[-1, -1] = format_to_sigfigs(grand_total)

    # Create a masked matrix for heatmap (set total row/column to np.nan)
    masked_matrix = full_matrix.astype(float)
    masked_matrix[-1, :] = np.nan
    
    masked_matrix[:, -1] = np.nan

    # Update labels
    pred_labels = list(pred_labels) + ["Row Total"]
    true_labels = list(true_labels) + ["Column Total"]

    # Plot
    cell_width = 1.6
    cell_height = 0.7
    fig_width = cell_width * full_matrix.shape[1]
    fig_height = cell_height * full_matrix.shape[0] * 1.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(masked_matrix, cmap="Blues")
    ax.set_aspect(cell_height / cell_width)

    # Move x-axis labels to top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_xticklabels(pred_labels)
    ax.set_yticklabels(true_labels)

    # Set up color normalization and colormap
    norm = plt.Normalize(vmin=np.nanmin(masked_matrix), vmax=np.nanmax(masked_matrix))
    cmap = plt.get_cmap("Blues")

    # Annotate all cells
    for i in range(full_matrix.shape[0]):
        for j in range(full_matrix.shape[1]):
            val = full_matrix[i, j]
            is_heatmap_cell = not (i == full_matrix.shape[0] - 1 or j == full_matrix.shape[1] - 1)
            rgba = cmap(norm(val)) if is_heatmap_cell else (1, 1, 1, 1)  # white background for totals
            brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
            text_color = "black" if brightness > 0.5 else "white"
            ax.text(j, i, extended_annotations[i, j], ha="center", va="center", color=text_color)

    ax.set_xlabel("Predicted Class (Top)")
    ax.set_ylabel("True Class (Left)")
    ax.set_title(title)
    #fig.colorbar(im, ax=ax)
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    
    fig.text(
        0.75, 0.90,  # x, y in normalized figure coordinates
        f"$\\int\\!\\mathcal{{L}}\\,dt = {int_lumi:.2f}\\ \\mathrm{{fb}}^{{-1}}$",  # LaTeX-style string
        ha="right",  # horizontal alignment to match TextAlign(31)
        va="top",
        fontsize=12,  
    )

    # Save figure
    os.makedirs("./table", exist_ok=True)
    save_path = os.path.join("./table", filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def cal_uncertainty(group, neutrino_lumi, realdata_lumi, kaon_lumi, neutron_lumi):
    #print(group)
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    values = group[numeric_cols]
    
    # Step 1: Compute relative uncertainties
    upper_uncertainty = values.applymap(lambda x: 1 / np.sqrt(x) if x > 0 else 1.14)
    #print('GNN yields and uncertainty')
    #print(group)
    #print(upper_uncertainty)
    
    # Step 2: Determine normalization factors
    norm_factors = []

    for _, row in group.iterrows():
        cls = row['true_class']
        if cls in ['ve', 'vm', 'vt', 'NC']:
            norm = realdata_lumi / neutrino_lumi
        elif cls == 'kaon':
            norm = realdata_lumi / kaon_lumi
        elif cls == 'neutron':
            norm = realdata_lumi / neutron_lumi
        elif cls == 'data_vetoTagged':
            norm = 1 / 1e8
        else:
            norm = 1  # default: no normalization
        norm_factors.append(norm)

    norm_factors = np.array(norm_factors).reshape(-1, 1)
    #print(norm_factors)
    # Step 3: Apply normalization
    normalized_values = values * norm_factors
    normalized_uncertainty = upper_uncertainty * norm_factors
    #print('after normalization')
    #print(normalized_values)
    #print(normalized_uncertainty)
    

    # Step 4: Add true_class as index
    index = group['true_class'].values
    normalized_values_df = normalized_values.copy()
    normalized_values_df.index = index

    normalized_uncertainty_df = normalized_uncertainty.copy()
    normalized_uncertainty_df.index = index

    return normalized_values_df, normalized_uncertainty_df
    
    
def plot_table(all_matrix, realdata_lumi):
    #print(all_matrix)
    drop_veto_tagged_data = False
    drop_veto_0_data = False
    drop_neutral = False
    drop_neutrino = False
    

    for cut_name, group in all_matrix.groupby('cut'):
        print(cut_name)
        # print(group)
        #if (not(cut_name == 'Prediction_0 > 0.9200000000000004' or cut_name == 'no_cut')):
        #    continue
        #if (cut_name != 'no_cut'):
        #    continue
        
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        normalized_values_df = group[numeric_cols]
        normalized_uncertainty_df = normalized_values_df # for debug

        #normalized_values_df, normalized_uncertainty_df = cal_uncertainty(group, neutrino_lumi, realdata_lumi, kaon_lumi, neutron_lumi)

        if drop_veto_tagged_data:
            mask = ~normalized_values_df.index.isin(['data_vetoTagged'])
            normalized_values_df = normalized_values_df[mask]
            normalized_uncertainty_df = normalized_uncertainty_df[mask]
            
        if drop_veto_0_data:
            mask = ~normalized_values_df.index.isin(['data_vetoFree'])
            normalized_values_df = normalized_values_df[mask]
            normalized_uncertainty_df = normalized_uncertainty_df[mask]

        if drop_neutral:
            mask = ~normalized_values_df.index.isin(['kaon', 'neutron'])
            normalized_values_df = normalized_values_df[mask]
            normalized_uncertainty_df = normalized_uncertainty_df[mask]
        
        if drop_neutrino:
            mask = ~normalized_values_df.index.isin(['ve', 'vm', 'vt', 'NC'])
            normalized_values_df = normalized_values_df[mask]
            normalized_uncertainty_df = normalized_uncertainty_df[mask]

        
        safe_name = cut_name.replace(" ", "_").replace(">", "gt").replace("<", "lt")
        filename = f"full_matrix_{safe_name}.pdf"
        if (cut_name == 'no_cut'):
            title = f""
        else:
            title = f"Confusion Matrix with Uncertainty — {cut_name}"
        title = f""
        print(filename)
        # print(normalized_values_df)
        # print(normalized_uncertainty_df)
        plot_confusion_matrix_with_uncertainty(normalized_values_df, normalized_uncertainty_df, realdata_lumi, title=title, filename=filename)


        #break


def process_row(row):
    matrix_path = row['matrix_baseline_muon_output_path']
    lumi_per_file = np.nan_to_num(row['lumi_per_file'], nan=0.0)
    data_type = row['data_type']
    
    
    matrix = pd.read_csv(matrix_path, index_col=0)
    if "real_data" in data_type:
        veto_ineff = row['veto_ineff']
    else:
        
        veto_ineff = 1
    
    return matrix, lumi_per_file, veto_ineff

def load_matrix_from_csv(df, max_file=1e6, int_lumi_real_data=1e5, process_real_data=False):
    #print(df)
    vetoFree_matrix = None
    vetoTagged_matrix = None
    
    vetoFree_lumi = 0
    vetoTagged_lumi = 0
    

    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        vetoFree_matrix_path = row[f'vetoFree_matrix_{model_name}_output_path']
        vetoFree_lumi_per_file = np.nan_to_num(row['lumi_per_file'], nan=0.0)
        
        if os.path.exists(vetoFree_matrix_path) and os.path.getsize(vetoFree_matrix_path) > 0:
            vf_matrix = pd.read_csv(vetoFree_matrix_path, index_col=0)
            vetoFree_lumi += vetoFree_lumi_per_file
            count+=1
            
            
            
        
            if vetoFree_matrix is None:
                vetoFree_matrix = vf_matrix.copy()
            else:
                vetoFree_matrix[target_columns] = vetoFree_matrix[target_columns].add(vf_matrix[target_columns], fill_value=0)
            
            if process_real_data and (vetoFree_lumi>int_lumi_real_data):
                break
        
        
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
    #print(vetoTagged_matrix)
    return vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi

def load_neutral_bkg_matrix(df):
    results = []

    # Ensure 'energy_range' exists in the DataFrame
    if 'energy_range' not in df.columns:
        raise ValueError("DataFrame must contain an 'energy_range' column.")

    # Group by energy_range
    grouped = df.groupby('energy_range')

    for energy_range, group in grouped:
        # Load matrix and lumi for this energy range
        vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi = load_matrix_from_csv(group)
        #print(matrix)

        results.append((energy_range, (vetoFree_matrix, vetoFree_lumi, vetoTagged_matrix, vetoTagged_lumi)))

    return results

def sum_normalized_neutral_bkg_matrix(matrix_list, factor=1.0):
    total_matrix = None

    for energy_range, (vetoFree_matrix, vetoFree_lumi, _, _) in matrix_list:
        if vetoFree_lumi == 0:
            continue  # Skip to avoid division by zero

        
        normalized_matrix = normalize_matrix(vetoFree_matrix, vetoFree_lumi, factor)

        if total_matrix is None:
            total_matrix = normalized_matrix.copy()
        else:
            total_matrix[target_columns] = total_matrix[target_columns].add(normalized_matrix[target_columns], fill_value=0)

    return total_matrix

def normalize_matrix(matrix, lumi, factor):
    #print(matrix, lumi, factor)
    if lumi == 0:
        raise ValueError("Cannot normalize matrix with lumi=0")
    numeric_cols = matrix.select_dtypes(include='number').columns
    matrix = matrix.copy()
    matrix[numeric_cols] = matrix[numeric_cols] / lumi * factor
    return matrix


def process_matrix(realdata_metadata, MC_neutrino_metadata,MC_muon_metadata,  MC_kaon_metadata, MC_neutron_metadata):
    
    print('reading real data')
    realdata_vetoFree_matrix, realdata_vetoFree_lumi, realdata_vetoTagged_matrix, realdata_vetoTagged_lumi = load_matrix_from_csv(realdata_metadata)
    
    print('reading neutrino')
    MC_neutrino_vetoFree_matrix, MC_neutrino_vetoFree_lumi, _, _ = load_matrix_from_csv(MC_neutrino_metadata)
    
    print('reading muon')
    muon_vetoFree_matrix, muon_vetoFree_lumi, muon_vetoTagged_matrix, muon_vetoTagged_lumi = load_matrix_from_csv(MC_muon_metadata)
    
    print('reading kaon')
    MC_kaon_matrix_list= load_neutral_bkg_matrix(MC_kaon_metadata)
    print('reading neutron')
    MC_neutron_matrix_list = load_neutral_bkg_matrix(MC_neutron_metadata)
    
    normalized_MC_neutrino_vetoFree_matrix = normalize_matrix(MC_neutrino_vetoFree_matrix, MC_neutrino_vetoFree_lumi, realdata_vetoFree_lumi)
    
    #normalized_realdata_vetoTagged_matrix = normalize_matrix(realdata_vetoTagged_matrix, realdata_vetoTagged_lumi, realdata_vetoFree_lumi)
    #print(muon_vetoTagged_matrix)
    muon_combine_matrix = muon_vetoFree_matrix.copy()
    muon_combine_matrix[target_columns] =  muon_vetoFree_matrix[target_columns].add(muon_vetoTagged_matrix[target_columns], fill_value=0)
    muon_combine_lumi = muon_vetoFree_lumi+muon_vetoTagged_lumi
    
    normalise_muon_matrix = normalize_matrix(muon_combine_matrix, muon_combine_lumi, realdata_vetoFree_lumi)
    
    normalized_MC_kaon_vetoFree_matrix = sum_normalized_neutral_bkg_matrix(MC_kaon_matrix_list, realdata_vetoFree_lumi)
    
    normalized_MC_neutron_vetoFree_matrix = sum_normalized_neutral_bkg_matrix(MC_neutron_matrix_list, realdata_vetoFree_lumi)
    #print((normalized_MC_neutron_matrix))
    #print(normalized_MC_neutrino_matrix)
    #print()
    # print(realdata_vetoFree_matrix)
    # print(normalized_MC_neutrino_vetoFree_matrix)
    # print(normalized_MC_kaon_vetoFree_matrix)
    # print(normalized_MC_neutron_vetoFree_matrix)
    
    all_matrix = pd.concat([
        normalized_MC_neutrino_vetoFree_matrix,
        realdata_vetoFree_matrix,
        normalise_muon_matrix,
    #    normalized_realdata_vetoTagged_matrix,
        normalized_MC_kaon_vetoFree_matrix,
        normalized_MC_neutron_vetoFree_matrix
    ], ignore_index=False)
    
    plot_table(all_matrix, realdata_vetoFree_lumi)




METADATA_dict = read_metadata()
model_name = 'GravNet_v4'

def main():

    
    neutrino_df = METADATA_dict['MC_neutrino']
    muon_df = METADATA_dict['MC_muon']
    kaon_df = METADATA_dict['MC_kaon']
    neutron_df = METADATA_dict['MC_neutron']
    real_data = METADATA_dict['real_data_2024']
    
    
    
    
    process_matrix(real_data, neutrino_df,muon_df, kaon_df, neutron_df)
    
    
    



if __name__ == "__main__": 

    
    main()
