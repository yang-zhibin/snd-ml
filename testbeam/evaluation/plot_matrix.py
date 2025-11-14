import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

particle_2_class = {
    'e-': 0,
    'pi': 1,
}
class_2_particle = {v: k for k, v in particle_2_class.items()}
target_columns = ['e-', 'pi']


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

    true_labels = true_labels.str.replace('data_veto_tagged', 'data (veto tagged)')
    true_labels = true_labels.str.replace('data_zero_veto', 'data (veto clean)')

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
        elif cls == 'data_veto_tagged':
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
        #print(cut_name)
        #if (not(cut_name == 'Prediction_0 > 0.9200000000000004' or cut_name == 'no_cut')):
        #    continue
        #if (cut_name != 'no_cut'):
        #    continue
        
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        normalized_values_df = group[numeric_cols]
        normalized_uncertainty_df = normalized_values_df # for debug

        #normalized_values_df, normalized_uncertainty_df = cal_uncertainty(group, neutrino_lumi, realdata_lumi, kaon_lumi, neutron_lumi)

        if drop_veto_tagged_data:
            mask = ~normalized_values_df.index.isin(['data_veto_tagged'])
            normalized_values_df = normalized_values_df[mask]
            normalized_uncertainty_df = normalized_uncertainty_df[mask]
            
        if drop_veto_0_data:
            mask = ~normalized_values_df.index.isin(['data_zero_veto'])
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
        
        #drop rows data_veto_tagged and data_zero_veto if drop_data
        #drop rows kaon and muon if drop_neutral
        
        safe_name = cut_name.replace(" ", "_").replace(">", "gt").replace("<", "lt")
        filename = f"new_normalisation_full_matrix_{safe_name}.pdf"
        if (cut_name == 'no_cut'):
            title = f""
        else:
            title = f"Confusion Matrix with Uncertainty — {cut_name}"
        title = f""
        print(filename)
        plot_confusion_matrix_with_uncertainty(normalized_values_df, normalized_uncertainty_df, realdata_lumi, title=title, filename=filename)


        #break



def select_eval_neutrion(MC_neutrino):
    train_csv = '/eos/user/z/zhibin/sndData/converted/combined_train.csv'
    train_df = pd.read_csv(train_csv)
    
    # Filter to Neutrinos
    train_df = train_df[train_df['partition'] == 'Neutrinos'].copy()
    
    # Extract partition
    train_df['partition'] = train_df['file'].str.extract(r'/Neutrinos/(\d+)/sndLHC')[0]
    train_df.dropna(subset=['partition'], inplace=True)

    # Ensure type consistency
    train_partitions = train_df['partition'].astype(str).unique()
    MC_neutrino['partition'] = MC_neutrino['partition'].astype(str)

    # Debug print of matching rows
    matching_rows = MC_neutrino[MC_neutrino['partition'].isin(train_partitions)]
    #print("Dropping the following paths:")
    #print(matching_rows[['partition', 'digi_path']])

    # Filter out training partitions
    MC_neutrino = MC_neutrino[~MC_neutrino['partition'].isin(train_partitions)]
    #print(MC_neutrino)
    return MC_neutrino


def plot_scifi_cut_eff(matrix):
    Scifi_hit_cuts = [
            "no_cut",
            "scifi_gt_50", "scifi_gt_100", "scifi_gt_150", "scifi_gt_200",
            "scifi_gt_250", "scifi_gt_300", "scifi_gt_350", "scifi_gt_400", "scifi_gt_450",
            "scifi_gt_500",
    ]
    # Define x-axis: corresponding number of scifi hits for each cut
    x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    assert len(Scifi_hit_cuts) == len(x), "Mismatch between cut labels and x values"
    
    # Prepare storage for yields
    kaon_yields = []
    neutron_yields = []
    muon_yields = []

    # Sort to ensure consistent order
    matrix = matrix.set_index('cut').loc[Scifi_hit_cuts].reset_index()
    print(matrix)

    for cut_name in Scifi_hit_cuts:
        group = matrix[matrix['cut'] == cut_name]
        row = group[group['true_class'] == 'data_zero_veto']

        if not row.empty:
            kaon_yield = row['kaon'].values[0]
            neutron_yield = row['neutron'].values[0]
            muon_yield = row['muon'].values[0]
        else:
            kaon_yield = neutron_yield = muon_yield = 0  # fallback if row is missing

        kaon_yields.append(kaon_yield)
        neutron_yields.append(neutron_yield)
        muon_yields.append(muon_yield)

    # Create output directory if it doesn't exist
    os.makedirs('./cut_eff/', exist_ok=True)

    colors = {
        'kaon': '#ffa500',     # ROOT.kOrange + 7 (approx)
        'neutron': '#00ffff',  # ROOT.kCyan
        'muon': '#000000'      # ROOT.kBlack
    }

        
    fig, ax = plt.subplots(figsize=(8, 3.5))

    def inverse_sqrt_safe(arr):
        arr = np.array(arr, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 1 / np.sqrt(arr)
            result[~np.isfinite(result)] = 0  # Set inf and nan to 0
        return result

    kaon_errors = inverse_sqrt_safe(kaon_yields)
    neutron_errors = inverse_sqrt_safe(neutron_yields)
    muon_errors = inverse_sqrt_safe(muon_yields)
    # Plot with error bars
    ax.errorbar(x, kaon_yields, yerr=kaon_errors, marker='o', linestyle='--',
                color=colors['kaon'], label='Kaon', capsize=3)

    ax.errorbar(x, neutron_yields, yerr=neutron_errors, marker='o', linestyle='--',
                color=colors['neutron'], label='Neutron', capsize=3)

    ax.errorbar(x, muon_yields, yerr=muon_errors, marker='o', linestyle='--',
                color=colors['muon'], label='Muon', capsize=3)

    # Labels, grid, formatting
    ax.set_xlabel("SciFi Hit Cut Threshold")
    ax.set_ylabel("Yield")
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    # Title outside the plot
    fig.text(
        0.01, 0.99, "Effect of SciFi Hit Cut at Control Region",
        ha='left', va='top',
        fontsize=13, #fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    plt.savefig('./cut_eff/scifi_cut_eff.pdf')
    plt.close()
        
    
    #create a plot and save to ./cut_eff/

def plot_score_cut_eff(all_matrix):
    Prediction_0_score_cuts = [
        "no_cut",
        "Prediction_0 > 0.50", "Prediction_0 > 0.52", "Prediction_0 > 0.54", "Prediction_0 > 0.56",
        "Prediction_0 > 0.58", "Prediction_0 > 0.60", "Prediction_0 > 0.62", "Prediction_0 > 0.64",
        "Prediction_0 > 0.66", "Prediction_0 > 0.68", "Prediction_0 > 0.70", "Prediction_0 > 0.72",
        "Prediction_0 > 0.74", "Prediction_0 > 0.76", "Prediction_0 > 0.78", "Prediction_0 > 0.80",
        "Prediction_0 > 0.82", "Prediction_0 > 0.84", "Prediction_0 > 0.86", "Prediction_0 > 0.88",
        "Prediction_0 > 0.90", "Prediction_0 > 0.92", "Prediction_0 > 0.94", "Prediction_0 > 0.96",
        "Prediction_0 > 0.98", "Prediction_0 > 0.99",
    ]

    # Define x-axis: corresponding score values (extract float from string)
    x = [0.0] + [float(cut.split(">")[1].strip()) for cut in Prediction_0_score_cuts[1:]]
    assert len(Prediction_0_score_cuts) == len(x), "Mismatch between cut labels and x values"

    # Prepare storage for yields
    sig_yields = []

    # Ensure consistent cut ordering
    matrix = all_matrix.set_index('cut').loc[Prediction_0_score_cuts].reset_index()
    print(matrix)

    for cut_name in Prediction_0_score_cuts:
        group = matrix[matrix['cut'] == cut_name]
        row = group[group['true_class'] == 'data_zero_veto']

        if not row.empty:
            kaon_yield = row['kaon'].values[0]
            neutron_yield = row['neutron'].values[0]
            muon_yield = row['muon'].values[0]
        else:
            kaon_yield = neutron_yield = muon_yield = 0

        kaon_yields.append(kaon_yield)
        neutron_yields.append(neutron_yield)
        muon_yields.append(muon_yield)

    os.makedirs('./cut_eff/', exist_ok=True)

    colors = {
        'kaon': '#ffa500',
        'neutron': '#00ffff',
        'muon': '#000000'
    }

    def inverse_sqrt_safe(arr):
        arr = np.array(arr, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 1 / np.sqrt(arr)
            result[~np.isfinite(result)] = 0
        return result

    kaon_errors = inverse_sqrt_safe(kaon_yields)
    neutron_errors = inverse_sqrt_safe(neutron_yields)
    muon_errors = inverse_sqrt_safe(muon_yields)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.errorbar(x, kaon_yields, yerr=kaon_errors, marker='o', linestyle='--',
                color=colors['kaon'], label='Kaon', capsize=3)
    ax.errorbar(x, neutron_yields, yerr=neutron_errors, marker='o', linestyle='--',
                color=colors['neutron'], label='Neutron', capsize=3)
    ax.errorbar(x, muon_yields, yerr=muon_errors, marker='o', linestyle='--',
                color=colors['muon'], label='Muon', capsize=3)

    ax.set_xlabel("Prediction_0 Score Threshold")
    ax.set_ylabel("Yield")
    ax.set_yscale('log')
    ax.set_xticks(x[::2])  # Show every second tick for clarity
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    fig.text(
        0.01, 0.99, "Effect of Prediction_0 Score Cut at Control Region",
        ha='left', va='top',
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('./cut_eff/prediction0_cut_eff.pdf')
    plt.close()
    

def process_cut_efficiency(full_matrix):
    neutrino_matrix, neutrino_lumi = full_matrix["neutrino"]
    realdata_matrix, realdata_lumi = full_matrix["real_data_2024"]
    kaon_matrix, kaon_lumi = full_matrix["kaon"]
    neutron_matrix, neutron_lumi = full_matrix["neutron"]
    
    all_matrix = pd.concat([
        neutrino_matrix,
        realdata_matrix,
        kaon_matrix,
        neutron_matrix
    ], ignore_index=False)
    
    all_matrix.index.name = 'true_class'
    all_matrix = all_matrix.reset_index()
    
    plot_scifi_cut_eff(all_matrix)
    
    #plot_score_cut_eff(all_matrix)
    
    Prediction_0_score_cuts =  [
            "no_cut",
            "Prediction_0 > 0.50",
            "Prediction_0 > 0.52",
            "Prediction_0 > 0.54",
            "Prediction_0 > 0.56",
            "Prediction_0 > 0.58",
            "Prediction_0 > 0.60",
            "Prediction_0 > 0.62",
            "Prediction_0 > 0.64",
            "Prediction_0 > 0.66",
            "Prediction_0 > 0.68",
            "Prediction_0 > 0.70",
            "Prediction_0 > 0.72",
            "Prediction_0 > 0.74",
            "Prediction_0 > 0.76",
            "Prediction_0 > 0.78",
            "Prediction_0 > 0.80",
            "Prediction_0 > 0.82",
            "Prediction_0 > 0.84",
            "Prediction_0 > 0.86",
            "Prediction_0 > 0.88",
            "Prediction_0 > 0.90",
            "Prediction_0 > 0.92",
            "Prediction_0 > 0.94",
            "Prediction_0 > 0.96",
            "Prediction_0 > 0.98",
            "Prediction_0 > 0.99",
        ]


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

def load_matrix_from_csv(df, max_file=1e5):
    #print(df)
    particle_matrix = None
    particle_lumi = 0
    

    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        #print(idx,row)
        matrix, lumi_per_file, veto_ineff = process_row(row)
        if "data_veto_tagged" in matrix.index:
            numeric_cols = matrix.select_dtypes(include='number').columns
            matrix.loc["data_veto_tagged", numeric_cols] *= veto_ineff
            #print(matrix)
        
        #print(matrix)
        # add lumi to particle_matrix
        particle_lumi += lumi_per_file

        # add matrix to particle_matrix
        if particle_matrix is None:
            particle_matrix = matrix.copy()
        else:
            # Add values for summable keys only
            particle_matrix[target_columns] = particle_matrix[target_columns].add(matrix[target_columns], fill_value=0)
        
        count+=1
        if count > max_file:
            break
    #print(particle_matrix)
    return particle_matrix, particle_lumi

def load_neutral_bkg_matrix(df):
    results = []

    # Ensure 'energy_range' exists in the DataFrame
    if 'energy_range' not in df.columns:
        raise ValueError("DataFrame must contain an 'energy_range' column.")

    # Group by energy_range
    grouped = df.groupby('energy_range')

    for energy_range, group in grouped:
        # Load matrix and lumi for this energy range
        matrix, lumi = load_matrix_from_csv(group)
        #print(matrix)

        # Append the result as a tuple: (energy_range, (matrix, lumi))
        results.append((energy_range, (matrix, lumi)))

    return results

def sum_normalized_neutral_bkg_matrix(matrix_list, factor=1.0):
    total_matrix = None

    for energy_range, (matrix, lumi) in matrix_list:
        if lumi == 0:
            continue  # Skip to avoid division by zero

        
        normalized_matrix = normalize_matrix(matrix, lumi, factor)

        if total_matrix is None:
            total_matrix = normalized_matrix.copy()
        else:
            total_matrix[target_columns] = total_matrix[target_columns].add(normalized_matrix[target_columns], fill_value=0)

    return total_matrix

def normalize_matrix(matrix, lumi, factor):
    if lumi == 0:
        raise ValueError("Cannot normalize matrix with lumi=0")
    numeric_cols = matrix.select_dtypes(include='number').columns
    matrix = matrix.copy()
    matrix[numeric_cols] = matrix[numeric_cols] / lumi * factor
    return matrix


def process_matrix(realdata_metadata, MC_neutrino_metadata, MC_kaon_metadata, MC_neutron_metadata):
    
    realdata_matrix, realdata_lumi = load_matrix_from_csv(realdata_metadata, 700)
    #print(realdata_matrix)
    MC_neutrino_matrix, MC_neutrino_lumi = load_matrix_from_csv(MC_neutrino_metadata)
    
    MC_kaon_matrix_list= load_neutral_bkg_matrix(MC_kaon_metadata)
    MC_neutron_matrix_list = load_neutral_bkg_matrix(MC_neutron_metadata)
    
    normalized_MC_neutrino_matrix = normalize_matrix(MC_neutrino_matrix, MC_neutrino_lumi, realdata_lumi)
    
    normalized_MC_kaon_matrix = sum_normalized_neutral_bkg_matrix(MC_kaon_matrix_list, realdata_lumi)
    
    
    #print((MC_kaon_matrix_list))
    normalized_MC_neutron_matrix = sum_normalized_neutral_bkg_matrix(MC_neutron_matrix_list, realdata_lumi)
    #print((normalized_MC_neutron_matrix))
    #print(normalized_MC_neutrino_matrix)
    #print()
    
    all_matrix = pd.concat([
        normalized_MC_neutrino_matrix,
        realdata_matrix,
        normalized_MC_kaon_matrix,
        normalized_MC_neutron_matrix
    ], ignore_index=False)
    
    plot_table(all_matrix, realdata_lumi)


def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data
        
def drop_missing_files(df: pd.DataFrame, column_name: str, metadata_name: str = "") -> pd.DataFrame:
    """Drop rows where the file in column_name does not exist. Print summary per metadata."""
    exists_mask = df[column_name].apply(lambda path: os.path.exists(path))
    missing_count = (~exists_mask).sum()

    if metadata_name:
        print(f"{metadata_name}: {missing_count} missing files in '{column_name}'")
    else:
        print(f"{missing_count} missing files in '{column_name}'")

    return df[exists_mask].reset_index(drop=True)

def select_eval_neutrion(MC_neutrino):
    train_csv = '/eos/user/z/zhibin/sndData/converted/combined_train.csv'
    train_df = pd.read_csv(train_csv)
    
    # Filter to Neutrinos
    train_df = train_df[train_df['partition'] == 'Neutrinos'].copy()
    
    # Extract partition
    train_df['partition'] = train_df['file'].str.extract(r'/Neutrinos/(\d+)/sndLHC')[0]
    train_df.dropna(subset=['partition'], inplace=True)

    # Ensure type consistency
    train_partitions = train_df['partition'].astype(str).unique()
    MC_neutrino['partition'] = MC_neutrino['partition'].astype(str)

    # Debug print of matching rows
    matching_rows = MC_neutrino[MC_neutrino['partition'].isin(train_partitions)]
    print("Dropping the following paths:")
    #print(matching_rows[['partition', 'digi_path']])

    # Filter out training partitions
    MC_neutrino = MC_neutrino[~MC_neutrino['partition'].isin(train_partitions)]
    #print(MC_neutrino)
    return MC_neutrino

def main():
    mc_files = [
        "MC_kaon_FTFP_BERT_metadata.csv",
        "MC_neutron_FTFP_BERT_metadata.csv",
        "MC_muon_down_metadata.csv",
        "MC_muon_horizontal_metadata.csv",
        "MC_muon_up_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
        "real_data_2024_metadata.csv",
    ]

    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'

    metadata_vars = load_metadata_files(mc_files, root_path)

    MC_kaon_FTFP_BERT = metadata_vars["MC_kaon_FTFP_BERT"]
    MC_neutron_FTFP_BERT = metadata_vars["MC_neutron_FTFP_BERT"]
    MC_muon_down = metadata_vars["MC_muon_down"]
    MC_muon_horizontal = metadata_vars["MC_muon_horizontal"]
    MC_muon_up = metadata_vars["MC_muon_up"]
    MC_neutrino_volTarget_100fb_1 = metadata_vars["MC_neutrino_volTarget_100fb_1"]
    real_data_2024 = metadata_vars["real_data_2024"]
    
    # Drop missing 'feature_path' files
    MC_muon_down = drop_missing_files(MC_muon_down, "eval_baseline_muon_output_path", "MC_muon_down")
    MC_muon_horizontal = drop_missing_files(MC_muon_horizontal, "eval_baseline_muon_output_path", "MC_muon_horizontal")
    MC_muon_up = drop_missing_files(MC_muon_up, "eval_baseline_muon_output_path", "MC_muon_up")
    MC_muon = pd.concat([MC_muon_down, MC_muon_horizontal, MC_muon_up], ignore_index=True)
    
    
    #Drop missing 'eval_baseline_muon_output_path' files
    MC_kaon_FTFP_BERT = drop_missing_files(MC_kaon_FTFP_BERT, "matrix_baseline_muon_output_path", "MC_kaon_FTFP_BERT")
    MC_neutron_FTFP_BERT = drop_missing_files(MC_neutron_FTFP_BERT, "matrix_baseline_muon_output_path", "MC_neutron_FTFP_BERT")
    MC_neutrino_volTarget_100fb_1 = drop_missing_files(MC_neutrino_volTarget_100fb_1, "matrix_baseline_muon_output_path", "MC_neutrino_volTarget_100fb_1")
    
    real_data_2024 = drop_missing_files(real_data_2024, "matrix_baseline_muon_output_path", "real_data_2024")
    
    
    #process_n_hits_in_diff_energy(MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT)
    
    #select portion of kaon and neutron()
    MC_neutrino_volTarget_100fb_1 = select_eval_neutrion(MC_neutrino_volTarget_100fb_1)
    
    
    process_matrix(real_data_2024, MC_neutrino_volTarget_100fb_1, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT)
    
    
    
    
    


if __name__ == "__main__": 
    main()
