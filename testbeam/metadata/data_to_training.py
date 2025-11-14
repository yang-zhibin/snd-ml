import pandas as pd
import os
import re
import matplotlib.pyplot as plt
# import ROOT
from collections import defaultdict
from tqdm import tqdm

report_outdir = "./metadata_reports/"
os.makedirs(report_outdir, exist_ok=True)


particle_to_target = {
        12: 0, -12: 0,
        14: 1, -14: 1,
        16: 2, -16: 2,
        112: 3, -112: 3, 114: 3, -114: 3, 116: 3, -116: 3,
        130: 4, 310: 4,
        2112: 5,
        13:6, -13:6,
        0:6
}
particle_mapping = {
    12: 've', -12: 've',
    14: 'vm', -14: 'vm',
    16: 'vt', -16: 'vt',
    112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC', 116: 'NC', -116: 'NC',
    130: 'kaon', 310: 'kaon',
    2112: 'neutron',
    13:'muon', -13:'muon',
    0:'data'
}

def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data

def check_neutral_bkg_dataset(df, dataset_name):
    pat = re.compile(r"""
        (?P<model>[^/]+)/                    # mc_model_type
        (?P<particle>[^_]+)_                 # particle
        (?P<E_low>\d+\.?\d*)_                # E_low
        (?P<E_high>\d+\.?\d*)                # E_high
        (?:_.*)?                             # optional suffix
    """, re.VERBOSE)


    def _parse(row):
        m = pat.fullmatch(row["subfolder"])
        if m is None:
            raise ValueError(f"Unparsable subfolder: {row['subfolder']}")
        gd = m.groupdict()
        return pd.Series(
            {
                "mc_model": gd["model"],
                "particle": gd["particle"],
                "E_low": float(gd["E_low"]),
                "E_high": float(gd["E_high"]),
            }
        )

    df = df.join(df.apply(_parse, axis=1))
    # List to hold aggregated values
    summary = []

    # Group by 'E_low'
    for e_low, group in df.groupby('E_low'):
        total_lumi = group['lumi_per_file'].sum()
        total_events = group['n_event'].sum()
        e_high = group['E_high'].iloc[0]
        summary.append((e_low, e_high, total_lumi, total_events))

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary, columns=['E_low', 'E_high', 'lumi', 'n_event'])

    # Format labels with units
    lumi_labels = [
        f"{row.E_low}-{row.E_high} GeV ({row.lumi:.2e} pb⁻¹)"
        for row in summary_df.itertuples()
    ]
    event_labels = [
        f"{row.E_low}-{row.E_high} GeV ({row.n_event:.2e} events)"
        for row in summary_df.itertuples()
    ]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def autopct_func(pct):
        return f'{pct:.1f}%' if pct > 5 else ''

    # Lumi pie chart
    wedges1, _, _ = axes[0].pie(
        summary_df['lumi'],
        autopct=autopct_func,
        startangle=90
    )
    axes[0].set_title('Lumi Distribution')
    axes[0].legend(wedges1, lumi_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].axis('equal')

    # Event pie chart
    wedges2, _, _ = axes[1].pie(
        summary_df['n_event'],
        autopct=autopct_func,
        startangle=90
    )
    axes[1].set_title('Event Count Distribution')
    axes[1].legend(wedges2, event_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].axis('equal')

    plt.tight_layout()
    output_path = os.path.join(report_outdir, f"{dataset_name}_lumi_and_event_pie_charts.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Pie charts saved to: {output_path}")
    

def plot_muon_bkg_dataset(df, dataset_name):
    summary = []

    # Summarize lumi, n_event, and preSelect per partition
    for partition, group in df.groupby('partition'):
        total_lumi = group['lumi_per_file'].sum()
        total_events = group['n_event'].sum()
        total_preselect = group['preSelect'].sum()  # assuming this column exists
        summary.append((partition, total_lumi, total_events, total_preselect))

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary, columns=['partition', 'lumi', 'n_event', 'preSelect'])

    # Prepare labels with scientific notation
    lumi_labels = [
        f"{row.partition} ({row.lumi:.2e} pb⁻¹)"
        for row in summary_df.itertuples()
    ]
    event_labels = [
        f"{row.partition} ({row.n_event:.2e} events)"
        for row in summary_df.itertuples()
    ]
    preSelect_labels = [
        f"{row.partition} ({row.preSelect:.2e} events)"
        for row in summary_df.itertuples()
    ]


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # adjust layout for 3 charts

    def autopct_func(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    # Lumi pie
    wedges1, _, _ = axes[0].pie(
        summary_df['lumi'], autopct=autopct_func, startangle=90
    )
    axes[0].set_title('Lumi Distribution')
    axes[0].legend(wedges1, lumi_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].axis('equal')

    # Event pie
    wedges2, _, _ = axes[1].pie(
        summary_df['n_event'], autopct=autopct_func, startangle=90
    )
    axes[1].set_title('Event Count Distribution')
    axes[1].legend(wedges2, event_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].axis('equal')

    # PreSelect pie
    wedges3, _, _ = axes[2].pie(
        summary_df['preSelect'], autopct=autopct_func, startangle=90
    )
    axes[2].set_title('PreSelected Event Distribution')
    axes[2].legend(wedges3, preSelect_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].axis('equal')

    # Save plot
    plt.tight_layout()
    output_path = os.path.join(report_outdir, f"{dataset_name}_partition_pie_charts.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Saved partition pie charts to: {output_path}")

def check_muon_bkg_dataset(df,dataset_name):
    tagged_counts = []
    free_counts = []

    for i, row in df.iterrows():
        root_file = row["preSelect_path"]

        try:
            rdf = ROOT.RDataFrame("sndData", root_file)

            # Count where the branch equals 1
            tagged_count = rdf.Filter("preSelect_vetoTagged == 1").Count().GetValue()
            free_count = rdf.Filter("preSelect_vetoFree == 1").Count().GetValue()
        except Exception as e:
            print(f"Error processing {root_file}: {e}")
            tagged_count = -1
            free_count = -1

        tagged_counts.append(tagged_count)
        free_counts.append(free_count)

    df[f"preSelect_vetoTagged"] = tagged_counts
    df[f"preSelect_vetoFree"] = free_counts
    df[f"preSelect"] = df[f"preSelect_vetoTagged"] + df[f"preSelect_vetoFree"]
    
   
    plot_muon_bkg_dataset(df, dataset_name)
    
def check_neutrion_bkg_dataset(df,dataset_name):
    # read ve,vm,vt, NC for total, preSelect_vetoTagged and preSelect_vetoFree
    #
    # Initialize counters
    total_counts = defaultdict(int)
    veto_tagged_counts = defaultdict(int)
    veto_free_counts = defaultdict(int)

    for i, row in df.iterrows():
        root_file = row["preSelect_path"]

        try:
            f = ROOT.TFile.Open(root_file)
            if not f or f.IsZombie():
                raise IOError("File could not be opened")

            tree = f.Get("sndData")
            if not tree:
                raise ValueError("Tree 'sndData' not found in file")

            for event in tree:
                pdg = event.pdgCode
                total_counts[particle_mapping[pdg]] += 1
                veto = event.veto
                if veto > 0:
                    veto_tagged_counts[particle_mapping[pdg]] += 1
                else:
                    veto_free_counts[particle_mapping[pdg]] += 1
            f.Close()

        except Exception as e:
            print(f"Error processing {root_file}: {e}")
            continue
        
    particle_colors = {
        've': '#1f77b4',
        'vm': '#ff7f0e',
        'vt': '#2ca02c',
        'NC': '#d62728',
        'kaon': '#9467bd',
        'neutron': '#8c564b',
        'muon': '#e377c2',
        'data': '#7f7f7f'
    }
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    def plot_pie(ax, counts_dict, title):
        items = [(label, counts_dict[label]) for label in particle_colors if counts_dict[label] > 0]
        if not items:
            ax.set_title(f"{title}\n(no data)")
            ax.axis('off')
            return

        labels, sizes = zip(*items)
        colors = [particle_colors[label] for label in labels]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors
        )
        ax.set_title(title)

        # Add custom legend with counts
        legend_labels = [f"{lbl} ({cnt})" for lbl, cnt in zip(labels, sizes)]
        ax.legend(wedges, legend_labels, title="Particles", loc='upper right', bbox_to_anchor=(1.3, 1.0))



    plot_pie(axs[0], total_counts, "Total Particles")
    plot_pie(axs[1], veto_tagged_counts, "Veto Tagged")
    plot_pie(axs[2], veto_free_counts, "Veto Free")

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(report_outdir, f"{dataset_name}_pie_charts.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pie chart to {output_path}")
    
    # Print total event counts per category
    print(f"\nSummary for {dataset_name}:")
    print(f"  Total particles:       {sum(total_counts.values())}")
    print(f"  Veto-tagged particles: {sum(veto_tagged_counts.values())}")
    print(f"  Veto-free particles:   {sum(veto_free_counts.values())}")

def check_MC_data():
    mc_files = [
        "MC_data_testbeam2024_metadata.csv",
        "MC_data_testbeam2023_metadata.csv"
    ]

    root_path = '/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)

    MC_data_testbeam2023 = metadata_vars["MC_data_testbeam2023"]
    MC_data_testbeam2024 = metadata_vars["MC_data_testbeam2024"]
    
    # Print event counts by subfolder for 2023 data
    print("\nMC_data_testbeam2023 event counts by subfolder:")
    subfolder_counts_2023 = MC_data_testbeam2023.groupby('partition')['n_event'].sum()
    for subfolder, count in subfolder_counts_2023.items():
        print(f"{subfolder}: {count:,} events")
        
    # Print event counts by subfolder for 2024 data  
    print("\nMC_data_testbeam2024 event counts by subfolder:")
    subfolder_counts_2024 = MC_data_testbeam2024.groupby('subfolder')['n_event'].sum()
    for subfolder, count in subfolder_counts_2024.items():
        print(f"{subfolder}: {count:,} events")


def get_2024_train_set():
    mc_files = [
        "MC_data_testbeam2024_metadata.csv",
        "MC_data_testbeam2023_metadata.csv"
    ]

    root_path = '/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)

    MC_data_testbeam2024 = metadata_vars["MC_data_testbeam2024"]

    # ---- Split by cumulative count (not row count) ----
    # Work on a copy
    df = MC_data_testbeam2024.copy()
    # Ensure numeric
    df["n_event"] = pd.to_numeric(df["n_event"], errors="coerce").fillna(0).astype(int)

    # Default everything to 'test'
    df["split"] = "test"

    # ---- 1) Split beam_type == 'pi+' into 4:1:5 (train:val:test) by cumulative n_event per beam_energy ----
    pi_mask = df["beam_type"] == "pi+"
    pi_energies = sorted(df.loc[pi_mask, "beam_energy"].dropna().unique())

    for energy in pi_energies:
        mask_e = pi_mask & (df["beam_energy"] == energy)
        idxs = df[mask_e].index.tolist()
        total_pi_energy = df.loc[idxs, "n_event"].sum()
        if total_pi_energy <= 0:
            print(f"No pi+ events for energy {energy}, skipping.")
            continue

        train_target = int(0.3 * total_pi_energy)
        val_target = int(0.1 * total_pi_energy)

        acc_train = 0
        acc_val = 0

        for idx in idxs:
            cnt = int(df.at[idx, "n_event"])
            if acc_train < train_target:
                df.at[idx, "split"] = "train"
                acc_train += cnt
            elif acc_val < val_target:
                df.at[idx, "split"] = "val"
                acc_val += cnt
            else:
                df.at[idx, "split"] = "test"

        print(f"pi+ energy {energy}: total={total_pi_energy}, train={acc_train}, val={acc_val}, test={total_pi_energy - acc_train - acc_val}")

    # ---- 2) For beam_type == 'e-' select same amount of events per beam_energy for train/val as pi+ got ----
    e_mask = df["beam_type"] == "e-"
    e_energies = sorted(df.loc[e_mask, "beam_energy"].dropna().unique())

    for energy in e_energies:
        # Determine targets from pi+ assignments for this energy
        pi_train_assigned = df.loc[(df["beam_type"] == "pi+") & (df["beam_energy"] == energy) & (df["split"] == "train"), "n_event"].sum()
        pi_val_assigned = df.loc[(df["beam_type"] == "pi+") & (df["beam_energy"] == energy) & (df["split"] == "val"), "n_event"].sum()

        # If no pi+ for this energy, fall back to a 4:1 split within e- itself
        if pi_train_assigned == 0 and pi_val_assigned == 0:
            # fallback: split e- for this energy into 40%/10%/50%
            mask_e = e_mask & (df["beam_energy"] == energy)
            idxs = df[mask_e].index.tolist()
            total_e_energy = df.loc[idxs, "n_event"].sum()
            if total_e_energy <= 0:
                print(f"No e- events for energy {energy}, skipping.")
                continue
            train_target = int(0.4 * total_e_energy)
            val_target = int(0.1 * total_e_energy)

            acc_train = 0
            acc_val = 0
            for idx in idxs:
                cnt = int(df.at[idx, "n_event"])
                if acc_train < train_target:
                    df.at[idx, "split"] = "train"
                    acc_train += cnt
                elif acc_val < val_target:
                    df.at[idx, "split"] = "val"
                    acc_val += cnt
                else:
                    df.at[idx, "split"] = "test"
            print(f"e- energy {energy} (fallback): total={total_e_energy}, train={acc_train}, val={acc_val}, test={total_e_energy - acc_train - acc_val}")
            continue

        remaining_train = int(pi_train_assigned)
        remaining_val = int(pi_val_assigned)

        acc_train_e = 0
        acc_val_e = 0

        mask_e = e_mask & (df["beam_energy"] == energy)
        idxs = df[mask_e].index.tolist()

        for idx in idxs:
            cnt = int(df.at[idx, "n_event"])
            if acc_train_e < remaining_train:
                df.at[idx, "split"] = "train"
                acc_train_e += cnt
            elif acc_val_e < remaining_val:
                df.at[idx, "split"] = "val"
                acc_val_e += cnt
            else:
                df.at[idx, "split"] = "test"

        print(f"e- energy {energy}: assigned train={acc_train_e} (target {remaining_train}), val={acc_val_e} (target {remaining_val})")
    

    
    

    return df[df["split"].isin(["train", "val"])].copy()


def get_vetoFree_event_count(root_path):
    try:
        f = ROOT.TFile.Open(root_path)
        if not f or f.IsZombie():
            return 0
        tree = f.Get("sndData")
        if not tree:
            return 0
        count = tree.GetEntries("preSelect_vetoFree == 1")
        
        f.Close()
        return count
    except Exception as e:
        print(f"Error reading {root_path}: {e}")
        return 0

def process_neutral_bkg_df(df):
    target_events = {
        '5-10': 50_000,
        '10-20': 10_000,
        '20-30': 5_000,
        '30-40': 5_000,
        '40-50': 5_000,
        '50-60': 5_000,
        '60-70': 5_000,
        '70-90': 5_000,
    }
    def canonical_energy_range(er_str):
        try:
            er_tuple = eval(er_str)
            return f"{int(er_tuple[0])}-{int(er_tuple[1])}"
        except Exception:
            return "unknown"

    df["energy_bin"] = df["energy_range"].apply(canonical_energy_range)
    selected_rows = []
    grouped = df.groupby("energy_bin")
    
    for energy_range, group in grouped:
        print(energy_range)
        target = target_events.get(energy_range, 0)
        if target == 0:
            continue

        train_target = target
        val_target = int(0.2 * target)
        train_accum = 0
        val_accum = 0

        print(f"\nProcessing energy range {energy_range} (target: {target} events)")

        for _, row in tqdm(group.iterrows(), total=len(group), desc=f"{energy_range} progress", unit="file"):
            root_file = row["preSelect_path"]
            vetoFree_count = get_vetoFree_event_count(root_file)  # or use get_vetoFree_event_count if filtering is needed

            if vetoFree_count == 0:
                continue

            row = row.copy()
            row["vetoFree_count"] = vetoFree_count

            if train_accum < train_target:
                row["split"] = "train"
                train_accum += vetoFree_count
                selected_rows.append(row)

            elif val_accum < val_target:
                row["split"] = "val"
                val_accum += vetoFree_count
                selected_rows.append(row)

            # Update tqdm description with % completed
            total_accum = train_accum + val_accum
            tqdm.write(f"  -> {total_accum}/{target} accumulated ({100 * total_accum / target:.1f}%)")

            if train_accum >= train_target and val_accum >= val_target:
                break

    return pd.DataFrame(selected_rows)

def get_neutral_bkg_train_set():
    mc_files = [
    "MC_kaon_FTFP_BERT_metadata.csv",
    "MC_neutron_FTFP_BERT_metadata.csv",
    ]
    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)
    
    MC_kaon_FTFP_BERT = metadata_vars["MC_kaon_FTFP_BERT"]
    MC_neutron_FTFP_BERT = metadata_vars["MC_neutron_FTFP_BERT"]
    MC_kaon_train_df = process_neutral_bkg_df(MC_kaon_FTFP_BERT)
    MC_neutron_train_df = process_neutral_bkg_df(MC_neutron_FTFP_BERT)
    
    #print(MC_kaon_train_df)
    #print(MC_neutron_train_df)

    combined_df = pd.concat([MC_kaon_train_df, MC_neutron_train_df], ignore_index=True)
    return combined_df
    
def get_muon_train_set():
    mc_files = [
    "MC_muon_down_metadata.csv",
    "MC_muon_horizontal_metadata.csv",
    "MC_muon_up_metadata.csv",
    ]
    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)
    
    MC_muon_down = metadata_vars["MC_muon_down"]
    MC_muon_horizontal = metadata_vars["MC_muon_horizontal"]
    MC_muon_up = metadata_vars["MC_muon_up"]
    
    MC_muon_df = pd.concat([MC_muon_down, MC_muon_horizontal, MC_muon_up], ignore_index=True)
    
    updated_rows = []

    print("Reading vetoFree and vetoTagged event counts...")
    for _, row in tqdm(MC_muon_df.iterrows(), total=len(MC_muon_df), unit="file"):
        row = row.copy()

        # Veto-free
        if "vetoFree_hit_path" in row and isinstance(row["vetoFree_hit_path"], str):
            path = row["vetoFree_hit_path"]
            if os.path.exists(path):
                try:
                    f = ROOT.TFile.Open(path)
                    if f and not f.IsZombie():
                        tree = f.Get("sndData")
                        if tree:
                            row["vetoFree_count"] = tree.GetEntries()
                    f.Close()
                except Exception as e:
                    print(f"Error reading vetoFree file: {path}, error: {e}")

        # Veto-tagged
        if "vetoTagged_hit_path" in row and isinstance(row["vetoTagged_hit_path"], str):
            path = row["vetoTagged_hit_path"]
            if os.path.exists(path):
                try:
                    f = ROOT.TFile.Open(path)
                    if f and not f.IsZombie():
                        tree = f.Get("sndData")
                        if tree:
                            row["vetoTagged_count"] = tree.GetEntries()
                    f.Close()
                except Exception as e:
                    print(f"Error reading vetoTagged file: {path}, error: {e}")

        updated_rows.append(row)

    df = pd.DataFrame(updated_rows)
    df["vetoFree_count"] = df["vetoFree_count"].fillna(0).astype(int)
    df["vetoTagged_count"] = df["vetoTagged_count"].fillna(0).astype(int)

    # ---- VetoFree split ----
    total_vf = df["vetoFree_count"].sum()
    train_threshold_vf = 0.8 * total_vf
    accum_vf = 0
    vf_split = []

    for count in df["vetoFree_count"]:
        if accum_vf < train_threshold_vf:
            vf_split.append("train")
        else:
            vf_split.append("val")
        accum_vf += count

    df["vetoFree_split"] = vf_split

    # ---- VetoTagged split ----
    total_vt = df["vetoTagged_count"].sum()
    train_threshold_vt = 0.8 * total_vt
    accum_vt = 0
    vt_split = []

    for count in df["vetoTagged_count"]:
        if accum_vt < train_threshold_vt:
            vt_split.append("train")
        else:
            vt_split.append("val")
        accum_vt += count

    df["vetoTagged_split"] = vt_split

    # ---- Summary ----
    vf_train = df[df["vetoFree_split"] == "train"]["vetoFree_count"].sum()
    vf_val  = df[df["vetoFree_split"] == "val"]["vetoFree_count"].sum()
    vt_train = df[df["vetoTagged_split"] == "train"]["vetoTagged_count"].sum()
    vt_val  = df[df["vetoTagged_split"] == "val"]["vetoTagged_count"].sum()

    print("\nVetoFree Split:")
    print(f"  Train: {vf_train:.3e}")
    print(f"  Val : {vf_val:.3e}")

    print("\nVetoTagged Split:")
    print(f"  Train: {vt_train:.3e}")
    print(f"  Val : {vt_val:.3e}")

    return df


def cal_avg_veto_ineff():
    pass
    

def generate_train_dataset():
    # train:val:test=4:1:5
    train_2024_df = get_2024_train_set()
    

    
    
    # Output directory and model name
    out_dir = "./training"
    split_name = 'split_v1'
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save individual splits
    train_2024_path = f"{out_dir}/train_2024_{split_name}.csv"

    
    
    train_2024_df.to_csv(train_2024_path, index=False)


    print(f"Saved:")
    print(f" - Neutrino samples to {train_2024_path}")

   
def read_train_splts(splits='train'):
    out_dir = "./training"
    split_name = 'train_2024_split_v1'

    train_df_path = f"{out_dir}/{split_name}.csv"
  

    # Read CSVs
    train_df = pd.read_csv(train_df_path)
    
    # Print event counts by subfolder for 2024 data  
    print("\n Training data set MC_data_testbeam2024 event counts by subfolder:")
    subfolder_counts_2024 = train_df.groupby('subfolder')['n_event'].sum()
    for subfolder, count in subfolder_counts_2024.items():
        print(f"{subfolder}: {count:,} events")


     
if __name__ == "__main__":
    check_MC_data()
    # generate_train_dataset()
    read_train_splts()
    
    




# weight for each file
