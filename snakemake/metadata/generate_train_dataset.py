import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import ROOT
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
    # "MC_kaon_FTFP_BERT_metadata.csv",
    # "MC_neutron_FTFP_BERT_metadata.csv",
    
    # "MC_kaon_QGSP_BERT_HP_PEN_metadata.csv",
    # "MC_neutron_QGSP_BERT_HP_PEN_metadata.csv",
    
    # "MC_muon_down_metadata.csv",
    # "MC_muon_horizontal_metadata.csv",
    # "MC_muon_up_metadata.csv",
    
    # "MC_neutrino_volMuFilter_20fb-1_metadata.csv",
     "MC_neutrino_volTarget_100fb-1_metadata.csv",
    
    # "real_data_2022_metadata.csv",
    # "real_data_2023_metadata.csv",
    # "real_data_2024_metadata.csv",
    ]

    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'

    metadata_vars = load_metadata_files(mc_files, root_path)

    

    # MC_neutrino_volMuFilter_20fb_1 = metadata_vars["MC_neutrino_volMuFilter_20fb_1"]
    MC_neutrino_volTarget_100fb_1 = metadata_vars["MC_neutrino_volTarget_100fb_1"]

    # real_data_2022 = metadata_vars["real_data_2022"]
    # real_data_2023 = metadata_vars["real_data_2023"]
    # real_data_2024 = metadata_vars["real_data_2024"]
    
    #MC_muon = pd.concat([MC_muon_down, MC_muon_horizontal, MC_muon_up], ignore_index=True)
    
    #create_dataset_report
    # MC_kaon_FTFP_BERT = metadata_vars["MC_kaon_FTFP_BERT"]
    # MC_neutron_FTFP_BERT = metadata_vars["MC_neutron_FTFP_BERT"]
    
    # MC_neutron_QGSP_BERT_HP_PEN = metadata_vars["MC_neutron_QGSP_BERT_HP_PEN"]
    # MC_kaon_QGSP_BERT_HP_PEN = metadata_vars["MC_kaon_QGSP_BERT_HP_PEN"]
    # check_neutral_bkg_dataset(MC_kaon_FTFP_BERT, "MC_kaon_FTFP_BERT")
    # check_neutral_bkg_dataset(MC_neutron_FTFP_BERT, "MC_neutron_FTFP_BERT")
    # check_neutral_bkg_dataset(MC_neutron_QGSP_BERT_HP_PEN, "MC_neutron_QGSP_BERT_HP_PEN")
    # check_neutral_bkg_dataset(MC_kaon_QGSP_BERT_HP_PEN, "MC_kaon_QGSP_BERT_HP_PEN")
    
    
    
    # MC_muon_down = metadata_vars["MC_muon_down"]
    # MC_muon_horizontal = metadata_vars["MC_muon_horizontal"]
    # MC_muon_up = metadata_vars["MC_muon_up"]
    # check_muon_bkg_dataset(MC_muon_down, "MC_muon_down")
    # check_muon_bkg_dataset(MC_muon_horizontal, "MC_muon_horizontal")
    # check_muon_bkg_dataset(MC_muon_up, "MC_muon_up")
    
    #check_neutrino_dataset
    check_neutrion_bkg_dataset(MC_neutrino_volTarget_100fb_1, "MC_neutrino_volTarget_100fb_1")
    
    # check_real_data

def get_neutrino_train_set():
    mc_files = [
        # "MC_neutrino_volMuFilter_20fb-1_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
    ]
    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)

    df = metadata_vars["MC_neutrino_volTarget_100fb_1"]
    updated_rows = []

    print("Reading vetoFree and vetoTagged counts from ROOT files...")
    for _, row in tqdm(df.iterrows(), total=len(df), unit="file"):
        row = row.copy()

        if "preSelect_path" in row and isinstance(row["preSelect_path"], str):
            path = row["preSelect_path"]
            if os.path.exists(path):
                try:
                    f = ROOT.TFile.Open(path)
                    if f and not f.IsZombie():
                        tree = f.Get("sndData")
                        if tree:
                            row["vetoFree_count"] = tree.GetEntries("preSelect_vetoFree == 1")
                            row["vetoTagged_count"] = tree.GetEntries("preSelect_vetoTagged == 1")
                    f.Close()
                except Exception as e:
                    print(f"Error reading {path}: {e}")

        updated_rows.append(row)

    df = pd.DataFrame(updated_rows)
    df["vetoFree_count"] = df["vetoFree_count"].fillna(0).astype(int)
    df["vetoTagged_count"] = df["vetoTagged_count"].fillna(0).astype(int)

    # ---- Split by cumulative count (not row count) ----
    total_events = df["vetoFree_count"].sum()
    train_target = int(0.4 * total_events)
    val_target = int(0.1 * total_events)

    print(f"\nTotal vetoFree events: {total_events:.3e}")
    print(f"Target: train={train_target:.3e}, val={val_target:.3e}")

    df["split"] = ""
    accum = 0

    for idx, row in df.iterrows():
        if accum < train_target:
            df.at[idx, "split"] = "train"
        elif accum < train_target + val_target:
            df.at[idx, "split"] = "val"
        accum += row["vetoFree_count"]

    # Summary
    train_sum = df[df["split"] == "train"]["vetoFree_count"].sum()
    val_sum = df[df["split"] == "val"]["vetoFree_count"].sum()

    print(f"Train vetoFree count: {train_sum:.3e}")
    print(f"Val  vetoFree count: {val_sum:.3e}")
    
    

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
    # Neutrino total: 318547, train:val:test=4:1:5, train=127k (vetoFree)
    neutrino_df = get_neutrino_train_set()
    
    
    # MC kaon(vetoFree)
    # MC Neutron (vetoFree)
    neutral_bkg_df = get_neutral_bkg_train_set()
    

    # MC muon (vetoFree + vetoTagged but drop veto hits)
    muon_bkg_df = get_muon_train_set()
    
    
    # Output directory and model name
    out_dir = "./training"
    model_name = 'GravNet_v2'
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save individual splits
    neutrino_path = f"{out_dir}/{model_name}_neutrino_split.csv"
    neutral_bkg_path = f"{out_dir}/{model_name}_neutral_bkg_split.csv"
    muon_bkg_path = f"{out_dir}/{model_name}_muon_bkg_split.csv"
    
    
    neutrino_df.to_csv(neutrino_path, index=False)
    neutral_bkg_df.to_csv(neutral_bkg_path, index=False)
    muon_bkg_df.to_csv(muon_bkg_path, index=False)

    print(f"Saved:")
    print(f" - Neutrino samples to {neutrino_path}")
    print(f" - Neutral background samples to {neutral_bkg_path}")
    print(f" - Muon background samples to {muon_bkg_path}")
   
def read_train_splts(splits='train'):
    out_dir = "./training"
    model_name = 'GravNet_v2'

    neutrino_path = f"{out_dir}/{model_name}_neutrino_split.csv"
    neutral_bkg_path = f"{out_dir}/{model_name}_neutral_bkg_split.csv"
    muon_bkg_path = f"{out_dir}/{model_name}_muon_bkg_split.csv"
    

    # Read CSVs
    neutrino_df = pd.read_csv(neutrino_path)
    neutral_bkg_df = pd.read_csv(neutral_bkg_path)
    muon_bkg_df = pd.read_csv(muon_bkg_path)
    
    neutrino_df['weight'] = neutrino_df['lumi_per_file']/neutrino_df['vetoFree_count']
    neutral_bkg_df['weight'] = neutral_bkg_df['lumi_per_file']/neutral_bkg_df['vetoFree_count']
    
    neutrion_lumi = neutrino_df['lumi_per_file'].sum()
    muon_factor = neutrion_lumi/muon_bkg_df['lumi_per_file'].sum()
    muon_bkg_df['weight'] = muon_bkg_df['lumi_per_file']/muon_bkg_df['vetoTagged_count'] * muon_factor
    
    # Group neutral background by energy bin and data_type
    grouped = neutral_bkg_df.groupby(['energy_bin', 'data_type'])
    weights = {}
    # Normalize rates to get weights
    for (energy_bin, data_type), group_df in grouped:
        factor = neutrion_lumi / group_df['lumi_per_file'].sum()
        for idx, row in group_df.iterrows():
            weights[idx] = row['lumi_per_file'] / row['vetoFree_count'] * factor * 0.15
            #write weight to row

    # Assign weights
    neutral_bkg_df['weight'] = neutral_bkg_df.index.map(weights)
    
    
    #update neutrino_df, neutral_bkg_df, muon_bkg_df which has new columns to its csv file
    neutrino_df.to_csv(neutrino_path, index=False)
    neutral_bkg_df.to_csv(neutral_bkg_path, index=False)
    muon_bkg_df.to_csv(muon_bkg_path, index=False)
    
    neutrino_df['total_weights'] = neutrino_df['weight'] * neutrino_df['vetoFree_count']
    neutral_bkg_df['total_weights'] = neutral_bkg_df['weight'] * neutral_bkg_df['vetoFree_count']
    muon_bkg_df['total_weights'] = muon_bkg_df['weight'] * muon_bkg_df['vetoTagged_count']
    print(neutrino_df)
    print(neutral_bkg_df)
    print(muon_bkg_df)

        # Determine which splits to include
    if splits == 'train':
        split_list = ['train']
    elif splits == 'val':
        split_list = ['val']
    else:
        split_list = ['train', 'val']

    # Print counts, lumi, and weight sum for neutrino and neutral background
    def print_counts(df, label):
        for split in split_list:
            split_df = df[df['split'] == split]
            vf_count = split_df['vetoFree_count'].sum() if 'vetoFree_count' in split_df.columns else 0
            lumi_sum = split_df['lumi_per_file'].sum() if 'lumi_per_file' in split_df.columns else 0
            weight_sum = split_df['total_weights'].sum() if 'total_weights' in split_df.columns else 0
            print(f"{label} [{split}]: vetoFree_count = {vf_count}, lumi_per_file sum = {lumi_sum}, weight sum = {weight_sum}")

    print_counts(neutrino_df, 'Neutrino')
    # Separate neutral background into kaon and neutron components
    kaon_df = neutral_bkg_df[neutral_bkg_df['data_type'] == 'MC_kaon'].copy()
    neutron_df = neutral_bkg_df[neutral_bkg_df['data_type'] == 'MC_neutron'].copy()

    # Print separate stats
    print_counts(kaon_df, 'Neutral Background (Kaon)')
    print_counts(neutron_df, 'Neutral Background (Neutron)')


    # Print counts, lumi, and weight sum for muon background
    def print_muon_counts(df):
        for split in split_list:
            vf_mask = df['vetoFree_split'] == split
            vt_mask = df['vetoTagged_split'] == split

            vf_count = df.loc[vf_mask, 'vetoFree_count'].sum()
            vt_count = df.loc[vt_mask, 'vetoTagged_count'].sum()

            lumi_vf = df.loc[vf_mask, 'lumi_per_file'].sum()
            lumi_vt = df.loc[vt_mask, 'lumi_per_file'].sum()

            weight_sum = df.loc[vt_mask, 'total_weights'].sum() if 'total_weights' in df.columns else 0

            print(f"Muon Background [{split}]: vetoFree_count = {vf_count}, lumi = {lumi_vf}; "
                  f"vetoTagged_count = {vt_count}, lumi = {lumi_vt}, weight sum = {weight_sum}")

    print_muon_counts(muon_bkg_df)

    

     
if __name__ == "__main__":
    #check_MC_data()
    #generate_train_dataset()
    read_train_splts()
    
    




# weight for each file
