import os
import pandas as pd
from tqdm import tqdm

# ------------------------
# IO: read all metadata CSVs
# ------------------------
def read_metadata():
    metadata_csv_list = [
        "MC_kaon_FTFP_BERT_metadata_subset.csv",
        "MC_neutron_FTFP_BERT_metadata_subset.csv",
        "real_data_2024_skim_runs_metadata.csv",
        "MC_neutrino_2024_vm_metadata.csv",
        "MC_neutrino_2024_ve_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
    ]

    metadata_rootpath = "/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/"
    metadata_dict = {}

    for csv_file in metadata_csv_list:
        full_path = os.path.join(metadata_rootpath, csv_file)
        if not os.path.exists(full_path):
            print(f"⚠️  Warning: file not found: {full_path}")
            continue
        try:
            df = pd.read_csv(full_path)
            key = os.path.splitext(csv_file)[0]
            metadata_dict[key] = df
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")

    return metadata_dict


def process_preCut_csv(df, max_lumi: float = 0.1,
                       path_col: str = "preCutEff_path",
                       lumi_col: str = "lumi_per_file"):
    total_table = None
    total_lumi = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing preCut CSVs"):
        csv_path = row.get(path_col)
        lumi = row.get(lumi_col, 0.0)

        if not isinstance(csv_path, str) or not os.path.exists(csv_path):
            tqdm.write(f"⚠️ Skipping invalid or missing path at row {idx}: {csv_path}")
            continue

        try:
            table = pd.read_csv(csv_path)
        except Exception as e:
            tqdm.write(f"❌ Error reading {csv_path}: {e}")
            continue

        # Separate numeric and non-numeric columns
        numeric_cols = table.select_dtypes(include="number").columns
        non_numeric_cols = [c for c in table.columns if c not in numeric_cols]

        if total_table is None:
            # Start with full table (preserve all columns)
            total_table = table.copy()
        else:
            # Only add numeric columns, leave non-numeric as in first table
            total_table[numeric_cols] = total_table[numeric_cols].add(
                table[numeric_cols], fill_value=0
            )

        total_lumi += float(lumi) if pd.notna(lumi) else 0.0
        if total_lumi >= max_lumi:
            tqdm.write(f"⏸️ Reached max_lumi={max_lumi:.3g}; stopping early.")
            break

    if total_table is None:
        tqdm.write("⚠️ No valid tables found.")
        total_table = pd.DataFrame()

    return total_table, total_lumi


def process_normalised_preCut_csv(df, max_lumi: float = 1000):
    total_table, total_lumi = process_preCut_csv(df, max_lumi=max_lumi)
    print(f"total_lumi : {total_lumi}")
    print(total_table)
    if total_table.empty:
        return total_table
    if total_lumi > 0:
        # Only normalize numeric columns
        numeric_cols = total_table.select_dtypes(include="number").columns
        total_table[numeric_cols] = total_table[numeric_cols] / total_lumi
    else:
        tqdm.write("⚠️ Total luminosity is zero; skipping normalization.")
        
    
    return total_table

# ------------------------
# Utility: scale normalized tables to a target luminosity
# ------------------------
def scale_to_lumi(norm_table: pd.DataFrame, target_lumi: float) -> pd.DataFrame:
    if norm_table is None or norm_table.empty:
        return pd.DataFrame()

    out = norm_table.copy()

    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s * target_lumi
        else:
            # Try to coerce numeric-looking strings; leave pure strings alone
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().any():
                out[col] = coerced * target_lumi
            # else: keep non-numeric text columns unchanged

    return out


def _coerce_count(df: pd.DataFrame) -> pd.DataFrame:
    if "count" in df.columns and not pd.api.types.is_numeric_dtype(df["count"]):
        df = df.copy()
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
    return df


def _expand_by_subtype(df: pd.DataFrame, base_name: str, subtype_col: str,
                       keep_values: list[str], label_fmt: str):
    out = []
    #print(df)
    if subtype_col not in df.columns:
        tmp = df[["cut", "count"]].copy()
        tmp["particle"] = base_name
        out.append(tmp)
        return out

    for subtype in df[subtype_col].unique():
        if subtype not in keep_values:
            continue
        sub_df = df[df[subtype_col] == subtype]
        tmp = sub_df[["cut", "count"]].copy()
        tmp["particle"] = label_fmt.format(subtype=subtype)
        out.append(tmp)
    return out


def make_summary_table(particle_tables: dict, subtype_col: str = "Unnamed: 0") -> pd.DataFrame:
    """
    Combine all particle preCut tables into a single table:
      - Splits vm_2024 (vm + NC), ve_2024 (ve + NC), volTarget (vm, ve, NC, vt)
      - Merges NC_vm_2024 + NC_ve_2024 into NC_2024
      - Rows = cut, Columns = particle types, Values = count
    """

    combined = []

    for name, df in particle_tables.items():
        if df is None or len(df) == 0:
            continue
        if not {"cut", "count"}.issubset(df.columns):
            print(f"⚠️ Skipping {name}: missing 'cut' or 'count'")
            continue

        df = _coerce_count(df)
        lname = name.lower()

        if lname == "vm_2024":
            combined += _expand_by_subtype(
                df, name, subtype_col,
                keep_values=["vm", "NC"],
                label_fmt="{subtype}_2024"
            )
        elif lname == "ve_2024":
            combined += _expand_by_subtype(
                df, name, subtype_col,
                keep_values=["ve", "NC"],
                label_fmt="{subtype}_2024"
            )
        elif "neutrino2022" in lname:
            combined += _expand_by_subtype(
                df, name, subtype_col,
                keep_values=["vm", "ve", "NC", "vt"],
                label_fmt="{subtype}_2022"
            )
        else:
            tmp = df[["cut", "count"]].copy()
            tmp["particle"] = name
            combined.append(tmp)

    if not combined:
        print("❌ No valid tables to combine.")
        return pd.DataFrame()

    combined_df = pd.concat(combined, ignore_index=True)

    # Pivot wide
    pivot = combined_df.pivot_table(index="cut", columns="particle", values="count", aggfunc="sum", fill_value=0)

    # ---- Merge NC_vm_2024 + NC_ve_2024 → NC_2024 ----
    nc_cols = [c for c in pivot.columns if c in ("NC_vm_2024", "NC_ve_2024")]
    if nc_cols:
        pivot["NC_2024"] = pivot[nc_cols].sum(axis=1)
        pivot.drop(columns=nc_cols, inplace=True)

    # Nice order of cuts
    order = [
        "a_raw", "b_non_veto", "1_scifi>200",
        "veto_1ns", "veto1_1ns", "veto2_1ns", "veto3_1ns",
        "veto_2ns", "veto1_2ns", "veto2_2ns", "veto3_2ns",
        "veto_3ns", "veto1_3ns", "veto2_3ns", "veto3_3ns",
        "veto_4ns", "veto1_4ns", "veto2_4ns", "veto3_4ns",
        "veto_5ns", "veto1_5ns", "veto2_5ns", "veto3_5ns",
    ]
    pivot = pivot.reindex([c for c in order if c in pivot.index], axis=0)

    # Column order grouping
    col_order = []
    col_order += [c for c in ["data", "kaon", "neutron"] if c in pivot.columns]
    col_order += [c for c in ["vm_vm_2024", "ve_ve_2024", "NC_2024"] if c in pivot.columns]
    vt_cols = ["vm_2022", "ve_2022", "NC_2022", "vt_2022"]
    col_order += [c for c in vt_cols if c in pivot.columns]
    col_order += [c for c in pivot.columns if c not in col_order]

    pivot = pivot[col_order]

    return pivot

# ------------------------
# Main
# ------------------------
def main():
    metadata = read_metadata()
    print(metadata.keys())

    # Show a peek of real data metadata if present
    real_key = "real_data_2024_skim_runs_metadata"
    if real_key not in metadata:
        print(f"❌ Missing key '{real_key}' in metadata. Available: {list(metadata.keys())}")
        return

    print(metadata[real_key].head())

    # 1) Real data: accumulate and get its total lumi (un-normalized)
    real_table, real_lumi = process_preCut_csv(metadata[real_key])
    print(f"Real data total lumi = {real_lumi:.6g}")

    # 2) MC/other sources: normalized per unit lumi
    neutron_key = "MC_neutron_FTFP_BERT_metadata_subset"
    kaon_key   = "MC_kaon_FTFP_BERT_metadata_subset"
    vm_key     = "MC_neutrino_2024_vm_metadata"
    ve_key     = "MC_neutrino_2024_ve_metadata"
    neutrino_key     = "MC_neutrino_volTarget_100fb-1_metadata"

    neutron_norm = process_normalised_preCut_csv(metadata.get(neutron_key, pd.DataFrame()))
    kaon_norm    = process_normalised_preCut_csv(metadata.get(kaon_key, pd.DataFrame()))
    vm_norm      = process_normalised_preCut_csv(metadata.get(vm_key, pd.DataFrame()))
    ve_norm      = process_normalised_preCut_csv(metadata.get(ve_key, pd.DataFrame()))
    neutrino_norm      = process_normalised_preCut_csv(metadata.get(neutrino_key, pd.DataFrame()))

    # 3) Scale MC tables to the *real data* integrated luminosity
    neutron_scaled = scale_to_lumi(neutron_norm, real_lumi)
    kaon_scaled    = scale_to_lumi(kaon_norm, real_lumi)
    vm_scaled      = scale_to_lumi(vm_norm, real_lumi)
    ve_scaled      = scale_to_lumi(ve_norm, real_lumi)
    neutrino_scaled      = scale_to_lumi(neutrino_norm, real_lumi)

    # # Example: print basic summaries
    # def brief(df, name):
    #     if df is None or df.empty:
    #         print(f"{name}: (empty)")
    #     else:
    #         print(f"{name}: shape={df.shape}")
    #         print(df)

    # brief(real_table,     "real_table (accumulated)")
    # brief(neutron_scaled, "neutron_scaled (MC→real lumi)")
    # brief(kaon_scaled,    "kaon_scaled (MC→real lumi)")
    # brief(vm_scaled,      "vm_scaled (MC→real lumi)")
    # brief(ve_scaled,      "ve_scaled (MC→real lumi)")
    # brief(neutrino_scaled,      "neutrino_scaled (MC→real lumi)")
    
    particle_tables = {
        "data": real_table,
        "kaon": kaon_scaled,
        "neutron": neutron_scaled,
        "vm_2024": vm_scaled,
        "ve_2024": ve_scaled,
        "neutrino2022": neutrino_scaled,
    }

    summary = make_summary_table(particle_tables)
    print(summary)


if __name__ == "__main__":
    main()
