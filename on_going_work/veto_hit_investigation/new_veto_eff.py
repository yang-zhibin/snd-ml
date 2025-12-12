import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def process_preCut_csv(df, max_lumi: float = 1e8,
                       path_col: str = "preCutEff_path",
                       lumi_col: str = "lumi_per_file"):
    total_table = None
    total_lumi = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing preCut CSVs"):
        csv_path = row.get(path_col)
        lumi = row.get(lumi_col)

        if not isinstance(csv_path, str) or not os.path.exists(csv_path):
            tqdm.write(f"⚠️ Skipping invalid or missing path at row {idx}: {csv_path}")
            continue

        try:
            table = pd.read_csv(csv_path)
        except Exception as e:
            tqdm.write(f"❌ Error reading {csv_path}: {e}")
            continue
        
        # check if table has "veto2_and_veto3_1ns" in column "cut"
        if "cut" not in table.columns or "veto2_and_veto3_1ns" not in table["cut"].values:
            tqdm.write(f"⚠️ Skipping {csv_path}: required cut not found.")
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


def process_neutralHadron_preCut_csv(df: pd.DataFrame,
                                     outdir: str,
                                     csv_name: str,
                                     max_lumi: float = 1e8) -> pd.DataFrame:
    """
    Group by 'energy_range', run process_preCut_csv on each group,
    save one pre-normalization CSV across all groups,
    then normalize each group's numeric columns by its own lumi,
    accumulate them into one DataFrame, and finally sum all numeric columns.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with the sum of all normalized numeric columns.
    """
    os.makedirs(outdir, exist_ok=True)

    pre_norm_rows = []
    normalized_total = None  # will accumulate normalized numeric columns
    original_total = None

    if "energy_range" not in df.columns:
        raise KeyError("DataFrame must contain an 'energy_range' column.")

    for energy_range, gdf in df.groupby("energy_range", dropna=False):
        total_table, total_lumi = process_preCut_csv(gdf, max_lumi=max_lumi)

        if total_table is None or total_table.empty:
            continue

        # 1) Add to pre-normalization table (with lumi info)
        tt = total_table.copy()
        tt.insert(0, "energy_range", energy_range)
        tt["group_total_lumi"] = float(total_lumi)
        pre_norm_rows.append(tt)

        # 2) Normalize numeric columns by lumi and accumulate
        if total_lumi and total_lumi > 0:
            nt = total_table.copy()
            num_cols = nt.select_dtypes(include="number").columns
            nt[num_cols] = nt[num_cols] / total_lumi

            # Append only numeric columns to normalized_tables
            if normalized_total is None:
                normalized_total = nt.copy()
                original_total = total_table.copy()
            else:
                normalized_total[num_cols] = normalized_total[num_cols].add(nt[num_cols], fill_value=0 )
                original_total[num_cols] = original_total[num_cols].add(total_table[num_cols], fill_value=0 )
        else:
            print(f"⚠️ Total luminosity is zero or None for group {energy_range!r}; skipping normalization.")

    # Save pre-normalization CSV
    if pre_norm_rows:
        pre_norm_df = pd.concat(pre_norm_rows, ignore_index=True, sort=False)
        pre_norm_path = os.path.join(outdir, f"{csv_name}_prenorm_by_group.csv")
        pre_norm_df.to_csv(pre_norm_path, index=False)

    # Sum normalized numeric columns
    if normalized_total.empty:
        return pd.DataFrame(), pd.DataFrame()

    #print(normalized_total)
    #sum_path = os.path.join(outdir, f"{csv_name}_normalized_sum.csv")
    #summed_norm.to_csv(sum_path, index=False)

    return normalized_total, original_total



def process_normalised_preCut_csv(df, max_lumi: float = 1e8):
    total_table, total_lumi = process_preCut_csv(df, max_lumi=max_lumi)
    original_total = total_table.copy()
    #print(f"total_lumi : {total_lumi}")
    #print(total_table)
    if total_table.empty:
        return total_table
    if total_lumi > 0:
        # Only normalize numeric columns
        numeric_cols = total_table.select_dtypes(include="number").columns
        total_table[numeric_cols] = total_table[numeric_cols] / total_lumi
    else:
        tqdm.write("⚠️ Total luminosity is zero; skipping normalization.")
        
    
    return total_table, original_total

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
        "veto_1ns", "veto1_1ns", "veto2_1ns", "veto3_1ns", "veto2_and_veto3_1ns",
        "veto_2ns", "veto1_2ns", "veto2_2ns", "veto3_2ns", "veto2_and_veto3_2ns",
        "veto_3ns", "veto1_3ns", "veto2_3ns", "veto3_3ns", "veto2_and_veto3_3ns",
        "veto_4ns", "veto1_4ns", "veto2_4ns", "veto3_4ns", "veto2_and_veto3_4ns",
        "veto_5ns", "veto1_5ns", "veto2_5ns", "veto3_5ns", "veto2_and_veto3_5ns",
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




def compute_and_save_tables(
    df: pd.DataFrame,
    outdir: str = "./preCutEff_table",
    round_decimals: int = 4,
    save_csv: bool = True,
    dpi: int = 200,
    int_lumi: float = None,
):
    os.makedirs(outdir, exist_ok=True)
    if "a_raw" not in df.index:
        raise ValueError("The DataFrame must contain an 'a_raw' row for normalization.")

    df_eff = df.div(df.loc["a_raw"])

    cols_2022 = ["data", "kaon", "neutron", "vm_2022", "ve_2022", "NC_2022", "vt_2022"]
    cols_2024 = ["data", "kaon", "neutron", "vm_2024", "ve_2024", "NC_2024", "vt_2024"]

    cuts_template = [
        ("1ns", ["a_raw", "b_non_veto", "1_scifi>200", "veto1_1ns", "veto2_1ns", "veto3_1ns", "veto_1ns", "veto2_and_veto3_1ns"]),
        ("2ns", ["a_raw", "b_non_veto", "1_scifi>200", "veto1_2ns", "veto2_2ns", "veto3_2ns", "veto_2ns", "veto2_and_veto3_2ns"]),
        ("3ns", ["a_raw", "b_non_veto", "1_scifi>200", "veto1_3ns", "veto2_3ns", "veto3_3ns", "veto_3ns", "veto2_and_veto3_3ns"]),
        ("4ns", ["a_raw", "b_non_veto", "1_scifi>200", "veto1_4ns", "veto2_4ns", "veto3_4ns", "veto_4ns", "veto2_and_veto3_4ns"]),
        ("5ns", ["a_raw", "b_non_veto", "1_scifi>200", "veto1_5ns", "veto2_5ns", "veto3_5ns", "veto_5ns", "veto2_and_veto3_5ns"]),
    ]

    def save_table_as_image(df_table: pd.DataFrame, title: str, path_png: str):
        disp = df_table.round(round_decimals).astype(float)
        n_rows, n_cols = disp.shape

        fig_w = max(5, 0.7 * n_cols + 1.6)
        fig_h = max(2, 0.35 * n_rows + 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # Table anchored at top, very close to the title
        tbl = ax.table(
            cellText=disp.values,
            rowLabels=disp.index.tolist(),
            colLabels=disp.columns.tolist(),
            loc="upper center",
            cellLoc="center",
        )
        tbl.scale(1.0, 1.05)

        # Make header bold
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")

        # Title and luminosity closer to the table (just above top row)
        # Coordinates: slightly below top margin (0.96 instead of 0.98)
        fig.text(0.02, 0.96, title, ha="left", va="top", fontsize=12)
        if int_lumi is not None:
            fig.text(
                0.98,
                0.955,  # slightly lower to match title height
                f"$\\int\\!\\mathcal{{L}}\\,dt = {int_lumi:.2f}\\ \\mathrm{{fb}}^{{-1}}$",
                ha="right",
                va="top",
                fontsize=11.5,
            )

        # Tight save, remove padding entirely
        fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    outputs = {}
    for ns, cuts in cuts_template:
        for year, cols in (("2022", cols_2022), ("2024", cols_2024)):
            name = f"eff_Veto{ns}_{year}"
            valid_cuts = [c for c in cuts if c in df_eff.index]
            valid_cols = [c for c in cols if c in df_eff.columns]
            if not valid_cuts or not valid_cols:
                continue
            sub = df_eff.loc[valid_cuts, valid_cols]
            outputs[name] = sub
            if save_csv:
                sub.round(round_decimals).to_csv(os.path.join(outdir, f"{name}.csv"))
            save_table_as_image(sub, name.replace("_", " ").upper(), os.path.join(outdir, f"{name}.png"))

    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write("Pre-cut efficiencies normalized to a_raw. Tables saved as CSV and PNG.\n")
        for key in sorted(outputs.keys()):
            f.write(f" - {key}\n")

    
    orig_csv = os.path.join(outdir, "original_df.csv")
    df.to_csv(orig_csv, float_format=f"%.{round_decimals}f")
    return outputs


def lighten_color(color, amount=0.2):
    """
    Lighten a given color by blending it with white.
    amount=0 → no change, amount=1 → white
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        return color  # if color not valid
    return tuple(min(1, x + (1 - x) * amount) for x in c)


def compute_and_save_eff_and_yield_tables(
    df: pd.DataFrame,
    outdir: str = "./preCutEff_table",
    call_original_df: pd.DataFrame | None = None,  # << NEW
    sig_digits: int = 2,
    save_csv: bool = True,
    dpi: int = 200,
    int_lumi: float | None = None,
):
    """
    Build and save one table per veto window (1–5 ns), for ALL columns combined:
      - Efficiency tables (normalized to 'a_raw')
      - Yield tables (original numbers)
      - Optionally yield tables from a second DataFrame (call_original_df), 
        saved as 'original_Veto{ns}.csv/.png'
    """

    os.makedirs(outdir, exist_ok=True)
    if "a_raw" not in df.index:
        raise ValueError("The DataFrame must contain an 'a_raw' row for normalization.")
    if call_original_df is not None and "a_raw" not in call_original_df.index:
        raise ValueError("call_original_df must also contain an 'a_raw' row if provided.")

    # ---------- Helpers ----------

    def lighten_color(color, amount=0.35):
        try:
            r, g, b = mcolors.to_rgb(color)
        except ValueError:
            return color
        return (min(1.0, r + (1 - r) * amount),
                min(1.0, g + (1 - g) * amount),
                min(1.0, b + (1 - b) * amount))

    base_hues_2022 = {
        "vm": "#a5c9ff",
        "ve": "#a5ffd5",
        "NC": "#d0d0ff",
        "vt": "#e0a5ff",
    }
    base_colors = {
        k: (v, mcolors.to_hex(lighten_color(v, 0.35)))
        for k, v in base_hues_2022.items()
    }

    neutral_colors = {"data": "#eeeeee", "kaon": "#dddddd", "neutron": "#cccccc"}
    default_other = "#f5f5f5"

    def get_col_color(col: str) -> str:
        for part, (c22, c24) in base_colors.items():
            if col.endswith(f"{part}_2022"):
                return c22
            if col.endswith(f"{part}_2024"):
                return c24
        for k, c in neutral_colors.items():
            if col == k or col.startswith(k):
                return c
        return default_other

    def sort_columns(columns):
        fixed = [c for c in ["data", "kaon", "neutron"] if c in columns]
        cols_2022 = [c for c in columns if c.endswith("_2022")]
        cols_2024 = [c for c in columns if c.endswith("_2024")]
        others = [c for c in columns if c not in fixed + cols_2022 + cols_2024]
        return fixed + cols_2022 + cols_2024 + others

    def format_sig(x, sig=sig_digits):
        try:
            if pd.isna(x):
                return ""
            x = float(x)
        except Exception:
            return str(x)
        if x == 0:
            return "0"
        ax = abs(x)
        if ax >= 1e3 or ax < 1e-3:
            return f"{x:.{sig-1}e}"
        else:
            digits = sig - int(np.floor(np.log10(ax))) - 1
            digits = max(0, digits)
            return f"{x:.{digits}f}"

    def df_format_sig(df_in: pd.DataFrame, sig=sig_digits) -> pd.DataFrame:
        return df_in.applymap(lambda v: format_sig(v, sig))

    # ---------- Core computation ----------

    df_eff = df.div(df.loc["a_raw"])

    cuts_template = [
        ("1ns", ["a_raw", "b_non_veto", "1_scifi>200",
                 "veto1_1ns", "veto2_1ns", "veto3_1ns", "veto_1ns", "veto2_and_veto3_1ns"]),
        ("2ns", ["a_raw", "b_non_veto", "1_scifi>200",
                 "veto1_2ns", "veto2_2ns", "veto3_2ns", "veto_2ns", "veto2_and_veto3_2ns"]),
        ("3ns", ["a_raw", "b_non_veto", "1_scifi>200",
                 "veto1_3ns", "veto2_3ns", "veto3_3ns", "veto_3ns", "veto2_and_veto3_3ns"]),
        ("4ns", ["a_raw", "b_non_veto", "1_scifi>200",
                 "veto1_4ns", "veto2_4ns", "veto3_4ns", "veto_4ns", "veto2_and_veto3_4ns"]),
        ("5ns", ["a_raw", "b_non_veto", "1_scifi>200",
                 "veto1_5ns", "veto2_5ns", "veto3_5ns", "veto_5ns", "veto2_and_veto3_5ns"]),
    ]

    # ---------- Rendering ----------

    def save_table_as_image(df_table: pd.DataFrame, title: str, path_png: str):
        disp = df_format_sig(df_table, sig_digits)
        n_rows, n_cols = disp.shape
        fig_w = max(5, 0.7 * n_cols + 1.6)
        fig_h = max(2, 0.35 * n_rows + 1.2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=disp.values,
            rowLabels=disp.index.tolist(),
            colLabels=disp.columns.tolist(),
            loc="upper center",
            cellLoc="center",
        )
        tbl.scale(1.0, 1.05)

        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#cccccc")
            else:
                col_name = disp.columns[col]
                cell.set_facecolor(get_col_color(col_name))

        fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=12)
        if int_lumi is not None:
            fig.text(
                0.98, 0.96,
                f"$\\int\\!\\mathcal{{L}}\\,dt = {int_lumi:.2f}\\ \\mathrm{{fb}}^{{-1}}$",
                ha="right", va="top", fontsize=11.0,
            )

        # Legend
        legend_entries = [
            ("vm (2022/2024)", base_colors["vm"]),
            ("ve (2022/2024)", base_colors["ve"]),
            ("NC (2022/2024)", base_colors["NC"]),
            ("vt (2022/2024)", base_colors["vt"]),
        ]
        x0, y0 = 0.02, 0.06
        box_w, box_h = 0.016, 0.018
        for i, (label, (c22, c24)) in enumerate(legend_entries):
            y = y0 - i * 0.03
            fig.patches.extend([
                plt.Rectangle((x0, y), box_w, box_h, facecolor=c22, transform=fig.transFigure, clip_on=False),
                plt.Rectangle((x0 + box_w + 0.006, y), box_w, box_h, facecolor=c24, transform=fig.transFigure, clip_on=False),
            ])
            fig.text(x0 + 2 * box_w + 0.025, y + box_h / 2, label, va="center", fontsize=9)

        fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ---------- Build & save ----------

    outputs_eff, outputs_yields, outputs_original = {}, {}, {}

    for ns, cuts in cuts_template:
        valid_cuts = [c for c in cuts if c in df.index]
        if not valid_cuts:
            continue

        name_eff = f"eff_Veto{ns}"
        name_yld = f"yields_Veto{ns}"

        sub_eff = df_eff.loc[valid_cuts]
        sub_yld = df.loc[valid_cuts]

        sub_eff = sub_eff[sort_columns(sub_eff.columns)]
        sub_yld = sub_yld[sort_columns(sub_yld.columns)]

        outputs_eff[name_eff] = sub_eff
        outputs_yields[name_yld] = sub_yld

        if save_csv:
            df_format_sig(sub_eff).to_csv(os.path.join(outdir, f"{name_eff}.csv"))
            df_format_sig(sub_yld).to_csv(os.path.join(outdir, f"{name_yld}.csv"))

        save_table_as_image(sub_eff, name_eff.replace("_", " ").upper(),
                            os.path.join(outdir, f"{name_eff}.png"))
        save_table_as_image(sub_yld, name_yld.replace("_", " ").upper(),
                            os.path.join(outdir, f"{name_yld}.png"))

        # ---------- Optional: call_original_df yields ----------
        if call_original_df is not None:
            valid_cuts_orig = [c for c in cuts if c in call_original_df.index]
            if not valid_cuts_orig:
                continue
            name_orig = f"original_Veto{ns}"
            sub_orig = call_original_df.loc[valid_cuts_orig]
            sub_orig = sub_orig[sort_columns(sub_orig.columns)]
            outputs_original[name_orig] = sub_orig

            if save_csv:
                df_format_sig(sub_orig).to_csv(os.path.join(outdir, f"{name_orig}.csv"))
            save_table_as_image(sub_orig, name_orig.replace("_", " ").upper(),
                                os.path.join(outdir, f"{name_orig}.png"))

    # README
    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write(f"Efficiencies (normalized to a_raw) and yields (original numbers).\n")
        if call_original_df is not None:
            f.write("Also includes original yields from call_original_df as 'original_Veto*.csv/.png'.\n")
        f.write(f"All values formatted to {sig_digits} significant digits; scientific notation for |x|>1000 or |x|<0.001.\n")
        f.write("Column order: data, kaon, neutron, *_2022, *_2024, others.\n")
        f.write("Color: same hue per particle (2024 lighter).\n\n")

    df.to_csv(os.path.join(outdir, "original_df.csv"), float_format="%.6g")
    if call_original_df is not None:
        call_original_df.to_csv(os.path.join(outdir, "call_original_df.csv"), float_format="%.6g")

    return {"eff": outputs_eff, "yields": outputs_yields, "original": outputs_original}

# ------------------------
# Main
# ------------------------
def main():
    metadata = read_metadata()
    print(metadata.keys())
    outdir = "./preCutEff_table"
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

    neutron_norm, neutron_original = process_neutralHadron_preCut_csv(metadata.get(neutron_key, pd.DataFrame()), outdir, neutron_key)
    kaon_norm, kaon_original    = process_neutralHadron_preCut_csv(metadata.get(kaon_key, pd.DataFrame()), outdir, kaon_key)
    vm_norm, vm_original      = process_normalised_preCut_csv(metadata.get(vm_key, pd.DataFrame()))
    ve_norm, ve_original      = process_normalised_preCut_csv(metadata.get(ve_key, pd.DataFrame()))
    neutrino_norm, neutrino_original    = process_normalised_preCut_csv(metadata.get(neutrino_key, pd.DataFrame()))

    # 3) Scale MC tables to the *real data* integrated luminosity
    #real_lumi = 1500
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
    
    particle_original = {
        "data": real_table,
        "kaon": kaon_original,
        "neutron": neutron_original,
        "vm_2024": vm_original,
        "ve_2024": ve_original,
        "neutrino2022": neutrino_original,
    }

    summary = make_summary_table(particle_tables)
    summary_original = make_summary_table(particle_original)
    print(summary)
    print(summary_original)
    
    # tables = compute_and_save_eff_tables(summary, outdir="./preCutEff_table", round_decimals=8,
    #                                  save_csv=False, dpi=200, int_lumi=real_lumi)
    tables = compute_and_save_eff_and_yield_tables(summary, outdir="./preCutEff_table", call_original_df=summary_original, sig_digits=3,
                                     save_csv=False, dpi=200, int_lumi=real_lumi)



if __name__ == "__main__":
    main()
