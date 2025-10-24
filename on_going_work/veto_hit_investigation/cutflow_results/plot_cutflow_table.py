#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_to_data_lumi(df):
    """Scale MC columns to match data lumi."""
    if "int_lumi" not in df.index or "data" not in df.columns:
        return df.copy()
    out = df.copy()
    data_lumi = out.loc["int_lumi", "data"]
    for col in out.columns:
        if col == "data":
            continue
        col_lumi = out.loc["int_lumi", col]
        if pd.notna(col_lumi) and col_lumi != 0:
            out.loc[out.index != "int_lumi", col] *= (data_lumi / col_lumi)
    return out, data_lumi

def pick_family(df, kind):
    """Pick rows for either 'veto1' or 'all' families."""
    if kind == "veto1":
        mask = df.index.str.startswith("vetoHitTime_earlist_veto1>")
    else:
        mask = df.index.str.startswith("vetoHitTime_earlist>") & ~df.index.str.startswith("vetoHitTime_earlist_veto1>")
    mask |= (df.index == "total")
    sub = df[mask].copy()
    sub["thr"] = np.nan
    for idx in sub.index:
        if idx != "total":
            m = re.search(r">([\d.]+)", idx)
            if m:
                sub.loc[idx, "thr"] = float(m.group(1))

    # Sort by threshold, keeping 'total' at the top
    sub = sub.sort_values("thr", na_position="first")
    return sub

def eff_to_total(df, sub):
    """Compute efficiencies vs total."""
    if "total" not in df.index:
        raise ValueError("Missing 'total' row in CSV.")
    total = df.loc["total"].replace(0, np.nan)
    eff = sub.drop(columns="thr").div(total)
    eff.index = [f">{thr:g} ns" for thr in sub["thr"].values]
    return eff.clip(upper=1.0).fillna(0.0)

def _format_values(df: pd.DataFrame, digits: int, mode: str) -> list[list[str]]:
    """
    Return a 2D list of strings for the table cells.
    mode = 'counts' uses thousands separators for small numbers,
            scientific notation for very large ones;
    mode = 'eff' uses fixed decimals.
    """
    out = []
    if mode == "counts":
        for _, row in df.iterrows():
            formatted = []
            for x in row.values:
                if pd.isna(x):
                    formatted.append("")
                elif abs(x) >= 1e4:
                    formatted.append(f"{x:.2e}")  # scientific notation
                else:
                    formatted.append(f"{x:,.0f}")  # normal integer with separators
            out.append(formatted)
    else:  # efficiencies
        fmt = f"{{:.{digits}f}}"
        for _, row in df.iterrows():
            out.append([fmt.format(x) if pd.notna(x) else "" for x in row.values])
    return out

def plot_table(df, data_lumi, title, outpng, mode, digits=3):
    """Render a DataFrame as a PNG table."""
    outpng = Path(outpng)
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(df) + 1))
    ax.axis("off")

    # Format values
    cell_text = _format_values(df, digits, mode)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    
    fig.text(
        0.8, 0.80,  # x, y in normalized figure coordinates
        f"$\\int\\!\\mathcal{{L}}\\,dt = {data_lumi:.2f}\\ \\mathrm{{fb}}^{{-1}}$",  # LaTeX-style string
        ha="right",  # horizontal alignment to match TextAlign(31)
        va="top",
        fontsize=12,  
    )
    
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.2)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()
    print(f"Saved table: {outpng}")

def rename_rows(sub, label_prefix):
    new_index = []
    for thr in sub["thr"]:
        if pd.isna(thr):
            new_index.append("Total")
        else:
            new_index.append(f"{label_prefix} > {thr:g} ns")
    return new_index

def main():
    df = pd.read_csv("cutflow_summary_2024.csv", index_col=0).apply(pd.to_numeric, errors="coerce")
    veto_setup = "ThreeVetoPlanes"
    
    if veto_setup == "ThreeVetoPlanes":
        #drop column kaon and neutron
        df = df.drop(columns=["kaon", "neutron"])

    # normalize to data lumi
    dfn, data_lumi = normalize_to_data_lumi(df)

    # select families
    sub_v1 = pick_family(dfn, "veto1")
    sub_all = pick_family(dfn, "all")

    # compute efficiencies vs total
    eff_v1 = eff_to_total(dfn, sub_v1)
    eff_all = eff_to_total(dfn, sub_all)

    # rename rows for readability
    sub_v1.index = rename_rows(sub_v1, "Station 1 Veto Hit Time")
    sub_all.index = rename_rows(sub_all, "All Veto Hit Time")
    eff_v1.index = rename_rows(sub_v1, "Station 1 Veto Hit Time")
    eff_all.index = rename_rows(sub_all, "All Veto Hit Time")

    # plot count tables
    plot_table(sub_v1.drop(columns="thr"),data_lumi, "Counts: Veto Station 1 Hit Time", f"./plots/{veto_setup}_counts_veto_station1.png",mode="counts", digits=0)
    plot_table(sub_all.drop(columns="thr"),data_lumi, "Counts: All Veto Hit Time", f"./plots/{veto_setup}_counts_all_veto.png",mode="counts", digits=0)

    # plot efficiency tables
    plot_table(eff_v1,data_lumi, "Efficiency vs Total: Veto Station 1", f"./plots/{veto_setup}_eff_veto_station1.png",mode="eff", digits=3)
    plot_table(eff_all,data_lumi, "Efficiency vs Total: All Veto", f"./plots/{veto_setup}_eff_all_veto.png",mode="eff", digits=3)

if __name__ == "__main__":
    main()
