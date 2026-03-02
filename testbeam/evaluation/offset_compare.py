#!/usr/bin/env python3
import glob
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



INPUT_DIR = Path("./old_plots/TestbeamPlot/2024/")
PATTERN = "QDC_MPV_by_channel_E*GeV_*_st2_mat0_ori0.csv"
OUTPUT_PDF = INPUT_DIR / "mpv_data_vs_channel_by_energy_st2_mat0_ori0.pdf"
OUTPUT_PDF_MPV_VIOLIN = INPUT_DIR / "mpv_data_distribution_by_energy_st2_mat0_ori0.pdf"
OUTPUT_PDF_MPV_SCATTER_HIST = INPUT_DIR / "mpv_data_scatter_hist_by_energy_st2_mat0_ori0.pdf"


OUT_DIR = Path("./QDC_offset")
OUTPUT_CSV = OUT_DIR / "QDC_offset_st2_mat0_ori0.csv"

ENERGY_RE = re.compile(r"E(\d+)GeV")



def plot_mpv_scatter_hist_selected_energies(bins=60, use_density=True):
    selected_energies = [100, 200, 300]

    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    frames = []

    for f in files:
        e = extract_energy(f)

        # Skip unwanted energies
        if e not in selected_energies:
            continue

        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["energy"] = e
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        df = df.groupby(["energy", "channel"], as_index=False)["mpv_data"].mean()
        frames.append(df)

    if not frames:
        raise RuntimeError("No data found for selected energies.")

    all_data = pd.concat(frames, ignore_index=True)

    # Global binning across selected energies
    global_vals = all_data["mpv_data"].values
    _, bin_edges = np.histogram(global_vals, bins=bins, density=use_density)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.figure(figsize=(10, 6))

    markers = {
        100: "o",
        200: "s",
        300: "^"
    }

    for e in selected_energies:
        vals = all_data.loc[all_data["energy"] == e, "mpv_data"].values
        if len(vals) == 0:
            print(f"[warning] No data for {e} GeV")
            continue

        counts, _ = np.histogram(vals, bins=bin_edges, density=use_density)

        plt.scatter(
            bin_centers,
            counts,
            marker=markers[e],
            label=f"{e} GeV",
            s=50
        )

    plt.xlabel("MPV (Data)")
    plt.ylabel("Density" if use_density else "Counts")
    plt.title("MPV(Data) distribution — 100, 200, 300 GeV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF_MPV_SCATTER_HIST)
    plt.close()

    print(f"[saved] {OUTPUT_PDF_MPV_SCATTER_HIST}")

def extract_energy(filepath: str) -> int:
    m = ENERGY_RE.search(os.path.basename(filepath))
    if not m:
        raise ValueError(f"Could not extract energy from filename: {filepath}")
    return int(m.group(1))


def plot_offset():
    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    # energy -> dataframe(channel, mpv_data)
    by_energy = {}

    for f in files:
        e = extract_energy(f)
        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        # If duplicates exist per channel, average them (safe default)
        df = df.groupby("channel", as_index=False)["mpv_data"].mean()

        by_energy[e] = df

    energies = sorted(by_energy.keys())

    # Union of all channels across energies so lines share a common x-grid
    all_channels = sorted(set().union(*[set(df["channel"].tolist()) for df in by_energy.values()]))

    # Build aligned series per energy (NaN where missing)
    aligned = {}
    for e in energies:
        s = by_energy[e].set_index("channel")["mpv_data"]
        aligned[e] = pd.Series(index=all_channels, data=[s.get(ch, float("nan")) for ch in all_channels])

    # Plot: x=channel, y=mpv_data, one line per energy
    plt.figure(figsize=(120, 6))

    for e in energies:
        y = aligned[e].values
        plt.plot(all_channels, y, marker="o", linewidth=1, label=f"{e} GeV")

    plt.xlabel("Channel")
    plt.ylabel("MPV (Data)")
    plt.title("MPV(Data) vs Channel — overlay by beam energy")
    plt.grid(True, alpha=0.3)

    # If many channels, this helps readability
    if len(all_channels) > 40:
        # show fewer x tick labels
        step = max(1, len(all_channels) // 20)
        plt.xticks(all_channels[::step], rotation=45, ha="right")
    else:
        plt.xticks(all_channels, rotation=45, ha="right")

    plt.legend(title="Energy", ncol=min(4, len(energies)))
    plt.tight_layout()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF)
    plt.close()

    print(f"[saved] {OUTPUT_PDF}")

def cal_avg_offset():
    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    frames = []

    for f in files:
        e = extract_energy(f)
        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["energy"] = e
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        # If duplicates exist per channel within this file, average them (same as your plot script)
        df = df.groupby(["energy", "channel"], as_index=False)["mpv_data"].mean()

        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    # Average across energies per channel
    offsets = (
        all_data.groupby("channel")["mpv_data"]
        .agg(mpv_data_avg="mean", mpv_data_std="std", n_energies="count")
        .reset_index()
        .sort_values("channel")
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    offsets.to_csv(OUTPUT_CSV, index=False)

    print(f"[saved] {OUTPUT_CSV}")
    print(f"channels: {offsets['channel'].nunique()}, energies used (total rows): {len(all_data)}")

def plot_mpv_data_distribution_by_energy(bins=60, use_density=True, logy=False):
    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    frames = []
    for f in files:
        e = extract_energy(f)
        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["energy"] = e
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        # average duplicates per (energy, channel), same convention as your other code
        df = df.groupby(["energy", "channel"], as_index=False)["mpv_data"].mean()

        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    energies = sorted(all_data["energy"].unique())

    plt.figure(figsize=(12, 6))
    for e in energies:
        vals = all_data.loc[all_data["energy"] == e, "mpv_data"].dropna().values
        if len(vals) == 0:
            continue
        plt.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=use_density,
            label=f"{e} GeV (n={len(vals)})",
        )

    plt.xlabel("MPV (Data)")
    plt.ylabel("Density" if use_density else "Counts")
    plt.title("Distribution of MPV(Data) across channels — grouped by beam energy")
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    plt.legend(ncol=min(3, len(energies)))
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF_MPV_DIST)
    plt.close()

    print(f"[saved] {OUTPUT_PDF_MPV_DIST}")


def plot_mpv_scatter_hist_by_energy(bins=60, use_density=True):
    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    frames = []

    for f in files:
        e = extract_energy(f)
        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["energy"] = e
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        df = df.groupby(["energy", "channel"], as_index=False)["mpv_data"].mean()
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    energies = sorted(all_data["energy"].unique())

    # Global binning (important!)
    global_vals = all_data["mpv_data"].values
    counts_global, bin_edges = np.histogram(global_vals, bins=bins, density=use_density)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, e in enumerate(energies):
        vals = all_data.loc[all_data["energy"] == e, "mpv_data"].values
        if len(vals) == 0:
            continue

        counts, _ = np.histogram(vals, bins=bin_edges, density=use_density)

        plt.scatter(
            bin_centers,
            counts,
            marker=markers[i % len(markers)],
            label=f"{e} GeV",
            s=40
        )

    plt.xlabel("MPV (Data)")
    plt.ylabel("Density" if use_density else "Counts")
    plt.title("MPV(Data) distribution across channels — scatter histogram")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF_MPV_SCATTER_HIST)
    plt.close()

    print(f"[saved] {OUTPUT_PDF_MPV_SCATTER_HIST}")

def plot_mpv_violin_by_energy():
    files = sorted(glob.glob(str(INPUT_DIR / PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found for: {INPUT_DIR / PATTERN}")

    frames = []

    for f in files:
        e = extract_energy(f)
        df = pd.read_csv(f)

        required = {"channel", "mpv_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {sorted(missing)}")

        df = df.copy()
        df["energy"] = e
        df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
        df["mpv_data"] = pd.to_numeric(df["mpv_data"], errors="coerce")
        df = df.dropna(subset=["channel", "mpv_data"])
        df["channel"] = df["channel"].astype(int)

        # Average duplicates per (energy, channel)
        df = df.groupby(["energy", "channel"], as_index=False)["mpv_data"].mean()

        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    energies = sorted(all_data["energy"].unique())

    # Collect data per energy
    data = [
        all_data.loc[all_data["energy"] == e, "mpv_data"].values
        for e in energies
    ]

    plt.figure(figsize=(10, 6))

    parts = plt.violinplot(
        data,
        showmeans=True,
        showmedians=False,
        showextrema=False
    )

    # Optional: slightly improve appearance
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    plt.xticks(
        range(1, len(energies) + 1),
        [f"{e} GeV" for e in energies],
        rotation=45
    )

    plt.xlabel("Beam Energy")
    plt.ylabel("MPV (Data)")
    plt.title("MPV(Data) distribution across channels")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF_MPV_VIOLIN)
    plt.close()

    print(f"[saved] {OUTPUT_PDF_MPV_VIOLIN}")
    
if __name__ == "__main__":
    # cal_avg_offset()
    
    plot_mpv_scatter_hist_selected_energies(bins=40, use_density=False)
    # plot_mpv_violin_by_energy()
    
    