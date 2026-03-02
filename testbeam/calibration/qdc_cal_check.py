import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pathlib import Path


ID_COLUMNS = ["board_id", "fe_id", "channel"]


def load_csv(path):
    return pd.read_csv(path)


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compare_global(df1, df2, label1, label2, output_dir):
    """
    Compare global distributions (not channel matched)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    summary_rows = []

    numeric_cols = list(set(get_numeric_columns(df1)) &
                        set(get_numeric_columns(df2)))

    for col in numeric_cols:
        if col in ID_COLUMNS:
            continue

        v1 = df1[col].dropna().values
        v2 = df2[col].dropna().values

        if len(v1) == 0 or len(v2) == 0:
            continue

        ks_stat, ks_p = ks_2samp(v1, v2)

        summary_rows.append({
            "parameter": col,
            f"{label1}_mean": np.mean(v1),
            f"{label2}_mean": np.mean(v2),
            f"{label1}_std": np.std(v1),
            f"{label2}_std": np.std(v2),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
        })

        plt.figure(figsize=(7, 5))
        plt.hist(v1, bins=100, density=True,
                 histtype="step", linewidth=1.5, label=label1)
        plt.hist(v2, bins=100, density=True,
                 histtype="step", linewidth=1.5, label=label2)

        plt.xlabel(col)
        plt.ylabel("Normalized entries")
        plt.title(f"{col} comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}.png")
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    return summary_df


def compare_channel_matched(df1, df2, label1, label2, output_dir):
    """
    Compare channel-by-channel differences (much more meaningful)
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    merged = df1.merge(
        df2,
        on=ID_COLUMNS,
        suffixes=(f"_{label1}", f"_{label2}")
    )

    numeric_cols = get_numeric_columns(df1)

    for col in numeric_cols:
        if col in ID_COLUMNS:
            continue

        col1 = f"{col}_{label1}"
        col2 = f"{col}_{label2}"

        if col1 not in merged or col2 not in merged:
            continue

        diff = merged[col1] - merged[col2]

        plt.figure(figsize=(7, 5))
        plt.hist(diff.dropna(), bins=100, histtype="step")
        plt.xlabel(f"{col} difference ({label1} - {label2})")
        plt.ylabel("Entries")
        plt.title(f"Channel-matched difference: {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_difference.png")
        plt.close()

    merged.to_csv(output_dir / "channel_matched_table.csv", index=False)
    return merged


def run_comparison(file1, file2, tag):
    df1 = load_csv(file1)
    df2 = load_csv(file2)

    base_dir = Path(f"comparison_{tag}")

    print(f"\n=== Global comparison: {tag} ===")
    compare_global(df1, df2, "lab", "testbeam",
                   base_dir / "global")

    print(f"\n=== Channel-matched comparison: {tag} ===")
    compare_channel_matched(df1, df2, "lab", "testbeam",
                            base_dir / "channel_matched")


def main():
    run_comparison("/eos/user/z/zhibin/TestBeam/run_000041/qdc_cal.csv",
                   "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/qdc_cal.csv",
                   "qdc")

    run_comparison("/eos/user/z/zhibin/TestBeam/run_000041/tdc_cal.csv",
                   "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/tdc_cal.csv",
                   "tac")


if __name__ == "__main__":
    main()
