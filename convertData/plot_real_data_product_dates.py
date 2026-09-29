#!/usr/bin/env python3
"""Plot modification-date distributions for real-data feature and hit3D files."""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def resolve(row, column):
    value = str(row[column])
    if value.startswith("/"):
        return value
    return str(row["output_base_path"]).rstrip("/") + "/" + value.lstrip("/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    directory_entries = {}
    for directory in metadata["output_base_path"].astype(str).unique():
        try:
            directory_entries[directory] = {
                entry.name: entry.stat()
                for entry in os.scandir(directory)
                if entry.is_file()
            }
        except OSError:
            directory_entries[directory] = {}
    records = []
    for product, column in (("feature", "feature_path"), ("hit3d", "hit3d_path")):
        for _, row in metadata.iterrows():
            path = resolve(row, column)
            try:
                directory = str(row["output_base_path"]).rstrip("/")
                stat = directory_entries[directory][os.path.basename(path)]
                timestamp = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
                records.append({
                    "product": product,
                    "partition": row.get("partition", ""),
                    "run_number": row.get("run_number", row.get("run", "")),
                    "path": path,
                    "date_utc": timestamp.floor("D"),
                    "mtime_utc": timestamp,
                    "size_bytes": stat.st_size,
                    "exists": 1,
                })
            except (OSError, KeyError):
                records.append({
                    "product": product, "partition": row.get("partition", ""),
                    "run_number": row.get("run_number", row.get("run", "")),
                    "path": path, "date_utc": pd.NaT, "mtime_utc": pd.NaT,
                    "size_bytes": -1, "exists": 0,
                })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "real_data_product_file_dates.csv", index=False)
    present = frame[frame["exists"] == 1].copy()
    daily = present.groupby(["date_utc", "product"]).size().unstack(fill_value=0)
    daily.to_csv(output / "real_data_product_daily_counts.csv")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    colors = {"feature": "#3366cc", "hit3d": "#dc3912"}
    for axis, product in zip(axes, ("feature", "hit3d")):
        counts = daily[product] if product in daily else pd.Series(dtype=int)
        axis.bar(counts.index, counts.values, width=0.85, color=colors[product])
        axis.set_title(f"Real-data {product} ROOT files by modification date (UTC)")
        axis.set_ylabel("Number of files")
        axis.grid(axis="y", alpha=0.25)
        axis.text(0.99, 0.95, f"present: {(present['product'] == product).sum():,}\nmissing: {((frame['product'] == product) & (frame['exists'] == 0)).sum():,}",
                  transform=axis.transAxes, ha="right", va="top")
    axes[-1].set_xlabel("Modification date")
    fig.savefig(output / "real_data_product_file_date_histogram.png", dpi=170)

    print(daily.to_string())
    print(f"\nWrote outputs to {output}")


if __name__ == "__main__":
    main()
