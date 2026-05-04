import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def load_npz_events(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    events = list(data["events"])
    return events


def save_merged(events, split_name, split_tag, out_dir, source_files, metadata_csv):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split_tag}_{split_name}.npz"

    output = {
        "events": events,
        "metadata": {
            "split_name": split_name,
            "split_tag": split_tag,
            "n_selected": len(events),
            "n_source_files": len(source_files),
            "source_files": source_files,
            "metadata_csv": str(metadata_csv),
        },
    }

    np.savez_compressed(out_path, **output)

    print(f"[{split_name}] saved {len(events)} events")
    print(f"[{split_name}] output: {out_path}")


def main(args):
    df = pd.read_csv(args.metadata_csv)

    train_events = []
    val_events = []

    train_sources = []
    val_sources = []

    for _, row in df.iterrows():
        split = str(row["split"]).strip().lower()

        if split not in ("train", "val"):
            continue

        full_path = Path(row["output_base_path"]) / row["npz_hit_path"]

        if not full_path.exists():
            print(f"[WARNING] missing file: {full_path}")
            continue

        events = load_npz_events(full_path)

        if split == "train":
            train_events.extend(events)
            train_sources.append(str(full_path))

        elif split == "val":
            val_events.extend(events)
            val_sources.append(str(full_path))

        print(f"[{split}] loaded {len(events)} events from {full_path}")

    save_merged(
        train_events,
        "train",
        args.split,
        args.output_dir,
        train_sources,
        args.metadata_csv,
    )

    save_merged(
        val_events,
        "val",
        args.split,
        args.output_dir,
        val_sources,
        args.metadata_csv,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--metadata-csv",
        required=True,
        help="CSV file with columns: output_base_path, npz_hit_path, split",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save merged files",
    )

    parser.add_argument(
        "-s",
        "--split",
        required=True,
        help="Split tag used in output name, e.g. 0, 1, foldA",
    )

    args = parser.parse_args()

    main(args)