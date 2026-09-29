#!/usr/bin/env python3
"""Select and validate one run from passing-muon track-feature metadata."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_csv(frame, output):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--expected-lumi", type=float)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {
        "run",
        "n_saved_event",
        "lumi_per_file",
        "output_base_path",
        "feature_path",
        "track_feature_schema",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Track metadata lacks columns: {sorted(missing)}")
    selected = frame.loc[frame["run"].astype(int) == args.run].copy()
    if selected.empty:
        raise ValueError(f"No rows selected for run {args.run}")
    if set(selected["track_feature_schema"].astype(str)) != {"flat_v1"}:
        raise ValueError("Selected metadata contains a non-flat_v1 track schema")
    full_paths = (
        selected["output_base_path"].astype(str).str.rstrip("/")
        + "/"
        + selected["feature_path"].astype(str).str.lstrip("/")
    )
    if full_paths.duplicated().any():
        raise ValueError("Selected metadata contains duplicate feature paths")

    n_files = len(selected)
    n_events = int(pd.to_numeric(selected["n_saved_event"]).sum())
    luminosity = float(pd.to_numeric(selected["lumi_per_file"]).sum())
    if args.expected_files is not None and n_files != args.expected_files:
        raise ValueError(f"Expected {args.expected_files} files, found {n_files}")
    if args.expected_events is not None and n_events != args.expected_events:
        raise ValueError(f"Expected {args.expected_events} events, found {n_events}")
    if args.expected_lumi is not None and abs(luminosity - args.expected_lumi) > 1e-12:
        raise ValueError(
            f"Expected luminosity {args.expected_lumi:.15g}, found {luminosity:.15g}"
        )

    selected = selected.reset_index(drop=True)
    selected["source_metadata_row"] = selected.index.astype(int)
    atomic_csv(selected, args.output)
    print(
        f"Wrote run {args.run}: {n_files} files, {n_events} events, "
        f"{luminosity:.15g} fb^-1"
    )


if __name__ == "__main__":
    main()
