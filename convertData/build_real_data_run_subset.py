#!/usr/bin/env python3
"""Select a deterministic, file-position-stratified subset from every data run."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import resolve_product_path


FILE_NUMBER = re.compile(r"(?:sndsw_raw-|raw-)(\d+)")
REQUIRED_COLUMNS = {
    "run", "n_event", "lumi_per_file", "output_base_path",
    "feature_path", "hit3d_path",
}


def normalized_run(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or not text.lstrip("-").isdigit():
        raise ValueError(f"Invalid run value {value!r}")
    return text


def file_order_key(row: pd.Series) -> tuple[int, str]:
    paths = (str(row.get("digi_path", "")), str(row["feature_path"]))
    for path in paths:
        match = FILE_NUMBER.search(path)
        if match:
            return int(match.group(1)), path
    return 2**63 - 1, "|".join(paths)


def stable_choice_key(seed: int, run: str, feature_path: str, hit3d_path: str) -> str:
    value = f"{seed}\x1f{run}\x1f{feature_path}\x1f{hit3d_path}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_subset(metadata: pd.DataFrame, files_per_run: int, seed: int,
                 workers: int = 32, expected_runs: int | None = None):
    if files_per_run <= 0:
        raise ValueError("files_per_run must be positive")
    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Input metadata lacks columns: {sorted(missing)}")

    frame = metadata.copy()
    frame["_ml_run"] = frame["run"].map(normalized_run)
    if expected_runs is not None and frame["_ml_run"].nunique() != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs, found {frame['_ml_run'].nunique()}"
        )
    frame["_ml_feature_full_path"] = [
        resolve_product_path(row, "feature_path") for _, row in frame.iterrows()
    ]
    frame["_ml_hit3d_full_path"] = [
        resolve_product_path(row, "hit3d_path") for _, row in frame.iterrows()
    ]
    path_pairs = list(zip(
        frame["_ml_feature_full_path"], frame["_ml_hit3d_full_path"]
    ))

    def available(pair):
        feature_path, hit3d_path = pair
        return os.path.exists(feature_path) and os.path.exists(hit3d_path)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        frame["_ml_input_available"] = list(executor.map(available, path_pairs))

    selected_parts = []
    summary_rows = []
    for run, all_rows in frame.groupby("_ml_run", sort=True):
        available_rows = all_rows.loc[all_rows["_ml_input_available"]].copy()
        if len(available_rows) < files_per_run:
            raise ValueError(
                f"Run {run} has only {len(available_rows)} available feature/Hit3D "
                f"pairs; {files_per_run} are required"
            )
        available_rows["_ml_order_key"] = [
            file_order_key(row) for _, row in available_rows.iterrows()
        ]
        available_rows = available_rows.sort_values(
            "_ml_order_key", kind="stable"
        ).reset_index(drop=False).rename(columns={"index": "_ml_original_row"})
        available_rows["_ml_run_file_rank"] = range(len(available_rows))

        chosen = []
        total = len(available_rows)
        for stratum in range(files_per_run):
            start = stratum * total // files_per_run
            stop = (stratum + 1) * total // files_per_run
            candidates = available_rows.iloc[start:stop].copy()
            candidates["_ml_choice_key"] = [
                stable_choice_key(seed, run, feature_path, hit3d_path)
                for feature_path, hit3d_path in zip(
                    candidates["_ml_feature_full_path"],
                    candidates["_ml_hit3d_full_path"],
                )
            ]
            row = candidates.sort_values("_ml_choice_key", kind="stable").iloc[0].copy()
            row["ml_subset_version"] = (
                f"real_data_2024_{files_per_run}_files_per_run_v1"
            )
            row["ml_subset_seed"] = int(seed)
            row["ml_run_file_rank"] = int(row["_ml_run_file_rank"])
            row["ml_run_available_files"] = int(total)
            row["ml_run_stratum"] = int(stratum)
            chosen.append(row)

        selected_run = pd.DataFrame(chosen)
        selected_parts.append(selected_run)
        summary_rows.append({
            "run": run,
            "metadata_files": int(len(all_rows)),
            "available_feature_hit3d_pairs": int(total),
            "selected_files": int(len(selected_run)),
            "selected_original_events": int(
                pd.to_numeric(selected_run["n_event"]).sum()
            ),
            "selected_lumi_per_file_sum": float(
                pd.to_numeric(selected_run["lumi_per_file"]).sum()
            ),
            "first_selected_file_rank": int(selected_run["ml_run_file_rank"].min()),
            "last_selected_file_rank": int(selected_run["ml_run_file_rank"].max()),
        })

    result = pd.concat(selected_parts, ignore_index=True)
    internal = [name for name in result.columns if name.startswith("_ml_")]
    result = result.drop(columns=internal)
    if result.duplicated(["run", "ml_run_stratum"]).any():
        raise RuntimeError("A run/stratum was selected more than once")
    if result.groupby(result["run"].map(normalized_run)).size().ne(files_per_run).any():
        raise RuntimeError("The output does not contain exactly files_per_run per run")
    if result["feature_path"].astype(str).duplicated().any():
        raise RuntimeError("The selected feature paths are not unique")
    summary = pd.DataFrame(summary_rows)
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--files-per-run", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--expected-runs", type=int)
    args = parser.parse_args()

    metadata = pd.read_csv(args.input, low_memory=False)
    result, summary = build_subset(
        metadata,
        files_per_run=args.files_per_run,
        seed=args.seed,
        workers=args.workers,
        expected_runs=args.expected_runs,
    )
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary.to_csv(summary_path, index=False)
    print(
        f"Wrote {len(result)} rows from {len(summary)} runs to {output}; "
        f"selected luminosity={summary['selected_lumi_per_file_sum'].sum():.6g}"
    )


if __name__ == "__main__":
    main()
