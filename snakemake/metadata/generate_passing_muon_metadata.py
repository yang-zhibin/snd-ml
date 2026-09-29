#!/usr/bin/env python3
"""Build exposure-correct metadata for track-selected 2024 proton data.

The reconstructed tree entry count is a selected yield.  Exposure is therefore
allocated using the number of source events attempted by each reconstruction
job, and the random-sampling scale is encoded exactly once in lumi_per_file.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml


WHOLE_RE = re.compile(r"muonReco_RUN-(?P<run>\d+)_f(?P<source>\d+)\.0\.0\.root$")
CHUNK_RE = re.compile(
    r"sndsw_raw-(?P<chunk>\d+)_(?P<source>\d+)_(?P<run>\d+)_muonReco\.root$"
)
SOURCE_TEMPLATE = "sndsw_raw-{source:04d}.root"
SOURCE_RE = re.compile(r"sndsw_raw-(?P<source>\d{4})\.root$")


def root_entries(path: str, tree_name: str = "cbmsim") -> int:
    """Read cbmsim entries, importing ROOT only in the production process."""
    import ROOT

    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Missing {tree_name} tree: {path}")
        return int(tree.GetEntriesFast())
    finally:
        root_file.Close()


def _files(directory: str) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted(path.glob("*.root"), key=lambda item: item.name)


def _load_lumi(pp24_csv: str, pp24_run: int) -> tuple[int, float, float]:
    table = pd.read_csv(pp24_csv)
    required = {"Run", "Fill", "scale", "lumi_nb"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"pp24 CSV lacks columns: {sorted(missing)}")
    match = table.loc[table["Run"].astype(int) == pp24_run]
    if len(match) != 1:
        raise ValueError(f"Expected one pp24 row for run {pp24_run}, found {len(match)}")
    row = match.iloc[0]
    scale = float(row["scale"])
    lumi_nb = float(row["lumi_nb"])
    if scale <= 0 or lumi_nb <= 0:
        raise ValueError(f"Non-positive scale/luminosity for pp24 run {pp24_run}")
    return int(row["Fill"]), scale, lumi_nb


def _source_path(source_dir: str, source_index: int) -> Path:
    return Path(source_dir) / SOURCE_TEMPLATE.format(source=source_index)


def discover_run(
    spec: dict,
    entry_reader: Callable[[str], int] = root_entries,
) -> tuple[list[dict], list[dict]]:
    """Map canonical reconstruction outputs to source files.

    Files carrying a non-canonical run number are reported and excluded.  Any
    malformed ROOT filename is an error rather than silently receiving lumi.
    """
    run = int(spec["run"])
    mode = spec["mode"]
    pattern = WHOLE_RE if mode == "whole_file" else CHUNK_RE if mode == "chunked" else None
    if pattern is None:
        raise ValueError(f"Unsupported reconstruction mode {mode!r} for run {run}")

    rows, excluded = [], []
    seen_keys = set()
    source_entry_cache = {}
    for reco_path in _files(spec["reco_dir"]):
        match = pattern.fullmatch(reco_path.name)
        if not match:
            raise ValueError(f"Unexpected ROOT filename in run {run}: {reco_path}")
        file_run = int(match.group("run"))
        source_index = int(match.group("source"))
        if file_run != run:
            excluded.append({
                "canonical_run": run,
                "file_run": file_run,
                "reco_path": str(reco_path),
                "reason": "filename_run_mismatch",
            })
            continue

        source_path = _source_path(spec["source_dir"], source_index)
        if not source_path.is_file():
            raise FileNotFoundError(f"No source file for {reco_path}: {source_path}")
        source_key = str(source_path)
        if source_key not in source_entry_cache:
            try:
                source_entry_cache[source_key] = entry_reader(source_key, "cbmsim")
            except TypeError:
                source_entry_cache[source_key] = entry_reader(source_key)
        source_n_event = source_entry_cache[source_key]
        input_tree = str(spec.get("reco_tree", "cbmsim"))
        try:
            saved_n_event = entry_reader(str(reco_path), input_tree)
        except TypeError:
            saved_n_event = entry_reader(str(reco_path))
        if source_n_event < 0 or saved_n_event < 0:
            raise ValueError(f"Negative entry count for {reco_path}")

        if mode == "whole_file":
            chunk_index, start, stop = None, 0, source_n_event
            key = source_index
        else:
            chunk_index = int(match.group("chunk"))
            chunk_size = int(spec["chunk_size"])
            if chunk_size <= 0:
                raise ValueError(f"Invalid chunk_size for run {run}: {chunk_size}")
            start = chunk_index * chunk_size
            stop = min(start + chunk_size, source_n_event)
            if start >= source_n_event:
                raise ValueError(f"Chunk starts beyond source entries: {reco_path}")
            key = (source_index, chunk_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate reconstruction range for run {run}: {key}")
        seen_keys.add(key)

        rows.append({
            "run": run,
            "source_file_index": source_index,
            "chunk_index": chunk_index,
            "attempt_start": start,
            "attempt_stop": stop,
            "source_n_event": source_n_event,
            "n_attempted_source_events": stop - start,
            "n_event": saved_n_event,
            "n_saved_event": saved_n_event,
            "source_path": str(source_path),
            "reco_path": str(reco_path),
            "digi_path": str(reco_path),
            "reconstruction_mode": mode,
            "input_tree": input_tree,
        })

    if not rows:
        raise ValueError(f"No canonical reconstruction files found for run {run}")
    available_sources = sorted(
        int(match.group("source"))
        for path in _files(spec["source_dir"])
        if (match := SOURCE_RE.fullmatch(path.name))
    )
    missing_sources = _validate_ranges(
        run, mode, rows,
        available_sources=available_sources,
        allow_missing=bool(spec.get("allow_missing_source_outputs", False)),
    )
    for source_index in missing_sources:
        excluded.append({
            "canonical_run": run,
            "file_run": None,
            "reco_path": None,
            "source_path": str(_source_path(spec["source_dir"], source_index)),
            "reason": "no_reconstruction_output",
        })
    return rows, excluded


def _validate_ranges(
    run: int,
    mode: str,
    rows: list[dict],
    available_sources: list[int] | None = None,
    allow_missing: bool = False,
) -> list[int]:
    by_source: dict[int, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source_file_index"], []).append(row)
    source_indices = sorted(by_source)
    expected_indices = available_sources or list(range(source_indices[-1] + 1))
    missing_sources = sorted(set(expected_indices) - set(source_indices))
    if missing_sources and not allow_missing:
        raise ValueError(f"Run {run} has missing source-file indices: {missing_sources}")

    for source_index, source_rows in by_source.items():
        ordered = sorted(source_rows, key=lambda row: row["attempt_start"])
        expected = 0
        for row in ordered:
            if row["attempt_start"] != expected:
                raise ValueError(
                    f"Run {run}, source {source_index}: gap/overlap at {expected}, "
                    f"next start is {row['attempt_start']}"
                )
            expected = row["attempt_stop"]
        if expected != ordered[0]["source_n_event"]:
            raise ValueError(
                f"Run {run}, source {source_index}: coverage stops at {expected}, "
                f"source has {ordered[0]['source_n_event']} events"
            )
        if mode == "whole_file" and len(ordered) != 1:
            raise ValueError(f"Run {run}, source {source_index}: multiple whole-file outputs")
    return missing_sources


def _add_output_paths(frame: pd.DataFrame, dataset: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["output_base_path"] = frame.apply(
        lambda row: (
            f"{dataset['output_root']}/{dataset['data_type']}/"
            f"{dataset['subfolder']}/{row['partition']}"
        ),
        axis=1,
    )
    suffixes = {
        "newRaw_path": ".root", "newDigi_path": ".root", "preSelect_path": ".root",
        "nueAnalysis_path": ".root", "nueAnalysisFilter_path": ".root",
        "preCutEff_path": ".csv", "hit_path": ".root", "hit3d_path": ".root",
        "feature_path": ".root", "pt_hit_path": ".pt.gz", "npz_hit_path": ".npz",
    }
    for column, suffix in suffixes.items():
        tag = column.removesuffix("_path")
        frame[column] = frame.apply(
            lambda row: f"{tag}_{dataset['data_type']}_{dataset['subfolder']}_"
            f"{row['partition']}_{Path(row['reco_path']).stem}{suffix}",
            axis=1,
        )
    return frame


def build_metadata(config: dict, entry_reader: Callable[[str], int] = root_entries):
    dataset = config["dataset"]
    if dataset.get("beam_type") != "proton" or int(dataset.get("year", 0)) != 2024:
        raise ValueError("This builder accepts only 2024 proton data")

    all_rows, all_excluded = [], []
    for spec in config["runs"]:
        rows, excluded = discover_run(spec, entry_reader)
        fill, scale, run_lumi_nb = _load_lumi(dataset["pp24_csv"], int(spec["pp24_run"]))
        total_attempted = sum(row["n_attempted_source_events"] for row in rows)
        if total_attempted <= 0:
            raise ValueError(f"Run {spec['run']} has no attempted source events")
        for row in rows:
            fraction = row["n_attempted_source_events"] / total_attempted
            full_lumi_fb = run_lumi_nb / 1_000_000.0
            row.update({
                "data_type": dataset["data_type"],
                "category": dataset["data_type"],
                "is_real_data": True,
                "year": 2024,
                "beam_type": "proton",
                "subfolder": str(dataset["subfolder"]),
                "partition": f"run_{int(spec['run']):06d}",
                "production_profile": "real_data_default",
                "selection_mode": "passing_muon_direct",
                "uses_nue_analysis_filter": False,
                "pp24_run": int(spec["pp24_run"]),
                "fill": fill,
                "scale_factor": scale,
                "run_lumi_nb": run_lumi_nb,
                "run_lumi_fb": full_lumi_fb,
                "effective_run_lumi_fb": full_lumi_fb / scale,
                "lumi_fraction": fraction,
                "full_lumi_per_file": full_lumi_fb * fraction,
                "lumi_per_file": full_lumi_fb * fraction / scale,
                "geo_path": dataset["geo_path"],
            })
        _validate_lumi_closure(int(spec["run"]), rows)
        all_rows.extend(rows)
        all_excluded.extend(excluded)

    frame = pd.DataFrame(all_rows).sort_values(
        ["run", "source_file_index", "attempt_start"]
    ).reset_index(drop=True)
    excluded_columns = ["canonical_run", "file_run", "reco_path", "source_path", "reason"]
    return _add_output_paths(frame, dataset), pd.DataFrame(all_excluded, columns=excluded_columns)


def _validate_lumi_closure(run: int, rows: list[dict]) -> None:
    run_lumi = rows[0]["run_lumi_fb"]
    scale = rows[0]["scale_factor"]
    full_sum = sum(row["full_lumi_per_file"] for row in rows)
    effective_sum = sum(row["lumi_per_file"] for row in rows)
    tolerance = max(1e-12, abs(run_lumi) * 1e-10)
    if abs(full_sum - run_lumi) > tolerance:
        raise ValueError(f"Run {run}: full-luminosity closure failed")
    if abs(effective_sum * scale - run_lumi) > tolerance:
        raise ValueError(f"Run {run}: effective-luminosity closure failed")


def _atomic_csv(frame: pd.DataFrame, output: str) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded-report", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metadata, excluded = build_metadata(config)
    _atomic_csv(metadata, args.output)
    _atomic_csv(excluded, args.excluded_report)
    print(f"Wrote {len(metadata)} rows to {args.output}")
    print(f"Recorded {len(excluded)} excluded/absent files; report: {args.excluded_report}")


if __name__ == "__main__":
    main()
