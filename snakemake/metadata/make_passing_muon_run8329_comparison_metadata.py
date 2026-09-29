#!/usr/bin/env python3
"""Build source-paired metadata for existing run-8329 DQ and passing features."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def atomic_csv(frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} metadata lacks columns: {sorted(missing)}")


def full_feature_path(row: pd.Series) -> str:
    return str(Path(str(row["output_base_path"])) / str(row["feature_path"]))


def real_dq_feature_path(source_index: int, config: dict) -> str:
    root = config["output"].get("real_dq_feature_root")
    if not root:
        raise ValueError("output.real_dq_feature_root is required")
    return str(
        Path(str(root))
        / f"realDataDQ_run008329_source{int(source_index):03d}.root"
    )


def validate_ranges(chunks: pd.DataFrame, source_entries: int, source_index: int) -> None:
    ordered = chunks.sort_values(["attempt_start", "attempt_stop", "chunk_index"])
    cursor = 0
    for _, row in ordered.iterrows():
        start = int(row["attempt_start"])
        stop = int(row["attempt_stop"])
        if start != cursor:
            raise ValueError(
                f"Source {source_index} has a gap/overlap: expected start {cursor}, found {start}"
            )
        if stop <= start or stop > source_entries:
            raise ValueError(
                f"Source {source_index} has invalid range [{start}, {stop}) for {source_entries} entries"
            )
        if int(row["n_attempted_source_events"]) != stop - start:
            raise ValueError(f"Source {source_index} range count disagrees with metadata")
        cursor = stop
    if cursor != source_entries:
        raise ValueError(
            f"Source {source_index} chunk coverage ends at {cursor}, expected {source_entries}"
        )


def assign_source_partitions(source_rows: pd.DataFrame, n_partitions: int) -> pd.Series:
    """Assign contiguous sources once; both samples inherit this assignment."""
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive")
    if n_partitions > len(source_rows):
        raise ValueError("n_partitions cannot exceed the source-file count")
    weights = pd.to_numeric(source_rows["source_n_event"]).astype(int).tolist()
    total = sum(weights)
    assignments = []
    current_part = 1
    cumulative = 0
    for index, weight in enumerate(weights):
        assignments.append(current_part)
        cumulative += weight
        remaining_sources = len(weights) - index - 1
        remaining_parts = n_partitions - current_part
        if (
            current_part < n_partitions
            and cumulative >= total * current_part / n_partitions
            and remaining_sources >= remaining_parts
        ):
            current_part += 1
    if sorted(set(assignments)) != list(range(1, n_partitions + 1)):
        raise RuntimeError("Failed to create every requested paired partition")
    return pd.Series(assignments, index=source_rows.index, dtype=int)


def build_tables(inclusive: pd.DataFrame, passing: pd.DataFrame, config: dict):
    run = int(config["run"])
    expected = config["expected"]
    require_columns(
        inclusive,
        {
            "run", "n_event", "digi_path", "geo_path", "lumi_per_file",
            "output_base_path", "feature_path",
        },
        "real-data DQ",
    )
    require_columns(
        passing,
        {
            "run", "source_file_index", "chunk_index", "attempt_start",
            "attempt_stop", "source_n_event", "n_attempted_source_events",
            "n_saved_event", "source_path", "reco_path", "geo_path",
            "input_tree", "output_base_path", "feature_path", "lumi_per_file",
        },
        "passing",
    )

    inclusive = inclusive.loc[pd.to_numeric(inclusive["run"]) == run].copy()
    passing = passing.loc[pd.to_numeric(passing["run"]) == run].copy()
    if inclusive.empty or passing.empty:
        raise ValueError(f"Run {run} is missing from one or both metadata inputs")

    inclusive["source_path"] = inclusive["digi_path"].astype(str)
    if inclusive["source_path"].duplicated().any():
        raise ValueError("Inclusive metadata contains duplicate source paths")
    if passing["reco_path"].astype(str).duplicated().any():
        raise ValueError("Passing metadata contains duplicate reconstructed paths")

    source_map = (
        passing[["source_file_index", "source_path", "source_n_event"]]
        .drop_duplicates()
        .sort_values("source_file_index")
    )
    if source_map["source_file_index"].duplicated().any() or source_map["source_path"].duplicated().any():
        raise ValueError("Passing metadata source mapping is not one-to-one")
    expected_indices = list(range(len(source_map)))
    observed_indices = source_map["source_file_index"].astype(int).tolist()
    if observed_indices != expected_indices:
        raise ValueError("source_file_index must be contiguous and zero based")

    inclusive = source_map.merge(inclusive, on="source_path", how="left", validate="one_to_one")
    if inclusive["digi_path"].isna().any():
        missing = inclusive.loc[inclusive["digi_path"].isna(), "source_path"].tolist()
        raise ValueError(f"Passing sources absent from inclusive metadata: {missing[:10]}")
    if len(inclusive) != len(source_map):
        raise ValueError("Inclusive/passing source join changed the source count")
    if (pd.to_numeric(inclusive["n_event"]) != pd.to_numeric(inclusive["source_n_event"])).any():
        raise ValueError("Inclusive n_event disagrees with passing source_n_event")

    for source_index, chunks in passing.groupby("source_file_index", sort=True):
        source_entries = int(chunks["source_n_event"].iloc[0])
        validate_ranges(chunks, source_entries, int(source_index))

    attempted = int(pd.to_numeric(passing["n_attempted_source_events"]).sum())
    passing_events = int(pd.to_numeric(passing["n_saved_event"]).sum())
    source_events = int(pd.to_numeric(source_map["source_n_event"]).sum())
    checks = {
        "source_files": len(source_map),
        "reconstructed_chunks": len(passing),
        "attempted_events": attempted,
        "passing_events": passing_events,
    }
    for key, observed in checks.items():
        if observed != int(expected[key]):
            raise ValueError(f"Expected {key}={expected[key]}, found {observed}")
    if source_events != attempted:
        raise ValueError(f"Source-event total {source_events} != attempted total {attempted}")

    authoritative_lumi = float(expected["luminosity_fb"])
    lumi_tolerance = float(expected.get("luminosity_tolerance_fb", 1e-12))
    passing_lumi = float(pd.to_numeric(passing["lumi_per_file"]).sum())
    if abs(passing_lumi - authoritative_lumi) > lumi_tolerance:
        raise ValueError(
            f"Passing luminosity {passing_lumi:.15g} != expected {authoritative_lumi:.15g}"
        )

    inclusive = inclusive.sort_values("source_file_index").reset_index(drop=True)
    inclusive["data_type"] = "real_data"
    inclusive["selection_mode"] = "external_cutflow_data_quality"
    inclusive["source_metadata_row"] = inclusive["source_file_index"].astype(int)
    inclusive["feature_full_path"] = inclusive["source_file_index"].map(
        lambda value: real_dq_feature_path(int(value), config)
    )
    # The legacy real-data metadata carries a rounded run exposure (0.5325
    # fb^-1 for run 8329).  Preserve it for audit, but use the authoritative
    # exposure validated against the attempted reconstruction chunks.  Source
    # shares follow attempted source entries and therefore sum exactly once.
    inclusive["source_lumi_original"] = pd.to_numeric(inclusive["lumi_per_file"])
    inclusive["lumi_per_file"] = (
        pd.to_numeric(inclusive["source_n_event"]) / attempted * authoritative_lumi
    )
    n_partitions = int(config.get("partitioning", {}).get("n_partitions", 20))
    inclusive["paired_part_index"] = assign_source_partitions(inclusive, n_partitions)

    passing = passing.sort_values(["source_file_index", "attempt_start"]).reset_index(drop=True)
    passing["source_metadata_row"] = passing.index.astype(int)
    passing["selection_mode"] = "passing_muon_reco_before_data_quality"
    passing["feature_full_path"] = passing.apply(full_feature_path, axis=1)
    source_to_part = dict(
        zip(inclusive["source_file_index"].astype(int), inclusive["paired_part_index"].astype(int))
    )
    passing["paired_part_index"] = passing["source_file_index"].astype(int).map(source_to_part)

    pair_rows = []
    for _, source in inclusive.iterrows():
        source_index = int(source["source_file_index"])
        chunks = passing.loc[pd.to_numeric(passing["source_file_index"]) == source_index]
        pair_rows.append(
            {
                "run": run,
                "source_file_index": source_index,
                "source_path": source["source_path"],
                "source_n_event": int(source["source_n_event"]),
                "real_dq_feature_path": source["feature_full_path"],
                "passing_chunk_count": len(chunks),
                "passing_chunk_paths": ";".join(chunks["reco_path"].astype(str)),
                "passing_feature_paths": ";".join(chunks["feature_full_path"].astype(str)),
                "attempted_event_count": int(chunks["n_attempted_source_events"].sum()),
                "passing_before_dq": int(chunks["n_saved_event"].sum()),
                "lumi_per_source": float(source["lumi_per_file"]),
                "paired_part_index": int(source["paired_part_index"]),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    validation = pd.DataFrame(
        [{
            "status": "pass",
            **checks,
            "source_events": source_events,
            "luminosity_fb": authoritative_lumi,
            "raw_efficiency": passing_events / attempted,
        }]
    )
    return inclusive, passing, pairs, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # Do not resolve AFS symlinks: user and work-space aliases can have
    # different write permissions even when they identify the same checkout.
    repo_root = Path(args.repo_root).absolute()
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else repo_root / path
    inclusive = pd.read_csv(resolve(config["inputs"]["inclusive_metadata"]))
    passing = pd.read_csv(resolve(config["inputs"]["passing_metadata"]))
    tables = build_tables(inclusive, passing, config)
    output_dir = Path(args.output_dir) if args.output_dir else resolve(config["output"]["metadata_dir"])
    names = ["inclusive_metadata", "passing_metadata", "source_pairs", "validation"]
    for name, table in zip(names, tables):
        atomic_csv(table, output_dir / config["output"][name])
    print(
        f"Validated run {config['run']}: {len(tables[2])} sources, "
        f"{len(tables[1])} chunks, efficiency={tables[3].iloc[0]['raw_efficiency']:.6%}"
    )


if __name__ == "__main__":
    main()
