#!/usr/bin/env python3
"""Build the deterministic source manifest for 2024 real-data indexes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


SOURCE_FAMILY = "real_data_2024_skim"


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def resolve(row: pd.Series, column: str) -> str:
    value = clean(row.get(column, ""))
    if not value:
        return ""
    if value.startswith("root://") or Path(value).is_absolute():
        return value
    base = clean(row.get("output_base_path", "")).rstrip("/")
    return f"{base}/{value.lstrip('/')}" if base else value


def source_uid(metadata_file: str, source_row: int, digi_path: str) -> str:
    digest = hashlib.sha256(
        f"{SOURCE_FAMILY}\x1f{metadata_file}\x1f{source_row}\x1f{digi_path}".encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    return f"unified_src_{digest}"


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_manifest(config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    metadata_path = str(config["source_metadata"])
    source = pd.read_csv(metadata_path, keep_default_na=False)
    if len(source) != int(config["expected_source_files"]):
        raise ValueError(
            f"expected {config['expected_source_files']} sources, found {len(source)}"
        )
    index_root = str(config["index_root"]).rstrip("/")
    records = []
    for source_row, row in source.iterrows():
        digi_path = clean(row.get("digi_path"))
        cutflow_path = resolve(row, "nueAnalysisFilter_path")
        if not digi_path or not cutflow_path:
            raise ValueError(f"missing input path at source row {source_row}")
        uid = source_uid(metadata_path, source_row, digi_path)
        output_dir = f"{index_root}/{uid}"
        lumi = clean(row.get("lumi_per_file", ""))
        records.append(
            {
                "manifest_version": str(config["version"]),
                "source_uid": uid,
                "source_metadata_file": metadata_path,
                "source_metadata_row": int(source_row),
                "digi_path": digi_path,
                "base_cutflow_path": cutflow_path,
                "input_tree": str(config.get("input_tree", "cbmsim")),
                "input_event_count": int(row["n_event"]),
                "lumi_per_file": float(lumi) if lumi else "",
                "no_veto_index_path": f"{output_dir}/no_veto.root",
                "veto_10pct_index_path": f"{output_dir}/veto_10pct.root",
                "selection_summary_path": f"{output_dir}/selection_summary.json",
                "validation_record": f"{output_dir}/selection.validated.json",
                "is_canary": 0,
            }
        )
    frame = pd.DataFrame(records).sort_values("source_uid", kind="mergesort")
    frame = frame.reset_index(drop=True)
    for column in (
        "source_uid",
        "no_veto_index_path",
        "veto_10pct_index_path",
        "validation_record",
    ):
        if frame[column].duplicated().any():
            raise ValueError(f"duplicate {column}")
    canary_row = int(config["canary"]["source_metadata_row"])
    matches = frame.index[frame["source_metadata_row"].astype(int).eq(canary_row)]
    if len(matches) != 1:
        raise ValueError(f"expected one canary at metadata row {canary_row}")
    frame.loc[matches[0], "is_canary"] = 1
    return frame


def write_manifest(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = build_manifest(args.config)
    write_manifest(frame, args.output)
    print(f"Wrote {len(frame)} sources to {args.output}")


if __name__ == "__main__":
    main()
