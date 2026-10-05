#!/usr/bin/env python3
"""Map existing 2024 MC metadata to isolated, versioned feature outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def resolve_source_path(row: pd.Series, column: str) -> str:
    value = str(row.get(column, "") or "")
    if not value or value.lower() == "nan" or value.startswith("root://") or Path(value).is_absolute():
        return value
    return f"{str(row['output_base_path']).rstrip('/')}/{value.lstrip('/')}"


def prepare_metadata(config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    feature_root = str(config["outputs"]["feature_root"]).rstrip("/")
    template = config["feature"]["filename_template"]
    frames = []
    for source_path in config["source_metadata"]:
        source_path = str(source_path)
        frame = pd.read_csv(source_path)
        frame["source_metadata_file"] = source_path
        frame["source_row"] = range(len(frame))
        frame["source_output_base_path"] = frame["output_base_path"]
        frame["source_feature_path"] = frame.apply(lambda row: resolve_source_path(row, "feature_path"), axis=1)
        frame["source_hit3d_path"] = frame.apply(lambda row: resolve_source_path(row, "hit3d_path"), axis=1)
        frame["source_nueAnalysisFilter_path"] = frame.apply(
            lambda row: resolve_source_path(row, "nueAnalysisFilter_path"), axis=1
        )
        frame["output_base_path"] = frame.apply(
            lambda row: f"{feature_root}/{row['subfolder']}/{int(row['partition'])}", axis=1
        )
        frame["feature_path"] = frame.apply(
            lambda row: template.format(subfolder=row["subfolder"], partition=int(row["partition"])), axis=1
        )
        frame["hit3d_path"] = frame["source_hit3d_path"]
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    keys = ["subfolder", "partition"]
    if result.duplicated(keys).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicate source keys:\n{duplicates.to_string(index=False)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} source rows to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_metadata(args.config, args.output)


if __name__ == "__main__":
    main()

