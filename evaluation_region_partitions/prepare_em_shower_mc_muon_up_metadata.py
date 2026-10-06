#!/usr/bin/env python3
"""Normalize MC-muon-up metadata for isolated EM-shower feature production."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


def resolve_path(row: pd.Series, column: str) -> str:
    value = str(row.get(column, "") or "").strip()
    if not value or value.lower() == "nan":
        return ""
    if value.startswith("root://") or Path(value).is_absolute():
        return value
    return f"{str(row['output_base_path']).rstrip('/')}/{value.lstrip('/')}"


def prepare(config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    source_path = str(config["source_metadata"])
    frame = pd.read_csv(source_path)
    selected = frame.loc[
        (frame["data_type"].astype(str).str.lower() == str(config["data_type"]).lower())
        & (frame["subfolder"].astype(str) == str(config["subfolder"]))
    ].copy()
    expected = int(config["expected_source_files"])
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} MC-muon-up rows, found {len(selected)}")

    production_map = {
        str(source): str(key) for key, source in config["productions"].items()
    }
    found = set(selected["partition"].astype(str))
    required = set(production_map)
    if found != required:
        raise ValueError(
            f"MC-muon-up production mismatch: expected {sorted(required)}, found {sorted(found)}"
        )

    feature_root = str(config["outputs"]["feature_root"]).rstrip("/")
    rows = []
    for source_row, row in selected.reset_index(drop=True).iterrows():
        digi_path = resolve_path(row, "digi_path")
        geo_path = resolve_path(row, "geo_path")
        if not digi_path or not geo_path:
            raise ValueError(f"Missing digi/geometry path for MC-muon row {source_row}")
        source_key = hashlib.sha256(digi_path.encode("utf-8")).hexdigest()[:12]
        production = production_map[str(row["partition"])]
        filename = f"feature_scifi_cluster_v1_MC_muon_up_{source_key}.root"
        feature_path = f"{feature_root}/{production}/{source_key}/{filename}"
        record = row.to_dict()
        record.update(
            {
                "source_metadata_file": source_path,
                "source_row": int(source_row),
                "source_key": source_key,
                "production": production,
                "digi_path": digi_path,
                "geo_path": geo_path,
                "input_tree": str(config.get("tree_name", "cbmsim")),
                "source_feature_path": resolve_path(row, "feature_path"),
                "output_base_path": str(Path(feature_path).parent),
                "feature_path": feature_path,
                "hit3d_path": "",
            }
        )
        rows.append(record)

    result = pd.DataFrame(rows).sort_values(["production", "source_key"]).reset_index(drop=True)
    for column in ("source_key", "digi_path", "feature_path"):
        if result[column].duplicated().any():
            raise ValueError(f"Duplicate MC-muon-up {column}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote {len(result)} MC-muon-up source rows to {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
