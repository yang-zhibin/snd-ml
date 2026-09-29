"""Shared, ROOT-free utilities for the data-driven 3D-CNN dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_repo_path(value: str, repo_root: str | Path) -> str:
    text = str(value)
    if text.startswith("root://") or os.path.isabs(text):
        return text
    return str(Path(repo_root) / text)


def resolve_product_path(row: pd.Series, column: str) -> str:
    value = str(row[column])
    if value.startswith("root://") or os.path.isabs(value):
        return value
    return str(Path(str(row["output_base_path"])) / value)


def stable_uid(*parts: object, prefix: str = "src") -> str:
    canonical = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def configured_density_bins(config: dict) -> list[dict]:
    density = config.get("density_sampling", {})
    raw_bins = density.get("bins")
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError("density_sampling.bins must be a non-empty list")
    result = []
    previous_high = None
    for index, raw in enumerate(raw_bins):
        if not isinstance(raw, dict):
            raise ValueError(f"Density bin {index} must be a mapping")
        low = float(raw["low"])
        high_raw = raw.get("high")
        high = math.inf if high_raw is None else float(high_raw)
        if not math.isfinite(low) or high <= low:
            raise ValueError(f"Invalid density bin {index}: [{low}, {high})")
        if previous_high is not None and low != previous_high:
            raise ValueError("Density bins must be contiguous and non-overlapping")
        result.append({
            "index": index,
            "name": str(raw.get("name", f"bin_{index}")),
            "low": low,
            "high": high,
        })
        previous_high = high
    if not math.isinf(result[-1]["high"]):
        raise ValueError("The final density bin must have high: null")
    return result


def density_bin_index(value: object, bins: list[dict]) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    if not math.isfinite(number) or number < bins[0]["low"]:
        return -1
    for item in bins:
        if item["low"] <= number < item["high"]:
            return int(item["index"])
    return -1
