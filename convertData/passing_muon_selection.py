"""Pure helpers for the detector-only passing-muon working points.

This module deliberately has no ROOT dependency.  Dataset production and the
V7 diagnostic workflow can therefore construct the same selection expression
from the same YAML configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def joined(parts) -> str:
    values = [str(part).strip() for part in parts]
    if not values or any(not value for value in values):
        raise ValueError("Selection components must be non-empty")
    return " && ".join(f"({value})" for value in values)


def coverage_components(working_point: dict) -> OrderedDict:
    """Build the ordered detector-coverage components used by V7."""
    required = {"scifi_min_active_xy_stations", "ds_min_active_stations"}
    missing = required - set(working_point)
    if missing:
        raise ValueError(f"Coverage working point lacks {sorted(missing)}")

    components = OrderedDict((
        ("all_veto_stations", "has_all_veto"),
        (
            "scifi_xy_station_coverage",
            "scifi_n_active_stations_xy >= "
            f"{int(working_point['scifi_min_active_xy_stations'])}",
        ),
        ("all_us_stations", "has_all_us"),
        (
            "ds_station_coverage",
            f"n_active_ds >= {int(working_point['ds_min_active_stations'])}",
        ),
    ))

    for component, detector, station_count in (
        ("scifi_station_multiplicity", "scifi", 5),
        ("veto_station_multiplicity", "veto", 3),
        ("us_station_multiplicity", "us", 5),
    ):
        exclusive_key = f"{detector}_max_hits_per_station_exclusive"
        inclusive_key = f"{detector}_max_hits_per_station_inclusive"
        exclusive = working_point.get(exclusive_key)
        inclusive = working_point.get(inclusive_key)
        if exclusive is not None and inclusive is not None:
            raise ValueError(
                f"Configure only one of {exclusive_key} and {inclusive_key}"
            )
        if exclusive is not None:
            components[component] = joined(
                f"count_{detector}{station} < {int(exclusive)}"
                for station in range(1, station_count + 1)
            )
        elif inclusive is not None:
            components[component] = joined(
                f"count_{detector}{station} <= {int(inclusive)}"
                for station in range(1, station_count + 1)
            )

        total = working_point.get(f"{detector}_max_hits_total_inclusive")
        if total is not None:
            components[f"{detector}_total_multiplicity"] = (
                f"count_{detector} <= {int(total)}"
            )
    return components


def configured_coverage_selection(config: dict, working_point: str) -> dict:
    points = config.get("coverage_working_points")
    if not isinstance(points, dict) or working_point not in points:
        raise ValueError(f"Unknown coverage working point {working_point!r}")
    components = coverage_components(points[working_point])
    expression = joined(components.values())
    payload = {
        "working_point": str(working_point),
        "definition": points[working_point],
        "derived_columns": config.get("derived_columns", {}),
        "components": list(components.items()),
        "expression": expression,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "name": str(working_point),
        "label": str(points[working_point].get("label", working_point)),
        "components": components,
        "derived_columns": OrderedDict(
            (str(key), str(value))
            for key, value in config.get("derived_columns", {}).items()
        ),
        "expression": expression,
        "fingerprint": fingerprint,
    }


def load_configured_coverage_selection(
    config_path: str | Path, working_point: str,
) -> dict:
    return configured_coverage_selection(load_yaml(config_path), working_point)
