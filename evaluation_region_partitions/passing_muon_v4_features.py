#!/usr/bin/env python3
"""Expand V4-only station-position templates into generic 2D features."""

from __future__ import annotations

from copy import deepcopy


EXPECTED_DETECTORS = ("scifi", "ds")
REQUIRED_FIELDS = {
    "stations", "feature_name", "x_expression", "y_expression",
    "x_bins", "x_min", "x_max", "y_bins", "y_min", "y_max",
    "x_title", "y_title",
}


def _format(template: object, station: int, field: str, detector: str) -> str:
    try:
        value = str(template).format(station=station)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Invalid {detector}.{field} station template: {template!r}"
        ) from error
    if not value.strip():
        raise ValueError(f"Empty {detector}.{field} for station {station}")
    return value


def station_position_features(region_config: dict) -> dict[str, dict]:
    """Return validated per-station x-y feature definitions in config order."""
    block = region_config.get("station_average_position_2d")
    if not isinstance(block, dict):
        raise ValueError("V4 config requires station_average_position_2d")
    detectors = block.get("detectors")
    if not isinstance(detectors, dict) or tuple(detectors) != EXPECTED_DETECTORS:
        raise ValueError(
            "station_average_position_2d.detectors must be ordered as scifi, ds"
        )

    result = {}
    for detector, definition in detectors.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Station-position detector {detector!r} must be a mapping")
        missing = REQUIRED_FIELDS - set(definition)
        if missing:
            raise ValueError(
                f"Station-position detector {detector!r} lacks: {sorted(missing)}"
            )
        stations = definition["stations"]
        if not isinstance(stations, list) or not stations:
            raise ValueError(f"{detector}.stations must be a nonempty list")
        stations = [int(station) for station in stations]
        if any(station < 1 for station in stations) or len(stations) != len(set(stations)):
            raise ValueError(f"{detector}.stations must contain unique positive integers")
        x_bins, y_bins = int(definition["x_bins"]), int(definition["y_bins"])
        x_min, x_max = float(definition["x_min"]), float(definition["x_max"])
        y_min, y_max = float(definition["y_min"]), float(definition["y_max"])
        if x_bins < 1 or y_bins < 1 or x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid station-position binning for {detector}")

        for station in stations:
            name = _format(definition["feature_name"], station, "feature_name", detector)
            if name in result:
                raise ValueError(f"Duplicate station-position feature {name!r}")
            x_expression = _format(
                definition["x_expression"], station, "x_expression", detector
            )
            y_expression = _format(
                definition["y_expression"], station, "y_expression", detector
            )
            result[name] = {
                "dimension": 2,
                "x_expression": x_expression,
                "y_expression": y_expression,
                "feature_cut": (
                    f"std::isfinite({x_expression}) && "
                    f"std::isfinite({y_expression}) && "
                    f"{x_expression} > -900 && {y_expression} > -900"
                ),
                "x_bins": x_bins,
                "x_min": x_min,
                "x_max": x_max,
                "y_bins": y_bins,
                "y_min": y_min,
                "y_max": y_max,
                "x_title": _format(
                    definition["x_title"], station, "x_title", detector
                ),
                "y_title": _format(
                    definition["y_title"], station, "y_title", detector
                ),
                "fold_x_underflow": bool(
                    definition.get("fold_x_underflow", False)
                ),
                "fold_x_overflow": bool(
                    definition.get("fold_x_overflow", False)
                ),
                "fold_y_underflow": bool(
                    definition.get("fold_y_underflow", False)
                ),
                "fold_y_overflow": bool(
                    definition.get("fold_y_overflow", False)
                ),
                "logz": bool(definition.get("logz", True)),
            }
    return result


def expanded_v4_hist_config(hist_config: dict, region_config: dict) -> dict:
    """Copy the shared histogram config and append only V4 station features."""
    result = deepcopy(hist_config)
    features = result.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("Histogram config requires a nonempty features mapping")
    additions = station_position_features(region_config)
    collisions = set(features) & set(additions)
    if collisions:
        raise ValueError(
            f"V4 station-position features collide with shared features: "
            f"{sorted(collisions)}"
        )
    features.update(additions)
    return result
