"""Pure configuration helpers for the Region V1 DS--SciFi rejection."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def joined(parts) -> str:
    values = [str(value).strip() for value in parts]
    if not values or any(not value for value in values):
        raise ValueError("Selection components must be non-empty")
    return " && ".join(f"({value})" for value in values)


def configured_veto_edge_selection(config: dict) -> dict:
    """Build the frozen Veto edge-muon selection and its diagnostics."""
    selection = config["selection"]
    edge = selection.get("edge", {})
    missing_minimum = float(edge.get("valid_position_min_exclusive", -900.0))
    boundaries = edge.get("boundaries_cm", {})
    required_planes = {"veto1_y", "veto2_y", "veto3_x"}
    if set(boundaries) != required_planes:
        raise ValueError(
            "Veto edge selection requires veto1_y, veto2_y, and veto3_x boundaries"
        )
    parsed = {}
    for plane, values in boundaries.items():
        low = float(values["low_edge_max"])
        high = float(values["high_edge_min"])
        if not low < high:
            raise ValueError(f"Invalid edge thresholds for {plane}: {low}, {high}")
        parsed[plane] = (low, high)

    valid = {
        "v1": (
            f"(count_veto1 > 0 && std::isfinite(avg_veto1_y) "
            f"&& avg_veto1_y > {missing_minimum:.12g})"
        ),
        "v2": (
            f"(count_veto2 > 0 && std::isfinite(avg_veto2_y) "
            f"&& avg_veto2_y > {missing_minimum:.12g})"
        ),
        "v3": (
            f"(count_veto3 > 0 && std::isfinite(avg_veto3_x) "
            f"&& avg_veto3_x > {missing_minimum:.12g})"
        ),
    }
    v1_low, v1_high = parsed["veto1_y"]
    v2_low, v2_high = parsed["veto2_y"]
    v3_low, v3_high = parsed["veto3_x"]
    edge_flags = {
        "v1_low": f"({valid['v1']} && avg_veto1_y <= {v1_low:.12g})",
        "v1_high": f"({valid['v1']} && avg_veto1_y >= {v1_high:.12g})",
        "v2_low": f"({valid['v2']} && avg_veto2_y <= {v2_low:.12g})",
        "v2_high": f"({valid['v2']} && avg_veto2_y >= {v2_high:.12g})",
    }
    any_y = f"({valid['v1']} || {valid['v2']})"
    any_valid = f"({any_y} || {valid['v3']})"
    y_edge = (
        f"(({valid['v1']} && {valid['v2']} && "
        f"(({edge_flags['v1_low']} && {edge_flags['v2_low']}) || "
        f"({edge_flags['v1_high']} && {edge_flags['v2_high']}))) || "
        f"({valid['v1']} && !{valid['v2']} && "
        f"({edge_flags['v1_low']} || {edge_flags['v1_high']})) || "
        f"(!{valid['v1']} && {valid['v2']} && "
        f"({edge_flags['v2_low']} || {edge_flags['v2_high']})) || "
        f"(!{valid['v1']} && !{valid['v2']}))"
    )
    x_edge = (
        f"(!{valid['v3']} || avg_veto3_x <= {v3_low:.12g} "
        f"|| avg_veto3_x >= {v3_high:.12g})"
    )
    tag_expression = joined([any_valid, y_edge, x_edge])
    keep_expression = f"!({tag_expression})"
    base_expression = str(selection.get("base_expression", "")).strip()
    if not base_expression:
        raise ValueError("selection.base_expression is required")
    categories = {
        "both_orientations": f"({any_y} && {valid['v3']})",
        "y_only": f"({any_y} && !{valid['v3']})",
        "x_only": f"(!{any_y} && {valid['v3']})",
        "no_position": f"(!{any_y} && !{valid['v3']})",
        "both_y_planes": f"({valid['v1']} && {valid['v2']})",
        "veto1_y_only": f"({valid['v1']} && !{valid['v2']})",
        "veto2_y_only": f"(!{valid['v1']} && {valid['v2']})",
    }
    payload = {
        "base_expression": base_expression,
        "valid_veto1_expression": valid["v1"],
        "valid_veto2_expression": valid["v2"],
        "valid_veto3_expression": valid["v3"],
        "any_y_valid_expression": any_y,
        "any_valid_expression": any_valid,
        "y_edge_expression": y_edge,
        "x_edge_expression": x_edge,
        "tag_expression": tag_expression,
        "keep_expression": keep_expression,
        "required_branches": [
            "density_sndsw_scifi",
            "count_veto1", "count_veto2", "count_veto3",
            "avg_veto1_y", "avg_veto2_y", "avg_veto3_x",
        ],
        "count_branches": ["count_veto1", "count_veto2", "count_veto3"],
        "diagnostic_categories": categories,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload | {"fingerprint": fingerprint}


def configured_veto_central_selection(config: dict) -> dict:
    """Build the Veto central-muon rejection requested for Region V1."""
    selection = config["selection"]
    central = selection.get("central", {})
    missing_minimum = float(
        central.get("valid_position_min_exclusive", -900.0)
    )
    intervals = central.get("intervals_cm", {})
    required_planes = {"veto1_y", "veto2_y", "veto3_x"}
    if set(intervals) != required_planes:
        raise ValueError(
            "Central selection requires veto1_y, veto2_y, and veto3_x intervals"
        )
    parsed = {}
    for plane, values in intervals.items():
        low = float(values["min_exclusive"])
        high = float(values["max_exclusive"])
        if not low < high:
            raise ValueError(f"Invalid central interval for {plane}: {low}, {high}")
        parsed[plane] = (low, high)

    valid = {
        "v1": (
            f"(count_veto1 > 0 && std::isfinite(avg_veto1_y) "
            f"&& avg_veto1_y > {missing_minimum:.12g})"
        ),
        "v2": (
            f"(count_veto2 > 0 && std::isfinite(avg_veto2_y) "
            f"&& avg_veto2_y > {missing_minimum:.12g})"
        ),
        "v3": (
            f"(count_veto3 > 0 && std::isfinite(avg_veto3_x) "
            f"&& avg_veto3_x > {missing_minimum:.12g})"
        ),
    }
    v1_low, v1_high = parsed["veto1_y"]
    v2_low, v2_high = parsed["veto2_y"]
    v3_low, v3_high = parsed["veto3_x"]
    v1_central = (
        f"({valid['v1']} && avg_veto1_y > {v1_low:.12g} "
        f"&& avg_veto1_y < {v1_high:.12g})"
    )
    v2_central = (
        f"({valid['v2']} && avg_veto2_y > {v2_low:.12g} "
        f"&& avg_veto2_y < {v2_high:.12g})"
    )
    x_central = (
        f"({valid['v3']} && avg_veto3_x > {v3_low:.12g} "
        f"&& avg_veto3_x < {v3_high:.12g})"
    )
    any_y = f"({valid['v1']} || {valid['v2']})"
    y_central = (
        f"(({valid['v1']} && {valid['v2']} && "
        f"{v1_central} && {v2_central}) || "
        f"({valid['v1']} && !{valid['v2']} && {v1_central}) || "
        f"(!{valid['v1']} && {valid['v2']} && {v2_central}))"
    )
    tag_expression = joined([y_central, x_central])
    complement_expression = f"!({tag_expression})"
    base_expression = str(selection.get("base_expression", "")).strip()
    if not base_expression:
        raise ValueError("selection.base_expression is required")
    categories = {
        "both_orientations": f"({any_y} && {valid['v3']})",
        "y_only": f"({any_y} && !{valid['v3']})",
        "x_only": f"(!{any_y} && {valid['v3']})",
        "no_position": f"(!{any_y} && !{valid['v3']})",
        "both_y_planes": f"({valid['v1']} && {valid['v2']})",
        "veto1_y_only": f"({valid['v1']} && !{valid['v2']})",
        "veto2_y_only": f"(!{valid['v1']} && {valid['v2']})",
    }
    payload = {
        "base_expression": base_expression,
        "valid_veto1_expression": valid["v1"],
        "valid_veto2_expression": valid["v2"],
        "valid_veto3_expression": valid["v3"],
        "any_y_valid_expression": any_y,
        "v1_central_expression": v1_central,
        "v2_central_expression": v2_central,
        "y_central_expression": y_central,
        "x_central_expression": x_central,
        "tag_expression": tag_expression,
        "keep_expression": complement_expression,
        "required_branches": [
            "density_sndsw_scifi",
            "count_veto1", "count_veto2", "count_veto3",
            "avg_veto1_y", "avg_veto2_y", "avg_veto3_x",
        ],
        "count_branches": ["count_veto1", "count_veto2", "count_veto3"],
        "diagnostic_categories": categories,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload | {"fingerprint": fingerprint}


def configured_selection(config: dict) -> dict:
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Configuration lacks selection mapping")
    mode = str(selection.get("mode", "ds_scifi"))
    if mode == "veto_edge_muon":
        return configured_veto_edge_selection(config)
    if mode == "veto_central_muon_rejection":
        return configured_veto_central_selection(config)
    if mode != "ds_scifi":
        raise ValueError(f"Unsupported selection mode: {mode}")
    scifi = selection.get("scifi", {})
    ds = selection.get("ds", {})
    active_xy = int(scifi.get("active_xy_stations_equal", 0))
    scifi_max = int(scifi.get("max_hits_per_station_exclusive", 0))
    ds_station_value = ds.get("max_hits_per_station_inclusive")
    ds_total_value = ds.get("max_hits_total_inclusive")
    ds_station_max = None if ds_station_value is None else int(ds_station_value)
    ds_total_max = None if ds_total_value is None else int(ds_total_value)
    if active_xy != 5:
        raise ValueError("V1 requires exactly five active SciFi XY stations")
    if scifi_max <= 0:
        raise ValueError("SciFi multiplicity threshold must be positive")
    if ds_station_max is not None and ds_station_max <= 0:
        raise ValueError("DS station multiplicity threshold must be positive")
    if ds_total_max is not None and ds_total_max <= 0:
        raise ValueError("DS total multiplicity threshold must be positive")
    if not bool(ds.get("require_all_stations", False)):
        raise ValueError("V1 DS selection must require all four stations")

    scifi_expression = joined([
        f"scifi_n_active_stations_xy == {active_xy}",
        *(
            f"count_scifi{station} < {scifi_max}"
            for station in range(1, 6)
        ),
    ])
    ds_components = [
        *(f"count_ds{station} > 0" for station in range(1, 5)),
    ]
    if ds_station_max is not None:
        ds_components.extend(
            f"count_ds{station} <= {ds_station_max}"
            for station in range(1, 5)
        )
    if ds_total_max is not None:
        ds_components.append(f"count_ds <= {ds_total_max}")
    ds_expression = joined(ds_components)
    tag_expression = joined([scifi_expression, ds_expression])
    keep_expression = f"!({tag_expression})"
    base_expression = str(selection.get("base_expression", "")).strip()
    if not base_expression:
        raise ValueError("selection.base_expression is required")

    forbidden = {"count_veto", "count_us", "reco", "pdgCode"}
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", tag_expression))
    offenders = sorted(
        identifier for identifier in identifiers
        if any(identifier == item or identifier.startswith(item) for item in forbidden)
    )
    if offenders:
        raise ValueError(f"Forbidden identifiers in DS--SciFi tag: {offenders}")

    required_branches = [
        "density_sndsw_scifi", "scifi_n_active_stations_xy", "count_scifi",
        *(f"count_scifi{station}" for station in range(1, 6)),
        "count_ds", *(f"count_ds{station}" for station in range(1, 5)),
    ]
    payload = {
        "base_expression": base_expression,
        "scifi_expression": scifi_expression,
        "ds_expression": ds_expression,
        "tag_expression": tag_expression,
        "keep_expression": keep_expression,
        "required_branches": required_branches,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload | {"fingerprint": fingerprint}


def configured_cutflow(config: dict) -> list[dict[str, str]]:
    """Return the ordered components of the cumulative tag cutflow."""
    selection = configured_selection(config)
    if (
        str(config["selection"].get("mode", "ds_scifi"))
        == "veto_central_muon_rejection"
    ):
        return [
            {
                "key": "base",
                "label": "Region V1 input population",
                "expression": selection["base_expression"],
            },
            {
                "key": "central_muon",
                "label": "central_muon = y_central AND x_central",
                "expression": selection["tag_expression"],
            },
        ]
    if str(config["selection"].get("mode", "ds_scifi")) == "veto_edge_muon":
        return [
            {
                "key": "base",
                "label": "SciFi count > 35",
                "expression": selection["base_expression"],
            },
            {
                "key": "any_veto_position",
                "label": "At least one valid Veto position",
                "expression": selection["any_valid_expression"],
            },
            {
                "key": "y_edge",
                "label": "Available y information is edge-compatible",
                "expression": selection["y_edge_expression"],
            },
            {
                "key": "x_edge",
                "label": "Available x information is edge-compatible",
                "expression": selection["x_edge_expression"],
            },
        ]
    scifi = config["selection"]["scifi"]
    ds = config["selection"]["ds"]
    active_xy = int(scifi["active_xy_stations_equal"])
    scifi_max = int(scifi["max_hits_per_station_exclusive"])
    ds_station_value = ds.get("max_hits_per_station_inclusive")
    ds_total_value = ds.get("max_hits_total_inclusive")
    stages = [
        {
            "key": "base",
            "label": "SciFi count > 35",
            "expression": selection["base_expression"],
        },
        {
            "key": "scifi_xy",
            "label": f"All {active_xy} SciFi stations have X and Y activity",
            "expression": f"scifi_n_active_stations_xy == {active_xy}",
        },
        {
            "key": "scifi_station_hits",
            "label": f"Every SciFi station has < {scifi_max} hits",
            "expression": joined(
                f"count_scifi{station} < {scifi_max}"
                for station in range(1, 6)
            ),
        },
        {
            "key": "ds_all_active",
            "label": "All 4 DS stations are active",
            "expression": joined(
                f"count_ds{station} > 0" for station in range(1, 5)
            ),
        },
    ]
    if ds_station_value is not None:
        ds_station_max = int(ds_station_value)
        stages.append({
            "key": "ds_station_hits",
            "label": f"Every DS station has <= {ds_station_max} hits",
            "expression": joined(
                f"count_ds{station} <= {ds_station_max}"
                for station in range(1, 5)
            ),
        })
    if ds_total_value is not None:
        ds_total_max = int(ds_total_value)
        stages.append({
            "key": "ds_total_hits",
            "label": f"Total DS hits <= {ds_total_max}",
            "expression": f"count_ds <= {ds_total_max}",
        })
    return stages


def cutflow_field(stage: dict[str, str]) -> str:
    return f"cutflow_{stage['key']}"


def expand_macros(expression: str, macros: dict) -> str:
    expanded = str(expression)
    while True:
        updated = expanded
        for name, value in macros.items():
            updated = updated.replace(str(name), f"({value})")
        if updated == expanded:
            return expanded
        expanded = updated


def configured_region_expressions(config: dict, region_config: dict) -> dict[str, str]:
    configured = list(config.get("regions", {}))
    available = region_config.get("regions", {})
    missing = sorted(set(configured) - set(available))
    if missing:
        raise KeyError(f"Missing Region V1 definitions: {missing}")
    macros = region_config.get("expression_macros", {})
    return {
        region: expand_macros(available[region]["expression"], macros).strip()
        for region in configured
    }
