#!/usr/bin/env python3
"""Configuration helpers for V7 loose-failure SciFi-density distributions."""

from __future__ import annotations

import re
from collections import OrderedDict

from make_passing_muon_binary_region_hist_v5 import configured_selections
from passing_muon_v7_event_display_config import selection_parts


DENSITY_CATEGORY_PREFIX = "scifi_density_"


def joined_or(expressions) -> str:
    expressions = [str(expression).strip() for expression in expressions]
    if not expressions or any(not expression for expression in expressions):
        raise ValueError("Veto-union expressions must be non-empty")
    return " || ".join(f"({expression})" for expression in expressions)


def configured_density_categories(data_config: dict, mc_config: dict) -> OrderedDict:
    """Return the common, cumulative SciFi-density display categories."""
    data_failures, data_additional = selection_parts(data_config)
    mc_failures, mc_additional = selection_parts(mc_config)
    if data_config.get("selection") != mc_config.get("selection"):
        raise ValueError("Data and MC event-display selections must be identical")
    if data_config.get("samples") != ["run8329"]:
        raise ValueError("Density distributions require data samples: [run8329]")
    if mc_config.get("samples") != ["mc_muon_up"]:
        raise ValueError("Density distributions require MC samples: [mc_muon_up]")
    if data_failures != mc_failures or data_additional != mc_additional:
        raise ValueError("Data and MC event-display category definitions differ")

    result = OrderedDict()
    for key, definition in data_additional.items():
        key = str(key)
        if not key.startswith(DENSITY_CATEGORY_PREFIX):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]+", key):
            raise ValueError(f"Unsafe SciFi-density category name {key!r}")
        expression = str(definition.get("expression", "")).strip()
        label = str(definition.get("label", "")).strip()
        if not expression or not label:
            raise ValueError(f"SciFi-density category {key!r} lacks label/expression")
        if "density_sndsw_scifi" not in expression:
            raise ValueError(
                f"SciFi-density category {key!r} does not use density_sndsw_scifi"
            )
        result[key] = {"label": label, "expression": expression}
    if not result:
        raise ValueError("No scifi_density_* event-display categories are configured")
    return result


def configured_density_selection(
    study_config: dict, data_config: dict, mc_config: dict,
) -> dict:
    """Build the exact loose-failure selection shared by data and MC."""
    working_point = str(data_config["selection"]["working_point"])
    if str(mc_config["selection"]["working_point"]) != working_point:
        raise ValueError("Data and MC event displays use different working points")
    selections = configured_selections(study_config)
    if working_point not in selections:
        raise ValueError(f"Unknown density-distribution working point {working_point!r}")
    failures, _ = selection_parts(data_config)
    veto_union = joined_or(
        definition["expression"] for definition in failures.values()
    )
    return {
        "working_point": working_point,
        "coverage_expression": selections[working_point]["expression"],
        "veto_union_expression": veto_union,
        "veto_categories": OrderedDict(
            (str(key), str(definition["expression"]))
            for key, definition in failures.items()
        ),
    }


def category_expression(selection: dict, category: dict) -> str:
    """Select the union of Veto groups for one loose-failure density threshold."""
    return (
        f"!({selection['coverage_expression']}) && "
        f"({category['expression']}) && "
        f"({selection['veto_union_expression']})"
    )
