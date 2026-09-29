#!/usr/bin/env python3
"""Configuration and population helpers for the Run-8329 V8 study."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import yaml

from make_passing_muon_binary_region_hist_v5 import (
    configured_selections,
    configured_v5_features,
    validate_region_config,
)


POPULATION_KEYS = (
    "passing_muon_candidate",
    "muon_dis_candidate",
    "signal_region",
)


def load_yaml(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_v8_config(path):
    path = Path(path)
    overlay = load_yaml(path)
    base_path = Path(str(overlay["base_config"]))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_yaml(base_path)
    validate_region_config(base)

    if int(overlay.get("run", -1)) != int(base.get("run", -2)):
        raise ValueError("V8 and base configuration runs do not match")
    selection = overlay.get("selection", {})
    base_key = str(selection.get("base_working_point", ""))
    base_selections = configured_selections(base)
    if base_key not in base_selections:
        raise ValueError(f"Unknown V8 base working point {base_key!r}")
    populations = overlay.get("populations", {})
    if tuple(populations) != POPULATION_KEYS:
        raise ValueError(
            "V8 populations must be ordered as " + ", ".join(POPULATION_KEYS)
        )
    for key in POPULATION_KEYS:
        if not str(populations[key].get("label", "")).strip():
            raise ValueError(f"V8 population {key!r} requires a label")
    for field in (
        "key", "label", "veto_present_expression", "signal_region_expression"
    ):
        if not str(selection.get(field, "")).strip():
            raise ValueError(f"V8 selection requires {field}")
    return overlay, base, base_selections[base_key]


def population_definitions(overlay, base_selection):
    selection = overlay["selection"]
    veto = str(selection["veto_present_expression"])
    signal = str(selection["signal_region_expression"])
    passing = str(base_selection["expression"])
    labels = overlay["populations"]
    return OrderedDict((
        (
            "passing_muon_candidate",
            {
                "label": str(labels["passing_muon_candidate"]["label"]),
                "expression": f"({veto}) && ({passing})",
            },
        ),
        (
            "muon_dis_candidate",
            {
                "label": str(labels["muon_dis_candidate"]["label"]),
                "expression": f"({veto}) && !({passing})",
            },
        ),
        (
            "signal_region",
            {
                "label": str(labels["signal_region"]["label"]),
                "expression": signal,
            },
        ),
    ))


def configured_features(hist_config, base_config):
    return configured_v5_features(hist_config, base_config)
