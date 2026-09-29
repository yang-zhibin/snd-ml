"""Configuration helpers for the V7 PassingMuonReco composition study."""

from __future__ import annotations

from collections import OrderedDict
import re

from make_passing_muon_track_type_hist import validate_hist_config
from passing_muon_v7_reco import joined


CLASS_ORDER = (
    "no_valid_track",
    "track_like_low_activity",
    "track_plus_activity",
)


def _expand(value, prefix):
    return value.replace("{prefix}", prefix) if isinstance(value, str) else value


def configured_background_composition(config):
    block = config.get("background_composition")
    if not isinstance(block, dict) or not bool(block.get("enabled", False)):
        raise ValueError("V7 background_composition must be enabled")
    if str(block.get("scope", "")) != "passing_muon_reco_only":
        raise ValueError(
            "Only the evidence-backed passing_muon_reco_only scope is supported"
        )
    if str(block.get("required_schema", "")) != "flat_v2":
        raise ValueError("V7 background composition requires flat_v2")
    input_group = str(block.get("input_group", "")).strip()
    if not input_group:
        raise ValueError("background_composition.input_group must be non-empty")

    track = block.get("track_definition")
    if not isinstance(track, dict):
        raise ValueError("background_composition.track_definition must be a mapping")
    track_key = str(track.get("key", "")).strip()
    prefix = str(track.get("property_prefix", "")).strip()
    label = str(track.get("label", "")).strip()
    valid_expression = str(track.get("valid_expression", "")).strip()
    required_prefixes = [
        str(value).strip() for value in track.get("required_prefixes", [])
    ]
    if not re.fullmatch(r"[A-Za-z0-9_]+", track_key) or not re.fullmatch(
        r"[A-Za-z0-9_]+", prefix
    ) or not label or not valid_expression or not required_prefixes:
        raise ValueError(
            "Composition track definition requires safe key/property_prefix, "
            "label, valid_expression and required_prefixes"
        )
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", value) for value in required_prefixes):
        raise ValueError("Composition required track prefixes must be safe identifiers")
    if prefix not in required_prefixes or len(required_prefixes) != len(set(required_prefixes)):
        raise ValueError(
            "property_prefix must occur once in unique required_prefixes"
        )

    class_order = tuple(str(value) for value in block.get("class_order", []))
    if class_order != CLASS_ORDER:
        raise ValueError(
            f"background_composition.class_order must be {list(CLASS_ORDER)}"
        )
    raw_labels = block.get("class_labels")
    if not isinstance(raw_labels, dict) or set(raw_labels) != set(CLASS_ORDER):
        raise ValueError("background_composition.class_labels must label every class")
    class_labels = OrderedDict(
        (key, str(raw_labels[key]).strip()) for key in CLASS_ORDER
    )
    if any(not value for value in class_labels.values()):
        raise ValueError("Composition class labels must be non-empty")

    defaults = dict(block.get("defaults", {}))
    raw_features = block.get("features")
    if not isinstance(raw_features, dict) or not raw_features:
        raise ValueError("background_composition.features must be non-empty")
    features = OrderedDict()
    for name, raw in raw_features.items():
        feature = dict(defaults)
        feature.update({key: _expand(value, prefix) for key, value in raw.items()})
        feature["dimension"] = int(feature.get("dimension", 1))
        requires_track = bool(feature.pop("requires_valid_track", False))
        if requires_track:
            feature["feature_cut"] = joined([
                f"{prefix}_has_best == 1", feature.get("feature_cut", "")
            ])
        features[str(name)] = feature
    validate_hist_config({"features": features})

    raw_sections = block.get("plot_sections")
    if not isinstance(raw_sections, dict) or not raw_sections:
        raise ValueError("background_composition.plot_sections must be non-empty")
    sections = OrderedDict()
    assigned = []
    for key, raw in raw_sections.items():
        section_key = str(key)
        label = str(raw.get("label", "")).strip()
        names = [str(value) for value in raw.get("features", [])]
        if not re.fullmatch(r"[A-Za-z0-9_]+", section_key) or not label or not names:
            raise ValueError(f"Invalid composition plot section {section_key!r}")
        unknown = set(names) - set(features)
        if unknown or len(names) != len(set(names)):
            raise ValueError(
                f"Composition section {section_key!r} has unknown/repeated features: "
                f"{sorted(unknown)}"
            )
        assigned.extend(names)
        sections[section_key] = {"label": label, "features": names}
    if len(assigned) != len(set(assigned)) or set(assigned) != set(features):
        raise ValueError("Every composition feature must occur in exactly one section")

    return {
        "scope": "passing_muon_reco_only",
        "required_schema": "flat_v2",
        "input_group": input_group,
        "track_key": track_key,
        "track_prefix": prefix,
        "track_valid_expression": valid_expression,
        "required_track_prefixes": required_prefixes,
        "track_label": str(track["label"]),
        "class_order": class_order,
        "class_labels": class_labels,
        "features": features,
        "sections": sections,
    }


def class_expressions(selection_expression, composition):
    """Return exhaustive, mutually exclusive observable class expressions."""
    valid = str(composition["track_valid_expression"])
    return OrderedDict((
        ("no_valid_track", f"!({valid})"),
        (
            "track_like_low_activity",
            joined([valid, selection_expression]),
        ),
        (
            "track_plus_activity",
            joined([valid, f"!({selection_expression})"]),
        ),
    ))
