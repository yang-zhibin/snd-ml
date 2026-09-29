"""Shared configuration helpers for Run-8329 flat-v2 track diagnostics."""

from __future__ import annotations

from collections import OrderedDict
import re

from make_passing_muon_track_type_hist import validate_hist_config


def joined(parts):
    return " && ".join(f"({part})" for part in parts if str(part).strip())


def _formatted_feature(definition, prefix, label):
    result = {}
    for key, value in definition.items():
        if isinstance(value, str):
            # ROOT axis titles legitimately contain braces (for example
            # ``#chi^{2}``).  Expand only the two tokens supported by this
            # configuration layer instead of treating every brace as Python
            # format syntax.
            result[key] = value.replace("{prefix}", prefix).replace(
                "{label}", label
            )
        else:
            result[key] = value
    result["dimension"] = int(result.get("dimension", 1))
    return result


def _configured_features(raw_features, defaults, context):
    if not isinstance(raw_features, dict) or not raw_features:
        raise ValueError(f"{context} must be a non-empty feature mapping")
    result = OrderedDict()
    for name, definition in raw_features.items():
        feature = dict(defaults)
        feature.update(definition)
        feature["dimension"] = int(feature.get("dimension", 1))
        result[str(name)] = feature
    validate_hist_config({"features": result})
    return result


def configured_reco_diagnostics(config):
    """Return validated event and per-track-type feature definitions."""
    block = config.get("reco_track_diagnostics")
    if not isinstance(block, dict):
        raise ValueError("V7 config requires reco_track_diagnostics")
    if str(block.get("required_schema", "")) != "flat_v2":
        raise ValueError("V7 reconstructed-track diagnostics require flat_v2")

    track_types = block.get("track_types")
    if not isinstance(track_types, dict) or not track_types:
        raise ValueError("reco_track_diagnostics.track_types must be non-empty")
    prefixes = set()
    for key, definition in track_types.items():
        prefix = str(definition.get("prefix", "")).strip()
        label = str(definition.get("label", "")).strip()
        if not prefix or not label:
            raise ValueError(f"Track type {key!r} requires prefix and label")
        if prefix in prefixes:
            raise ValueError(f"Duplicate reconstructed-track prefix {prefix!r}")
        prefixes.add(prefix)

    defaults = dict(block.get("defaults", {}))
    raw_event_features = block.get("event_features", {})
    raw_track_features = block.get("track_features", {})
    if not raw_event_features or not raw_track_features:
        raise ValueError("V7 requires event_features and track_features")

    event_features = _configured_features(
        raw_event_features, defaults, "reco_track_diagnostics.event_features"
    )

    track_features = OrderedDict()
    for track_key, track in track_types.items():
        prefix = str(track["prefix"])
        label = str(track["label"])
        expanded = OrderedDict()
        for name, definition in raw_track_features.items():
            feature = dict(defaults)
            feature.update(_formatted_feature(definition, prefix, label))
            scope = str(feature.pop("scope", "best"))
            if scope != "best":
                raise ValueError(
                    f"Track feature {name!r} has unsupported scope {scope!r}"
                )
            feature["feature_cut"] = joined(
                [f"{prefix}_has_best == 1", feature.get("feature_cut", "")]
            )
            expanded[str(name)] = feature
        validate_hist_config({"features": expanded})
        track_features[str(track_key)] = expanded

    raw_plot_groups = block.get("plot_groups")
    if not isinstance(raw_plot_groups, dict) or not raw_plot_groups:
        raise ValueError("reco_track_diagnostics.plot_groups must be non-empty")
    plot_groups = OrderedDict()
    assigned_track_types = set()
    for group_key, raw_group in raw_plot_groups.items():
        group_key = str(group_key)
        if not isinstance(raw_group, dict):
            raise ValueError(f"Plot group {group_key!r} must be a mapping")
        if not re.fullmatch(r"[A-Za-z0-9_]+", group_key):
            raise ValueError(f"Unsafe reconstructed-track plot group {group_key!r}")
        label = str(raw_group.get("label", "")).strip()
        raw_sections = raw_group.get("sections")
        if not label or not isinstance(raw_sections, dict) or not raw_sections:
            raise ValueError(f"Plot group {group_key!r} requires label and sections")
        sections = OrderedDict()
        group_track_types = []
        for section_key, raw_section in raw_sections.items():
            section_key = str(section_key)
            section_label = str(raw_section.get("label", "")).strip()
            section_tracks = [
                str(value) for value in raw_section.get("track_types", [])
            ]
            if not section_label or not section_tracks:
                raise ValueError(
                    f"Plot section {group_key}.{section_key} requires label and track_types"
                )
            unknown = set(section_tracks) - set(track_types)
            if unknown:
                raise ValueError(
                    f"Plot section {group_key}.{section_key} references unknown "
                    f"track types: {sorted(unknown)}"
                )
            duplicates = set(section_tracks) & set(group_track_types)
            if duplicates:
                raise ValueError(
                    f"Plot group {group_key!r} repeats track types: {sorted(duplicates)}"
                )
            group_track_types.extend(section_tracks)
            sections[section_key] = {
                "label": section_label,
                "track_types": section_tracks,
            }
        overlap = assigned_track_types & set(group_track_types)
        if overlap:
            raise ValueError(
                f"Track types assigned to more than one plot group: {sorted(overlap)}"
            )
        assigned_track_types.update(group_track_types)
        plot_groups[group_key] = {
            "label": label,
            "sections": sections,
            "track_types": group_track_types,
            "event_features": _configured_features(
                raw_group.get("event_features"),
                defaults,
                f"reco_track_diagnostics.plot_groups.{group_key}.event_features",
            ),
        }
    if assigned_track_types != set(track_types):
        raise ValueError(
            "Every reconstructed-track type must appear in exactly one plot group; "
            f"unassigned={sorted(set(track_types) - assigned_track_types)}"
        )

    input_group = str(block.get("input_group", "")).strip()
    if not input_group:
        raise ValueError("reco_track_diagnostics.input_group must be non-empty")
    return {
        "required_schema": "flat_v2",
        "input_group": input_group,
        "track_types": OrderedDict(
            (str(key), dict(value)) for key, value in track_types.items()
        ),
        "event_features": event_features,
        "track_features": track_features,
        "plot_groups": plot_groups,
    }


def required_reco_branches(diagnostics):
    """Return explicit scalar branches needed before RDataFrame expressions run."""
    required = {
        "reco_track_n_all",
        "reco_track_n_converged",
        "reco_track_n_types_present",
    }
    best_suffixes = {
        "n_all",
        "n_converged",
        "has_any",
        "has_converged",
        "has_best",
        "best_start_x",
        "best_start_y",
        "best_start_z",
        "best_stop_x",
        "best_stop_y",
        "best_stop_z",
        "best_chi2ndf",
        "best_ndf",
        "best_slope_xz",
        "best_slope_yz",
        "best_n_measurements",
        "best_n_unique_planes",
    }
    for track in diagnostics["track_types"].values():
        prefix = str(track["prefix"])
        required.update(f"{prefix}_{suffix}" for suffix in best_suffixes)
    return required
