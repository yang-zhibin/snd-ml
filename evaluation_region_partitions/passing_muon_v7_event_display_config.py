#!/usr/bin/env python3
"""Lightweight configuration helpers for V7 data and MC event displays."""

from __future__ import annotations


SUPPORTED_SAMPLES = ("run8329", "mc_muon_up")


def configured_samples(config: dict) -> list[str]:
    samples = [str(value) for value in config.get("samples", [])]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("V7 event-display samples must be non-empty and unique")
    unsupported = set(samples) - set(SUPPORTED_SAMPLES)
    if unsupported:
        raise ValueError(
            f"Unsupported V7 event-display samples: {sorted(unsupported)}"
        )
    return samples


def selection_parts(config: dict) -> tuple[dict, dict]:
    selection = config.get("selection", {})
    failures = selection.get("failure_veto_categories")
    if not isinstance(failures, dict) or not failures:
        raise ValueError("V7 event display requires failure_veto_categories")
    additional = selection.get("additional_failure_criteria", {})
    if not isinstance(additional, dict):
        raise ValueError("selection.additional_failure_criteria must be a mapping")
    return failures, additional


def inclusive_selection_parts(config: dict) -> tuple[dict, dict]:
    """Return veto and criterion mappings that do not apply the working point."""
    selection = config.get("selection", {})
    veto_categories = selection.get("inclusive_veto_categories", {})
    criteria = selection.get("inclusive_criteria", {})
    if not isinstance(veto_categories, dict):
        raise ValueError("selection.inclusive_veto_categories must be a mapping")
    if not isinstance(criteria, dict):
        raise ValueError("selection.inclusive_criteria must be a mapping")
    if bool(veto_categories) != bool(criteria):
        raise ValueError(
            "inclusive_veto_categories and inclusive_criteria must either both "
            "be populated or both be empty"
        )
    return veto_categories, criteria


def category_names(config: dict) -> list[str]:
    selection = config.get("selection", {})
    include_standard = bool(selection.get("include_standard_categories", True))
    failures, additional = selection_parts(config) if include_standard else ({}, {})
    inclusive_veto, inclusive_criteria = inclusive_selection_parts(config)
    categories = []
    for sample in configured_samples(config):
        if include_standard:
            categories.extend([
                f"{sample}__candidate",
                *(f"{sample}__failure__{veto}" for veto in failures),
                *(
                    f"{sample}__failure__{veto}__{criterion}"
                    for criterion in additional
                    for veto in failures
                ),
            ])
        categories.extend(
            f"{sample}__inclusive__{veto}__{criterion}"
            for criterion in inclusive_criteria
            for veto in inclusive_veto
        )
    if not categories:
        raise ValueError("V7 event display configuration defines no categories")
    return categories


def validate_sampling(config: dict) -> tuple[int, int, int]:
    sampling = config["sampling"]
    target = int(sampling["events_per_category"])
    reserve = target * int(sampling.get("reserve_factor", 1))
    minimum_sources = int(sampling.get("minimum_sources", 1))
    if target < 1 or reserve < target:
        raise ValueError("Sampling target must be positive and reserve must cover it")
    if minimum_sources < 1:
        raise ValueError("sampling.minimum_sources must be positive")
    if sampling.get("mode", "round_robin_sources") != "round_robin_sources":
        raise ValueError("Only sampling.mode=round_robin_sources is supported")
    return target, reserve, minimum_sources
