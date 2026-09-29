#!/usr/bin/env python3
"""Select auditable V7 data or muon-MC event-display reserve manifests."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from make_passing_muon_binary_region_hist_v5 import (
    configured_selections,
    validate_region_config,
)
from passing_muon_v7_event_display_config import (
    configured_samples,
    inclusive_selection_parts,
    selection_parts,
    validate_sampling,
)
from select_passing_muon_v6_event_display_events import (
    COUNT_FIELDS,
    MANIFEST_FIELDS,
    atomic_csv,
    dataframe_for_source,
    load_mc_sources,
    load_real_sources,
    load_yaml,
    resolve,
    round_robin_rows,
    sample_from_frame,
)


def category_definitions(config: dict, study: dict) -> OrderedDict[str, dict]:
    """Return configured data or MC coverage display categories."""
    validate_region_config(study)
    samples = configured_samples(config)
    working_point = str(config["selection"]["working_point"])
    selections = configured_selections(study)
    if working_point not in selections:
        raise ValueError(f"Unknown V7 event-display working point {working_point!r}")
    selection = selections[working_point]
    selection_config = config.get("selection", {})
    include_standard = bool(
        selection_config.get("include_standard_categories", True)
    )
    failures, additional = (
        selection_parts(config) if include_standard else ({}, {})
    )
    inclusive_veto, inclusive_criteria = inclusive_selection_parts(config)

    definitions: OrderedDict[str, dict] = OrderedDict()
    for sample in samples:
        baseline = "true" if sample == "run8329" else str(
            study["mc_muon_up"]["baseline_expression"]
        )
        if include_standard:
            definitions[f"{sample}__candidate"] = {
                "sample": sample,
                "population": "candidate",
                "veto_category": "all",
                "veto_category_label": f"{working_point} candidate",
                "expression": f"({baseline}) && ({selection['expression']})",
            }
            for veto_key, veto in failures.items():
                definitions[f"{sample}__failure__{veto_key}"] = {
                    "sample": sample,
                    "population": "failure",
                    "veto_category": str(veto_key),
                    "veto_category_label": str(veto["label"]),
                    "expression": (
                        f"({baseline}) && !({selection['expression']}) && "
                        f"({veto['expression']})"
                    ),
                }
        for criterion_key, criterion in additional.items():
            for veto_key, veto in failures.items():
                definitions[
                    f"{sample}__failure__{veto_key}__{criterion_key}"
                ] = {
                    "sample": sample,
                    "population": "failure",
                    "veto_category": str(veto_key),
                    "veto_category_label": (
                        f"{veto['label']}, {criterion['label']}"
                    ),
                    "expression": (
                        f"({baseline}) && !({selection['expression']}) && "
                        f"({veto['expression']}) && ({criterion['expression']})"
                    ),
                }
        for criterion_key, criterion in inclusive_criteria.items():
            for veto_key, veto in inclusive_veto.items():
                definitions[
                    f"{sample}__inclusive__{veto_key}__{criterion_key}"
                ] = {
                    "sample": sample,
                    "population": "inclusive",
                    "veto_category": str(veto_key),
                    "veto_category_label": (
                        f"{veto['label']}, {criterion['label']}"
                    ),
                    # Deliberately omit both selection['expression'] and its
                    # negation: membership is independent of the working point.
                    "expression": (
                        f"({baseline}) && ({veto['expression']}) && "
                        f"({criterion['expression']})"
                    ),
                }
    return definitions


def sources_for_sample(
    sample: str, config: dict, study: dict, repo_root: Path,
) -> list[dict]:
    if sample == "run8329":
        return load_real_sources(config, repo_root)
    if sample == "mc_muon_up":
        return load_mc_sources(config, study, repo_root)
    raise ValueError(f"Unsupported V7 event-display sample {sample!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-counts", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = load_yaml(resolve(args.config, repo_root))
    study = load_yaml(resolve(config["study_config"], repo_root))
    definitions = category_definitions(config, study)
    target, reserve, configured_minimum_sources = validate_sampling(config)
    working_point = str(config["selection"]["working_point"])
    selection = configured_selections(study)[working_point]
    tree_name = str(config["input"].get("tree_name", "sndData"))
    selected_rows = {}
    sampling_stats = {}
    for sample in configured_samples(config):
        sample_categories = [
            category for category, definition in definitions.items()
            if definition["sample"] == sample
        ]
        sources = sources_for_sample(sample, config, study, repo_root)
        minimum_sources = min(configured_minimum_sources, len(sources))
        source_pools = {category: [] for category in sample_categories}
        pool_sizes = {category: 0 for category in sample_categories}
        sources_with_events = {category: 0 for category in sample_categories}
        sources_scanned = 0
        for source_number, source in enumerate(sources, start=1):
            if (
                source_number > minimum_sources
                and all(
                    pool_sizes[category] >= reserve
                    for category in sample_categories
                )
            ):
                break
            categories_to_sample = (
                sample_categories
                if source_number <= minimum_sources
                else [
                    category for category in sample_categories
                    if pool_sizes[category] < reserve
                ]
            )
            print(
                f"[sample] {sample} source {source_number}/{len(sources)}: "
                f"{source['source_feature_path']}"
            )
            frame = dataframe_for_source(source, tree_name, study)
            for category in categories_to_sample:
                limit = reserve if source_number <= minimum_sources else (
                    reserve - pool_sizes[category]
                )
                rows = sample_from_frame(
                    frame,
                    definitions[category]["expression"],
                    limit,
                    source,
                    definitions[category],
                    category,
                    selection,
                )
                if rows:
                    source_pools[category].append(rows)
                    pool_sizes[category] += len(rows)
                    sources_with_events[category] += 1
            sources_scanned = source_number
        for category in sample_categories:
            selected_rows[category] = round_robin_rows(
                source_pools[category], reserve
            )
            sampling_stats[category] = {
                "sampled_events": pool_sizes[category],
                "sources_with_events": sources_with_events[category],
                "sources_scanned": sources_scanned,
                "sources_total": len(sources),
                "scan_complete": sources_scanned == len(sources),
                "minimum_sources": minimum_sources,
            }

    manifest = []
    count_rows = []
    seen = set()
    for category, definition in definitions.items():
        rows = selected_rows[category]
        if len(rows) < reserve:
            print(
                f"[warn] Category {category} has only {len(rows)} reserve events; "
                f"requested={reserve} after scanning "
                f"{sampling_stats[category]['sources_scanned']}/"
                f"{sampling_stats[category]['sources_total']} sources"
            )
        for row in rows:
            identity = (
                category,
                row["sample"],
                row["source_feature_path"],
                row["source_event_index"],
            )
            if identity in seen:
                raise RuntimeError(
                    f"Duplicate selected event identity within category: {identity}"
                )
            seen.add(identity)
            manifest.append(row)
        count_rows.append({
            "category": category,
            "sample": definition["sample"],
            "population": definition["population"],
            "veto_category": definition["veto_category"],
            **sampling_stats[category],
            "manifest_events": len(rows),
            "target_events": target,
            "reserve_events": reserve,
            "selection_expression": definition["expression"],
        })

    atomic_csv(manifest, MANIFEST_FIELDS, args.output_manifest)
    atomic_csv(count_rows, COUNT_FIELDS, args.output_counts)
    print(
        f"Wrote {len(manifest)} reserve rows across "
        f"{len(definitions)} categories to {args.output_manifest}"
    )


if __name__ == "__main__":
    main()
