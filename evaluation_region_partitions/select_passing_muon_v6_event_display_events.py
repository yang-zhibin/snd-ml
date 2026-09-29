#!/usr/bin/env python3
"""Select auditable V6 loose-candidate and failure event-display manifests."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import OrderedDict, deque
from pathlib import Path

import pandas as pd
import ROOT
import yaml


ROOT.gROOT.SetBatch(True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))
from make_passing_muon_binary_region_hist_v5 import (  # noqa: E402
    configured_selections,
    define_shared_columns,
    validate_region_config,
)


IDENTITY_COLUMNS = ("runId", "eventId", "eventIndex")
MANIFEST_FIELDS = [
    "category", "sample", "sample_label", "population", "veto_category",
    "veto_category_label", "production_key", "source_file_index",
    "source_feature_path", "feature_entry", "event_source_path",
    "event_tree_name", "source_event_index", "geo_path", "runId", "eventId",
    "failed_components", "selection_expression", "loose_expression",
    "count_veto", "count_veto1", "count_veto2", "count_veto3",
    *(f"count_scifi{i}" for i in range(1, 6)),
    *(f"count_us{i}" for i in range(1, 6)),
    *(f"count_ds{i}" for i in range(1, 5)),
]
COUNT_FIELDS = [
    "category", "sample", "population", "veto_category", "sampled_events",
    "sources_with_events", "sources_scanned", "sources_total", "scan_complete",
    "minimum_sources", "manifest_events", "target_events", "reserve_events",
    "selection_expression",
]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve(path: object, repo_root: Path) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    if value.startswith("root://") or os.path.isabs(value):
        return value
    return str(repo_root / value)


def root_input(path: str) -> str:
    if path.startswith("/eos/") and not os.path.exists(path):
        return f"root://eospublic.cern.ch/{path}"
    return path


def full_metadata_path(row: pd.Series, column: str) -> str:
    value = str(row.get(column, "")).strip()
    if not value:
        return ""
    if value.startswith("root://") or os.path.isabs(value):
        return value
    base = str(row.get("output_base_path", "")).rstrip("/")
    return f"{base}/{value.lstrip('/')}" if base else value


def interleave_groups(rows: list[dict], key: str) -> list[dict]:
    groups: OrderedDict[str, deque] = OrderedDict()
    for row in rows:
        groups.setdefault(str(row.get(key, "")), deque()).append(row)
    result = []
    while any(groups.values()):
        for queue in groups.values():
            if queue:
                result.append(queue.popleft())
    return result


def load_real_sources(config: dict, repo_root: Path) -> list[dict]:
    path = resolve(config["input"]["real_data_metadata"], repo_root)
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "source_file_index", "feature_full_path", "digi_path", "geo_path",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Run-8329 metadata lacks columns: {sorted(missing)}")
    sources = []
    for row in frame.sort_values("source_file_index").to_dict("records"):
        sources.append({
            "sample": "run8329",
            "sample_label": "Run 8329 RealDataDQ",
            "production_key": "",
            "source_file_index": int(row["source_file_index"]),
            "source_feature_path": str(row["feature_full_path"]),
            "event_source_path": str(row["digi_path"]),
            "geo_path": str(row["geo_path"]),
        })
    return sources


def load_mc_sources(config: dict, study: dict, repo_root: Path) -> list[dict]:
    path = resolve(config["input"]["mc_metadata"], repo_root)
    frame = pd.read_csv(path, keep_default_na=False)
    mc = study["mc_muon_up"]
    partition_to_key = {
        str(definition["source_partition"]): key
        for key, definition in mc["productions"].items()
    }
    selected = frame.loc[
        frame["data_type"].astype(str).eq("MC_muon")
        & frame["subfolder"].astype(str).eq(str(mc["subfolder"]))
        & frame["partition"].astype(str).isin(partition_to_key)
    ].copy()
    expected = int(mc["expected_source_files"])
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} MC sources, found {len(selected)}")
    sources = []
    for source_index, row in selected.iterrows():
        production_key = partition_to_key[str(row["partition"])]
        sources.append({
            "sample": "mc_muon_up",
            "sample_label": "Pooled MC muon up",
            "production_key": production_key,
            "source_file_index": int(source_index),
            "source_feature_path": full_metadata_path(row, "feature_path"),
            "event_source_path": full_metadata_path(row, "digi_path"),
            "geo_path": full_metadata_path(row, "geo_path"),
        })
    return interleave_groups(sources, "production_key")


def category_definitions(config: dict, study: dict) -> OrderedDict[str, dict]:
    validate_region_config(study)
    working_point = str(config["selection"]["working_point"])
    selections = configured_selections(study)
    if working_point not in selections:
        raise ValueError(f"Unknown V6 working point {working_point!r}")
    loose = selections[working_point]
    definitions: OrderedDict[str, dict] = OrderedDict()
    for sample in ("run8329", "mc_muon_up"):
        baseline = "true" if sample == "run8329" else str(
            study["mc_muon_up"]["baseline_expression"]
        )
        candidate_key = f"{sample}__candidate"
        definitions[candidate_key] = {
            "sample": sample,
            "population": "candidate",
            "veto_category": "all",
            "veto_category_label": "loose candidate",
            "expression": f"({baseline}) && ({loose['expression']})",
        }
        for veto_key, veto in config["selection"]["failure_veto_categories"].items():
            category = f"{sample}__failure__{veto_key}"
            definitions[category] = {
                "sample": sample,
                "population": "failure",
                "veto_category": str(veto_key),
                "veto_category_label": str(veto["label"]),
                "expression": (
                    f"({baseline}) && !({loose['expression']}) && "
                    f"({veto['expression']})"
                ),
            }
    return definitions


def python_value(value):
    return value.item() if hasattr(value, "item") else value


def dataframe_for_source(source: dict, tree_name: str, study: dict):
    frame = ROOT.RDataFrame(tree_name, root_input(source["source_feature_path"]))
    columns = {str(value) for value in frame.GetColumnNames()}
    required = set(IDENTITY_COLUMNS) | {
        "count_veto", "count_veto1", "count_veto2", "count_veto3",
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
    }
    missing = required - columns
    if missing:
        raise RuntimeError(
            f"{source['source_feature_path']} lacks branches: {sorted(missing)}"
        )
    frame = define_shared_columns(frame, study["derived_columns"])
    frame = frame.Define("__v6_feature_entry", "(long long)rdfentry_")
    return frame


def sample_from_frame(
    frame, expression: str, limit: int, source: dict, definition: dict,
    category: str, loose: dict,
) -> list[dict]:
    if limit <= 0:
        return []
    selected = frame.Filter(expression).Range(limit)
    component_columns = []
    for index, (name, component_expression) in enumerate(loose["components"].items()):
        column = f"__v6_component_{index}"
        selected = selected.Define(column, str(component_expression))
        component_columns.append((name, column))
    read_columns = [
        "__v6_feature_entry", *IDENTITY_COLUMNS,
        "count_veto", "count_veto1", "count_veto2", "count_veto3",
        *(f"count_scifi{i}" for i in range(1, 6)),
        *(f"count_us{i}" for i in range(1, 6)),
        *(f"count_ds{i}" for i in range(1, 5)),
        *(column for _, column in component_columns),
    ]
    values = selected.AsNumpy(read_columns)
    size = len(values["__v6_feature_entry"])
    rows = []
    for index in range(size):
        failed = [
            name for name, column in component_columns
            if not bool(python_value(values[column][index]))
        ]
        row = {field: "" for field in MANIFEST_FIELDS}
        row.update({
            "category": category,
            "sample": source["sample"],
            "sample_label": source["sample_label"],
            "population": definition["population"],
            "veto_category": definition["veto_category"],
            "veto_category_label": definition["veto_category_label"],
            "production_key": source["production_key"],
            "source_file_index": source["source_file_index"],
            "source_feature_path": source["source_feature_path"],
            "feature_entry": int(python_value(values["__v6_feature_entry"][index])),
            "event_source_path": source["event_source_path"],
            "event_tree_name": "",
            "source_event_index": int(python_value(values["eventIndex"][index])),
            "geo_path": source["geo_path"],
            "runId": python_value(values["runId"][index]),
            "eventId": python_value(values["eventId"][index]),
            "failed_components": ";".join(failed),
            "selection_expression": expression,
            "loose_expression": loose["expression"],
        })
        for feature in MANIFEST_FIELDS:
            if feature.startswith(("count_veto", "count_scifi", "count_us", "count_ds")):
                row[feature] = int(python_value(values[feature][index]))
        rows.append(row)
    return rows


def round_robin_rows(source_pools: list[list[dict]], limit: int) -> list[dict]:
    """Take rows evenly across sources while preserving source and entry order."""
    queues = [deque(rows) for rows in source_pools if rows]
    selected = []
    while queues and len(selected) < limit:
        next_queues = []
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return selected


def atomic_csv(rows: list[dict], fields: list[str], output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-counts", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = load_yaml(resolve(args.config, repo_root))
    study = load_yaml(resolve(config["study_config"], repo_root))
    definitions = category_definitions(config, study)
    working_point = str(config["selection"]["working_point"])
    loose = configured_selections(study)[working_point]
    tree_name = str(config["input"].get("tree_name", "sndData"))
    target = int(config["sampling"]["events_per_category"])
    reserve = target * int(config["sampling"].get("reserve_factor", 1))
    configured_minimum_sources = int(config["sampling"].get("minimum_sources", 1))
    if target < 1 or reserve < target:
        raise ValueError("Sampling target must be positive and reserve must cover the target")
    if configured_minimum_sources < 1:
        raise ValueError("sampling.minimum_sources must be positive")
    if config["sampling"].get("mode", "round_robin_sources") != "round_robin_sources":
        raise ValueError("Only sampling.mode=round_robin_sources is supported")

    sources = {
        "run8329": load_real_sources(config, repo_root),
        "mc_muon_up": load_mc_sources(config, study, repo_root),
    }
    selected_rows = {category: [] for category in definitions}
    sampling_stats = {}

    for sample, sample_sources in sources.items():
        sample_categories = [
            category for category, definition in definitions.items()
            if definition["sample"] == sample
        ]
        source_pools = {category: [] for category in sample_categories}
        pool_sizes = {category: 0 for category in sample_categories}
        sources_with_events = {category: 0 for category in sample_categories}
        minimum_sources = min(configured_minimum_sources, len(sample_sources))
        sources_scanned = 0
        for source_number, source in enumerate(sample_sources, start=1):
            if (
                source_number > minimum_sources
                and all(pool_sizes[category] >= reserve for category in sample_categories)
            ):
                break
            if source_number <= minimum_sources:
                categories_to_sample = sample_categories
            else:
                categories_to_sample = [
                    category for category in sample_categories
                    if pool_sizes[category] < reserve
                ]
            print(
                f"[sample] {sample} source {source_number}/{len(sample_sources)}: "
                f"{source['source_feature_path']}"
            )
            frame = dataframe_for_source(source, tree_name, study)
            for category in categories_to_sample:
                # During the minimum-source phase, keep a full bounded pool
                # from every source. This permits round-robin diversity without
                # starving categories that occur in only a few source files.
                limit = reserve if source_number <= minimum_sources else (
                    reserve - pool_sizes[category]
                )
                rows = sample_from_frame(
                    frame, definitions[category]["expression"], limit, source,
                    definitions[category], category, loose,
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
                "sources_total": len(sample_sources),
                "scan_complete": sources_scanned == len(sample_sources),
                "minimum_sources": minimum_sources,
            }
        print(
            f"[sample] {sample}: stopped after {sources_scanned}/{len(sample_sources)} "
            f"sources; reserves="
            + ", ".join(
                f"{category}={len(selected_rows[category])}"
                for category in sample_categories
            )
        )

    manifest = []
    count_rows = []
    seen = set()
    for category, definition in definitions.items():
        rows = selected_rows[category]
        if len(rows) < reserve:
            raise RuntimeError(
                f"Category {category} has only {len(rows)} reserved events; "
                f"required reserve={reserve}, target={target}"
            )
        for row in rows:
            identity = (row["sample"], row["source_feature_path"], row["source_event_index"])
            if identity in seen:
                raise RuntimeError(f"Duplicate selected event identity: {identity}")
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
    print(f"Wrote {len(manifest)} reserved event rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
