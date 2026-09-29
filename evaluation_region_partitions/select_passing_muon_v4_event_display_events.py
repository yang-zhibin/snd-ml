#!/usr/bin/env python3
"""Select bounded, auditable V4 veto-time event-display manifests."""

from __future__ import annotations

import argparse
import csv
import os
import re
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
from make_passing_muon_region_hist_v4 import validate_region_config  # noqa: E402


POPULATIONS = OrderedDict([
    ("passing", {"label": "PassingMuonReco", "metadata": "partition_metadata"}),
    ("missing", {"label": "NoPassingMuonReco", "metadata": "missing_metadata"}),
])
COUNT_COLUMNS = [
    "count_veto", "count_veto1", "count_veto2", "count_veto3",
    *(f"count_scifi{i}" for i in range(1, 6)),
    *(f"count_us{i}" for i in range(1, 6)),
    *(f"count_ds{i}" for i in range(1, 5)),
]
MANIFEST_FIELDS = [
    "category", "population", "population_label", "veto_time_category",
    "veto_time_label", "partition_id", "partition_path", "partition_entry",
    "source_file_index", "event_source_path", "event_tree_name",
    "source_event_index", "raw_event_index", "feature_event_index",
    "identity_resolution", "geo_path", "runId", "eventId", "eventIndex",
    "original_entry", "selection_expression", "hitTime_veto_min_ns",
    *COUNT_COLUMNS,
]
COUNT_FIELDS = [
    "category", "population", "veto_time_category", "reference_events",
    "sampled_events", "sources_with_events", "partitions_scanned",
    "partitions_total", "scan_complete", "minimum_partitions",
    "minimum_sources_per_category", "manifest_events", "target_events", "reserve_events",
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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def partition_number(value: object) -> int:
    match = re.search(r"part(\d+)$", str(value))
    if not match:
        raise ValueError(f"Cannot extract partition number from {value!r}")
    return int(match.group(1))


def load_partition_sources(
    config: dict, study: dict, population: str, repo_root: Path,
) -> list[dict]:
    metadata_key = POPULATIONS[population]["metadata"]
    path = resolve(config["input"][metadata_key], repo_root)
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"particle_group", "partition_id", "feature_partition_path"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} lacks partition columns: {sorted(missing)}")
    group = str(study["groups"][population])
    selected = frame.loc[frame["particle_group"].astype(str).eq(group)].copy()
    if "feature_available" in selected:
        selected = selected.loc[selected["feature_available"].map(truthy)]
    if selected.empty:
        raise RuntimeError(f"No available {population} partitions for group {group}")
    selected["__part"] = selected["partition_id"].map(partition_number)
    selected = selected.sort_values("__part")
    if selected["__part"].duplicated().any():
        raise RuntimeError(f"Duplicate {population} partition numbers")
    return [
        {
            "partition_id": str(row.partition_id),
            "partition_path": str(row.feature_partition_path),
        }
        for row in selected.itertuples()
    ]


def load_event_sources(config: dict, repo_root: Path) -> dict[int, dict]:
    path = resolve(config["input"]["real_data_metadata"], repo_root)
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "source_file_index", "source_n_event", "digi_path", "geo_path", "run",
        "feature_full_path",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} lacks source columns: {sorted(missing)}")
    if frame["source_file_index"].duplicated().any():
        raise RuntimeError("Real-data metadata contains duplicate source_file_index values")
    result = {}
    for row in frame.itertuples():
        source = int(row.source_file_index)
        result[source] = {
            "event_source_path": str(row.digi_path),
            "identity_feature_path": str(row.feature_full_path),
            "geo_path": str(row.geo_path),
            "run": int(row.run),
            "source_n_event": int(row.source_n_event),
        }
    return result


def resolve_raw_event_entries(
    rows: list[dict], event_sources: dict[int, dict], tree_name: str,
) -> None:
    """Resolve selected feature identities to entries in the original raw trees.

    PassingMuonReco feature ``eventIndex`` values address the passing-muon skim,
    whereas the corresponding RealDataDQ feature files retain the original raw
    entry.  Join on the stable ``(runId, eventId)`` identity and mutate only the
    display-specific entry fields; preserve the feature indices for provenance.
    """
    rows_by_source: OrderedDict[int, list[dict]] = OrderedDict()
    for row in rows:
        source_index = int(row["source_file_index"])
        rows_by_source.setdefault(source_index, []).append(row)

    for source_index, source_rows in rows_by_source.items():
        source = event_sources.get(source_index)
        if source is None:
            raise RuntimeError(
                f"No source metadata while resolving source_file_index={source_index}"
            )
        identity_path = str(source["identity_feature_path"])
        if not identity_path:
            raise RuntimeError(
                f"Source {source_index} has no RealDataDQ identity feature path"
            )

        requested = {
            (int(row["runId"]), int(row["eventId"])) for row in source_rows
        }
        frame = ROOT.RDataFrame(tree_name, root_input(identity_path))
        columns = {str(value) for value in frame.GetColumnNames()}
        required = {"runId", "eventId", "eventIndex"}
        missing = required - columns
        if missing:
            raise RuntimeError(
                f"RealDataDQ identity source {identity_path} lacks "
                f"branches: {sorted(missing)}"
            )
        values = frame.AsNumpy(["runId", "eventId", "eventIndex"])
        raw_entries = {}
        for run_value, event_value, entry_value in zip(
            values["runId"], values["eventId"], values["eventIndex"]
        ):
            identity = (
                int(python_value(run_value)), int(python_value(event_value))
            )
            if identity not in requested:
                continue
            if identity in raw_entries:
                raise RuntimeError(
                    f"Duplicate RealDataDQ identity {identity} in {identity_path}"
                )
            raw_entries[identity] = int(python_value(entry_value))

        unresolved = sorted(requested - set(raw_entries))
        if unresolved:
            preview = unresolved[:5]
            raise RuntimeError(
                f"Could not resolve {len(unresolved)} selected identities for "
                f"source {source_index} in {identity_path}; examples={preview}"
            )

        source_entries = int(source["source_n_event"])
        for row in source_rows:
            identity = (int(row["runId"]), int(row["eventId"]))
            raw_entry = raw_entries[identity]
            if raw_entry < 0 or raw_entry >= source_entries:
                raise RuntimeError(
                    f"Resolved raw entry {raw_entry} for {identity} is outside "
                    f"source {source_index} range 0..{source_entries - 1}"
                )
            row["feature_event_index"] = int(row["eventIndex"])
            row["raw_event_index"] = raw_entry
            row["source_event_index"] = raw_entry
            row["identity_resolution"] = "real_dq_run_event_lookup"


def category_definitions(config: dict, study: dict) -> OrderedDict[str, dict]:
    validate_region_config(study)
    axis_name = str(config["selection"].get("axis", ""))
    if axis_name != "veto_time":
        raise ValueError("V4 event displays require selection.axis=veto_time")
    available = study["axes"][axis_name]["categories"]
    requested = list(config["selection"].get("categories", []))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("selection.categories must be a nonempty unique list")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown V4 veto-time categories: {sorted(unknown)}")
    definitions: OrderedDict[str, dict] = OrderedDict()
    for population, population_definition in POPULATIONS.items():
        for veto_category in requested:
            region = available[veto_category]
            key = f"{population}__{veto_category}"
            definitions[key] = {
                "population": population,
                "population_label": population_definition["label"],
                "veto_time_category": veto_category,
                "veto_time_label": str(region["label"]),
                "expression": str(region["expression"]).strip(),
            }
    return definitions


def reference_count(histogram: ROOT.TFile, definition: dict) -> int:
    name = (
        "axis__veto_time__"
        f"{definition['veto_time_category']}__{definition['population']}__count"
    )
    value = histogram.Get(name)
    if not value:
        raise RuntimeError(f"Reference histogram lacks {name}")
    return int(value.GetVal() if hasattr(value, "GetVal") else value.GetTitle())


def python_value(value):
    return value.item() if hasattr(value, "item") else value


def dataframe_for_partition(partition: dict, tree_name: str):
    frame = ROOT.RDataFrame(tree_name, root_input(partition["partition_path"]))
    columns = {str(value) for value in frame.GetColumnNames()}
    required = {
        "source_file_index", "runId", "eventId", "eventIndex",
        "original_entry", "hitTime_veto_min_ns", *COUNT_COLUMNS,
    }
    missing = required - columns
    if missing:
        raise RuntimeError(
            f"{partition['partition_path']} lacks branches: {sorted(missing)}"
        )
    return frame.Define("__v4_partition_entry", "(long long)rdfentry_")


def sample_from_frame(
    frame, expression: str, limit: int, partition: dict, definition: dict,
    category: str, event_sources: dict[int, dict],
) -> list[dict]:
    if limit <= 0:
        return []
    read_columns = [
        "__v4_partition_entry", "source_file_index", "runId", "eventId",
        "eventIndex", "original_entry", "hitTime_veto_min_ns", *COUNT_COLUMNS,
    ]
    values = frame.Filter(expression).Range(limit).AsNumpy(read_columns)
    rows = []
    for index in range(len(values["__v4_partition_entry"])):
        source_index = int(python_value(values["source_file_index"][index]))
        if source_index not in event_sources:
            raise RuntimeError(f"No source metadata for source_file_index={source_index}")
        source = event_sources[source_index]
        run_id = int(python_value(values["runId"][index]))
        event_index = int(python_value(values["eventIndex"][index]))
        original_entry = int(python_value(values["original_entry"][index]))
        if run_id != source["run"]:
            raise RuntimeError(
                f"Run mismatch for source {source_index}: {run_id} != {source['run']}"
            )
        if event_index != original_entry:
            raise RuntimeError(
                f"Entry mismatch for source {source_index}: "
                f"eventIndex={event_index}, original_entry={original_entry}"
            )
        row = {field: "" for field in MANIFEST_FIELDS}
        row.update({
            "category": category,
            "population": definition["population"],
            "population_label": definition["population_label"],
            "veto_time_category": definition["veto_time_category"],
            "veto_time_label": definition["veto_time_label"],
            "partition_id": partition["partition_id"],
            "partition_path": partition["partition_path"],
            "partition_entry": int(
                python_value(values["__v4_partition_entry"][index])
            ),
            "source_file_index": source_index,
            "event_source_path": source["event_source_path"],
            "event_tree_name": "",
            # Resolved after bounded category sampling using the authoritative
            # RealDataDQ (runId, eventId) -> raw eventIndex mapping.
            "source_event_index": "",
            "raw_event_index": "",
            "feature_event_index": event_index,
            "identity_resolution": "",
            "geo_path": source["geo_path"],
            "runId": run_id,
            "eventId": int(python_value(values["eventId"][index])),
            "eventIndex": event_index,
            "original_entry": original_entry,
            "selection_expression": expression,
            "hitTime_veto_min_ns": float(
                python_value(values["hitTime_veto_min_ns"][index])
            ),
        })
        for column in COUNT_COLUMNS:
            row[column] = int(python_value(values[column][index]))
        rows.append(row)
    return rows


def round_robin_sources(rows: list[dict], limit: int) -> list[dict]:
    groups: OrderedDict[int, deque] = OrderedDict()
    for row in rows:
        groups.setdefault(int(row["source_file_index"]), deque()).append(row)
    result = []
    while any(groups.values()) and len(result) < limit:
        for queue in groups.values():
            if queue and len(result) < limit:
                result.append(queue.popleft())
    return result


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
    parser.add_argument("--partition-metadata")
    parser.add_argument("--missing-metadata")
    parser.add_argument("--real-data-metadata")
    parser.add_argument("--reference-histogram")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = load_yaml(resolve(args.config, repo_root))
    # Command-line overrides make the Snakemake rule's frozen inputs explicit
    # while retaining a self-contained YAML interface for direct use.
    if args.partition_metadata:
        config["input"]["partition_metadata"] = args.partition_metadata
    if args.missing_metadata:
        config["input"]["missing_metadata"] = args.missing_metadata
    if args.real_data_metadata:
        config["input"]["real_data_metadata"] = args.real_data_metadata
    if args.reference_histogram:
        config["reference_histogram"] = args.reference_histogram
    study = load_yaml(resolve(config["study_config"], repo_root))
    definitions = category_definitions(config, study)
    event_sources = load_event_sources(config, repo_root)
    tree_name = str(config["input"].get("tree_name", study.get("tree_name", "sndData")))
    target = int(config["sampling"]["events_per_category"])
    reserve = target * int(config["sampling"].get("reserve_factor", 1))
    configured_minimum = int(config["sampling"].get("minimum_partitions", 1))
    minimum_sources = int(
        config["sampling"].get("minimum_sources_per_category", 1)
    )
    if target < 1 or reserve < target:
        raise ValueError("Sampling target must be positive and reserve must cover the target")
    if configured_minimum < 1:
        raise ValueError("sampling.minimum_partitions must be positive")
    if minimum_sources < 1:
        raise ValueError("sampling.minimum_sources_per_category must be positive")
    if config["sampling"].get("mode", "round_robin_sources") != "round_robin_sources":
        raise ValueError("Only sampling.mode=round_robin_sources is supported")

    reference_path = resolve(config["reference_histogram"], repo_root)
    reference = ROOT.TFile.Open(root_input(reference_path))
    if not reference or reference.IsZombie():
        raise OSError(f"Cannot open V4 reference histogram {reference_path}")
    try:
        reference_counts = {
            category: reference_count(reference, definition)
            for category, definition in definitions.items()
        }
    finally:
        reference.Close()
    insufficient = {
        category: count for category, count in reference_counts.items()
        if count < reserve
    }
    if insufficient:
        raise RuntimeError(
            f"V4 categories cannot supply reserve={reserve}: {insufficient}"
        )

    selected_rows = {}
    sampling_stats = {}
    for population in POPULATIONS:
        categories = [
            category for category, definition in definitions.items()
            if definition["population"] == population
        ]
        partitions = load_partition_sources(config, study, population, repo_root)
        minimum = min(configured_minimum, len(partitions))
        pools = {category: [] for category in categories}
        pool_sources = {category: set() for category in categories}
        partitions_scanned = 0
        for partition_number_, partition in enumerate(partitions, start=1):
            if (
                partition_number_ > minimum
                and all(
                    len(pools[category]) >= reserve
                    and len(pool_sources[category]) >= minimum_sources
                    for category in categories
                )
            ):
                break
            categories_to_sample = categories if partition_number_ <= minimum else [
                category for category in categories
                if len(pools[category]) < reserve
                or len(pool_sources[category]) < minimum_sources
            ]
            print(
                f"[sample] {population} partition {partition_number_}/{len(partitions)}: "
                f"{partition['partition_path']}"
            )
            frame = dataframe_for_partition(partition, tree_name)
            for category in categories_to_sample:
                limit = reserve if (
                    partition_number_ <= minimum
                    or len(pools[category]) >= reserve
                ) else reserve - len(pools[category])
                new_rows = sample_from_frame(
                    frame, definitions[category]["expression"], limit,
                    partition, definitions[category], category, event_sources,
                )
                pools[category].extend(new_rows)
                pool_sources[category].update(
                    int(row["source_file_index"]) for row in new_rows
                )
            partitions_scanned = partition_number_
        for category in categories:
            selected_rows[category] = round_robin_sources(pools[category], reserve)
            sampling_stats[category] = {
                "sampled_events": len(pools[category]),
                "sources_with_events": len(pool_sources[category]),
                "partitions_scanned": partitions_scanned,
                "partitions_total": len(partitions),
                "scan_complete": partitions_scanned == len(partitions),
                "minimum_partitions": minimum,
                "minimum_sources_per_category": minimum_sources,
            }
        print(
            f"[sample] {population}: stopped after "
            f"{partitions_scanned}/{len(partitions)} partitions"
        )

    manifest = []
    count_rows = []
    seen = set()
    for category, definition in definitions.items():
        rows = selected_rows.get(category, [])
        if len(rows) < reserve:
            raise RuntimeError(
                f"Category {category} has only {len(rows)} reserved events; "
                f"required reserve={reserve}, target={target}"
            )
        observed_sources = int(sampling_stats[category]["sources_with_events"])
        if observed_sources < minimum_sources:
            raise RuntimeError(
                f"Category {category} spans only {observed_sources} source files; "
                f"required minimum={minimum_sources}"
            )
        for row in rows:
            identity = (
                int(row["source_file_index"]), int(row["runId"]), int(row["eventId"])
            )
            if identity in seen:
                raise RuntimeError(f"Duplicate selected event identity: {identity}")
            seen.add(identity)
            manifest.append(row)
        count_rows.append({
            "category": category,
            "population": definition["population"],
            "veto_time_category": definition["veto_time_category"],
            "reference_events": reference_counts[category],
            **sampling_stats[category],
            "manifest_events": len(rows),
            "target_events": target,
            "reserve_events": reserve,
            "selection_expression": definition["expression"],
        })

    resolve_raw_event_entries(manifest, event_sources, tree_name)
    atomic_csv(manifest, MANIFEST_FIELDS, args.output_manifest)
    atomic_csv(count_rows, COUNT_FIELDS, args.output_counts)
    print(f"Wrote {len(manifest)} reserved V4 event rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
