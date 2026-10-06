#!/usr/bin/env python3
"""Validate a unified shower-feature file and write its validation record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import ROOT

from sample_selection import (
    SELECTION_MODES,
    cutflow_selection_expression,
    resolve_selection_mode,
    setup_event_deltat_alias,
    uses_external_cutflow,
)
from scifi_plane_clustering import SCHEMA_VERSION, scalar_feature_dtypes
from source_provenance import (
    SOURCE_DOMAINS,
    UNAVAILABLE_ENERGY_GEV,
    UNAVAILABLE_PDG_CODE,
    UNIFIED_SCHEMA_VERSION,
)


REQUIRED_BRANCH_TYPES = {
    "runId": "Int_t",
    "eventId": "Int_t",
    "eventIndex": "Int_t",
    "source_energy_gev": "Double_t",
    "source_pdg_code": "Int_t",
    "source_domain_id": "Int_t",
}


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-7)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_title(root_file: ROOT.TFile, name: str) -> str:
    value = root_file.Get(name)
    if not value or not hasattr(value, "GetTitle"):
        raise RuntimeError(f"missing metadata object {name}")
    return str(value.GetTitle())


def metadata_integer(root_file: ROOT.TFile, name: str) -> int:
    value = root_file.Get(name)
    if not value or not hasattr(value, "GetVal"):
        raise RuntimeError(f"missing integer metadata object {name}")
    return int(value.GetVal())


def selected_cutflow_entries(tree: ROOT.TTree, selection: str | None) -> list[int]:
    if not selection:
        return list(range(int(tree.GetEntries())))
    list_name = "validator_selected_entries"
    tree.Draw(f">>{list_name}", selection, "entrylist")
    selected = ROOT.gDirectory.Get(list_name)
    if not selected or not isinstance(selected, ROOT.TEntryList):
        raise RuntimeError("failed to build validator source-cutflow entry list")
    return [int(selected.GetEntry(index)) for index in range(int(selected.GetN()))]


def raw_entries_for_cutflow_rows(tree: ROOT.TTree, rows: list[int]) -> list[int]:
    has_entry = bool(tree.GetBranch("entry") or tree.GetLeaf("entry"))
    raw_entries = []
    for row in rows:
        tree.GetEntry(row)
        raw_entries.append(int(tree.entry) if has_entry else int(row))
    return raw_entries


def inspect_source_selection(args: argparse.Namespace) -> dict:
    """Independently derive input and selected counts from frozen sources."""
    digi = ROOT.TFile.Open(args.input_digi_path)
    if not digi or digi.IsZombie():
        raise RuntimeError(f"cannot open input digi {args.input_digi_path}")
    try:
        digi_tree = digi.Get(args.input_tree)
        if not digi_tree:
            raise RuntimeError(f"missing input tree {args.input_tree}")
        input_entries = int(digi_tree.GetEntries())
    finally:
        digi.Close()
    if input_entries != int(args.expected_input_event_count):
        raise RuntimeError(
            f"input digi entry count {input_entries} does not match manifest "
            f"{args.expected_input_event_count}"
        )

    selection_mode = resolve_selection_mode(args.sample_type, args.selection_mode)
    selection = None
    if uses_external_cutflow(args.sample_type, selection_mode):
        if not args.input_cutflow_path:
            raise RuntimeError("external selection requires an input cutflow")
        cutflow_source = ROOT.TFile.Open(args.input_cutflow_path)
        if not cutflow_source or cutflow_source.IsZombie():
            raise RuntimeError(f"cannot open input cutflow {args.input_cutflow_path}")
        try:
            cutflow_tree = cutflow_source.Get("cutFlowSummary")
            if not cutflow_tree:
                raise RuntimeError("missing source cutFlowSummary tree")
            event_deltat_branch = setup_event_deltat_alias(cutflow_tree)
            selection = cutflow_selection_expression(
                args.sample_type,
                selection_mode,
                event_deltat_available=bool(event_deltat_branch),
            )
            selected_rows = selected_cutflow_entries(cutflow_tree, selection)
            selected_raw_entries = raw_entries_for_cutflow_rows(
                cutflow_tree, selected_rows
            )
        finally:
            cutflow_source.Close()
    else:
        selected_raw_entries = list(range(input_entries))

    max_events = int(args.expected_max_events)
    if max_events < 0:
        raise RuntimeError("expected max-events value must be non-negative")
    expected_raw_entries = (
        selected_raw_entries[:max_events] if max_events else selected_raw_entries
    )
    return {
        "input_event_count": input_entries,
        "selection_mode": selection_mode,
        "selection_expression": selection or "",
        "available_selected_event_count": len(selected_raw_entries),
        "expected_written_event_count": len(expected_raw_entries),
        "expected_raw_entries": expected_raw_entries,
        "max_events": max_events,
    }


def branch_type(tree: ROOT.TTree, name: str) -> str:
    branch = tree.GetBranch(name)
    if not branch:
        raise RuntimeError(f"missing required branch {name}")
    leaf = branch.GetLeaf(name)
    if not leaf and branch.GetListOfLeaves().GetEntries() == 1:
        leaf = branch.GetListOfLeaves().At(0)
    if not leaf:
        raise RuntimeError(f"could not determine ROOT type for branch {name}")
    return str(leaf.GetTypeName())


def validate_source_scalar_values(domain_id: int, energy: float, pdg: int) -> None:
    if domain_id not in SOURCE_DOMAINS or domain_id == 0:
        raise RuntimeError(f"invalid unified source domain ID {domain_id}")
    if not math.isfinite(float(energy)):
        raise RuntimeError("source_energy_gev is not finite")
    if domain_id == 3:
        if not close(energy, UNAVAILABLE_ENERGY_GEV) or pdg != UNAVAILABLE_PDG_CODE:
            raise RuntimeError("collision-data source sentinels do not match the contract")
        return
    if energy < 0 or pdg == 0:
        raise RuntimeError(f"source truth/beam values are invalid for domain {domain_id}")


def validate_cluster_closure(tree: ROOT.TTree, entry: int) -> None:
    vertical_count = int(tree.scifi_vertical_cluster_count)
    horizontal_count = int(tree.scifi_horizontal_cluster_count)
    vertical_hits = int(tree.scifi_vertical_total_nhits)
    horizontal_hits = int(tree.scifi_horizontal_total_nhits)
    vertical_qdc = float(tree.scifi_vertical_total_qdc)
    horizontal_qdc = float(tree.scifi_horizontal_total_qdc)
    count_scifi = int(tree.count_scifi)
    direct_expected = float(tree.qdc_scifi) / count_scifi if count_scifi else 0.0
    checks = (
        (tree.scifi_cluster_total_count, vertical_count + horizontal_count, "cluster count"),
        (tree.scifi_cluster_total_nhits, vertical_hits + horizontal_hits, "hit count"),
        (tree.scifi_cluster_total_qdc, vertical_qdc + horizontal_qdc, "QDC"),
        (tree.scifi_cluster_total_nhits, tree.count_scifi, "existing SciFi hit"),
        (tree.scifi_cluster_total_qdc, tree.qdc_scifi, "existing SciFi QDC"),
        (tree.scifi_qdc_per_hit_direct, direct_expected, "direct SciFi QDC/hit"),
        (
            tree.scifi_qdc_per_hit_direct,
            tree.scifi_cluster_qdc_per_hit,
            "direct/cluster SciFi QDC/hit",
        ),
        (
            tree.scifi_other_clusters_count,
            tree.scifi_vertical_other_cluster_count
            + tree.scifi_horizontal_other_cluster_count,
            "other count",
        ),
        (
            tree.scifi_other_clusters_nhits,
            tree.scifi_vertical_other_cluster_nhits
            + tree.scifi_horizontal_other_cluster_nhits,
            "other hits",
        ),
        (
            tree.scifi_other_clusters_qdc,
            tree.scifi_vertical_other_cluster_qdc
            + tree.scifi_horizontal_other_cluster_qdc,
            "other QDC",
        ),
    )
    for observed, expected, label in checks:
        if not close(observed, expected):
            raise RuntimeError(f"{label} closure failed at entry {entry}")


def write_record(path: str | Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def validate(args: argparse.Namespace) -> dict:
    source_contract = inspect_source_selection(args)
    source = ROOT.TFile.Open(args.feature)
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open {args.feature}")
    try:
        tree = source.Get(args.tree)
        if not tree:
            raise RuntimeError(f"missing tree {args.tree}")
        names = {branch.GetName() for branch in tree.GetListOfBranches()}
        required = {name for name, _ in scalar_feature_dtypes()}
        required.update(REQUIRED_BRANCH_TYPES)
        required.update({"count_scifi", "qdc_scifi", "scifi_qdc_per_hit_direct"})
        missing = sorted(required - names)
        if missing:
            raise RuntimeError("missing unified branches: " + ", ".join(missing))
        for name, expected_type in REQUIRED_BRANCH_TYPES.items():
            observed_type = branch_type(tree, name)
            if observed_type != expected_type:
                raise RuntimeError(
                    f"branch {name} has ROOT type {observed_type}, expected {expected_type}"
                )

        if metadata_title(source, "metadata_scifi_cluster_schema") != SCHEMA_VERSION:
            raise RuntimeError("wrong SciFi cluster schema marker")
        if (
            metadata_title(source, "metadata_unified_shower_feature_schema")
            != UNIFIED_SCHEMA_VERSION
        ):
            raise RuntimeError("wrong unified shower-feature schema marker")
        if metadata_title(source, "metadata_source_uid") != args.expected_source_uid:
            raise RuntimeError("source UID metadata does not match the manifest")
        if args.expected_source_geometry_id is not None and (
            metadata_title(source, "metadata_source_geometry_id")
            != args.expected_source_geometry_id
        ):
            raise RuntimeError("source geometry ID metadata does not match the manifest")
        if args.expected_source_geometry_checksum is not None and (
            metadata_title(source, "metadata_source_geometry_checksum")
            != args.expected_source_geometry_checksum
        ):
            raise RuntimeError("source geometry checksum metadata does not match the manifest")
        if (
            metadata_title(source, "metadata_selection_mode")
            != source_contract["selection_mode"]
        ):
            raise RuntimeError("selection-mode metadata does not match the source contract")
        if (
            metadata_integer(source, "metadata_input_entries")
            != source_contract["input_event_count"]
        ):
            raise RuntimeError("input-entry metadata does not match the source contract")
        if (
            metadata_integer(source, "metadata_available_selected_entries")
            != source_contract["available_selected_event_count"]
        ):
            raise RuntimeError(
                "available-selected-entry metadata does not match the source contract"
            )
        if (
            metadata_integer(source, "metadata_max_events")
            != source_contract["max_events"]
        ):
            raise RuntimeError("max-events metadata does not match the source contract")

        cutflow = source.Get("cutFlowSummary")
        if not cutflow:
            raise RuntimeError("missing cutFlowSummary tree")
        entries = int(tree.GetEntries())
        if int(cutflow.GetEntries()) != entries:
            raise RuntimeError("sndData/cutFlowSummary entry-count mismatch")
        if entries != source_contract["expected_written_event_count"]:
            raise RuntimeError(
                f"written entry count {entries} does not match expected "
                f"selected count {source_contract['expected_written_event_count']}"
            )

        hit_source = None
        hit_tree = None
        if args.hit3d:
            hit_source = ROOT.TFile.Open(args.hit3d)
            hit_tree = hit_source.Get(args.hit3d_tree) if hit_source else None
            if not hit_tree:
                raise RuntimeError("missing Hit3D tree")
            if int(hit_tree.GetEntries()) != entries:
                raise RuntimeError("feature/Hit3D entry-count mismatch")

        seen = set()
        limit = entries if args.max_events <= 0 else min(entries, args.max_events)
        for entry in range(limit):
            tree.GetEntry(entry)
            expected_event_index = source_contract["expected_raw_entries"][entry]
            if int(tree.eventIndex) != expected_event_index:
                raise RuntimeError(
                    f"source eventIndex mismatch at entry {entry}: "
                    f"{int(tree.eventIndex)} != {expected_event_index}"
                )
            cutflow.GetEntry(entry)
            if (cutflow.GetBranch("entry") or cutflow.GetLeaf("entry")) and (
                int(cutflow.entry) != expected_event_index
            ):
                raise RuntimeError(
                    f"output cutflow entry mismatch at entry {entry}: "
                    f"{int(cutflow.entry)} != {expected_event_index}"
                )
            identity = (int(tree.runId), int(tree.eventId), int(tree.eventIndex))
            if identity in seen:
                raise RuntimeError(f"duplicate event identity {identity} at entry {entry}")
            seen.add(identity)
            if hit_tree:
                hit_tree.GetEntry(entry)
                hit_identity = (
                    int(hit_tree.runId),
                    int(hit_tree.eventId),
                    int(hit_tree.eventIndex),
                )
                if hit_identity != identity:
                    raise RuntimeError(f"feature/Hit3D identity mismatch at entry {entry}")

            domain_id = int(tree.source_domain_id)
            if domain_id != args.expected_source_domain_id:
                raise RuntimeError(
                    f"source domain {domain_id} does not match expected "
                    f"{args.expected_source_domain_id} at entry {entry}"
                )
            validate_source_scalar_values(
                domain_id, float(tree.source_energy_gev), int(tree.source_pdg_code)
            )
            validate_cluster_closure(tree, entry)
            for name, _ in scalar_feature_dtypes():
                if not math.isfinite(float(getattr(tree, name))):
                    raise RuntimeError(f"non-finite {name} at entry {entry}")
            if not math.isfinite(float(tree.scifi_qdc_per_hit_direct)):
                raise RuntimeError(f"non-finite scifi_qdc_per_hit_direct at entry {entry}")
        if hit_source:
            hit_source.Close()
    finally:
        source.Close()

    return {
        "validation_passed": True,
        "manifest_version": args.manifest_version,
        "unified_schema_version": UNIFIED_SCHEMA_VERSION,
        "scifi_cluster_schema_version": SCHEMA_VERSION,
        "source_uid": args.expected_source_uid,
        "source_domain_id": args.expected_source_domain_id,
        "source_domain": SOURCE_DOMAINS[args.expected_source_domain_id],
        "source_geometry_id": args.expected_source_geometry_id or "",
        "source_geometry_checksum": args.expected_source_geometry_checksum or "",
        "input_digi_path": args.input_digi_path,
        "input_cutflow_path": args.input_cutflow_path or "",
        "feature_path": args.published_feature_path or args.feature,
        "feature_size_bytes": os.path.getsize(args.feature),
        "feature_sha256": sha256(args.feature),
        "expected_input_event_count": args.expected_input_event_count,
        "source_selection_expression": source_contract["selection_expression"],
        "available_selected_event_count": source_contract[
            "available_selected_event_count"
        ],
        "expected_written_event_count": source_contract[
            "expected_written_event_count"
        ],
        "conversion_max_events": source_contract["max_events"],
        "written_event_count": entries,
        "validated_event_count": limit,
        "identity_validation": (
            "source_selection_and_hit3d_entrywise"
            if args.hit3d
            else "source_selection_entrywise"
        ),
        "converter_commit": args.converter_commit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature")
    parser.add_argument("--tree", default="sndData")
    parser.add_argument("--hit3d")
    parser.add_argument("--hit3d-tree", default="hit3D")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--manifest-version", required=True)
    parser.add_argument("--expected-source-uid", required=True)
    parser.add_argument("--expected-source-domain-id", type=int, required=True)
    parser.add_argument("--expected-source-geometry-id")
    parser.add_argument("--expected-source-geometry-checksum")
    parser.add_argument("--expected-input-event-count", type=int, required=True)
    parser.add_argument("--sample-type", required=True)
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, required=True)
    parser.add_argument("--input-tree", default="cbmsim")
    parser.add_argument("--expected-max-events", type=int, default=0)
    parser.add_argument("--input-digi-path", required=True)
    parser.add_argument("--input-cutflow-path")
    parser.add_argument("--published-feature-path")
    parser.add_argument("--converter-commit", required=True)
    parser.add_argument("--validation-record", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = validate(args)
    write_record(args.validation_record, record)
    print(
        f"validated {record['validated_event_count']} entries; "
        f"schema={record['unified_schema_version']}"
    )


if __name__ == "__main__":
    main()
