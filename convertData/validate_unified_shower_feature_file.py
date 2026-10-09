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


def identity_cutflow_entries(tree: ROOT.TTree, expected_entries: int) -> range:
    """Validate an identity cutflow in compiled ROOT code and return its range."""
    entries = int(tree.GetEntries())
    if entries != expected_entries:
        raise RuntimeError(
            f"identity cutflow has {entries} rows, expected {expected_entries}"
        )
    has_entry = bool(tree.GetBranch("entry") or tree.GetLeaf("entry"))
    if has_entry and entries:
        mismatches = int(
            ROOT.RDataFrame(tree)
            .Filter("static_cast<Long64_t>(entry) != "
                    "static_cast<Long64_t>(rdfentry_)")
            .Count()
            .GetValue()
        )
        if mismatches:
            raise RuntimeError(
                f"identity cutflow has {mismatches} non-identity entry values"
            )
    return range(entries)


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
    expected_selected_event_count = getattr(
        args, "expected_selected_event_count", None
    )
    if expected_selected_event_count is not None:
        expected_selected_event_count = int(expected_selected_event_count)
        if expected_selected_event_count < 0:
            raise RuntimeError(
                "expected selected-event count must be non-negative"
            )
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
            if selection_mode == "identity_cutflow" and not selection:
                expected_identity_entries = (
                    input_entries
                    if expected_selected_event_count is None
                    else expected_selected_event_count
                )
                selected_raw_entries = identity_cutflow_entries(
                    cutflow_tree, expected_identity_entries
                )
            else:
                selected_rows = selected_cutflow_entries(cutflow_tree, selection)
                selected_raw_entries = raw_entries_for_cutflow_rows(
                    cutflow_tree, selected_rows
                )
        finally:
            cutflow_source.Close()
    else:
        selected_raw_entries = range(input_entries)
    if (
        expected_selected_event_count is not None
        and len(selected_raw_entries) != expected_selected_event_count
    ):
        raise RuntimeError(
            f"selected source has {len(selected_raw_entries)} entries, expected "
            f"{expected_selected_event_count}"
        )

    selected_entry_start_arg = getattr(args, "expected_selected_entry_start", None)
    selected_entry_stop_arg = getattr(args, "expected_selected_entry_stop", None)
    range_requested = (
        selected_entry_start_arg is not None
        or selected_entry_stop_arg is not None
    )
    selected_entry_start = int(selected_entry_start_arg or 0)
    selected_entry_stop = (
        len(selected_raw_entries)
        if selected_entry_stop_arg is None
        else int(selected_entry_stop_arg)
    )
    if selected_entry_start < 0:
        raise RuntimeError("expected selected-entry start must be non-negative")
    if selected_entry_stop < selected_entry_start:
        raise RuntimeError(
            "expected selected-entry stop must not precede the start"
        )
    if selected_entry_stop > len(selected_raw_entries):
        raise RuntimeError(
            f"expected selected-entry stop {selected_entry_stop} exceeds "
            f"available selected entries {len(selected_raw_entries)}"
        )
    selected_window = selected_raw_entries[
        selected_entry_start:selected_entry_stop
    ]

    max_events = int(args.expected_max_events)
    if max_events < 0:
        raise RuntimeError("expected max-events value must be non-negative")
    expected_raw_entries = (
        selected_window[:max_events] if max_events else selected_window
    )
    selected_entry_stop_effective = (
        selected_entry_start + len(expected_raw_entries)
    )
    return {
        "input_event_count": input_entries,
        "selection_mode": selection_mode,
        "selection_expression": selection or "",
        "available_selected_event_count": len(selected_raw_entries),
        "expected_written_event_count": len(expected_raw_entries),
        "expected_raw_entries": expected_raw_entries,
        "max_events": max_events,
        "range_requested": range_requested,
        "selected_entry_start": selected_entry_start,
        "selected_entry_stop": selected_entry_stop_effective,
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


def contiguous_entry_start(entries) -> int | None:
    """Return the first value when entries form a unit-step range."""
    if isinstance(entries, range):
        return entries.start if entries.step == 1 else None
    if not entries:
        return 0
    start = int(entries[0])
    for offset, value in enumerate(entries):
        if int(value) != start + offset:
            return None
    return start


def declare_compiled_validation_helpers() -> None:
    ROOT.gInterpreter.Declare(
        r"""
        #ifndef UNIFIED_SHOWER_VALIDATION_HELPERS_V1
        #define UNIFIED_SHOWER_VALIDATION_HELPERS_V1
        #include <algorithm>
        #include <cmath>
        inline bool unified_feature_close(double left, double right) {
            const double tolerance = std::max(
                1e-7,
                1e-9 * std::max(std::abs(left), std::abs(right))
            );
            return std::abs(left - right) <= tolerance;
        }
        #endif
        """
    )


def compiled_event_validation(
    tree: ROOT.TTree,
    cutflow: ROOT.TTree,
    expected_raw_entries,
    expected_domain_id: int,
    limit: int,
) -> int:
    """Validate a contiguous event range in one branch-pruned C++ scan."""
    start = contiguous_entry_start(expected_raw_entries)
    if start is None:
        raise RuntimeError("compiled validation requires contiguous event indices")
    if limit == 0:
        return 0

    declare_compiled_validation_helpers()
    finite_names = sorted(
        {name for name, _ in scalar_feature_dtypes()}
        | {"source_energy_gev", "scifi_qdc_per_hit_direct"}
    )
    checks = [
        (
            "static_cast<Long64_t>(eventIndex) == "
            f"{start}LL + static_cast<Long64_t>(rdfentry_)"
        ),
        f"static_cast<int>(source_domain_id) == {int(expected_domain_id)}",
        "std::isfinite(static_cast<double>(source_energy_gev))",
        (
            "(static_cast<int>(source_domain_id) == 3 ? "
            f"(unified_feature_close(source_energy_gev, {UNAVAILABLE_ENERGY_GEV}) "
            f"&& static_cast<int>(source_pdg_code) == {UNAVAILABLE_PDG_CODE}) : "
            "(source_energy_gev >= 0.0 && "
            "static_cast<int>(source_pdg_code) != 0))"
        ),
        (
            "unified_feature_close(scifi_cluster_total_count, "
            "scifi_vertical_cluster_count + scifi_horizontal_cluster_count)"
        ),
        (
            "unified_feature_close(scifi_cluster_total_nhits, "
            "scifi_vertical_total_nhits + scifi_horizontal_total_nhits)"
        ),
        (
            "unified_feature_close(scifi_cluster_total_qdc, "
            "scifi_vertical_total_qdc + scifi_horizontal_total_qdc)"
        ),
        "unified_feature_close(scifi_cluster_total_nhits, count_scifi)",
        "unified_feature_close(scifi_cluster_total_qdc, qdc_scifi)",
        (
            "unified_feature_close(scifi_qdc_per_hit_direct, "
            "count_scifi ? qdc_scifi / count_scifi : 0.0)"
        ),
        (
            "unified_feature_close(scifi_qdc_per_hit_direct, "
            "scifi_cluster_qdc_per_hit)"
        ),
        (
            "unified_feature_close(scifi_other_clusters_count, "
            "scifi_vertical_other_cluster_count + "
            "scifi_horizontal_other_cluster_count)"
        ),
        (
            "unified_feature_close(scifi_other_clusters_nhits, "
            "scifi_vertical_other_cluster_nhits + "
            "scifi_horizontal_other_cluster_nhits)"
        ),
        (
            "unified_feature_close(scifi_other_clusters_qdc, "
            "scifi_vertical_other_cluster_qdc + "
            "scifi_horizontal_other_cluster_qdc)"
        ),
    ]
    checks.extend(
        f"std::isfinite(static_cast<double>({name}))" for name in finite_names
    )
    expression = " && ".join(f"({check})" for check in checks)
    frame = ROOT.RDataFrame(tree).Range(0, limit)
    failures = int(frame.Filter(f"!({expression})").Count().GetValue())
    if failures:
        raise RuntimeError(
            f"compiled event validation failed for {failures}/{limit} entries"
        )

    if cutflow.GetBranch("entry") or cutflow.GetLeaf("entry"):
        cutflow_expression = (
            "static_cast<Long64_t>(entry) != "
            f"{start}LL + static_cast<Long64_t>(rdfentry_)"
        )
        cutflow_failures = int(
            ROOT.RDataFrame(cutflow)
            .Range(0, limit)
            .Filter(cutflow_expression)
            .Count()
            .GetValue()
        )
        if cutflow_failures:
            raise RuntimeError(
                "compiled cutflow identity validation failed for "
                f"{cutflow_failures}/{limit} entries"
            )
    return limit


def validate_merge_boundaries(
    tree: ROOT.TTree,
    cutflow: ROOT.TTree,
    expected_raw_entries,
    merge_record: dict,
    expected_domain_id: int,
) -> int:
    """Check the first and last event contributed by every validated shard."""
    ranges = merge_record.get("shard_selected_entry_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise RuntimeError("merge record lacks shard selected-entry ranges")
    indices = set()
    for selected_range in ranges:
        if not isinstance(selected_range, list) or len(selected_range) != 2:
            raise RuntimeError("invalid shard selected-entry range in merge record")
        start, stop = (int(value) for value in selected_range)
        if stop <= start:
            raise RuntimeError("empty shard selected-entry range in merge record")
        indices.update((start, stop - 1))
    for entry in sorted(indices):
        tree.GetEntry(entry)
        expected_event_index = int(expected_raw_entries[entry])
        if int(tree.eventIndex) != expected_event_index:
            raise RuntimeError(
                f"merged boundary eventIndex mismatch at entry {entry}"
            )
        if int(tree.source_domain_id) != expected_domain_id:
            raise RuntimeError(f"merged boundary source domain mismatch at {entry}")
        cutflow.GetEntry(entry)
        if (cutflow.GetBranch("entry") or cutflow.GetLeaf("entry")) and (
            int(cutflow.entry) != expected_event_index
        ):
            raise RuntimeError(f"merged boundary cutflow mismatch at entry {entry}")
    return len(indices)


def write_record(path: str | Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def validate(args: argparse.Namespace) -> dict:
    source_contract = inspect_source_selection(args)
    merge_record = None
    if getattr(args, "merge_record", None):
        with open(args.merge_record, encoding="utf-8") as source_record:
            merge_record = json.load(source_record)
        if merge_record.get("merge_passed") is not True:
            raise RuntimeError("feature-shard merge record did not pass")
        if merge_record.get("source_uid") != args.expected_source_uid:
            raise RuntimeError("feature-shard merge source UID mismatch")

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
        if source_contract["range_requested"]:
            if (
                metadata_integer(source, "metadata_selected_entry_start")
                != source_contract["selected_entry_start"]
            ):
                raise RuntimeError(
                    "selected-entry-start metadata does not match the source contract"
                )
            if (
                metadata_integer(source, "metadata_selected_entry_stop")
                != source_contract["selected_entry_stop"]
            ):
                raise RuntimeError(
                    "selected-entry-stop metadata does not match the source contract"
                )

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

        source_value_checks = []
        expected_source_pdg_code = getattr(args, "expected_source_pdg_code", None)
        expected_source_energy_gev = getattr(args, "expected_source_energy_gev", None)
        if expected_source_pdg_code is not None:
            source_value_checks.append(
                "static_cast<int>(source_pdg_code) != "
                f"{int(expected_source_pdg_code)}"
            )
        if expected_source_energy_gev is not None:
            source_value_checks.append(
                "!unified_feature_close(source_energy_gev, "
                f"{float(expected_source_energy_gev):.17g})"
            )
        if source_value_checks and entries:
            declare_compiled_validation_helpers()
            mismatches = int(
                ROOT.RDataFrame(tree)
                .Filter(" || ".join(f"({check})" for check in source_value_checks))
                .Count()
                .GetValue()
            )
            if mismatches:
                raise RuntimeError(
                    f"source PDG/energy mismatch in {mismatches}/{entries} entries"
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

        limit = entries if args.max_events <= 0 else min(entries, args.max_events)
        merge_boundary_count = 0
        if merge_record:
            if int(merge_record.get("merged_event_count", -1)) != entries:
                raise RuntimeError("feature-shard merge entry count mismatch")
            merge_boundary_count = validate_merge_boundaries(
                tree,
                cutflow,
                source_contract["expected_raw_entries"],
                merge_record,
                args.expected_source_domain_id,
            )
            validation_execution = "validated_shards_plus_merge_boundaries"
        elif hit_tree is None and contiguous_entry_start(
            source_contract["expected_raw_entries"]
        ) is not None:
            print(f"compiled branch-pruned validation of {limit} entries")
            compiled_event_validation(
                tree,
                cutflow,
                source_contract["expected_raw_entries"],
                args.expected_source_domain_id,
                limit,
            )
            validation_execution = "compiled_branch_pruned_full_scan"
        else:
            seen = set()
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
                    raise RuntimeError(
                        f"duplicate event identity {identity} at entry {entry}"
                    )
                seen.add(identity)
                if hit_tree:
                    hit_tree.GetEntry(entry)
                    hit_identity = (
                        int(hit_tree.runId),
                        int(hit_tree.eventId),
                        int(hit_tree.eventIndex),
                    )
                    if hit_identity != identity:
                        raise RuntimeError(
                            f"feature/Hit3D identity mismatch at entry {entry}"
                        )

                domain_id = int(tree.source_domain_id)
                if domain_id != args.expected_source_domain_id:
                    raise RuntimeError(
                        f"source domain {domain_id} does not match expected "
                        f"{args.expected_source_domain_id} at entry {entry}"
                    )
                validate_source_scalar_values(
                    domain_id,
                    float(tree.source_energy_gev),
                    int(tree.source_pdg_code),
                )
                validate_cluster_closure(tree, entry)
                for name, _ in scalar_feature_dtypes():
                    if not math.isfinite(float(getattr(tree, name))):
                        raise RuntimeError(f"non-finite {name} at entry {entry}")
                if not math.isfinite(float(tree.scifi_qdc_per_hit_direct)):
                    raise RuntimeError(
                        f"non-finite scifi_qdc_per_hit_direct at entry {entry}"
                    )
            validation_execution = "python_full_scan"
        if hit_source:
            hit_source.Close()
    finally:
        source.Close()

    feature_checksum = sha256(args.feature)
    if merge_record:
        if merge_record.get("output_sha256") != feature_checksum:
            raise RuntimeError("feature-shard merge checksum mismatch")

    record = {
        "validation_passed": True,
        "manifest_version": args.manifest_version,
        "unified_schema_version": UNIFIED_SCHEMA_VERSION,
        "scifi_cluster_schema_version": SCHEMA_VERSION,
        "source_uid": args.expected_source_uid,
        "source_domain_id": args.expected_source_domain_id,
        "source_domain": SOURCE_DOMAINS[args.expected_source_domain_id],
        "source_geometry_id": args.expected_source_geometry_id or "",
        "source_geometry_checksum": args.expected_source_geometry_checksum or "",
        "source_pdg_code": getattr(args, "expected_source_pdg_code", None),
        "source_energy_gev": getattr(args, "expected_source_energy_gev", None),
        "input_digi_path": args.input_digi_path,
        "input_cutflow_path": args.input_cutflow_path or "",
        "feature_path": args.published_feature_path or args.feature,
        "feature_size_bytes": os.path.getsize(args.feature),
        "feature_sha256": feature_checksum,
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
        "validation_execution": validation_execution,
        "identity_validation": (
            "source_selection_entrywise_sharded_merge"
            if merge_record
            else (
                "source_selection_and_hit3d_entrywise"
                if args.hit3d
                else "source_selection_entrywise"
            )
        ),
        "converter_commit": args.converter_commit,
    }
    if merge_record:
        record.update({
            "feature_shard_count": int(merge_record["shard_count"]),
            "merge_boundary_event_count": merge_boundary_count,
            "feature_shard_validation_records": merge_record[
                "shard_validation_records"
            ],
            "feature_shard_sha256": merge_record["shard_feature_sha256"],
        })
    if source_contract["range_requested"]:
        record.update({
            "selected_entry_start": source_contract["selected_entry_start"],
            "selected_entry_stop": source_contract["selected_entry_stop"],
        })
    return record


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
    parser.add_argument("--expected-source-pdg-code", type=int)
    parser.add_argument("--expected-source-energy-gev", type=float)
    parser.add_argument("--expected-input-event-count", type=int, required=True)
    parser.add_argument("--expected-selected-event-count", type=int)
    parser.add_argument("--sample-type", required=True)
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, required=True)
    parser.add_argument("--input-tree", default="cbmsim")
    parser.add_argument("--expected-max-events", type=int, default=0)
    parser.add_argument("--expected-selected-entry-start", type=int)
    parser.add_argument("--expected-selected-entry-stop", type=int)
    parser.add_argument("--input-digi-path", required=True)
    parser.add_argument("--input-cutflow-path")
    parser.add_argument("--published-feature-path")
    parser.add_argument("--merge-record")
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
