#!/usr/bin/env python3
"""Materialize no-Veto and deterministic veto-10% indexes for one 2024 file."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
from pathlib import Path

import ROOT

from real_data_2024_selection import (
    NO_VETO_PROFILE,
    SELECTION_PROFILE_VERSION,
    VETO_10PCT_PROFILE,
    VETO_SAMPLE_BUCKET,
    VETO_SAMPLE_DENOMINATOR,
    deterministic_sample_bucket,
    event_header_identity,
    has_valid_veto_hit,
)
from sample_selection import cutflow_selection_expression, setup_event_deltat_alias


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporary_path(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.tmp.{os.getpid()}")


def open_root(path: str, tree_name: str):
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot open ROOT file {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"missing tree {tree_name!r} in {path}")
    return root_file, tree


class IndexWriter:
    def __init__(self, destination: str, profile: str, source_uid: str):
        self.destination = Path(destination)
        self.temporary = temporary_path(destination)
        self.root_file = ROOT.TFile(str(self.temporary), "RECREATE")
        if not self.root_file or self.root_file.IsZombie():
            raise RuntimeError(f"cannot create {self.temporary}")
        self.tree = ROOT.TTree("cutFlowSummary", profile)
        self.entry = array.array("q", [0])
        self.run_id = array.array("i", [0])
        self.event_id = array.array("i", [0])
        self.veto_present = array.array("i", [0])
        self.sample_bucket = array.array("i", [-1])
        self.tree.Branch("entry", self.entry, "entry/L")
        self.tree.Branch("runId", self.run_id, "runId/I")
        self.tree.Branch("eventId", self.event_id, "eventId/I")
        self.tree.Branch("veto_present", self.veto_present, "veto_present/I")
        self.tree.Branch("sample_bucket", self.sample_bucket, "sample_bucket/I")
        self.profile = profile
        self.source_uid = source_uid

    def fill(self, identity: tuple[int, int, int], veto_present: bool, bucket: int):
        run_id, event_id, event_index = identity
        self.entry[0] = event_index
        self.run_id[0] = run_id
        self.event_id[0] = event_id
        self.veto_present[0] = int(veto_present)
        self.sample_bucket[0] = int(bucket)
        self.tree.Fill()

    def close_and_publish(self, input_entries: int, base_selected_entries: int) -> dict:
        self.root_file.cd()
        self.tree.Write()
        ROOT.TNamed("selection_profile", self.profile).Write()
        ROOT.TNamed("selection_profile_version", SELECTION_PROFILE_VERSION).Write()
        ROOT.TNamed("source_uid", self.source_uid).Write()
        ROOT.TNamed("sampling_hash", "sha256-first64-big-endian").Write()
        ROOT.TParameter("Long64_t")("input_event_count", int(input_entries)).Write()
        ROOT.TParameter("Long64_t")(
            "base_selected_event_count", int(base_selected_entries)
        ).Write()
        ROOT.TParameter("Long64_t")(
            "selected_event_count", int(self.tree.GetEntries())
        ).Write()
        ROOT.TParameter("int")(
            "sampling_denominator", int(VETO_SAMPLE_DENOMINATOR)
        ).Write()
        ROOT.TParameter("int")("sampling_bucket", int(VETO_SAMPLE_BUCKET)).Write()
        selected_entries = int(self.tree.GetEntries())
        self.root_file.Close()
        os.replace(self.temporary, self.destination)
        return {
            "path": str(self.destination),
            "sha256": sha256_file(self.destination),
            "selected_event_count": selected_entries,
        }


def build_indices(args: argparse.Namespace) -> dict:
    digi_file, digi_tree = open_root(args.digi_path, args.input_tree)
    cutflow_file, cutflow_tree = open_root(args.base_cutflow_path, "cutFlowSummary")
    no_veto_writer = IndexWriter(args.no_veto_output, NO_VETO_PROFILE, args.source_uid)
    veto_writer = IndexWriter(
        args.veto_10pct_output, VETO_10PCT_PROFILE, args.source_uid
    )
    published = False
    try:
        event_deltat_branch = setup_event_deltat_alias(cutflow_tree)
        selection = cutflow_selection_expression(
            "real_data", "external_cutflow", bool(event_deltat_branch)
        )
        list_name = f"real_data_2024_base_{os.getpid()}"
        cutflow_tree.Draw(f">>{list_name}", selection, "entrylist")
        selected = ROOT.gDirectory.Get(list_name)
        if not selected or not isinstance(selected, ROOT.TEntryList):
            raise RuntimeError("failed to build the base real-data entry list")

        input_entries = int(digi_tree.GetEntries())
        available_base_entries = int(selected.GetN())
        base_limit = int(args.max_base_events)
        if base_limit < 0:
            raise ValueError("--max-base-events must be non-negative")
        processed_base_entries = (
            min(available_base_entries, base_limit)
            if base_limit
            else available_base_entries
        )

        has_entry_branch = bool(
            cutflow_tree.GetBranch("entry") or cutflow_tree.GetLeaf("entry")
        )
        previous_raw_entry = -1
        all_veto_count = 0
        for ordinal in range(processed_base_entries):
            cutflow_row = int(selected.GetEntry(ordinal))
            cutflow_tree.GetEntry(cutflow_row)
            raw_entry = int(cutflow_tree.entry) if has_entry_branch else cutflow_row
            if raw_entry <= previous_raw_entry:
                raise RuntimeError(
                    "base cutflow raw entries must be unique and strictly increasing"
                )
            if raw_entry < 0 or raw_entry >= input_entries:
                raise RuntimeError(
                    f"base cutflow entry {raw_entry} is outside [0,{input_entries})"
                )
            previous_raw_entry = raw_entry
            if digi_tree.GetEntry(raw_entry) <= 0:
                raise RuntimeError(f"failed to read digi entry {raw_entry}")
            identity = event_header_identity(digi_tree.EventHeader, raw_entry)
            veto_present = has_valid_veto_hit(digi_tree.Digi_MuFilterHits)
            if not veto_present:
                no_veto_writer.fill(identity, False, -1)
                continue
            all_veto_count += 1
            bucket = deterministic_sample_bucket(*identity)
            if bucket == VETO_SAMPLE_BUCKET:
                veto_writer.fill(identity, True, bucket)

        no_veto = no_veto_writer.close_and_publish(
            input_entries, processed_base_entries
        )
        veto_10pct = veto_writer.close_and_publish(
            input_entries, processed_base_entries
        )
        published = True
        summary = {
            "selection_profile_version": SELECTION_PROFILE_VERSION,
            "source_uid": args.source_uid,
            "digi_path": args.digi_path,
            "base_cutflow_path": args.base_cutflow_path,
            "input_tree": args.input_tree,
            "selection_expression": selection,
            "input_event_count": input_entries,
            "available_base_selected_event_count": available_base_entries,
            "base_selected_event_count": processed_base_entries,
            "max_base_events": base_limit,
            "no_veto_event_count": no_veto["selected_event_count"],
            "all_veto_event_count": all_veto_count,
            "veto_10pct_event_count": veto_10pct["selected_event_count"],
            "count_conservation_passed": (
                no_veto["selected_event_count"] + all_veto_count
                == processed_base_entries
            ),
            "sampling": {
                "algorithm": "sha256-first64-big-endian",
                "identity": "runId:eventId:eventIndex",
                "denominator": VETO_SAMPLE_DENOMINATOR,
                "accepted_bucket": VETO_SAMPLE_BUCKET,
                "fraction": 1.0 / VETO_SAMPLE_DENOMINATOR,
                "weight": float(VETO_SAMPLE_DENOMINATOR),
            },
            "outputs": {
                NO_VETO_PROFILE: no_veto,
                VETO_10PCT_PROFILE: veto_10pct,
            },
        }
        if not summary["count_conservation_passed"]:
            raise RuntimeError("no-veto/veto count conservation failed")
        summary_path = Path(args.summary_output)
        summary_tmp = temporary_path(summary_path)
        with open(summary_tmp, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(summary_tmp, summary_path)
        return summary
    finally:
        cutflow_file.Close()
        digi_file.Close()
        if not published:
            for writer in (no_veto_writer, veto_writer):
                try:
                    writer.root_file.Close()
                except Exception:
                    pass
                writer.temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digi-path", required=True)
    parser.add_argument("--base-cutflow-path", required=True)
    parser.add_argument("--input-tree", default="cbmsim")
    parser.add_argument("--source-uid", required=True)
    parser.add_argument("--no-veto-output", required=True)
    parser.add_argument("--veto-10pct-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--max-base-events",
        type=int,
        default=0,
        help="Smoke-test limit; zero processes every base-selected event.",
    )
    args = parser.parse_args()
    summary = build_indices(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
