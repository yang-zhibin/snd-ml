#!/usr/bin/env python3
"""Independently validate one pair of 2024 real-data selection indexes."""

from __future__ import annotations

import argparse
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_title(root_file, name: str) -> str:
    value = root_file.Get(name)
    if not value:
        raise RuntimeError(f"missing ROOT metadata {name}")
    return str(value.GetTitle())


def metadata_integer(root_file, name: str) -> int:
    value = root_file.Get(name)
    if not value:
        raise RuntimeError(f"missing ROOT metadata {name}")
    return int(value.GetVal())


def read_and_validate_index(
    path: str,
    expected_profile: str,
    expected_source_uid: str,
    input_entries: int,
) -> list[int]:
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot open selection index {path}")
    try:
        tree = root_file.Get("cutFlowSummary")
        if not tree:
            raise RuntimeError(f"missing cutFlowSummary in {path}")
        if metadata_title(root_file, "selection_profile") != expected_profile:
            raise RuntimeError(f"selection profile mismatch in {path}")
        if (
            metadata_title(root_file, "selection_profile_version")
            != SELECTION_PROFILE_VERSION
        ):
            raise RuntimeError(f"selection profile version mismatch in {path}")
        if metadata_title(root_file, "source_uid") != expected_source_uid:
            raise RuntimeError(f"source UID mismatch in {path}")
        if metadata_integer(root_file, "input_event_count") != input_entries:
            raise RuntimeError(f"input count mismatch in {path}")
        if metadata_integer(root_file, "selected_event_count") != int(tree.GetEntries()):
            raise RuntimeError(f"selected count metadata mismatch in {path}")

        entries = []
        previous = -1
        for row in tree:
            entry = int(row.entry)
            if entry <= previous:
                raise RuntimeError(f"entries are not unique and increasing in {path}")
            if entry < 0 or entry >= input_entries:
                raise RuntimeError(f"entry {entry} is outside the digi range in {path}")
            entries.append(entry)
            previous = entry
        return entries
    finally:
        root_file.Close()


def validate(args: argparse.Namespace) -> dict:
    with open(args.summary, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("selection_profile_version") != SELECTION_PROFILE_VERSION:
        raise RuntimeError("summary selection profile version mismatch")
    if summary.get("source_uid") != args.source_uid:
        raise RuntimeError("summary source UID mismatch")

    digi = ROOT.TFile.Open(args.digi_path)
    if not digi or digi.IsZombie():
        raise RuntimeError(f"cannot open digi input {args.digi_path}")
    try:
        tree = digi.Get(args.input_tree)
        if not tree:
            raise RuntimeError(f"missing input tree {args.input_tree}")
        input_entries = int(tree.GetEntries())
        no_veto_entries = read_and_validate_index(
            args.no_veto_index, NO_VETO_PROFILE, args.source_uid, input_entries
        )
        veto_entries = read_and_validate_index(
            args.veto_10pct_index,
            VETO_10PCT_PROFILE,
            args.source_uid,
            input_entries,
        )
        if set(no_veto_entries).intersection(veto_entries):
            raise RuntimeError("no-veto and veto-10% indexes overlap")

        for entry in no_veto_entries:
            if tree.GetEntry(entry) <= 0:
                raise RuntimeError(f"failed to read no-veto digi entry {entry}")
            if has_valid_veto_hit(tree.Digi_MuFilterHits):
                raise RuntimeError(f"no-veto index contains veto entry {entry}")

        for entry in veto_entries:
            if tree.GetEntry(entry) <= 0:
                raise RuntimeError(f"failed to read veto digi entry {entry}")
            if not has_valid_veto_hit(tree.Digi_MuFilterHits):
                raise RuntimeError(f"veto index contains no-veto entry {entry}")
            identity = event_header_identity(tree.EventHeader, entry)
            if deterministic_sample_bucket(*identity) != VETO_SAMPLE_BUCKET:
                raise RuntimeError(f"veto index contains rejected hash bucket at {entry}")
    finally:
        digi.Close()

    expected_no_veto = int(summary["no_veto_event_count"])
    expected_veto = int(summary["veto_10pct_event_count"])
    if len(no_veto_entries) != expected_no_veto:
        raise RuntimeError("no-veto count differs from summary")
    if len(veto_entries) != expected_veto:
        raise RuntimeError("veto-10% count differs from summary")
    for profile, path in (
        (NO_VETO_PROFILE, args.no_veto_index),
        (VETO_10PCT_PROFILE, args.veto_10pct_index),
    ):
        expected_hash = summary["outputs"][profile]["sha256"]
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {profile}")

    if summary["sampling"]["denominator"] != VETO_SAMPLE_DENOMINATOR:
        raise RuntimeError("sampling denominator mismatch")
    if summary["sampling"]["accepted_bucket"] != VETO_SAMPLE_BUCKET:
        raise RuntimeError("sampling bucket mismatch")

    return {
        "validation_version": "real_data_2024_selection_validation_v1",
        "selection_profile_version": SELECTION_PROFILE_VERSION,
        "source_uid": args.source_uid,
        "input_event_count": input_entries,
        "no_veto_event_count": len(no_veto_entries),
        "veto_10pct_event_count": len(veto_entries),
        "no_veto_sha256": sha256_file(args.no_veto_index),
        "veto_10pct_sha256": sha256_file(args.veto_10pct_index),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digi-path", required=True)
    parser.add_argument("--input-tree", default="cbmsim")
    parser.add_argument("--source-uid", required=True)
    parser.add_argument("--no-veto-index", required=True)
    parser.add_argument("--veto-10pct-index", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--validation-record", required=True)
    args = parser.parse_args()
    record = validate(args)
    destination = Path(args.validation_record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
