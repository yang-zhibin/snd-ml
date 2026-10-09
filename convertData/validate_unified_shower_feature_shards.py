#!/usr/bin/env python3
"""Validate a complete shard set without merging its ROOT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merge_unified_shower_feature_shards import (
    COMMON_RECEIPT_FIELDS,
    load_receipts,
    validate_receipts,
    write_json,
)


def validate_complete_shards(args: argparse.Namespace) -> dict:
    receipts = validate_receipts(load_receipts(args.shard_record))
    reference = receipts[0]
    if str(reference["source_uid"]) != str(args.expected_source_uid):
        raise RuntimeError(
            f"shard source UID {reference['source_uid']} != expected "
            f"{args.expected_source_uid}"
        )
    selected_count = int(reference["available_selected_event_count"])
    if selected_count != int(args.expected_selected_event_count):
        raise RuntimeError(
            f"selected event count {selected_count} != expected "
            f"{args.expected_selected_event_count}"
        )
    if str(reference.get("input_digi_path", "")) != str(
        args.expected_input_digi_path
    ):
        raise RuntimeError("shard input digi path does not match the manifest")
    if str(reference.get("input_cutflow_path", "")) != str(
        args.expected_input_cutflow_path
    ):
        raise RuntimeError("shard input cutflow path does not match the manifest")

    report = {
        "validation_passed": True,
        "record_type": "unified_shower_feature_shards_complete_v1",
        "family": str(args.family),
        "source_uid": str(reference["source_uid"]),
        "selected_event_count": selected_count,
        "shard_count": len(receipts),
        "shards": [
            {
                "shard_index": index,
                "selected_entry_start": int(receipt["selected_entry_start"]),
                "selected_entry_stop": int(receipt["selected_entry_stop"]),
                "written_event_count": int(receipt["written_event_count"]),
                "feature_path": str(receipt["feature_path"]),
                "feature_sha256": str(receipt["feature_sha256"]),
                "validation_record": str(receipt["validation_record"]),
            }
            for index, receipt in enumerate(receipts)
        ],
    }
    report.update({field: reference.get(field) for field in COMMON_RECEIPT_FIELDS})
    write_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-record", action="append", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--expected-source-uid", required=True)
    parser.add_argument("--expected-selected-event-count", type=int, required=True)
    parser.add_argument("--expected-input-digi-path", required=True)
    parser.add_argument("--expected-input-cutflow-path", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    report = validate_complete_shards(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
