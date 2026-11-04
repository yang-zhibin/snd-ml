#!/usr/bin/env python3
"""Validate complete or explicitly partial feature-snapshot coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convertData"))
from snapshot_contract import validate_membership, validate_selection_coverage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_source_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if "source_uid" not in (rows.fieldnames or []):
            raise RuntimeError(f"snapshot manifest has no source_uid column: {path}")
        return len({str(row["source_uid"]) for row in rows})


def validate(
    coverage_path: str,
    families: list[str],
    policy: str = "complete",
    minimum_available_sources: int = 1,
    allowed_unavailable_statuses: list[str] | None = None,
) -> dict:
    coverage_file = Path(coverage_path)
    payload = json.loads(coverage_file.read_text(encoding="utf-8"))
    if policy not in {"complete", "partial"}:
        raise ValueError(f"unknown coverage policy {policy}")
    if minimum_available_sources < 1:
        raise ValueError("minimum_available_sources must be positive")
    allowed_missing = set(allowed_unavailable_statuses or ["missing_feature"])
    genuinely_missing = {"missing_feature", "missing_validation_record",
                         "missing_shard_completeness_record", "missing_shard_product"}
    if not allowed_missing <= genuinely_missing:
        raise ValueError("partial coverage cannot allow invalid or failed products")
    records = {str(item["family"]): item for item in payload.get("families", [])}
    if len(records) != len(payload.get("families", [])):
        raise RuntimeError("duplicate coverage family")
    missing = sorted(set(families) - set(records))
    if missing:
        raise RuntimeError(f"coverage metadata is missing families: {missing}")
    failures = []
    checked = []
    for family in families:
        record = records[family]
        available = int(record.get("available_source_count", -1))
        total = int(record.get("total_source_count", -1))
        unavailable = int(record.get("unavailable_source_count", -1))
        status_counts = record.get("status_counts", {})
        snapshot = Path(str(record.get("snapshot_manifest", "")))
        source_manifest = Path(str(record.get("source_manifest", "")))
        expected_source_sha256 = str(record.get("source_manifest_sha256", ""))
        reasons = []
        selection_accounting = None
        try:
            universe = validate_membership(record)
            if family.startswith("real_data_2024_"):
                selection_accounting = validate_selection_coverage(record, universe)
        except (ValueError, KeyError, OSError, TypeError) as error:
            reasons.append(str(error))
        if total < 1 or available < 0 or unavailable != total - available:
            reasons.append(f"available={available}, total={total}, unavailable={unavailable}")
        if available < minimum_available_sources:
            reasons.append(
                f"available={available} is below minimum={minimum_available_sources}"
            )
        available_statuses = {"available", "available_shards"}
        allowed_statuses = available_statuses | (allowed_missing if policy == "partial" else set())
        unexpected = {
            key: value for key, value in status_counts.items()
            if key not in allowed_statuses and int(value)
        }
        if unexpected:
            reasons.append(f"disallowed statuses={unexpected}")
        if sum(int(value) for value in status_counts.values()) != total:
            reasons.append(
                f"status count sum={sum(int(value) for value in status_counts.values())} "
                f"!= total={total}"
            )
        if policy == "complete" and (unavailable != 0 or available != total):
            reasons.append("complete policy requires every source")
        if not snapshot.is_file():
            reasons.append(f"snapshot manifest does not exist: {snapshot}")
            snapshot_count = -1
            snapshot_sha256 = ""
        else:
            snapshot_count = _snapshot_source_count(snapshot)
            snapshot_sha256 = _sha256(snapshot)
            if snapshot_count != available:
                reasons.append(
                    f"snapshot unique sources={snapshot_count}, coverage available={available}"
                )
        if not source_manifest.is_file():
            reasons.append(f"source manifest does not exist: {source_manifest}")
        elif not expected_source_sha256:
            reasons.append("coverage has no source manifest checksum")
        elif _sha256(source_manifest) != expected_source_sha256:
            reasons.append(f"source manifest checksum changed: {source_manifest}")
        fraction = float(record.get("available_fraction", -1.0))
        expected_fraction = available / total if total > 0 else -1.0
        if not math.isclose(fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12):
            reasons.append(
                f"available_fraction={fraction} != calculated={expected_fraction}"
            )
        if reasons:
            failures.append(f"{family}: " + "; ".join(reasons))
        checked.append({
            "family": family,
            "selection_accounting": selection_accounting,
            "available_source_count": available,
            "total_source_count": total,
            "unavailable_source_count": unavailable,
            "available_fraction": expected_fraction,
            "status_counts": status_counts,
            "snapshot_manifest": str(snapshot),
            "snapshot_manifest_sha256": snapshot_sha256,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": expected_source_sha256,
        })
    if failures:
        raise RuntimeError("invalid feature snapshot coverage:\n" + "\n".join(failures))
    return {
        "validation_passed": True,
        "coverage_policy": policy,
        "minimum_available_sources": minimum_available_sources,
        "allowed_unavailable_statuses": sorted(allowed_missing),
        "coverage": str(coverage_file),
        "coverage_sha256": _sha256(coverage_file),
        "families": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--policy", choices=("complete", "partial"), default="complete")
    parser.add_argument("--minimum-available-sources", type=int, default=1)
    parser.add_argument("--allowed-unavailable-status", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(
        args.coverage,
        args.family,
        policy=args.policy,
        minimum_available_sources=args.minimum_available_sources,
        allowed_unavailable_statuses=args.allowed_unavailable_status or None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
