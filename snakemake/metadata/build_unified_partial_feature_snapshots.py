#!/usr/bin/env python3
"""Freeze the currently validated unified feature files for partial plots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    families = config.get("source_manifests", {})
    if not families:
        raise ValueError("partial-distribution config has no source_manifests")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_unit(row: dict) -> tuple[str, str, list[dict]]:
    feature = Path(str(row["feature_scifi_cluster_v1_path"]))
    marker = Path(str(row["validation_record"]))
    if not feature.is_file():
        return "missing_feature", str(feature), []
    if not marker.is_file():
        return "missing_validation_record", str(marker), []
    try:
        with marker.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return "invalid_validation_record", f"{marker}: {error}", []
    if record.get("validation_passed") is not True:
        return "failed_validation", str(marker), []
    if str(record.get("source_uid", "")) != str(row["source_uid"]):
        return "source_uid_mismatch", str(marker), []
    if str(record.get("feature_path", "")) != str(feature):
        return "feature_path_mismatch", str(marker), []
    unit = dict(row)
    unit.update(
        {
            "feature_unit_uid": str(row["source_uid"]),
            "feature_layout": "canonical",
            "shard_index": -1,
            "selected_entry_start": 0,
            "selected_entry_stop": int(float(row.get("selected_event_count", 0) or 0)),
            "shard_completeness_record": "",
        }
    )
    return "available", "", [unit]


def _shard_units(row: dict, shard_root: str) -> tuple[str, str, list[dict]]:
    source_uid = str(row["source_uid"])
    complete = Path(shard_root) / source_uid / "shards.complete.json"
    if not complete.is_file():
        return "missing_shard_completeness_record", str(complete), []
    try:
        with complete.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return "invalid_shard_completeness_record", f"{complete}: {error}", []
    if record.get("validation_passed") is not True:
        return "failed_shard_completeness", str(complete), []
    if str(record.get("source_uid", "")) != source_uid:
        return "shard_source_uid_mismatch", str(complete), []
    expected_count = str(row.get("selected_event_count", ""))
    if expected_count and int(record.get("selected_event_count", -1)) != int(
        float(expected_count)
    ):
        return "shard_selected_count_mismatch", str(complete), []
    shards = list(record.get("shards", []))
    if not shards:
        return "empty_shard_completeness_record", str(complete), []

    units = []
    expected_start = 0
    for index, shard in enumerate(shards):
        feature = Path(str(shard.get("feature_path", "")))
        marker = Path(str(shard.get("validation_record", "")))
        start = int(shard.get("selected_entry_start", -1))
        stop = int(shard.get("selected_entry_stop", -1))
        if int(shard.get("shard_index", -1)) != index or start != expected_start:
            return "invalid_shard_order", str(complete), []
        if stop <= start or int(shard.get("written_event_count", -1)) != stop - start:
            return "invalid_shard_range", str(complete), []
        if not feature.is_file() or not marker.is_file():
            return "missing_shard_product", str(feature), []
        try:
            with marker.open("r", encoding="utf-8") as handle:
                validation = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            return "invalid_shard_validation_record", f"{marker}: {error}", []
        if validation.get("validation_passed") is not True:
            return "failed_shard_validation", str(marker), []
        if str(validation.get("source_uid", "")) != source_uid:
            return "shard_source_uid_mismatch", str(marker), []
        if str(validation.get("feature_path", "")) != str(feature):
            return "shard_feature_path_mismatch", str(marker), []
        unit = dict(row)
        unit.update(
            {
                "feature_unit_uid": f"{source_uid}:part_{index:04d}",
                "feature_layout": "shard",
                "shard_index": index,
                "selected_entry_start": start,
                "selected_entry_stop": stop,
                "feature_scifi_cluster_v1_path": str(feature),
                "validation_record": str(marker),
                "shard_completeness_record": str(complete),
            }
        )
        units.append(unit)
        expected_start = stop
    if expected_start != int(record.get("selected_event_count", -1)):
        return "incomplete_shard_coverage", str(complete), []
    return "available_shards", "", units


def _check_row(row: dict, shard_root: str = "") -> tuple[str, str, list[dict]]:
    canonical = _canonical_unit(row)
    if canonical[0] == "available" or not shard_root:
        return canonical
    shards = _shard_units(row, shard_root)
    return shards if shards[0] == "available_shards" else canonical


def _exposure_sum(frame: pd.DataFrame) -> float:
    sources = frame.drop_duplicates("source_uid", keep="first")
    values = pd.to_numeric(sources["lumi_per_file"], errors="coerce")
    return float(values.dropna().sum())


def build_snapshots(
    config_path: str | Path,
    output_root: str | Path,
    workers: int = 16,
) -> list[dict]:
    config = load_config(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    coverage = []

    for family, manifest_value in config["source_manifests"].items():
        manifest = Path(str(manifest_value))
        frame = pd.read_csv(manifest, keep_default_na=False)
        required = {
            "family",
            "source_uid",
            "feature_scifi_cluster_v1_path",
            "validation_record",
            "lumi_per_file",
            "exposure_kind",
        }
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{manifest} is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["family"].astype(str).eq(str(family))].copy()
        if frame.empty:
            raise ValueError(f"{manifest} contains no rows for family {family}")
        if frame["source_uid"].duplicated().any():
            raise ValueError(f"{manifest} contains duplicate source UIDs for {family}")

        records = frame.to_dict("records")
        shard_root = str(config.get("shard_receipt_roots", {}).get(family, ""))
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            checks = list(executor.map(lambda row: _check_row(row, shard_root), records))
        statuses = pd.Series([status for status, _, _ in checks], index=frame.index)
        available_records = [unit for _, _, units in checks for unit in units]
        available = pd.DataFrame(available_records)
        if not available.empty:
            available = available.sort_values(
                ["source_uid", "selected_entry_start"], kind="mergesort"
            ).reset_index(drop=True)
        if available.empty:
            raise RuntimeError(f"no validated feature files are available for {family}")
        if available["feature_unit_uid"].duplicated().any():
            raise RuntimeError(f"snapshot contains duplicate feature units for {family}")

        snapshot_path = output_root / f"{family}.csv"
        temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
        available.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(snapshot_path)

        status_counts = statuses.value_counts().to_dict()
        row = {
            "snapshot_version": str(config["version"]),
            "family": str(family),
            "source_manifest": str(manifest),
            "source_manifest_sha256": _sha256(manifest),
            "snapshot_manifest": str(snapshot_path),
            "total_source_count": int(len(frame)),
            "available_source_count": int(available["source_uid"].nunique()),
            "available_feature_file_count": int(len(available)),
            "unavailable_source_count": int(
                len(frame) - available["source_uid"].nunique()
            ),
            "available_fraction": float(
                available["source_uid"].nunique() / len(frame)
            ),
            "available_exposure": _exposure_sum(available),
            "exposure_kinds": sorted(set(available["exposure_kind"].astype(str))),
            "status_counts": {
                str(key): int(value) for key, value in sorted(status_counts.items())
            },
        }
        coverage.append(row)
        print(
            f"{family}: froze {available['source_uid'].nunique()}/{len(frame)} sources "
            f"in {len(available)} validated feature files "
            f"in {snapshot_path}"
        )
    return coverage


def write_coverage(
    coverage: list[dict], json_path: str | Path, csv_path: str | Path
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_tmp.write_text(
        json.dumps({"families": coverage}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat = []
    for row in coverage:
        item = dict(row)
        item["exposure_kinds"] = "|".join(item["exposure_kinds"])
        item["status_counts"] = json.dumps(item["status_counts"], sort_keys=True)
        flat.append(item)
    pd.DataFrame(flat).to_csv(csv_tmp, index=False, lineterminator="\n")
    json_tmp.replace(json_path)
    csv_tmp.replace(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--coverage-csv", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = build_snapshots(args.config, args.output_root, args.workers)
    write_coverage(coverage, args.coverage_json, args.coverage_csv)


if __name__ == "__main__":
    main()
