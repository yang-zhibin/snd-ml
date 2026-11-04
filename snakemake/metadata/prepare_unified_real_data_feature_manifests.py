#!/usr/bin/env python3
"""Freeze validated 2024 real-data selections into two feature manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def _clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    profiles = config.get("profiles", {})
    if set(profiles) != {"no_veto", "veto_10pct"}:
        raise ValueError("real-data distribution config must define no_veto and veto_10pct")
    return config


def _source_metadata_rows(selection: pd.DataFrame) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for metadata_path, group in selection.groupby("source_metadata_file", sort=False):
        source = pd.read_csv(str(metadata_path), keep_default_na=False)
        for source_row in group["source_metadata_row"].astype(int).unique():
            if source_row < 0 or source_row >= len(source):
                raise IndexError(f"source metadata row {source_row} is outside {metadata_path}")
            rows[(str(metadata_path), int(source_row))] = source.iloc[source_row].to_dict()
    return rows


def _inspect_selection(row: dict, profiles: dict, feature_root: str) -> dict:
    uid = str(row["source_uid"])
    validation_path = Path(str(row["validation_record"]))
    summary_path = Path(str(row["selection_summary_path"]))
    if not validation_path.is_file():
        return {"source_uid": uid, "status": "missing_selection_validation", "rows": {}}
    if not summary_path.is_file():
        return {"source_uid": uid, "status": "missing_selection_summary", "rows": {}}
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "source_uid": uid,
            "status": "invalid_selection_metadata",
            "detail": str(error),
            "rows": {},
        }
    if validation.get("status") != "passed":
        return {"source_uid": uid, "status": "failed_selection_validation", "rows": {}}
    if str(validation.get("source_uid", "")) != uid:
        return {"source_uid": uid, "status": "selection_source_uid_mismatch", "rows": {}}
    if str(summary.get("source_uid", "")) != uid:
        return {"source_uid": uid, "status": "summary_source_uid_mismatch", "rows": {}}
    if not bool(summary.get("count_conservation_passed", False)):
        return {"source_uid": uid, "status": "selection_count_conservation_failed", "rows": {}}
    if int(validation.get("input_event_count", -1)) != int(row["input_event_count"]):
        return {"source_uid": uid, "status": "selection_input_count_mismatch", "rows": {}}

    profile_rows = {}
    profile_counts = {}
    for profile, settings in profiles.items():
        count_field = str(settings["count_field"])
        checksum_field = str(settings["checksum_field"])
        index_path = Path(str(row[str(settings["index_column"])]))
        selected_count = int(validation.get(count_field, -1))
        if selected_count < 0 or selected_count != int(summary.get(count_field, -2)):
            return {
                "source_uid": uid,
                "status": f"{profile}_selected_count_mismatch",
                "rows": {},
            }
        if not index_path.is_file():
            return {"source_uid": uid, "status": f"{profile}_missing_index", "rows": {}}
        expected_checksum = str(validation.get(checksum_field, ""))
        if not expected_checksum or _sha256(index_path) != expected_checksum:
            return {
                "source_uid": uid,
                "status": f"{profile}_index_checksum_mismatch",
                "rows": {},
            }
        profile_counts[profile] = selected_count
        if selected_count == 0:
            profile_rows[profile] = None
            continue
        family = str(settings["family"])
        feature = (
            Path(feature_root)
            / profile
            / uid
            / f"feature_scifi_cluster_v1_{uid}.root"
        )
        profile_rows[profile] = {
            "manifest_version": "",
            "family": family,
            "source_uid": uid,
            "source_metadata_file": str(row["source_metadata_file"]),
            "source_metadata_row": int(row["source_metadata_row"]),
            "digi_path": str(row["digi_path"]),
            "geo_path": str(row["geo_path"]),
            "cutflow_path": str(index_path),
            "input_tree": str(row["input_tree"]),
            "sample_type": "real_data",
            "selection_mode": "indexed_cutflow",
            "selection_profile": family,
            "source_domain_id": 3,
            "source_domain": "real_data_2024_skim",
            "source_pdg_code": 0,
            "source_energy_gev": -1.0,
            "source_geometry_id": "",
            "source_geometry_checksum": "",
            "input_event_count": int(row["input_event_count"]),
            "selected_event_count": selected_count,
            "sampling_probability": float(settings["sampling_probability"]),
            "event_weight": float(settings["event_weight"]),
            "selection_index_sha256": expected_checksum,
            "selection_validation_record": str(validation_path),
            "lumi_per_file": 0.0,
            "source_lumi_per_file": _clean(row.get("lumi_per_file", "")),
            "exposure_kind": "event_count_only",
            "canonical_label": "real_data",
            "feature_scifi_cluster_v1_path": str(feature),
            "validation_record": str(feature) + ".validated.json",
            "validation_marker": str(feature) + ".validated.json",
            "is_canary": 0,
        }
    return {"source_uid": uid, "status": "available", "rows": profile_rows, "selected_counts": profile_counts}


def build_manifests(
    config_path: str | Path,
    workers: int = 16,
) -> tuple[dict[str, pd.DataFrame], dict]:
    config = load_config(config_path)
    selection_path = Path(str(config["inputs"]["selection_manifest"]))
    selection = pd.read_csv(selection_path, keep_default_na=False)
    required = {
        "source_uid",
        "source_metadata_file",
        "source_metadata_row",
        "digi_path",
        "input_tree",
        "input_event_count",
        "lumi_per_file",
        "no_veto_index_path",
        "veto_10pct_index_path",
        "selection_summary_path",
        "validation_record",
    }
    missing = required - set(selection.columns)
    if missing:
        raise KeyError(f"selection manifest is missing columns: {sorted(missing)}")
    if selection["source_uid"].duplicated().any():
        raise ValueError("selection manifest contains duplicate source UIDs")

    policy_path = Path(str(config["inputs"]["selection_policy"]))
    policy = yaml.safe_load(policy_path.read_text())
    exclusions = policy.get("excluded_source_uids", [])
    if len(set(exclusions)) != len(exclusions) or not set(exclusions) <= set(selection["source_uid"].astype(str)):
        raise ValueError("duplicate or unknown exclusion UID")
    reason = policy.get("exclusion_reason", "")
    if exclusions and not reason:
        raise ValueError("selection exclusions require an explicit reason")
    metadata_rows = _source_metadata_rows(selection)
    records = []
    for row in selection.to_dict("records"):
        source = metadata_rows[(str(row["source_metadata_file"]), int(row["source_metadata_row"]))]
        geo_path = _clean(source.get("geo_path", ""))
        if not geo_path:
            raise ValueError(f"missing geo_path for {row['source_uid']}")
        row["geo_path"] = geo_path
        records.append(row)

    profiles = config["profiles"]
    feature_root = str(config["outputs"]["feature_root"]).rstrip("/")
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        checks = list(
            executor.map(
                lambda row: ({"source_uid": str(row["source_uid"]), "status": "excluded",
                              "reason": reason, "rows": {}}
                             if str(row["source_uid"]) in exclusions else
                             _inspect_selection(row, profiles, feature_root)),
                records,
            )
        )

    version = str(config["version"])
    manifests = {}
    for profile, settings in profiles.items():
        family = str(settings["family"])
        rows = [
            check["rows"][profile]
            for check in checks
            if check["status"] == "available" and check["rows"].get(profile) is not None
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"no non-empty validated selections are available for {profile}")
        frame["manifest_version"] = version
        frame = frame.sort_values("source_uid", kind="mergesort").reset_index(drop=True)
        if frame["source_uid"].duplicated().any():
            raise ValueError(f"duplicate source UID in {profile} manifest")
        # Pick the smallest non-empty selection so that the local smoke test is
        # representative but intentionally cheap.
        canary_index = pd.to_numeric(frame["selected_event_count"]).idxmin()
        frame.loc[canary_index, "is_canary"] = 1
        manifests[family] = frame

    status_counts = pd.Series([check["status"] for check in checks]).value_counts()
    coverage = {
        "coverage_schema_version": 2,
        "version": version,
        "selection_policy": str(policy_path),
        "selection_policy_sha256": _sha256(policy_path),
        "source_statuses": [{k: v for k, v in check.items() if k != "rows"} for check in checks],
        "eligible_source_count": len(selection) - len(exclusions),
        "excluded_source_count": len(exclusions),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "total_source_count": int(len(selection)),
        "validated_source_count": int(sum(check["status"] == "available" for check in checks)),
        "status_counts": {str(k): int(v) for k, v in sorted(status_counts.items())},
        "families": {
            str(settings["family"]): {
                "profile": profile,
                "nonempty_source_count": int(len(manifests[str(settings["family"])])),
                "selected_event_count": int(
                    manifests[str(settings["family"])]["selected_event_count"].sum()
                ),
                "sampling_probability": float(settings["sampling_probability"]),
                "event_weight": float(settings["event_weight"]),
            }
            for profile, settings in profiles.items()
        },
    }
    return manifests, coverage


def write_outputs(
    manifests: dict[str, pd.DataFrame],
    coverage: dict,
    config: dict,
    coverage_json: str | Path,
    coverage_csv: str | Path,
) -> None:
    for family, frame in manifests.items():
        output = Path(str(config["source_manifests"][family]))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(output)

    json_path = Path(coverage_json)
    csv_path = Path(coverage_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_tmp.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for family, values in coverage["families"].items():
        rows.append({"family": family, **values})
    pd.DataFrame(rows).to_csv(csv_tmp, index=False, lineterminator="\n")
    json_tmp.replace(json_path)
    csv_tmp.replace(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--coverage-csv", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config = load_config(args.config)
    manifests, coverage = build_manifests(args.config, args.workers)
    write_outputs(manifests, coverage, config, args.coverage_json, args.coverage_csv)
    for family, frame in manifests.items():
        print(
            f"{family}: froze {len(frame)} non-empty sources with "
            f"{int(frame['selected_event_count'].sum())} selected events"
        )


if __name__ == "__main__":
    main()
