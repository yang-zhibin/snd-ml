"""Membership checks for versioned feature snapshots (no ROOT dependency)."""
import csv
from collections import Counter
from pathlib import Path
from feature_integrity import sha256


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_membership(record):
    if record.get("coverage_schema_version") != 2:
        raise ValueError("coverage schema is obsolete; refresh validated snapshots")
    snapshot = record["snapshot_manifest"]
    source = record["source_manifest"]
    if sha256(snapshot) != record.get("snapshot_manifest_sha256"):
        raise ValueError("frozen snapshot checksum mismatch")
    if sha256(source) != record.get("source_manifest_sha256"):
        raise ValueError("source manifest checksum mismatch")
    sources = read_rows(source)
    universe = {row["source_uid"]: row for row in sources}
    if len(universe) != len(sources) or any(row["family"] != record["family"] for row in sources):
        raise ValueError("duplicate source UID or wrong source family")
    statuses = record.get("source_statuses", {})
    if set(statuses) != set(universe):
        raise ValueError("source status membership mismatch")
    if dict(Counter(statuses.values())) != record["status_counts"]:
        raise ValueError("source status totals mismatch")
    expected = {uid for uid, status in statuses.items() if status in {"available", "available_shards"}}
    rows = read_rows(snapshot)
    if {row["source_uid"] for row in rows} != expected:
        raise ValueError("snapshot available membership mismatch")
    if len(universe) != record["total_source_count"] or len(expected) != record["available_source_count"]:
        raise ValueError("source membership counts mismatch")
    if len(rows) != record["available_feature_file_count"]:
        raise ValueError("feature unit count mismatch")
    for key in ("feature_unit_uid", "feature_scifi_cluster_v1_path"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {key}")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_uid"], []).append(row)
    for uid in expected:
        units = grouped[uid]
        original = universe[uid]
        for row in units:
            for key in original:
                if key not in {"feature_scifi_cluster_v1_path", "validation_record", "validation_marker"}:
                    if row.get(key) != original[key]:
                        raise ValueError(f"source provenance mismatch: {uid} {key}")
        layouts = {row["feature_layout"] for row in units}
        if layouts == {"canonical"}:
            unit = units[0]
            if statuses[uid] != "available":
                raise ValueError("canonical unit has inconsistent availability status")
            if int(unit["shard_index"]) != -1 or int(unit["selected_entry_start"]) != 0:
                raise ValueError("invalid canonical selected-entry range")
            if original.get("selected_event_count", "") != "":
                if int(unit["selected_entry_stop"]) != int(original["selected_event_count"]):
                    raise ValueError("incomplete canonical selected-entry coverage")
            if len(units) != 1 or units[0]["feature_scifi_cluster_v1_path"] != original["feature_scifi_cluster_v1_path"] or units[0]["validation_record"] != original["validation_record"]:
                raise ValueError("invalid canonical unit")
        elif layouts == {"shard"}:
            if statuses[uid] != "available_shards":
                raise ValueError("sharded unit has inconsistent availability status")
            stop = 0
            for index, row in enumerate(sorted(units, key=lambda row: int(row["shard_index"]))):
                start, end = int(row["selected_entry_start"]), int(row["selected_entry_stop"])
                if int(row["shard_index"]) != index or start != stop or end <= start:
                    raise ValueError("invalid shard coverage")
                stop = end
            if stop != int(original["selected_event_count"]):
                raise ValueError("incomplete shard coverage")
        else:
            raise ValueError("mixed or unknown feature layouts")
    return universe


def validate_selection_coverage(record, universe):
    import json
    import yaml
    path = record.get("selection_coverage", "")
    if not path or sha256(path) != record.get("selection_coverage_sha256"):
        raise ValueError("missing or changed upstream selection coverage; refresh snapshots")
    coverage = json.loads(Path(path).read_text())
    if coverage.get("coverage_schema_version") != 2:
        raise ValueError("obsolete selection coverage; refresh source manifests")
    for key in ("selection_manifest", "selection_policy"):
        if sha256(coverage[key]) != coverage[key + "_sha256"]:
            raise ValueError(f"changed {key}")
    original = read_rows(coverage["selection_manifest"])
    all_uids = {row["source_uid"] for row in original}
    ledger = coverage["source_statuses"]
    by_uid = {row["source_uid"]: row for row in ledger}
    if len(all_uids) != len(original) or len(by_uid) != len(ledger) or set(by_uid) != all_uids:
        raise ValueError("upstream selection membership mismatch")
    policy = yaml.safe_load(Path(coverage["selection_policy"]).read_text())
    excluded_list = policy.get("excluded_source_uids", [])
    excluded = set(excluded_list)
    if len(excluded) != len(excluded_list) or not excluded <= all_uids:
        raise ValueError("unknown or duplicate exclusion UID")
    if {uid for uid, row in by_uid.items() if row["status"] == "excluded"} != excluded:
        raise ValueError("selection exclusions mismatch")
    if any(not by_uid[uid].get("reason") for uid in excluded):
        raise ValueError("missing exclusion reason")
    eligible = all_uids - excluded
    if any(by_uid[uid]["status"] != "available" for uid in eligible):
        raise ValueError("incomplete upstream selection: eligible sources lack valid selection")
    expected_totals = {"total_source_count": len(all_uids), "eligible_source_count": len(eligible),
                       "excluded_source_count": len(excluded), "validated_source_count": len(eligible)}
    if any(coverage.get(key) != value for key, value in expected_totals.items()):
        raise ValueError("upstream selection totals disagree with ledger")
    if coverage.get("status_counts") != dict(Counter(row["status"] for row in ledger)):
        raise ValueError("upstream status counts disagree with ledger")
    profile = coverage["families"][record["family"]]["profile"]
    counts = {uid: by_uid[uid]["selected_counts"][profile] for uid in eligible}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("invalid selected count")
    required = {uid for uid, count in counts.items() if count > 0}
    family_counts = coverage["families"][record["family"]]
    if family_counts["nonempty_source_count"] != len(required) or family_counts["selected_event_count"] != sum(counts.values()):
        raise ValueError("profile totals disagree with ledger")
    if set(universe) != required:
        raise ValueError("feature source manifest does not match selected nonempty sources")
    if any(int(universe[uid]["selected_event_count"]) != counts[uid] for uid in required):
        raise ValueError("selected event count mismatch")
    return {"original_source_count": len(all_uids), "eligible_source_count": len(eligible),
            "excluded_source_count": len(excluded), "validated_source_count": len(eligible),
            "zero_selected_source_count": len(eligible - required),
            "required_feature_source_count": len(required),
            "selection_coverage": path, "selection_coverage_sha256": sha256(path),
            "selection_policy_sha256": coverage["selection_policy_sha256"]}
