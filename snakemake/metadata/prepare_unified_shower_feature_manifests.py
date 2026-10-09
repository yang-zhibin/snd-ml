#!/usr/bin/env python3
"""Build deterministic per-family manifests for unified shower features."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml


FAMILIES = (
    "mc_neutrino",
    "mc_muon_up",
    "real_data_2024_skim",
    "testbeam_mc_2024",
    "testbeam_data_2024",
)
DOMAIN_NAMES = {
    1: "mc_neutrino",
    2: "mc_muon_up",
    3: "real_data_2024_skim",
    4: "testbeam_mc_2024",
    5: "testbeam_data_2024",
}
SAMPLE_TYPE_BY_DOMAIN = {
    1: "MC_neutrino",
    2: "MC_muon",
    3: "real_data",
    4: "MC_testbeam",
    5: "real_testbeam",
}
EXTERNAL_SELECTION_MODES = {"external_cutflow", "identity_cutflow"}
SOURCE_UID_PATTERN = re.compile(r"(?:unified_src|tbfull_src)_[0-9a-f]{20}\Z")


def selected_entry_shard_ranges(
    input_event_count: int,
    selected_event_count: int,
    threshold_input_events: int,
    selected_events_per_shard: int,
) -> list[tuple[int, int, int]]:
    """Return selected-entry shard ranges for a large raw input source."""
    input_event_count = int(input_event_count)
    selected_event_count = int(selected_event_count)
    threshold_input_events = int(threshold_input_events)
    selected_events_per_shard = int(selected_events_per_shard)
    if input_event_count < 0 or selected_event_count < 0:
        raise ValueError("Event counts must be non-negative")
    if selected_event_count > input_event_count:
        raise ValueError(
            "Selected event count cannot exceed the raw input event count"
        )
    if threshold_input_events < 0 or selected_events_per_shard <= 0:
        raise ValueError("Shard threshold and size must be valid")
    if input_event_count <= threshold_input_events:
        return []
    return [
        (index, start, min(start + selected_events_per_shard, selected_event_count))
        for index, start in enumerate(
            range(0, selected_event_count, selected_events_per_shard)
        )
    ]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _resolve(row: pd.Series, column: str) -> str:
    value = _clean(row.get(column, ""))
    if not value:
        return ""
    if value.startswith("root://") or Path(value).is_absolute():
        return value
    base = _clean(row.get("output_base_path", "")).rstrip("/")
    return f"{base}/{value.lstrip('/')}" if base else value


def _source_uid(family: str, metadata_file: str, source_row: int, digi_path: str) -> str:
    digest = hashlib.sha256(
        f"{family}\x1f{metadata_file}\x1f{source_row}\x1f{digi_path}".encode("utf-8")
    ).hexdigest()[:20]
    return f"unified_src_{digest}"


def _canonical_label(family: str, row: pd.Series) -> str:
    if family == "real_data_2024_skim":
        return "real_data"
    if family == "mc_muon_up":
        return "MC_muon_up"
    if family.startswith("testbeam_"):
        particle = _clean(row.get("class_name", "")).lower()
        if particle not in {"electron", "pion"}:
            raise ValueError(f"Unsupported testbeam particle label {particle!r}")
        domain = "mc" if family == "testbeam_mc_2024" else "data"
        return f"testbeam_{domain}_{particle}"
    return ""


def _base_record(
    config: dict,
    family: str,
    row: pd.Series,
    metadata_file: str,
    source_row: int,
) -> dict:
    settings = config["families"][family]
    digi_path = _clean(row.get("digi_path", ""))
    geo_path = _clean(row.get("geo_path", ""))
    if not digi_path or not geo_path:
        raise ValueError(f"Missing digi/geometry path for {family} row {source_row}")
    source_uid = _clean(row.get("source_uid", ""))
    if not source_uid:
        source_uid = _source_uid(family, metadata_file, source_row, digi_path)
    if not SOURCE_UID_PATTERN.fullmatch(source_uid):
        raise ValueError(f"Invalid source UID {source_uid!r} for {family}")

    feature_root = str(config["feature_roots"][family]).rstrip("/")
    feature_path = (
        f"{feature_root}/{source_uid}/"
        f"feature_scifi_cluster_v1_{source_uid}.root"
    )
    validation_record = f"{feature_path}.validated.json"
    expected = row.get("n_event", row.get("n_events", ""))
    if expected == "" or pd.isna(expected):
        raise ValueError(f"Missing expected event count for {family}/{source_uid}")
    expected = int(expected)
    if expected < 0:
        raise ValueError(f"Negative expected event count for {family}/{source_uid}")

    lumi = row.get("lumi_per_file", "")
    if lumi == "" or pd.isna(lumi):
        lumi = ""
    else:
        lumi = float(lumi)
        if not pd.notna(lumi) or lumi < 0:
            raise ValueError(f"Invalid luminosity for {family}/{source_uid}")

    return {
        "manifest_version": str(config["version"]),
        "family": family,
        "source_uid": source_uid,
        "source_metadata_file": metadata_file,
        "source_metadata_row": int(source_row),
        "data_type": _clean(row.get("data_type", settings["sample_type"])),
        "subfolder": _clean(row.get("subfolder", "")),
        "partition": _clean(row.get("partition", "")),
        "digi_path": digi_path,
        "geo_path": geo_path,
        "cutflow_path": "",
        "input_tree": str(settings.get("input_tree", "cbmsim")),
        "sample_type": str(settings["sample_type"]),
        "selection_mode": str(settings["selection_mode"]),
        "canonical_label": _canonical_label(family, row),
        "canonical_labels": "|".join(settings["canonical_labels"]),
        "source_pdg_code": "",
        "source_energy_gev": "",
        "source_domain_id": int(settings["source_domain_id"]),
        "source_domain": str(settings["source_domain"]),
        "source_geometry_id": _clean(row.get("geometry_setup_id", "")),
        "source_geometry_checksum": _clean(row.get("geometry_checksum", "")),
        "lumi_per_file": lumi,
        "exposure_kind": str(settings["exposure_kind"]),
        "feature_scifi_cluster_v1_path": feature_path,
        "validation_record": validation_record,
        # Transitional alias for consumers that still use the old generic name.
        "validation_marker": validation_record,
        "input_event_count": expected,
        "selected_event_count": (
            expected if str(settings["selection_mode"]) == "direct_input" else ""
        ),
        "is_canary": 0,
    }


def _legacy_family(config: dict, family: str) -> pd.DataFrame:
    settings = config["families"][family]
    paths = settings["source_metadata"]
    if isinstance(paths, str):
        paths = [paths]
    records = []
    for source_path in paths:
        frame = pd.read_csv(source_path, keep_default_na=False)
        for source_row, row in frame.iterrows():
            if family == "mc_muon_up" and not (
                str(row["data_type"]).lower() == "mc_muon"
                and str(row["subfolder"]) == "up"
            ):
                continue
            record = _base_record(config, family, row, str(source_path), source_row)
            if family in {"mc_neutrino", "real_data_2024_skim"}:
                record["cutflow_path"] = _resolve(row, "nueAnalysisFilter_path")
            records.append(record)
    return pd.DataFrame(records)


def _testbeam_family(config: dict, family: str) -> pd.DataFrame:
    settings = config["families"][family]
    catalog_path = str(settings["source_metadata"])
    expected_hash = str(settings["source_metadata_sha256"])
    actual_hash = _sha256(catalog_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen testbeam catalogue hash mismatch: {actual_hash} != {expected_hash}"
        )
    frame = pd.read_csv(catalog_path, keep_default_na=False)
    selected = frame.loc[frame["domain"].astype(str).eq(settings["catalog_domain"])]
    records = []
    for catalog_row, row in selected.iterrows():
        if _clean(row.get("geometry_setup_id")) != "W_2walls_28plates":
            raise ValueError(f"Non-W2 testbeam source at catalogue row {catalog_row}")
        metadata_file = str(row["metadata_csv"])
        metadata_row = int(row["metadata_row"])
        record = _base_record(config, family, row, metadata_file, metadata_row)
        record.update(
            {
                "catalog_file": catalog_path,
                "catalog_row": int(catalog_row),
                "cutflow_path": str(row["cutflow_path"]),
                "source_pdg_code": int(row["pdg_code"]),
                "source_energy_gev": float(row["beam_energy_gev"]),
                "selected_event_count": int(row["n_events"]),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _validate_contract(config: dict, family: str, frame: pd.DataFrame) -> None:
    settings = config["families"][family]
    domain_id = int(settings["source_domain_id"])
    if DOMAIN_NAMES.get(domain_id) != family:
        raise ValueError(f"Domain ID/name mismatch for {family}")
    if str(settings["source_domain"]) != family:
        raise ValueError(f"Source-domain string mismatch for {family}")
    if str(settings["sample_type"]) != SAMPLE_TYPE_BY_DOMAIN[domain_id]:
        raise ValueError(f"Sample-type/domain mismatch for {family}")

    expected = int(settings["expected_source_files"])
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} {family} rows, found {len(frame)}")
    for column in ("digi_path", "geo_path"):
        if frame[column].astype(str).eq("").any():
            raise ValueError(f"Empty {family} {column}")
    for column in (
        "source_uid",
        "feature_scifi_cluster_v1_path",
        "validation_record",
    ):
        if frame[column].duplicated().any():
            raise ValueError(f"Duplicate {family} {column}")
        if frame[column].astype(str).eq("").any():
            raise ValueError(f"Empty {family} {column}")
    if str(settings["selection_mode"]) in EXTERNAL_SELECTION_MODES and frame[
        "cutflow_path"
    ].astype(str).eq("").any():
        raise ValueError(f"Empty {family} cutflow path")
    if str(settings["selection_mode"]) in {"direct_input", "identity_cutflow"}:
        if frame["selected_event_count"].astype(str).eq("").any():
            raise ValueError(f"Empty {family} selected event count")
        selected_counts = frame["selected_event_count"].astype(int)
        input_counts = frame["input_event_count"].astype(int)
        if selected_counts.lt(0).any():
            raise ValueError(f"Negative {family} selected event count")
        if selected_counts.gt(input_counts).any():
            raise ValueError(
                f"Selected event count exceeds input event count for {family}"
            )
    if family.startswith("testbeam_"):
        for column in (
            "source_pdg_code",
            "source_energy_gev",
            "source_geometry_id",
            "source_geometry_checksum",
            "canonical_label",
        ):
            if frame[column].astype(str).eq("").any():
                raise ValueError(f"Empty testbeam {column}")
    elif frame["source_pdg_code"].astype(str).ne("").any() or frame[
        "source_energy_gev"
    ].astype(str).ne("").any():
        raise ValueError(f"Non-testbeam source overrides found for {family}")


def _finish(config: dict, family: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("source_uid", kind="mergesort").reset_index(drop=True)
    _validate_contract(config, family, frame)
    canary = config["families"][family].get("canary")
    if canary:
        candidates = pd.Series(True, index=frame.index)
        metadata_suffix = str(canary.get("source_metadata_suffix", ""))
        if metadata_suffix:
            candidates &= frame["source_metadata_file"].astype(str).str.endswith(
                metadata_suffix
            )
        if "source_metadata_row" in canary:
            candidates &= frame["source_metadata_row"].astype(int).eq(
                int(canary["source_metadata_row"])
            )
        if "source_uid" in canary:
            candidates &= frame["source_uid"].astype(str).eq(str(canary["source_uid"]))
        matches = list(frame.index[candidates])
        if len(matches) != 1:
            raise ValueError(f"Expected one configured {family} canary, found {len(matches)}")
        frame.loc[matches[0], "is_canary"] = 1
    else:
        frame.loc[0, "is_canary"] = 1
    return frame


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    unknown = set(config["families"]) - set(FAMILIES)
    missing = set(FAMILIES) - set(config["families"])
    if unknown or missing:
        raise ValueError(f"Family mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return config


def build_manifests(config_path: str | Path) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    result = {}
    for family in FAMILIES:
        if family.startswith("testbeam_"):
            frame = _testbeam_family(config, family)
        else:
            frame = _legacy_family(config, family)
        result[family] = _finish(config, family, frame)
    combined = pd.concat(result.values(), ignore_index=True)
    for column in ("source_uid", "feature_scifi_cluster_v1_path", "validation_record"):
        if combined[column].duplicated().any():
            raise ValueError(f"Duplicate cross-family {column}")
    return result


def write_manifest(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    config = load_config(args.config)
    frames = build_manifests(args.config)
    output_dir = Path(args.output_dir or config["manifest_root"])
    families = [args.family] if args.family else list(FAMILIES)
    for family in families:
        destination = output_dir / f"{family}.csv"
        write_manifest(frames[family], destination)
        print(f"Wrote {len(frames[family])} {family} rows to {destination}")


if __name__ == "__main__":
    main()
