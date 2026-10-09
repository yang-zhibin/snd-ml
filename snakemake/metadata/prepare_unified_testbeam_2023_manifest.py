#!/usr/bin/env python3
"""Build the geometry-locked 2023 H8 pion manifest and balanced allocations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


GEOMETRY_ID = "Fe_3walls_H8_2023"
GEOMETRY_CHECKSUM = (
    "4194869a465bfbe2f049effe9407577eba99111334f1960c8d26f857c12de21c"
)
LEGACY_GEOMETRIES = {
    "geofile_sndlhc_H8_2023_1wall.root": {
        "geometry_setup_id": "Fe_1wall_H8_2023",
        "geometry_checksum": "cd57df7f4ab3450882f009c640f3d93d11352f9871db3bf16e62f898debcaaa6",
        "absorber_material": "iron",
        "wall_count": 1,
    },
    "geofile_sndlhc_H8_2023_2walls.root": {
        "geometry_setup_id": "Fe_2walls_H8_2023",
        "geometry_checksum": "1a46ca4ba1ba57a0fa7db581cabd84d2e4a35a56c932468ab94f854cbf1f94b6",
        "absorber_material": "iron",
        "wall_count": 2,
    },
    "geofile_sndlhc_H8_2023_3walls.root": {
        "geometry_setup_id": GEOMETRY_ID,
        "geometry_checksum": GEOMETRY_CHECKSUM,
        "absorber_material": "iron",
        "wall_count": 3,
    },
}
LEGACY_SELECTED_CELLS = (
    {"particle": "pi+", "pdg_code": 211, "energies_gev": (100, 140, 180)},
)
DEFAULT_SELECTED_CELLS = (
    {"particle": "pi+", "pdg_code": 211, "energies_gev": (100, 140, 180)},
    {"particle": "pi-", "pdg_code": -211, "energies_gev": (300,)},
)
TARGET_EVENTS_PER_CELL = 50_000


def stable_source_uid(digi_path: str) -> str:
    digest = hashlib.sha256(str(digi_path).encode("utf-8")).hexdigest()[:20]
    return f"unified_tb23_src_{digest}"


def parse_energy(value: object) -> int:
    energy = float(str(value).strip().lower().replace("gev", ""))
    if not energy.is_integer():
        raise ValueError(f"non-integral beam energy {value!r}")
    return int(energy)


def classify_legacy_geometry(geo_path: str | Path) -> dict:
    """Resolve only layouts whose active placements/checksums were audited."""
    filename = Path(str(geo_path)).name
    if filename not in LEGACY_GEOMETRIES:
        raise ValueError(f"unknown unaudited 2023 H8 geometry: {geo_path}")
    return dict(LEGACY_GEOMETRIES[filename])


def balanced_allocations(event_counts: list[int], target: int) -> list[int]:
    """Allocate exactly target entries while using every non-empty source."""
    if target < 0 or any(count < 0 for count in event_counts):
        raise ValueError("event counts and target must be non-negative")
    if sum(event_counts) < target:
        raise ValueError(f"only {sum(event_counts)} events are available for target {target}")
    allocation = [0] * len(event_counts)
    active = [index for index, count in enumerate(event_counts) if count]
    remaining = target
    while remaining and active:
        share = max(1, remaining // len(active))
        next_active = []
        for index in active:
            room = event_counts[index] - allocation[index]
            take = min(room, share, remaining)
            allocation[index] += take
            remaining -= take
            if allocation[index] < event_counts[index]:
                next_active.append(index)
        active = next_active
    if remaining:
        raise RuntimeError(f"allocation left {remaining} events unassigned")
    return allocation


def deterministic_entry_indices(input_count: int, selected_count: int) -> list[int]:
    """Return sorted, unique, evenly spaced entry indexes."""
    if input_count < 0 or selected_count < 0 or selected_count > input_count:
        raise ValueError("invalid input/selected event counts")
    if selected_count == 0:
        return []
    entries = [
        ((2 * index + 1) * input_count) // (2 * selected_count)
        for index in range(selected_count)
    ]
    if entries != sorted(set(entries)) or entries[-1] >= input_count:
        raise RuntimeError("deterministic index construction failed")
    return entries


def normalize_selected_cells(selected_cells: list[dict] | tuple[dict, ...]) -> list[dict]:
    normalized = []
    observed = set()
    for cell in selected_cells:
        particle = str(cell["particle"])
        pdg_code = int(cell["pdg_code"])
        expected_pdg = {"pi+": 211, "pi-": -211}.get(particle)
        if expected_pdg is None or pdg_code != expected_pdg:
            raise ValueError(f"unsupported particle/PDG cell: {particle}, {pdg_code}")
        energies = tuple(int(value) for value in cell["energies_gev"])
        if not energies or any(value <= 0 for value in energies):
            raise ValueError(f"invalid energies for {particle}: {energies}")
        for energy in energies:
            key = (pdg_code, energy)
            if key in observed:
                raise ValueError(f"duplicate selected cell {key}")
            observed.add(key)
        normalized.append(
            {"particle": particle, "pdg_code": pdg_code, "energies_gev": energies}
        )
    if not normalized:
        raise ValueError("at least one 2023 testbeam cell must be selected")
    return normalized


def build_manifest(
    metadata_path: str | Path,
    selection_root: str | Path,
    feature_root: str | Path,
    selected_cells: list[dict] | tuple[dict, ...] = DEFAULT_SELECTED_CELLS,
    target_events_per_cell: int = TARGET_EVENTS_PER_CELL,
    manifest_version: str = "unified_testbeam_2023_h8_v2",
) -> tuple[pd.DataFrame, dict]:
    metadata_path = Path(metadata_path)
    source = pd.read_csv(metadata_path, keep_default_na=False)
    required = {
        "data_type", "subfolder", "partition", "n_event", "digi_path",
        "geo_path", "beam_energy", "beam_type",
    }
    missing = required - set(source.columns)
    if missing:
        raise KeyError(f"2023 metadata is missing columns: {sorted(missing)}")
    source = source.copy()
    source["source_energy_gev"] = source["beam_energy"].map(parse_energy)
    geometry = source["geo_path"].map(classify_legacy_geometry)
    source["source_geometry_id"] = geometry.map(lambda value: value["geometry_setup_id"])
    source["source_geometry_checksum"] = geometry.map(
        lambda value: value["geometry_checksum"]
    )
    cells_config = normalize_selected_cells(selected_cells)
    particle_to_pdg = {
        str(cell["particle"]): int(cell["pdg_code"]) for cell in cells_config
    }
    selected_keys = {
        (int(cell["pdg_code"]), int(energy))
        for cell in cells_config
        for energy in cell["energies_gev"]
    }
    source["source_pdg_code"] = source["beam_type"].astype(str).map(particle_to_pdg)
    selected_cell_mask = pd.Series(
        [
            (int(pdg), int(energy)) in selected_keys if pd.notna(pdg) else False
            for pdg, energy in zip(
                source["source_pdg_code"], source["source_energy_gev"]
            )
        ],
        index=source.index,
    )
    selected = source.loc[
        source["source_geometry_id"].eq(GEOMETRY_ID)
        & selected_cell_mask
    ].copy()
    if selected.empty:
        raise ValueError("no configured 2023 H8 3-wall pion sources were selected")
    selected["source_pdg_code"] = selected["source_pdg_code"].astype("int64")
    selected["input_event_count"] = pd.to_numeric(
        selected["n_event"], errors="raise"
    ).astype("int64")
    if (selected["input_event_count"] <= 0).any():
        raise ValueError("selected source has a non-positive event count")
    if selected["digi_path"].astype(str).duplicated().any():
        raise ValueError("selected metadata has duplicate digi paths")
    selected = selected.sort_values(
        ["source_pdg_code", "source_energy_gev", "partition", "digi_path"],
        kind="mergesort",
    ).reset_index().rename(columns={"index": "source_metadata_row"})

    for pdg_code, energy in sorted(selected_keys):
        mask = selected["source_pdg_code"].eq(pdg_code) & selected[
            "source_energy_gev"
        ].eq(energy)
        counts = selected.loc[mask, "input_event_count"].astype(int).tolist()
        selected.loc[mask, "selected_event_count"] = balanced_allocations(
            counts, int(target_events_per_cell)
        )
    selected["selected_event_count"] = selected["selected_event_count"].astype("int64")

    selection_root = Path(selection_root)
    feature_root = Path(feature_root)
    records = []
    for row in selected.to_dict("records"):
        uid = stable_source_uid(str(row["digi_path"]))
        index_path = selection_root / uid / "selected_entries.root"
        feature_path = feature_root / uid / f"feature_scifi_cluster_v1_{uid}.root"
        records.append({
            "manifest_version": str(manifest_version),
            "family": "testbeam_data_2023_h8",
            "source_uid": uid,
            "source_metadata_file": str(metadata_path),
            "source_metadata_row": int(row["source_metadata_row"]) + 2,
            "data_type": str(row["data_type"]),
            "subfolder": str(row["subfolder"]),
            "partition": str(row["partition"]),
            "digi_path": str(row["digi_path"]),
            "geo_path": str(row["geo_path"]),
            "cutflow_path": str(index_path),
            "input_tree": "cbmsim",
            "sample_type": "real_testbeam",
            "selection_mode": "indexed_cutflow",
            "canonical_label": "testbeam_data_pion",
            "canonical_labels": "testbeam_data_pion",
            "source_pdg_code": int(row["source_pdg_code"]),
            "source_energy_gev": int(row["source_energy_gev"]),
            "source_domain_id": 6,
            "source_domain": "testbeam_data_2023_h8",
            "source_geometry_id": GEOMETRY_ID,
            "source_geometry_checksum": GEOMETRY_CHECKSUM,
            "lumi_per_file": "",
            "exposure_kind": "event_count_only",
            "feature_scifi_cluster_v1_path": str(feature_path),
            "validation_record": str(feature_path) + ".validated.json",
            "validation_marker": str(feature_path) + ".validated.json",
            "input_event_count": int(row["input_event_count"]),
            "selected_event_count": int(row["selected_event_count"]),
            "selection_index_path": str(index_path),
            "is_canary": False,
        })
    manifest = pd.DataFrame(records).sort_values(
        ["source_pdg_code", "source_energy_gev", "source_uid"], kind="mergesort"
    ).reset_index(drop=True)
    if manifest["source_uid"].duplicated().any():
        raise ValueError("stable source UID collision")
    cells = []
    particle_by_pdg = {value: key for key, value in particle_to_pdg.items()}
    for (pdg_code, energy), frame in manifest.groupby(
        ["source_pdg_code", "source_energy_gev"], sort=True
    ):
        cells.append({
            "particle": particle_by_pdg[int(pdg_code)],
            "pdg_code": int(pdg_code),
            "energy_gev": int(energy),
            "source_count": int(len(frame)),
            "raw_event_count": int(frame["input_event_count"].sum()),
            "selected_event_count": int(frame["selected_event_count"].sum()),
        })
    observed = {
        (row["pdg_code"], row["energy_gev"]): row["selected_event_count"]
        for row in cells
    }
    expected = {
        key: int(target_events_per_cell) for key in selected_keys
    }
    if observed != expected:
        raise RuntimeError(f"cell allocation mismatch: {observed} != {expected}")
    report = {
        "manifest_version": str(manifest_version),
        "source_metadata_file": str(metadata_path),
        "selection_policy": "balanced_across_sources_evenly_spaced_entries_v1",
        "geometry_classification": "audited_active_placements_and_checksum_v1",
        "geometry_setup_id": GEOMETRY_ID,
        "geometry_checksum": GEOMETRY_CHECKSUM,
        "absorber_material": "iron",
        "wall_count": 3,
        "selected_cells": cells_config,
        "target_events_per_cell": int(target_events_per_cell),
        "cells": cells,
        "source_count": int(len(manifest)),
    }
    return manifest, report


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    settings = config.get(
        "testbeam_2023",
        {
            "manifest_version": "unified_testbeam_2023_h8_v1",
            "target_events_per_cell": TARGET_EVENTS_PER_CELL,
            "selected_cells": LEGACY_SELECTED_CELLS,
        },
    )
    manifest, report = build_manifest(
        args.metadata,
        args.selection_root,
        args.feature_root,
        selected_cells=settings["selected_cells"],
        target_events_per_cell=int(settings["target_events_per_cell"]),
        manifest_version=str(settings["manifest_version"]),
    )
    atomic_csv(manifest, Path(args.output_manifest))
    atomic_json(report, Path(args.output_summary))
    print(
        f"wrote {len(manifest)} 2023 H8 sources with "
        f"{manifest['selected_event_count'].sum()} selected events"
    )


if __name__ == "__main__":
    main()
