#!/usr/bin/env python3
"""Validate the schema and invariants of one flat reco-track feature file."""

import argparse
import math

import ROOT

from reco_track_features import (
    DETAIL_TRACK_VECTOR_DTYPES,
    FLAT_V1,
    FLAT_V2,
    MEASUREMENT_VECTOR_DTYPES,
    RECO_TRACK_SCHEMAS,
    TRACK_TYPE_PREFIXES,
    VECTOR_DTYPES,
    decode_measurement_detid,
    scalar_dtypes_for_schema,
    vector_dtypes_for_schema,
)


def _validate_v1_entry(tree, index):
    lengths = [len(getattr(tree, name)) for name in VECTOR_DTYPES]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Track-vector length mismatch at entry {index}: {lengths}")
    n_tracks = int(tree.reco_track_n_all)
    if lengths[0] != n_tracks:
        raise RuntimeError(f"reco_track_n_all mismatch at entry {index}")
    known_sum = sum(
        int(getattr(tree, f"{prefix}_n_all"))
        for prefix in TRACK_TYPE_PREFIXES.values()
    )
    if known_sum + int(tree.reco_track_n_unknown_type) != n_tracks:
        raise RuntimeError(f"Per-type count mismatch at entry {index}")
    for prefix in TRACK_TYPE_PREFIXES.values():
        has_best = int(getattr(tree, f"{prefix}_has_best"))
        best_index = int(getattr(tree, f"{prefix}_best_index"))
        chi2ndf = float(getattr(tree, f"{prefix}_best_chi2ndf"))
        if has_best and (
            best_index < 0 or best_index >= n_tracks or not math.isfinite(chi2ndf)
        ):
            raise RuntimeError(f"Invalid best-track state for {prefix}, entry {index}")
        if not has_best and best_index != -1:
            raise RuntimeError(
                f"Invalid absent-track sentinel for {prefix}, entry {index}"
            )


def _validate_v2_entry(tree, index):
    n_tracks = int(tree.reco_track_n_all)
    track_lengths = [
        len(getattr(tree, name)) for name in DETAIL_TRACK_VECTOR_DTYPES
    ]
    if any(length != n_tracks for length in track_lengths):
        raise RuntimeError(
            f"Detail track-vector length mismatch at entry {index}: {track_lengths}"
        )

    offsets = [int(value) for value in tree.reco_track_measurement_offset]
    if len(offsets) != n_tracks + 1 or not offsets or offsets[0] != 0:
        raise RuntimeError(f"Invalid measurement offsets at entry {index}: {offsets}")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise RuntimeError(f"Non-monotonic measurement offsets at entry {index}")

    measurement_lengths = [
        len(getattr(tree, name)) for name in MEASUREMENT_VECTOR_DTYPES
    ]
    if len(set(measurement_lengths)) != 1:
        raise RuntimeError(
            f"Measurement-vector length mismatch at entry {index}: "
            f"{measurement_lengths}"
        )
    if offsets[-1] != measurement_lengths[0]:
        raise RuntimeError(f"Final measurement offset mismatch at entry {index}")

    for track_index in range(n_tracks):
        n_measurements = int(tree.reco_track_n_measurements[track_index])
        if offsets[track_index + 1] - offsets[track_index] != n_measurements:
            raise RuntimeError(
                f"Track measurement-count mismatch at entry {index}, "
                f"track {track_index}"
            )
        if int(tree.reco_track_npoints[track_index]) != n_measurements:
            raise RuntimeError(
                f"Track point/measurement mismatch at entry {index}, "
                f"track {track_index}"
            )
        if int(tree.reco_track_measurement_alignment_ok[track_index]) != 1:
            raise RuntimeError(
                f"Unaligned measurements at entry {index}, track {track_index}"
            )
        first_detid = int(tree.reco_track_first_measurement_detid[track_index])
        last_detid = int(tree.reco_track_last_measurement_detid[track_index])
        begin, end = offsets[track_index], offsets[track_index + 1]
        if n_measurements:
            if first_detid != int(tree.reco_track_measurement_detid[begin]):
                raise RuntimeError(
                    f"First detector-ID mismatch at entry {index}, track {track_index}"
                )
            if last_detid != int(tree.reco_track_measurement_detid[end - 1]):
                raise RuntimeError(
                    f"Last detector-ID mismatch at entry {index}, track {track_index}"
                )
        elif first_detid != -1 or last_detid != -1:
            raise RuntimeError(
                f"Invalid empty-track detector-ID sentinel at entry {index}, "
                f"track {track_index}"
            )

        subsystem_names = {
            0: "unknown",
            1: "scifi",
            2: "veto",
            3: "us",
            4: "ds",
        }
        actual_measurements = {name: 0 for name in subsystem_names.values()}
        actual_planes = {name: set() for name in ("scifi", "veto", "us", "ds")}
        for measurement_index in range(begin, end):
            detid = int(tree.reco_track_measurement_detid[measurement_index])
            decoded = decode_measurement_detid(detid)
            for field in ("subsystem", "station", "plane", "view", "plane_key"):
                saved = int(
                    getattr(tree, f"reco_track_measurement_{field}")[measurement_index]
                )
                if saved != decoded[field]:
                    raise RuntimeError(
                        f"Detector-ID decoding mismatch for {field} at entry "
                        f"{index}, track {track_index}, measurement {measurement_index}"
                    )
            name = subsystem_names[decoded["subsystem"]]
            actual_measurements[name] += 1
            if name != "unknown":
                actual_planes[name].add(decoded["plane_key"])

        for name, count in actual_measurements.items():
            saved = int(
                getattr(tree, f"reco_track_n_{name}_measurements")[track_index]
            )
            if saved != count:
                raise RuntimeError(
                    f"{name} measurement-count mismatch at entry {index}, "
                    f"track {track_index}"
                )
        for name, keys in actual_planes.items():
            saved = int(getattr(tree, f"reco_track_n_{name}_planes")[track_index])
            if saved != len(keys):
                raise RuntimeError(
                    f"{name} plane-count mismatch at entry {index}, track {track_index}"
                )
        if sum(len(keys) for keys in actual_planes.values()) != int(
            tree.reco_track_n_unique_planes[track_index]
        ):
            raise RuntimeError(
                f"Unique plane-count mismatch at entry {index}, track {track_index}"
            )

    for prefix in TRACK_TYPE_PREFIXES.values():
        if not int(getattr(tree, f"{prefix}_has_best")):
            continue
        best_index = int(getattr(tree, f"{prefix}_best_index"))
        for vector_name in DETAIL_TRACK_VECTOR_DTYPES:
            field = vector_name.removeprefix("reco_track_")
            scalar_value = int(getattr(tree, f"{prefix}_best_{field}"))
            vector_value = int(getattr(tree, vector_name)[best_index])
            if scalar_value != vector_value:
                raise RuntimeError(
                    f"Best-track detail mismatch for {prefix}_{field}, entry {index}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument(
        "--schema",
        choices=RECO_TRACK_SCHEMAS,
        default=None,
        help="Expected schema; by default use metadata_reco_track_features.",
    )
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(args.path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {args.path}")
    tree = root_file.Get("sndData")
    if not tree:
        raise RuntimeError("Missing sndData tree")
    metadata = root_file.Get("metadata_reco_track_features")
    schema = metadata.GetTitle() if metadata else None
    if schema not in RECO_TRACK_SCHEMAS:
        raise RuntimeError("Missing or invalid metadata_reco_track_features")
    if args.schema is not None and schema != args.schema:
        raise RuntimeError(
            f"Expected reconstructed-track schema {args.schema}, found {schema}"
        )
    missing = [
        name
        for name in list(scalar_dtypes_for_schema(schema))
        + list(vector_dtypes_for_schema(schema))
        if not tree.GetBranch(name)
    ]
    if missing:
        raise RuntimeError(f"Missing reconstructed-track branches: {missing}")

    # Sample entries throughout the file without turning validation into a
    # second full event loop.
    entries = int(tree.GetEntries())
    indices = sorted({0, entries // 2, entries - 1}) if entries else []
    for index in indices:
        tree.GetEntry(index)
        _validate_v1_entry(tree, index)
        if schema == FLAT_V2:
            _validate_v2_entry(tree, index)

    print(f"Validated {args.path}: {entries} entries, {schema} schema")
    root_file.Close()


if __name__ == "__main__":
    main()
