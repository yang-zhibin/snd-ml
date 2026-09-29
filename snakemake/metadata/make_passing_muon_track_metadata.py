#!/usr/bin/env python3
"""Derive versioned reconstructed-track feature metadata from passing-muon metadata."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_VERSION = "passing_muon_track_features_v1"
DEFAULT_SCHEMA = "flat_v1"


def derive_track_metadata(frame, version=DEFAULT_VERSION, schema=DEFAULT_SCHEMA):
    required = {
        "reco_path",
        "geo_path",
        "input_tree",
        "partition",
        "lumi_per_file",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Passing-muon metadata lacks columns: {sorted(missing)}")

    result = frame.copy()
    result["source_feature_path"] = result.get("feature_path")
    result["production_profile"] = version
    result["track_feature_schema"] = schema
    result["output_base_path"] = result["partition"].map(
        lambda partition: (
            "/eos/experiment/sndlhc/users/zhibin/"
            f"{version}/2024/{partition}"
        )
    )
    result["track_feature_path"] = result["reco_path"].map(
        lambda path: f"featureTracks_{Path(path).stem}.root"
    )
    # Keep feature_path compatible with generic inspection-input builders.
    result["feature_path"] = result["track_feature_path"]
    duplicate = result["track_feature_path"].duplicated(keep=False)
    if duplicate.any():
        names = result.loc[duplicate, "track_feature_path"].head(10).tolist()
        raise ValueError(f"Non-unique track feature names: {names}")
    return result


def atomic_csv(frame, output):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--schema", choices=("flat_v1", "flat_v2"), default=DEFAULT_SCHEMA
    )
    args = parser.parse_args()
    metadata = derive_track_metadata(
        pd.read_csv(args.input), version=args.version, schema=args.schema
    )
    atomic_csv(metadata, args.output)
    print(f"Wrote {len(metadata)} rows to {args.output}")


if __name__ == "__main__":
    main()
