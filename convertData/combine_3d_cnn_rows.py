#!/usr/bin/env python3
"""Combine one-row CSV products with strict schema and key validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unique-key")
    args = parser.parse_args()
    frames = []
    columns = None
    for path in args.inputs:
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != 1:
            raise ValueError(f"Expected exactly one row in {path}, found {len(frame)}")
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise ValueError(f"CSV schema differs in {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if args.unique_key:
        if args.unique_key not in result:
            raise KeyError(f"Unique key {args.unique_key!r} is missing")
        if result[args.unique_key].astype(str).duplicated().any():
            raise ValueError(f"Duplicate values found for {args.unique_key}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Combined {len(result)} rows into {output}")


if __name__ == "__main__":
    main()
