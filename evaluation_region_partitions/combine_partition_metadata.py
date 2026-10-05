#!/usr/bin/env python3
"""Combine one-row partition metadata products with consistency checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine(inputs: list[str], output: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in inputs]
    nonempty = [frame for frame in frames if not frame.empty]
    result = pd.concat(nonempty, ignore_index=True) if nonempty else frames[0].iloc[:0].copy()
    if not result.empty and result["partition_id"].duplicated().any():
        duplicates = result.loc[result["partition_id"].duplicated(False), "partition_id"].tolist()
        raise ValueError(f"Duplicate partition IDs: {duplicates}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} non-empty partition rows to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    combine(args.input, args.output)


if __name__ == "__main__":
    main()
