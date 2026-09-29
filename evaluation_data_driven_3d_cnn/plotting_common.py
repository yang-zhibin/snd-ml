"""Shared constants and output helpers for dataset diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import yaml


matplotlib.use("Agg")

CLASS_ORDER = (
    "CC_numu",
    "CC_nue",
    "NC",
    "neutral_kaon",
    "neutron",
    "passing_muon",
    "muon_DIS",
)
EXCLUDED_CLASSES = ("CC_nutau",)
MUON_CLASSES = ("passing_muon", "muon_DIS")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def validate_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("valid") is not True:
        raise ValueError(f"Dataset validation did not pass: {path}")
    return report


def _markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).write_text(markdown_table(frame), encoding="utf-8")


def save_figure(figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
