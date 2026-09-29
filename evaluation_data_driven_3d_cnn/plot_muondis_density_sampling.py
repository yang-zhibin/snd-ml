#!/usr/bin/env python3
"""Diagnose natural and uniform-bin muon-DIS sampling on the full dataset."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .plotting_common import (
        load_yaml, require_columns, save_figure, write_markdown_table,
    )
except ImportError:  # Direct script execution.
    from plotting_common import load_yaml, require_columns, save_figure, write_markdown_table


def density_bin_definitions(config: dict) -> list[dict]:
    definitions = []
    for index, item in enumerate(config["density_sampling"]["bins"]):
        definitions.append({
            "key": f"bin_{index}",
            "name": str(item["name"]),
            "low": float(item["low"]),
            "high": None if item.get("high") is None else float(item["high"]),
        })
    if not definitions:
        raise ValueError("No density bins are configured")
    return definitions


def expand_density_counts(catalog: pd.DataFrame, manifest: pd.DataFrame,
                          config: dict) -> pd.DataFrame:
    require_columns(catalog, {
        "dataset_file_uid", "class_name", "domain", "fold", "n_events",
    }, "folded catalog")
    require_columns(manifest, {
        "dataset_file_uid", "n_events", "density_counts",
    }, "density manifest")
    muon = catalog.loc[catalog["class_name"].astype(str).eq("muon_DIS")].copy()
    if muon["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Folded catalog contains duplicate muon-DIS dataset files")
    if manifest["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Density manifest contains duplicate dataset files")
    expected = set(muon["dataset_file_uid"].astype(str))
    actual = set(manifest["dataset_file_uid"].astype(str))
    if expected != actual:
        raise ValueError(
            f"Density manifest/catalog mismatch: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )
    joined = muon.merge(
        manifest[["dataset_file_uid", "n_events", "density_counts"]],
        on="dataset_file_uid", how="inner", validate="one_to_one",
        suffixes=("_catalog", "_manifest"),
    )
    definitions = density_bin_definitions(config)
    key_to_name = {item["key"]: item["name"] for item in definitions}
    key_to_name["invalid"] = "invalid"
    rows = []
    for row in joined.itertuples(index=False):
        catalog_events = int(row.n_events_catalog)
        manifest_events = int(row.n_events_manifest)
        counts = {
            str(key): int(value)
            for key, value in json.loads(row.density_counts_manifest).items()
        }
        unknown = set(counts) - set(key_to_name)
        if unknown:
            raise ValueError(f"Unknown density bins for {row.dataset_file_uid}: {sorted(unknown)}")
        if catalog_events != manifest_events or sum(counts.values()) != catalog_events:
            raise ValueError(f"Density count closure failed for {row.dataset_file_uid}")
        for key, name in key_to_name.items():
            rows.append({
                "dataset_file_uid": str(row.dataset_file_uid),
                "domain": str(row.domain),
                "fold": int(row.fold),
                "density_key": key,
                "density_bin": name,
                "is_valid": key != "invalid",
                "n_events": int(counts.get(key, 0)),
            })
    return pd.DataFrame(rows)


def _sampling_rows(counts: pd.DataFrame, group_columns: list[str],
                   valid_names: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in counts.groupby(group_columns, sort=True, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        values = group.groupby("density_bin")["n_events"].sum()
        total_all = int(values.sum())
        total_valid = int(values.reindex(valid_names, fill_value=0).sum())
        populated = [name for name in valid_names if int(values.get(name, 0)) > 0]
        for name in [*valid_names, "invalid"]:
            count = int(values.get(name, 0))
            valid = name != "invalid"
            natural = count / total_valid if valid and total_valid else 0.0
            target = 1.0 / len(populated) if valid and name in populated else 0.0
            result = dict(zip(group_columns, group_key))
            result.update({
                "density_bin": name,
                "n_events": count,
                "natural_fraction": natural,
                "fraction_of_all_events": count / total_all if total_all else 0.0,
                "uniform_training_fraction": target,
                "relative_sampling_multiplier": target / natural if natural else 0.0,
            })
            rows.append(result)
    return pd.DataFrame(rows)


def build_density_tables(catalog: pd.DataFrame, manifest: pd.DataFrame,
                         config: dict) -> dict[str, pd.DataFrame]:
    expanded = expand_density_counts(catalog, manifest, config)
    valid_names = [item["name"] for item in density_bin_definitions(config)]
    full = _sampling_rows(expanded, ["domain"], valid_names)
    by_fold = _sampling_rows(expanded, ["domain", "fold"], valid_names)

    cv_frames = []
    n_folds = int(config["cross_validation"]["n_folds"])
    for held_out_fold in range(n_folds):
        training = expanded.loc[~expanded["fold"].eq(held_out_fold)]
        table = _sampling_rows(training, ["domain"], valid_names)
        table.insert(0, "held_out_fold", held_out_fold)
        cv_frames.append(table)
    cv = pd.concat(cv_frames, ignore_index=True)
    return {"expanded": expanded, "full": full, "by_fold": by_fold, "cv": cv}


def density_warnings(cv: pd.DataFrame) -> list[str]:
    warnings = []
    valid = cv.loc[~cv["density_bin"].eq("invalid")]
    for row in valid.itertuples(index=False):
        multiplier = float(row.relative_sampling_multiplier)
        if int(row.n_events) == 0:
            warnings.append(
                f"Fold {row.held_out_fold} {row.domain} {row.density_bin} has no training events."
            )
        elif multiplier > 50:
            warnings.append(
                f"Fold {row.held_out_fold} {row.domain} {row.density_bin} has severe "
                f"repetition risk (multiplier={multiplier:.1f})."
            )
        elif multiplier > 10:
            warnings.append(
                f"Fold {row.held_out_fold} {row.domain} {row.density_bin} has strong "
                f"repetition risk (multiplier={multiplier:.1f})."
            )
        elif multiplier > 5:
            warnings.append(
                f"Fold {row.held_out_fold} {row.domain} {row.density_bin} has notable "
                f"repetition (multiplier={multiplier:.1f})."
            )
    return warnings


def _grouped_bars(table, value, title, ylabel, path, log=False):
    domains = sorted(table["domain"].unique())
    bins = list(dict.fromkeys(table["density_bin"]))
    x = np.arange(len(bins))
    width = 0.8 / max(len(domains), 1)
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, domain in enumerate(domains):
        values = (
            table.loc[table["domain"].eq(domain)].set_index("density_bin")[value]
            .reindex(bins, fill_value=0).to_numpy(dtype=float)
        )
        axis.bar(x + (index - (len(domains) - 1) / 2) * width, values,
                 width=width, label=domain)
    axis.set_xticks(x, bins, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if log:
        axis.set_yscale("log")
    axis.legend()
    save_figure(figure, path)
    plt.close(figure)


def generate_density_outputs(catalog: pd.DataFrame, manifest: pd.DataFrame,
                             config: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_density_tables(catalog, manifest, config)
    tables["full"].to_csv(output_dir / "muondis_density_table.csv", index=False)
    write_markdown_table(tables["full"], output_dir / "muondis_density_table.md")
    tables["by_fold"].to_csv(output_dir / "muondis_density_by_fold.csv", index=False)
    tables["cv"].to_csv(output_dir / "muondis_cv_sampling_multipliers.csv", index=False)

    _grouped_bars(
        tables["full"], "n_events", "Natural muon-DIS density counts",
        "Events", output_dir / "muondis_density_counts.png", log=True,
    )

    valid = tables["full"].loc[~tables["full"]["density_bin"].eq("invalid")]
    domains = sorted(valid["domain"].unique())
    figure, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 4.5), squeeze=False)
    for axis, domain in zip(axes[0], domains):
        subset = valid.loc[valid["domain"].eq(domain)]
        x = np.arange(len(subset))
        axis.bar(x - 0.2, subset["natural_fraction"], 0.4, label="natural")
        axis.bar(x + 0.2, subset["uniform_training_fraction"], 0.4, label="uniform target")
        axis.set_xticks(x, subset["density_bin"], rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(domain)
        axis.legend()
    figure.suptitle("Natural versus uniform muon-DIS density fractions")
    save_figure(figure, output_dir / "muondis_natural_vs_uniform.png")
    plt.close(figure)

    _grouped_bars(
        valid, "relative_sampling_multiplier", "Muon-DIS training exposure multiplier",
        "Uniform target / natural fraction",
        output_dir / "muondis_sampling_multiplier.png", log=True,
    )

    by_fold = tables["by_fold"].loc[~tables["by_fold"]["density_bin"].eq("invalid")]
    row_labels = [
        f"{domain}:{density_bin}"
        for domain, density_bin in product(
            sorted(by_fold["domain"].unique()),
            [item["name"] for item in density_bin_definitions(config)],
        )
    ]
    matrix = np.zeros((len(row_labels), int(config["cross_validation"]["n_folds"])))
    lookup = {
        (f"{row.domain}:{row.density_bin}", int(row.fold)): int(row.n_events)
        for row in by_fold.itertuples(index=False)
    }
    for row_index, label in enumerate(row_labels):
        for fold in range(matrix.shape[1]):
            matrix[row_index, fold] = lookup.get((label, fold), 0)
    figure, axis = plt.subplots(figsize=(9, max(5, len(row_labels) * 0.4)))
    image = axis.imshow(np.log10(matrix + 1), aspect="auto", cmap="viridis")
    axis.set_xticks(range(matrix.shape[1]), range(matrix.shape[1]))
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Fold")
    axis.set_title("Natural muon-DIS density counts by fold")
    figure.colorbar(image, ax=axis, label="log10(events + 1)")
    save_figure(figure, output_dir / "muondis_density_by_fold.png")
    plt.close(figure)
    tables["warnings"] = density_warnings(tables["cv"])
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folded-catalog", required=True)
    parser.add_argument("--density-index-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_density_outputs(
        pd.read_csv(args.folded_catalog, keep_default_na=False),
        pd.read_csv(args.density_index_manifest, keep_default_na=False),
        load_yaml(args.config), args.output_dir,
    )


if __name__ == "__main__":
    main()
