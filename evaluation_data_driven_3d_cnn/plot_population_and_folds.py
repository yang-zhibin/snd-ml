#!/usr/bin/env python3
"""Build natural-population and five-fold balance diagnostics."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .plotting_common import (
        CLASS_ORDER, EXCLUDED_CLASSES, MUON_CLASSES, load_yaml, require_columns,
        save_figure, write_markdown_table,
    )
except ImportError:  # Direct script execution.
    from plotting_common import (
        CLASS_ORDER, EXCLUDED_CLASSES, MUON_CLASSES, load_yaml, require_columns,
        save_figure, write_markdown_table,
    )


POPULATION_COLUMNS = {
    "fold", "class_name", "domain", "n_events", "dataset_file_uid",
    "source_uid", "fold_group_uid", "lumi_source_uid", "lumi_per_file",
}


def aggregate_population(catalog: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Return a complete class/fold/domain grid using natural catalog counts."""
    require_columns(catalog, POPULATION_COLUMNS, "folded catalog")
    observed_classes = set(catalog["class_name"].astype(str))
    unexpected = observed_classes - set(CLASS_ORDER) - set(EXCLUDED_CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected classes in folded catalog: {sorted(unexpected)}")
    domains = sorted(set(catalog["domain"].astype(str)))
    if not domains:
        raise ValueError("Folded catalog contains no domains")

    rows = []
    grouped = catalog.groupby(["class_name", "fold", "domain"], dropna=False)
    for class_name, fold, domain in product(CLASS_ORDER, range(n_folds), domains):
        key = (class_name, fold, domain)
        group = grouped.get_group(key) if key in grouped.groups else catalog.iloc[0:0]
        luminosity = group[["lumi_source_uid", "lumi_per_file"]].drop_duplicates(
            "lumi_source_uid"
        )
        rows.append({
            "class_name": class_name,
            "fold": int(fold),
            "domain": domain,
            "n_events": int(pd.to_numeric(group["n_events"]).sum()),
            "dataset_rows": int(len(group)),
            "source_files": int(group["source_uid"].astype(str).nunique()),
            "source_groups": int(group["fold_group_uid"].astype(str).nunique()),
            "lumi_sources": int(group["lumi_source_uid"].astype(str).nunique()),
            "deduplicated_luminosity": float(
                pd.to_numeric(luminosity["lumi_per_file"], errors="coerce").fillna(0).sum()
            ),
        })
    return pd.DataFrame(rows)


def fold_balance_metrics(population: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    totals = population.groupby(["class_name", "fold"], as_index=False)["n_events"].sum()
    rows = []
    for class_name in CLASS_ORDER:
        counts = (
            totals.loc[totals["class_name"].eq(class_name)]
            .set_index("fold")["n_events"]
            .reindex(range(n_folds), fill_value=0)
            .astype(float)
        )
        total = float(counts.sum())
        mean = total / n_folds
        fractions = counts / total if total else counts * 0.0
        relative_deviation = (
            float(np.max(np.abs(counts - mean) / mean)) if mean else float("inf")
        )
        minimum = float(counts.min())
        maximum = float(counts.max())
        empty_folds = int((counts == 0).sum())
        if empty_folds or relative_deviation > 0.5:
            status = "WARNING"
        elif relative_deviation > 0.2:
            status = "CHECK"
        else:
            status = "OK"
        rows.append({
            "class_name": class_name,
            "total_events": int(total),
            "minimum_fold_events": int(minimum),
            "maximum_fold_events": int(maximum),
            "maximum_minimum_ratio": maximum / minimum if minimum else float("inf"),
            "mean_fold_events": mean,
            "std_fold_events": float(counts.std(ddof=0)),
            "coefficient_of_variation": float(counts.std(ddof=0) / mean) if mean else float("inf"),
            "largest_absolute_share_deviation": float(np.max(np.abs(fractions - 1 / n_folds))),
            "largest_relative_deviation": relative_deviation,
            "empty_folds": empty_folds,
            "status": status,
        })
    return pd.DataFrame(rows)


def population_warnings(catalog: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    warnings = []
    tau_events = int(pd.to_numeric(
        catalog.loc[catalog["class_name"].astype(str).eq("CC_nutau"), "n_events"]
    ).sum())
    if tau_events:
        warnings.append(
            f"CC_nutau unexpectedly contains {tau_events} events and must remain excluded."
        )
    for row in metrics.itertuples(index=False):
        if row.status != "OK":
            warnings.append(
                f"{row.class_name} fold balance is {row.status}: maximum relative "
                f"deviation={row.largest_relative_deviation:.1%}, empty folds={row.empty_folds}."
            )
    all_folds = set(pd.to_numeric(catalog["fold"]).astype(int))
    for (domain, class_name), group in catalog.groupby(
        ["domain", "class_name"], sort=True, dropna=False
    ):
        present_folds = set(pd.to_numeric(group["fold"]).astype(int))
        missing_folds = sorted(all_folds - present_folds)
        if missing_folds:
            warnings.append(
                f"{domain} {class_name} has no independent source group in "
                f"fold(s) {', '.join(map(str, missing_folds))}."
            )
    return warnings


def _class_fold_matrix(population: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        population.groupby(["class_name", "fold"])[value].sum()
        .unstack(fill_value=0)
        .reindex(index=CLASS_ORDER)
        .sort_index(axis=1)
    )


def _annotated_heatmap(values, annotations, title, colorbar, path, cmap="viridis",
                       center=None):
    figure, axis = plt.subplots(figsize=(9, 5.6))
    if center is None:
        image = axis.imshow(values, aspect="auto", cmap=cmap)
    else:
        span = max(abs(np.nanmin(values) - center), abs(np.nanmax(values) - center), 0.05)
        image = axis.imshow(
            values, aspect="auto", cmap=cmap, vmin=center - span, vmax=center + span
        )
    axis.set_xticks(range(values.shape[1]), [str(index) for index in range(values.shape[1])])
    axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axis.set_xlabel("Fold")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, annotations[row, column], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label=colorbar)
    save_figure(figure, path)
    plt.close(figure)


def _format_luminosity(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 1000 or value < 0.01:
        return f"{value:.2e}"
    return f"{value:.3f}"


def plot_luminosity_by_class_and_fold(population: pd.DataFrame, n_folds: int,
                                      path: str | Path) -> None:
    """Plot deduplicated integrated/equivalent luminosity in fb^-1."""
    matrix = (
        population.groupby(["domain", "class_name", "fold"])[
            "deduplicated_luminosity"
        ]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(n_folds), fill_value=0.0)
    )
    source_rows = (
        population.groupby(["domain", "class_name", "fold"])["dataset_rows"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=matrix.index, columns=range(n_folds), fill_value=0)
    )
    populated = source_rows.sum(axis=1).gt(0)
    matrix = matrix.loc[populated]
    source_rows = source_rows.loc[populated]
    if matrix.empty:
        raise ValueError("No luminosity source rows are available to plot")

    values = matrix.to_numpy(dtype=float)
    log_values = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0
    log_values[positive] = np.log10(values[positive])
    labels = [f"{domain}: {class_name}" for domain, class_name in matrix.index]

    figure, axis = plt.subplots(figsize=(9, max(5, 0.55 * len(labels))))
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad(color="lightgray")
    image = axis.imshow(log_values, aspect="auto", cmap=colormap)
    axis.set_xticks(range(n_folds), [str(fold) for fold in range(n_folds)])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Fold")
    axis.set_title("Deduplicated luminosity by class and fold (gray = no source group)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            annotation = (
                "no source" if int(source_rows.iloc[row, column]) == 0
                else _format_luminosity(values[row, column])
            )
            axis.text(
                column, row, annotation,
                ha="center", va="center", fontsize=8,
            )
    figure.colorbar(image, ax=axis, label=r"log10(luminosity / fb$^{-1}$)")
    save_figure(figure, path)
    plt.close(figure)


def generate_population_outputs(catalog: pd.DataFrame, config: dict,
                                output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_folds = int(config["cross_validation"]["n_folds"])
    population = aggregate_population(catalog, n_folds)
    metrics = fold_balance_metrics(population, n_folds)
    population.to_csv(output_dir / "population_table.csv", index=False)
    write_markdown_table(population, output_dir / "population_table.md")
    metrics.to_csv(output_dir / "fold_balance_metrics.csv", index=False)

    counts = _class_fold_matrix(population, "n_events")
    _annotated_heatmap(
        np.log10(counts.to_numpy(dtype=float) + 1),
        np.vectorize(lambda value: f"{int(value):,}")(counts.to_numpy()),
        "Natural event population by class and fold",
        "log10(events + 1)", output_dir / "population_by_class_and_fold.png",
    )
    class_totals = counts.sum(axis=1).replace(0, np.nan)
    fractions = counts.div(class_totals, axis=0).fillna(0)
    _annotated_heatmap(
        fractions.to_numpy(),
        np.vectorize(lambda value: f"{value:.1%}")(fractions.to_numpy()),
        "Per-class fraction assigned to each fold",
        "class fraction", output_dir / "class_fraction_by_fold.png",
        cmap="Blues",
    )
    ratios = fractions * n_folds
    _annotated_heatmap(
        ratios.to_numpy(),
        np.vectorize(lambda value: f"{value:.2f}")(ratios.to_numpy()),
        "Fold balance ratio (1 = ideal)",
        "balance ratio", output_dir / "fold_balance_ratio.png",
        cmap="coolwarm", center=1.0,
    )

    muons = population.loc[population["class_name"].isin(MUON_CLASSES)]
    domain = muons.pivot_table(
        index=["class_name", "fold"], columns="domain", values="n_events",
        aggfunc="sum", fill_value=0,
    )
    figure, axes = plt.subplots(1, len(MUON_CLASSES), figsize=(12, 4.5), sharey=False)
    for axis, class_name in zip(np.atleast_1d(axes), MUON_CLASSES):
        subset = domain.loc[class_name].reindex(range(n_folds), fill_value=0)
        bottom = np.zeros(n_folds)
        for domain_name in subset.columns:
            values = subset[domain_name].to_numpy(dtype=float)
            axis.bar(range(n_folds), values, bottom=bottom, label=domain_name)
            bottom += values
        axis.set_title(class_name)
        axis.set_xlabel("Fold")
        axis.set_ylabel("Natural events")
        axis.legend()
    figure.suptitle("Muon-class domain composition")
    save_figure(figure, output_dir / "domain_composition.png")
    plt.close(figure)

    groups = _class_fold_matrix(population, "source_groups")
    _annotated_heatmap(
        groups.to_numpy(dtype=float),
        np.vectorize(lambda value: str(int(value)))(groups.to_numpy()),
        "Independent source groups by class and fold",
        "source groups", output_dir / "source_groups_by_fold.png",
        cmap="magma",
    )
    plot_luminosity_by_class_and_fold(
        population, n_folds, output_dir / "luminosity_by_class_and_fold.png"
    )
    return {
        "population": population,
        "metrics": metrics,
        "warnings": population_warnings(catalog, metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folded-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_population_outputs(
        pd.read_csv(args.folded_catalog, keep_default_na=False),
        load_yaml(args.config), args.output_dir,
    )


if __name__ == "__main__":
    main()
