#!/usr/bin/env python3
"""Generate the complete validated data-driven 3D-CNN diagnostics report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .plot_muondis_density_sampling import generate_density_outputs
    from .plot_population_and_folds import generate_population_outputs
    from .plotting_common import (
        CLASS_ORDER, EXCLUDED_CLASSES, load_yaml, markdown_table, validate_report,
    )
except ImportError:  # Direct script execution.
    from plot_muondis_density_sampling import generate_density_outputs
    from plot_population_and_folds import generate_population_outputs
    from plotting_common import (
        CLASS_ORDER, EXCLUDED_CLASSES, load_yaml, markdown_table, validate_report,
    )


PLOTS = (
    "3d_event_displays/training_events_3d_cv0.png",
    "population_by_class_and_fold.png",
    "class_fraction_by_fold.png",
    "fold_balance_ratio.png",
    "domain_composition.png",
    "source_groups_by_fold.png",
    "luminosity_by_class_and_fold.png",
    "muondis_density_counts.png",
    "muondis_natural_vs_uniform.png",
    "muondis_sampling_multiplier.png",
    "muondis_density_by_fold.png",
)


def build_report(config_path: str, validation_path: str, folded_catalog_path: str,
                 density_manifest_path: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    validation = validate_report(validation_path)
    catalog = pd.read_csv(folded_catalog_path, keep_default_na=False)
    density_manifest = pd.read_csv(density_manifest_path, keep_default_na=False)
    population = generate_population_outputs(catalog, config, output_dir)
    density = generate_density_outputs(catalog, density_manifest, config, output_dir)
    warnings = [*population["warnings"], *density["warnings"]]

    lines = [
        "# Data-driven 3D-CNN dataset diagnostics",
        "",
        f"- Dataset version: `{validation['dataset_version']}`",
        "- Validation: **PASS**",
        f"- Training classes: **{len(CLASS_ORDER)}**",
        f"- Excluded class: `{EXCLUDED_CLASSES[0]}`",
        f"- Folds: **{validation['folds']}**",
        f"- Logical dataset rows: **{validation['dataset_rows']:,}**",
        f"- Muon-DIS density-index rows: **{validation['density_index_rows']:,}**",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None.")
    lines.extend([
        "",
        "## Training class order",
        "",
        *[f"{index}. `{name}`" for index, name in enumerate(CLASS_ORDER)],
        "",
        "## Fold-balance metrics",
        "",
        markdown_table(population["metrics"]),
        "## Muon-DIS density sampling",
        "",
        "Uniform-bin multipliers describe training exposure, not physics weights. ",
        "Validation remains natural and includes invalid-density events.",
        "",
        "## Plots",
        "",
        *[f"### {Path(plot).stem.replace('_', ' ').title()}\n\n![{plot}]({plot})\n" for plot in PLOTS],
    ])
    report = output_dir / "dataset_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--folded-catalog", required=True)
    parser.add_argument("--density-index-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = build_report(
        args.config, args.validation_report, args.folded_catalog,
        args.density_index_manifest, args.output_dir,
    )
    print(f"Wrote data-driven 3D-CNN diagnostics to {path}")


if __name__ == "__main__":
    main()
