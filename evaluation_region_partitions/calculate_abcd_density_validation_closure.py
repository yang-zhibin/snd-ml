#!/usr/bin/env python3
"""Calculate arithmetic ABCD closure for frozen validation constructions only."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROLES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--merge-audit", required=True)
    parser.add_argument("--merge-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-tex", required=True)
    parser.add_argument("--output-plot-dir", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_figure(fig: plt.Figure, path: str | Path, dpi: int = 180) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_scope(config: dict, merge_config: dict) -> tuple[dict, set[str]]:
    scope = config["scope"]
    required_false = ("signal_subtraction", "nonclosure_correction", "likelihood_fit")
    if not bool(scope.get("validation_only")) or any(bool(scope.get(key)) for key in required_false):
        raise ValueError("Closure configuration violates the validation-only scope")
    excluded = {str(region) for region in scope["excluded_regions"]}
    allowed = {
        str(item["region"]): str(item["label"])
        for item in merge_config["selection"]["regions"]
    }
    if not excluded or excluded != set(merge_config["selection"]["excluded_regions"]):
        raise ValueError("Closure exclusions do not match the frozen merge policy")
    for baseline, baseline_config in config["baselines"].items():
        roles = baseline_config["roles"]
        if set(roles) != set(ROLES):
            raise ValueError(f"{baseline} must define exactly A/B/C/D")
        regions = [str(roles[role]["region"]) for role in ROLES]
        if len(set(regions)) != 4 or any(region not in allowed for region in regions):
            raise ValueError(f"{baseline} uses invalid or repeated validation regions")
        if any(region in excluded for region in regions):
            raise ValueError(f"{baseline} includes an excluded nominal region")
    return allowed, excluded


def load_merge_audit(
    path: str | Path, policy_version: str, allowed: dict[str, str], edges: list[float]
) -> dict[str, list[int]]:
    counts: dict[str, dict[int, int]] = {region: {} for region in allowed}
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "policy_version",
            "merged_bin_index",
            "bin_low",
            "bin_high",
            "region",
            "region_label",
            "raw_data_count",
            "validation_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Merge audit is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            region = row["region"]
            if region not in allowed:
                raise ValueError(f"Unexpected region in merge audit: {region}")
            if row["policy_version"] != policy_version or row["validation_status"] != "PASS":
                raise ValueError("Merge audit has an invalid version or status")
            index = int(row["merged_bin_index"])
            if not math.isclose(float(row["bin_low"]), edges[index - 1], abs_tol=1e-9):
                raise ValueError("Merge-audit lower edge mismatch")
            if not math.isclose(float(row["bin_high"]), edges[index], abs_tol=1e-9):
                raise ValueError("Merge-audit upper edge mismatch")
            if row["region_label"] != allowed[region]:
                raise ValueError("Merge-audit region-label mismatch")
            counts[region][index] = int(row["raw_data_count"])
    expected = set(range(1, len(edges)))
    for region, region_counts in counts.items():
        if set(region_counts) != expected:
            raise ValueError(f"Incomplete merge audit for {region}")
    return {
        region: [region_counts[index] for index in range(1, len(edges))]
        for region, region_counts in counts.items()
    }


def load_validation_inputs(
    path: str | Path,
    table_version: str,
    allowed: dict[str, str],
    edges: list[float],
    audit_counts: dict[str, list[int]],
) -> dict[str, dict]:
    regular: dict[str, list[dict]] = {region: [] for region in allowed}
    underflow: dict[str, int] = {}
    inclusive: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "table_version",
            "region",
            "region_label",
            "row_type",
            "root_bin_index",
            "bin_low",
            "bin_high",
            "data_exposure",
            "data_raw_events",
            "cc_nue_sum_weights",
            "cc_nue_sum_weights_squared",
            "source_validation_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Analysis table is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            region = row["region"]
            if region not in allowed:
                continue
            if row["table_version"] != table_version or row["source_validation_status"] != "PASS":
                raise ValueError(f"Invalid source row for {region}")
            if row["region_label"] != allowed[region]:
                raise ValueError(f"Region-label mismatch for {region}")
            count_float = float(row["data_raw_events"])
            count = int(round(count_float))
            if count < 0 or not math.isclose(count_float, count, abs_tol=1e-9):
                raise ValueError(f"Invalid raw count for {region}")
            parsed = {
                "count": count,
                "exposure": float(row["data_exposure"]),
                "signal": float(row["cc_nue_sum_weights"]),
                "signal_sumw2": float(row["cc_nue_sum_weights_squared"]),
            }
            if any(not math.isfinite(value) or value < 0 for value in parsed.values()):
                raise ValueError(f"Invalid numeric input for {region}")
            if row["row_type"] == "bin":
                parsed.update(
                    {
                        "base_index": int(row["root_bin_index"]),
                        "low": float(row["bin_low"]),
                        "high": float(row["bin_high"]),
                    }
                )
                regular[region].append(parsed)
            elif row["row_type"] == "underflow":
                underflow[region] = count
            elif row["row_type"] == "inclusive":
                inclusive[region] = parsed

    result: dict[str, dict] = {}
    for region in allowed:
        rows = sorted(regular[region], key=lambda row: row["base_index"])
        if len(rows) != 80 or region not in underflow or region not in inclusive:
            raise ValueError(f"Incomplete analysis-table inputs for {region}")
        if underflow[region] != 0:
            raise ValueError(f"Nonzero sideband underflow is excluded by policy for {region}")
        exposures = {row["exposure"] for row in rows} | {inclusive[region]["exposure"]}
        if len(exposures) != 1:
            raise ValueError(f"Inconsistent exposure for {region}")
        merged = []
        for index, (low, high) in enumerate(zip(edges, edges[1:])):
            selected = [row for row in rows if row["low"] >= low and row["high"] <= high]
            if not selected or not math.isclose(selected[0]["low"], low, abs_tol=1e-9):
                raise ValueError(f"Missing base bins for {region}, [{low}, {high})")
            if not math.isclose(selected[-1]["high"], high, abs_tol=1e-9):
                raise ValueError(f"Incomplete base-bin coverage for {region}, [{low}, {high})")
            item = {
                "count": sum(row["count"] for row in selected),
                "signal": sum(row["signal"] for row in selected),
                "signal_sumw2": sum(row["signal_sumw2"] for row in selected),
            }
            if item["count"] != audit_counts[region][index]:
                raise ValueError(f"Merged raw count disagrees with frozen audit for {region}")
            merged.append(item)
        if sum(item["count"] for item in merged) != inclusive[region]["count"]:
            raise ValueError(f"Inclusive raw-count reconstruction failed for {region}")
        if not math.isclose(
            sum(item["signal"] for item in merged),
            inclusive[region]["signal"],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Inclusive signal-yield reconstruction failed for {region}")
        result[region] = {
            "label": allowed[region],
            "exposure": exposures.pop(),
            "bins": merged,
            "inclusive": {
                "count": sum(item["count"] for item in merged),
                "signal": sum(item["signal"] for item in merged),
                "signal_sumw2": sum(item["signal_sumw2"] for item in merged),
            },
        }
    return result


def arithmetic_result(counts: dict[str, int], exposures: dict[str, float]) -> dict:
    if any(counts[role] <= 0 for role in ROLES):
        raise ValueError(f"Arithmetic closure requires positive A/B/C/D counts: {counts}")
    exposure_factor = (
        exposures["A"] * exposures["D"] / (exposures["B"] * exposures["C"])
    )
    transfer = counts["C"] / counts["D"] * exposures["D"] / exposures["C"]
    transfer_rel_var = 1.0 / counts["C"] + 1.0 / counts["D"]
    predicted = counts["B"] * counts["C"] / counts["D"] * exposure_factor
    predicted_rel_var = 1.0 / counts["B"] + transfer_rel_var
    predicted_variance = predicted * predicted * predicted_rel_var
    kappa = counts["A"] / predicted
    kappa_rel_var = 1.0 / counts["A"] + predicted_rel_var
    pull_denominator = math.sqrt(counts["A"] + predicted_variance)
    return {
        "exposure_factor": exposure_factor,
        "transfer_factor": transfer,
        "transfer_factor_stat_uncertainty": transfer * math.sqrt(transfer_rel_var),
        "predicted": predicted,
        "prediction_stat_uncertainty": math.sqrt(predicted_variance),
        "prediction_variance": predicted_variance,
        "kappa": kappa,
        "kappa_stat_uncertainty": kappa * math.sqrt(kappa_rel_var),
        "fractional_nonclosure": kappa - 1.0,
        "pull": (counts["A"] - predicted) / pull_denominator,
    }


def add_common_fields(
    result: dict,
    baseline: str,
    baseline_label: str,
    row_type: str,
    bin_index: str,
    bin_low: str,
    bin_high: str,
    role_regions: dict[str, str],
    inputs: dict[str, dict],
    role_values: dict[str, dict],
) -> dict:
    output = {
        "baseline": baseline,
        "baseline_label": baseline_label,
        "row_type": row_type,
        "merged_bin_index": bin_index,
        "bin_low": bin_low,
        "bin_high": bin_high,
        **result,
        "calculation_status": "PASS",
    }
    for role in ROLES:
        region = role_regions[role]
        value = role_values[role]
        output[f"region_{role.lower()}"] = region
        output[f"region_{role.lower()}_label"] = inputs[region]["label"]
        output[f"exposure_{role.lower()}"] = inputs[region]["exposure"]
        output[f"count_{role.lower()}"] = value["count"]
        output[f"signal_yield_{role.lower()}"] = value["signal"]
        output[f"signal_stat_uncertainty_{role.lower()}"] = math.sqrt(
            value["signal_sumw2"]
        )
        output[f"signal_fraction_of_data_{role.lower()}"] = (
            value["signal"] / value["count"] if value["count"] else 0.0
        )
    return output


def calculate_results(
    config: dict, edges: list[float], inputs: dict[str, dict]
) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {}
    for baseline, baseline_config in config["baselines"].items():
        role_regions = {
            role: str(baseline_config["roles"][role]["region"]) for role in ROLES
        }
        exposures = {role: inputs[region]["exposure"] for role, region in role_regions.items()}
        bin_results = []
        for index, (low, high) in enumerate(zip(edges, edges[1:]), start=1):
            role_values = {
                role: inputs[region]["bins"][index - 1]
                for role, region in role_regions.items()
            }
            counts = {role: value["count"] for role, value in role_values.items()}
            arithmetic = arithmetic_result(counts, exposures)
            bin_results.append(
                add_common_fields(
                    arithmetic,
                    baseline,
                    str(baseline_config["label"]),
                    "bin",
                    str(index),
                    f"{low:.12g}",
                    f"{high:.12g}",
                    role_regions,
                    inputs,
                    role_values,
                )
            )

        inclusive_values = {
            role: inputs[region]["inclusive"] for role, region in role_regions.items()
        }
        inclusive_counts = {
            role: value["count"] for role, value in inclusive_values.items()
        }
        direct = add_common_fields(
            arithmetic_result(inclusive_counts, exposures),
            baseline,
            str(baseline_config["label"]),
            "inclusive_direct",
            "",
            f"{edges[0]:.12g}",
            f"{edges[-1]:.12g}",
            role_regions,
            inputs,
            inclusive_values,
        )

        binned_prediction = sum(row["predicted"] for row in bin_results)
        binned_variance = sum(row["prediction_variance"] for row in bin_results)
        observed = inclusive_counts["A"]
        binned_kappa = observed / binned_prediction
        binned_kappa_rel_var = 1.0 / observed + binned_variance / (binned_prediction**2)
        binned_arithmetic = {
            "exposure_factor": "",
            "transfer_factor": "",
            "transfer_factor_stat_uncertainty": "",
            "predicted": binned_prediction,
            "prediction_stat_uncertainty": math.sqrt(binned_variance),
            "prediction_variance": binned_variance,
            "kappa": binned_kappa,
            "kappa_stat_uncertainty": binned_kappa * math.sqrt(binned_kappa_rel_var),
            "fractional_nonclosure": binned_kappa - 1.0,
            "pull": (observed - binned_prediction) / math.sqrt(observed + binned_variance),
        }
        binned_sum = add_common_fields(
            binned_arithmetic,
            baseline,
            str(baseline_config["label"]),
            "binned_sum",
            "",
            f"{edges[0]:.12g}",
            f"{edges[-1]:.12g}",
            role_regions,
            inputs,
            inclusive_values,
        )
        all_results[baseline] = [*bin_results, direct, binned_sum]
    return all_results


def csv_content(results: dict[str, list[dict]]) -> str:
    rows = [row for baseline_rows in results.values() for row in baseline_rows]
    fieldnames = list(rows[0])
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def number(value: float | str, significant: int = 6) -> str:
    if value == "":
        return "--"
    numeric = float(value)
    if numeric == 0:
        return "0"
    if abs(numeric) >= 10000 or abs(numeric) < 0.001:
        return f"{numeric:.{significant - 1}e}"
    return f"{numeric:.{significant}g}"


def interval(row: dict) -> str:
    return f"[{float(row['bin_low']):g}, {float(row['bin_high']):g})"


def make_plots(
    results: dict[str, list[dict]], output_plot_dir: str | Path
) -> dict[str, Path]:
    output_directory = Path(output_plot_dir)
    paths = {}
    for baseline, rows in results.items():
        bins = [row for row in rows if row["row_type"] == "bin"]
        x = list(range(len(bins)))
        labels = [interval(row) for row in bins]
        observed = [row["count_a"] for row in bins]
        predicted = [row["predicted"] for row in bins]
        predicted_error = [row["prediction_stat_uncertainty"] for row in bins]
        kappas = [row["kappa"] for row in bins]
        kappa_error = [row["kappa_stat_uncertainty"] for row in bins]
        transfer = [row["transfer_factor"] for row in bins]
        transfer_error = [row["transfer_factor_stat_uncertainty"] for row in bins]

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10.5, 9.0),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.15, 1.15], "hspace": 0.08},
        )
        axes[0].errorbar(
            x,
            observed,
            yerr=[math.sqrt(value) for value in observed],
            fmt="o",
            color="black",
            label="Observed validation A",
        )
        axes[0].errorbar(
            x,
            predicted,
            yerr=predicted_error,
            fmt="s",
            color="#d62728",
            label="ABCD prediction",
        )
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Events")
        axes[0].set_title(rows[0]["baseline_label"] + " (arithmetic, validation only)")
        axes[0].legend(frameon=False)
        axes[0].grid(True, which="both", axis="y", alpha=0.25)

        axes[1].errorbar(x, kappas, yerr=kappa_error, fmt="o", color="#1f77b4")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axes[1].set_ylabel(r"$\kappa=A_{obs}/A_{pred}$")
        axes[1].grid(True, axis="y", alpha=0.25)

        axes[2].errorbar(x, transfer, yerr=transfer_error, fmt="o", color="#2ca02c")
        axes[2].set_ylabel("Transfer factor")
        axes[2].set_xlabel(r"density_sndsw_scifi merged interval")
        axes[2].set_xticks(x, labels, rotation=25, ha="right")
        axes[2].grid(True, axis="y", alpha=0.25)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.16, hspace=0.08)
        png_destination = output_directory / f"{baseline}.png"
        pdf_destination = output_directory / f"{baseline}.pdf"
        atomic_figure(fig, png_destination)
        atomic_figure(fig, pdf_destination)
        plt.close(fig)
        paths[baseline] = pdf_destination.resolve()
    return paths


def markdown_content(
    config: dict,
    results: dict[str, list[dict]],
    plot_paths: dict[str, Path],
    input_table: str,
    merge_audit: str,
) -> str:
    lines = [
        f"# {config['version']}",
        "",
        "Status: **PASS**",
        "",
        "Scope: arithmetic closure for the two validation constructions only. The nominal signal region was excluded; no signal subtraction, nonclosure correction, bin retuning, or likelihood fit was performed.",
        "",
        f"Source table: `{Path(input_table).resolve()}`",
        "",
        f"Frozen merge audit: `{Path(merge_audit).resolve()}`",
        "",
        "The prediction is `A_pred = (B*C/D)*(L_A*L_D)/(L_B*L_C)`. The reported uncertainties and pulls use first-order independent-Poisson propagation and are statistical diagnostics, not likelihood intervals.",
        "",
        "## Inclusive summary",
        "",
        "| Validation | Summary | A observed | A predicted | Stat. uncertainty | Kappa | Kappa stat. uncertainty | Nonclosure | Pull |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rows in results.values():
        for row in rows:
            if row["row_type"] not in {"inclusive_direct", "binned_sum"}:
                continue
            summary = "Direct inclusive ABCD" if row["row_type"] == "inclusive_direct" else "Sum of binned predictions"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["baseline_label"],
                        summary,
                        f"{row['count_a']:,}",
                        number(row["predicted"]),
                        number(row["prediction_stat_uncertainty"]),
                        number(row["kappa"]),
                        number(row["kappa_stat_uncertainty"]),
                        f"{100.0 * row['fractional_nonclosure']:.3f}%",
                        number(row["pull"]),
                    ]
                )
                + " |"
            )
    for baseline, rows in results.items():
        bins = [row for row in rows if row["row_type"] == "bin"]
        lines.extend(
            [
                "",
                f"## {rows[0]['baseline_label']}",
                "",
                f"Mapping: A={rows[0]['region_a_label']}, B={rows[0]['region_b_label']}, C={rows[0]['region_c_label']}, D={rows[0]['region_d_label']}.",
                "",
                "| Density interval | A observed | A predicted | Prediction stat. uncertainty | Transfer factor | Kappa | Kappa stat. uncertainty | Nonclosure | Pull |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bins:
            lines.append(
                "| "
                + " | ".join(
                    [
                        interval(row),
                        f"{row['count_a']:,}",
                        number(row["predicted"]),
                        number(row["prediction_stat_uncertainty"]),
                        number(row["transfer_factor"]),
                        number(row["kappa"]),
                        number(row["kappa_stat_uncertainty"]),
                        f"{100.0 * row['fractional_nonclosure']:.3f}%",
                        number(row["pull"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"Plot: `{plot_paths[baseline]}`",
                "",
                "Inclusive signal-MC contamination diagnostics (not subtracted):",
                "",
                "| Role | Region | Signal yield | MC stat. uncertainty | Signal/Data |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        inclusive = next(row for row in rows if row["row_type"] == "inclusive_direct")
        for role in ROLES:
            key = role.lower()
            lines.append(
                f"| {role} | {inclusive[f'region_{key}_label']} | "
                f"{number(inclusive[f'signal_yield_{key}'])} | "
                f"{number(inclusive[f'signal_stat_uncertainty_{key}'])} | "
                f"{100.0 * inclusive[f'signal_fraction_of_data_{key}']:.6g}% |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These arithmetic values quantify validation nonclosure. They do not authorize a correction or a nominal signal-region prediction. Sparse/non-Gaussian inference and nuisance parameters require the later simultaneous likelihood step.",
            "",
        ]
    )
    return "\n".join(lines)


def tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def latex_content(
    config: dict, results: dict[str, list[dict]], plot_paths: dict[str, Path]
) -> str:
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\usepackage{booktabs,array,pdflscape,graphicx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{xcolor}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        r"\title{Region V1 ABCD Validation Closure}",
        r"\author{SND@LHC analysis workflow}",
        r"\date{}",
        r"\maketitle",
        r"\begin{center}\fcolorbox{black}{gray!10}{\parbox{0.92\textwidth}{\textbf{Validation-only arithmetic result.} The nominal signal region is excluded. No signal subtraction, nonclosure correction, bin retuning, or likelihood fit is included.}}\end{center}",
        r"\begin{landscape}",
        r"\section{Inclusive summary}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Validation & Summary & $A_{obs}$ & $A_{pred}\pm\sigma$ & $\kappa\pm\sigma$ & Nonclosure & Pull \\",
        r"\midrule",
    ]
    for rows in results.values():
        for row in rows:
            if row["row_type"] not in {"inclusive_direct", "binned_sum"}:
                continue
            summary = "Direct inclusive" if row["row_type"] == "inclusive_direct" else "Binned sum"
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["baseline_label"].replace(" validation", "")),
                        summary,
                        f"{row['count_a']:,}",
                        f"{number(row['predicted'])} $\\pm$ {number(row['prediction_stat_uncertainty'])}",
                        f"{number(row['kappa'])} $\\pm$ {number(row['kappa_stat_uncertainty'])}",
                        f"{100.0 * row['fractional_nonclosure']:.3f}\\%",
                        number(row["pull"]),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{landscape}"])
    for baseline, rows in results.items():
        bins = [row for row in rows if row["row_type"] == "bin"]
        lines.extend(
            [
                r"\clearpage",
                r"\begin{landscape}",
                rf"\section{{{tex_escape(rows[0]['baseline_label'])}}}",
                r"\scriptsize",
                rf"Mapping: $A={rows[0]['region_a_label']}$, $B={rows[0]['region_b_label']}$, $C={rows[0]['region_c_label']}$, $D={rows[0]['region_d_label']}$. ",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Density interval & $A_{obs}$ & $A_{pred}\pm\sigma$ & Transfer & $\kappa\pm\sigma$ & Nonclosure & Pull \\",
                r"\midrule",
            ]
        )
        for row in bins:
            lines.append(
                " & ".join(
                    [
                        "{" + tex_escape(interval(row)) + "}",
                        f"{row['count_a']:,}",
                        f"{number(row['predicted'])} $\\pm$ {number(row['prediction_stat_uncertainty'])}",
                        number(row["transfer_factor"]),
                        f"{number(row['kappa'])} $\\pm$ {number(row['kappa_stat_uncertainty'])}",
                        f"{100.0 * row['fractional_nonclosure']:.3f}\\%",
                        number(row["pull"]),
                    ]
                )
                + r" \\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\begin{center}",
                rf"\includegraphics[width=0.72\linewidth]{{{plot_paths[baseline]}}}",
                r"\end{center}",
                r"\small The arithmetic prediction is $A_{pred}=(BC/D)(L_A L_D)/(L_B L_C)$. Uncertainties use first-order independent-Poisson propagation. Signal MC is diagnostic only and is not subtracted.",
                r"\end{landscape}",
            ]
        )
    lines.extend(
        [
            r"\section{Calculation definitions and boundary}",
            r"For each frozen density bin, $T=(C/D)(L_D/L_C)$ and $A_{pred}=B(L_A/L_B)T$. The closure factor is $\kappa=A_{obs}/A_{pred}$, fractional nonclosure is $\kappa-1$, and the diagnostic pull denominator is $\sqrt{A_{obs}+\sigma_{pred}^{2}}$.",
            r"The direct-inclusive result applies the ABCD formula to region totals. The binned-sum result sums the six independently calculated bin predictions and their approximate variances.",
            r"These results quantify validation nonclosure only. They do not constitute a nominal signal-region prediction, a nonclosure correction, or a likelihood result.",
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    merge_config = load_yaml(args.merge_config)
    if config["source"]["merge_policy_version"] != merge_config["version"]:
        raise ValueError("Closure and merge-policy versions do not match")
    allowed, _ = validate_scope(config, merge_config)
    edges = [float(edge) for edge in merge_config["merged_regular_bin_edges"]]
    audit_counts = load_merge_audit(args.merge_audit, merge_config["version"], allowed, edges)
    inputs = load_validation_inputs(
        args.input_table,
        str(config["source"]["table_version"]),
        allowed,
        edges,
        audit_counts,
    )
    results = calculate_results(config, edges, inputs)
    plot_paths = make_plots(results, args.output_plot_dir)
    atomic_text(args.output_csv, csv_content(results))
    atomic_text(
        args.output_markdown,
        markdown_content(config, results, plot_paths, args.input_table, args.merge_audit),
    )
    atomic_text(args.output_tex, latex_content(config, results, plot_paths))
    print(
        f"Calculated {len(config['baselines'])} validation constructions with "
        f"{len(edges) - 1} frozen bins; nominal signal region excluded"
    )


if __name__ == "__main__":
    main()
