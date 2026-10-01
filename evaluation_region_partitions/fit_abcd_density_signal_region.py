#!/usr/bin/env python3
"""Fit a global ABCD background scale and CC_nue signal with cabinetry."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import tempfile
from io import StringIO
from pathlib import Path

import cabinetry
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import yaml
from cabinetry.fit.results_containers import FitResults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-inputs-csv", required=True)
    parser.add_argument("--output-workspace-json", required=True)
    parser.add_argument("--output-fit-json", required=True)
    parser.add_argument("--output-mu-scan-csv", required=True)
    parser.add_argument("--output-alpha-scan-csv", required=True)
    parser.add_argument("--output-likelihood-2d-csv", required=True)
    parser.add_argument("--output-report-markdown", required=True)
    parser.add_argument("--output-report-tex", required=True)
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


def atomic_figure(figure: plt.Figure, path: str | Path, dpi: int = 180) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_text(rows: list[dict], fieldnames: list[str] | None = None) -> str:
    if not rows:
        raise ValueError("Cannot serialize an empty CSV")
    columns = fieldnames or list(rows[0])
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def finite_float(row: dict, key: str, *, nonnegative: bool = True) -> float:
    value = float(row[key])
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {key}: {row[key]}")
    return value


def load_model_inputs(path: str | Path, config: dict) -> list[dict]:
    source = config["source"]
    expected_roles = {role.lower(): region for role, region in source["expected_roles"].items()}
    edges = [float(edge) for edge in source["merged_regular_bin_edges"]]
    required = {
        "baseline",
        "row_type",
        "merged_bin_index",
        "bin_low",
        "bin_high",
        "predicted",
        "prediction_stat_uncertainty",
        "calculation_status",
        "count_a",
        "signal_yield_a",
        "signal_stat_uncertainty_a",
        "region_a",
        "region_b",
        "region_c",
        "region_d",
    }
    selected = []
    binned_sum = None
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Closure CSV is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            if row["baseline"] != source["baseline"]:
                continue
            if row["calculation_status"] != "PASS":
                raise ValueError("Selected closure row does not have PASS status")
            for role, expected_region in expected_roles.items():
                if row[f"region_{role}"] != expected_region:
                    raise ValueError(f"Unexpected region mapping for role {role.upper()}")
            if row["row_type"] == source["row_type"]:
                selected.append(row)
            elif row["row_type"] == "binned_sum":
                binned_sum = row

    selected.sort(key=lambda row: int(row["merged_bin_index"]))
    if len(selected) != len(edges) - 1 or binned_sum is None:
        raise ValueError("Expected six bin rows and one binned-sum row")

    inputs = []
    for index, row in enumerate(selected, start=1):
        if int(row["merged_bin_index"]) != index:
            raise ValueError("Merged bin indices are not contiguous")
        low = finite_float(row, "bin_low")
        high = finite_float(row, "bin_high")
        if not math.isclose(low, edges[index - 1], abs_tol=1e-9) or not math.isclose(
            high, edges[index], abs_tol=1e-9
        ):
            raise ValueError("Closure bins do not match the frozen configured edges")
        observed_float = finite_float(row, "count_a")
        observed = int(round(observed_float))
        if not math.isclose(observed_float, observed, abs_tol=1e-9):
            raise ValueError("Observed data are not integer raw counts")
        background = finite_float(row, "predicted")
        signal = finite_float(row, "signal_yield_a")
        if background <= 0:
            raise ValueError("ABCD background template must be strictly positive")
        inputs.append(
            {
                "bin_index": index,
                "bin_low": low,
                "bin_high": high,
                "observed_data": observed,
                "abcd_background": background,
                "abcd_stat_uncertainty_diagnostic": finite_float(
                    row, "prediction_stat_uncertainty"
                ),
                "cc_nue_signal": signal,
                "cc_nue_mc_stat_uncertainty_diagnostic": finite_float(
                    row, "signal_stat_uncertainty_a"
                ),
            }
        )

    expected_sums = {
        "observed_data": int(round(finite_float(binned_sum, "count_a"))),
        "abcd_background": finite_float(binned_sum, "predicted"),
        "cc_nue_signal": finite_float(binned_sum, "signal_yield_a"),
    }
    for key, expected in expected_sums.items():
        actual = sum(row[key] for row in inputs)
        if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9):
            raise ValueError(f"{key} does not reproduce the binned-sum row")
    return inputs


def parameter_config(config: dict, name: str) -> dict:
    item = config["model"]["parameters"][name]
    bounds = [float(value) for value in item["bounds"]]
    initial = float(item["initial"])
    if len(bounds) != 2 or bounds[0] >= bounds[1] or not bounds[0] <= initial <= bounds[1]:
        raise ValueError(f"Invalid parameter configuration for {name}")
    return {"name": name, "inits": [initial], "bounds": [bounds]}


def build_workspace(config: dict, inputs: list[dict]) -> dict:
    model_config = config["model"]
    if model_config.get("include_template_uncertainties"):
        raise ValueError("The v1 diagnostic must not enable template uncertainties")
    if model_config.get("background_scale_scope") != "global_across_six_bins":
        raise ValueError("The v1 diagnostic requires one global background scale")
    alpha_name = str(model_config["background_parameter"])
    mu_name = str(model_config["signal_parameter"])
    if model_config["parameter_of_interest"] != mu_name or alpha_name == mu_name:
        raise ValueError("Invalid parameter-of-interest configuration")
    return {
        "channels": [
            {
                "name": str(model_config["channel"]),
                "samples": [
                    {
                        "name": str(model_config["background_sample"]),
                        "data": [row["abcd_background"] for row in inputs],
                        "modifiers": [
                            {"name": alpha_name, "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": str(model_config["signal_sample"]),
                        "data": [row["cc_nue_signal"] for row in inputs],
                        "modifiers": [
                            {"name": mu_name, "type": "normfactor", "data": None}
                        ],
                    },
                ],
            }
        ],
        "observations": [
            {
                "name": str(model_config["channel"]),
                "data": [row["observed_data"] for row in inputs],
            }
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {
                    "poi": mu_name,
                    "parameters": [
                        parameter_config(config, mu_name),
                        parameter_config(config, alpha_name),
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def serialize_fit_results(
    config: dict,
    inputs: list[dict],
    model: pyhf.pdf.Model,
    fit_results,
) -> dict:
    boundary_tolerance = float(config["fit"]["boundary_tolerance"])
    configured = config["model"]["parameters"]
    parameters = []
    for index, name in enumerate(fit_results.labels):
        lower, upper = [float(value) for value in configured[name]["bounds"]]
        value = float(fit_results.bestfit[index])
        parameters.append(
            {
                "name": name,
                "bestfit": value,
                "uncertainty_hesse": float(fit_results.uncertainty[index]),
                "lower_bound": lower,
                "upper_bound": upper,
                "at_lower_bound": abs(value - lower) <= boundary_tolerance,
                "at_upper_bound": abs(value - upper) <= boundary_tolerance,
            }
        )
    parameter_values = {item["name"]: item["bestfit"] for item in parameters}
    alpha = parameter_values[config["model"]["background_parameter"]]
    mu = parameter_values[config["model"]["signal_parameter"]]
    return {
        "analysis_version": config["version"],
        "status": "FIT_RETURNED",
        "scope": "explicitly unblinded diagnostic; no template uncertainties",
        "software": {
            "cabinetry": importlib.metadata.version("cabinetry"),
            "pyhf": importlib.metadata.version("pyhf"),
        },
        "model": {
            "channels": list(model.config.channels),
            "samples": list(model.config.samples),
            "parameter_order": list(model.config.par_order),
            "parameter_of_interest": model.config.poi_name,
            "n_bins": model.config.nmaindata,
        },
        "input_sums": {
            "observed_data": sum(row["observed_data"] for row in inputs),
            "abcd_background": sum(row["abcd_background"] for row in inputs),
            "cc_nue_signal": sum(row["cc_nue_signal"] for row in inputs),
        },
        "parameters": parameters,
        "correlation_matrix": {
            "labels": list(fit_results.labels),
            "values": np.asarray(fit_results.corr_mat, dtype=float).tolist(),
        },
        "best_twice_nll": float(fit_results.best_twice_nll),
        "goodness_of_fit_p_value": float(fit_results.goodness_of_fit),
        "postfit_sums": {
            "background": alpha * sum(row["abcd_background"] for row in inputs),
            "signal": mu * sum(row["cc_nue_signal"] for row in inputs),
            "total": alpha * sum(row["abcd_background"] for row in inputs)
            + mu * sum(row["cc_nue_signal"] for row in inputs),
        },
    }


def scan_rows(scan_results) -> list[dict]:
    return [
        {
            "parameter": scan_results.name,
            "parameter_value": float(value),
            "delta_twice_nll": float(delta),
        }
        for value, delta in zip(scan_results.parameter_values, scan_results.delta_nlls)
    ]


def two_dimensional_likelihood(
    config: dict,
    model: pyhf.pdf.Model,
    data: list[float],
    fit_results,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    alpha_name = config["model"]["background_parameter"]
    mu_name = config["model"]["signal_parameter"]
    labels = list(fit_results.labels)
    alpha_index = labels.index(alpha_name)
    mu_index = labels.index(mu_name)
    two_d = config["scans"]["two_dimensional"]
    alpha_best = float(fit_results.bestfit[alpha_index])
    alpha_uncertainty = float(fit_results.uncertainty[alpha_index])
    alpha_bounds = [float(value) for value in config["model"]["parameters"][alpha_name]["bounds"]]
    span = float(two_d["alpha_uncertainty_span"]) * alpha_uncertainty
    alpha_low = max(alpha_bounds[0], alpha_best - span)
    alpha_high = min(alpha_bounds[1], alpha_best + span)
    if not alpha_low < alpha_high:
        raise ValueError("Could not construct a finite alpha range for the 2D scan")
    mu_low, mu_high = [float(value) for value in two_d["mu_range"]]
    alpha_values = np.linspace(alpha_low, alpha_high, int(two_d["alpha_points"]))
    mu_values = np.linspace(mu_low, mu_high, int(two_d["mu_points"]))
    delta = np.empty((len(mu_values), len(alpha_values)), dtype=float)
    rows = []
    best_twice_nll = float(fit_results.best_twice_nll)
    for mu_position, mu in enumerate(mu_values):
        for alpha_position, alpha in enumerate(alpha_values):
            parameters = np.asarray(fit_results.bestfit, dtype=float).copy()
            parameters[alpha_index] = alpha
            parameters[mu_index] = mu
            twice_nll = float(
                np.asarray(pyhf.infer.mle.twice_nll(parameters, data, model)).reshape(-1)[0]
            )
            value = max(0.0, twice_nll - best_twice_nll)
            delta[mu_position, alpha_position] = value
            rows.append(
                {
                    alpha_name: alpha,
                    mu_name: mu,
                    "twice_nll": twice_nll,
                    "delta_twice_nll": value,
                }
            )
    return rows, alpha_values, mu_values, delta


def save_cabinetry_plots(
    output_directory: Path,
    model: pyhf.pdf.Model,
    data: list[float],
    fit_results,
    mu_scan,
    alpha_scan,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_labels = list(model.config.par_names)
    before_values = np.asarray(model.config.suggested_init(), dtype=float)
    before_values[parameter_labels.index("alpha_bkg")] = 1.0
    before_values[parameter_labels.index("mu")] = 0.0
    before_fit_results = FitResults(
        bestfit=before_values,
        uncertainty=np.zeros_like(before_values),
        labels=parameter_labels,
        corr_mat=np.eye(len(parameter_labels)),
        best_twice_nll=float("nan"),
        goodness_of_fit=None,
        minos_uncertainty={},
    )
    before_fit = cabinetry.model_utils.prediction(
        model, fit_results=before_fit_results, label="Before fit: raw ABCD"
    )
    prefit = cabinetry.model_utils.prediction(model, label="Pre-fit")
    postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results, label="Post-fit")
    for label, prediction in (
        ("before_fit", before_fit),
        ("prefit", prefit),
        ("postfit", postfit),
    ):
        plots = cabinetry.visualize.data_mc(
            prediction, data, save_figure=False, close_figure=False
        )
        if not plots or len(plots) != 1:
            raise ValueError("Expected exactly one cabinetry data/MC figure")
        figure = plots[0]["figure"]
        atomic_figure(figure, output_directory / f"{label}_data_model.pdf")
        atomic_figure(figure, output_directory / f"{label}_data_model.png")
        plt.close(figure)

    correlation = cabinetry.visualize.correlation_matrix(
        fit_results, save_figure=False, close_figure=False
    )
    atomic_figure(correlation, output_directory / "correlation_matrix.pdf")
    atomic_figure(correlation, output_directory / "correlation_matrix.png")
    plt.close(correlation)

    for label, result in (("mu", mu_scan), ("alpha_bkg", alpha_scan)):
        figure = cabinetry.visualize.scan(result, save_figure=False, close_figure=False)
        atomic_figure(figure, output_directory / f"likelihood_scan_{label}.pdf")
        atomic_figure(figure, output_directory / f"likelihood_scan_{label}.png")
        plt.close(figure)


def save_two_dimensional_plot(
    output_directory: Path,
    config: dict,
    alpha_values: np.ndarray,
    mu_values: np.ndarray,
    delta: np.ndarray,
    fit_results,
) -> None:
    alpha_name = config["model"]["background_parameter"]
    mu_name = config["model"]["signal_parameter"]
    labels = list(fit_results.labels)
    figure, axis = plt.subplots(figsize=(7.5, 5.8))
    mesh = axis.pcolormesh(alpha_values, mu_values, delta, shading="auto", cmap="viridis")
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(r"$\Delta(-2\log L)$")
    available_levels = [level for level in (2.30, 5.99) if level < float(np.max(delta))]
    if available_levels:
        contours = axis.contour(
            alpha_values,
            mu_values,
            delta,
            levels=available_levels,
            colors="white",
            linewidths=1.2,
        )
        axis.clabel(contours, inline=True, fontsize=9, fmt="%.2f")
    axis.plot(
        fit_results.bestfit[labels.index(alpha_name)],
        fit_results.bestfit[labels.index(mu_name)],
        marker="*",
        markersize=12,
        color="red",
        label="best fit",
    )
    axis.set_xlabel(alpha_name)
    axis.set_ylabel(mu_name)
    axis.set_title("Two-parameter likelihood surface")
    axis.legend(frameon=False)
    figure.tight_layout()
    atomic_figure(figure, output_directory / "likelihood_2d.pdf")
    atomic_figure(figure, output_directory / "likelihood_2d.png")
    plt.close(figure)


def markdown_report(
    config: dict,
    inputs: list[dict],
    fit_payload: dict,
    workspace_path: str,
    closure_path: str,
) -> str:
    parameters = {item["name"]: item for item in fit_payload["parameters"]}
    alpha_name = config["model"]["background_parameter"]
    mu_name = config["model"]["signal_parameter"]
    alpha = parameters[alpha_name]
    mu = parameters[mu_name]
    labels = fit_payload["correlation_matrix"]["labels"]
    correlation = fit_payload["correlation_matrix"]["values"]
    alpha_index = labels.index(alpha_name)
    mu_index = labels.index(mu_name)
    lines = [
        f"# {config['version']}",
        "",
        "Status: **FIT RETURNED**",
        "",
        "Scope: explicitly unblinded six-bin diagnostic. One global ABCD background scale and one CC_nue signal-strength parameter are fitted simultaneously. Template uncertainties are intentionally absent in this first model.",
        "",
        f"Source closure table: `{Path(closure_path).resolve()}`",
        "",
        f"Workspace: `{Path(workspace_path).resolve()}`",
        "",
        "Model: `nu_i = alpha_bkg * b_i_ABCD + mu * s_i_CC_nue`.",
        "",
        "## Input templates",
        "",
        "| Density interval | Observed data | ABCD background | CC_nue signal |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in inputs:
        lines.append(
            f"| [{row['bin_low']:g}, {row['bin_high']:g}) | {row['observed_data']} | "
            f"{row['abcd_background']:.9g} | {row['cc_nue_signal']:.9g} |"
        )
    sums = fit_payload["input_sums"]
    lines.extend(
        [
            f"| **Total** | **{sums['observed_data']}** | **{sums['abcd_background']:.9g}** | **{sums['cc_nue_signal']:.9g}** |",
            "",
            "## Fit result",
            "",
            "| Parameter | Best fit | Hesse uncertainty | Bounds | At boundary |",
            "| --- | ---: | ---: | --- | --- |",
            f"| `{alpha_name}` | {alpha['bestfit']:.9g} | {alpha['uncertainty_hesse']:.9g} | [{alpha['lower_bound']:g}, {alpha['upper_bound']:g}] | {alpha['at_lower_bound'] or alpha['at_upper_bound']} |",
            f"| `{mu_name}` | {mu['bestfit']:.9g} | {mu['uncertainty_hesse']:.9g} | [{mu['lower_bound']:g}, {mu['upper_bound']:g}] | {mu['at_lower_bound'] or mu['at_upper_bound']} |",
            "",
            f"- Best-fit twice negative log-likelihood: `{fit_payload['best_twice_nll']:.9g}`.",
            f"- Goodness-of-fit p-value reported by cabinetry: `{fit_payload['goodness_of_fit_p_value']:.9g}`.",
            f"- Correlation `corr({alpha_name}, {mu_name})`: `{correlation[alpha_index][mu_index]:.9g}`.",
            f"- Post-fit background total: `{fit_payload['postfit_sums']['background']:.9g}`.",
            f"- Post-fit signal total: `{fit_payload['postfit_sums']['signal']:.9g}`.",
            "",
            "## Interpretation boundary",
            "",
            "The signal-region data determine the freely floating background normalization. Consequently, normalization information does not independently constrain the signal; only the assumed ABCD-versus-signal shape difference does. The ABCD shape has not passed validation, and the signal region was used during development. This is a diagnostic likelihood fit, not a discovery significance or limit result.",
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


def latex_row(*cells: object) -> str:
    return " & ".join(str(cell) for cell in cells) + r" \\"


def latex_report(
    config: dict,
    inputs: list[dict],
    fit_payload: dict,
    workspace_path: str,
    closure_path: str,
    plot_directory: Path,
) -> str:
    parameters = {item["name"]: item for item in fit_payload["parameters"]}
    alpha_name = config["model"]["background_parameter"]
    mu_name = config["model"]["signal_parameter"]
    alpha = parameters[alpha_name]
    mu = parameters[mu_name]
    labels = fit_payload["correlation_matrix"]["labels"]
    correlation = fit_payload["correlation_matrix"]["values"]
    corr_value = correlation[labels.index(alpha_name)][labels.index(mu_name)]
    sums = fit_payload["input_sums"]
    postfit = fit_payload["postfit_sums"]
    software = fit_payload["software"]
    plot_paths = {
        name: (plot_directory / filename).resolve().as_posix()
        for name, filename in {
            "before": "before_fit_data_model.pdf",
            "prefit": "prefit_data_model.pdf",
            "postfit": "postfit_data_model.pdf",
            "mu_scan": "likelihood_scan_mu.pdf",
            "alpha_scan": "likelihood_scan_alpha_bkg.pdf",
            "likelihood_2d": "likelihood_2d.pdf",
            "correlation": "correlation_matrix.pdf",
        }.items()
    }
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\usepackage{booktabs,array,graphicx,xcolor}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        r"\title{ABCD background and CC electron-neutrino signal fit}",
        r"\author{SND@LHC analysis workflow}",
        r"\date{}",
        r"\maketitle",
        (
            r"\begin{center}\fcolorbox{black}{yellow!12}{"
            r"\parbox{0.92\textwidth}{\textbf{Diagnostic, explicitly unblinded "
            r"result.} The signal-region data were used in model development. "
            r"The ABCD shape has not passed validation, and template "
            r"uncertainties are not included. This is not a discovery "
            r"significance or an exclusion-limit result.}}\end{center}"
        ),
        r"\tableofcontents",
        r"\section{Analysis question and model}",
        (
            r"The six bins of the \texttt{density\_scifi\_sndsw} observable in "
            r"signal region A are fitted with the HistFactory expectation"
        ),
        (
            r"\[\nu_i(\alpha_{\mathrm{bkg}},\mu)="
            r"\alpha_{\mathrm{bkg}}b_i^{\mathrm{ABCD}}+"
            r"\mu s_i^{\mathrm{CC}\,\nu_e}.\]"
        ),
        (
            r"The observed count in bin $i$ is modeled as "
            r"$n_i\sim\mathrm{Pois}(\nu_i)$. A single freely floating "
            r"$\alpha_{\mathrm{bkg}}$ scales all six background bins; $\mu$ "
            r"scales the normalized charged-current electron-neutrino MC "
            r"template. Both parameters are constrained to be nonnegative. "
            r"No per-bin background scales or template-statistical/systematic "
            r"nuisance parameters are present in this v1 diagnostic."
        ),
        r"\subsection{Definitions}",
        r"\begin{tabular}{p{0.20\textwidth}p{0.72\textwidth}}",
        r"\toprule",
        latex_row("Symbol", "Definition and calculation"),
        r"\midrule",
        latex_row(
            r"$n_i$",
            "Raw 2024 skim-data count in signal region A and density bin $i$.",
        ),
        latex_row(
            r"$b_i^{\mathrm{ABCD}}$",
            "ABCD prediction from the frozen A/B/C/D construction in bin $i$.",
        ),
        latex_row(
            r"$s_i^{\mathrm{CC}\,\nu_e}$",
            "Luminosity-normalized charged-current electron-neutrino MC yield.",
        ),
        latex_row(
            r"$\alpha_{\mathrm{bkg}}$",
            r"One global ABCD-background multiplier; initial 1, bounds $[0,5]$.",
        ),
        latex_row(
            r"$\mu$",
            r"Signal strength and parameter of interest; initial 1, bounds $[0,100]$.",
        ),
        latex_row(
            r"$\nu_i$",
            "Expected total supplied to the Poisson likelihood in bin $i$.",
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\section{Inputs}",
        r"\begin{center}\small",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        latex_row(
            "Density interval",
            r"Observed $n_i$",
            r"ABCD $b_i$",
            r"CC $\nu_e$ $s_i$",
        ),
        r"\midrule",
    ]
    for row in inputs:
        lines.append(
            latex_row(
                "{" + f"[{row['bin_low']:g}, {row['bin_high']:g})" + "}",
                f"{row['observed_data']:,}",
                f"{row['abcd_background']:.6g}",
                f"{row['cc_nue_signal']:.6g}",
            )
        )
    lines.extend(
        [
            r"\midrule",
            latex_row(
                "Total",
                f"{sums['observed_data']:,}",
                f"{sums['abcd_background']:.6g}",
                f"{sums['cc_nue_signal']:.6g}",
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                r"The intervals are left-inclusive and right-exclusive. ABCD "
                r"and signal entries are expected yields and need not be "
                r"integers. The fit uses all six rows, not only the totals."
            ),
            r"\section{Tools, runtime, and reproducibility}",
            r"\begin{tabular}{p{0.25\textwidth}p{0.65\textwidth}}",
            r"\toprule",
            latex_row("Tool or artifact", "Role and setup"),
            r"\midrule",
            latex_row(
                f"cabinetry {tex_escape(software['cabinetry'])}",
                (
                    "Predictions, fit, one-dimensional profile-likelihood "
                    "scans, goodness of fit, and statistical plots."
                ),
            ),
            latex_row(
                f"pyhf {tex_escape(software['pyhf'])}",
                (
                    "HistFactory backend for the binned Poisson likelihood and "
                    "two-parameter likelihood grid."
                ),
            ),
            latex_row(
                "Matplotlib",
                "Rendering backend used by cabinetry and the two-dimensional scan.",
            ),
            latex_row(
                "ROOT inputs",
                (
                    "Earlier stages materialized and validated histogram yields. "
                    "This fit consumes the tabular closure product; ROOT does not "
                    "refit or redraw this statistical model."
                ),
            ),
            latex_row(
                "Snakemake",
                (
                    "Orchestration, LCG activation, preflight checks, output "
                    "tracking, and LaTeX compilation."
                ),
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\subsection{Input provenance}",
            r"\small",
            "Closure CSV: \\path{" + str(Path(closure_path).resolve()) + "}",
            r"\par",
            "HistFactory workspace: \\path{"
            + str(Path(workspace_path).resolve())
            + "}",
            r"\normalsize",
            r"\section{Fit results}",
            r"\begin{center}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            latex_row(
                "Parameter", "Best fit", "Hesse uncertainty", "Bounds", "Boundary"
            ),
            r"\midrule",
            latex_row(
                tex_escape(alpha_name),
                f"{alpha['bestfit']:.7g}",
                f"{alpha['uncertainty_hesse']:.7g}",
                f"[{alpha['lower_bound']:g}, {alpha['upper_bound']:g}]",
                str(alpha["at_lower_bound"] or alpha["at_upper_bound"]),
            ),
            latex_row(
                tex_escape(mu_name),
                f"{mu['bestfit']:.7g}",
                f"{mu['uncertainty_hesse']:.7g}",
                f"[{mu['lower_bound']:g}, {mu['upper_bound']:g}]",
                str(mu["at_lower_bound"] or mu["at_upper_bound"]),
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\begin{center}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            latex_row("Quantity", "Value"),
            r"\midrule",
            latex_row(
                r"Best-fit $-2\log L$", f"{fit_payload['best_twice_nll']:.7g}"
            ),
            latex_row(
                r"Goodness-of-fit $p$-value",
                f"{fit_payload['goodness_of_fit_p_value']:.7g}",
            ),
            latex_row(
                r"$\mathrm{corr}(\alpha_{\mathrm{bkg}},\mu)$",
                f"{corr_value:.7g}",
            ),
            latex_row("Post-fit background total", f"{postfit['background']:.7g}"),
            latex_row("Post-fit signal total", f"{postfit['signal']:.7g}"),
            latex_row("Post-fit total expectation", f"{postfit['total']:.7g}"),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                r"The global scale brings the total background close to the "
                r"observed total, but the goodness-of-fit $p$-value is "
                r"effectively zero. The six-bin ABCD shape is therefore "
                r"incompatible with the observed density distribution in this "
                r"model. The fitted signal is effectively at its nonnegative "
                r"boundary and must not be interpreted as evidence against or "
                r"for a signal while the background model fails."
            ),
            r"\clearpage",
            r"\section{Before-fit, pre-fit, and post-fit comparisons}",
            (
                r"\textbf{Before fit} is an extra input diagnostic defined here "
                r"as raw ABCD background only: $\alpha_{\mathrm{bkg}}=1$, "
                r"$\mu=0$. \textbf{Pre-fit} is the standard configured nominal "
                r"model at $\alpha_{\mathrm{bkg}}=1$, $\mu=1$. "
                r"\textbf{Post-fit} uses the simultaneous maximum-likelihood "
                r"estimates."
            ),
        ]
    )
    for title, name in (
        ("Before fit: raw ABCD background", "before"),
        ("Pre-fit: nominal signal plus background", "prefit"),
        ("Post-fit: simultaneous fitted model", "postfit"),
    ):
        lines.extend(
            [
                rf"\subsection{{{tex_escape(title)}}}",
                r"\begin{center}",
                rf"\includegraphics[width=0.80\linewidth]{{{plot_paths[name]}}}",
                r"\end{center}",
            ]
        )
    lines.extend(
        [
            r"\clearpage",
            r"\section{Likelihood diagnostics}",
            (
                r"The one-dimensional curves are profile-likelihood scans: at "
                r"each displayed value, the other parameter is refitted. The "
                r"vertical scale is the increase in $-2\log L$ from the best "
                r"fit. The two-dimensional surface evaluates both displayed "
                r"values together. White contours, when present, mark "
                r"$\Delta(-2\log L)=2.30$ and 5.99."
            ),
            r"\begin{center}",
            rf"\includegraphics[width=0.72\linewidth]{{{plot_paths['mu_scan']}}}",
            r"\end{center}",
            r"\begin{center}",
            rf"\includegraphics[width=0.72\linewidth]{{{plot_paths['alpha_scan']}}}",
            r"\end{center}",
            r"\begin{center}",
            rf"\includegraphics[width=0.76\linewidth]{{{plot_paths['likelihood_2d']}}}",
            r"\end{center}",
            r"\subsection{Fitted-parameter correlation}",
            r"\begin{center}",
            rf"\includegraphics[width=0.62\linewidth]{{{plot_paths['correlation']}}}",
            r"\end{center}",
            r"\section{Conclusion and analysis boundary}",
            (
                r"The machinery is operational and reproducible, and a "
                r"simultaneous two-parameter fit is returned. It does not yet "
                r"provide a defensible physics measurement: a freely floating "
                r"global normalization cannot repair the strong bin-to-bin ABCD "
                r"nonclosure. An independently validated background-shape "
                r"treatment is required before a significance or limit."
            ),
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    inputs = load_model_inputs(args.closure_csv, config)
    workspace_spec = build_workspace(config, inputs)
    workspace = pyhf.Workspace(workspace_spec)
    model, data = cabinetry.model_utils.model_and_data(workspace)

    if model.config.nmaindata != len(inputs):
        raise ValueError("Workspace main-data length does not match the six inputs")
    if model.config.poi_name != config["model"]["parameter_of_interest"]:
        raise ValueError("Workspace parameter of interest is incorrect")
    expected_parameters = {
        config["model"]["background_parameter"],
        config["model"]["signal_parameter"],
    }
    if set(model.config.par_order) != expected_parameters or model.config.npars != 2:
        raise ValueError("Workspace must contain exactly alpha_bkg and mu")

    fit_config = config["fit"]
    fit_results = cabinetry.fit.fit(
        model,
        data,
        goodness_of_fit=bool(fit_config["goodness_of_fit"]),
        maxiter=int(fit_config["maxiter"]),
        tolerance=float(fit_config["tolerance"]),
    )
    fit_payload = serialize_fit_results(config, inputs, model, fit_results)

    mu_name = config["model"]["signal_parameter"]
    alpha_name = config["model"]["background_parameter"]
    mu_scan_config = config["scans"]["mu"]
    alpha_scan_config = config["scans"]["alpha_bkg"]
    mu_scan = cabinetry.fit.scan(
        model,
        data,
        mu_name,
        par_range=tuple(float(value) for value in mu_scan_config["range"]),
        n_steps=int(mu_scan_config["points"]),
        maxiter=int(fit_config["maxiter"]),
        tolerance=float(fit_config["tolerance"]),
    )
    alpha_scan = cabinetry.fit.scan(
        model,
        data,
        alpha_name,
        par_range=tuple(float(value) for value in alpha_scan_config["range"]),
        n_steps=int(alpha_scan_config["points"]),
        maxiter=int(fit_config["maxiter"]),
        tolerance=float(fit_config["tolerance"]),
    )
    likelihood_rows, alpha_values, mu_values, likelihood_delta = (
        two_dimensional_likelihood(config, model, data, fit_results)
    )

    plot_directory = Path(args.output_plot_dir)
    save_cabinetry_plots(
        plot_directory, model, data, fit_results, mu_scan, alpha_scan
    )
    save_two_dimensional_plot(
        plot_directory,
        config,
        alpha_values,
        mu_values,
        likelihood_delta,
        fit_results,
    )

    atomic_text(args.output_inputs_csv, csv_text(inputs))
    atomic_text(args.output_workspace_json, json.dumps(workspace_spec, indent=2) + "\n")
    atomic_text(args.output_fit_json, json.dumps(fit_payload, indent=2) + "\n")
    atomic_text(args.output_mu_scan_csv, csv_text(scan_rows(mu_scan)))
    atomic_text(args.output_alpha_scan_csv, csv_text(scan_rows(alpha_scan)))
    atomic_text(args.output_likelihood_2d_csv, csv_text(likelihood_rows))
    atomic_text(
        args.output_report_markdown,
        markdown_report(
            config,
            inputs,
            fit_payload,
            args.output_workspace_json,
            args.closure_csv,
        ),
    )
    atomic_text(
        args.output_report_tex,
        latex_report(
            config,
            inputs,
            fit_payload,
            args.output_workspace_json,
            args.closure_csv,
            plot_directory,
        ),
    )
    print(
        f"Completed {config['version']}: "
        f"{alpha_name}={fit_payload['parameters'][model.config.par_map[alpha_name]['slice'].start]['bestfit']:.8g}, "
        f"{mu_name}={fit_payload['parameters'][model.config.par_map[mu_name]['slice'].start]['bestfit']:.8g}, "
        f"goodness_of_fit={fit_payload['goodness_of_fit_p_value']:.8g}"
    )


if __name__ == "__main__":
    main()
