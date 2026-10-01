#!/usr/bin/env python3
"""Run the strict one-bin blinded control fit and expected inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

import cabinetry
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from scipy.stats import norm

from abcd_blinded_model import (
    CONTROL_ROLES,
    ROLES,
    a_background_prediction,
    covariance_matrix,
    fit_control,
    full_template,
    load_blinded_input,
    mapped_covariance,
    profile_interval,
    q0_value,
    replace_observations,
    sample_yields,
    transfer_parameters,
    value,
)
from fit_abcd_density_signal_region import (
    atomic_figure,
    atomic_text,
    csv_text,
    latex_row,
    load_yaml,
    tex_escape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-dirty-test", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def fit_payload(results) -> dict:
    covariance = covariance_matrix(results)
    return {
        "labels": list(results.labels),
        "bestfit": np.asarray(results.bestfit, dtype=float).tolist(),
        "uncertainty_hesse": np.asarray(results.uncertainty, dtype=float).tolist(),
        "covariance": covariance.tolist(),
        "correlation": np.asarray(results.corr_mat, dtype=float).tolist(),
        "best_twice_nll": float(results.best_twice_nll),
    }


def scan_rows(scan) -> list[dict]:
    return [
        {"mu": float(parameter), "delta_twice_nll": float(delta)}
        for parameter, delta in zip(scan.parameter_values, scan.delta_nlls)
    ]


def expected_upper_limit(config: dict, data, model) -> dict:
    low, high, points = config["fit"]["upper_limit_scan"]
    scan = np.linspace(float(low), float(high), int(points))
    observed, expected = pyhf.infer.intervals.upper_limits.upper_limit(
        data,
        model,
        scan=scan,
        level=float(config["fit"]["upper_limit_level"]),
    )
    return {
        "background_asimov_upper_limit": float(np.asarray(observed)),
        "expected_band_minus2_minus1_median_plus1_plus2": [
            float(np.asarray(item)) for item in expected
        ],
        "confidence_level": 1.0 - float(config["fit"]["upper_limit_level"]),
        "scan": [float(low), float(high), int(points)],
    }


def control_covariance_rows(results) -> list[dict]:
    covariance = covariance_matrix(results)
    rows = []
    for i, left in enumerate(results.labels):
        for j, right in enumerate(results.labels):
            rows.append(
                {
                    "parameter_i": left,
                    "parameter_j": right,
                    "covariance": float(covariance[i, j]),
                    "correlation": float(results.corr_mat[i, j]),
                }
            )
    return rows


def save_plots(
    output: Path,
    config: dict,
    row: dict,
    control_model,
    control_data,
    control_fit,
    full_model,
    background_parameters,
    signal_parameters,
    scan,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    prefit = np.asarray(
        control_model.expected_data(
            control_model.config.suggested_init(), include_auxdata=False
        ),
        dtype=float,
    )
    postfit = np.asarray(
        control_model.expected_data(control_fit.bestfit, include_auxdata=False),
        dtype=float,
    )
    observed = []
    prefit_values = []
    postfit_values = []
    for role in CONTROL_ROLES:
        channel_slice = control_model.config.channel_slices[f"{role}_region"]
        observed.append(float(np.asarray(control_data)[channel_slice][0]))
        prefit_values.append(float(prefit[channel_slice][0]))
        postfit_values.append(float(postfit[channel_slice][0]))
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.2, prefit_values, 0.4, label="Pre-fit", color="tab:orange")
    axis.bar(x + 0.2, postfit_values, 0.4, label="Control-region post-fit", color="tab:blue")
    axis.errorbar(x, observed, yerr=np.sqrt(observed), fmt="ko", label="Observed control data")
    axis.set_xticks(x, CONTROL_ROLES)
    axis.set_ylabel("Events in [5000, 40000]")
    axis.set_yscale("log")
    axis.set_title("Blinded B/C/D control fit; observed A absent")
    axis.legend()
    atomic_figure(figure, output / "control_region_fit.pdf")
    atomic_figure(figure, output / "control_region_fit.png")
    plt.close(figure)

    background_components = sample_yields(full_model, background_parameters, "A")
    signal_components = sample_yields(full_model, signal_parameters, "A")
    names = [
        "ABCD residual", "CC numu", "NC nue", "NC numu", "CC nue signal"
    ]
    keys = [
        "abcd_residual_background", "cc_numu", "nc_nue", "nc_numu",
        "cc_nue_signal",
    ]
    values_for_plot = [
        background_components[key] if key != "cc_nue_signal"
        else signal_components[key]
        for key in keys
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bottom = 0.0
    colors = ["tab:gray", "tab:orange", "tab:green", "tab:purple", "tab:blue"]
    for name, amount, color in zip(names, values_for_plot, colors):
        axis.bar([0], [amount], bottom=bottom, label=name, color=color)
        bottom += amount
    axis.set_xticks([0], ["A expectation\n(no observed data)"])
    axis.set_ylabel("Expected events in [5000, 40000]")
    axis.set_title("Control-region post-fit A prediction and injected signal")
    axis.legend(fontsize=8)
    atomic_figure(figure, output / "signal_region_expected.pdf")
    atomic_figure(figure, output / "signal_region_expected.png")
    plt.close(figure)

    figure = cabinetry.visualize.scan(scan, save_figure=False, close_figure=False)
    atomic_figure(figure, output / "expected_likelihood_scan_mu.pdf")
    atomic_figure(figure, output / "expected_likelihood_scan_mu.png")
    plt.close(figure)
    figure = cabinetry.visualize.correlation_matrix(
        control_fit, save_figure=False, close_figure=False
    )
    atomic_figure(figure, output / "control_correlation.pdf")
    atomic_figure(figure, output / "control_correlation.png")
    plt.close(figure)


def make_payload(
    config: dict,
    row: dict,
    control_model,
    control_fit,
    full_model,
    expected_fit,
    background_parameters,
    a_prediction: tuple[float, float],
    scan,
    p0: float,
    q0: float,
    upper_limit: dict,
) -> dict:
    interval = profile_interval(
        scan.parameter_values,
        scan.delta_nlls,
        float(config["fit"]["confidence_level"]),
    )
    mu_index = list(expected_fit.labels).index(config["model"]["signal_parameter"])
    mu_hat = float(expected_fit.bestfit[mu_index])
    half_width = (
        0.5 * (interval[1] - interval[0])
        if interval[0] is not None and interval[1] is not None
        else None
    )
    a_components = sample_yields(full_model, background_parameters, "A")
    return {
        "analysis_version": config["version"],
        "status": "PROVISIONAL_EXPECTED_ONLY",
        "blinding_status": config["blinding"]["status"],
        "statistical_model": "one-bin [5000,40000] simultaneous ABCD",
        "density_bin_index_in_model": False,
        "signal_region_observation_used": False,
        "control_fit_signal_strength": 0.0,
        "control_fit_label": "control-region post-fit prediction",
        "signal_contamination": {
            "decision": row["signal_contamination_decision"],
            "threshold": value(row, "signal_contamination_threshold"),
            "fractions": {
                role: value(row, f"signal_contamination_fraction_{role.lower()}")
                for role in CONTROL_ROLES
            },
        },
        "kappa": {
            "central": value(row, "kappa_central"),
            "uncertainty": value(row, "kappa_uncertainty"),
            "relative_uncertainty": value(row, "kappa_relative_uncertainty"),
            "constraint": row["kappa_constraint_form"],
            "source": row["kappa_constraint_source"],
            "transport_status": row["kappa_transport_status"],
        },
        "control_fit": fit_payload(control_fit),
        "a_prediction": {
            "total_background": a_prediction[0],
            "uncertainty_delta_method": a_prediction[1],
            "components": a_components,
        },
        "expected_measurement": {
            "injected_mu": float(config["model"]["signal_injection"]),
            "mu_hat": mu_hat,
            "confidence_level": float(config["fit"]["confidence_level"]),
            "profile_interval": list(interval),
            "half_width": half_width,
            "relative_precision": half_width / mu_hat if half_width is not None and mu_hat > 0 else None,
            "fit": fit_payload(expected_fit),
        },
        "expected_discovery": {
            "q0": q0,
            "p0_asymptotic": p0,
            "z_asymptotic": float(norm.isf(p0)),
            "toy_validation_performed": False,
        },
        "expected_exclusion": upper_limit,
        "software": {
            "cabinetry": importlib.metadata.version("cabinetry"),
            "pyhf": pyhf.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "workspace": {
            "control_channels": list(control_model.config.channels),
            "full_channels": list(full_model.config.channels),
            "full_parameters": list(full_model.config.par_order),
        },
    }


def report_markdown(payload: dict) -> str:
    discovery = payload["expected_discovery"]
    measurement = payload["expected_measurement"]
    exclusion = payload["expected_exclusion"]
    return "\n".join(
        [
            "# Strict one-bin blinded ABCD expected report",
            "",
            "**Expected-only reblinded-development result. Observed A data were not read.**",
            "",
            f"Control-fit label: **{payload['control_fit_label']}**.",
            "",
            f"Kappa: `{payload['kappa']['central']:.6g} +/- {payload['kappa']['uncertainty']:.6g}` from `{payload['kappa']['source']}`.",
            "",
            f"Expected discovery at mu=1: `q0={discovery['q0']:.6g}`, `p0={discovery['p0_asymptotic']:.6g}`, `Z={discovery['z_asymptotic']:.6g}`.",
            "",
            f"Expected 68% profile interval: `{measurement['profile_interval']}`; relative precision `{measurement['relative_precision']}`.",
            "",
            f"Expected background-only 95% upper limit on mu: `{exclusion['background_asimov_upper_limit']:.6g}`.",
            "",
            "Observed inference remains locked. Kappa transport and 30% MC-normalization uncertainties remain provisional.",
            "",
        ]
    )


def report_latex(config: dict, row: dict, payload: dict, plot_dir: Path) -> str:
    plot = lambda name: (plot_dir / name).resolve().as_posix()
    measurement = payload["expected_measurement"]
    discovery = payload["expected_discovery"]
    exclusion = payload["expected_exclusion"]
    interval = measurement["profile_interval"]
    interval_text = "[{}, {}]".format(
        "open" if interval[0] is None else f"{interval[0]:.5g}",
        "open" if interval[1] is None else f"{interval[1]:.5g}",
    )
    precision_text = (
        "undefined for an open scan interval"
        if measurement["relative_precision"] is None
        else f"{measurement['relative_precision']:.3g}"
    )
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage{booktabs,graphicx,xcolor}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        r"\title{Strict one-bin blinded ABCD CC electron-neutrino analysis}",
        r"\author{SND@LHC analysis workflow}", r"\date{}", r"\maketitle",
        r"\begin{center}\fcolorbox{black}{yellow!12}{\parbox{0.92\textwidth}{\textbf{Expected-only reblinded-development result.} Observed signal-region A data are absent from the dependency graph, sanitized input, and likelihood. Historical blindness cannot be restored because A was inspected in earlier development.}}\end{center}",
        r"\section{Analysis definition and strict blinding}",
        r"The primary analysis is one count over $5000\leq\texttt{density\_scifi\_sndsw}<40000$. There is no density-bin index in the statistical model. Observed inputs are B, C, D, plus an independent validation plane used for $\kappa$.",
        r"\section{ABCD likelihood and component separation}",
        r"The residual non-neutrino component is $b_D=L_D\rho$, $b_B=L_D\rho u$, $b_C=L_C\rho v$, and $b_A=L_A\rho u v\kappa$. The channel expectation is $\nu_R=b_R+\mu S_R+\sum_k\gamma_k q_{k,R}$. Explicit CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ MC components are not included in the residual ABCD component.",
        r"\section{One-bin inputs and signal contamination}",
        r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
        latex_row("Quantity", "A", "B", "C", "D"), r"\midrule",
        latex_row("Observed data", "MASKED", row["count_b"], row["count_c"], row["count_d"]),
        latex_row("CC nue signal, mu=1", *[f"{value(row, f'cc_nue_{role.lower()}'):.5g}" for role in ROLES]),
        latex_row("Other-neutrino MC", *[f"{sum(value(row, f'{sample.lower()}_{role.lower()}') for sample in config['source']['background_samples']):.5g}" for role in ROLES]),
        r"\bottomrule\end{tabular}\end{center}",
        rf"The CC $\nu_e$ contamination fractions in B/C/D are {value(row, 'signal_contamination_fraction_b'):.3%}, {value(row, 'signal_contamination_fraction_c'):.3%}, and {value(row, 'signal_contamination_fraction_d'):.3%}. The 1\% criterion therefore classifies contamination as \texttt{{{tex_escape(row['signal_contamination_decision'])}}}. The blinded background-only control fit fixes $\mu=0$; the final simultaneous model releases it.",
        r"\section{Independent nonclosure constraint}",
        rf"The validation plane gives $\kappa={payload['kappa']['central']:.5g}\pm{payload['kappa']['uncertainty']:.5g}$, represented by a HistFactory normsys constraint. Its source is {tex_escape(payload['kappa']['source'])}. The transport status is \texttt{{{tex_escape(payload['kappa']['transport_status'])}}}.",
        r"\section{Blinded control-region fit}",
        r"The following is a \textbf{control-region post-fit prediction}, not a signal-region post-fit result.",
        r"\begin{center}", rf"\includegraphics[width=0.82\linewidth]{{{plot('control_region_fit.pdf')}}}", r"\end{center}",
        rf"The predicted total A background is ${payload['a_prediction']['total_background']:.5g}\pm{payload['a_prediction']['uncertainty_delta_method']:.5g}$ using covariance and independent constrained nuisances.",
        r"\begin{center}", rf"\includegraphics[width=0.76\linewidth]{{{plot('signal_region_expected.pdf')}}}", r"\end{center}",
        r"\section{Expected measurement, discovery, and exclusion}",
        rf"For signal-injected Asimov data ($\mu_{{inj}}=1$), the expected best fit is $\hat\mu={measurement['mu_hat']:.5g}$ and the 68\% profile interval is {interval_text}, with relative precision {precision_text}. The expected discovery test gives $q_0={discovery['q0']:.5g}$, $p_0={discovery['p0_asymptotic']:.5g}$, and $Z={discovery['z_asymptotic']:.5g}$. The background-only Asimov 95\% upper limit is $\mu<{exclusion['background_asimov_upper_limit']:.5g}$.",
        r"\begin{center}", rf"\includegraphics[width=0.76\linewidth]{{{plot('expected_likelihood_scan_mu.pdf')}}}", r"\end{center}",
        r"\section{Nuisance parameters and correlations}",
        r"The model includes validation nonclosure, shared 30\% normalization constraints for each explicit neutrino background, and per-channel MC-statistical constraints. The 30\% values remain placeholders.",
        r"\begin{center}", rf"\includegraphics[width=0.74\linewidth]{{{plot('control_correlation.pdf')}}}", r"\end{center}",
        r"\section{Freeze and unblinding boundary}",
        r"The freeze manifest records Git state, regions, range, exposures, components, constraints, expected results, and checksums. Only the separate explicitly authorized target may read A data. That target must profile $\mu$ and all nuisance parameters simultaneously in A+B+C+D and validate $p_0$ with toys.",
        r"\section{Observed results}", r"\textbf{LOCKED. No observed A count, fit, significance, or limit was produced.}",
        r"\section{Limitations}",
        r"The $\kappa$ transport has no secondary validation, the MC normalization uncertainties are placeholders, and the asymptotic expected result has not been toy-validated. These prevent a final physics interpretation.",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def run_expected(
    config: dict,
    row: dict,
    output_root: Path,
) -> dict:
    control_spec, control_model, control_data, control_fit = fit_control(config, row)
    template_spec, template_model = full_template(config, row)
    background_parameters = transfer_parameters(
        control_model, control_fit.bestfit, template_model, 0.0
    )
    signal_parameters = transfer_parameters(
        control_model,
        control_fit.bestfit,
        template_model,
        float(config["model"]["signal_injection"]),
    )
    background_spec = replace_observations(
        template_spec, template_model, background_parameters
    )
    signal_spec = replace_observations(
        template_spec, template_model, signal_parameters
    )
    background_workspace = pyhf.Workspace(background_spec)
    background_model, background_data = cabinetry.model_utils.model_and_data(
        background_workspace
    )
    signal_workspace = pyhf.Workspace(signal_spec)
    expected_model, expected_data = cabinetry.model_utils.model_and_data(
        signal_workspace
    )
    expected_fit = cabinetry.fit.fit(
        expected_model,
        expected_data,
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    scan = cabinetry.fit.scan(
        expected_model,
        expected_data,
        config["model"]["signal_parameter"],
        par_range=tuple(float(x) for x in config["fit"]["mu_scan_range"]),
        n_steps=int(config["fit"]["mu_scan_points"]),
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    q0, _, _ = q0_value(expected_data, expected_model)
    p0 = float(norm.sf(math.sqrt(max(0.0, q0))))
    upper_limit = expected_upper_limit(config, background_data, background_model)
    full_covariance = mapped_covariance(control_model, control_fit, template_model)
    a_prediction = a_background_prediction(
        template_model, background_parameters, full_covariance
    )
    payload = make_payload(
        config, row, control_model, control_fit, expected_model, expected_fit,
        background_parameters, a_prediction, scan, p0, q0, upper_limit,
    )

    output = output_root / config["output"]["expected_directory"]
    plots = output / config["output"]["plot_directory"]
    save_plots(
        plots, config, row, control_model, control_data, control_fit,
        template_model, background_parameters, signal_parameters, scan,
    )
    atomic_text(output / config["output"]["model_inputs"], csv_text([row]))
    atomic_text(
        output / config["output"]["control_workspace"],
        json.dumps(control_spec, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["expected_background_workspace"],
        json.dumps(background_spec, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["expected_signal_workspace"],
        json.dumps(signal_spec, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["control_fit"],
        json.dumps(fit_payload(control_fit), indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["control_covariance"],
        csv_text(control_covariance_rows(control_fit)),
    )
    atomic_text(
        output / config["output"]["expected_results"],
        json.dumps(payload, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["mu_scan"], csv_text(scan_rows(scan))
    )
    atomic_text(
        output / config["output"]["report_markdown"], report_markdown(payload)
    )
    atomic_text(
        output / config["output"]["report_tex"],
        report_latex(config, row, payload, plots),
    )
    return payload


def write_manifest(
    config: dict,
    row: dict,
    input_path: str | Path,
    config_path: str | Path,
    output_root: Path,
    payload: dict,
    state: dict,
    allow_dirty_test: bool,
) -> dict:
    if (
        config["freeze"]["require_clean_git_for_production"]
        and state["dirty"]
        and not allow_dirty_test
    ):
        raise ValueError("Refusing to freeze expected outputs from a dirty worktree")
    output = output_root / config["output"]["expected_directory"]
    artifacts = {}
    for key in (
        "control_workspace", "expected_background_workspace",
        "expected_signal_workspace", "expected_results", "mu_scan",
    ):
        path = output / config["output"][key]
        artifacts[key] = {"path": str(path), "sha256": sha256(path)}
    manifest = {
        "analysis_version": config["version"],
        "freeze_status": "TEST_OVERRIDE_DIRTY" if state["dirty"] else "FROZEN",
        "git": state,
        "one_bin_range": [
            float(config["source"]["counting_range"][0]),
            float(config["source"]["counting_range"][1]),
        ],
        "event_selection": config["source"]["event_selection"],
        "abcd_regions": config["source"]["roles"],
        "validation_regions": config["source"]["validation_roles"],
        "exposures": {
            role: value(row, f"exposure_{role.lower()}") for role in ROLES
        },
        "signal_definition": config["source"]["signal_sample"],
        "mc_background_definitions": config["source"]["background_samples"],
        "normalization": "MC yields normalized to each region exposure",
        "nuisance_parameters": payload["workspace"]["full_parameters"],
        "nuisance_constraints": {
            "kappa": payload["kappa"],
            "background_mc_normalization": config["model"]["background_mc_normalization"],
            "mc_statistical": config["model"]["include_mc_statistical_uncertainty"],
        },
        "expected_inference_results": {
            "measurement": payload["expected_measurement"],
            "discovery": payload["expected_discovery"],
            "exclusion": payload["expected_exclusion"],
        },
        "sanitized_input": {"path": str(input_path), "sha256": sha256(input_path)},
        "configuration": {"path": str(config_path), "sha256": sha256(config_path)},
        "artifacts": artifacts,
        "unblinding_criteria": config["freeze"]["unblinding_criteria"],
        "unblinding_authorized": False,
        "signal_region_observation_used": False,
    }
    atomic_text(
        output_root / config["output"]["freeze_manifest"],
        json.dumps(manifest, indent=2) + "\n",
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config["blinding"].get("observed_inference_authorized"):
        raise ValueError("Expected runner refuses observed-inference authorization")
    row = load_blinded_input(args.input, config)
    output_root = Path(args.output_root)
    payload = run_expected(config, row, output_root)
    manifest = write_manifest(
        config, row, args.input, args.config, output_root, payload,
        git_state(), args.allow_dirty_test,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "z_expected": payload["expected_discovery"]["z_asymptotic"],
                "upper_limit_expected": payload["expected_exclusion"]["background_asimov_upper_limit"],
                "freeze_status": manifest["freeze_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
