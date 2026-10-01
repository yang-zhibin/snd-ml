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


def signed_value(row: dict, key: str) -> float:
    result = float(row[key])
    if not math.isfinite(result):
        raise ValueError(f"Invalid finite signed value for {key}: {row[key]}")
    return result


def validation_closure_payload(row: dict) -> dict:
    """Return the independently observed one-bin closure result (never signal A)."""
    suffixes = ("numerator", "b", "c", "d")
    return {
        "density_interval": [value(row, "range_low"), value(row, "range_high")],
        "regions": {suffix: row[f"validation_region_{suffix}"] for suffix in suffixes},
        "exposures": {
            suffix: value(row, f"validation_exposure_{suffix}") for suffix in suffixes
        },
        "raw_observed": {
            suffix: value(row, f"validation_observed_{suffix}") for suffix in suffixes
        },
        "explicit_neutrino_background": {
            suffix: value(row, f"validation_other_neutrino_{suffix}")
            for suffix in suffixes
        },
        "residual_observed": {
            suffix: value(row, f"validation_residual_{suffix}") for suffix in suffixes
        },
        "residual_variance": {
            suffix: value(row, f"validation_residual_variance_{suffix}")
            for suffix in suffixes
        },
        "transfer": value(row, "validation_transfer"),
        "transfer_uncertainty": value(row, "validation_transfer_uncertainty"),
        "prediction": value(row, "validation_prediction"),
        "prediction_uncertainty": value(row, "validation_prediction_uncertainty"),
        "kappa": value(row, "kappa_central"),
        "kappa_uncertainty": value(row, "kappa_uncertainty"),
        "fractional_nonclosure": signed_value(row, "validation_fractional_nonclosure"),
        "pull": signed_value(row, "validation_pull"),
    }


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
        "validation_closure": validation_closure_payload(row),
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
    closure = payload["validation_closure"]
    low, high = closure["density_interval"]
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
            "## Independent validation-plane one-bin closure",
            "",
            "This is A-prime in the validation plane, not the masked signal-region A observation.",
            "",
            "| Density interval | A'obs residual | A'pred +/- sigma_pred | Transfer T +/- sigma_T | kappa +/- sigma_kappa | Nonclosure | Pull |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| [{low:g}, {high:g}) | {closure['residual_observed']['numerator']:.6g} | "
                f"{closure['prediction']:.6g} +/- {closure['prediction_uncertainty']:.6g} | "
                f"{closure['transfer']:.6g} +/- {closure['transfer_uncertainty']:.6g} | "
                f"{closure['kappa']:.6g} +/- {closure['kappa_uncertainty']:.6g} | "
                f"{closure['fractional_nonclosure']:.3%} | {closure['pull']:.6g} |"
            ),
            "",
            "Definitions: `N_R'` is the raw validation-region data count; `Q_R'` is the explicit CC-numu + NC-nue + NC-numu MC yield; `R_R'=N_R'-Q_R'` is the residual count used by ABCD; and `L_R'` is the region exposure. `T=(R_C'/R_D')(L_D'/L_C')`, `A'pred=R_B'(L_A'/L_B')T`, `kappa=A'obs/A'pred`, nonclosure is `kappa-1`, and pull is `(A'obs-A'pred)/sqrt(V_A'+sigma_pred^2)`. Uncertainties use first-order propagation of validation-data Poisson variance, MC statistical variance, and the configured MC-normalization terms.",
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


def definition_block(entries: list[tuple[str, str, str, str, str]]) -> list[str]:
    lines = [r"\begin{description}"]
    for symbol, definition, purpose, meaning, reference in entries:
        lines.append(
            rf"\item[{symbol}] \textbf{{Definition:}} {definition} "
            rf"\textbf{{Why it is used:}} {purpose} "
            rf"\textbf{{Interpretation:}} {meaning} {reference}"
        )
    lines.append(r"\end{description}")
    return lines


def likelihood_definition_lines() -> list[str]:
    return definition_block([
        (r"$R$", r"One of the four disjoint analysis regions $A$, $B$, $C$, or $D$.", r"It indexes every observed count and expected contribution in the simultaneous likelihood.", r"A is signal-enhanced and masked here; B, C, and D constrain background behavior.", r"\cite{histfactory,choi-oh}"),
        (r"$L_R$", r"The integrated data exposure assigned to region $R$.", r"It converts the fitted residual-background rate into an expected event count and accounts for regions produced with different luminosities.", r"A larger $L_R$ gives proportionally more residual-background events at fixed rate and transfer factors.", r"\cite{histfactory}"),
        (r"$\rho$", r"The residual-background rate per unit exposure in reference region D.", r"It supplies one common normalization shared by all four ABCD regions.", r"$L_D\rho$ is the residual event expectation in D before explicit neutrino components are added.", r"\cite{histfactory}"),
        (r"$u$", r"The multiplicative transfer factor from D to B along the first ABCD axis, after exposure scaling.", r"It lets B data constrain how the residual background changes when the first region selection changes.", r"$u=1$ means no rate change per unit exposure along that axis.", r"\cite{histfactory,choi-oh}"),
        (r"$v$", r"The multiplicative transfer factor from D to C along the second ABCD axis, after exposure scaling.", r"It lets C data constrain the other ABCD-axis dependence.", r"$v=1$ means no rate change per unit exposure along that axis.", r"\cite{histfactory,choi-oh}"),
        (r"$\kappa$", r"A multiplicative correction for residual non-factorization when extrapolating to A.", r"ABCD is exact only when the two background-selection axes factorize; $\kappa$ represents measured departure from that ideal.", r"$\kappa=1$ is closure, while values below or above one reduce or increase the naive A prediction.", r"\cite{choi-oh,histfactory}"),
        (r"$b_R$", r"The residual non-neutrino background expectation in region $R$.", r"It is the component estimated through the shared ABCD parameterization rather than explicit neutrino MC.", r"The model uses $b_D=L_D\rho$, $b_B=L_B\rho u$, $b_C=L_C\rho v$, and $b_A=L_A\rho uv\kappa$.", r"\cite{histfactory}"),
        (r"$\nu_R$", r"The total expected event count in region $R$.", r"It is the mean of the Poisson term compared with the region's event count.", r"It sums residual ABCD background, signal, and every explicit neutrino-background component.", r"\cite{histfactory}"),
        (r"$\mu$", r"The signal-strength parameter multiplying the nominal CC $\nu_e$ signal template.", r"It is the parameter of interest used to measure, discover, or limit the signal.", r"$\mu=0$ is background-only and $\mu=1$ is the nominal simulated signal normalization.", r"\cite{cowan}"),
        (r"$S_R$", r"The nominal CC $\nu_e$ MC signal yield in region $R$ at $\mu=1$.", r"It specifies the expected signal distribution across the four channels.", r"The fitted signal contribution is $\mu S_R$.", r"\cite{histfactory}"),
        (r"$q_{k,R}$", r"The nominal MC yield for explicit neutrino-background component $k$ in region $R$.", r"It keeps CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ separate from the residual ABCD background and avoids double counting.", r"Each component can have its own normalization and MC-statistical uncertainty.", r"\cite{histfactory}"),
        (r"$\gamma_k$", r"A constrained nuisance scale multiplying $q_{k,R}$.", r"It propagates systematic normalization uncertainty for explicit MC backgrounds into the likelihood.", r"The nominal value is one; movement away from one is penalized by its constraint term.", r"\cite{histfactory}"),
    ])


def input_definition_lines() -> list[str]:
    return definition_block([
        (r"$N_R$", r"The raw observed event count in region $R$ and the frozen density interval.", r"Counting experiments are modeled with Poisson probabilities whose observations are nonnegative integer counts.", r"A is deliberately absent; only B, C, D and independent validation counts enter the blinded workflow.", r"\cite{histfactory}"),
        (r"$Q_R$", r"The sum of nominal CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ MC yields in region $R$.", r"It identifies neutrino backgrounds modeled explicitly rather than absorbed into the ABCD residual.", r"$Q_R=\sum_k q_{k,R}$ and is normalized to $L_R$.", r"\cite{histfactory}"),
        (r"signal fraction", r"The nominal ratio $S_R/N_R$ in a control region.", r"It tests whether fixing $\mu=0$ in the B+C+D control-only fit is a negligible approximation.", r"The configured threshold is 1\%; B exceeds it, so the control fit is explicitly conditional and the final simultaneous fit must release $\mu$.", r"\cite{histfactory}"),
    ])


def closure_definition_lines() -> list[str]:
    return definition_block([
        (r"$N_{R'}$", r"The raw data count in validation region $R'$.", r"The validation plane provides an observed closure test without using signal-region A.", r"The prime distinguishes every validation quantity from the nominal analysis region.", r"\cite{choi-oh}"),
        (r"$Q_{R'}$", r"The explicit non-signal neutrino MC yield in validation region $R'$.", r"It is removed before testing factorization of the residual background, because those MC samples are separate likelihood components.", r"It equals the sum of CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ yields.", r"\cite{histfactory}"),
        (r"$R_{R'}$", r"The residual validation count $R_{R'}=N_{R'}-Q_{R'}$.", r"This is the population to which the ABCD factorization calculation is applied.", r"The table's $A'_{\rm obs}$ is $R_{A'}$, not a raw count and not signal-region A.", r"\cite{choi-oh}"),
        (r"$V_{R'}$", r"The variance assigned to $R_{R'}$: $N_{R'}+\sum_k\sigma^2_{\mathrm{MCstat},k,R'}+\sum_k(\delta_kq_{k,R'})^2$.", r"It combines validation-data Poisson fluctuation with explicit-MC statistical and configured normalization terms.", r"It is used only for first-order uncertainty propagation in this closure diagnostic.", r"\cite{histfactory}"),
        (r"$T$", r"The validation transfer $T=(R_{C'}/R_{D'})(L_{D'}/L_{C'})$.", r"It measures the residual-background change along the second ABCD axis per unit exposure.", r"It is multiplied by the $B'$ residual to predict $A'$.", r"\cite{choi-oh,histfactory}"),
        (r"$\sigma_T$", r"The propagated uncertainty $\sigma_T=T\sqrt{V_{C'}/R_{C'}^2+V_{D'}/R_{D'}^2}$.", r"It quantifies uncertainty in the transfer inferred from finite $C'$ and $D'$ samples.", r"A larger relative uncertainty in either region increases $\sigma_T$.", r"\cite{choi-oh}"),
        (r"$A'_{\rm pred}$", r"The validation prediction $R_{B'}(L_{A'}/L_{B'})T$.", r"It is the ABCD extrapolation obtained without looking at validation $A'$ until the prediction is defined.", r"Agreement with $A'_{\rm obs}$ is closure; disagreement measures nonclosure.", r"\cite{choi-oh,histfactory}"),
        (r"$\sigma_{\rm pred}$", r"The propagated uncertainty $A'_{\rm pred}\sqrt{V_{B'}/R_{B'}^2+V_{C'}/R_{C'}^2+V_{D'}/R_{D'}^2}$.", r"It carries the finite-sample and explicit-MC uncertainty from $B'$, $C'$, and $D'$ into the prediction.", r"It does not include $A'$ because $A'$ is the independent quantity used to test the prediction.", r"\cite{choi-oh}"),
        (r"$\kappa_{\rm val}$", r"The closure ratio $A'_{\rm obs}/A'_{\rm pred}$.", r"It summarizes the departure from ideal ABCD factorization and supplies the provisional likelihood correction.", r"One means exact closure; the measured value below one means the naive prediction is too high.", r"\cite{choi-oh}"),
        (r"$\sigma_\kappa$", r"The first-order uncertainty on $\kappa_{\rm val}$, including $V_{A'}$ and the $B'/C'/D'$ prediction terms.", r"It defines the width of the provisional HistFactory normalization constraint.", r"It measures uncertainty, not evidence that the correction transports to nominal A.", r"\cite{histfactory}"),
        (r"nonclosure", r"The fractional quantity $\kappa_{\rm val}-1$.", r"It expresses closure failure on a relative scale centered at zero.", r"Negative values mean the uncorrected ABCD prediction exceeds the validation observation.", r"\cite{choi-oh}"),
        (r"pull", r"$(A'_{\rm obs}-A'_{\rm pred})/\sqrt{V_{A'}+\sigma_{\rm pred}^2}$.", r"It reports the discrepancy in units of the combined diagnostic uncertainty.", r"Its sign gives the direction of the discrepancy; its magnitude is only diagnostic because the propagation is approximate.", r"\cite{choi-oh}"),
    ])


def inference_definition_lines() -> list[str]:
    return definition_block([
        (r"$\theta$", r"The full set of nuisance parameters, including ABCD transfers, nonclosure, MC normalizations, and MC-statistical modifiers.", r"Profiling them propagates systematic and statistical uncertainty into inference on $\mu$.", r"They are fitted simultaneously rather than fixed to the control-fit estimates in final inference.", r"\cite{histfactory,cowan}"),
        (r"$\lambda(\mu)$", r"The profile-likelihood ratio comparing the best fit at fixed $\mu$ with the global best fit.", r"It is the basis for confidence intervals, discovery tests, and limits while accounting for nuisance parameters.", r"Small $-2\ln\lambda$ values indicate compatibility with that $\mu$.", r"\cite{cowan}"),
        (r"Asimov data", r"A deterministic data set equal to model expectations at chosen parameter values.", r"It gives the median expected sensitivity without statistical fluctuations.", r"This report uses $\mu=1$ for expected measurement/discovery and $\mu=0$ for the expected upper limit.", r"\cite{cowan}"),
        (r"$q_0$", r"The one-sided profile-likelihood discovery statistic for testing $\mu=0$.", r"It quantifies disagreement with the background-only hypothesis while setting downward fluctuations to zero evidence for positive signal.", r"Larger $q_0$ implies stronger background-only incompatibility.", r"\cite{cowan}"),
        (r"$p_0$", r"The background-only tail probability for obtaining $q_0$ at least as signal-like as the tested result.", r"It expresses discovery compatibility on a probability scale.", r"Small $p_0$ means less compatibility with background-only.", r"\cite{cowan}"),
        (r"$Z$", r"The one-sided Gaussian-equivalent significance $Z=\Phi^{-1}(1-p_0)$.", r"It translates $p_0$ into the conventional standard-normal scale.", r"It is expected sensitivity here, not an observed discovery significance.", r"\cite{cowan}"),
        (r"68\% interval", r"The set of $\mu$ values selected from the profile scan at the configured one-parameter threshold.", r"It summarizes expected measurement precision around the fitted signal strength.", r"An interval reaching the physical boundary $\mu=0$ indicates weak sensitivity.", r"\cite{cowan}"),
        (r"95\% upper limit", r"The background-only Asimov upper bound on $\mu$ at the configured confidence level.", r"It summarizes expected exclusion power when no signal is present.", r"It is an expected limit and must not be described as a limit from observed signal-region data.", r"\cite{cowan}"),
    ])


def analysis_step_lines(config: dict, row: dict, payload: dict) -> list[str]:
    closure = payload["validation_closure"]
    return [
        r"\section{Analysis procedure: why and how each step is performed}",
        r"\subsection*{Step 1: freeze the counting definition and enforce blinding}",
        r"\textbf{Why:} choosing selections or binning after viewing signal-region data can bias the result. The expected workflow must therefore have a fixed observable, range, region map, nuisance model, and an input graph that excludes observed A.",
        rf"\textbf{{How:}} use exactly one interval, ${closure['density_interval'][0]:g}\leq\texttt{{density\_scifi\_sndsw}}<{closure['density_interval'][1]:g}$, require the configured event selection, and reject any sanitized field or DAG dependency representing observed A. The freeze manifest records Git state and artifact hashes. This implements the separation between expected sensitivity and observed inference described for likelihood-based searches in Ref.~\cite{{cowan}}.",
        r"\subsection*{Step 2: materialize data and MC inputs}",
        r"\textbf{Why:} every likelihood term must have one auditable numerical source, and explicit MC backgrounds must not also be absorbed into the ABCD residual.",
        rf"\textbf{{How:}} integrate each frozen ROOT histogram over the one interval. The blinded control observations are $N_B={row['count_b']}$, $N_C={row['count_c']}$, and $N_D={row['count_d']}$. Normalize each MC sample to the exposure of its destination region, store its yield and sum-of-weights-squared variance, then define $Q_R=\sum_kq_{{k,R}}$. The HistFactory construction follows the component-and-modifier model of Ref.~\cite{{histfactory}} and is serialized with pyhf \cite{{pyhf}}.",
        r"\subsection*{Step 3: calculate closure in an independent validation plane}",
        r"\textbf{Why:} the two ABCD axes are only approximately independent in real data. Closure must be measured away from nominal A before a nonclosure factor is transported into its prediction \cite{choi-oh}.",
        rf"\textbf{{How:}} subtract explicit neutrino MC to obtain residuals $R_{{R'}}=N_{{R'}}-Q_{{R'}}$. Numerically, $R_{{A'}}={closure['residual_observed']['numerator']:.6g}$, $R_{{B'}}={closure['residual_observed']['b']:.6g}$, $R_{{C'}}={closure['residual_observed']['c']:.6g}$, and $R_{{D'}}={closure['residual_observed']['d']:.6g}$. Then",
        r"\[T=\frac{R_{C'}}{R_{D'}}\frac{L_{D'}}{L_{C'}},\qquad A'_{\rm pred}=R_{B'}\frac{L_{A'}}{L_{B'}}T,\qquad \kappa_{\rm val}=\frac{R_{A'}}{A'_{\rm pred}}.\]",
        rf"For these inputs, $T={closure['residual_observed']['c']:.6g}/{closure['residual_observed']['d']:.6g}\times {closure['exposures']['d']:.6g}/{closure['exposures']['c']:.6g}={closure['transfer']:.6g}$, $A'_{{\rm pred}}={closure['residual_observed']['b']:.6g}\times {closure['exposures']['numerator']:.6g}/{closure['exposures']['b']:.6g}\times {closure['transfer']:.6g}={closure['prediction']:.6g}$, and $\kappa_{{\rm val}}={closure['residual_observed']['numerator']:.6g}/{closure['prediction']:.6g}={closure['kappa']:.6g}$. Uncertainties are propagated with the formulas printed below the closure table.",
        r"\subsection*{Step 4: build the four-channel likelihood}",
        r"\textbf{Why:} a simultaneous Poisson likelihood preserves the finite counts in B, C, and D and propagates shared transfer and systematic uncertainties into A instead of treating the background estimate as an exact number.",
        r"\textbf{How:} for each channel use $\operatorname{Pois}(N_R\mid\nu_R)$ with $\nu_R=b_R+\mu S_R+\sum_k\gamma_kq_{k,R}$, multiply the four channel terms, and multiply by constraint terms for $\kappa$, MC normalizations, and MC statistics. This is the binned likelihood construction used by HistFactory \cite{histfactory}.",
        r"\subsection*{Step 5: fit B+C+D conditionally with $\mu=0$}",
        r"\textbf{Why:} the control-only fit estimates $\rho$, $u$, and $v$ without accessing observed A and supplies a blinded background prediction and covariance.",
        rf"\textbf{{How:}} maximize the constrained B+C+D likelihood with $\mu$ fixed to zero, save all fitted parameters and their covariance, transfer the fitted values into the A expectation, and use the covariance gradient to obtain ${payload['a_prediction']['total_background']:.6g}\pm{payload['a_prediction']['uncertainty_delta_method']:.6g}$ expected background events. Because B has {100.0 * value(row, 'signal_contamination_fraction_b'):.3f}\% nominal signal contamination, this is explicitly a conditional diagnostic; the final simultaneous fit must refit all nuisances with $\mu$ free.",
        r"\subsection*{Step 6: construct expected Asimov data}",
        r"\textbf{Why:} expected sensitivity must be evaluated without substituting the masked A observation. The Asimov data set gives the median expected behavior under a chosen model \cite{cowan}.",
        r"\textbf{How:} evaluate the full A+B+C+D model at the transferred nuisance values once with $\mu=1$ for expected measurement/discovery and once with $\mu=0$ for expected exclusion. These model expectations become the channel observations in separate expected-only workspaces.",
        r"\subsection*{Step 7: profile $\mu$ and calculate expected inference}",
        r"\textbf{Why:} profiling retains nuisance-parameter uncertainty and correlations while testing the signal strength. The asymptotic profile-likelihood framework is described in Ref.~\cite{cowan}.",
        rf"\textbf{{How:}} fit the $\mu=1$ Asimov workspace, scan $-2\ln\lambda(\mu)$ for the expected interval, compute the one-sided $q_0$ test of $\mu=0$, convert $p_0$ to $Z=\Phi^{{-1}}(1-p_0)$, and use the $\mu=0$ Asimov workspace for the 95\% upper limit. The resulting expected values are $Z={payload['expected_discovery']['z_asymptotic']:.6g}$ and $\mu_{{95}}<{payload['expected_exclusion']['background_asimov_upper_limit']:.6g}$.",
        r"\subsection*{Step 8: freeze outputs and keep observed inference locked}",
        r"\textbf{Why:} an observed result is interpretable only after selections, nonclosure treatment, nuisance model, and inference procedure are fixed independently of A.",
        r"\textbf{How:} hash the sanitized input, workspaces, scans, and result JSON; record software versions and Git state; and require a separate authorization matching the exact manifest hash before the unblinding target can open A. This report does not execute that target.",
    ]


def report_latex(config: dict, row: dict, payload: dict, plot_dir: Path) -> str:
    plot = lambda name: (plot_dir / name).resolve().as_posix()
    measurement = payload["expected_measurement"]
    discovery = payload["expected_discovery"]
    exclusion = payload["expected_exclusion"]
    closure = payload["validation_closure"]
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
        r"\usepackage{amsmath,booktabs,graphicx,xcolor}",
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
        r"The residual non-neutrino component is $b_D=L_D\rho$, $b_B=L_B\rho u$, $b_C=L_C\rho v$, and $b_A=L_A\rho u v\kappa$. The channel expectation is $\nu_R=b_R+\mu S_R+\sum_k\gamma_k q_{k,R}$. Explicit CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ MC components are not included in the residual ABCD component.",
        *likelihood_definition_lines(),
        r"\section{One-bin inputs and signal contamination}",
        r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
        latex_row("Quantity", "A", "B", "C", "D"), r"\midrule",
        latex_row("Observed data", "MASKED", row["count_b"], row["count_c"], row["count_d"]),
        latex_row("CC nue signal, mu=1", *[f"{value(row, f'cc_nue_{role.lower()}'):.5g}" for role in ROLES]),
        latex_row("Other-neutrino MC", *[f"{sum(value(row, f'{sample.lower()}_{role.lower()}') for sample in config['source']['background_samples']):.5g}" for role in ROLES]),
        r"\bottomrule\end{tabular}\end{center}",
        *input_definition_lines(),
        rf"The CC $\nu_e$ contamination fractions in B/C/D are {100.0 * value(row, 'signal_contamination_fraction_b'):.3f}\%, {100.0 * value(row, 'signal_contamination_fraction_c'):.3f}\%, and {100.0 * value(row, 'signal_contamination_fraction_d'):.3f}\%. The 1\% criterion therefore classifies contamination as \texttt{{{tex_escape(row['signal_contamination_decision'])}}}. The blinded background-only control fit fixes $\mu=0$; the final simultaneous model releases it.",
        *analysis_step_lines(config, row, payload),
        r"\section{Independent nonclosure constraint}",
        r"The table below is the one-bin analogue of the nominal-closure table. Every primed quantity belongs to the independent validation plane; $A'$ is not the masked signal region A.",
        r"\begin{center}\scriptsize\begin{tabular}{lrrrrrr}\toprule",
        latex_row(r"Density interval", r"$A'_{\rm obs}$", r"$A'_{\rm pred}\pm\sigma_{\rm pred}$", r"$T\pm\sigma_T$", r"$\kappa_{\rm val}\pm\sigma_\kappa$", r"Nonclosure", r"Pull"),
        r"\midrule",
        latex_row(
            rf"{{[{closure['density_interval'][0]:g}, {closure['density_interval'][1]:g})}}",
            f"{closure['residual_observed']['numerator']:.6g}",
            rf"${closure['prediction']:.6g}\pm{closure['prediction_uncertainty']:.6g}$",
            rf"${closure['transfer']:.6g}\pm{closure['transfer_uncertainty']:.6g}$",
            rf"${closure['kappa']:.6g}\pm{closure['kappa_uncertainty']:.6g}$",
            rf"{100.0 * closure['fractional_nonclosure']:.3f}\%",
            f"{closure['pull']:.6g}",
        ),
        r"\bottomrule\end{tabular}\end{center}",
        r"The validation-region mapping is:",
        r"\begin{description}",
        rf"\item[$A'$ (numerator)] \nolinkurl{{{closure['regions']['numerator']}}}",
        rf"\item[$B'$] \nolinkurl{{{closure['regions']['b']}}}",
        rf"\item[$C'$] \nolinkurl{{{closure['regions']['c']}}}",
        rf"\item[$D'$ (anchor)] \nolinkurl{{{closure['regions']['d']}}}",
        r"\end{description}",
        *closure_definition_lines(),
        rf"Thus the validation plane gives $\kappa_{{\rm val}}={payload['kappa']['central']:.5g}\pm{payload['kappa']['uncertainty']:.5g}$, represented by a HistFactory normsys constraint. Its source is {tex_escape(payload['kappa']['source'])}. The transport status is \nolinkurl{{{payload['kappa']['transport_status']}}}.",
        r"\section{Blinded control-region fit}",
        r"The following is a \textbf{control-region post-fit prediction}, not a signal-region post-fit result.",
        r"\begin{center}", rf"\includegraphics[width=0.82\linewidth]{{{plot('control_region_fit.pdf')}}}", r"\end{center}",
        rf"The predicted total A background is ${payload['a_prediction']['total_background']:.5g}\pm{payload['a_prediction']['uncertainty_delta_method']:.5g}$ using covariance and independent constrained nuisances.",
        r"\begin{center}", rf"\includegraphics[width=0.76\linewidth]{{{plot('signal_region_expected.pdf')}}}", r"\end{center}",
        r"\section{Expected measurement, discovery, and exclusion}",
        *inference_definition_lines(),
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
        r"\begin{thebibliography}{9}",
        r"\bibitem{histfactory} K. Cranmer, G. Lewis, L. Moneta, A. Shibata, and W. Verkerke, \emph{HistFactory: A tool for creating statistical models for use with RooFit and RooStats}, CERN-OPEN-2012-016 (2012), doi:10.17181/CERN-OPEN-2012-016. Section 5.2 gives an ABCD likelihood example. \url{https://cds.cern.ch/record/1456844}",
        r"\bibitem{choi-oh} S. Choi and H. Oh, \emph{Improved Extrapolation Methods of Data-driven Background Estimation in High-Energy Physics}, arXiv:1906.10831 (2019). \url{https://arxiv.org/abs/1906.10831}",
        r"\bibitem{cowan} G. Cowan, K. Cranmer, E. Gross, and O. Vitells, \emph{Asymptotic formulae for likelihood-based tests of new physics}, Eur. Phys. J. C 71 (2011) 1554, doi:10.1140/epjc/s10052-011-1554-0. \url{https://arxiv.org/abs/1007.1727}",
        r"\bibitem{pyhf} L. Heinrich, M. Feickert, G. Stark, and K. Cranmer, \emph{pyhf: pure-Python implementation of HistFactory statistical models}, JOSS 6 (2021) 2823, doi:10.21105/joss.02823. \url{https://doi.org/10.21105/joss.02823}",
        r"\end{thebibliography}",
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
