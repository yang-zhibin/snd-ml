#!/usr/bin/env python3
"""Run control-only ABCD fits and blinded Asimov expected inference."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import cabinetry
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from scipy.stats import norm

from fit_abcd_density_signal_region import (
    atomic_figure,
    atomic_text,
    csv_text,
    latex_row,
    load_yaml,
    tex_escape,
)


ROLES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sanitized(path: str | Path, config: dict) -> dict[str, list[dict]]:
    samples = [config["source"]["signal_sample"], *config["source"]["background_samples"]]
    required = {
        "input_version", "blinding_status", "role", "region", "bin_index",
        "bin_low", "bin_high", "exposure", "observed_data",
    }
    for sample in samples:
        required.update({f"{sample.lower()}_sum_weights", f"{sample.lower()}_sum_weights_squared"})
    output = {role: [] for role in ROLES}
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Sanitized input is missing: " + ", ".join(sorted(missing)))
        for row in reader:
            role = row["role"]
            if role not in output:
                raise ValueError(f"Unexpected role {role}")
            if row["input_version"] != config["version"]:
                raise ValueError("Input version mismatch")
            if row["region"] != config["source"]["roles"][role]:
                raise ValueError(f"Region mismatch for role {role}")
            if role == "A" and row["observed_data"].strip():
                raise ValueError("Observed signal-region data found in sanitized input")
            if role != "A" and not row["observed_data"].strip():
                raise ValueError(f"Missing observed control data for role {role}")
            output[role].append(row)
    for role, rows in output.items():
        rows.sort(key=lambda row: int(row["bin_index"]))
        if len(rows) != 80:
            raise ValueError(f"Expected 80 regular bins for role {role}")
        if [int(row["bin_index"]) for row in rows] != list(range(1, 81)):
            raise ValueError(f"Non-contiguous source bins for role {role}")
    return output


def aggregate(source: dict[str, list[dict]], edges: list[float], config: dict) -> list[dict]:
    if edges[-1] != 40000 or any(right <= left or left % 500 or right % 500 for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("Edges must be increasing 500-unit boundaries ending at 40000")
    signal = config["source"]["signal_sample"].lower()
    backgrounds = [sample.lower() for sample in config["source"]["background_samples"]]
    output = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        item = {"bin_index": index, "bin_low": low, "bin_high": high}
        for role in ROLES:
            selected = [row for row in source[role] if low <= float(row["bin_low"]) < high]
            if len(selected) != int((high - low) / 500):
                raise ValueError(f"Incomplete coverage for {role} [{low}, {high})")
            key = role.lower()
            item[f"exposure_{key}"] = float(selected[0]["exposure"])
            item[f"observed_{key}"] = "" if role == "A" else sum(int(row["observed_data"]) for row in selected)
            item[f"signal_{key}"] = sum(float(row[f"{signal}_sum_weights"]) for row in selected)
            item[f"signal_variance_{key}"] = sum(float(row[f"{signal}_sum_weights_squared"]) for row in selected)
            for sample in backgrounds:
                item[f"{sample}_{key}"] = sum(float(row[f"{sample}_sum_weights"]) for row in selected)
                item[f"{sample}_variance_{key}"] = sum(float(row[f"{sample}_sum_weights_squared"]) for row in selected)
            item[f"other_neutrino_{key}"] = sum(item[f"{sample}_{key}"] for sample in backgrounds)
        output.append(item)
    return output


def residual_initials(inputs: list[dict]) -> tuple[list[float], list[float], list[float]]:
    residual = {}
    for role in ("B", "C", "D"):
        key = role.lower()
        residual[role] = [float(row[f"observed_{key}"]) - row[f"other_neutrino_{key}"] for row in inputs]
        if min(residual[role]) <= 0:
            raise ValueError(f"Non-positive residual background initialization in {role}")
    rho, u, v = [], [], []
    for i, row in enumerate(inputs):
        rho.append(residual["D"][i] / row["exposure_d"])
        u.append(residual["B"][i] * row["exposure_d"] / (residual["D"][i] * row["exposure_b"]))
        v.append(residual["C"][i] * row["exposure_d"] / (residual["D"][i] * row["exposure_c"]))
    return rho, u, v


def parameter(name: str, values: list[float]) -> dict:
    return {"name": name, "inits": values, "bounds": [[0.0, max(20.0 * value, 20.0)] for value in values]}


def build_workspace(config: dict, inputs: list[dict], roles: tuple[str, ...], observations: dict[str, list[float]]) -> dict:
    rho, u, v = residual_initials(inputs)
    kappa = float(config["model"]["nonclosure"]["nominal"])
    modifiers = {
        "A": [{"name": "rho", "type": "shapefactor", "data": None}, {"name": "u", "type": "shapefactor", "data": None}, {"name": "v", "type": "shapefactor", "data": None}],
        "B": [{"name": "rho", "type": "shapefactor", "data": None}, {"name": "u", "type": "shapefactor", "data": None}],
        "C": [{"name": "rho", "type": "shapefactor", "data": None}, {"name": "v", "type": "shapefactor", "data": None}],
        "D": [{"name": "rho", "type": "shapefactor", "data": None}],
    }
    channels = []
    for role in roles:
        key = role.lower()
        samples = [{
            "name": "abcd_residual_background",
            "data": [row[f"exposure_{key}"] * (kappa if role == "A" else 1.0) for row in inputs],
            "modifiers": modifiers[role],
        }]
        for sample in config["source"]["background_samples"]:
            relative = float(config["model"]["background_mc_normalization"][sample])
            samples.append({
                "name": sample.lower(),
                "data": [row[f"{sample.lower()}_{key}"] for row in inputs],
                "modifiers": [{"name": f"norm_{sample.lower()}", "type": "normsys", "data": {"lo": max(0.01, 1.0 - relative), "hi": 1.0 + relative}}],
            })
        samples.append({
            "name": "cc_nue_signal",
            "data": [row[f"signal_{key}"] for row in inputs],
            "modifiers": [{"name": config["model"]["signal_parameter"], "type": "normfactor", "data": None}],
        })
        channels.append({"name": f"{role}_region", "samples": samples})
    mu = config["model"]["signal_parameter"]
    return {
        "channels": channels,
        "observations": [{"name": f"{role}_region", "data": observations[role]} for role in roles],
        "measurements": [{"name": "measurement", "config": {"poi": mu, "parameters": [
            {"name": mu, "inits": [float(config["model"]["signal_initial"])], "bounds": [[float(x) for x in config["model"]["signal_bounds"]]]},
            parameter("rho", rho), parameter("u", u), parameter("v", v),
        ]}}],
        "version": "1.0.0",
    }


def transfer_parameters(source_model, source_values, target_model, mu: float) -> list[float]:
    target = np.asarray(target_model.config.suggested_init(), dtype=float)
    for name in source_model.config.par_order:
        if name in target_model.config.par_order:
            target[target_model.config.par_slice(name)] = np.asarray(source_values)[source_model.config.par_slice(name)]
    target[target_model.config.poi_index] = mu
    return target.tolist()


def replace_observations(spec: dict, model, parameters: list[float]) -> dict:
    output = copy.deepcopy(spec)
    expected = np.asarray(model.expected_data(parameters, include_auxdata=False), dtype=float)
    for observation in output["observations"]:
        observation["data"] = expected[model.config.channel_slices[observation["name"]]].tolist()
    return output


def expected_components(model, parameters: list[float], role: str) -> np.ndarray:
    values = np.asarray(model.expected_data(parameters, include_auxdata=False), dtype=float)
    return values[model.config.channel_slices[f"{role}_region"]]


def save_plots(out: Path, edges: list[float], inputs: list[dict], control_model, control_data, control_fit, full_model, background_parameters, signal_parameters, scan) -> None:
    out.mkdir(parents=True, exist_ok=True)
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
    widths = (np.asarray(edges[1:]) - np.asarray(edges[:-1])) / 2.0
    prefit = np.asarray(control_model.expected_data(control_model.config.suggested_init(), include_auxdata=False))
    postfit = np.asarray(control_model.expected_data(control_fit.bestfit, include_auxdata=False))
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for axis, role in zip(axes, ("B", "C", "D")):
        slc = control_model.config.channel_slices[f"{role}_region"]
        observed = np.asarray(control_data[slc], dtype=float)
        axis.stairs(prefit[slc], edges, label="Pre-fit", color="tab:orange")
        axis.stairs(postfit[slc], edges, label="Post-control-fit", color="tab:blue")
        axis.errorbar(centers, observed, yerr=np.sqrt(observed), xerr=widths, fmt="ko", label="Control data")
        axis.set_ylabel(f"{role} events")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("density_scifi_sndsw")
    figure.suptitle("Blinded control-region fit (A data absent)")
    atomic_figure(figure, out / "control_fit.pdf")
    atomic_figure(figure, out / "control_fit.png")
    plt.close(figure)

    background = expected_components(full_model, background_parameters, "A")
    signal_total = expected_components(full_model, signal_parameters, "A")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.stairs(background, edges, fill=True, alpha=0.45, label="Expected background")
    axis.stairs(signal_total, edges, label="Background + injected CC nue", color="tab:red", linewidth=2)
    axis.set_xlabel("density_scifi_sndsw")
    axis.set_ylabel("Expected events")
    axis.set_title("Blinded signal-region expectation; no A data points")
    axis.legend()
    atomic_figure(figure, out / "signal_region_expected.pdf")
    atomic_figure(figure, out / "signal_region_expected.png")
    plt.close(figure)

    figure = cabinetry.visualize.scan(scan, save_figure=False, close_figure=False)
    atomic_figure(figure, out / "expected_likelihood_scan_mu.pdf")
    atomic_figure(figure, out / "expected_likelihood_scan_mu.png")
    plt.close(figure)
    figure = cabinetry.visualize.correlation_matrix(control_fit, save_figure=False, close_figure=False)
    atomic_figure(figure, out / "control_correlation.pdf")
    atomic_figure(figure, out / "control_correlation.png")
    plt.close(figure)


def make_payload(config: dict, analysis: dict, inputs: list[dict], control_model, control_fit, full_model, expected_fit, p0: float, z: float) -> dict:
    return {
        "analysis_version": config["version"],
        "analysis": analysis,
        "status": "PROVISIONAL_EXPECTED_ONLY",
        "blinding_status": config["blinding"]["status"],
        "signal_region_observation_used": False,
        "nonclosure_validation_status": config["model"]["nonclosure"]["validation_status"],
        "background_mc_uncertainty_status": config["model"]["background_mc_uncertainty_status"],
        "software": {"cabinetry": importlib.metadata.version("cabinetry"), "pyhf": pyhf.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "model": {"control_channels": list(control_model.config.channels), "full_channels": list(full_model.config.channels), "poi": full_model.config.poi_name, "n_bins": len(inputs)},
        "control_fit": {"labels": list(control_fit.labels), "bestfit": np.asarray(control_fit.bestfit).tolist(), "uncertainty": np.asarray(control_fit.uncertainty).tolist()},
        "expected_signal_fit": {"labels": list(expected_fit.labels), "bestfit": np.asarray(expected_fit.bestfit).tolist(), "uncertainty": np.asarray(expected_fit.uncertainty).tolist()},
        "expected_discovery": {"injected_mu": float(config["model"]["signal_injection"]), "p0_asymptotic": p0, "z_asymptotic": z, "toy_validation_performed": False},
        "inputs": inputs,
    }


def report_markdown(payload: dict) -> str:
    return "\n".join([
        f"# {payload['analysis']['title']}", "",
        "**Provisional expected-only reblinded-development result.** No observed A-region count was read.", "",
        f"Expected asymptotic discovery sensitivity: `Z = {payload['expected_discovery']['z_asymptotic']:.6g}` for injected `mu = {payload['expected_discovery']['injected_mu']:g}`.", "",
        f"Nonclosure status: `{payload['nonclosure_validation_status']}`. MC-normalization uncertainty status: `{payload['background_mc_uncertainty_status']}`.", "",
        "This is not an observed result and is not suitable for a physics claim until nonclosure and systematic uncertainties are independently validated.", "",
    ])


def report_latex(config: dict, payload: dict, plot_dir: Path) -> str:
    analysis = payload["analysis"]
    plot = lambda name: (plot_dir / name).resolve().as_posix()
    rows = payload["inputs"]
    lines = [
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage{booktabs,graphicx,xcolor}", r"\usepackage[T1]{fontenc}", r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{5pt}", r"\begin{document}",
        rf"\title{{{tex_escape(analysis['title'])}}}", r"\author{SND@LHC analysis workflow}", r"\date{}", r"\maketitle",
        r"\begin{center}\fcolorbox{black}{yellow!12}{\parbox{0.92\textwidth}{\textbf{Provisional expected-only reblinded-development result.} No observed signal-region A count is an input. Earlier development inspected A, so historical blindness cannot be restored.}}\end{center}",
        r"\section{Analysis status and blinding}",
        r"The control fit uses observed B/C/D data. Region A contains MC templates and an Asimov observation only. Observed inference is locked and was not run.",
        r"\section{Dataset, exposure, selection, and binning}",
        rf"The observable is \texttt{{{tex_escape(config['source']['observable'])}}}; edges are $[{', '.join(f'{x:g}' for x in analysis['bin_edges'])}]$. Counts are raw Poisson observations and MC is normalized to each region exposure.",
        r"\section{ABCD background construction}",
        r"For each bin, $b_D=L_D\rho$, $b_B=L_B\rho u$, $b_C=L_C\rho v$, and $b_A=L_A\rho u v\kappa$. The fitted component is residual background after explicitly modelled neutrino MC.",
        r"\section{Signal and other-neutrino MC}",
        r"CC $\nu_e$ is the signal and is scaled by the shared parameter $\mu$. CC $\nu_\mu$, NC $\nu_e$, and NC $\nu_\mu$ are separate components with shared 30\% placeholder normalization constraints. MC statistical uncertainties are not yet included.",
        r"\section{Likelihood}",
        r"The blinded likelihood is $\prod_{R=B,C,D}\prod_i\mathrm{Pois}(n_{R,i}\mid\nu_{R,i})\prod_j\pi_j(\theta_j)$. The expected full model adds an Asimov A observation; it never substitutes observed A data.",
        r"\section{Input yields}", r"\begin{center}\small\begin{tabular}{rrrrrrr}\toprule",
        latex_row("Density interval", "B", "C", "D", "CC nue A", "Other nu A", "A observed"), r"\midrule",
    ]
    for row in rows:
        lines.append(latex_row("{" + f"[{row['bin_low']:g},{row['bin_high']:g})" + "}", row["observed_b"], row["observed_c"], row["observed_d"], f"{row['signal_a']:.4g}", f"{row['other_neutrino_a']:.4g}", "MASKED"))
    result = payload["expected_discovery"]
    lines.extend([
        r"\bottomrule\end{tabular}\end{center}",
        r"\section{Validation closure and nonclosure constraint}",
        r"The nonclosure factor is provisionally fixed to $\kappa=1$. Its status is \textbf{UNVALIDATED}; current closure studies do not justify final inference.",
        r"\section{Systematic-uncertainty model}",
        r"The 30\% other-neutrino normalization constraints are placeholders. Detector, flux, cross-section, signal-efficiency, MC-statistical, and validated nonclosure uncertainties remain to be frozen.",
        r"\section{Blinded control-region fit}", r"\begin{center}", rf"\includegraphics[width=0.82\linewidth]{{{plot('control_fit.pdf')}}}", r"\end{center}",
        r"\section{Predicted signal-region composition}", r"\begin{center}", rf"\includegraphics[width=0.82\linewidth]{{{plot('signal_region_expected.pdf')}}}", r"\end{center}",
        r"\section{Expected Asimov sensitivity}",
        rf"For injected $\mu={result['injected_mu']:g}$, the asymptotic background-only test gives $p_0={result['p0_asymptotic']:.5g}$ and $Z={result['z_asymptotic']:.5g}$. Toy validation has not been performed.",
        r"\begin{center}", rf"\includegraphics[width=0.76\linewidth]{{{plot('expected_likelihood_scan_mu.pdf')}}}", r"\end{center}",
        r"\section{Nuisance correlations}", r"\begin{center}", rf"\includegraphics[width=0.76\linewidth]{{{plot('control_correlation.pdf')}}}", r"\end{center}",
        r"\section{Pre-unblinding freeze checklist}",
        r"The configuration, sanitized input, and workspaces are hashed in the freeze manifest. Before unblinding, validate $\kappa$, replace placeholder uncertainties, add MC statistical uncertainties, validate toys, and approve the primary binning.",
        r"\section{Observed result after unblinding}", r"\textbf{LOCKED: no observed signal-region fit or significance was run.}",
        r"\section{Goodness of fit and robustness}",
        r"The one-bin model is saturated and cannot test shape. The two-bin construction is exploratory and previously showed poor development-data compatibility; neither is promoted to a final model here.",
        r"\section{Interpretation and limitations}",
        r"This report verifies the blinded software path and quantifies provisional expected sensitivity. It is not an observed measurement, discovery claim, or exclusion result.", r"\end{document}",
    ])
    return "\n".join(lines) + "\n"


def run_analysis(config: dict, source: dict[str, list[dict]], name: str, output_root: Path) -> dict:
    analysis = dict(config["analyses"][name]); analysis["name"] = name
    edges = [float(value) for value in analysis["bin_edges"]]; analysis["bin_edges"] = edges
    inputs = aggregate(source, edges, config)
    control_observations = {role: [float(row[f"observed_{role.lower()}"]) for row in inputs] for role in ("B", "C", "D")}
    control_spec = build_workspace(config, inputs, ("B", "C", "D"), control_observations)
    control_workspace = pyhf.Workspace(control_spec); control_model, control_data = cabinetry.model_utils.model_and_data(control_workspace)
    init = control_model.config.suggested_init(); init[control_model.config.poi_index] = 0.0
    fixed = control_model.config.suggested_fixed(); fixed[control_model.config.poi_index] = True
    control_fit = cabinetry.fit.fit(control_model, control_data, init_pars=init, fix_pars=fixed, maxiter=int(config["fit"]["maxiter"]), tolerance=float(config["fit"]["tolerance"]))

    placeholder = {role: [0.0] * len(inputs) for role in ROLES}
    full_template = build_workspace(config, inputs, ROLES, placeholder)
    template_workspace = pyhf.Workspace(full_template); full_model = template_workspace.model()
    background_parameters = transfer_parameters(control_model, control_fit.bestfit, full_model, 0.0)
    signal_parameters = transfer_parameters(control_model, control_fit.bestfit, full_model, float(config["model"]["signal_injection"]))
    background_spec = replace_observations(full_template, full_model, background_parameters)
    signal_spec = replace_observations(full_template, full_model, signal_parameters)
    signal_workspace = pyhf.Workspace(signal_spec); expected_model, expected_data = cabinetry.model_utils.model_and_data(signal_workspace)
    expected_fit = cabinetry.fit.fit(expected_model, expected_data, maxiter=int(config["fit"]["maxiter"]), tolerance=float(config["fit"]["tolerance"]))
    scan = cabinetry.fit.scan(expected_model, expected_data, config["model"]["signal_parameter"], par_range=tuple(float(x) for x in config["fit"]["mu_scan_range"]), n_steps=int(config["fit"]["mu_scan_points"]), maxiter=int(config["fit"]["maxiter"]), tolerance=float(config["fit"]["tolerance"]))
    p0 = float(np.asarray(pyhf.infer.hypotest(0.0, expected_data, expected_model, test_stat="q0")).reshape(-1)[0])
    z = float(norm.isf(p0))
    payload = make_payload(config, analysis, inputs, control_model, control_fit, expected_model, expected_fit, p0, z)
    out = output_root / analysis["output_directory"]; plots = out / config["output"]["plot_directory"]
    save_plots(plots, edges, inputs, control_model, control_data, control_fit, expected_model, background_parameters, signal_parameters, scan)
    atomic_text(out / config["output"]["model_inputs"], csv_text(inputs))
    atomic_text(out / config["output"]["control_workspace"], json.dumps(control_spec, indent=2) + "\n")
    atomic_text(out / config["output"]["expected_background_workspace"], json.dumps(background_spec, indent=2) + "\n")
    atomic_text(out / config["output"]["expected_signal_workspace"], json.dumps(signal_spec, indent=2) + "\n")
    atomic_text(out / config["output"]["expected_results"], json.dumps(payload, indent=2) + "\n")
    atomic_text(out / config["output"]["mu_scan"], csv_text([{"mu": float(x), "delta_twice_nll": float(y)} for x, y in zip(scan.parameter_values, scan.delta_nlls)]))
    atomic_text(out / config["output"]["report_markdown"], report_markdown(payload))
    atomic_text(out / config["output"]["report_tex"], report_latex(config, payload, plots))
    return {"analysis": name, "output_directory": str(out), "z": z, "p0": p0, "workspace_sha256": sha256(out / config["output"]["expected_signal_workspace"])}


def main() -> None:
    args = parse_args(); config = load_yaml(args.config)
    if config["blinding"].get("observed_inference_authorized"):
        raise ValueError("Expected-only runner refuses an observed-inference configuration")
    source = load_sanitized(args.input, config); output_root = Path(args.output_root)
    results = [run_analysis(config, source, name, output_root) for name in ("cutcount", "shape")]
    manifest = {"analysis_version": config["version"], "blinding_status": config["blinding"]["status"], "signal_region_observation_used": False, "sanitized_input_sha256": sha256(args.input), "config_sha256": sha256(args.config), "analyses": results, "unblinding_authorized": False}
    atomic_text(output_root / config["output"]["freeze_manifest"], json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PROVISIONAL_EXPECTED_ONLY", "analyses": results}))


if __name__ == "__main__":
    main()
