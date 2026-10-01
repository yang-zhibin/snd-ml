#!/usr/bin/env python3
"""Explicitly authorized one-bin A+B+C+D simultaneous unblinding inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cabinetry
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import ROOT
from scipy.stats import norm

from abcd_blinded_model import (
    ROLES,
    build_workspace,
    load_blinded_input,
    profile_interval,
    q0_value,
    sample_yields,
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
from run_abcd_blinded_expected import fit_payload, scan_rows, sha256


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--signal-data-root", required=True)
    parser.add_argument("--signal-data-summary", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--toys", type=int)
    return parser.parse_args()


def validate_authorization(
    authorization_path: str | Path,
    manifest_path: str | Path,
    config: dict,
) -> tuple[dict, dict]:
    authorization = load_yaml(authorization_path)
    if authorization.get("analysis_version") != config["version"]:
        raise ValueError("Unblinding authorization has the wrong analysis version")
    if authorization.get("authorized") is not True:
        raise PermissionError("Observed signal-region inference is not authorized")
    if not str(authorization.get("approved_by", "")).strip():
        raise ValueError("Unblinding authorization is missing approved_by")
    if not str(authorization.get("approved_at", "")).strip():
        raise ValueError("Unblinding authorization is missing approved_at")
    if authorization.get("freeze_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Authorization does not match the frozen manifest SHA256")
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("analysis_version") != config["version"]:
        raise ValueError("Freeze manifest version mismatch")
    if manifest.get("signal_region_observation_used") is not False:
        raise ValueError("Freeze manifest is not blinded")
    if manifest.get("freeze_status") != "FROZEN":
        raise ValueError("Only a clean production freeze can be unblinded")
    return authorization, manifest


def read_observed_a(
    root_path: str | Path,
    summary_path: str | Path,
    data_config: dict,
    config: dict,
) -> int:
    with open(summary_path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("status") != "PASS":
        raise ValueError("Signal-region data summary is not one PASS row")
    expected_region = config["source"]["roles"]["A"]
    if rows[0].get("region") != expected_region:
        raise ValueError("Signal-region data summary has the wrong region")
    histogram_config = data_config["histogram"]
    low, high = [float(x) for x in config["source"]["counting_range"]]
    width = float(histogram_config["bin_width"])
    first = int(round((low - float(histogram_config["min"])) / width)) + 1
    last = int(histogram_config["bins"])
    source = ROOT.TFile.Open(str(root_path))
    if not source or source.IsZombie():
        raise OSError(f"Could not open {root_path}")
    histogram = source.Get(str(histogram_config["name"]))
    if not histogram or int(histogram.GetNbinsX()) != last:
        raise ValueError("Signal-region data histogram has the wrong schema")
    if high != float(histogram_config["max"]):
        raise ValueError("Configured counting range does not end at histogram maximum")
    observed = float(histogram.Integral(first, last))
    source.Close()
    rounded = int(round(observed))
    if not math.isclose(observed, rounded, abs_tol=1.0e-9):
        raise ValueError("Signal-region observation is not an integer raw count")
    return rounded


def run_toys(
    model,
    constrained_parameters: np.ndarray,
    observed_q0: float,
    number: int,
    seed: int,
    max_failed_fraction: float,
) -> tuple[list[dict], float, float]:
    if number <= 0:
        raise ValueError("Toy count must be positive")
    np.random.seed(seed)
    samples = np.asarray(
        model.make_pdf(constrained_parameters).sample((number,)), dtype=float
    )
    rows = []
    failures = 0
    q0_values = []
    for index, sample in enumerate(samples):
        try:
            q0, _, unconstrained = q0_value(sample.tolist(), model)
            mu_hat = float(unconstrained[model.config.poi_index])
            rows.append(
                {
                    "toy_index": index,
                    "q0": q0,
                    "mu_hat": mu_hat,
                    "fit_status": "PASS",
                }
            )
            q0_values.append(q0)
        except Exception as error:  # optimizer failure is recorded and bounded
            failures += 1
            rows.append(
                {
                    "toy_index": index,
                    "q0": "",
                    "mu_hat": "",
                    "fit_status": f"FAIL:{type(error).__name__}",
                }
            )
    failed_fraction = failures / number
    if failed_fraction > max_failed_fraction:
        raise RuntimeError(
            f"Toy fit failure fraction {failed_fraction:.3%} exceeds threshold"
        )
    exceedances = sum(q0 >= observed_q0 for q0 in q0_values)
    p0 = (exceedances + 1.0) / (len(q0_values) + 1.0)
    return rows, p0, failed_fraction


def save_plots(output: Path, result: dict, scan, toy_rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    components = result["fitted_a_components"]
    names = list(components)
    values = [components[name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    bottom = 0.0
    for name, amount in zip(names, values):
        axis.bar([0], [amount], bottom=bottom, label=name)
        bottom += amount
    axis.errorbar(
        [0], [result["observed_a"]],
        yerr=[math.sqrt(result["observed_a"])], fmt="ko", label="Observed A",
    )
    axis.set_xticks([0], ["A [5000, 40000]"])
    axis.set_ylabel("Events")
    axis.set_title("Simultaneous A+B+C+D post-fit")
    axis.legend(fontsize=8)
    atomic_figure(figure, output / "observed_postfit_a.pdf")
    atomic_figure(figure, output / "observed_postfit_a.png")
    plt.close(figure)

    figure = cabinetry.visualize.scan(scan, save_figure=False, close_figure=False)
    atomic_figure(figure, output / "observed_likelihood_scan_mu.pdf")
    atomic_figure(figure, output / "observed_likelihood_scan_mu.png")
    plt.close(figure)

    passed = [float(row["q0"]) for row in toy_rows if row["fit_status"] == "PASS"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(passed, bins=40, histtype="stepfilled", alpha=0.65)
    axis.axvline(result["q0"], color="tab:red", linewidth=2, label="Observed q0")
    axis.set_xlabel("q0 under background-only toys")
    axis.set_ylabel("Pseudoexperiments")
    axis.legend()
    atomic_figure(figure, output / "toy_q0_distribution.pdf")
    atomic_figure(figure, output / "toy_q0_distribution.png")
    plt.close(figure)


def final_markdown(result: dict) -> str:
    return "\n".join(
        [
            "# Authorized one-bin ABCD simultaneous result",
            "",
            "The likelihood profiles mu and all nuisance parameters simultaneously in A+B+C+D.",
            "",
            f"mu_hat = `{result['mu_hat']:.6g}`; interval = `{result['profile_interval']}`.",
            f"Asymptotic p0 = `{result['p0_asymptotic']:.6g}` (`Z={result['z_asymptotic']:.6g}`).",
            f"Toy p0 = `{result['p0_toys']:.6g}` (`Z={result['z_toys']:.6g}`).",
            "",
        ]
    )


def final_latex(config: dict, result: dict, plot_dir: Path) -> str:
    plot = lambda name: (plot_dir / name).resolve().as_posix()
    interval = result["profile_interval"]
    interval_text = "[{}, {}]".format(
        "open" if interval[0] is None else f"{interval[0]:.6g}",
        "open" if interval[1] is None else f"{interval[1]:.6g}",
    )
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage{booktabs,graphicx,xcolor}",
        r"\usepackage[T1]{fontenc}", r"\begin{document}",
        r"\title{Authorized one-bin ABCD CC electron-neutrino result}",
        r"\author{SND@LHC analysis workflow}", r"\date{}", r"\maketitle",
        r"\section{Authorization and frozen analysis}",
        rf"This result used the explicitly authorized target for analysis version \texttt{{{tex_escape(config['version'])}}}. The one-bin range, regions, constraints, and statistical procedure were frozen before reading A.",
        r"\section{Simultaneous likelihood}",
        r"The final likelihood is $\prod_{R=A,B,C,D}\mathrm{Pois}(n_R\mid\nu_R)\prod_j\pi_j(\theta_j)$. The global fit profiles $\mu$ and every nuisance parameter simultaneously; B/C/D best-fit parameters are not frozen.",
        r"\section{Fit results}",
        r"\begin{center}\begin{tabular}{lr}\toprule",
        latex_row("Quantity", "Result"), r"\midrule",
        latex_row("Observed A", result["observed_a"]),
        latex_row(r"Best-fit $\hat\mu$", f"{result['mu_hat']:.6g}"),
        latex_row("68% profile interval", interval_text),
        latex_row("Fitted CC nue yield in A", f"{result['fitted_signal_a']:.6g}"),
        latex_row("Fitted total background in A", f"{result['fitted_background_a']:.6g}"),
        latex_row(r"Observed $q_0$", f"{result['q0']:.6g}"),
        latex_row("Asymptotic p0 / Z", f"{result['p0_asymptotic']:.6g} / {result['z_asymptotic']:.6g}"),
        latex_row("Toy p0 / Z", f"{result['p0_toys']:.6g} / {result['z_toys']:.6g}"),
        r"\bottomrule\end{tabular}\end{center}",
        r"\subsection{Pre-fit and post-fit parameters}",
        r"\begin{center}\small\begin{tabular}{lrrr}\toprule",
        latex_row("Parameter", "Pre-fit", "Post-fit", "Hesse uncertainty"),
        r"\midrule",
    ]
    for parameter in result["parameters"]:
        lines.append(
            latex_row(
                tex_escape(parameter["name"]),
                f"{parameter['prefit']:.5g}",
                f"{parameter['postfit']:.5g}",
                f"{parameter['uncertainty_hesse']:.5g}",
            )
        )
    lines.extend([
        r"\bottomrule\end{tabular}\end{center}",
        r"\section{Post-fit A composition}", r"\begin{center}",
        rf"\includegraphics[width=0.78\linewidth]{{{plot('observed_postfit_a.pdf')}}}", r"\end{center}",
        r"\section{Profile likelihood}", r"\begin{center}",
        rf"\includegraphics[width=0.78\linewidth]{{{plot('observed_likelihood_scan_mu.pdf')}}}", r"\end{center}",
        r"\section{Toy-MC validation}",
        rf"The toy ensemble used {result['toys']['requested']} background-only pseudoexperiments, with fit-failure fraction {result['toys']['failed_fraction']:.3%}. A downward fluctuation has $q_0=0$ by construction.",
        r"\begin{center}", rf"\includegraphics[width=0.78\linewidth]{{{plot('toy_q0_distribution.pdf')}}}", r"\end{center}",
        r"\section{Limitations}",
        r"Interpretation remains conditional on the independently derived but not secondarily validated nonclosure transport and on the configured MC systematics.",
        r"\end{document}",
    ])
    return "\n".join(lines) + "\n"


def run_observed(
    config: dict,
    row: dict,
    observed_a: int,
    output_root: Path,
    number_of_toys: int,
) -> dict:
    observations = {
        "A": [float(observed_a)],
        **{
            role: [value(row, f"count_{role.lower()}")]
            for role in ("B", "C", "D")
        },
    }
    spec = build_workspace(config, row, ROLES, observations)
    workspace = pyhf.Workspace(spec)
    model, data = cabinetry.model_utils.model_and_data(workspace)
    fit = cabinetry.fit.fit(
        model, data,
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    scan = cabinetry.fit.scan(
        model, data, config["model"]["signal_parameter"],
        par_range=tuple(float(x) for x in config["fit"]["mu_scan_range"]),
        n_steps=int(config["fit"]["mu_scan_points"]),
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    interval = profile_interval(scan.parameter_values, scan.delta_nlls)
    q0, constrained, unconstrained = q0_value(data, model)
    mu_hat = float(unconstrained[model.config.poi_index])
    if mu_hat <= 0.0:
        q0 = 0.0
    p0_asymptotic = float(norm.sf(math.sqrt(q0)))
    toy_rows, p0_toys, failed_fraction = run_toys(
        model, constrained, q0, number_of_toys,
        int(config["toy_mc"]["seed"]),
        float(config["toy_mc"]["max_failed_fraction"]),
    )
    components = sample_yields(model, fit.bestfit, "A")
    signal = components["cc_nue_signal"]
    background = sum(amount for name, amount in components.items() if name != "cc_nue_signal")
    prefit = np.asarray(model.config.suggested_init(), dtype=float)
    parameters = [
        {
            "name": name,
            "prefit": float(prefit[index]),
            "postfit": float(fit.bestfit[index]),
            "uncertainty_hesse": float(fit.uncertainty[index]),
        }
        for index, name in enumerate(fit.labels)
    ]
    z_toys = max(0.0, float(norm.isf(p0_toys)))
    result = {
        "analysis_version": config["version"],
        "status": "AUTHORIZED_OBSERVED_SIMULTANEOUS_FIT",
        "model": "one-bin simultaneous A+B+C+D with all nuisances profiled",
        "control_parameters_frozen": False,
        "observed_a": observed_a,
        "mu_hat": mu_hat,
        "profile_interval": list(interval),
        "fitted_signal_a": signal,
        "fitted_background_a": background,
        "fitted_total_a": signal + background,
        "fitted_a_components": components,
        "parameters": parameters,
        "fit": fit_payload(fit),
        "q0": q0,
        "p0_asymptotic": p0_asymptotic,
        "z_asymptotic": float(norm.isf(p0_asymptotic)),
        "p0_toys": p0_toys,
        "z_toys": z_toys,
        "asymptotic_toy_difference": {
            "delta_p0": p0_toys - p0_asymptotic,
            "delta_z": z_toys - float(norm.isf(p0_asymptotic)),
        },
        "toys": {
            "requested": number_of_toys,
            "passed": sum(row["fit_status"] == "PASS" for row in toy_rows),
            "failed_fraction": failed_fraction,
            "seed": int(config["toy_mc"]["seed"]),
        },
    }
    output = output_root / config["output"]["unblinded_directory"]
    plots = output / config["output"]["final_plot_directory"]
    save_plots(plots, result, scan, toy_rows)
    atomic_text(
        output / config["output"]["observed_workspace"],
        json.dumps(spec, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["observed_results"],
        json.dumps(result, indent=2) + "\n",
    )
    atomic_text(
        output / config["output"]["observed_mu_scan"],
        csv_text(scan_rows(scan)),
    )
    atomic_text(
        output / config["output"]["toy_q0"], csv_text(toy_rows)
    )
    atomic_text(
        output / config["output"]["final_report_markdown"], final_markdown(result)
    )
    atomic_text(
        output / config["output"]["final_report_tex"],
        final_latex(config, result, plots),
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    # Authorization and freeze validation deliberately happen before A is opened.
    validate_authorization(args.authorization, args.freeze_manifest, config)
    row = load_blinded_input(args.input, config)
    data_config = load_yaml(args.data_config)
    observed_a = read_observed_a(
        args.signal_data_root, args.signal_data_summary, data_config, config
    )
    number_of_toys = args.toys or int(config["toy_mc"]["default_toys"])
    result = run_observed(
        config, row, observed_a, Path(args.output_root), number_of_toys
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "mu_hat": result["mu_hat"],
                "z_asymptotic": result["z_asymptotic"],
                "z_toys": result["z_toys"],
            }
        )
    )


if __name__ == "__main__":
    main()
