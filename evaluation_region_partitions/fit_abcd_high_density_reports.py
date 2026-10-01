#!/usr/bin/env python3
"""Build two versioned high-density simultaneous A/B/C/D fit reports."""

from __future__ import annotations

import argparse
import csv
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
from cabinetry.fit.results_containers import FitResults

from fit_abcd_density_signal_region import (
    atomic_figure,
    atomic_text,
    csv_text,
    latex_row,
    load_yaml,
    tex_escape,
)


REGION_LABELS = {
    "A": "A signal",
    "B": "B control",
    "C": "C control",
    "D": "D control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def load_source_rows(path: str | Path, config: dict) -> dict[str, list[dict]]:
    required = {
        "table_version",
        "region",
        "row_type",
        "bin_low",
        "bin_high",
        "data_exposure",
        "data_raw_events",
        "cc_nue_sum_weights",
        "cc_nue_sum_weights_squared",
        "source_validation_status",
    }
    region_names = config["source"]["regions"]
    reverse = {name: role for role, name in region_names.items()}
    selected = {role: [] for role in region_names}
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Input table is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            role = reverse.get(row["region"])
            if role is None or row["row_type"] != "bin":
                continue
            if row["table_version"] != config["source"]["table_version"]:
                raise ValueError("Unexpected source table version")
            if row["source_validation_status"] != "PASS":
                raise ValueError("Source row did not pass validation")
            selected[role].append(row)
    for role, rows in selected.items():
        rows.sort(key=lambda row: float(row["bin_low"]))
        if len(rows) != 80:
            raise ValueError(f"Expected 80 regular source bins for role {role}")
        edges = [float(rows[0]["bin_low"])] + [float(row["bin_high"]) for row in rows]
        if edges != [float(value) for value in range(0, 40001, 500)]:
            raise ValueError(f"Unexpected 500-unit source binning for role {role}")
        exposures = {float(row["data_exposure"]) for row in rows}
        if len(exposures) != 1:
            raise ValueError(f"Exposure changes across source bins for role {role}")
    return selected


def aggregate_inputs(
    source_rows: dict[str, list[dict]], edges: list[float]
) -> list[dict]:
    if edges[0] < 0 or edges[-1] != 40000 or any(
        right <= left or left % 500 or right % 500
        for left, right in zip(edges[:-1], edges[1:])
    ):
        raise ValueError("Analysis edges must be increasing 500-unit edges ending at 40000")
    output = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        item: dict[str, float | int] = {
            "bin_index": index,
            "bin_low": low,
            "bin_high": high,
        }
        for role in ("A", "B", "C", "D"):
            rows = [
                row
                for row in source_rows[role]
                if low <= float(row["bin_low"]) < high
            ]
            if len(rows) != int((high - low) / 500):
                raise ValueError(f"Source-bin coverage failure for {role} [{low}, {high})")
            item[f"count_{role.lower()}"] = sum(
                int(row["data_raw_events"]) for row in rows
            )
            item[f"signal_{role.lower()}"] = sum(
                float(row["cc_nue_sum_weights"]) for row in rows
            )
            item[f"signal_variance_{role.lower()}"] = sum(
                float(row["cc_nue_sum_weights_squared"]) for row in rows
            )
            item[f"exposure_{role.lower()}"] = float(rows[0]["data_exposure"])
        b = float(item["count_b"])
        c = float(item["count_c"])
        d = float(item["count_d"])
        if min(b, c, d) <= 0:
            raise ValueError("Every high-density ABCD bin must have positive B/C/D counts")
        exposure_factor = (
            float(item["exposure_a"])
            * float(item["exposure_d"])
            / (float(item["exposure_b"]) * float(item["exposure_c"]))
        )
        prediction = b * c / d * exposure_factor
        relative_variance = 1.0 / b + 1.0 / c + 1.0 / d
        item["abcd_prediction_a"] = prediction
        item["abcd_stat_uncertainty"] = prediction * math.sqrt(relative_variance)
        item["abcd_exposure_factor"] = exposure_factor
        output.append(item)
    return output


def parameter_config(name: str, inits: list[float], upper_scale: float) -> dict:
    bounds = [[0.0, max(upper_scale * value, upper_scale)] for value in inits]
    return {"name": name, "inits": inits, "bounds": bounds}


def build_workspace(config: dict, inputs: list[dict]) -> dict:
    n_bins = len(inputs)
    exposures = {
        role: [float(row[f"exposure_{role.lower()}"]) for row in inputs]
        for role in ("A", "B", "C", "D")
    }
    rho = [float(row["count_d"]) / exposures["D"][i] for i, row in enumerate(inputs)]
    u = [
        float(row["count_b"]) * exposures["D"][i]
        / (float(row["count_d"]) * exposures["B"][i])
        for i, row in enumerate(inputs)
    ]
    v = [
        float(row["count_c"]) * exposures["D"][i]
        / (float(row["count_d"]) * exposures["C"][i])
        for i, row in enumerate(inputs)
    ]
    kappa = float(config["model"]["nonclosure_kappa"])
    modifiers = {
        "A": [
            {"name": "rho", "type": "shapefactor", "data": None},
            {"name": "u", "type": "shapefactor", "data": None},
            {"name": "v", "type": "shapefactor", "data": None},
        ],
        "B": [
            {"name": "rho", "type": "shapefactor", "data": None},
            {"name": "u", "type": "shapefactor", "data": None},
        ],
        "C": [
            {"name": "rho", "type": "shapefactor", "data": None},
            {"name": "v", "type": "shapefactor", "data": None},
        ],
        "D": [{"name": "rho", "type": "shapefactor", "data": None}],
    }
    channels = []
    observations = []
    for role in ("A", "B", "C", "D"):
        scale = kappa if role == "A" else 1.0
        channels.append(
            {
                "name": REGION_LABELS[role].replace(" ", "_"),
                "samples": [
                    {
                        "name": "abcd_background",
                        "data": [scale * value for value in exposures[role]],
                        "modifiers": modifiers[role],
                    },
                    {
                        "name": "cc_nue_signal",
                        "data": [float(row[f"signal_{role.lower()}"]) for row in inputs],
                        "modifiers": [
                            {
                                "name": config["model"]["signal_parameter"],
                                "type": "normfactor",
                                "data": None,
                            }
                        ],
                    },
                ],
            }
        )
        observations.append(
            {
                "name": REGION_LABELS[role].replace(" ", "_"),
                "data": [int(row[f"count_{role.lower()}"]) for row in inputs],
            }
        )
    mu_name = config["model"]["signal_parameter"]
    return {
        "channels": channels,
        "observations": observations,
        "measurements": [
            {
                "name": "measurement",
                "config": {
                    "poi": mu_name,
                    "parameters": [
                        {
                            "name": mu_name,
                            "inits": [float(config["model"]["signal_initial"])],
                            "bounds": [
                                [float(x) for x in config["model"]["signal_bounds"]]
                            ],
                        },
                        parameter_config("rho", rho, 20.0),
                        parameter_config("u", u, 20.0),
                        parameter_config("v", v, 20.0),
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def point_results(model: pyhf.pdf.Model, mu: float, label_source) -> FitResults:
    values = np.asarray(model.config.suggested_init(), dtype=float)
    values[model.config.poi_index] = mu
    return FitResults(
        bestfit=values,
        uncertainty=np.zeros_like(values),
        labels=list(label_source.labels),
        corr_mat=np.eye(len(values)),
        best_twice_nll=float("nan"),
        goodness_of_fit=None,
        minos_uncertainty={},
    )


def save_plots(
    output_dir: Path,
    config: dict,
    edges: list[float],
    model: pyhf.pdf.Model,
    data: list[float],
    fit_results,
    mu_scan,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_config = {
        "Regions": [
            {
                "Name": "A_signal",
                "Variable": config["source"]["observable"],
                "Binning": edges,
            }
        ]
    }
    predictions = {
        "before_fit": cabinetry.model_utils.prediction(
            model,
            fit_results=point_results(model, 0.0, fit_results),
            label="Before fit: control-derived ABCD",
        ),
        "prefit": cabinetry.model_utils.prediction(model, label="Pre-fit"),
        "postfit": cabinetry.model_utils.prediction(
            model, fit_results=fit_results, label="Post-fit"
        ),
    }
    for name, prediction in predictions.items():
        plots = cabinetry.visualize.data_mc(
            prediction,
            data,
            config=plot_config,
            channels="A_signal",
            save_figure=False,
            close_figure=False,
        )
        if not plots or len(plots) != 1:
            raise ValueError("Expected one signal-region comparison plot")
        figure = plots[0]["figure"]
        atomic_figure(figure, output_dir / f"{name}_signal_region.pdf")
        atomic_figure(figure, output_dir / f"{name}_signal_region.png")
        plt.close(figure)
    figure = cabinetry.visualize.scan(mu_scan, save_figure=False, close_figure=False)
    atomic_figure(figure, output_dir / "likelihood_scan_mu.pdf")
    atomic_figure(figure, output_dir / "likelihood_scan_mu.png")
    plt.close(figure)
    figure = cabinetry.visualize.correlation_matrix(
        fit_results, save_figure=False, close_figure=False
    )
    atomic_figure(figure, output_dir / "correlation_matrix.pdf")
    atomic_figure(figure, output_dir / "correlation_matrix.png")
    plt.close(figure)


def result_payload(config: dict, analysis: dict, model, inputs, fit_results) -> dict:
    parameters = [
        {
            "name": name,
            "bestfit": float(fit_results.bestfit[index]),
            "uncertainty_hesse": float(fit_results.uncertainty[index]),
            "at_lower_bound": abs(float(fit_results.bestfit[index]))
            <= float(config["fit"]["boundary_tolerance"]),
        }
        for index, name in enumerate(fit_results.labels)
    ]
    gof = float(fit_results.goodness_of_fit)
    return {
        "version": config["version"],
        "analysis": analysis,
        "scope": "explicitly unblinded diagnostic; kappa fixed to one",
        "software": {
            "cabinetry": importlib.metadata.version("cabinetry"),
            "pyhf": importlib.metadata.version("pyhf"),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "model": {
            "channels": list(model.config.channels),
            "samples": list(model.config.samples),
            "n_bins_total": model.config.nmaindata,
            "n_parameters": model.config.npars,
            "parameter_of_interest": model.config.poi_name,
            "degrees_of_freedom": model.config.nmaindata - model.config.npars,
        },
        "inputs": inputs,
        "parameters": parameters,
        "best_twice_nll": float(fit_results.best_twice_nll),
        "goodness_of_fit_p_value": gof if math.isfinite(gof) else None,
        "correlation_matrix": {
            "labels": list(fit_results.labels),
            "values": np.asarray(fit_results.corr_mat, dtype=float).tolist(),
        },
    }


def report_markdown(config: dict, payload: dict) -> str:
    mu = next(item for item in payload["parameters"] if item["name"] == "mu")
    boundary = [item["name"] for item in payload["parameters"] if item["at_lower_bound"]]
    lines = [
        f"# {payload['analysis']['title']}",
        "",
        "**Diagnostic, explicitly unblinded result.** The ABCD nonclosure factor is fixed to one and has no independently validated constraint. This is not a discovery or limit result.",
        "",
        f"Edges: `{payload['analysis']['bin_edges']}`.",
        "",
        "| Bin | A | B | C | D | ABCD prediction | CC_nue in A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["inputs"]:
        lines.append(
            f"| [{row['bin_low']:g}, {row['bin_high']:g}) | {row['count_a']} | "
            f"{row['count_b']} | {row['count_c']} | {row['count_d']} | "
            f"{row['abcd_prediction_a']:.6g} +/- {row['abcd_stat_uncertainty']:.6g} | "
            f"{row['signal_a']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Best-fit `mu = {mu['bestfit']:.8g} +/- {mu['uncertainty_hesse']:.8g}`.",
            "",
            f"Goodness-of-fit degrees of freedom: `{payload['model']['degrees_of_freedom']}`; p-value: `{payload['goodness_of_fit_p_value']}`.",
            "",
            f"Parameters at their lower boundary: `{boundary}`.",
            "",
        ]
    )
    return "\n".join(lines)


def report_latex(config: dict, payload: dict, plot_dir: Path) -> str:
    analysis = payload["analysis"]
    mu = next(item for item in payload["parameters"] if item["name"] == "mu")
    gof = payload["goodness_of_fit_p_value"]
    boundary = [item["name"] for item in payload["parameters"] if item["at_lower_bound"]]
    plot = lambda name: (plot_dir / name).resolve().as_posix()
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\usepackage{booktabs,array,graphicx,xcolor}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        rf"\title{{{tex_escape(analysis['title'])}}}",
        r"\author{SND@LHC analysis workflow}",
        r"\date{}",
        r"\maketitle",
        r"\begin{center}\fcolorbox{black}{yellow!12}{\parbox{0.92\textwidth}{\textbf{Diagnostic, explicitly unblinded result.} The nonclosure factor $\kappa$ is fixed to one without an independently validated constraint. This is not a discovery significance or exclusion-limit result.}}\end{center}",
        r"\section{Selection, regions, and model}",
        rf"The selected observable is \texttt{{{tex_escape(config['source']['observable'])}}} with edges $[{', '.join(f'{x:g}' for x in analysis['bin_edges'])}]$. The existing six-bin baseline outputs are not modified.",
        r"The simultaneous likelihood contains the signal region A and control regions B, C, and D. In each bin $i$:",
        r"\[b_{D,i}=L_D\rho_i,\quad b_{B,i}=L_B\rho_i u_i,\quad b_{C,i}=L_C\rho_i v_i,\quad b_{A,i}=L_A\rho_i u_i v_i\kappa_i.\]",
        r"Observed counts follow $n_{R,i}\sim\mathrm{Pois}(b_{R,i}+\mu s_{R,i})$. The same $\mu$ scales standard CC $\nu_e$ MC leakage in all four regions. Here $\kappa_i=1$ is fixed. Other neutrino MC samples are not added separately, avoiding double counting with the data-driven background.",
        r"\section{Input yields}",
        r"\begin{center}\small\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        latex_row("Density interval", "A", "B", "C", "D", r"$A_{pred}$", r"$\sigma_{pred}$", r"CC $\nu_e$ A"),
        r"\midrule",
    ]
    for row in payload["inputs"]:
        lines.append(
            latex_row(
                "{" + f"[{row['bin_low']:g}, {row['bin_high']:g})" + "}",
                row["count_a"], row["count_b"], row["count_c"], row["count_d"],
                f"{row['abcd_prediction_a']:.4g}",
                f"{row['abcd_stat_uncertainty']:.4g}",
                f"{row['signal_a']:.4g}",
            )
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}\end{center}",
            r"The arithmetic diagnostic is $A_{pred}=(BC/D)(L_A L_D)/(L_B L_C)$ and $\sigma_{pred}/A_{pred}=\sqrt{1/B+1/C+1/D}$. The likelihood itself fits A/B/C/D simultaneously instead of treating this propagated value as a Gaussian observation.",
            r"\subsection{CC $\nu_e$ signal leakage used by the likelihood}",
            r"\begin{center}\small\begin{tabular}{rrrrr}",
            r"\toprule",
            latex_row("Density interval", r"$s_A$", r"$s_B$", r"$s_C$", r"$s_D$"),
            r"\midrule",
        ]
    )
    for row in payload["inputs"]:
        lines.append(
            latex_row(
                "{" + f"[{row['bin_low']:g}, {row['bin_high']:g})" + "}",
                f"{row['signal_a']:.4g}",
                f"{row['signal_b']:.4g}",
                f"{row['signal_c']:.4g}",
                f"{row['signal_d']:.4g}",
            )
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}\end{center}",
            r"\section{Fit result}",
            r"\begin{center}\begin{tabular}{lr}\toprule",
            latex_row("Quantity", "Value"),
            r"\midrule",
            latex_row(r"Best-fit $\mu$", f"{mu['bestfit']:.7g}"),
            latex_row(r"Hesse uncertainty on $\mu$", f"{mu['uncertainty_hesse']:.7g}"),
            latex_row(r"Best-fit $-2\log L$", f"{payload['best_twice_nll']:.7g}"),
            latex_row("Goodness-of-fit degrees of freedom", payload["model"]["degrees_of_freedom"]),
            latex_row("Goodness-of-fit p-value", "undefined" if gof is None else f"{gof:.7g}"),
            r"\bottomrule\end{tabular}\end{center}",
            r"\subsection{All fitted parameters}",
            r"\begin{center}\small\begin{tabular}{lrrl}\toprule",
            latex_row("Parameter", "Best fit", "Hesse uncertainty", "At lower boundary"),
            r"\midrule",
        ]
    )
    for item in payload["parameters"]:
        lines.append(
            latex_row(
                tex_escape(item["name"]),
                f"{item['bestfit']:.6g}",
                f"{item['uncertainty_hesse']:.6g}",
                str(item["at_lower_bound"]),
            )
        )
    if payload["model"]["degrees_of_freedom"] == 0:
        interpretation = (
            r"This cut-and-count model is saturated: it has zero goodness-of-fit "
            r"degrees of freedom, so no p-value exists. In addition, "
            + tex_escape(", ".join(boundary) if boundary else "no parameter")
            + r" is at its lower boundary. Its Hesse covariance is therefore only diagnostic."
        )
    elif gof is not None and gof < 0.05:
        interpretation = (
            rf"The goodness-of-fit p-value is {gof:.4g}, below 0.05. The requested "
            r"two-bin factorized model is incompatible with the observations and "
            r"cannot support a final signal inference."
        )
    else:
        interpretation = r"No additional fit-quality failure was identified."
    lines.extend(
        [
            r"\bottomrule\end{tabular}\end{center}",
            interpretation,
            r"\clearpage\section{Signal-region comparisons}",
            r"The x-axis uses the true unequal density-bin widths. The likelihood uses raw event counts.",
        ]
    )
    for title, filename in (
        ("Before fit: control-derived background, $\mu=0$", "before_fit_signal_region.pdf"),
        ("Pre-fit: configured initial model, $\mu=1$", "prefit_signal_region.pdf"),
        ("Post-fit simultaneous model", "postfit_signal_region.pdf"),
    ):
        lines.extend(
            [
                rf"\subsection{{{title}}}",
                r"\begin{center}",
                rf"\includegraphics[width=0.78\linewidth]{{{plot(filename)}}}",
                r"\end{center}",
            ]
        )
    lines.extend(
        [
            r"\clearpage\section{Likelihood and correlation diagnostics}",
            r"\begin{center}",
            rf"\includegraphics[width=0.75\linewidth]{{{plot('likelihood_scan_mu.pdf')}}}",
            r"\end{center}",
            r"\begin{center}",
            rf"\includegraphics[width=0.72\linewidth]{{{plot('correlation_matrix.pdf')}}}",
            r"\end{center}",
            r"\section{Interpretation boundary}",
            r"This report demonstrates the requested high-density model and its numerical behavior. It cannot be promoted to a final physics result until $\kappa$ is constrained by independent closure validation and the signal production choice and systematic uncertainties are frozen.",
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(config: dict, source_rows: dict, name: str, output_root: Path) -> None:
    analysis = dict(config["analyses"][name])
    analysis["name"] = name
    edges = [float(value) for value in analysis["bin_edges"]]
    analysis["bin_edges"] = edges
    inputs = aggregate_inputs(source_rows, edges)
    spec = build_workspace(config, inputs)
    workspace = pyhf.Workspace(spec)
    model, data = cabinetry.model_utils.model_and_data(workspace)
    expected_npars = 1 + 3 * len(inputs)
    if model.config.npars != expected_npars or model.config.nmaindata != 4 * len(inputs):
        raise ValueError("Unexpected simultaneous-model dimensions")
    fit_results = cabinetry.fit.fit(
        model,
        data,
        goodness_of_fit=True,
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    mu_scan = cabinetry.fit.scan(
        model,
        data,
        config["model"]["signal_parameter"],
        par_range=tuple(float(x) for x in config["fit"]["mu_scan_range"]),
        n_steps=int(config["fit"]["mu_scan_points"]),
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    payload = result_payload(config, analysis, model, inputs, fit_results)
    out = output_root / analysis["output_directory"]
    plots = out / config["output"]["plot_directory"]
    save_plots(plots, config, edges, model, data, fit_results, mu_scan)
    atomic_text(out / config["output"]["inputs_csv"], csv_text(inputs))
    atomic_text(out / config["output"]["workspace_json"], json.dumps(spec, indent=2) + "\n")
    atomic_text(out / config["output"]["fit_results_json"], json.dumps(payload, indent=2) + "\n")
    scan_rows = [
        {"mu": float(value), "delta_twice_nll": float(delta)}
        for value, delta in zip(mu_scan.parameter_values, mu_scan.delta_nlls)
    ]
    atomic_text(out / config["output"]["mu_scan_csv"], csv_text(scan_rows))
    atomic_text(out / config["output"]["report_markdown"], report_markdown(config, payload))
    atomic_text(out / config["output"]["report_tex"], report_latex(config, payload, plots))
    print(
        f"completed {name}: mu="
        f"{next(x['bestfit'] for x in payload['parameters'] if x['name'] == 'mu'):.8g}, "
        f"gof={payload['goodness_of_fit_p_value']}"
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    source_rows = load_source_rows(args.input_table, config)
    output_root = Path(args.output_root)
    for name in ("cutcount", "shape"):
        run_analysis(config, source_rows, name, output_root)


if __name__ == "__main__":
    main()
