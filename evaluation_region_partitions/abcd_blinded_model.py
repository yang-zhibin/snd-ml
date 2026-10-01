#!/usr/bin/env python3
"""Shared one-bin blinded/unblinded ABCD HistFactory model utilities."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import cabinetry
import numpy as np
import pyhf

from make_abcd_blinded_inputs import (
    FORBIDDEN_SIGNAL_OBSERVATION_FIELDS,
    assert_blinded_schema,
)


ROLES = ("A", "B", "C", "D")
CONTROL_ROLES = ("B", "C", "D")


def load_blinded_input(path: str | Path, config: dict) -> dict:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = {str(name).lower() for name in (reader.fieldnames or [])}
        forbidden = FORBIDDEN_SIGNAL_OBSERVATION_FIELDS & fieldnames
        if forbidden:
            raise ValueError(
                "Blinded artifact contains forbidden signal-observation fields: "
                + ", ".join(sorted(forbidden))
            )
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("Strict one-bin input must contain exactly one row")
    row = rows[0]
    assert_blinded_schema(row)
    if row["input_version"] != config["version"]:
        raise ValueError("Blinded input/config version mismatch")
    low, high = [float(value) for value in config["source"]["counting_range"]]
    if float(row["range_low"]) != low or float(row["range_high"]) != high:
        raise ValueError("Blinded input has the wrong counting range")
    for role in ROLES:
        key = role.lower()
        if row[f"region_{key}"] != config["source"]["roles"][role]:
            raise ValueError(f"Region mapping mismatch for {role}")
    return row


def value(row: dict, key: str) -> float:
    result = float(row[key])
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid nonnegative value for {key}: {row[key]}")
    return result


def residual_initials(row: dict, config: dict) -> tuple[float, float, float]:
    residual = {}
    for role in CONTROL_ROLES:
        key = role.lower()
        residual[role] = value(row, f"count_{key}") - sum(
            value(row, f"{sample.lower()}_{key}")
            for sample in config["source"]["background_samples"]
        )
        if residual[role] <= 0.0:
            raise ValueError(f"Non-positive residual background initialization in {role}")
    rho = residual["D"] / value(row, "exposure_d")
    u = residual["B"] * value(row, "exposure_d") / (
        residual["D"] * value(row, "exposure_b")
    )
    v = residual["C"] * value(row, "exposure_d") / (
        residual["D"] * value(row, "exposure_c")
    )
    return rho, u, v


def scalar_parameter(name: str, initial: float) -> dict:
    return {
        "name": name,
        "inits": [initial],
        "bounds": [[0.0, max(20.0 * initial, 20.0)]],
    }


def build_workspace(
    config: dict,
    row: dict,
    roles: tuple[str, ...],
    observations: dict[str, list[float]],
) -> dict:
    """Build one bin per channel; there is no density-bin parameter index."""
    rho, u, v = residual_initials(row, config)
    modifiers = {
        "A": [
            {"name": "rho", "type": "normfactor", "data": None},
            {"name": "u", "type": "normfactor", "data": None},
            {"name": "v", "type": "normfactor", "data": None},
            {
                "name": "kappa_validation",
                "type": "normsys",
                "data": {
                    "lo": value(row, "kappa_constraint_lo"),
                    "hi": value(row, "kappa_constraint_hi"),
                },
            },
        ],
        "B": [
            {"name": "rho", "type": "normfactor", "data": None},
            {"name": "u", "type": "normfactor", "data": None},
        ],
        "C": [
            {"name": "rho", "type": "normfactor", "data": None},
            {"name": "v", "type": "normfactor", "data": None},
        ],
        "D": [{"name": "rho", "type": "normfactor", "data": None}],
    }
    channels = []
    include_mc_stat = bool(config["model"]["include_mc_statistical_uncertainty"])
    for role in roles:
        key = role.lower()
        scale = value(row, "kappa_central") if role == "A" else 1.0
        samples = [
            {
                "name": "abcd_residual_background",
                "data": [value(row, f"exposure_{key}") * scale],
                "modifiers": modifiers[role],
            }
        ]
        for sample in config["source"]["background_samples"]:
            prefix = sample.lower()
            relative = float(config["model"]["background_mc_normalization"][sample])
            sample_modifiers = [
                {
                    "name": f"norm_{prefix}",
                    "type": "normsys",
                    "data": {
                        "lo": max(0.01, 1.0 - relative),
                        "hi": 1.0 + relative,
                    },
                }
            ]
            if include_mc_stat:
                sample_modifiers.append(
                    {
                        "name": f"mcstat_{prefix}_{key}",
                        "type": "staterror",
                        "data": [math.sqrt(value(row, f"{prefix}_variance_{key}"))],
                    }
                )
            samples.append(
                {
                    "name": prefix,
                    "data": [value(row, f"{prefix}_{key}")],
                    "modifiers": sample_modifiers,
                }
            )
        signal_modifiers = [
            {
                "name": config["model"]["signal_parameter"],
                "type": "normfactor",
                "data": None,
            }
        ]
        if include_mc_stat:
            signal_modifiers.append(
                {
                    "name": f"mcstat_cc_nue_{key}",
                    "type": "staterror",
                    "data": [math.sqrt(value(row, f"cc_nue_variance_{key}"))],
                }
            )
        samples.append(
            {
                "name": "cc_nue_signal",
                "data": [value(row, f"cc_nue_{key}")],
                "modifiers": signal_modifiers,
            }
        )
        channels.append({"name": f"{role}_region", "samples": samples})

    mu_name = config["model"]["signal_parameter"]
    return {
        "channels": channels,
        "observations": [
            {"name": f"{role}_region", "data": observations[role]}
            for role in roles
        ],
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
                        scalar_parameter("rho", rho),
                        scalar_parameter("u", u),
                        scalar_parameter("v", v),
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def control_observations(row: dict) -> dict[str, list[float]]:
    return {
        role: [value(row, f"count_{role.lower()}")]
        for role in CONTROL_ROLES
    }


def fit_control(config: dict, row: dict):
    spec = build_workspace(config, row, CONTROL_ROLES, control_observations(row))
    workspace = pyhf.Workspace(spec)
    model, data = cabinetry.model_utils.model_and_data(workspace)
    initial = model.config.suggested_init()
    initial[model.config.poi_index] = 0.0
    fixed = model.config.suggested_fixed()
    fixed[model.config.poi_index] = True
    results = cabinetry.fit.fit(
        model,
        data,
        init_pars=initial,
        fix_pars=fixed,
        maxiter=int(config["fit"]["maxiter"]),
        tolerance=float(config["fit"]["tolerance"]),
    )
    return spec, model, data, results


def transfer_parameters(source_model, source_values, target_model, mu: float) -> list[float]:
    target = np.asarray(target_model.config.suggested_init(), dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    for name in source_model.config.par_order:
        if name in target_model.config.par_order:
            target[target_model.config.par_slice(name)] = source_values[
                source_model.config.par_slice(name)
            ]
    target[target_model.config.poi_index] = mu
    return target.tolist()


def replace_observations(spec: dict, model, parameters: list[float]) -> dict:
    output = copy.deepcopy(spec)
    expected = np.asarray(
        model.expected_data(parameters, include_auxdata=False), dtype=float
    )
    for observation in output["observations"]:
        observation["data"] = expected[
            model.config.channel_slices[observation["name"]]
        ].tolist()
    return output


def full_template(config: dict, row: dict) -> tuple[dict, pyhf.pdf.Model]:
    placeholder = {role: [0.0] for role in ROLES}
    spec = build_workspace(config, row, ROLES, placeholder)
    return spec, pyhf.Workspace(spec).model()


def covariance_matrix(fit_results) -> np.ndarray:
    uncertainty = np.asarray(fit_results.uncertainty, dtype=float)
    correlation = np.asarray(fit_results.corr_mat, dtype=float)
    return correlation * np.outer(uncertainty, uncertainty)


def scalar_labels(model) -> list[str]:
    labels = []
    for name in model.config.par_order:
        parameter_set = model.config.param_set(name)
        if parameter_set.n_parameters == 1:
            labels.append(name)
        else:
            labels.extend(f"{name}[{index}]" for index in range(parameter_set.n_parameters))
    return labels


def mapped_covariance(control_model, control_fit, full_model) -> np.ndarray:
    control_covariance = covariance_matrix(control_fit)
    result = np.zeros((full_model.config.npars, full_model.config.npars), dtype=float)
    for left in control_model.config.par_order:
        if left not in full_model.config.par_order:
            continue
        left_control = control_model.config.par_slice(left)
        left_full = full_model.config.par_slice(left)
        for right in control_model.config.par_order:
            if right not in full_model.config.par_order:
                continue
            right_control = control_model.config.par_slice(right)
            right_full = full_model.config.par_slice(right)
            result[left_full, right_full] = control_covariance[left_control, right_control]
    for name in full_model.config.par_order:
        if name in control_model.config.par_order or name == full_model.config.poi_name:
            continue
        parameter_slice = full_model.config.par_slice(name)
        result[parameter_slice, parameter_slice] = np.eye(
            parameter_slice.stop - parameter_slice.start
        )
    return result


def a_background_prediction(
    model,
    parameters: list[float],
    covariance: np.ndarray,
) -> tuple[float, float]:
    parameters = np.asarray(parameters, dtype=float)
    poi_index = model.config.poi_index
    parameters[poi_index] = 0.0

    def prediction(values: np.ndarray) -> float:
        expected = np.asarray(
            model.expected_data(values.tolist(), include_auxdata=False), dtype=float
        )
        return float(expected[model.config.channel_slices["A_region"]][0])

    central = prediction(parameters)
    gradient = np.zeros(model.config.npars, dtype=float)
    bounds = model.config.suggested_bounds()
    for index in range(model.config.npars):
        if index == poi_index:
            continue
        step = 1.0e-4 * max(1.0, abs(parameters[index]))
        low, high = bounds[index]
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] = min(high, parameters[index] + step)
        minus[index] = max(low, parameters[index] - step)
        denominator = plus[index] - minus[index]
        if denominator > 0.0:
            gradient[index] = (prediction(plus) - prediction(minus)) / denominator
    variance = float(gradient @ covariance @ gradient)
    return central, math.sqrt(max(0.0, variance))


def sample_yields(model, parameters: list[float], role: str) -> dict[str, float]:
    by_sample = np.asarray(
        model.main_model.expected_data(parameters, return_by_sample=True), dtype=float
    )
    channel_slice = model.config.channel_slices[f"{role}_region"]
    return {
        sample: float(by_sample[index, channel_slice][0])
        for index, sample in enumerate(model.config.samples)
    }


def profile_interval(parameter_values, delta_nlls, level: float = 0.68) -> tuple[float | None, float | None]:
    # One-parameter Wilks interval: Delta(-2 log L)=1 at 68.27%.
    threshold = 1.0 if math.isclose(level, 0.68, abs_tol=0.02) else 1.0
    x = np.asarray(parameter_values, dtype=float)
    y = np.asarray(delta_nlls, dtype=float)
    minimum = int(np.nanargmin(y))

    def crossing(indices) -> float | None:
        for left, right in indices:
            y_left, y_right = y[left] - threshold, y[right] - threshold
            if y_left == 0.0:
                return float(x[left])
            if y_left * y_right <= 0.0 and y_right != y_left:
                fraction = -y_left / (y_right - y_left)
                return float(x[left] + fraction * (x[right] - x[left]))
        return None

    lower = crossing((index, index + 1) for index in range(minimum - 1, -1, -1))
    upper = crossing((index, index + 1) for index in range(minimum, len(x) - 1))
    if lower is None and y[0] <= threshold:
        lower = float(x[0])
    if upper is None and y[-1] <= threshold:
        upper = None
    return lower, upper


def q0_value(data, model) -> tuple[float, np.ndarray, np.ndarray]:
    result = pyhf.infer.test_statistics.q0(
        0.0,
        data,
        model,
        model.config.suggested_init(),
        model.config.suggested_bounds(),
        model.config.suggested_fixed(),
        return_fitted_pars=True,
    )
    q0, fitted = result
    constrained, unconstrained = fitted
    return (
        float(np.asarray(q0).reshape(-1)[0]),
        np.asarray(constrained, dtype=float),
        np.asarray(unconstrained, dtype=float),
    )
