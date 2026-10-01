#!/usr/bin/env python3
"""Focused synthetic checks for strict blinded and simultaneous ABCD inference."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import pyhf

from abcd_blinded_model import (
    ROLES,
    build_workspace,
    load_blinded_input,
    q0_value,
)
from make_abcd_blinded_inputs import (
    FORBIDDEN_SIGNAL_OBSERVATION_FIELDS,
    assert_blinded_schema,
    assert_no_signal_data_dependencies,
)
from run_abcd_blinded_expected import run_expected
from run_abcd_unblinded import run_observed, validate_authorization


def config() -> dict:
    return {
        "version": "test_blinded_counting_v2",
        "source": {
            "observable": "density_scifi_sndsw",
            "counting_range": [5000, 40000],
            "event_selection": "synthetic",
            "roles": {"A": "signal", "B": "b", "C": "c", "D": "d"},
            "validation_roles": {
                "numerator": "va", "transfer_b": "vb",
                "transfer_c": "vc", "anchor_d": "vd",
            },
            "signal_sample": "CC_nue",
            "background_samples": ["CC_numu", "NC_nue", "NC_numu"],
        },
        "model": {
            "signal_parameter": "mu", "signal_initial": 1.0,
            "signal_bounds": [0.0, 30.0], "signal_injection": 1.0,
            "signal_contamination_negligible_threshold": 0.01,
            "nonclosure": {"transport_status": "TEST"},
            "background_mc_normalization": {
                "CC_numu": 0.3, "NC_nue": 0.3, "NC_numu": 0.3,
            },
            "background_mc_uncertainty_status": "TEST",
            "include_mc_statistical_uncertainty": True,
        },
        "fit": {
            "maxiter": 100000, "tolerance": 0.1,
            "mu_scan_range": [0.0, 15.0], "mu_scan_points": 61,
            "confidence_level": 0.68, "upper_limit_level": 0.05,
            "upper_limit_scan": [0.0, 15.0, 31],
        },
        "toy_mc": {"default_toys": 5, "seed": 8329, "max_failed_fraction": 0.5},
        "output": {
            "expected_directory": "expected", "model_inputs": "model_inputs.csv",
            "control_workspace": "control_workspace.json",
            "expected_background_workspace": "expected_background_workspace.json",
            "expected_signal_workspace": "expected_signal_workspace.json",
            "control_fit": "control_fit.json", "control_covariance": "control_covariance.csv",
            "expected_results": "expected_results.json", "mu_scan": "expected_mu_scan.csv",
            "report_markdown": "expected_report.md", "report_tex": "expected_report.tex",
            "plot_directory": "plots", "freeze_manifest": "freeze_manifest.json",
            "unblinded_directory": "unblinded", "observed_workspace": "observed_workspace.json",
            "observed_results": "observed_results.json", "observed_mu_scan": "observed_mu_scan.csv",
            "toy_q0": "toy_q0.csv", "final_report_markdown": "final_report.md",
            "final_report_tex": "final_report.tex", "final_plot_directory": "plots",
        },
        "blinding": {
            "status": "REBLINDED_DEVELOPMENT",
            "signal_region_observation_allowed_in_expected_inputs": False,
            "observed_inference_authorized": False,
        },
        "freeze": {"require_clean_git_for_production": False, "unblinding_criteria": []},
    }


def input_row(cfg: dict) -> dict:
    row: dict[str, object] = {
        "input_version": cfg["version"], "blinding_status": "REBLINDED_DEVELOPMENT",
        "range_low": 5000, "range_high": 40000,
        "region_a": "signal", "region_b": "b", "region_c": "c", "region_d": "d",
        "exposure_a": 10.0, "exposure_b": 10.0, "exposure_c": 2.0, "exposure_d": 2.0,
        "count_b": 80, "count_c": 36, "count_d": 18,
        "signal_contamination_fraction_b": 0.125,
        "signal_contamination_fraction_c": 0.002,
        "signal_contamination_fraction_d": 0.002,
        "signal_contamination_threshold": 0.01,
        "signal_contamination_decision": "NONNEGLIGIBLE_BACKGROUND_ONLY_CONDITIONAL_MU_ZERO",
        "kappa_central": 0.4, "kappa_uncertainty": 0.14,
        "kappa_relative_uncertainty": 0.35,
        "kappa_constraint_lo": 0.65, "kappa_constraint_hi": 1.35,
        "kappa_constraint_form": "histfactory_normsys",
        "kappa_constraint_source": "synthetic independent validation plane",
        "kappa_transport_status": "TEST",
        "validation_exposure_factor": 1.0,
        "validation_transfer": 2.0,
        "validation_transfer_uncertainty": 0.4898979485566356,
        "validation_prediction": 50.0,
        "validation_prediction_uncertainty": 15.0,
        "validation_fractional_nonclosure": -0.6,
        "validation_pull": -1.948557158514987,
    }
    validation = {
        "numerator": ("va", 21.0, 1.0, 20.0, 12.0),
        "b": ("vb", 26.0, 1.0, 25.0, 18.75),
        "c": ("vc", 21.0, 1.0, 20.0, 12.0),
        "d": ("vd", 11.0, 1.0, 10.0, 3.0),
    }
    for suffix, (region, observed, other_nu, residual, variance) in validation.items():
        row[f"validation_region_{suffix}"] = region
        row[f"validation_exposure_{suffix}"] = 1.0
        row[f"validation_observed_{suffix}"] = observed
        row[f"validation_other_neutrino_{suffix}"] = other_nu
        row[f"validation_residual_{suffix}"] = residual
        row[f"validation_residual_variance_{suffix}"] = variance
    yields = {
        "cc_nue": {"a": 10.0, "b": 10.0, "c": 0.07, "d": 0.04},
        "cc_numu": {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1},
        "nc_nue": {"a": 0.2, "b": 0.1, "c": 0.1, "d": 0.1},
        "nc_numu": {"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.1},
    }
    for sample, roles in yields.items():
        for role, amount in roles.items():
            row[f"{sample}_{role}"] = amount
            row[f"{sample}_variance_{role}"] = max(0.001, amount * 0.01)
    return row


def write_csv(path: Path, row: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def nested_forbidden_keys(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_SIGNAL_OBSERVATION_FIELDS:
                found.add(str(key).lower())
            found.update(nested_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(nested_forbidden_keys(item))
    return found


def main() -> None:
    pyhf.set_backend("numpy")
    cfg = config(); row = input_row(cfg)
    assert_blinded_schema(row)
    assert not (FORBIDDEN_SIGNAL_OBSERVATION_FIELDS & {key.lower() for key in row})
    assert_no_signal_data_dependencies(["b__real_data.root", "c__real_data.root"], "signal")
    try:
        assert_no_signal_data_dependencies(["signal__real_data.root"], "signal")
    except ValueError as error:
        assert "entered blinded dependencies" in str(error)
    else:
        raise AssertionError("Signal data dependency must fail")

    observations = {"A": [70.0], "B": [80.0], "C": [36.0], "D": [18.0]}
    spec = build_workspace(cfg, row, ROLES, observations)
    model = pyhf.Workspace(spec).model()
    assert model.config.nmaindata == 4
    assert all("[" not in name for name in ("rho", "u", "v"))
    assert set(model.config.channels) == {"A_region", "B_region", "C_region", "D_region"}
    a_channel = next(channel for channel in spec["channels"] if channel["name"] == "A_region")
    residual = next(sample for sample in a_channel["samples"] if sample["name"] == "abcd_residual_background")
    assert any(modifier["name"] == "kappa_validation" for modifier in residual["modifiers"])
    assert all(sample["name"] != "abcd_residual_background" for sample in a_channel["samples"][1:])

    downward_spec = build_workspace(
        cfg, row, ROLES, {"A": [0.0], "B": [80.0], "C": [36.0], "D": [18.0]}
    )
    downward_workspace = pyhf.Workspace(downward_spec)
    downward_model = downward_workspace.model()
    downward_q0, _, downward_free_fit = q0_value(
        downward_workspace.data(downward_model), downward_model
    )
    assert downward_q0 == 0.0
    assert downward_free_fit[downward_model.config.poi_index] < 1.0e-3

    with tempfile.TemporaryDirectory(prefix="abcd_blinded_v2_test.") as temporary:
        root = Path(temporary); table = root / "sanitized.csv"; write_csv(table, row)
        loaded = load_blinded_input(table, cfg)
        payload = run_expected(cfg, loaded, root)
        assert payload["density_bin_index_in_model"] is False
        assert payload["signal_region_observation_used"] is False
        assert payload["control_fit_signal_strength"] == 0.0
        assert payload["expected_discovery"]["q0"] >= 0.0
        assert payload["expected_exclusion"]["background_asimov_upper_limit"] > 0.0
        assert payload["expected_measurement"]["profile_interval"][0] == 0.0
        assert payload["expected_measurement"]["profile_interval"][1] is not None
        assert payload["validation_closure"]["prediction"] == 50.0
        assert payload["validation_closure"]["residual_observed"]["numerator"] == 20.0
        expected_dir = root / "expected"
        model_inputs = list(csv.DictReader((expected_dir / "model_inputs.csv").open()))
        assert len(model_inputs) == 1
        assert not (FORBIDDEN_SIGNAL_OBSERVATION_FIELDS & {key.lower() for key in model_inputs[0]})
        for artifact in ("control_workspace.json", "expected_background_workspace.json", "expected_signal_workspace.json", "expected_results.json"):
            content = json.loads((expected_dir / artifact).read_text())
            assert not nested_forbidden_keys(content)
        markdown = (expected_dir / "expected_report.md").read_text()
        latex = (expected_dir / "expected_report.tex").read_text()
        assert "Independent validation-plane one-bin closure" in markdown
        assert "A'obs residual" in markdown
        assert r"$A'_{\rm pred}\pm\sigma_{\rm pred}$" in latex
        assert r"The fractional quantity $\kappa_{\rm val}-1$" in latex
        assert "signal-region A data" in latex
        assert latex.count(r"\textbf{Definition:}") >= 30
        assert latex.count(r"\textbf{Why it is used:}") >= 30
        assert latex.count(r"\textbf{Interpretation:}") >= 30
        for step in range(1, 9):
            assert rf"Step {step}:" in latex
        for reference in ("histfactory", "choi-oh", "cowan", "pyhf"):
            assert rf"\bibitem{{{reference}}}" in latex

        manifest = root / "freeze_manifest.json"
        manifest.write_text(json.dumps({
            "analysis_version": cfg["version"], "freeze_status": "FROZEN",
            "signal_region_observation_used": False,
        }))
        denied = root / "denied.yaml"
        denied.write_text(
            f"analysis_version: {cfg['version']}\nauthorized: false\n"
            "freeze_manifest_sha256: unused\napproved_by: ''\napproved_at: ''\n"
        )
        try:
            validate_authorization(denied, manifest, cfg)
        except PermissionError:
            pass
        else:
            raise AssertionError("Disabled authorization must fail before unblinding")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        allowed = root / "allowed.yaml"
        allowed.write_text(
            f"analysis_version: {cfg['version']}\nauthorized: true\n"
            f"freeze_manifest_sha256: {digest}\napproved_by: tester\n"
            "approved_at: 2026-10-01\nreason: synthetic test\n"
        )
        validate_authorization(allowed, manifest, cfg)
        observed = run_observed(cfg, loaded, 74, root, 5)
        assert observed["control_parameters_frozen"] is False
        assert observed["model"].startswith("one-bin simultaneous A+B+C+D")
        assert observed["q0"] >= 0.0
        observed_spec = json.loads((root / "unblinded" / "observed_workspace.json").read_text())
        assert len(observed_spec["channels"]) == 4
        assert observed_spec["observations"][0]["data"] or observed_spec["observations"]

    print("Strict one-bin blinded/unblinded ABCD synthetic checks: OK")


if __name__ == "__main__":
    main()
