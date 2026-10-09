"""Dataset-neutral source provenance policy for unified feature products."""

from __future__ import annotations

import math


UNIFIED_SCHEMA_VERSION = "unified_shower_features_v1"
SOURCE_DOMAINS = {
    0: "unknown",
    1: "mc_neutrino",
    2: "mc_muon_up",
    3: "real_data_2024_skim",
    4: "testbeam_mc_2024",
    5: "testbeam_data_2024",
    6: "testbeam_data_2023_h8",
}
MC_DOMAINS = {1, 2, 4}
REAL_DOMAINS = {3, 5, 6}
TESTBEAM_DOMAINS = {4, 5, 6}
UNAVAILABLE_ENERGY_GEV = -1.0
UNAVAILABLE_PDG_CODE = 0
SAMPLE_TYPE_BY_DOMAIN = {
    1: "mc_neutrino",
    2: "mc_muon",
    3: "real_data",
    4: "mc_testbeam",
    5: "real_testbeam",
    6: "real_testbeam",
}


def validate_source_options(args) -> None:
    domain_id = int(getattr(args, "source_domain_id", 0))
    if domain_id not in SOURCE_DOMAINS:
        raise ValueError(
            f"Unknown source domain ID {domain_id}; expected one of "
            f"{sorted(SOURCE_DOMAINS)}"
        )

    pdg = getattr(args, "source_pdg_code", None)
    energy = getattr(args, "source_energy_gev", None)
    if (pdg is None) != (energy is None):
        raise ValueError(
            "--source-pdg-code and --source-energy-gev must be provided together"
        )
    if energy is not None and (not math.isfinite(float(energy)) or float(energy) < 0):
        raise ValueError("--source-energy-gev must be finite and non-negative")

    sample_type = str(args.type).strip().lower()
    is_mc = sample_type.startswith("mc_")
    expected_sample_type = SAMPLE_TYPE_BY_DOMAIN.get(domain_id)
    if expected_sample_type and sample_type != expected_sample_type:
        raise ValueError(
            f"Source domain {SOURCE_DOMAINS[domain_id]} requires sample type "
            f"{expected_sample_type}, got {sample_type}"
        )
    if domain_id in MC_DOMAINS and not is_mc:
        raise ValueError(f"Source domain {SOURCE_DOMAINS[domain_id]} requires MC input")
    if domain_id in REAL_DOMAINS and is_mc:
        raise ValueError(f"Source domain {SOURCE_DOMAINS[domain_id]} requires real-data input")
    if domain_id in TESTBEAM_DOMAINS and pdg is None:
        raise ValueError("Testbeam domains require paired source PDG and energy values")
    if domain_id not in TESTBEAM_DOMAINS and pdg is not None:
        raise ValueError("Paired source PDG and energy values are testbeam-only")

    if domain_id and not str(getattr(args, "source_uid", "") or "").strip():
        raise ValueError("Unified source domains require --source-uid")

    if domain_id in TESTBEAM_DOMAINS:
        for name in ("source_geometry_id", "source_geometry_checksum"):
            if not str(getattr(args, name, "") or "").strip():
                raise ValueError(f"Testbeam domains require --{name.replace('_', '-')}")


def event_source_values(args, track0=None) -> tuple[float, int]:
    """Return source energy/PDG, validating testbeam MC catalogue values."""
    domain_id = int(getattr(args, "source_domain_id", 0))
    is_mc = str(args.type).strip().lower().startswith("mc_")
    supplied_pdg = getattr(args, "source_pdg_code", None)
    supplied_energy = getattr(args, "source_energy_gev", None)

    if is_mc and track0 is not None:
        truth_pdg = int(track0.GetPdgCode())
        truth_energy = float(track0.GetEnergy())
        if domain_id == 4:
            if truth_pdg != int(supplied_pdg):
                raise ValueError(
                    f"Testbeam MC PDG mismatch: truth={truth_pdg}, catalogue={supplied_pdg}"
                )
            if not math.isclose(
                truth_energy, float(supplied_energy), rel_tol=0.02, abs_tol=0.5
            ):
                raise ValueError(
                    "Testbeam MC energy mismatch: "
                    f"truth={truth_energy}, catalogue={supplied_energy}"
                )
        return truth_energy, truth_pdg

    if is_mc and domain_id in MC_DOMAINS:
        raise ValueError(
            f"Source domain {SOURCE_DOMAINS[domain_id]} requires incoming MCTrack[0] truth"
        )
    if domain_id in {5, 6}:
        return float(supplied_energy), int(supplied_pdg)
    return UNAVAILABLE_ENERGY_GEV, UNAVAILABLE_PDG_CODE
