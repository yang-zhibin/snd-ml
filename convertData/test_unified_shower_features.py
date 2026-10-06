#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "convertData"))
sys.path.insert(0, str(REPO_ROOT / "snakemake/metadata"))

from prepare_unified_shower_feature_manifests import build_manifests  # noqa: E402
from sample_selection import (  # noqa: E402
    IDENTITY_CUTFLOW,
    applies_collision_data_quality,
    uses_external_cutflow,
)
from source_provenance import event_source_values, validate_source_options  # noqa: E402
from validate_unified_shower_feature_file import (  # noqa: E402
    validate_source_scalar_values,
)


class Track:
    def __init__(self, pdg: int, energy: float):
        self.pdg = pdg
        self.energy = energy

    def GetPdgCode(self):
        return self.pdg

    def GetEnergy(self):
        return self.energy


def source_args(**overrides):
    values = {
        "type": "MC_neutrino",
        "source_domain_id": 1,
        "source_pdg_code": None,
        "source_energy_gev": None,
        "source_uid": "unified_src_0123456789abcdef0123",
        "source_geometry_id": "",
        "source_geometry_checksum": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class UnifiedSourceContractTest(unittest.TestCase):
    def test_neutrino_truth_is_unmodified(self):
        args = source_args()
        validate_source_options(args)
        self.assertEqual(event_source_values(args, Track(-14, 125.0)), (125.0, -14))

    def test_unified_domains_require_source_uid_and_matching_sample_type(self):
        with self.assertRaisesRegex(ValueError, "source-uid"):
            validate_source_options(source_args(source_uid=""))
        with self.assertRaisesRegex(ValueError, "requires sample type"):
            validate_source_options(source_args(type="MC_muon"))

    def test_testbeam_requires_paired_values_and_geometry(self):
        args = source_args(
            type="MC_testbeam",
            source_domain_id=4,
            source_pdg_code=11,
            source_energy_gev=None,
        )
        with self.assertRaisesRegex(ValueError, "provided together"):
            validate_source_options(args)
        args.source_energy_gev = 100.0
        with self.assertRaisesRegex(ValueError, "source-geometry-id"):
            validate_source_options(args)

    def test_testbeam_mc_catalogue_validates_truth(self):
        args = source_args(
            type="MC_testbeam",
            source_domain_id=4,
            source_pdg_code=11,
            source_energy_gev=100.0,
            source_geometry_id="W_2walls_28plates",
            source_geometry_checksum="abc",
        )
        validate_source_options(args)
        self.assertEqual(event_source_values(args, Track(11, 101.0)), (101.0, 11))
        with self.assertRaisesRegex(ValueError, "PDG mismatch"):
            event_source_values(args, Track(211, 100.0))
        with self.assertRaisesRegex(ValueError, "energy mismatch"):
            event_source_values(args, Track(11, 110.0))

    def test_real_data_sentinels_and_testbeam_values(self):
        collision = source_args(
            type="real_data", source_domain_id=3, source_uid="unified_src_0123456789abcdef0123"
        )
        validate_source_options(collision)
        self.assertEqual(event_source_values(collision), (-1.0, 0))
        validate_source_scalar_values(3, -1.0, 0)
        with self.assertRaisesRegex(RuntimeError, "sentinels"):
            validate_source_scalar_values(3, 0.0, 0)

        testbeam = source_args(
            type="real_testbeam",
            source_domain_id=5,
            source_pdg_code=-211,
            source_energy_gev=180.0,
            source_geometry_id="W_2walls_28plates",
            source_geometry_checksum="abc",
        )
        validate_source_options(testbeam)
        self.assertEqual(event_source_values(testbeam), (180.0, -211))

    def test_identity_cutflow_is_external_without_collision_quality(self):
        self.assertTrue(uses_external_cutflow("real_testbeam", IDENTITY_CUTFLOW))
        self.assertFalse(
            applies_collision_data_quality("real_testbeam", IDENTITY_CUTFLOW)
        )
        self.assertTrue(
            applies_collision_data_quality("real_data", "external_cutflow")
        )


class UnifiedManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_obeys_frozen_contract(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)

            def write_csv(name, rows):
                path = directory / name
                pd.DataFrame(rows).to_csv(path, index=False)
                return str(path)

            neutrino = write_csv(
                "neutrino.csv",
                [{
                    "data_type": "MC_neutrino",
                    "subfolder": "2024_ve",
                    "partition": 1,
                    "n_event": 48,
                    "digi_path": "/input/nu.root",
                    "geo_path": "/input/nu_geo.root",
                    "output_base_path": "/legacy/nu",
                    "nueAnalysisFilter_path": "cutflow.root",
                    "lumi_per_file": 15.0,
                }],
            )
            muon = write_csv(
                "muon.csv",
                [{
                    "data_type": "MC_muon",
                    "subfolder": "up",
                    "partition": "one",
                    "n_event": 10,
                    "digi_path": "/input/muon.root",
                    "geo_path": "/input/muon_geo.root",
                    "lumi_per_file": 0.5,
                }],
            )
            real = write_csv(
                "real.csv",
                [{
                    "data_type": "real_data",
                    "subfolder": "run_1",
                    "partition": "file_1",
                    "n_event": 20,
                    "digi_path": "/input/data.root",
                    "geo_path": "/input/data_geo.root",
                    "output_base_path": "/legacy/data",
                    "nueAnalysisFilter_path": "cutflow.root",
                    "lumi_per_file": 0.01,
                }],
            )
            catalog_rows = []
            for index, (domain, pdg, particle) in enumerate(
                (("mc_2024", 11, "electron"), ("data_2024", -211, "pion"))
            ):
                catalog_rows.append({
                    "data_type": "MC_data" if domain == "mc_2024" else "real_data",
                    "subfolder": "100GeV" if domain == "mc_2024" else "run_1",
                    "partition": f"source_{index}",
                    "n_events": 5,
                    "digi_path": f"/input/tb_{index}.root",
                    "geo_path": f"/input/tb_geo_{index}.root",
                    "metadata_csv": f"/metadata/tb_{index}.csv",
                    "metadata_row": index,
                    "domain": domain,
                    "geometry_setup_id": "W_2walls_28plates",
                    "geometry_checksum": f"checksum_{index}",
                    "beam_energy_gev": 100.0 if domain == "mc_2024" else 180.0,
                    "class_name": particle,
                    "pdg_code": pdg,
                    "source_uid": f"tbfull_src_{index:020x}",
                    "cutflow_path": f"/input/tb_cutflow_{index}.root",
                })
            catalog = write_csv("catalog.csv", catalog_rows)
            catalog_hash = hashlib.sha256(Path(catalog).read_bytes()).hexdigest()

            family_settings = {
                "mc_neutrino": {
                    "source_metadata": [neutrino],
                    "sample_type": "MC_neutrino",
                    "selection_mode": "external_cutflow",
                    "source_domain_id": 1,
                    "canonical_labels": ["CC_nue", "CC_numu", "NC_nue", "NC_numu"],
                    "exposure_kind": "simulated_luminosity_fb",
                },
                "mc_muon_up": {
                    "source_metadata": muon,
                    "sample_type": "MC_muon",
                    "selection_mode": "direct_input",
                    "source_domain_id": 2,
                    "canonical_labels": ["MC_muon_up"],
                    "exposure_kind": "unvalidated_equivalent_luminosity",
                },
                "real_data_2024_skim": {
                    "source_metadata": real,
                    "sample_type": "real_data",
                    "selection_mode": "external_cutflow",
                    "source_domain_id": 3,
                    "canonical_labels": ["real_data"],
                    "exposure_kind": "recorded_luminosity_fb",
                },
                "testbeam_mc_2024": {
                    "source_metadata": catalog,
                    "source_metadata_sha256": catalog_hash,
                    "catalog_domain": "mc_2024",
                    "sample_type": "MC_testbeam",
                    "selection_mode": "identity_cutflow",
                    "source_domain_id": 4,
                    "canonical_labels": ["testbeam_mc_electron", "testbeam_mc_pion"],
                    "exposure_kind": "event_count_only",
                },
                "testbeam_data_2024": {
                    "source_metadata": catalog,
                    "source_metadata_sha256": catalog_hash,
                    "catalog_domain": "data_2024",
                    "sample_type": "real_testbeam",
                    "selection_mode": "identity_cutflow",
                    "source_domain_id": 5,
                    "canonical_labels": ["testbeam_data_electron", "testbeam_data_pion"],
                    "exposure_kind": "event_count_only",
                },
            }
            for family, settings in family_settings.items():
                settings.update({
                    "expected_source_files": 1,
                    "input_tree": "cbmsim",
                    "source_domain": family,
                })
            config = {
                "version": "unified_shower_features_v1",
                "manifest_root": str(directory / "manifests"),
                "feature_roots": {
                    family: f"/features/{family}" for family in family_settings
                },
                "families": family_settings,
            }
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            first = build_manifests(config_path)
            second = build_manifests(config_path)
            for family in first:
                pd.testing.assert_frame_equal(first[family], second[family])
                frame = first[family]
                self.assertEqual(len(frame), 1)
                self.assertEqual(int(frame.loc[0, "is_canary"]), 1)
                self.assertEqual(frame.loc[0, "source_domain"], family)
                self.assertTrue(
                    frame.loc[0, "feature_scifi_cluster_v1_path"].startswith(
                        f"/features/{family}/"
                    )
                )
                self.assertTrue(
                    frame.loc[0, "validation_record"].endswith(".validated.json")
                )
            self.assertEqual(
                first["testbeam_mc_2024"].loc[0, "canonical_label"],
                "testbeam_mc_electron",
            )
            self.assertEqual(
                first["testbeam_data_2024"].loc[0, "canonical_label"],
                "testbeam_data_pion",
            )

    def test_rule_contract_uses_json_validation_records(self):
        rules = (
            REPO_ROOT / "snakemake/rules/unified_shower_features.smk"
        ).read_text(encoding="utf-8")
        self.assertIn("validate_unified_shower_feature_file.py", rules)
        self.assertIn(".root.validated.json", rules)
        self.assertIn("--validation-record", rules)
        self.assertNotIn('touch "$tmp_marker"', rules)
        for family in (
            "mc_neutrino",
            "mc_muon_up",
            "real_data_2024_skim",
            "testbeam_mc_2024",
            "testbeam_data_2024",
        ):
            self.assertIn(f"unified_features_{family}_canary_v1", rules)
            self.assertIn(f"unified_features_{family}_v1", rules)


if __name__ == "__main__":
    unittest.main()
