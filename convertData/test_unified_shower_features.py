#!/usr/bin/env python3

from __future__ import annotations

import array
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import ROOT
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "convertData"))
sys.path.insert(0, str(REPO_ROOT / "snakemake/metadata"))

from prepare_unified_shower_feature_manifests import build_manifests  # noqa: E402
from merge_unified_shower_feature_shards import (  # noqa: E402
    merge as merge_feature_shards,
    sha256 as feature_sha256,
)
from sample_selection import (  # noqa: E402
    IDENTITY_CUTFLOW,
    applies_collision_data_quality,
    cutflow_selection_expression,
    uses_external_cutflow,
)
from source_provenance import event_source_values, validate_source_options  # noqa: E402
from validate_unified_shower_feature_file import (  # noqa: E402
    inspect_source_selection,
    validate_merge_boundaries,
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

    def test_frozen_cutflow_selection_expression(self):
        self.assertEqual(
            cutflow_selection_expression("MC_neutrino", "external_cutflow"),
            "SciFiMinHits == 1",
        )
        self.assertIsNone(
            cutflow_selection_expression("real_testbeam", "identity_cutflow")
        )
        self.assertIsNone(
            cutflow_selection_expression("MC_muon", "direct_input")
        )


class UnifiedSourceSelectionValidationTest(unittest.TestCase):
    @staticmethod
    def write_source_files(directory: Path) -> tuple[str, str]:
        digi_path = directory / "digi.root"
        digi = ROOT.TFile(str(digi_path), "RECREATE")
        digi_tree = ROOT.TTree("cbmsim", "synthetic digi")
        value = array.array("i", [0])
        digi_tree.Branch("value", value, "value/I")
        for index in range(3):
            value[0] = index
            digi_tree.Fill()
        digi_tree.Write()
        digi.Close()

        cutflow_path = directory / "cutflow.root"
        cutflow = ROOT.TFile(str(cutflow_path), "RECREATE")
        cutflow_tree = ROOT.TTree("cutFlowSummary", "synthetic cutflow")
        raw_entry = array.array("q", [0])
        scifi_minhits = array.array("i", [0])
        cutflow_tree.Branch("entry", raw_entry, "entry/L")
        cutflow_tree.Branch("SciFiMinHits", scifi_minhits, "SciFiMinHits/I")
        for index, passed in enumerate((1, 0, 1)):
            raw_entry[0] = index
            scifi_minhits[0] = passed
            cutflow_tree.Fill()
        cutflow_tree.Write()
        cutflow.Close()
        return str(digi_path), str(cutflow_path)

    def test_selected_count_and_event_indices_are_derived_from_cutflow(self):
        with tempfile.TemporaryDirectory() as directory_name:
            digi, cutflow = self.write_source_files(Path(directory_name))
            args = SimpleNamespace(
                input_digi_path=digi,
                input_cutflow_path=cutflow,
                input_tree="cbmsim",
                expected_input_event_count=3,
                sample_type="MC_neutrino",
                selection_mode="external_cutflow",
                expected_max_events=0,
                expected_selected_entry_start=None,
                expected_selected_entry_stop=None,
            )
            contract = inspect_source_selection(args)
            self.assertEqual(contract["input_event_count"], 3)
            self.assertEqual(contract["available_selected_event_count"], 2)
            self.assertEqual(contract["expected_written_event_count"], 2)
            self.assertEqual(contract["expected_raw_entries"], [0, 2])

            args.expected_max_events = 1
            bounded = inspect_source_selection(args)
            self.assertEqual(bounded["available_selected_event_count"], 2)
            self.assertEqual(bounded["expected_written_event_count"], 1)
            self.assertEqual(bounded["expected_raw_entries"], [0])

            args.expected_max_events = 0
            args.expected_selected_entry_start = 1
            args.expected_selected_entry_stop = 2
            shard = inspect_source_selection(args)
            self.assertTrue(shard["range_requested"])
            self.assertEqual(shard["selected_entry_start"], 1)
            self.assertEqual(shard["selected_entry_stop"], 2)
            self.assertEqual(shard["expected_raw_entries"], [2])

            args.sample_type = "real_testbeam"
            args.selection_mode = "identity_cutflow"
            args.expected_selected_entry_start = None
            args.expected_selected_entry_stop = None
            identity = inspect_source_selection(args)
            self.assertIsInstance(identity["expected_raw_entries"], range)
            self.assertEqual(list(identity["expected_raw_entries"]), [0, 1, 2])


class UnifiedFeatureShardMergeTest(unittest.TestCase):
    @staticmethod
    def write_shard(path: Path, entries: list[int]) -> None:
        output = ROOT.TFile(str(path), "RECREATE")
        for tree_name in ("sndData", "cutFlowSummary"):
            tree = ROOT.TTree(tree_name, tree_name)
            value = array.array("q", [0])
            branch_name = "eventIndex" if tree_name == "sndData" else "entry"
            tree.Branch(branch_name, value, f"{branch_name}/L")
            domain = array.array("i", [2])
            if tree_name == "sndData":
                tree.Branch("source_domain_id", domain, "source_domain_id/I")
            for entry in entries:
                value[0] = entry
                tree.Fill()
            tree.Write()
        ROOT.TNamed("metadata_source_uid", "unified_src_test").Write()
        output.Close()

    @staticmethod
    def write_receipt(path: Path, feature: Path, start: int, stop: int) -> None:
        payload = {
            "validation_passed": True,
            "manifest_version": "unified_shower_features_v1",
            "unified_schema_version": "unified_shower_features_v1",
            "scifi_cluster_schema_version": "scifi_orientation_clusters_v1",
            "source_uid": "unified_src_test",
            "source_domain_id": 2,
            "source_domain": "mc_muon_up",
            "source_geometry_id": "",
            "source_geometry_checksum": "",
            "input_digi_path": "/input.root",
            "input_cutflow_path": "",
            "expected_input_event_count": 4,
            "source_selection_expression": "",
            "available_selected_event_count": 4,
            "conversion_max_events": 0,
            "converter_commit": "test",
            "selected_entry_start": start,
            "selected_entry_stop": stop,
            "written_event_count": stop - start,
            "feature_path": str(feature),
            "feature_sha256": feature_sha256(feature),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_merge_preserves_shard_order_and_tree_alignment(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            shard0 = directory / "shard0.root"
            shard1 = directory / "shard1.root"
            self.write_shard(shard0, [0, 1])
            self.write_shard(shard1, [2, 3])
            receipt0 = directory / "shard0.json"
            receipt1 = directory / "shard1.json"
            self.write_receipt(receipt0, shard0, 0, 2)
            self.write_receipt(receipt1, shard1, 2, 4)
            merged = directory / "merged.root"
            merge_record = directory / "merge.json"

            report = merge_feature_shards(SimpleNamespace(
                shard_record=[str(receipt1), str(receipt0)],
                output=str(merged),
                merge_record=str(merge_record),
            ))
            self.assertTrue(report["merge_passed"])
            self.assertEqual(report["shard_count"], 2)
            self.assertEqual(report["merged_event_count"], 4)
            self.assertEqual(
                report["shard_selected_entry_ranges"], [[0, 2], [2, 4]]
            )

            source = ROOT.TFile.Open(str(merged))
            try:
                feature = source.Get("sndData")
                cutflow = source.Get("cutFlowSummary")
                self.assertEqual(int(feature.GetEntries()), 4)
                self.assertEqual(int(cutflow.GetEntries()), 4)
                observed_feature = []
                observed_cutflow = []
                for entry in range(4):
                    feature.GetEntry(entry)
                    cutflow.GetEntry(entry)
                    observed_feature.append(int(feature.eventIndex))
                    observed_cutflow.append(int(cutflow.entry))
                self.assertEqual(observed_feature, [0, 1, 2, 3])
                self.assertEqual(observed_cutflow, observed_feature)
                self.assertEqual(
                    validate_merge_boundaries(
                        feature, cutflow, range(4), report, 2
                    ),
                    4,
                )
                self.assertEqual(
                    int(source.Get("metadata_feature_shard_count").GetVal()), 2
                )
            finally:
                source.Close()


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
            self.assertEqual(
                int(first["mc_neutrino"].loc[0, "input_event_count"]), 48
            )
            self.assertNotIn("expected_event_count", first["mc_neutrino"].columns)

    def test_rule_contract_uses_json_validation_records(self):
        rules = (
            REPO_ROOT / "snakemake/rules/unified_shower_features.smk"
        ).read_text(encoding="utf-8")
        self.assertIn("validate_unified_shower_feature_file.py", rules)
        self.assertIn(".root.validated.json", rules)
        self.assertIn("--validation-record", rules)
        self.assertIn("--expected-input-event-count", rules)
        self.assertIn("--expected-max-events", rules)
        self.assertIn("unified_shower_feature_shard_v1", rules)
        self.assertIn("unified_merge_shower_feature_shards_v1", rules)
        self.assertIn("--expected-selected-entry-start", rules)
        self.assertIn("--merge-record", rules)
        self.assertNotIn("--expected-source-event-count", rules)
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

    def test_large_family_shard_sizes_are_dataset_specific(self):
        config = yaml.safe_load(
            (
                REPO_ROOT
                / "snakemake/metadata/configs/unified_shower_features_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        sizes = config["sharding"]["selected_events_per_shard"]
        self.assertEqual(sizes["mc_muon_up"], 50000)
        self.assertEqual(sizes["testbeam_data_2024"], 10000)


if __name__ == "__main__":
    unittest.main()
