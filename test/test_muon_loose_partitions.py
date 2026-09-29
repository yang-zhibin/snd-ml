import argparse
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import pandas as pd
import ROOT
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from build_muon_loose_partitions import process_source, validate_thread_count
from build_muondis_density_indices import build_indices
from data_driven_3d_dataset import configured_density_bins


CONFIG = REPO / "snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml"


COUNT_BRANCHES = [
    "count_veto", "count_scifi", "count_us", "count_ds",
    *(f"count_veto{i}" for i in range(1, 4)),
    *(f"count_scifi{i}" for i in range(1, 6)),
    *(f"count_us{i}" for i in range(1, 6)),
    *(f"count_ds{i}" for i in range(1, 5)),
    "scifi_n_active_stations_xy",
]


def make_inputs(directory: Path):
    feature_path = directory / "feature.root"
    hit_path = directory / "hit3d.root"
    feature_file = ROOT.TFile(str(feature_path), "RECREATE")
    feature = ROOT.TTree("sndData", "")
    ints = {
        name: array("i", [0])
        for name in ["runId", "eventId", "eventIndex", "pdgCode", *COUNT_BRANCHES]
    }
    density = array("d", [0.0])
    for name, value in ints.items():
        feature.Branch(name, value, f"{name}/I")
    feature.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")

    events = [
        {"density": 250.0, "scifi": [2, 2, 2, 2, 2], "xy": 5, "veto": 0},
        {"density": 50.0, "scifi": [2, 2, 2, 2, 2], "xy": 5},
        {"density": 600.0, "scifi": [25, 2, 2, 2, 2], "xy": 5},
        {"density": 6000.0, "scifi": [2, 2, 2, 2, 2], "xy": 4},
    ]
    for event_id, event in enumerate(events):
        ints["runId"][0] = 8329
        ints["eventId"][0] = event_id
        ints["eventIndex"][0] = event_id
        ints["pdgCode"][0] = 13
        veto = event.get("veto", 3)
        for i in range(1, 4):
            ints[f"count_veto{i}"][0] = 1 if veto else 0
        ints["count_veto"][0] = veto
        for i, value in enumerate(event["scifi"], start=1):
            ints[f"count_scifi{i}"][0] = value
        ints["count_scifi"][0] = sum(event["scifi"])
        for i in range(1, 6):
            ints[f"count_us{i}"][0] = 1
        ints["count_us"][0] = 5
        for i in range(1, 5):
            ints[f"count_ds{i}"][0] = 1
        ints["count_ds"][0] = 4
        ints["scifi_n_active_stations_xy"][0] = event["xy"]
        density[0] = event["density"]
        feature.Fill()
    feature.Write()
    feature_file.Close()

    hit_file = ROOT.TFile(str(hit_path), "RECREATE")
    hit = ROOT.TTree("hit3D", "")
    hit_values = {
        name: array("i", [0]) for name in ("runId", "eventId", "eventIndex", "pdgCode")
    }
    dummy = array("d", [0.0])
    for name, value in hit_values.items():
        hit.Branch(name, value, f"{name}/I")
    hit.Branch("dummy_hit", dummy, "dummy_hit/D")
    for event_id in range(len(events)):
        hit_values["runId"][0] = 8329
        hit_values["eventId"][0] = event_id
        hit_values["eventIndex"][0] = event_id
        hit_values["pdgCode"][0] = 13
        dummy[0] = event_id + 0.5
        hit.Fill()
    hit.Write()
    hit_file.Close()
    return feature_path, hit_path


class MuonLoosePartitionTest(unittest.TestCase):
    def test_aligned_pass_fail_outputs_and_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            feature, hit = make_inputs(directory)
            catalog = directory / "sources.csv"
            pd.DataFrame([{
                "source_uid": "src_test",
                "role": "muon_split",
                "domain": "data",
                "dataset_key": "test_data",
                "fold_group_uid": "data_run_8329",
                "lumi_source_uid": "src_test",
                "run": "8329",
                "production": "test",
                "lumi_per_file": 0.1,
                "feature_path": str(feature),
                "hit3d_path": str(hit),
                "input_available": True,
            }]).to_csv(catalog, index=False)
            args = argparse.Namespace(
                config=str(CONFIG), source_catalog=str(catalog), source_uid="src_test",
                pass_feature=str(directory / "pass_feature.root"),
                pass_hit3d=str(directory / "pass_hit.root"),
                fail_feature=str(directory / "fail_feature.root"),
                fail_hit3d=str(directory / "fail_hit.root"),
                signal_feature=str(directory / "signal_feature.root"),
                signal_hit3d=str(directory / "signal_hit.root"),
                audit=str(directory / "audit.csv"), threads=0,
            )
            result = process_source(args)
            self.assertEqual(result["input_entries"], 4)
            self.assertEqual(result["parent_entries"], 3)
            self.assertEqual(result["blinded_entries"], 1)
            self.assertEqual(result["blinded_fail_leakage"], 0)
            self.assertEqual(result["signal_region_entries"], 1)
            self.assertEqual(result["pass_entries"], 1)
            self.assertEqual(result["fail_entries"], 2)
            self.assertEqual(result["identity_mismatches"], 0)
            self.assertEqual(result["output_identity_mismatches"], 0)

            for path, expected in ((args.pass_hit3d, 1), (args.fail_hit3d, 2)):
                root_file = ROOT.TFile.Open(path)
                tree = root_file.Get("hit3D")
                self.assertEqual(tree.GetEntries(), expected)
                self.assertTrue(tree.GetBranch("density_sndsw_scifi"))
                self.assertTrue(tree.GetBranch("ml_source_entry"))
                self.assertTrue(tree.GetBranch("ml_class_id"))
                root_file.Close()

            signal_file = ROOT.TFile.Open(args.signal_hit3d)
            signal_tree = signal_file.Get("hit3D")
            self.assertEqual(signal_tree.GetEntries(), 1)
            signal_tree.GetEntry(0)
            self.assertEqual(int(signal_tree.eventId), 0)
            signal_file.Close()

            config = yaml.safe_load(CONFIG.read_text())
            arrays, invalid = build_indices(
                args.fail_hit3d,
                "hit3D",
                "density_sndsw_scifi",
                configured_density_bins(config),
            )
            self.assertEqual([len(values) for values in arrays], [0, 1, 0, 1, 0])
            self.assertEqual(len(invalid), 0)

    def test_implicit_multithreading_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "single-threaded"):
            validate_thread_count(2)

    def test_unequal_input_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            feature, hit = make_inputs(directory)
            original_file = ROOT.TFile.Open(str(hit))
            short_path = directory / "short_hit3d.root"
            short_file = ROOT.TFile(str(short_path), "RECREATE")
            original_file.Get("hit3D").CloneTree(2).Write()
            short_file.Close()
            original_file.Close()
            catalog = directory / "sources.csv"
            pd.DataFrame([{
                "source_uid": "src_short", "role": "muon_split", "domain": "data",
                "dataset_key": "test_data", "fold_group_uid": "data_run_8329",
                "lumi_source_uid": "src_short", "run": "8329", "production": "test",
                "lumi_per_file": 0.1, "feature_path": str(feature),
                "hit3d_path": str(short_path), "input_available": True,
            }]).to_csv(catalog, index=False)
            args = argparse.Namespace(
                config=str(CONFIG), source_catalog=str(catalog), source_uid="src_short",
                pass_feature=str(directory / "pass_feature.root"),
                pass_hit3d=str(directory / "pass_hit.root"),
                fail_feature=str(directory / "fail_feature.root"),
                fail_hit3d=str(directory / "fail_hit.root"),
                signal_feature=str(directory / "signal_feature.root"),
                signal_hit3d=str(directory / "signal_hit.root"),
                audit=str(directory / "audit.csv"), threads=0,
            )
            with self.assertRaisesRegex(RuntimeError, "Entry mismatch"):
                process_source(args)


if __name__ == "__main__":
    unittest.main()
