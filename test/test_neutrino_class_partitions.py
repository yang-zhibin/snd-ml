import argparse
import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import pandas as pd
import ROOT


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from build_neutrino_class_partitions import process_source


CONFIG = REPO / "snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml"


def write_tree(path, tree_name, codes):
    root_file = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree(tree_name, "")
    values = {name: array("i", [0]) for name in (
        "runId", "eventId", "eventIndex", "pdgCode"
    )}
    for name, value in values.items():
        tree.Branch(name, value, f"{name}/I")
    for entry, code in enumerate(codes):
        values["runId"][0] = 1
        values["eventId"][0] = entry
        values["eventIndex"][0] = entry
        values["pdgCode"][0] = code
        tree.Fill()
    tree.Write()
    root_file.Close()


class NeutrinoClassPartitionTest(unittest.TestCase):
    def run_partition(self, directory, codes):
        feature = directory / "feature.root"
        hit3d = directory / "hit3d.root"
        write_tree(feature, "sndData", codes)
        write_tree(hit3d, "hit3D", codes)
        catalog = directory / "sources.csv"
        pd.DataFrame([{
            "source_uid": "src_test", "role": "neutrino_split",
            "domain": "mc", "input_available": True,
            "fold_group_uid": "src_test", "lumi_source_uid": "src_test",
            "dataset_key": "nu", "run": "", "production": "test",
            "energy_bin": "", "lumi_per_file": 15.0,
            "feature_path": str(feature), "hit3d_path": str(hit3d),
        }]).to_csv(catalog, index=False)
        args = argparse.Namespace(
            config=str(CONFIG), source_catalog=str(catalog),
            source_uid="src_test", output_root=str(directory / "out"),
            audit=str(directory / "audit.csv"), threads=0,
        )
        return process_source(args)

    def test_four_way_partition_closes_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            codes = [14, -14, 12, -12, 16, -16, 112, -114, 116]
            result = self.run_partition(directory, codes)
            self.assertEqual(result["partitioned_entries"], len(codes))
            self.assertEqual(result["unclassified_entries"], 0)
            self.assertEqual(result["output_identity_mismatches"], 0)
            partitions = json.loads(result["partitions"])
            self.assertEqual(
                {name: value["entries"] for name, value in partitions.items()},
                {"CC_numu": 2, "CC_nue": 2, "CC_nutau": 2, "NC": 3},
            )
            for class_name, expected in (
                ("CC_numu", 2), ("CC_nue", 2), ("CC_nutau", 2), ("NC", 3),
            ):
                root_file = ROOT.TFile.Open(str(directory / "out" / f"{class_name}_hit3d.root"))
                tree = root_file.Get("hit3D")
                self.assertEqual(tree.GetEntries(), expected)
                self.assertTrue(tree.GetBranch("ml_source_entry"))
                root_file.Close()

    def test_empty_flavor_outputs_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result = self.run_partition(directory, [12, -12, 112])
            partitions = json.loads(result["partitions"])
            self.assertEqual(partitions["CC_numu"]["entries"], 0)
            self.assertEqual(partitions["CC_nutau"]["entries"], 0)
            root_file = ROOT.TFile.Open(str(directory / "out" / "CC_nutau_hit3d.root"))
            self.assertIsNotNone(root_file.Get("hit3D"))
            self.assertEqual(root_file.Get("hit3D").GetEntries(), 0)
            root_file.Close()


if __name__ == "__main__":
    unittest.main()
