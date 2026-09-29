import json
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

from build_3d_cnn_dataset_catalog import build_catalog
from build_3d_cnn_entry_indices import build_indices


CONFIG_PATH = REPO / "snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml"


def make_hit_file(path: Path):
    root_file = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("hit3D", "")
    pdg = array("i", [0])
    tree.Branch("pdgCode", pdg, "pdgCode/I")
    for value in (12, -12, 112, -112):
        pdg[0] = value
        tree.Fill()
    tree.Write()
    root_file.Close()


def source(uid, mode, path, direct_class="", pdg_map=None):
    return {
        "source_uid": uid, "fold_group_uid": f"group_{uid}",
        "lumi_source_uid": uid, "dataset_key": uid, "domain": "mc",
        "run": "", "production": "test", "energy_bin": "10_20",
        "feature_path": str(path), "hit3d_path": str(path),
        "lumi_per_file": 2.5, "input_available": True,
        "role": "prepared_physics", "class_mode": mode,
        "direct_class": direct_class,
        "pdg_class_map": json.dumps(pdg_map or {}), "n_event_metadata": 4,
    }


class DatasetCatalogTest(unittest.TestCase):
    def test_cc_nc_share_original_source_and_luminosity(self):
        with tempfile.TemporaryDirectory() as tmp:
            hit_path = Path(tmp) / "hit.root"
            make_hit_file(hit_path)
            sources = pd.DataFrame([
                source("src_nue", "pdg_map", hit_path, pdg_map={
                    "12": "CC_nue", "-12": "CC_nue",
                    "112": "NC", "-112": "NC",
                }),
                source("src_kaon", "direct", hit_path, direct_class="neutral_kaon"),
            ])
            config = yaml.safe_load(CONFIG_PATH.read_text())
            catalog = build_catalog(sources, pd.DataFrame(), config, inspect_root=True)
            counts = catalog.set_index("class_name")["n_events"].to_dict()
            self.assertEqual(counts, {"CC_nue": 2, "NC": 2, "neutral_kaon": 4})
            neutrino = catalog.loc[catalog["source_uid"].eq("src_nue")]
            self.assertEqual(neutrino["fold_group_uid"].nunique(), 1)
            self.assertEqual(neutrino["lumi_source_uid"].nunique(), 1)
            self.assertEqual(neutrino["lumi_per_file"].tolist(), [2.5, 2.5])
            cc_expression = neutrino.loc[neutrino["class_name"].eq("CC_nue"), "event_filter"].iloc[0]
            self.assertEqual(
                build_indices(str(hit_path), "hit3D", cc_expression).tolist(),
                [0, 1],
            )

    def test_physical_neutrino_partitions_need_no_runtime_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            hit_path = Path(tmp) / "hit.root"
            make_hit_file(hit_path)
            row = source("src_nu", "physical_pdg_split", hit_path)
            row["role"] = "neutrino_split"
            sources = pd.DataFrame([row])
            partitions = {
                "CC_numu": {"entries": 0, "feature_path": "numu_f.root", "hit3d_path": "numu_h.root"},
                "CC_nue": {"entries": 2, "feature_path": "nue_f.root", "hit3d_path": "nue_h.root"},
                "CC_nutau": {"entries": 0, "feature_path": "nutau_f.root", "hit3d_path": "nutau_h.root"},
                "NC": {"entries": 2, "feature_path": "nc_f.root", "hit3d_path": "nc_h.root"},
            }
            neutrino_audits = pd.DataFrame([{
                "source_uid": "src_nu", "partition_version": "truth_pdg_partition_v1",
                "partitions": json.dumps(partitions),
            }])
            config = yaml.safe_load(CONFIG_PATH.read_text())
            catalog = build_catalog(
                sources, pd.DataFrame(), config, neutrino_audits, inspect_root=False
            )
            self.assertEqual(set(catalog["class_name"]), {"CC_nue", "NC"})
            self.assertEqual(set(catalog["event_filter"]), {"true"})


if __name__ == "__main__":
    unittest.main()
