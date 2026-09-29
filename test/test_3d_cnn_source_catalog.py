import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import uproot


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from build_3d_cnn_source_catalog import build_catalog


class SourceCatalogTest(unittest.TestCase):
    def test_configured_production_column_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            feature = directory / "feature.root"
            hit3d = directory / "hit3d.root"
            with uproot.recreate(feature) as root_file:
                root_file.mktree("sndData", {"dummy": "int32"})
            with uproot.recreate(hit3d) as root_file:
                root_file.mktree("hit3D", {"dummy": "int32"})
            metadata = directory / "metadata.csv"
            pd.DataFrame([{
                "data_type": "MC_muon_up",
                "subfolder": "sample",
                "partition": "scoring_at449m",
                "production_profile": "mc_default",
                "n_event": 10,
                "lumi_per_file": 1.5,
                "output_base_path": str(directory),
                "feature_path": str(feature),
                "hit3d_path": str(hit3d),
            }]).to_csv(metadata, index=False)
            config = {
                "dataset_version": "test",
                "availability_workers": 1,
                "source_datasets": {
                    "mc_muon": {
                        "metadata": str(metadata),
                        "role": "muon_split",
                        "domain": "mc",
                        "class_mode": "v7_loose_split",
                        "fold_group": "source",
                        "production_column": "partition",
                    }
                },
            }
            result = build_catalog(config, directory / "a/b/c/d/config.yaml")
            self.assertEqual(result.loc[0, "production"], "scoring_at449m")
            self.assertTrue(bool(result.loc[0, "input_available"]))


if __name__ == "__main__":
    unittest.main()
