import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import pandas as pd
import ROOT
import yaml


REPO = Path(__file__).resolve().parents[1]


def make_file(path, offset):
    root_file = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("hit3D", "")
    vectors = {
        "hit_x": ROOT.std.vector("float")(),
        "hit_y": ROOT.std.vector("float")(),
        "hit_z": ROOT.std.vector("float")(),
        "hit_qdc": ROOT.std.vector("float")(),
        "hit_detType": ROOT.std.vector("short")(),
    }
    for name, value in vectors.items():
        tree.Branch(name, value)
    for name, values in {
        "hit_x": [offset + 1.0, 999.0],
        "hit_y": [2.0, 999.0],
        "hit_z": [3.0, 999.0],
        "hit_qdc": [4.0, 1000.0],
        "hit_detType": [1, 2],
    }.items():
        for value in values:
            vectors[name].push_back(value)
    tree.Fill()
    tree.Write()
    root_file.Close()


class BuildSciFiStatisticsTest(unittest.TestCase):
    def test_one_two_pass_scan_builds_all_fold_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = []
            for fold in range(5):
                path = directory / f"fold_{fold}.root"
                make_file(path, fold)
                paths.append(path)
            rows = []
            for cv_fold in range(5):
                for fold, path in enumerate(paths):
                    rows.append({
                        "cv_fold": cv_fold,
                        "split": "validation" if fold == cv_fold else "train",
                        "fold": fold,
                        "dataset_file_uid": f"file_{fold}",
                        "hit3d_path": str(path),
                        "class_id": 0,
                        "class_name": "CC_numu",
                        "domain": "mc",
                        "n_events": 1,
                        "event_filter": "true",
                        "entry_index_path": "",
                        "density_index_path": "",
                        "sampling_strategy": "natural",
                    })
            views = directory / "views.csv"
            pd.DataFrame(rows).to_csv(views, index=False)
            config = directory / "config.yaml"
            config.write_text(yaml.safe_dump({
                "dataset_version": "data_driven_3d_cnn_v1",
                "data": {
                    "fold_views": str(views), "n_folds": 5,
                    "tree_name": "hit3D", "max_open_files": 2,
                },
                "representation": {"voxel_size": [1.0, 1.0, 1.0]},
            }))
            output = directory / "output"
            subprocess.run([
                sys.executable, str(REPO / "train_3d/build_scifi_statistics.py"),
                "--config", str(config), "--output-root", str(output),
            ], check=True, cwd=REPO)
            for fold in range(5):
                normalization = yaml.safe_load(
                    (output / f"fold_{fold}/normalization.json").read_text()
                )
                self.assertEqual(normalization["training_events"], 4)
                manifest = pd.read_csv(output / f"fold_{fold}/event_sizes.csv")
                self.assertEqual(len(manifest), 5)
                self.assertTrue(manifest["n_candidates"].eq(1).all())
                self.assertTrue(manifest["n_voxels"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
