import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from array import array

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CLASSES = [
    "CC_numu", "CC_nue", "NC", "neutral_kaon", "neutron",
    "passing_muon", "muon_DIS",
]


def write_hits(path):
    import ROOT
    output = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("hit3D", "")
    scalars = {name: array("i", [0]) for name in (
        "runId", "eventId", "eventIndex", "pdgCode")}
    energy = array("f", [0])
    for name, value in scalars.items():
        tree.Branch(name, value, f"{name}/I")
    tree.Branch("energy", energy, "energy/F")
    vectors = {
        "hit_x": ROOT.std.vector("float")(),
        "hit_y": ROOT.std.vector("float")(),
        "hit_z": ROOT.std.vector("float")(),
        "hit_qdc": ROOT.std.vector("float")(),
        "hit_detType": ROOT.std.vector("short")(),
    }
    for name, value in vectors.items():
        tree.Branch(name, value)
    for entry in range(7):
        scalars["runId"][0] = 100
        scalars["eventId"][0] = entry
        scalars["eventIndex"][0] = entry
        scalars["pdgCode"][0] = 14
        energy[0] = 10
        for value in vectors.values():
            value.clear()
        for hit in range(4):
            vectors["hit_x"].push_back(entry + hit * 0.1)
            vectors["hit_y"].push_back(hit)
            vectors["hit_z"].push_back(entry + hit)
            vectors["hit_qdc"].push_back(1 + entry + hit)
            vectors["hit_detType"].push_back(1)
        tree.Fill()
    tree.Write()
    output.Close()


@unittest.skipUnless(importlib.util.find_spec("ROOT"), "PyROOT unavailable")
@unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyG unavailable")
class TrainClassifierSmokeTest(unittest.TestCase):
    def test_one_fold_writes_checkpoints_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            rows = []
            density_paths = {}
            index_paths = {}
            for fold in range(5):
                root_path = tmp / f"fold_{fold}.root"
                write_hits(root_path)
                density = tmp / f"density_{fold}.npz"
                np.savez_compressed(density, bin_0=np.asarray([6], dtype=np.int64))
                density_paths[fold] = density
                for class_id, class_name in enumerate(CLASSES):
                    index = tmp / f"index_{fold}_{class_id}.npz"
                    np.savez_compressed(index, entries=np.asarray([class_id], dtype=np.int64))
                    index_paths[(fold, class_id)] = index
                    uid = f"fold{fold}_{class_name}"
                    for cv_fold in range(5):
                        rows.append({
                            "cv_fold": cv_fold,
                            "split": "validation" if fold == cv_fold else "train",
                            "fold": fold, "dataset_file_uid": uid,
                            "hit3d_path": str(root_path), "class_id": class_id,
                            "class_name": class_name, "domain": "data" if class_name == "muon_DIS" else "mc",
                            "n_events": 1, "event_filter": f"entry == {class_id}",
                            "entry_index_path": str(index),
                            "density_index_path": str(density) if class_name == "muon_DIS" else "",
                            "sampling_strategy": "natural" if fold == cv_fold else (
                                "uniform_populated_density_bins" if class_name == "muon_DIS" else "natural"),
                        })
            views = tmp / "views.csv"
            pd.DataFrame(rows).to_csv(views, index=False)
            validation = tmp / "validation.json"
            validation.write_text(json.dumps({
                "valid": True, "dataset_version": "data_driven_3d_cnn_v1", "folds": 5,
            }))
            metadata_root = tmp / "metadata"
            identities = [
                {"dataset_file_uid": f"fold{fold}_{name}", "source_entry": class_id,
                 "n_candidates": 4, "n_voxels": 4}
                for fold in range(5) for class_id, name in enumerate(CLASSES)
            ]
            normalization = {
                "version": "scifi_xyz_qdc_dettype_v1",
                "dataset_version": "data_driven_3d_cnn_v1", "cv_fold": 0,
                "position_mean": [3, 1.5, 4.5], "position_scale": [2, 1, 2],
                "qdc_transform": "log1p", "qdc_mean": 1.5, "qdc_scale": 1,
                "training_events": 28, "training_candidates": 112,
            }
            fold_dir = metadata_root / "fold_0"
            fold_dir.mkdir(parents=True)
            (fold_dir / "normalization.json").write_text(json.dumps(normalization))
            pd.DataFrame(identities).to_csv(fold_dir / "event_sizes.csv", index=False)
            (fold_dir / "summary.json").write_text(json.dumps({
                "cv_fold": 0, "coordinate_min": [-1, -1, -1],
                "coordinate_max": [20, 20, 20], "voxel_size": [1, 1, 1],
            }))
            config = {
                "model_version": "pyg_gravnet_smoke", "task": "classification",
                "dataset_version": "data_driven_3d_cnn_v1", "num_classes": 7,
                "class_names": CLASSES, "excluded_classes": ["CC_nutau"],
                "data": {
                    "fold_views": str(views), "validation_report": str(validation),
                    "normalization_path": str(metadata_root / "fold_{cv_fold}" / "normalization.json"),
                    "event_size_manifest": str(metadata_root / "fold_{cv_fold}" / "event_sizes.csv"),
                    "representation_summary": str(metadata_root / "fold_{cv_fold}" / "summary.json"),
                    "cv_fold": 0, "n_folds": 5, "epoch_size": 14,
                    "class_probabilities": "uniform", "domain_probabilities": None,
                    "muondis_density_probabilities": "uniform",
                    "validation_sampling": "natural",
                    "allowed_detector_types": [1],
                    "raw_input_fields": ["hit_x", "hit_y", "hit_z", "hit_qdc", "hit_detType"],
                    "tree_name": "hit3D", "reader_backend": "pyroot", "max_open_files": 2,
                    "num_workers": 0, "pin_memory": False,
                    "max_events_per_batch": 4, "max_nodes_per_batch": 100,
                },
                "representation": {
                    "version": "scifi_xyz_qdc_dettype_v1", "qdc_transform": "log1p",
                    "voxel_size": [1, 1, 1], "coordinate_min": None, "coordinate_max": None,
                    "include_station": False, "include_occupancy_as_input": False,
                    "include_time": False, "gravnet_nodes": "candidate_crossings",
                },
                "model": {"family": "pyg_gravnet", "in_channels": 5,
                          "hidden_channels": 8, "blocks": 1, "space_dimensions": 3,
                          "propagate_dimensions": 4, "k": 2, "dropout": 0.0},
                "optim": {"lr": 0.001, "weight_decay": 0},
                "objective": {"name": "cross_entropy", "class_weights": None},
                "trainer": {"max_epochs": 1, "accelerator": "cpu", "devices": 1,
                            "precision": "32-true", "log_every_n_steps": 1,
                            "enable_progress_bar": False, "enable_model_summary": False},
                "checkpoint": {"monitor": "val/macro_f1", "mode": "max",
                               "save_top_k": 1, "save_last": True},
                "early_stopping": {"patience": 2},
                "wandb": {"project": "snd-ml-data-driven-3d", "mode": "disabled",
                          "prediction_table_max_rows": 20,
                          "diagnostic_table_every_n_epochs": 1},
                "runtime": {"seed": 7},
            }
            config_path = tmp / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output = tmp / "output"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = (
                str(REPO / "train_3d" / "src")
                + os.pathsep + environment.get("PYTHONPATH", "")
            )
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run([
                sys.executable, str(REPO / "train_3d" / "train_classifier.py"),
                "--config", str(config_path), "--cv-fold", "0",
                "--output-dir", str(output),
            ], check=True, cwd=REPO, env=environment, timeout=180)
            run_dir = output / "pyg_gravnet_smoke" / "fold_0"
            manifest = json.loads((run_dir / "training_complete.json").read_text())
            self.assertTrue(Path(manifest["best_checkpoint"]).is_file())
            self.assertTrue(Path(manifest["last_checkpoint"]).is_file())
            self.assertEqual(manifest["cv_fold"], 0)
            shard_paths = []
            for shard in range(2):
                shard_path = tmp / f"shard_{shard}.json"
                shard_paths.append(shard_path)
                subprocess.run([
                    sys.executable,
                    str(REPO / "train_3d" / "evaluate_classifier_shard.py"),
                    "--training-manifest", str(run_dir / "training_complete.json"),
                    "--shard-index", str(shard), "--num-shards", "2",
                    "--num-threads", "1", "--device", "cpu",
                    "--output", str(shard_path),
                ], check=True, cwd=REPO, env=environment, timeout=180)
            merged_path = tmp / "merged.json"
            subprocess.run([
                sys.executable,
                str(REPO / "train_3d" / "merge_classifier_evaluation.py"),
                "--shards", *(str(path) for path in shard_paths),
                "--output", str(merged_path),
            ], check=True, cwd=REPO, env=environment, timeout=60)
            merged = json.loads(merged_path.read_text())
            self.assertEqual(merged["global"]["event_count"], 7)
            self.assertEqual(sum(map(sum, merged["global"]["confusion"])), 7)


if __name__ == "__main__":
    unittest.main()
