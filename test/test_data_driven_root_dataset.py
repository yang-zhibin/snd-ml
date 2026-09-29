import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))

from train_3d.data.data_driven_fold_dataset import EventReference
from train_3d.data.data_driven_root_dataset import (
    RootBackedClassificationDataset,
    RootHit3DEventReader,
    build_root_fold_datasets,
    classification_sparse_collate_fn,
)
from train_3d.data.scifi_hit_event_dataset import (
    RootSciFiEventReader,
    UprootSciFiEventReader,
)


def make_hit3d(path: Path):
    root_file = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("hit3D", "")
    scalars = {
        name: array("i", [0])
        for name in ("runId", "eventId", "eventIndex", "pdgCode")
    }
    energy = array("f", [0.0])
    for name, value in scalars.items():
        tree.Branch(name, value, f"{name}/I")
    tree.Branch("energy", energy, "energy/F")
    vectors = {
        "hit_x": ROOT.std.vector("float")(),
        "hit_y": ROOT.std.vector("float")(),
        "hit_z": ROOT.std.vector("float")(),
        "hit_qdc": ROOT.std.vector("float")(),
        "hit_station": ROOT.std.vector("short")(),
        "hit_detType": ROOT.std.vector("short")(),
    }
    for name, value in vectors.items():
        tree.Branch(name, value)

    events = [
        [(1.2, 2.2, 3.2, 4.0, 1, 1), (1.8, 2.7, 3.9, 6.0, 3, 2)],
        [(5.2, 6.2, 7.2, 8.0, 4, 3)],
    ]
    for entry, hits in enumerate(events):
        scalars["runId"][0] = 8329
        scalars["eventId"][0] = 100 + entry
        scalars["eventIndex"][0] = entry
        scalars["pdgCode"][0] = 14
        energy[0] = 10.0 + entry
        for value in vectors.values():
            value.clear()
        for hit in hits:
            for branch, value in zip(vectors, hit):
                vectors[branch].push_back(value)
        tree.Fill()
    tree.Write()
    root_file.Close()


class RootDatasetTest(unittest.TestCase):
    def references(self, path):
        return [
            EventReference("a", str(path), 0, 0, "CC_numu", "mc"),
            EventReference("a", str(path), 1, 0, "CC_numu", "mc"),
        ]

    def test_reader_fetches_exact_physical_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.root"
            make_hit3d(path)
            reader = RootHit3DEventReader(max_open_files=1)
            event = reader.read(self.references(path)[1])
            self.assertEqual(event["runId"], 8329)
            self.assertEqual(event["eventId"], 101)
            self.assertEqual(event["eventIndex"], 1)
            np.testing.assert_allclose(event["all_3dHits"]["qdc"], [8.0])
            reader.close()
            with self.assertRaises(IndexError):
                reader.read(EventReference("a", str(path), 2, 0, "CC_numu", "mc"))

    def test_pyroot_and_uproot_scifi_readers_are_identical(self):
        import uproot
        from packaging.version import Version
        if Version(uproot.__version__) < Version("5.4.0"):
            self.skipTest(
                "LCG uproot <5.4 can hang decoding ROOT std::vector branches"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.root"
            make_hit3d(path)
            reference = self.references(path)[0]
            pyroot = RootSciFiEventReader(max_open_files=1)
            first = pyroot.read(reference)
            # Do not keep the same file open through both ROOT I/O libraries.
            # PyROOT and uproot can otherwise contend in some LCG builds.
            pyroot.close()
            uproot = UprootSciFiEventReader(max_open_files=1)
            second = uproot.read(reference)
            for name in ("position", "qdc", "detector_type"):
                np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(first["eventId"], second["eventId"])
            # The detType=2 hit is excluded identically by both readers.
            self.assertEqual(len(first["qdc"]), 1)
            uproot.close()

    def test_dataset_voxelizes_and_collates_labels_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.root"
            make_hit3d(path)
            dataset = RootBackedClassificationDataset(
                self.references(path),
                ranges={"x_range": (0, 10), "y_range": (0, 10), "z_range": (0, 10)},
                voxel_size=(1, 1, 1),
                feature_channels=("qdc_sum", "occupancy", "dettype_1", "station_mean"),
            )
            first = dataset[0]
            self.assertEqual(first["label"].item(), 0)
            self.assertEqual(first["source_class_id"].item(), 0)
            self.assertEqual(first["event_id"], 100)
            self.assertEqual(tuple(first["coords"].shape), (1, 3))
            np.testing.assert_allclose(first["features"].numpy()[0], [10.0, 2.0, 1.0, 2.0])

            batch = classification_sparse_collate_fn([first, dataset[1]])
            self.assertEqual(batch["label"].tolist(), [0, 0])
            self.assertEqual(batch["source_class_id"].tolist(), [0, 0])
            self.assertEqual(batch["event_id"].tolist(), [100, 101])
            self.assertEqual(batch["coords"][:, 0].tolist(), [0, 1])
            self.assertEqual(tuple(batch["features"].shape), (2, 4))

            muondis = RootBackedClassificationDataset(
                [EventReference("dis", str(path), 0, 7, "muon_DIS", "data")],
                ranges={"x_range": (0, 10), "y_range": (0, 10), "z_range": (0, 10)},
                voxel_size=(1, 1, 1), feature_channels=("occupancy",),
            )[0]
            self.assertEqual(muondis["label"].item(), 6)
            self.assertEqual(muondis["source_class_id"].item(), 7)

    def test_dataset_rejects_epoch_change_for_natural_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.root"
            make_hit3d(path)
            dataset = RootBackedClassificationDataset(
                self.references(path),
                ranges={"x_range": (0, 10), "y_range": (0, 10), "z_range": (0, 10)},
                voxel_size=(1, 1, 1), feature_channels=("occupancy",),
            )
            with self.assertRaises(TypeError):
                dataset.set_epoch(1)

    def test_fold_factory_builds_sampled_train_and_natural_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.root"
            make_hit3d(path)
            views = pd.DataFrame([
                {
                    "cv_fold": 0, "split": "train", "fold": 1,
                    "dataset_file_uid": "train", "hit3d_path": str(path),
                    "class_id": 0, "class_name": "CC_numu", "domain": "mc",
                    "n_events": 2, "event_filter": "true",
                    "entry_index_path": "", "density_index_path": "",
                    "sampling_strategy": "natural",
                },
                {
                    "cv_fold": 0, "split": "validation", "fold": 0,
                    "dataset_file_uid": "validation", "hit3d_path": str(path),
                    "class_id": 0, "class_name": "CC_numu", "domain": "mc",
                    "n_events": 2, "event_filter": "true",
                    "entry_index_path": "", "density_index_path": "",
                    "sampling_strategy": "natural",
                },
            ])
            datasets = build_root_fold_datasets(
                views, cv_fold=0, epoch_size=30,
                ranges={"x_range": (0, 10), "y_range": (0, 10), "z_range": (0, 10)},
                voxel_size=(1, 1, 1), feature_channels=("occupancy",), seed=3,
            )
            self.assertEqual(len(datasets["train"]), 30)
            self.assertEqual(len(datasets["validation"]), 2)
            epoch_zero = list(datasets["train"].references.references)
            datasets["train"].set_epoch(1)
            self.assertNotEqual(epoch_zero, datasets["train"].references.references)
            self.assertEqual(datasets["validation"][1]["event_id"], 101)


if __name__ == "__main__":
    unittest.main()
