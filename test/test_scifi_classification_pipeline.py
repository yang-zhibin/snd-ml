import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))

from train_3d.data.classification_collate import (
    GravNetClassificationCollator,
    SparseClassificationCollator,
    SparseRepresentation,
    build_scifi_sparse_voxels,
)
from train_3d.data.data_driven_fold_dataset import EventReference
from train_3d.data.node_budget_batch_sampler import EventSizeIndex, NodeBudgetBatchSampler
from train_3d.data.scifi_hit_event_dataset import (
    SciFiHitEventDataset,
    SciFiNormalization,
    validate_and_select_scifi,
)
from train_3d.data.scifi_statistics import RunningMoments, count_sparse_voxels
from train_3d.classification_metrics import confusion_matrix, metrics_from_confusion


def normalization():
    return SciFiNormalization(
        version="scifi_xyz_qdc_dettype_v1",
        dataset_version="data_driven_3d_cnn_v1",
        cv_fold=0,
        position_mean=(1.0, 2.0, 3.0),
        position_scale=(2.0, 2.0, 2.0),
        qdc_transform="log1p",
        qdc_mean=0.0,
        qdc_scale=1.0,
        training_events=2,
        training_candidates=4,
    )


def sample(uid="a", entry=0):
    return {
        "position": np.asarray([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], np.float32),
        "qdc": np.asarray([2.0, 3.0], np.float32),
        "detector_type": np.asarray([1, 1], np.int16),
        "label": 0,
        "source_class_id": 0,
        "class_name": "CC_numu",
        "domain": "mc",
        "density_bin": "",
        "dataset_file_uid": uid,
        "source_entry": entry,
        "run_id": 1,
        "event_id": entry,
        "event_index": entry,
        "pdg_code": 14,
        "energy": 10.0,
        "n_candidates": 2,
        # These must never be consumed by either collator.
        "station": np.asarray([99, 99]),
        "occupancy": np.asarray([999.0]),
        "time": np.asarray([123.0]),
    }


class FakeReader:
    def read(self, reference):
        result = sample(reference.dataset_file_uid, reference.source_entry)
        result.update({
            "runId": 1, "eventId": reference.source_entry,
            "eventIndex": reference.source_entry, "pdgCode": 14, "energy": 10.0,
        })
        return result


class SciFiClassificationPipelineTest(unittest.TestCase):
    def test_filter_is_strict_and_vectors_must_align(self):
        selected = validate_and_select_scifi({
            "x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9],
            "qdc": [10, 20, 30], "detector_type": [1, 2, 1],
        })
        np.testing.assert_array_equal(selected["position"], [[1, 4, 7], [3, 6, 9]])
        np.testing.assert_array_equal(selected["qdc"], [10, 30])
        np.testing.assert_array_equal(selected["detector_type"], [1, 1])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            validate_and_select_scifi({
                "x": [1], "y": [1, 2], "z": [1], "qdc": [1],
                "detector_type": [1],
            })

    def test_shared_dataset_preserves_labels_and_only_raw_contract(self):
        reference = EventReference("a", "unused.root", 4, 7, "muon_DIS", "data")
        dataset = SciFiHitEventDataset([reference], reader=FakeReader())
        result = dataset[0]
        self.assertEqual(result["label"], 6)
        self.assertEqual(result["source_class_id"], 7)
        self.assertEqual(set(result) & {"station", "occupancy", "time"}, set())

    def test_both_collators_start_from_identical_candidates(self):
        event = sample()
        gravnet = GravNetClassificationCollator(normalization())([event])
        sparse = SparseClassificationCollator(
            normalization(), SparseRepresentation((1, 1, 1), (0, 0, 0), (10, 10, 10))
        )([event])
        self.assertEqual(tuple(gravnet["x"].shape), (2, 5))
        self.assertEqual(tuple(gravnet["pos"].shape), (2, 3))
        self.assertEqual(set(gravnet["batch"].tolist()), {0})
        self.assertEqual(tuple(sparse["features"].shape), (1, 2))
        self.assertEqual(sparse["n_nodes"].tolist(), [1])
        # Position occupies x[:,:3] for GravNet; prohibited metadata is absent.
        np.testing.assert_allclose(
            gravnet["x"][:, :3].numpy(), normalization().normalize_position(event["position"])
        )
        self.assertNotIn("station", gravnet)
        self.assertNotIn("station", sparse)

    def test_sparse_collisions_sum_raw_qdc_then_transform(self):
        event = sample()
        coords, features = build_scifi_sparse_voxels(
            event, normalization(),
            SparseRepresentation((1, 1, 1), (0, 0, 0), (10, 10, 10)),
        )
        self.assertEqual(len(coords), 1)
        self.assertAlmostEqual(features[0, 0], np.log1p(5.0), places=6)
        self.assertEqual(features[0, 1], 1.0)
        self.assertEqual(
            count_sparse_voxels(event["position"], (0, 0, 0), (10, 10, 10), (1, 1, 1)),
            1,
        )

    def test_node_budget_never_mixes_across_limit(self):
        references = [
            EventReference("a", "", 0, 0, "CC_numu", "mc"),
            EventReference("a", "", 1, 0, "CC_numu", "mc"),
            EventReference("a", "", 2, 0, "CC_numu", "mc"),
            EventReference("a", "", 3, 0, "CC_numu", "mc"),
        ]
        sizes = EventSizeIndex(pd.DataFrame([
            {"dataset_file_uid": "a", "source_entry": index,
             "n_candidates": size, "n_voxels": size}
            for index, size in enumerate([4, 5, 12, 2])
        ]))
        batches = list(NodeBudgetBatchSampler(
            references, sizes, "gravnet", max_events=3, max_nodes=10
        ))
        self.assertEqual([[item.source_entry for item in batch] for batch in batches],
                         [[0, 1], [2], [3]])

    def test_node_budget_deterministically_skips_empty_events(self):
        references = [
            EventReference("a", "", index, 0, "CC_numu", "mc")
            for index in range(3)
        ]
        sizes = EventSizeIndex(pd.DataFrame([
            {"dataset_file_uid": "a", "source_entry": index,
             "n_candidates": size, "n_voxels": size}
            for index, size in enumerate([2, 0, 3])
        ]))
        sampler = NodeBudgetBatchSampler(
            references, sizes, "gravnet", max_events=10, max_nodes=10
        )
        self.assertEqual(
            [[item.source_entry for item in batch] for batch in sampler], [[0, 2]]
        )
        self.assertEqual(len(sampler), 1)

    def test_reference_objects_are_resolved_by_dataloader(self):
        references = [
            EventReference("a", "unused", index, 0, "CC_numu", "mc")
            for index in range(3)
        ]
        sizes = EventSizeIndex(pd.DataFrame([
            {"dataset_file_uid": "a", "source_entry": index,
             "n_candidates": 2, "n_voxels": 1}
            for index in range(3)
        ]))
        dataset = SciFiHitEventDataset(references=None, reader=FakeReader())
        loader = DataLoader(
            dataset,
            batch_sampler=NodeBudgetBatchSampler(
                references, sizes, "gravnet", max_events=2, max_nodes=10
            ),
            collate_fn=GravNetClassificationCollator(normalization()),
            num_workers=0,
        )
        batches = list(loader)
        self.assertEqual([batch["event_id"].tolist() for batch in batches], [[0, 1], [2]])

    def test_streaming_moments_match_numpy(self):
        values = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        moments = RunningMoments(3)
        moments.update(values[:1])
        moments.update(values[1:])
        np.testing.assert_allclose(moments.mean, values.mean(axis=0))
        np.testing.assert_allclose(moments.scale, values.std(axis=0))

    def test_metrics_are_computed_from_global_confusion_counts(self):
        import torch

        labels = torch.tensor([0, 0, 1, 1, 2])
        predictions = torch.tensor([0, 1, 1, 1, 0])
        matrix = confusion_matrix(labels, predictions, 3)
        np.testing.assert_array_equal(
            matrix.numpy(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
        )
        metrics = metrics_from_confusion(matrix)
        self.assertAlmostEqual(float(metrics["accuracy"]), 0.6)
        self.assertAlmostEqual(float(metrics["balanced_accuracy"]), 0.5)


if __name__ == "__main__":
    unittest.main()
