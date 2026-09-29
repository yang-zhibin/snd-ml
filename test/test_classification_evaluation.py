import sys
from pathlib import Path
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d" / "src"))
sys.path.insert(0, str(REPO / "train_3d"))

from train_3d.classification_evaluation import (
    add_observations, empty_additive, finalize_additive,
)
from merge_classifier_evaluation import merge_payloads


class ClassificationEvaluationTest(unittest.TestCase):
    def shard(self, index, start, labels, predictions, losses):
        additive = empty_additive(3)
        add_observations(
            additive, torch.tensor(labels), torch.tensor(predictions),
            torch.tensor(losses), torch.tensor([2] * len(labels)), 3,
        )
        return {
            "class_names": ["a", "b", "c"], "cv_fold": 0,
            "num_shards": 2, "shard_index": index,
            "reference_start": start, "reference_stop": start + len(labels),
            "reference_count": len(labels), "checkpoint_sha256": "checkpoint",
            "config_sha256": "config", "additive": additive,
            "grouped": {"domain": {}, "density_bin": {}, "node_count_bin": {}},
            "skipped_empty_events": 0, "wall_seconds": 1.0,
        }

    def test_unequal_shards_match_single_process_metrics(self):
        left = self.shard(0, 0, [0], [0], [0.2])
        right = self.shard(1, 1, [0, 1, 1, 2], [1, 1, 2, 2], [1, 2, 3, 4])
        merged = merge_payloads([right, left])
        single = empty_additive(3)
        add_observations(
            single, torch.tensor([0, 0, 1, 1, 2]),
            torch.tensor([0, 1, 1, 2, 2]),
            torch.tensor([0.2, 1, 2, 3, 4]), torch.tensor([2] * 5), 3,
        )
        expected = finalize_additive(single, ["a", "b", "c"])
        self.assertEqual(merged["global"]["confusion"], expected["confusion"])
        self.assertAlmostEqual(merged["global"]["loss"], 2.04, places=6)
        self.assertAlmostEqual(merged["global"]["macro_f1"], expected["macro_f1"])

    def test_missing_shard_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_payloads([self.shard(0, 0, [0], [0], [1.0])])


if __name__ == "__main__":
    unittest.main()
