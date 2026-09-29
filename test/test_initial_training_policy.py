import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))
sys.path.insert(0, str(REPO / "train_3d"))

from train_classifier import validate_resume_config
from train_3d.classification_module import classification_loss_and_accuracy
from train_3d.data.classification_labels import validate_scifi_classification_config


class InitialTrainingPolicyTest(unittest.TestCase):
    def load_config(self, name):
        with (REPO / "train_3d/configs" / name).open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_both_active_configs_enforce_the_frozen_policy(self):
        for name in (
            "classifier_pyg_gravnet_v1.yaml",
            "classifier_sparse_resnet_v1.yaml",
            "classifier_native_scifi_gravnet_v1.yaml",
            "classifier_two_view_sparse_resnet_v1.yaml",
        ):
            with self.subTest(config=name):
                config = self.load_config(name)
                labels = validate_scifi_classification_config(config)
                self.assertEqual(len(labels), 7)
                self.assertNotIn("CC_nutau", labels)

    def test_policy_validator_rejects_accidental_reweighting(self):
        config = self.load_config("classifier_pyg_gravnet_v1.yaml")
        config["objective"]["class_weights"] = [1.0] * 7
        with self.assertRaisesRegex(ValueError, "unweighted"):
            validate_scifi_classification_config(config)

    def test_loss_is_plain_unweighted_cross_entropy(self):
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
        labels = torch.tensor([0, 1])
        loss, _ = classification_loss_and_accuracy(logits, labels, 2)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(loss, expected)

    def resume_config(self):
        return {
            "model_version": "pyg_gravnet_v1",
            "dataset_version": "data_driven_3d_cnn_v1",
            "class_names": ["a", "b"],
            "excluded_classes": ["CC_nutau"],
            "model": {"family": "pyg_gravnet", "in_channels": 5},
            "representation": {
                "version": "scifi_xyz_qdc_dettype_v1",
                "voxel_size": [1.0, 1.0, 1.0],
                "qdc_transform": "log1p",
            },
            "objective": {"name": "cross_entropy", "class_weights": None},
            "data": {"cv_fold": 0},
            "runtime": {"seed": 42},
            "provenance": {
                "fold_views_sha256": "folds",
                "validation_report_sha256": "validation",
                "normalization_sha256": "normalization",
                "representation_summary_sha256": "representation",
                "event_size_manifest_sha256": "sizes",
            },
        }

    def test_resume_accepts_an_identical_training_contract(self):
        existing = self.resume_config()
        validate_resume_config(existing, copy.deepcopy(existing))

    def test_resume_rejects_changed_model_or_input_contract(self):
        existing = self.resume_config()
        changes = {
            "representation": lambda value: value["representation"].update(
                version="incompatible_v2"
            ),
            "model": lambda value: value["model"].update(in_channels=6),
            "event_size_manifest_sha256": lambda value: value["provenance"].update(
                event_size_manifest_sha256="changed"
            ),
            "metadata_index_sha256": lambda value: value["provenance"].update(
                metadata_index_sha256="changed"
            ),
        }
        for expected_key, mutate in changes.items():
            with self.subTest(expected_key=expected_key):
                current = copy.deepcopy(existing)
                mutate(current)
                with self.assertRaisesRegex(ValueError, expected_key):
                    validate_resume_config(existing, current)


if __name__ == "__main__":
    unittest.main()
