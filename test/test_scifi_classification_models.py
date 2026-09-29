import importlib.util
import sys
import unittest
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))

from train_3d.models.pyg_gravnet_classifier import PyGGravNetClassifier


class SciFiClassificationModelTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyG unavailable")
    def test_native_gravnet_cpu_forward_backward(self):
        model = PyGGravNetClassifier(
            in_channels=7, hidden_channels=16, num_classes=7, blocks=2,
            space_dimensions=3, propagate_dimensions=8, k=3, dropout=0.0,
        )
        features = torch.randn(13, 7)
        batch = torch.tensor([0] * 6 + [1] * 7)
        logits = model(features, batch)
        self.assertEqual(tuple(logits.shape), (2, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 5]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(importlib.util.find_spec("MinkowskiEngine"), "MinkowskiEngine unavailable")
    def test_two_view_sparse_resnet_handles_missing_views(self):
        from train_3d.models.two_view_sparse_resnet_classifier import (
            TwoViewSparseResNetClassifier,
        )

        model = TwoViewSparseResNetClassifier(
            in_channels=2, num_classes=7, channels=(4, 8, 12, 16), dropout=0.0
        )
        vertical_coords = torch.tensor([
            [0, 0, 2], [0, 1, 4], [0, 2, 6], [0, 3, 8],
        ], dtype=torch.int32)
        horizontal_coords = torch.tensor([
            [1, 0, 3], [1, 1, 5], [1, 2, 7], [1, 3, 9],
        ], dtype=torch.int32)
        logits = model(
            vertical_coords, torch.randn(4, 2),
            horizontal_coords, torch.randn(4, 2),
            torch.tensor([True, False]), torch.tensor([False, True]),
        )
        self.assertEqual(tuple(logits.shape), (2, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyG unavailable")
    def test_gravnet_cpu_forward_backward_variable_events(self):
        model = PyGGravNetClassifier(
            in_channels=5, hidden_channels=16, num_classes=7, blocks=2,
            space_dimensions=3, propagate_dimensions=8, k=3, dropout=0.0,
        )
        features = torch.randn(12, 5)
        batch = torch.tensor([0] * 5 + [1] * 7)
        logits = model(features, batch)
        self.assertEqual(tuple(logits.shape), (2, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 6]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    @unittest.skipUnless(importlib.util.find_spec("MinkowskiEngine"), "MinkowskiEngine unavailable")
    def test_sparse_resnet_one_voxel_event_forward_backward(self):
        from train_3d.models.sparse_resnet_classifier import SparseResNetClassifier

        model = SparseResNetClassifier(
            in_channels=2, num_classes=7, channels=(4, 8, 12, 16), dropout=0.0
        )
        coordinates = torch.tensor([[0, 4, 8, 12]], dtype=torch.int32)
        features = torch.randn(1, 2)
        logits = model(coordinates, features)
        self.assertEqual(tuple(logits.shape), (1, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    @unittest.skipUnless(importlib.util.find_spec("MinkowskiEngine"), "MinkowskiEngine unavailable")
    def test_sparse_resnet_mixed_event_sizes_are_batch_independent(self):
        from train_3d.models.sparse_resnet_classifier import SparseResNetClassifier

        torch.manual_seed(7)
        model = SparseResNetClassifier(
            in_channels=2, num_classes=7, channels=(4, 8, 12, 16), dropout=0.0
        ).eval()
        first_coords = torch.tensor([[0, 4, 8, 12]], dtype=torch.int32)
        first_features = torch.randn(1, 2)
        second_xyz = torch.tensor([
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
            [8, 8, 8], [10, 8, 8], [8, 10, 8], [8, 8, 10],
        ], dtype=torch.int32)
        together_coords = torch.cat([
            first_coords,
            torch.cat([
                torch.ones((len(second_xyz), 1), dtype=torch.int32), second_xyz
            ], dim=1),
        ])
        together_features = torch.cat([first_features, torch.randn(8, 2)])

        with torch.no_grad():
            alone = model(first_coords, first_features)[0]
            together_logits = model(together_coords, together_features)
        self.assertEqual(tuple(together_logits.shape), (2, 7))
        torch.testing.assert_close(
            alone, together_logits[0], rtol=1e-5, atol=1e-6
        )

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyG unavailable")
    def test_gravnet_batch_prevents_cross_event_neighbors(self):
        torch.manual_seed(4)
        model = PyGGravNetClassifier(
            in_channels=5, hidden_channels=12, num_classes=7, blocks=2,
            space_dimensions=3, propagate_dimensions=8, k=3, dropout=0.0,
        ).eval()
        first = torch.randn(6, 5)
        second = torch.randn(8, 5) * 100
        with torch.no_grad():
            alone = model(first, torch.zeros(6, dtype=torch.long))[0]
            together = model(
                torch.cat([first, second]),
                torch.tensor([0] * 6 + [1] * 8),
            )[0]
        torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("MinkowskiEngine"), "MinkowskiEngine unavailable")
    def test_sparse_resnet_cpu_forward_backward(self):
        from train_3d.models.sparse_resnet_classifier import SparseResNetClassifier

        model = SparseResNetClassifier(
            in_channels=2, num_classes=7, channels=(8, 16, 24, 32), dropout=0.0
        )
        coordinates = torch.tensor([
            [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3],
            [1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2], [1, 3, 3, 3],
        ], dtype=torch.int32)
        features = torch.randn(len(coordinates), 2)
        logits = model(coordinates, features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 5]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
