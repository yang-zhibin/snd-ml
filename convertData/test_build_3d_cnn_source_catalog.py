import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_3d_cnn_source_catalog import inspect_product_pair


class ProductPairInspectionTest(unittest.TestCase):
    @staticmethod
    def write_tree(path: Path, tree_name: str, entries: int) -> None:
        with uproot.recreate(path) as root_file:
            tree = root_file.mktree(tree_name, {"eventId": np.int64})
            tree.extend({"eventId": np.arange(entries, dtype=np.int64)})

    def test_aligned_empty_pair_is_classified_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "feature.root"
            hit3d = Path(directory) / "hit3d.root"
            self.write_tree(feature, "sndData", 0)
            self.write_tree(hit3d, "hit3D", 0)

            state = inspect_product_pair(
                (str(feature), str(hit3d)), "sndData", "hit3D",
            )

            self.assertEqual(state, (True, True, True, 0, 0, True))

    def test_nonempty_pair_remains_processable(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "feature.root"
            hit3d = Path(directory) / "hit3d.root"
            self.write_tree(feature, "sndData", 3)
            self.write_tree(hit3d, "hit3D", 3)

            state = inspect_product_pair(
                (str(feature), str(hit3d)), "sndData", "hit3D",
            )

            self.assertEqual(state, (True, True, True, 3, 3, False))

    def test_missing_product_is_not_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "feature.root"
            self.write_tree(feature, "sndData", 0)

            state = inspect_product_pair(
                (str(feature), str(Path(directory) / "missing.root")),
                "sndData",
                "hit3D",
            )

            self.assertEqual(state, (True, False, False, -1, -1, False))

    def test_available_uninspected_pair_keeps_unknown_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "feature.root"
            hit3d = Path(directory) / "hit3d.root"
            self.write_tree(feature, "sndData", 0)
            self.write_tree(hit3d, "hit3D", 0)

            state = inspect_product_pair(
                (str(feature), str(hit3d)),
                "sndData",
                "hit3D",
                inspect_entries=False,
            )

            self.assertEqual(state, (True, True, True, -1, -1, False))


if __name__ == "__main__":
    unittest.main()
