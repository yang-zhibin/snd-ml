import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))

from train_3d.data.data_driven_fold_dataset import (
    DeterministicStratifiedValidationEvents, DeterministicTrainingReferenceSampler,
    HierarchicalTrainingSampler, NaturalValidationEvents, RankSlicedReferences,
    SampledTrainingEvents, luminosity_exposure,
)


class DataDrivenFoldDatasetTest(unittest.TestCase):
    def make_views(self, directory):
        density = directory / "density.npz"
        filtered = directory / "filtered.npz"
        np.savez_compressed(
            density,
            bin_0=np.asarray([0], dtype=np.int32),
            bin_1=np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
            invalid=np.asarray([9], dtype=np.int32),
        )
        np.savez_compressed(filtered, entries=np.asarray([4, 9], dtype=np.int32))
        rows = [
            {
                "cv_fold": 0, "split": "train", "fold": 1,
                "dataset_file_uid": "cc", "hit3d_path": "nu.root",
                "class_id": 0, "class_name": "CC_numu", "domain": "mc",
                "n_events": 2, "event_filter": "pdgCode == 14",
                "entry_index_path": str(filtered), "density_index_path": "",
                "sampling_strategy": "natural", "lumi_source_uid": "nu_file",
                "lumi_per_file": 2.0, "dataset_key": "nu",
            },
            {
                "cv_fold": 0, "split": "train", "fold": 2,
                "dataset_file_uid": "dis", "hit3d_path": "dis.root",
                "class_id": 7, "class_name": "muon_DIS", "domain": "data",
                "n_events": 10, "event_filter": "true", "entry_index_path": "",
                "density_index_path": str(density),
                "sampling_strategy": "uniform_populated_density_bins",
                "lumi_source_uid": "data_file", "lumi_per_file": 0.5,
                "dataset_key": "data",
            },
            {
                "cv_fold": 0, "split": "validation", "fold": 0,
                "dataset_file_uid": "val", "hit3d_path": "val.root",
                "class_id": 5, "class_name": "neutron", "domain": "mc",
                "n_events": 3, "event_filter": "true", "entry_index_path": "",
                "density_index_path": "", "sampling_strategy": "natural",
                "lumi_source_uid": "n_file", "lumi_per_file": 3.0,
                "dataset_key": "neutron",
            },
        ]
        return pd.DataFrame(rows)

    def test_hierarchy_density_and_physical_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            sampler = HierarchicalTrainingSampler(
                views, 0, seed=17,
                class_probabilities="uniform",
            )
            sampled = sampler.sample(20000)
            cc = [item for item in sampled if item.class_name == "CC_numu"]
            dis = [item for item in sampled if item.class_name == "muon_DIS"]
            self.assertAlmostEqual(len(cc) / len(sampled), 0.5, delta=0.02)
            self.assertEqual({item.source_entry for item in cc}, {4, 9})
            low = sum(item.density_bin == "bin_0" for item in dis) / len(dis)
            self.assertAlmostEqual(low, 0.5, delta=0.03)
            self.assertNotIn(9, {item.source_entry for item in dis})

    def test_muondis_density_policy_can_be_natural_or_uniform(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            uniform = HierarchicalTrainingSampler(
                views, 0, seed=4,
                class_probabilities={"CC_numu": 0.0, "muon_DIS": 1.0},
                muondis_density_probabilities="uniform",
            ).sample_epoch(20000, 0)
            natural = HierarchicalTrainingSampler(
                views, 0, seed=4,
                class_probabilities={"CC_numu": 0.0, "muon_DIS": 1.0},
                muondis_density_probabilities=None,
            ).sample_epoch(20000, 0)
            uniform_low = sum(item.density_bin == "bin_0" for item in uniform) / len(uniform)
            natural_low = sum(item.density_bin == "bin_0" for item in natural) / len(natural)
            self.assertAlmostEqual(uniform_low, 0.5, delta=0.02)
            self.assertAlmostEqual(natural_low, 1.0 / 9.0, delta=0.02)

    def test_validation_is_natural_and_fold_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            validation = NaturalValidationEvents(views, 0)
            self.assertEqual(len(validation), 3)
            refs = list(validation)
            self.assertEqual([item.source_entry for item in refs], [0, 1, 2])
            self.assertEqual({item.dataset_file_uid for item in refs}, {"val"})
            self.assertEqual(validation[-1].source_entry, 2)
            with self.assertRaises(IndexError):
                _ = validation[3]

    def test_invalid_density_event_remains_in_natural_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            density_path = Path(tmp) / "density.npz"
            np.savez_compressed(
                density_path, bin_0=np.arange(9, dtype=np.int64),
                invalid=np.asarray([9], dtype=np.int64),
            )
            row = pd.DataFrame([{
                "cv_fold": 0, "split": "validation", "fold": 0,
                "dataset_file_uid": "dis_validation", "hit3d_path": "dis.root",
                "class_id": 7, "class_name": "muon_DIS", "domain": "data",
                "n_events": 10, "event_filter": "true", "entry_index_path": "",
                "density_index_path": str(density_path),
                "sampling_strategy": "natural",
            }])
            validation = NaturalValidationEvents(row, 0)
            self.assertEqual([item.source_entry for item in validation], list(range(10)))
            self.assertEqual(validation[0].density_bin, "bin_0")
            self.assertEqual(validation[9].density_bin, "invalid")

    def test_stratified_validation_is_exact_fixed_and_excludes_empty_events(self):
        class_names = ("a", "b", "c")
        rows = []
        for class_id, class_name in enumerate(class_names):
            rows.append({
                "cv_fold": 0, "split": "validation", "fold": 0,
                "dataset_file_uid": f"val_{class_name}",
                "hit3d_path": f"{class_name}.root", "class_id": class_id,
                "class_name": class_name, "domain": "mc", "n_events": 8,
                "event_filter": "true", "entry_index_path": "",
                "density_index_path": "", "sampling_strategy": "natural",
            })
        views = pd.DataFrame(rows)
        accept = lambda reference: reference.source_entry != 0
        first = DeterministicStratifiedValidationEvents(
            views, 0, class_names, total_size=12, seed=42, accept=accept
        )
        second = DeterministicStratifiedValidationEvents(
            views, 0, class_names, total_size=12, seed=42, accept=accept
        )
        self.assertEqual(list(first), list(second))
        self.assertEqual(len(first), 12)
        self.assertEqual(
            {name: sum(item.class_name == name for item in first)
             for name in class_names},
            {"a": 4, "b": 4, "c": 4},
        )
        self.assertEqual(len({
            (item.dataset_file_uid, item.source_entry) for item in first
        }), 12)
        self.assertNotIn(0, {item.source_entry for item in first})
        # Round-robin output makes even the first diagnostic group balanced.
        self.assertEqual([item.class_name for item in list(first)[:3]], list(class_names))

    def test_stratified_validation_fails_when_a_class_quota_is_unavailable(self):
        rows = pd.DataFrame([
            {
                "cv_fold": 0, "split": "validation", "fold": 0,
                "dataset_file_uid": name, "hit3d_path": f"{name}.root",
                "class_id": index, "class_name": name, "domain": "mc",
                "n_events": count, "event_filter": "true",
                "entry_index_path": "", "density_index_path": "",
                "sampling_strategy": "natural",
            }
            for index, (name, count) in enumerate((("a", 5), ("b", 1)))
        ])
        with self.assertRaisesRegex(ValueError, "fewer than its requested quota"):
            DeterministicStratifiedValidationEvents(
                rows, 0, ("a", "b"), total_size=6, seed=1
            )

    def test_epoch_sampling_is_reproducible_and_call_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            first = HierarchicalTrainingSampler(views, 0, seed=1701)
            expected = first.sample_epoch(100, 3)
            first.sample(250)
            self.assertEqual(expected, first.sample_epoch(100, 3))
            second = HierarchicalTrainingSampler(views, 0, seed=1701)
            self.assertEqual(expected, second.sample_epoch(100, 3))
            self.assertNotEqual(expected, second.sample_epoch(100, 4))
            with self.assertRaisesRegex(ValueError, "epoch cannot be negative"):
                second.sample_epoch(1, -1)

    def test_ddp_reference_sequences_are_disjoint_and_equal_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            sampler = HierarchicalTrainingSampler(views, 0, seed=18)
            ranks = [
                list(DeterministicTrainingReferenceSampler(
                    sampler, epoch_size=10, epoch=2, rank=rank, world_size=3
                ))
                for rank in range(3)
            ]
            self.assertEqual([len(items) for items in ranks], [4, 4, 4])
            expected = sampler.sample_epoch(12, 2)
            interleaved = [ranks[index % 3][index // 3] for index in range(12)]
            self.assertEqual(interleaved, expected)
            validation = NaturalValidationEvents(views, 0)
            slices = [list(RankSlicedReferences(validation, rank, 2)) for rank in range(2)]
            self.assertEqual(
                sorted(item.source_entry for items in slices for item in items),
                [0, 1, 2],
            )

    def test_indexable_training_epoch_can_be_advanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            sampler = HierarchicalTrainingSampler(views, 0, seed=91)
            events = SampledTrainingEvents(sampler, epoch_size=30, epoch=0)
            epoch_zero = list(events.references)
            events.set_epoch(1)
            self.assertEqual(len(events), 30)
            self.assertNotEqual(epoch_zero, events.references)
            events.set_epoch(0)
            self.assertEqual(epoch_zero, events.references)

    def test_muondis_is_file_proportional_within_each_density_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            small = directory / "small.npz"
            large = directory / "large.npz"
            np.savez_compressed(small, bin_0=np.arange(10, dtype=np.int64))
            np.savez_compressed(large, bin_0=np.arange(10, 100, dtype=np.int64))
            rows = []
            for uid, path, count in (("small", small, 10), ("large", large, 90)):
                rows.append({
                    "cv_fold": 0, "split": "train", "fold": 1,
                    "dataset_file_uid": uid, "hit3d_path": f"{uid}.root",
                    "class_id": 7, "class_name": "muon_DIS", "domain": "data",
                    "n_events": count, "event_filter": "true",
                    "entry_index_path": "", "density_index_path": str(path),
                    "sampling_strategy": "uniform_populated_density_bins",
                })
            sampler = HierarchicalTrainingSampler(pd.DataFrame(rows), 0, seed=7)
            sampled = sampler.sample_epoch(20000, 0)
            small_fraction = sum(item.dataset_file_uid == "small" for item in sampled) / len(sampled)
            self.assertAlmostEqual(small_fraction, 0.1, delta=0.015)

    def test_requested_domain_balance_is_applied_inside_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = []
            for domain in ("data", "mc"):
                index = directory / f"{domain}.npz"
                np.savez_compressed(index, bin_0=np.arange(20, dtype=np.int64))
                rows.append({
                    "cv_fold": 0, "split": "train", "fold": 1,
                    "dataset_file_uid": domain, "hit3d_path": f"{domain}.root",
                    "class_id": 7, "class_name": "muon_DIS", "domain": domain,
                    "n_events": 20, "event_filter": "true",
                    "entry_index_path": "", "density_index_path": str(index),
                    "sampling_strategy": "uniform_populated_density_bins",
                })
            sampler = HierarchicalTrainingSampler(
                pd.DataFrame(rows), 0, seed=11,
                domain_probabilities={"muon_DIS": {"data": 0.25, "mc": 0.75}},
            )
            sampled = sampler.sample_epoch(20000, 0)
            data_fraction = sum(item.domain == "data" for item in sampled) / len(sampled)
            self.assertAlmostEqual(data_fraction, 0.25, delta=0.02)

    def test_unspecified_domain_balance_uses_natural_eligible_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = []
            for domain, count in (("data", 10), ("mc", 90)):
                index = directory / f"{domain}.npz"
                np.savez_compressed(index, bin_0=np.arange(count, dtype=np.int64))
                rows.append({
                    "cv_fold": 0, "split": "train", "fold": 1,
                    "dataset_file_uid": domain, "hit3d_path": f"{domain}.root",
                    "class_id": 7, "class_name": "muon_DIS", "domain": domain,
                    "n_events": count, "event_filter": "true",
                    "entry_index_path": "", "density_index_path": str(index),
                    "sampling_strategy": "uniform_populated_density_bins",
                })
            sampler = HierarchicalTrainingSampler(
                pd.DataFrame(rows), 0, seed=12,
                class_probabilities="uniform", domain_probabilities=None,
                muondis_density_probabilities="uniform",
            )
            sampled = sampler.sample_epoch(20000, 0)
            data_fraction = sum(item.domain == "data" for item in sampled) / len(sampled)
            self.assertAlmostEqual(data_fraction, 0.1, delta=0.015)

    def test_luminosity_is_deduplicated(self):
        rows = pd.DataFrame([
            {"dataset_key": "nu", "lumi_source_uid": "file_a", "lumi_per_file": 2.0},
            {"dataset_key": "nu", "lumi_source_uid": "file_a", "lumi_per_file": 2.0},
            {"dataset_key": "nu", "lumi_source_uid": "file_b", "lumi_per_file": 3.0},
        ])
        exposure = luminosity_exposure(rows, ["dataset_key"])
        self.assertEqual(exposure.iloc[0]["lumi_per_file"], 5.0)


if __name__ == "__main__":
    unittest.main()
