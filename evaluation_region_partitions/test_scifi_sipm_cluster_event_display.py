#!/usr/bin/env python3
"""Unit tests for SciFi plane clustering without ROOT dependencies."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from scifi_sipm_cluster_event_display import (
    cluster_event_hits,
    cluster_plane_hits,
    layer_channel,
)


@dataclass(frozen=True)
class FakeHit:
    station: int = 1
    orientation: str = "horizontal"
    mat: int = 0
    sipm: int = 0
    channel: int = 0
    qdc: float = 1.0
    valid: bool = True


def hit_at(index: int, **changes) -> FakeHit:
    values = {
        "mat": index // 512,
        "sipm": (index % 512) // 128,
        "channel": index % 128,
    }
    values.update(changes)
    return FakeHit(**values)


class SciFiSipmClusterTest(unittest.TestCase):
    def test_layer_channel_spans_all_hardware_boundaries(self):
        self.assertEqual(layer_channel(hit_at(0)), 0)
        self.assertEqual(layer_channel(hit_at(63)), 63)
        self.assertEqual(layer_channel(hit_at(64)), 64)
        self.assertEqual(layer_channel(hit_at(127)), 127)
        self.assertEqual(layer_channel(hit_at(128)), 128)
        self.assertEqual(layer_channel(hit_at(511)), 511)
        self.assertEqual(layer_channel(hit_at(512)), 512)
        self.assertEqual(layer_channel(hit_at(1535)), 1535)

    def test_die_sipm_and_mat_boundaries_do_not_split_clusters(self):
        for indices in ((62, 63, 64, 65), (126, 127, 128, 129), (510, 511, 512, 513)):
            with self.subTest(indices=indices):
                result = cluster_plane_hits([hit_at(index) for index in indices], 1, "horizontal")
                self.assertEqual(len(result.clusters), 1)
                self.assertEqual(result.clusters[0].channels, indices)

    def test_a_real_missing_channel_splits_clusters(self):
        result = cluster_plane_hits(
            [hit_at(index) for index in (20, 21, 23, 24)], 1, "horizontal"
        )
        self.assertEqual([cluster.channels for cluster in result.clusters], [(20, 21), (23, 24)])

    def test_invalid_and_nonfinite_hits_are_not_clustered(self):
        result = cluster_plane_hits(
            [
                hit_at(20),
                hit_at(21, valid=False),
                hit_at(22, qdc=math.nan),
                hit_at(23),
            ],
            1,
            "horizontal",
        )
        self.assertEqual([cluster.channels for cluster in result.clusters], [(20,), (23,)])
        self.assertEqual(result.invalid_hit_count, 1)
        self.assertEqual(result.nonfinite_valid_hit_count, 1)

    def test_largest_qdc_and_largest_hit_clusters_are_distinct(self):
        result = cluster_plane_hits(
            [
                hit_at(1, qdc=1.0),
                hit_at(2, qdc=1.0),
                hit_at(3, qdc=1.0),
                hit_at(10, qdc=10.0),
            ],
            1,
            "horizontal",
        )
        self.assertEqual(result.largest_hit_cluster_id, 0)
        self.assertEqual(result.largest_qdc_cluster_id, 1)

    def test_largest_cluster_ties_prefer_lower_channel(self):
        result = cluster_plane_hits(
            [hit_at(10, qdc=2.0), hit_at(20, qdc=2.0)], 1, "horizontal"
        )
        self.assertEqual(result.largest_qdc_cluster_id, 0)
        self.assertEqual(result.largest_hit_cluster_id, 0)

    def test_duplicate_plane_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate layer channel 10"):
            cluster_plane_hits([hit_at(10), hit_at(10)], 1, "horizontal")

    def test_event_result_has_ten_planes_and_exact_closure(self):
        hits = [hit_at(1, qdc=2.5), hit_at(2, qdc=-1.0)]
        results = cluster_event_hits(hits)
        self.assertEqual(len(results), 10)
        audit = results[(1, "horizontal")].audit()
        self.assertEqual(audit["valid_finite_hit_count"], 2)
        self.assertEqual(audit["closure"]["clustered_hit_count"], 2)
        self.assertEqual(audit["closure"]["hit_count_difference"], 0)
        self.assertEqual(audit["closure"]["clustered_qdc_sum"], 2.5)
        self.assertEqual(audit["closure"]["qdc_sum_difference"], 0.0)

    def test_invalid_hardware_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "channel outside"):
            layer_channel(FakeHit(channel=128))


if __name__ == "__main__":
    unittest.main()
