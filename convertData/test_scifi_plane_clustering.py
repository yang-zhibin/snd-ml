#!/usr/bin/env python3
"""Tests for ROOT-free SciFi orientation cluster features."""
import math
import unittest

from scifi_plane_clustering import (
    UNDEFINED_FLOAT, UNDEFINED_INT, cluster_feature_payload,
    cluster_scifi_hits, scalar_feature_dtypes,
)

def hit(channel, qdc=1.0, station=1, vertical=True):
    return {"station":station, "isVertical":vertical,
            "layer_channel":channel, "qdc":qdc}

class SciFiClusteringTest(unittest.TestCase):
    def test_hardware_boundaries_do_not_split(self):
        for channels in ((62,63,64,65),(126,127,128,129),(510,511,512,513)):
            plane=cluster_scifi_hits([hit(x) for x in channels])["planes"][(1,True)]
            self.assertEqual(len(plane["clusters"]),1)
            self.assertEqual(plane["clusters"][0]["channels"],channels)

    def test_real_gap_splits_and_orientation_ranking(self):
        analysis=cluster_scifi_hits([
            hit(10,2),hit(11,2),hit(30,8,station=3),
            hit(40,4),hit(50,4),
        ])
        plane=analysis["planes"][(1,True)]
        self.assertEqual(len(plane["clusters"]),3)
        ranked=analysis["orientations"]["vertical"]["ranked_clusters"]
        self.assertEqual([(x["station"],x["first_channel"]) for x in ranked],
                         [(3,30),(1,10),(1,40),(1,50)])

    def test_ties_use_hits_station_and_channel(self):
        ranked=cluster_scifi_hits([
            hit(30,4,station=2),hit(10,2),hit(11,2),hit(20,4),
        ])["orientations"]["vertical"]["ranked_clusters"]
        self.assertEqual([(x["station"],x["first_channel"]) for x in ranked],
                         [(1,10),(1,20),(2,30)])

    def test_shape_and_fractions(self):
        values=cluster_feature_payload([
            hit(1,1),hit(2,2),hit(3,3),hit(10,2),hit(20,1),
        ])["scalars"]
        self.assertEqual(values["scifi_vertical_cluster1_nhits"],3)
        self.assertEqual(values["scifi_vertical_cluster1_qdc"],6)
        self.assertEqual(values["scifi_vertical_cluster1_qdc_per_hit"],2)
        self.assertAlmostEqual(values["scifi_vertical_cluster1_qdc_std"],
                               math.sqrt(2.0/3.0))
        self.assertAlmostEqual(values["scifi_vertical_cluster1_qdc_cv"],
                               math.sqrt(2.0/3.0)/2.0)
        self.assertEqual(values["scifi_vertical_cluster1_qdc_roughness"],0.5)
        self.assertEqual(values["scifi_vertical_cluster1_plane_hit_fraction"],0.6)
        self.assertEqual(values["scifi_vertical_cluster1_orientation_qdc_fraction"],2.0/3.0)
        self.assertEqual(values["scifi_vertical_other_cluster_count"],1)
        self.assertEqual(values["scifi_vertical_other_cluster_nhits"],1)

    def test_top_two_same_station(self):
        values=cluster_feature_payload([hit(1,3),hit(10,2)])["scalars"]
        self.assertEqual(values["scifi_vertical_top2_same_station"],1)
        self.assertEqual(values["scifi_vertical_top2_station_delta"],0)
        self.assertEqual(values["scifi_vertical_top2_qdc_downstream_asymmetry"],0)
        self.assertEqual(values["scifi_vertical_top2_hit_downstream_asymmetry"],0)

    def test_downstream_asymmetry_and_ratios(self):
        values=cluster_feature_payload([
            hit(1,4,station=1),hit(2,4,station=1),hit(20,3,station=4),
        ])["scalars"]
        self.assertEqual(values["scifi_vertical_cluster2_to_cluster1_hit_ratio"],0.5)
        self.assertEqual(values["scifi_vertical_cluster2_to_cluster1_qdc_ratio"],3.0/8.0)
        self.assertEqual(values["scifi_vertical_top2_station_delta"],3)
        self.assertEqual(values["scifi_vertical_top2_qdc_downstream_asymmetry"],-5.0/11.0)
        self.assertEqual(values["scifi_vertical_top2_hit_downstream_asymmetry"],-1.0/3.0)

    def test_profiles_moments_slope_and_curvature(self):
        values=cluster_feature_payload([
            hit(10+s*10+offset,1,station=s) for s in range(1,6)
            for offset in range(s)
        ])["scalars"]
        self.assertAlmostEqual(sum(values["scifi_vertical_hit_frac{}".format(s)]
                                   for s in range(1,6)),1.0)
        self.assertAlmostEqual(values["scifi_vertical_hit_mean_station"],55.0/15.0)
        self.assertGreater(values["scifi_vertical_hit_station_slope"],0)
        self.assertAlmostEqual(values["scifi_vertical_hit_station_curvature"],0)

    def test_effective_counts_and_event_closure(self):
        values=cluster_feature_payload([
            hit(1,2),hit(10,2),hit(20,4,vertical=False),
        ])["scalars"]
        self.assertEqual(values["scifi_vertical_effective_hit_cluster_count"],2)
        self.assertEqual(values["scifi_vertical_effective_qdc_cluster_count"],2)
        self.assertEqual(values["scifi_cluster_total_count"],3)
        self.assertEqual(values["scifi_cluster_total_nhits"],3)
        self.assertEqual(values["scifi_cluster_total_qdc"],8)
        self.assertEqual(values["scifi_vertical_cluster_fraction"],2.0/3.0)

    def test_empty_and_one_cluster_conventions(self):
        empty=cluster_feature_payload([])["scalars"]
        self.assertEqual(empty["scifi_vertical_cluster1_exists"],0)
        self.assertEqual(empty["scifi_vertical_cluster1_station"],UNDEFINED_INT)
        self.assertEqual(empty["scifi_vertical_cluster1_qdc_std"],UNDEFINED_FLOAT)
        self.assertEqual(empty["scifi_vertical_hit_mean_station"],UNDEFINED_FLOAT)
        one=cluster_feature_payload([hit(1,0)])["scalars"]
        self.assertEqual(one["scifi_vertical_cluster1_exists"],1)
        self.assertEqual(one["scifi_vertical_cluster1_qdc_cv"],0)
        self.assertEqual(one["scifi_vertical_cluster2_exists"],0)
        self.assertEqual(one["scifi_vertical_top2_station_delta"],UNDEFINED_INT)

    def test_schema_duplicates_nonfinite_and_duplicate_channel(self):
        payload=cluster_feature_payload([hit(1)])
        self.assertEqual(set(payload["scalars"]),
                         {name for name,_ in scalar_feature_dtypes()})
        self.assertNotIn("vectors",payload)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            cluster_scifi_hits([hit(1),hit(1)])
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            cluster_scifi_hits([hit(1,float("nan"))])

if __name__ == "__main__":
    unittest.main()
