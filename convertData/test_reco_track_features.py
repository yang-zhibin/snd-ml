import array
import math
import unittest

from reco_track_features import (
    DETAIL_SCALAR_DTYPES,
    DETAIL_TRACK_VECTOR_DTYPES,
    FLAT_V1,
    FLAT_V2,
    MEASUREMENT_OFFSET_VECTOR_DTYPES,
    MEASUREMENT_VECTOR_DTYPES,
    SCALAR_DTYPES,
    VECTOR_DTYPES,
    decode_measurement_detid,
    fill_reco_track_buffers,
    reset_reco_track_buffers,
    scalar_dtypes_for_schema,
    vector_dtypes_for_schema,
)


class Vec:
    def __init__(self, x, y, z):
        self.xyz = (x, y, z)

    def X(self):
        return self.xyz[0]

    def Y(self):
        return self.xyz[1]

    def Z(self):
        return self.xyz[2]


class Track:
    def __init__(
        self,
        track_type,
        flag=True,
        chi2=4.0,
        ndf=2,
        npoints=3,
        points=None,
        detids=None,
    ):
        self.values = track_type, flag, chi2, ndf, npoints
        self.points = points
        self.detids = detids

    def getTrackType(self):
        return self.values[0]

    def getTrackFlag(self):
        return self.values[1]

    def getChi2(self):
        return self.values[2]

    def getNdf(self):
        return self.values[3]

    def getTrackPoints(self):
        if self.points is not None:
            return self.points
        return range(self.values[4])

    def getRawMeasDetIDs(self):
        if self.detids is not None:
            return self.detids
        return []

    def getStart(self):
        return Vec(1, 2, 3)

    def getStop(self):
        return Vec(4, 5, 6)

    def getTrackMom(self):
        return Vec(2, 4, 2)


def buffers(schema=FLAT_V1):
    scalar_dtypes = scalar_dtypes_for_schema(schema)
    vector_dtypes = vector_dtypes_for_schema(schema)
    scalars = {
        name: array.array(dtype, [0 if dtype == "i" else math.nan])
        for name, dtype in scalar_dtypes.items()
    }
    vectors = {name: [] for name in vector_dtypes}
    return scalars, vectors


class RecoTrackFeaturesTest(unittest.TestCase):
    def test_flat_v1_schema_is_unchanged_by_v2_definitions(self):
        self.assertEqual(scalar_dtypes_for_schema(FLAT_V1), SCALAR_DTYPES)
        self.assertEqual(vector_dtypes_for_schema(FLAT_V1), VECTOR_DTYPES)
        self.assertFalse(set(DETAIL_SCALAR_DTYPES) & set(SCALAR_DTYPES))
        self.assertFalse(set(DETAIL_TRACK_VECTOR_DTYPES) & set(VECTOR_DTYPES))
        self.assertFalse(set(MEASUREMENT_VECTOR_DTYPES) & set(VECTOR_DTYPES))
        self.assertFalse(set(MEASUREMENT_OFFSET_VECTOR_DTYPES) & set(VECTOR_DTYPES))

    def test_absent_collection_has_explicit_sentinels(self):
        scalars, vectors = buffers()
        reset_reco_track_buffers(scalars, vectors, True)
        fill_reco_track_buffers([], scalars, vectors)
        self.assertEqual(scalars["reco_track_branch_available"][0], 1)
        self.assertEqual(scalars["reco_scifi_simple_n_all"][0], 0)
        self.assertEqual(scalars["reco_scifi_simple_best_index"][0], -1)
        self.assertTrue(math.isnan(scalars["reco_scifi_simple_best_chi2ndf"][0]))

    def test_counts_unknown_types_and_failed_fits(self):
        scalars, vectors = buffers()
        reset_reco_track_buffers(scalars, vectors, True)
        fill_reco_track_buffers(
            [Track(1, False), Track(1, True), Track(99, True)], scalars, vectors
        )
        self.assertEqual(scalars["reco_track_n_all"][0], 3)
        self.assertEqual(scalars["reco_track_n_converged"][0], 2)
        self.assertEqual(scalars["reco_track_n_unknown_type"][0], 1)
        self.assertEqual(scalars["reco_scifi_simple_n_all"][0], 2)
        self.assertEqual(scalars["reco_scifi_simple_n_converged"][0], 1)
        self.assertEqual(vectors["reco_track_type"], [1, 1, 99])

    def test_best_track_tie_breaks_on_points_then_input_index(self):
        scalars, vectors = buffers()
        reset_reco_track_buffers(scalars, vectors, True)
        fill_reco_track_buffers(
            [Track(11, chi2=4, ndf=2, npoints=3), Track(11, chi2=6, ndf=3, npoints=5)],
            scalars,
            vectors,
        )
        self.assertEqual(scalars["reco_scifi_hough_best_index"][0], 1)
        self.assertEqual(scalars["reco_scifi_hough_best_npoints"][0], 5)
        self.assertAlmostEqual(scalars["reco_scifi_hough_best_slope_xz"][0], 1)
        self.assertAlmostEqual(scalars["reco_scifi_hough_best_slope_yz"][0], 2)

    def test_invalid_ndf_does_not_create_best_track(self):
        scalars, vectors = buffers()
        reset_reco_track_buffers(scalars, vectors, False)
        fill_reco_track_buffers([Track(15, True, chi2=1, ndf=0)], scalars, vectors)
        self.assertEqual(scalars["reco_track_branch_available"][0], 0)
        self.assertEqual(scalars["reco_combined_hough_has_converged"][0], 1)
        self.assertEqual(scalars["reco_combined_hough_has_best"][0], 0)

    def test_detector_id_decoding_uses_documented_encodings(self):
        self.assertEqual(
            decode_measurement_detid(1_100_123),
            {
                "subsystem": 1,
                "station": 1,
                "plane": -1,
                "view": 1,
                "plane_key": 111,
            },
        )
        self.assertEqual(decode_measurement_detid(12_034)["subsystem"], 2)
        self.assertEqual(decode_measurement_detid(12_034)["plane"], 2)
        self.assertEqual(decode_measurement_detid(24_056)["subsystem"], 3)
        self.assertEqual(decode_measurement_detid(37_012)["subsystem"], 4)
        self.assertEqual(decode_measurement_detid(999)["subsystem"], 0)
        self.assertEqual(decode_measurement_detid(999)["plane_key"], -1)

    def test_flat_v2_saves_aligned_measurements_offsets_and_plane_counts(self):
        detids = [
            1_000_001,
            1_000_002,
            1_100_003,
            2_000_004,
            10_001,
            20_002,
            33_003,
            999,
        ]
        points = [Vec(i, i + 0.5, 300 + i) for i in range(len(detids))]
        second_detids = [3_000_010, 3_100_011]
        second_points = [Vec(10, 20, 400), Vec(11, 21, 401)]
        scalars, vectors = buffers(FLAT_V2)
        reset_reco_track_buffers(scalars, vectors, True)
        fill_reco_track_buffers(
            [
                Track(11, points=points, detids=detids, npoints=len(points)),
                Track(
                    13,
                    points=second_points,
                    detids=second_detids,
                    npoints=len(second_points),
                ),
            ],
            scalars,
            vectors,
            schema=FLAT_V2,
        )

        self.assertEqual(vectors["reco_track_measurement_offset"], [0, 8, 10])
        self.assertEqual(
            vectors["reco_track_measurement_detid"], detids + second_detids
        )
        self.assertEqual(vectors["reco_track_measurement_point_z"][:2], [300.0, 301.0])
        self.assertEqual(vectors["reco_track_n_measurements"], [8, 2])
        self.assertEqual(vectors["reco_track_n_unique_planes"], [6, 2])
        self.assertEqual(vectors["reco_track_n_scifi_measurements"], [4, 2])
        self.assertEqual(vectors["reco_track_n_veto_measurements"], [1, 0])
        self.assertEqual(vectors["reco_track_n_us_measurements"], [1, 0])
        self.assertEqual(vectors["reco_track_n_ds_measurements"], [1, 0])
        self.assertEqual(vectors["reco_track_n_unknown_measurements"], [1, 0])
        self.assertEqual(vectors["reco_track_n_scifi_planes"], [3, 2])
        self.assertEqual(
            vectors["reco_track_first_measurement_detid"],
            [detids[0], second_detids[0]],
        )
        self.assertEqual(
            vectors["reco_track_last_measurement_detid"],
            [detids[-1], second_detids[-1]],
        )
        self.assertEqual(scalars["reco_scifi_hough_best_n_measurements"][0], 8)
        self.assertEqual(scalars["reco_scifi_hough_best_n_unique_planes"][0], 6)

    def test_flat_v2_empty_event_has_one_zero_offset(self):
        scalars, vectors = buffers(FLAT_V2)
        reset_reco_track_buffers(scalars, vectors, True)
        fill_reco_track_buffers([], scalars, vectors, schema=FLAT_V2)
        self.assertEqual(vectors["reco_track_measurement_offset"], [0])
        self.assertEqual(vectors["reco_track_measurement_detid"], [])
        self.assertEqual(
            scalars["reco_scifi_hough_best_first_measurement_detid"][0], -1
        )
        self.assertEqual(
            scalars["reco_scifi_hough_best_last_measurement_detid"][0], -1
        )

    def test_flat_v2_rejects_unaligned_points_and_detector_ids(self):
        scalars, vectors = buffers(FLAT_V2)
        reset_reco_track_buffers(scalars, vectors, True)
        track = Track(11, points=[Vec(1, 2, 3)], detids=[])
        with self.assertRaisesRegex(RuntimeError, "measurement alignment failure"):
            fill_reco_track_buffers([track], scalars, vectors, schema=FLAT_V2)


if __name__ == "__main__":
    unittest.main()
