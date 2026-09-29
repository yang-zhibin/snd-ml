"""Flat, analysis-friendly features derived from ``sndRecoTrack`` objects.

This module intentionally does not import ROOT.  The caller owns the scalar
buffers (one-element ``array.array`` objects) and vector buffers (ROOT vectors
or Python lists), which also makes the extraction logic straightforward to
unit-test.
"""

from __future__ import annotations

import math


TRACK_TYPE_PREFIXES = {
    1: "reco_scifi_simple",
    3: "reco_ds_simple",
    11: "reco_scifi_hough",
    13: "reco_ds_hough",
    15: "reco_combined_hough",
}

FLAT_V1 = "flat_v1"
FLAT_V2 = "flat_v2"
RECO_TRACK_SCHEMAS = (FLAT_V1, FLAT_V2)

# Stable values written to reco_track_measurement_subsystem.  Zero is kept for
# detector IDs that do not match the documented SciFi or MuFilter encodings.
MEASUREMENT_SUBSYSTEM_UNKNOWN = 0
MEASUREMENT_SUBSYSTEM_SCIFI = 1
MEASUREMENT_SUBSYSTEM_VETO = 2
MEASUREMENT_SUBSYSTEM_US = 3
MEASUREMENT_SUBSYSTEM_DS = 4

VECTOR_DTYPES = {
    "reco_track_type": "int",
    "reco_track_flag": "int",
    "reco_track_chi2": "double",
    "reco_track_ndf": "int",
    "reco_track_chi2ndf": "double",
    "reco_track_npoints": "int",
    "reco_track_start_x": "double",
    "reco_track_start_y": "double",
    "reco_track_start_z": "double",
    "reco_track_stop_x": "double",
    "reco_track_stop_y": "double",
    "reco_track_stop_z": "double",
    "reco_track_mom_x": "double",
    "reco_track_mom_y": "double",
    "reco_track_mom_z": "double",
    "reco_track_slope_xz": "double",
    "reco_track_slope_yz": "double",
    "reco_track_angle_xz": "double",
    "reco_track_angle_yz": "double",
}

# flat_v2 additions with one element per track.  Keep these separate from
# VECTOR_DTYPES: existing callers rely on every flat_v1 vector having
# reco_track_n_all elements.
DETAIL_TRACK_VECTOR_DTYPES = {
    "reco_track_n_measurements": "int",
    "reco_track_n_unique_planes": "int",
    "reco_track_n_scifi_measurements": "int",
    "reco_track_n_veto_measurements": "int",
    "reco_track_n_us_measurements": "int",
    "reco_track_n_ds_measurements": "int",
    "reco_track_n_unknown_measurements": "int",
    "reco_track_n_scifi_planes": "int",
    "reco_track_n_veto_planes": "int",
    "reco_track_n_us_planes": "int",
    "reco_track_n_ds_planes": "int",
    "reco_track_measurement_alignment_ok": "int",
    "reco_track_first_measurement_detid": "int",
    "reco_track_last_measurement_detid": "int",
}

# For track i, its associated measurements occupy [offset[i], offset[i + 1])
# in every measurement vector.
MEASUREMENT_OFFSET_VECTOR_DTYPES = {
    "reco_track_measurement_offset": "int",
}

MEASUREMENT_VECTOR_DTYPES = {
    "reco_track_measurement_detid": "int",
    "reco_track_measurement_point_x": "double",
    "reco_track_measurement_point_y": "double",
    "reco_track_measurement_point_z": "double",
    "reco_track_measurement_subsystem": "int",
    "reco_track_measurement_station": "int",
    "reco_track_measurement_plane": "int",
    "reco_track_measurement_view": "int",
    "reco_track_measurement_plane_key": "int",
}

SCALAR_DTYPES = {
    "reco_track_branch_available": "i",
    "reco_track_n_all": "i",
    "reco_track_n_converged": "i",
    "reco_track_n_types_present": "i",
    "reco_track_n_unknown_type": "i",
}

_BEST_DETAIL_INT_FIELDS = (
    "n_measurements",
    "n_unique_planes",
    "n_scifi_measurements",
    "n_veto_measurements",
    "n_us_measurements",
    "n_ds_measurements",
    "n_unknown_measurements",
    "n_scifi_planes",
    "n_veto_planes",
    "n_us_planes",
    "n_ds_planes",
    "measurement_alignment_ok",
    "first_measurement_detid",
    "last_measurement_detid",
)

DETAIL_SCALAR_DTYPES = {}

_BEST_FLOAT_FIELDS = (
    "chi2",
    "chi2ndf",
    "start_x",
    "start_y",
    "start_z",
    "stop_x",
    "stop_y",
    "stop_z",
    "mom_x",
    "mom_y",
    "mom_z",
    "slope_xz",
    "slope_yz",
    "angle_xz",
    "angle_yz",
)
_BEST_INT_FIELDS = ("flag", "ndf", "npoints")

for _prefix in TRACK_TYPE_PREFIXES.values():
    SCALAR_DTYPES.update(
        {
            f"{_prefix}_n_all": "i",
            f"{_prefix}_n_converged": "i",
            f"{_prefix}_has_any": "i",
            f"{_prefix}_has_converged": "i",
            f"{_prefix}_has_best": "i",
            f"{_prefix}_best_index": "i",
        }
    )
    SCALAR_DTYPES.update(
        {f"{_prefix}_best_{field}": "i" for field in _BEST_INT_FIELDS}
    )
    SCALAR_DTYPES.update(
        {f"{_prefix}_best_{field}": "d" for field in _BEST_FLOAT_FIELDS}
    )
    DETAIL_SCALAR_DTYPES.update(
        {f"{_prefix}_best_{field}": "i" for field in _BEST_DETAIL_INT_FIELDS}
    )


def scalar_dtypes_for_schema(schema):
    """Return scalar branch definitions without mutating the v1 schema."""
    if schema == FLAT_V1:
        return dict(SCALAR_DTYPES)
    if schema == FLAT_V2:
        return {**SCALAR_DTYPES, **DETAIL_SCALAR_DTYPES}
    raise ValueError(f"Unsupported reconstructed-track schema: {schema!r}")


def vector_dtypes_for_schema(schema):
    """Return vector branch definitions for one supported schema."""
    if schema == FLAT_V1:
        return dict(VECTOR_DTYPES)
    if schema == FLAT_V2:
        return {
            **VECTOR_DTYPES,
            **DETAIL_TRACK_VECTOR_DTYPES,
            **MEASUREMENT_OFFSET_VECTOR_DTYPES,
            **MEASUREMENT_VECTOR_DTYPES,
        }
    raise ValueError(f"Unsupported reconstructed-track schema: {schema!r}")


def scalar_default(name):
    """Return the documented sentinel/default for one scalar branch."""
    if name.endswith(
        (
            "_best_index",
            "_best_first_measurement_detid",
            "_best_last_measurement_detid",
        )
    ):
        return -1
    dtype = SCALAR_DTYPES.get(name, DETAIL_SCALAR_DTYPES.get(name))
    if dtype is None:
        raise KeyError(name)
    if dtype == "d":
        return math.nan
    return 0


def reset_reco_track_buffers(scalars, vectors, branch_available):
    """Reset all buffers before processing one event."""
    for name, buffer in scalars.items():
        buffer[0] = scalar_default(name)
    scalars["reco_track_branch_available"][0] = int(bool(branch_available))
    for vector in vectors.values():
        vector.clear()


def _append(vector, value):
    if hasattr(vector, "push_back"):
        vector.push_back(value)
    else:
        vector.append(value)


def _safe_call(obj, method, default):
    try:
        return getattr(obj, method)()
    except Exception:
        return default


def _component(vector, name):
    try:
        value = getattr(vector, name)()
        return float(value) if math.isfinite(float(value)) else math.nan
    except Exception:
        return math.nan


def _as_list(obj, method):
    """Read one required sndRecoTrack sequence with a useful failure mode."""
    try:
        return list(getattr(obj, method)())
    except Exception as exc:
        raise RuntimeError(f"Cannot read sndRecoTrack.{method}()") from exc


def decode_measurement_detid(detid):
    """Decode a confirmed sndsw SciFi or MuFilter measurement detector ID.

    SciFi planes are identified by (station, view); MuFilter planes by
    (subsystem, plane). Values that do not satisfy either documented encoding
    are retained as unknown instead of being guessed.
    """
    detid = int(detid)
    station = detid // 1_000_000
    view = (detid // 100_000) % 10
    if 1 <= station <= 5 and view in (0, 1):
        subsystem = MEASUREMENT_SUBSYSTEM_SCIFI
        return {
            "subsystem": subsystem,
            "station": station,
            "plane": -1,
            "view": view,
            "plane_key": 100 + 10 * station + view,
        }

    mufi_system = detid // 10_000
    if mufi_system in (1, 2, 3):
        subsystem = {
            1: MEASUREMENT_SUBSYSTEM_VETO,
            2: MEASUREMENT_SUBSYSTEM_US,
            3: MEASUREMENT_SUBSYSTEM_DS,
        }[mufi_system]
        plane = (detid // 1_000) % 10
        return {
            "subsystem": subsystem,
            "station": -1,
            "plane": plane,
            "view": -1,
            "plane_key": 100 * subsystem + plane,
        }

    return {
        "subsystem": MEASUREMENT_SUBSYSTEM_UNKNOWN,
        "station": -1,
        "plane": -1,
        "view": -1,
        "plane_key": -1,
    }


def _track_detail_record(track):
    points = _as_list(track, "getTrackPoints")
    detids = [int(value) for value in _as_list(track, "getRawMeasDetIDs")]
    alignment_ok = len(points) == len(detids)
    if not alignment_ok:
        raise RuntimeError(
            "sndRecoTrack measurement alignment failure: "
            f"{len(points)} fitted points but {len(detids)} detector IDs"
        )

    measurements = []
    subsystem_counts = {
        MEASUREMENT_SUBSYSTEM_SCIFI: 0,
        MEASUREMENT_SUBSYSTEM_VETO: 0,
        MEASUREMENT_SUBSYSTEM_US: 0,
        MEASUREMENT_SUBSYSTEM_DS: 0,
        MEASUREMENT_SUBSYSTEM_UNKNOWN: 0,
    }
    plane_keys = {subsystem: set() for subsystem in subsystem_counts}
    for point, detid in zip(points, detids):
        decoded = decode_measurement_detid(detid)
        subsystem = decoded["subsystem"]
        subsystem_counts[subsystem] += 1
        if decoded["plane_key"] >= 0:
            plane_keys[subsystem].add(decoded["plane_key"])
        measurements.append(
            {
                "detid": detid,
                "point_x": _component(point, "X"),
                "point_y": _component(point, "Y"),
                "point_z": _component(point, "Z"),
                **decoded,
            }
        )

    known_plane_keys = set().union(
        *(keys for subsystem, keys in plane_keys.items() if subsystem != 0)
    )
    detail = {
        "n_measurements": len(detids),
        "n_unique_planes": len(known_plane_keys),
        "n_scifi_measurements": subsystem_counts[MEASUREMENT_SUBSYSTEM_SCIFI],
        "n_veto_measurements": subsystem_counts[MEASUREMENT_SUBSYSTEM_VETO],
        "n_us_measurements": subsystem_counts[MEASUREMENT_SUBSYSTEM_US],
        "n_ds_measurements": subsystem_counts[MEASUREMENT_SUBSYSTEM_DS],
        "n_unknown_measurements": subsystem_counts[MEASUREMENT_SUBSYSTEM_UNKNOWN],
        "n_scifi_planes": len(plane_keys[MEASUREMENT_SUBSYSTEM_SCIFI]),
        "n_veto_planes": len(plane_keys[MEASUREMENT_SUBSYSTEM_VETO]),
        "n_us_planes": len(plane_keys[MEASUREMENT_SUBSYSTEM_US]),
        "n_ds_planes": len(plane_keys[MEASUREMENT_SUBSYSTEM_DS]),
        "measurement_alignment_ok": int(alignment_ok),
        "first_measurement_detid": detids[0] if detids else -1,
        "last_measurement_detid": detids[-1] if detids else -1,
    }
    return detail, measurements


def _track_record(track):
    track_type = int(_safe_call(track, "getTrackType", -1))
    flag = int(bool(_safe_call(track, "getTrackFlag", False)))
    chi2 = float(_safe_call(track, "getChi2", math.nan))
    ndf = int(_safe_call(track, "getNdf", 0))
    points = _safe_call(track, "getTrackPoints", None)
    try:
        npoints = int(points.size())
    except Exception:
        try:
            npoints = len(points)
        except Exception:
            npoints = 0

    chi2ndf = chi2 / ndf if ndf > 0 and math.isfinite(chi2) else math.nan
    start = stop = momentum = None
    if flag:
        start = _safe_call(track, "getStart", None)
        stop = _safe_call(track, "getStop", None)
        momentum = _safe_call(track, "getTrackMom", None)

    sx, sy, sz = (_component(start, axis) for axis in ("X", "Y", "Z"))
    ex, ey, ez = (_component(stop, axis) for axis in ("X", "Y", "Z"))
    px, py, pz = (_component(momentum, axis) for axis in ("X", "Y", "Z"))
    slope_xz = px / pz if math.isfinite(px) and math.isfinite(pz) and pz != 0 else math.nan
    slope_yz = py / pz if math.isfinite(py) and math.isfinite(pz) and pz != 0 else math.nan

    return {
        "type": track_type,
        "flag": flag,
        "chi2": chi2,
        "ndf": ndf,
        "chi2ndf": chi2ndf,
        "npoints": npoints,
        "start_x": sx,
        "start_y": sy,
        "start_z": sz,
        "stop_x": ex,
        "stop_y": ey,
        "stop_z": ez,
        "mom_x": px,
        "mom_y": py,
        "mom_z": pz,
        "slope_xz": slope_xz,
        "slope_yz": slope_yz,
        "angle_xz": math.atan(slope_xz) if math.isfinite(slope_xz) else math.nan,
        "angle_yz": math.atan(slope_yz) if math.isfinite(slope_yz) else math.nan,
    }


def _is_best_candidate(record):
    return (
        record["flag"] == 1
        and record["ndf"] > 0
        and math.isfinite(record["chi2ndf"])
        and record["chi2ndf"] >= 0
    )


def fill_reco_track_buffers(collection, scalars, vectors, schema=FLAT_V1):
    """Extract one event's reconstructed-track collection into flat buffers."""
    if schema not in RECO_TRACK_SCHEMAS:
        raise ValueError(f"Unsupported reconstructed-track schema: {schema!r}")
    detailed = schema == FLAT_V2
    records = []
    try:
        iterator = iter(collection)
    except TypeError:
        iterator = iter(())

    if detailed:
        _append(vectors["reco_track_measurement_offset"], 0)

    for track in iterator:
        record = _track_record(track)
        measurements = []
        if detailed:
            detail, measurements = _track_detail_record(track)
            record.update(detail)
        records.append(record)
        for field in VECTOR_DTYPES:
            key = field.removeprefix("reco_track_")
            _append(vectors[field], record[key])
        if detailed:
            for field in DETAIL_TRACK_VECTOR_DTYPES:
                key = field.removeprefix("reco_track_")
                _append(vectors[field], record[key])
            for measurement in measurements:
                for field in MEASUREMENT_VECTOR_DTYPES:
                    key = field.removeprefix("reco_track_measurement_")
                    _append(vectors[field], measurement[key])
            _append(
                vectors["reco_track_measurement_offset"],
                len(vectors["reco_track_measurement_detid"]),
            )

    scalars["reco_track_n_all"][0] = len(records)
    scalars["reco_track_n_converged"][0] = sum(r["flag"] for r in records)
    scalars["reco_track_n_unknown_type"][0] = sum(
        r["type"] not in TRACK_TYPE_PREFIXES for r in records
    )

    present = 0
    for track_type, prefix in TRACK_TYPE_PREFIXES.items():
        indexed = [(index, r) for index, r in enumerate(records) if r["type"] == track_type]
        converged = [(index, r) for index, r in indexed if r["flag"]]
        candidates = [(index, r) for index, r in indexed if _is_best_candidate(r)]
        scalars[f"{prefix}_n_all"][0] = len(indexed)
        scalars[f"{prefix}_n_converged"][0] = len(converged)
        scalars[f"{prefix}_has_any"][0] = int(bool(indexed))
        scalars[f"{prefix}_has_converged"][0] = int(bool(converged))
        if indexed:
            present += 1
        if not candidates:
            continue

        best_index, best = min(
            candidates,
            key=lambda item: (item[1]["chi2ndf"], -item[1]["npoints"], item[0]),
        )
        scalars[f"{prefix}_has_best"][0] = 1
        scalars[f"{prefix}_best_index"][0] = best_index
        for field in _BEST_INT_FIELDS + _BEST_FLOAT_FIELDS:
            scalars[f"{prefix}_best_{field}"][0] = best[field]
        if detailed:
            for field in _BEST_DETAIL_INT_FIELDS:
                scalars[f"{prefix}_best_{field}"][0] = best[field]

    scalars["reco_track_n_types_present"][0] = present
