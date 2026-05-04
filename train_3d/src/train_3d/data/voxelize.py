from collections import defaultdict

import numpy as np


def infer_coordinate_ranges(events, padding=1e-3):
    mins = defaultdict(lambda: np.inf)
    maxs = defaultdict(lambda: -np.inf)

    found_any = False
    for event in events:
        hits = event.get("all_3dHits", {})
        for key in ("x", "y", "z"):
            values = np.asarray(hits.get(key, []), dtype=np.float32)
            if values.size == 0:
                continue
            found_any = True
            mins[key] = min(mins[key], float(values.min()))
            maxs[key] = max(maxs[key], float(values.max()))

    if not found_any:
        return {
            "x_range": (-1.0, 1.0),
            "y_range": (-1.0, 1.0),
            "z_range": (-1.0, 1.0),
        }

    ranges = {}
    for key in ("x", "y", "z"):
        lo = mins[key]
        hi = maxs[key]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        ranges[f"{key}_range"] = (lo - padding, hi + padding)

    return ranges


def build_sparse_voxels(
    all_3d_hits,
    ranges,
    voxel_size,
    feature_channels,
    clamp_out_of_bounds=False,
):
    x = np.asarray(all_3d_hits.get("x", []), dtype=np.float32)
    y = np.asarray(all_3d_hits.get("y", []), dtype=np.float32)
    z = np.asarray(all_3d_hits.get("z", []), dtype=np.float32)
    qdc = np.asarray(all_3d_hits.get("qdc", []), dtype=np.float32)
    det_type = np.asarray(all_3d_hits.get("detType", []), dtype=np.int16)
    station = np.asarray(all_3d_hits.get("station", []), dtype=np.float32)

    if x.size == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, len(feature_channels)), dtype=np.float32),
        )

    x_min, x_max = ranges["x_range"]
    y_min, y_max = ranges["y_range"]
    z_min, z_max = ranges["z_range"]
    dx, dy, dz = [float(v) for v in voxel_size]

    if clamp_out_of_bounds:
        valid = np.ones_like(x, dtype=bool)
    else:
        valid = (
            (x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max)
        )

    x = x[valid]
    y = y[valid]
    z = z[valid]
    qdc = qdc[valid]
    det_type = det_type[valid]
    station = station[valid]

    if x.size == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, len(feature_channels)), dtype=np.float32),
        )

    ix = np.floor((x - x_min) / max(dx, 1e-12)).astype(np.int32)
    iy = np.floor((y - y_min) / max(dy, 1e-12)).astype(np.int32)
    iz = np.floor((z - z_min) / max(dz, 1e-12)).astype(np.int32)

    coords = np.stack([ix, iy, iz], axis=1)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    features = np.zeros((len(unique_coords), len(feature_channels)), dtype=np.float32)

    for channel_index, name in enumerate(feature_channels):
        if name in ("qdc", "qdc_sum"):
            np.add.at(features[:, channel_index], inverse, qdc)
        elif name == "occupancy":
            np.add.at(features[:, channel_index], inverse, 1.0)
        elif name == "dettype_1":
            np.add.at(features[:, channel_index], inverse[det_type == 1], 1.0)
        elif name == "dettype_2":
            np.add.at(features[:, channel_index], inverse[det_type == 2], 1.0)
        elif name == "dettype_3":
            np.add.at(features[:, channel_index], inverse[det_type == 3], 1.0)
        elif name == "station_mean":
            sums = np.zeros(len(unique_coords), dtype=np.float32)
            counts = np.zeros(len(unique_coords), dtype=np.float32)
            np.add.at(sums, inverse, station)
            np.add.at(counts, inverse, 1.0)
            features[:, channel_index] = sums / np.maximum(counts, 1.0)
        else:
            raise ValueError(f"Unsupported sparse feature channel: {name}")

    return unique_coords.astype(np.int32, copy=False), features
