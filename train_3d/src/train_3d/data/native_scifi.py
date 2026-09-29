"""Lossless detector-native SciFi hit contract and pure transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


VERTICAL = 0
HORIZONTAL = 1
NATIVE_SCIFI_BRANCHES = {
    "station": "scifi_native_station",
    "orientation": "scifi_native_orientation",
    "channel": "scifi_native_channel",
    "det_id": "scifi_native_det_id",
    "x": "scifi_native_x",
    "y": "scifi_native_y",
    "z": "scifi_native_z",
    "qdc": "scifi_native_qdc",
    "time_ns": "scifi_native_time_ns",
}
NATIVE_SCIFI_DTYPES = {
    "station": np.int16,
    "orientation": np.int16,
    "channel": np.int32,
    "det_id": np.int64,
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
    "qdc": np.float32,
    "time_ns": np.float32,
}


def validate_native_scifi_hits(raw: dict) -> dict[str, np.ndarray]:
    """Return typed, aligned native hit arrays after strict validation."""
    missing = set(NATIVE_SCIFI_BRANCHES) - set(raw)
    if missing:
        raise KeyError(f"Native SciFi event lacks fields: {sorted(missing)}")
    arrays = {
        name: np.asarray(raw[name], dtype=NATIVE_SCIFI_DTYPES[name])
        for name in NATIVE_SCIFI_BRANCHES
    }
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Native SciFi fields must be one-dimensional vectors")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Native SciFi vector length mismatch: {sorted(lengths)}")
    for name in ("x", "y", "z", "qdc", "time_ns"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Native SciFi field {name} contains a non-finite value")
    if np.any(arrays["qdc"] < 0):
        raise ValueError("Native SciFi QDC must be non-negative")
    if np.any((arrays["station"] < 1) | (arrays["station"] > 5)):
        raise ValueError("Native SciFi station values must be in [1,5]")
    if np.any((arrays["orientation"] != VERTICAL) &
              (arrays["orientation"] != HORIZONTAL)):
        raise ValueError("Native SciFi orientation must be 0 or 1")
    if np.any(arrays["channel"] < 0):
        raise ValueError("Native SciFi channels must be non-negative")
    if np.any(arrays["det_id"] < 0):
        raise ValueError("Native SciFi detector IDs must be non-negative")
    return arrays


def measured_coordinate(event: dict[str, np.ndarray]) -> np.ndarray:
    """Select X for vertical fibers and Y for horizontal fibers."""
    orientation = np.asarray(event["orientation"])
    return np.where(
        orientation == VERTICAL, event["x"], event["y"]
    ).astype(np.float32, copy=False)


def native_hit_counts(event: dict[str, np.ndarray]) -> dict[str, int]:
    orientation = np.asarray(event["orientation"])
    vertical = orientation == VERTICAL
    horizontal = orientation == HORIZONTAL
    vertical_sites = np.unique(
        np.column_stack([event["station"][vertical], event["channel"][vertical]]),
        axis=0,
    ) if vertical.any() else np.empty((0, 2), dtype=np.int64)
    horizontal_sites = np.unique(
        np.column_stack([event["station"][horizontal], event["channel"][horizontal]]),
        axis=0,
    ) if horizontal.any() else np.empty((0, 2), dtype=np.int64)
    return {
        "n_native_hits": len(orientation),
        "n_vertical_hits": int(vertical.sum()),
        "n_horizontal_hits": int(horizontal.sum()),
        "n_vertical_sites": len(vertical_sites),
        "n_horizontal_sites": len(horizontal_sites),
    }


@dataclass(frozen=True)
class NativeSciFiNormalization:
    version: str
    dataset_version: str
    cv_fold: int
    u_mean: float
    u_scale: float
    z_mean: float
    z_scale: float
    qdc_transform: str
    qdc_mean: float
    qdc_scale: float
    channel_min: float
    channel_max: float
    station_min: float
    station_max: float
    training_events: int
    training_native_hits: int

    def __post_init__(self):
        if self.version != "scifi_native_hit_normalization_v1":
            raise ValueError("Unsupported native SciFi normalization version")
        if self.qdc_transform != "log1p":
            raise ValueError("Only native qdc_transform='log1p' is supported")
        values = np.asarray([
            self.u_mean, self.u_scale, self.z_mean, self.z_scale,
            self.qdc_mean, self.qdc_scale, self.channel_min, self.channel_max,
            self.station_min, self.station_max,
        ], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Native normalization constants must be finite")
        if self.u_scale <= 0 or self.z_scale <= 0 or self.qdc_scale <= 0:
            raise ValueError("Native normalization scales must be positive")
        if self.channel_max <= self.channel_min or self.station_max <= self.station_min:
            raise ValueError("Native detector normalization ranges must be increasing")

    @classmethod
    def load(cls, path, expected_fold: int | None = None):
        result = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if expected_fold is not None and result.cv_fold != int(expected_fold):
            raise ValueError(
                f"Native normalization fold {result.cv_fold} does not match "
                f"fold {expected_fold}"
            )
        return result

    def save(self, path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _standardize(values, mean, scale):
        return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _range_scale(values, lower, upper):
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        return ((np.asarray(values, dtype=np.float32) - midpoint) / half_range).astype(
            np.float32, copy=False
        )

    def transform_qdc(self, qdc):
        qdc = np.asarray(qdc, dtype=np.float32)
        if np.any(qdc < 0) or not np.isfinite(qdc).all():
            raise ValueError("Native SciFi QDC must be finite and non-negative")
        return self._standardize(np.log1p(qdc), self.qdc_mean, self.qdc_scale)

    def node_features(self, event: dict[str, np.ndarray]) -> np.ndarray:
        orientation = np.asarray(event["orientation"])
        return np.column_stack([
            self._standardize(measured_coordinate(event), self.u_mean, self.u_scale),
            self._standardize(event["z"], self.z_mean, self.z_scale),
            self.transform_qdc(event["qdc"]),
            self._range_scale(event["channel"], self.channel_min, self.channel_max),
            self._range_scale(event["station"], self.station_min, self.station_max),
            (orientation == VERTICAL).astype(np.float32),
            (orientation == HORIZONTAL).astype(np.float32),
        ]).astype(np.float32, copy=False)


def build_sparse_view(event: dict[str, np.ndarray], orientation: int,
                      normalization: NativeSciFiNormalization):
    """Build exact detector cells with QDC-sum and occupancy features."""
    selected = np.asarray(event["orientation"]) == int(orientation)
    if not selected.any():
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.float32)
    coordinates = np.column_stack([
        np.asarray(event["station"])[selected].astype(np.int32) - 1,
        np.asarray(event["channel"])[selected].astype(np.int32),
    ])
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    qdc_sum = np.zeros(len(unique), dtype=np.float32)
    occupancy = np.zeros(len(unique), dtype=np.float32)
    np.add.at(qdc_sum, inverse, np.asarray(event["qdc"], dtype=np.float32)[selected])
    np.add.at(occupancy, inverse, 1.0)
    features = np.column_stack([
        normalization.transform_qdc(qdc_sum), occupancy,
    ]).astype(np.float32, copy=False)
    return unique.astype(np.int32, copy=False), features


def reconstruct_native_from_candidates(raw: dict, tolerance: float = 1e-5):
    """Approximate native hits from legacy crossed candidates for auditing.

    Separate fiber Z/time and detector IDs were not stored. The returned Z is
    the mean candidate Z for a key, time is the minimum candidate time, and
    detector ID is a deterministic surrogate. This function must not be used
    for final physics training.
    """
    required = {
        "station", "detector_type", "x", "y", "z", "time_ns",
        "v_channel", "h_channel", "v_qdc", "h_qdc",
    }
    missing = required - set(raw)
    if missing:
        raise KeyError(f"Legacy candidate event lacks fields: {sorted(missing)}")
    arrays = {name: np.asarray(raw[name]) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Legacy candidate vector length mismatch: {sorted(lengths)}")
    selected = arrays["detector_type"] == 1
    records = {}
    for row in np.nonzero(selected)[0]:
        for orientation, channel_name, qdc_name, coordinate_name in (
            (VERTICAL, "v_channel", "v_qdc", "x"),
            (HORIZONTAL, "h_channel", "h_qdc", "y"),
        ):
            channel = int(arrays[channel_name][row])
            if channel < 0:
                continue
            station = int(arrays["station"][row])
            key = (station, orientation, channel)
            qdc = float(arrays[qdc_name][row])
            coordinate = float(arrays[coordinate_name][row])
            if key in records:
                previous = records[key]
                if not np.isclose(previous["qdc"], qdc, rtol=tolerance, atol=tolerance):
                    raise ValueError(f"Contradictory QDC for reconstructed key {key}")
                if not np.isclose(
                    previous["u"], coordinate, rtol=tolerance, atol=tolerance
                ):
                    raise ValueError(f"Contradictory coordinate for reconstructed key {key}")
                previous["z_values"].append(float(arrays["z"][row]))
                previous["time_values"].append(float(arrays["time_ns"][row]))
            else:
                records[key] = {
                    "station": station, "orientation": orientation,
                    "channel": channel, "qdc": qdc, "u": coordinate,
                    "z_values": [float(arrays["z"][row])],
                    "time_values": [float(arrays["time_ns"][row])],
                }
    ordered = list(records.values())
    result = {
        "station": np.asarray([item["station"] for item in ordered], np.int16),
        "orientation": np.asarray([item["orientation"] for item in ordered], np.int16),
        "channel": np.asarray([item["channel"] for item in ordered], np.int32),
        "det_id": np.asarray([
            item["station"] * 100000 + item["orientation"] * 10000 + item["channel"]
            for item in ordered
        ], np.int64),
        "x": np.asarray([
            item["u"] if item["orientation"] == VERTICAL else 0.0
            for item in ordered
        ], np.float32),
        "y": np.asarray([
            item["u"] if item["orientation"] == HORIZONTAL else 0.0
            for item in ordered
        ], np.float32),
        "z": np.asarray([np.mean(item["z_values"]) for item in ordered], np.float32),
        "qdc": np.asarray([item["qdc"] for item in ordered], np.float32),
        "time_ns": np.asarray([
            np.min(item["time_values"]) for item in ordered
        ], np.float32),
    }
    return validate_native_scifi_hits(result)
