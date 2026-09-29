#!/usr/bin/env python3
"""Pure geometry arithmetic for the SciFi edge-transform validation.

This module deliberately has no ROOT or SNDSW dependency.  The adapter is
responsible for constructing physical plane geometry and fitted track points;
the functions here choose an edge, translate hit coordinates, and validate the
mathematical invariants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class PlaneKey:
    station: int
    view: int

    def __post_init__(self):
        if self.station not in range(1, 6):
            raise ValueError(f"SciFi station must be 1..5, got {self.station}")
        if self.view not in (0, 1):
            raise ValueError(f"SciFi view must be 0 or 1, got {self.view}")

    @property
    def orientation(self) -> str:
        return "vertical" if self.view == 1 else "horizontal"

    @property
    def coordinate(self) -> str:
        return "x" if self.view == 1 else "y"


@dataclass(frozen=True)
class ActiveInterval:
    low_cm: float
    high_cm: float
    mat: int
    sipm: int
    channel: int

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.low_cm, self.high_cm)):
            raise ValueError("Active channel interval contains a non-finite boundary")
        if self.high_cm <= self.low_cm:
            raise ValueError("Active channel interval must have positive width")


@dataclass(frozen=True)
class PlaneGeometry:
    key: PlaneKey
    z_cm: float
    low_edge_cm: float
    high_edge_cm: float
    intervals: tuple[ActiveInterval, ...]
    z_spread_cm: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.z_cm, self.low_edge_cm, self.high_edge_cm, self.z_spread_cm,
        )):
            raise ValueError(f"Plane {self.key} geometry is not finite")
        if self.high_edge_cm <= self.low_edge_cm:
            raise ValueError(f"Plane {self.key} has invalid detector edges")
        if self.z_spread_cm < 0:
            raise ValueError(f"Plane {self.key} has a negative z spread")
        if not self.intervals:
            raise ValueError(f"Plane {self.key} has no active channel intervals")


@dataclass(frozen=True)
class TrackProjection:
    coordinate: str
    slope: float
    intercept: float
    n_points: int
    n_distinct_z: int
    max_abs_residual_cm: float

    def at(self, z_cm: float) -> float:
        value = self.intercept + self.slope * z_cm
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {self.coordinate}(z) at z={z_cm}")
        return value


@dataclass(frozen=True)
class PlaneTransform:
    key: PlaneKey
    z_cm: float
    locked_side: str
    plane_local_closest_side: str
    low_edge_cm: float
    high_edge_cm: float
    target_cm: float
    track_original_cm: float
    track_transformed_cm: float
    delta_cm: float


@dataclass(frozen=True)
class HitCoordinate:
    key: PlaneKey
    coordinate_cm: float
    detector_id: int = -1
    qdc: float = math.nan
    time_ns: float = math.nan
    valid: bool = True


@dataclass(frozen=True)
class TransformedHit:
    source: HitCoordinate
    transformed_coordinate_cm: float
    location: str
    destination_channel: tuple[int, int, int] | None


def _require_finite(values: Iterable[float], label: str):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} contains a non-finite value")


def fit_linear_projection(
    points: Sequence[tuple[float, float]], coordinate: str,
    min_distinct_z: int = 2,
) -> TrackProjection:
    """Fit ``u(z) = intercept + slope*z`` with ordinary least squares."""
    if coordinate not in ("x", "y"):
        raise ValueError(f"Unsupported projection coordinate {coordinate!r}")
    if len(points) < min_distinct_z:
        raise ValueError(
            f"{coordinate}(z) needs at least {min_distinct_z} points; got {len(points)}"
        )
    clean = [(float(z), float(value)) for z, value in points]
    _require_finite((item for pair in clean for item in pair), f"{coordinate}(z) points")
    distinct_z = len({z for z, _ in clean})
    if distinct_z < min_distinct_z:
        raise ValueError(
            f"{coordinate}(z) needs at least {min_distinct_z} distinct z values; "
            f"got {distinct_z}"
        )
    mean_z = sum(z for z, _ in clean) / len(clean)
    mean_u = sum(value for _, value in clean) / len(clean)
    denominator = sum((z - mean_z) ** 2 for z, _ in clean)
    if denominator <= 0:
        raise ValueError(f"Degenerate {coordinate}(z) fit")
    slope = sum((z - mean_z) * (value - mean_u) for z, value in clean) / denominator
    intercept = mean_u - slope * mean_z
    residuals = [value - (intercept + slope * z) for z, value in clean]
    _require_finite((slope, intercept, *residuals), f"{coordinate}(z) fit")
    return TrackProjection(
        coordinate=coordinate,
        slope=slope,
        intercept=intercept,
        n_points=len(clean),
        n_distinct_z=distinct_z,
        max_abs_residual_cm=max(abs(value) for value in residuals),
    )


def choose_locked_side(
    planes: Sequence[PlaneGeometry],
    track_positions: Mapping[PlaneKey, float],
) -> str:
    """Choose one low/high side by summed distance over an orientation."""
    if not planes:
        raise ValueError("Cannot choose an edge side without planes")
    missing = [plane.key for plane in planes if plane.key not in track_positions]
    if missing:
        raise KeyError(f"Missing track positions for planes: {missing}")
    low_distance = sum(
        abs(float(track_positions[plane.key]) - plane.low_edge_cm) for plane in planes
    )
    high_distance = sum(
        abs(plane.high_edge_cm - float(track_positions[plane.key])) for plane in planes
    )
    # A deterministic tie goes to the lower edge.
    return "low" if low_distance <= high_distance else "high"


def build_plane_transforms(
    planes: Sequence[PlaneGeometry],
    track_positions: Mapping[PlaneKey, float],
    inward_margin_cm: float = 0.0,
    lock_side_per_orientation: bool = True,
) -> dict[PlaneKey, PlaneTransform]:
    if not math.isfinite(inward_margin_cm) or inward_margin_cm < 0:
        raise ValueError("inward_margin_cm must be finite and non-negative")
    by_orientation: dict[str, list[PlaneGeometry]] = {"horizontal": [], "vertical": []}
    for plane in planes:
        by_orientation[plane.key.orientation].append(plane)
    output = {}
    for orientation, group in by_orientation.items():
        if not group:
            continue
        locked = choose_locked_side(group, track_positions)
        for plane in group:
            original = float(track_positions[plane.key])
            if not math.isfinite(original):
                raise ValueError(f"Track position for {plane.key} is non-finite")
            low_distance = abs(original - plane.low_edge_cm)
            high_distance = abs(plane.high_edge_cm - original)
            local = "low" if low_distance <= high_distance else "high"
            side = locked if lock_side_per_orientation else local
            target = (
                plane.low_edge_cm + inward_margin_cm
                if side == "low"
                else plane.high_edge_cm - inward_margin_cm
            )
            if target < plane.low_edge_cm or target > plane.high_edge_cm:
                raise ValueError(
                    f"Margin {inward_margin_cm} places target outside plane {plane.key}"
                )
            delta = target - original
            output[plane.key] = PlaneTransform(
                key=plane.key,
                z_cm=plane.z_cm,
                locked_side=side,
                plane_local_closest_side=local,
                low_edge_cm=plane.low_edge_cm,
                high_edge_cm=plane.high_edge_cm,
                target_cm=target,
                track_original_cm=original,
                track_transformed_cm=original + delta,
                delta_cm=delta,
            )
    return output


def classify_transformed_coordinate(
    coordinate_cm: float, plane: PlaneGeometry, tolerance_cm: float = 1.0e-10,
) -> tuple[str, tuple[int, int, int] | None]:
    """Classify a coordinate as active channel, internal gap, or outside."""
    if not math.isfinite(coordinate_cm):
        raise ValueError("Cannot classify a non-finite coordinate")
    for interval in plane.intervals:
        if interval.low_cm - tolerance_cm <= coordinate_cm <= interval.high_cm + tolerance_cm:
            return "inside_active", (interval.mat, interval.sipm, interval.channel)
    if plane.low_edge_cm - tolerance_cm <= coordinate_cm <= plane.high_edge_cm + tolerance_cm:
        return "inside_gap", None
    return "outside_geometry", None


def transform_hits(
    hits: Sequence[HitCoordinate],
    planes: Mapping[PlaneKey, PlaneGeometry],
    transforms: Mapping[PlaneKey, PlaneTransform],
) -> list[TransformedHit]:
    output = []
    for hit in hits:
        if hit.key not in planes or hit.key not in transforms:
            raise KeyError(f"Missing plane/transform for hit plane {hit.key}")
        if not math.isfinite(hit.coordinate_cm):
            raise ValueError(f"Hit {hit.detector_id} has a non-finite coordinate")
        transformed = hit.coordinate_cm + transforms[hit.key].delta_cm
        location, destination = classify_transformed_coordinate(transformed, planes[hit.key])
        output.append(TransformedHit(hit, transformed, location, destination))
    return output


def validate_transform(
    hits: Sequence[HitCoordinate], transformed_hits: Sequence[TransformedHit],
    transforms: Mapping[PlaneKey, PlaneTransform],
    residual_tolerance_cm: float = 1.0e-9,
    track_edge_tolerance_cm: float = 1.0e-8,
    require_locked_sides: bool = True,
) -> dict:
    if len(hits) != len(transformed_hits):
        raise ValueError("Original/transformed hit collections have different sizes")
    failures = []
    max_residual_difference = 0.0
    for key, transform in transforms.items():
        difference = abs(transform.track_transformed_cm - transform.target_cm)
        if difference > track_edge_tolerance_cm:
            failures.append(
                f"{key}: transformed track misses target by {difference:.6g} cm"
            )
    sides: dict[str, set[str]] = {"horizontal": set(), "vertical": set()}
    for transform in transforms.values():
        sides[transform.key.orientation].add(transform.locked_side)
    if require_locked_sides:
        for orientation, values in sides.items():
            if len(values) > 1:
                failures.append(f"{orientation}: multiple locked sides {sorted(values)}")

    locations = {"inside_active": 0, "inside_gap": 0, "outside_geometry": 0}
    for original, transformed in zip(hits, transformed_hits):
        if transformed.source != original:
            failures.append(f"Hit ordering/source mismatch at detector ID {original.detector_id}")
            continue
        transform = transforms[original.key]
        before = original.coordinate_cm - transform.track_original_cm
        after = transformed.transformed_coordinate_cm - transform.track_transformed_cm
        difference = abs(after - before)
        max_residual_difference = max(max_residual_difference, difference)
        if difference > residual_tolerance_cm:
            failures.append(
                f"hit {original.detector_id}: residual changed by {difference:.6g} cm"
            )
        if transformed.location not in locations:
            failures.append(f"hit {original.detector_id}: unknown location {transformed.location}")
        else:
            locations[transformed.location] += 1
    if sum(locations.values()) != len(hits):
        failures.append("Transformed location counts do not close to original hit count")
    finite_values = []
    for transform in transforms.values():
        finite_values.extend((
            transform.z_cm, transform.low_edge_cm, transform.high_edge_cm,
            transform.target_cm, transform.track_original_cm,
            transform.track_transformed_cm, transform.delta_cm,
        ))
    finite_values.extend(hit.coordinate_cm for hit in hits)
    finite_values.extend(hit.transformed_coordinate_cm for hit in transformed_hits)
    if not all(math.isfinite(value) for value in finite_values):
        failures.append("Transform contains non-finite values")
    return {
        "passed": not failures,
        "failures": failures,
        "n_original_hits": len(hits),
        "n_transformed_hits": len(transformed_hits),
        "locations": locations,
        "max_residual_difference_cm": max_residual_difference,
        "sides_by_orientation": {
            orientation: sorted(values) for orientation, values in sides.items()
        },
    }
