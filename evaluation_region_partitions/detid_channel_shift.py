#!/usr/bin/env python3
"""ROOT-independent detector-ID channel-sequence transformation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, isfinite, sqrt
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class PlaneKey:
    detector: str
    system: int
    plane: int
    view: int

    @property
    def coordinate(self) -> str:
        return "x" if self.view == 1 else "y"


@dataclass(frozen=True)
class ChannelAddress:
    det_id: int
    key: PlaneKey
    ordinal: int
    centre_cm: float
    low_cm: float
    high_cm: float
    hierarchy: tuple[int, ...]
    z_cm: float = 0.0


@dataclass(frozen=True)
class ChannelLayout:
    key: PlaneKey
    channels: tuple[ChannelAddress, ...]

    @classmethod
    def ordered(cls, key: PlaneKey, channels: Iterable[ChannelAddress]):
        values = sorted(channels, key=lambda item: (item.centre_cm, item.det_id))
        values = tuple(replace(item, ordinal=index) for index, item in enumerate(values))
        layout = cls(key, values)
        layout.validate()
        return layout

    def validate(self) -> None:
        if not self.channels:
            raise ValueError(f"Empty channel layout for {self.key}")
        ids = [item.det_id for item in self.channels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate detector ID in {self.key}")
        centres = [item.centre_cm for item in self.channels]
        if not all(isfinite(value) for value in centres):
            raise ValueError(f"Non-finite channel centre in {self.key}")
        if any(right <= left for left, right in zip(centres, centres[1:])):
            raise ValueError(f"Channel centres are not strictly ordered in {self.key}")
        for index, item in enumerate(self.channels):
            if item.key != self.key or item.ordinal != index:
                raise ValueError(f"Inconsistent channel address at {self.key}/{index}")
            if not item.low_cm < item.high_cm:
                raise ValueError(f"Invalid channel interval for {item.det_id}")

    @property
    def by_det_id(self) -> dict[int, ChannelAddress]:
        return {item.det_id: item for item in self.channels}


@dataclass(frozen=True)
class ChannelShift:
    side: str
    anchor_det_id: int
    edge_det_id: int
    anchor_ordinal: int
    edge_ordinal: int
    offset: int
    requested_shift_cm: float
    realized_anchor_shift_cm: float
    anchor_error_cm: float


@dataclass(frozen=True)
class TranslationShift:
    """One constant ordinal offset approximating a physical translation."""

    selection_policy: str
    offset: int
    requested_shift_cm: float
    reference_source_det_id: int
    reference_destination_det_id: int
    realized_shift_mean_cm: float
    realized_shift_min_cm: float
    realized_shift_max_cm: float
    translation_error_cm: float
    rms_error_cm: float
    max_abs_error_cm: float
    quantization_tolerance_cm: float
    mapped_hits: int
    dropped_hits: int


@dataclass(frozen=True)
class MappedHit:
    source_det_id: int
    destination_det_id: int | None
    source_ordinal: int
    destination_ordinal: int
    status: str
    source_coordinate_cm: float
    destination_coordinate_cm: float | None
    realized_shift_cm: float | None


@dataclass(frozen=True)
class TrackLine:
    """SciFi track line expressed at one reference z coordinate."""

    reference_z_cm: float
    reference_x_cm: float
    reference_y_cm: float
    slope_xz: float
    slope_yz: float

    def coordinate_at(self, axis: str, z_cm: float) -> float:
        if axis == "x":
            origin, slope = self.reference_x_cm, self.slope_xz
        elif axis == "y":
            origin, slope = self.reference_y_cm, self.slope_yz
        else:
            raise ValueError(f"Unsupported track axis {axis!r}")
        values = (origin, slope, self.reference_z_cm, z_cm)
        if not all(isfinite(value) for value in values):
            raise ValueError("Track projection contains a non-finite value")
        return origin + slope * (z_cm - self.reference_z_cm)


@dataclass(frozen=True)
class StationActivityPoint:
    """Unweighted SciFi activity averages for both views of one station."""

    station: int
    x_cm: float
    y_cm: float
    z_x_cm: float
    z_y_cm: float
    n_x_hits: int
    n_y_hits: int


@dataclass(frozen=True)
class TwoStationActivityLine:
    """Straight activity axis defined by the first two active SciFi stations."""

    direction: str
    first: StationActivityPoint
    second: StationActivityPoint
    line: TrackLine


@dataclass(frozen=True)
class PlaneActivityPoint:
    """Unweighted activity average for one SciFi station/orientation plane."""

    station: int
    view: int
    coordinate_cm: float
    z_cm: float
    n_hits: int


@dataclass(frozen=True)
class AllPlaneActivityLine:
    """Straight activity axis fitted to every nonempty SciFi plane average."""

    direction: str
    x_points: tuple[PlaneActivityPoint, ...]
    y_points: tuple[PlaneActivityPoint, ...]
    line: TrackLine


def build_all_plane_activity_line(
    coordinates: Mapping[tuple[int, int], Iterable[float]],
    plane_z_cm: Mapping[tuple[int, int], float],
    pz_sign: int = 1,
) -> AllPlaneActivityLine:
    """Fit x-z and y-z lines to all nonempty SciFi plane averages.

    View 1 measures x and view 0 measures y.  Every active plane contributes
    one equal-weight point, independently of the number of digis in that
    plane.  This preserves the existing ``avg_scifiN_x/y`` convention while
    avoiding a fit dominated by a high-occupancy shower plane.
    """
    if pz_sign not in (-1, 1):
        raise ValueError(f"pz_sign must be -1 or +1, got {pz_sign}")
    values = {
        key: tuple(float(value) for value in samples)
        for key, samples in coordinates.items()
    }
    for key, samples in values.items():
        if not all(isfinite(value) for value in samples):
            raise ValueError(f"Non-finite SciFi activity coordinate in {key}")

    station_order = tuple(range(1, 6) if pz_sign > 0 else range(5, 0, -1))

    def points_for_view(view: int) -> tuple[PlaneActivityPoint, ...]:
        points = []
        for station in station_order:
            samples = values.get((station, view), ())
            if not samples:
                continue
            z_cm = float(plane_z_cm[(station, view)])
            if not isfinite(z_cm):
                raise ValueError(
                    f"Non-finite SciFi plane z coordinate at station {station}, view {view}"
                )
            points.append(PlaneActivityPoint(
                station=station,
                view=view,
                coordinate_cm=sum(samples) / len(samples),
                z_cm=z_cm,
                n_hits=len(samples),
            ))
        if len(points) < 2:
            axis = "x" if view == 1 else "y"
            raise ValueError(
                f"All-plane activity line needs at least two active {axis}-projection planes"
            )
        return tuple(points)

    def linear_fit(points: tuple[PlaneActivityPoint, ...]) -> tuple[float, float, float]:
        mean_z = sum(point.z_cm for point in points) / len(points)
        mean_coordinate = sum(point.coordinate_cm for point in points) / len(points)
        denominator = sum((point.z_cm - mean_z) ** 2 for point in points)
        if denominator == 0.0:
            raise ValueError("Active SciFi planes do not have distinct z coordinates")
        slope = sum(
            (point.z_cm - mean_z) * (point.coordinate_cm - mean_coordinate)
            for point in points
        ) / denominator
        if not all(isfinite(value) for value in (mean_z, mean_coordinate, slope)):
            raise ValueError("All-plane activity fit produced a non-finite value")
        return mean_z, mean_coordinate, slope

    x_points = points_for_view(1)
    y_points = points_for_view(0)
    mean_z_x, mean_x, slope_xz = linear_fit(x_points)
    mean_z_y, mean_y, slope_yz = linear_fit(y_points)
    reference_z = 0.5 * (mean_z_x + mean_z_y)
    line = TrackLine(
        reference_z_cm=reference_z,
        reference_x_cm=mean_x + slope_xz * (reference_z - mean_z_x),
        reference_y_cm=mean_y + slope_yz * (reference_z - mean_z_y),
        slope_xz=slope_xz,
        slope_yz=slope_yz,
    )
    return AllPlaneActivityLine(
        direction="positive_z" if pz_sign > 0 else "negative_z",
        x_points=x_points,
        y_points=y_points,
        line=line,
    )


def build_five_station_activity_line(
    coordinates_by_plane: Mapping[tuple[int, int], Iterable[float]],
    z_by_plane: Mapping[tuple[int, int], float],
    pz_sign: int = 1,
) -> AllPlaneActivityLine:
    """Fit equal-weight station averages after requiring all five X/Y planes.

    This is the production definition of the reconstruction-free Run-8329
    transform.  Every station must contain at least one vertical (X) and one
    horizontal (Y) Digi_ScifiHit.  Requiring the complete ten-plane set here
    prevents the more permissive validation-only all-active-plane fit from
    silently accepting a two-plane event.
    """
    required = {(station, view) for station in range(1, 6) for view in (0, 1)}
    values = {
        key: tuple(float(value) for value in samples)
        for key, samples in coordinates_by_plane.items()
    }
    missing = sorted(
        key for key in required
        if key not in values or not values[key]
    )
    if missing:
        formatted = ", ".join(f"station {station} view {view}" for station, view in missing)
        raise ValueError(f"Five-station activity line is missing: {formatted}")
    missing_z = sorted(key for key in required if key not in z_by_plane)
    if missing_z:
        raise ValueError(f"Five-station activity line lacks plane z values: {missing_z}")
    selected_coordinates = {
        key: values[key] for key in sorted(required)
    }
    selected_z = {key: float(z_by_plane[key]) for key in sorted(required)}
    return build_all_plane_activity_line(selected_coordinates, selected_z, pz_sign)


def build_two_station_activity_line(
    coordinates: Mapping[tuple[int, int], Iterable[float]],
    plane_z_cm: Mapping[tuple[int, int], float],
    pz_sign: int = 1,
) -> TwoStationActivityLine:
    """Build x-z and y-z lines from the first two stations active in both views.

    View 1 measures x and view 0 measures y.  The arithmetic mean deliberately
    matches the existing ``avg_scifiN_x/y`` feature definition.
    """
    if pz_sign not in (-1, 1):
        raise ValueError(f"pz_sign must be -1 or +1, got {pz_sign}")
    values = {
        key: tuple(float(value) for value in samples)
        for key, samples in coordinates.items()
    }
    for key, samples in values.items():
        if not all(isfinite(value) for value in samples):
            raise ValueError(f"Non-finite SciFi activity coordinate in {key}")
    station_order = range(1, 6) if pz_sign > 0 else range(5, 0, -1)
    active = [
        station for station in station_order
        if values.get((station, 1)) and values.get((station, 0))
    ]
    if len(active) < 2:
        raise ValueError(
            "Two-station activity line needs at least two SciFi stations "
            "active in both orientations"
        )

    def activity_point(station: int) -> StationActivityPoint:
        x_values = values[(station, 1)]
        y_values = values[(station, 0)]
        z_x = float(plane_z_cm[(station, 1)])
        z_y = float(plane_z_cm[(station, 0)])
        if not all(isfinite(value) for value in (z_x, z_y)):
            raise ValueError(f"Non-finite SciFi plane z coordinate at station {station}")
        return StationActivityPoint(
            station=station,
            x_cm=sum(x_values) / len(x_values),
            y_cm=sum(y_values) / len(y_values),
            z_x_cm=z_x,
            z_y_cm=z_y,
            n_x_hits=len(x_values),
            n_y_hits=len(y_values),
        )

    first, second = activity_point(active[0]), activity_point(active[1])
    delta_z_x = second.z_x_cm - first.z_x_cm
    delta_z_y = second.z_y_cm - first.z_y_cm
    if delta_z_x == 0.0 or delta_z_y == 0.0:
        raise ValueError("First two active SciFi stations do not have distinct z coordinates")
    slope_xz = (second.x_cm - first.x_cm) / delta_z_x
    slope_yz = (second.y_cm - first.y_cm) / delta_z_y
    reference_z = 0.5 * (first.z_x_cm + first.z_y_cm)
    line = TrackLine(
        reference_z_cm=reference_z,
        reference_x_cm=first.x_cm + slope_xz * (reference_z - first.z_x_cm),
        reference_y_cm=first.y_cm + slope_yz * (reference_z - first.z_y_cm),
        slope_xz=slope_xz,
        slope_yz=slope_yz,
    )
    return TwoStationActivityLine(
        direction="positive_z" if pz_sign > 0 else "negative_z",
        first=first,
        second=second,
        line=line,
    )


@dataclass(frozen=True)
class AxisDecision:
    """Auditable choice of one coordinate and the track entry edge."""

    slope_xz: float
    slope_yz: float
    selected_axis: str
    selected_view: int
    selected_slope: float
    axis_dominance_ratio: float
    pz_sign: int
    direction_source: str
    travel_slope: float
    entry_side: str


@dataclass(frozen=True)
class EdgeDecision:
    """Select the target edge without changing the one-axis decision."""

    selection_policy: str
    slope_entry_side: str
    closest_side: str
    target_side: str
    track_coordinate_cm: float
    low_edge_coordinate_cm: float
    high_edge_coordinate_cm: float
    distance_to_low_edge_cm: float
    distance_to_high_edge_cm: float
    closest_edge_distance_cm: float
    near_edge_threshold_cm: float
    near_edge_override: bool


def choose_transform_axis(
    slope_xz: float,
    slope_yz: float,
    pz_sign: int,
    direction_source: str,
    min_abs_slope: float = 0.0,
    min_axis_dominance_ratio: float = 1.0,
) -> AxisDecision:
    """Select the dominant projection and the side from which it enters.

    ``pz_sign`` supplies the otherwise missing direction of travel.  For a
    downstream-going line, a positive selected slope enters from the low edge;
    a negative slope enters from the high edge.
    """
    sx, sy = float(slope_xz), float(slope_yz)
    if not all(isfinite(value) for value in (sx, sy)):
        raise ValueError("Track slopes must be finite")
    if int(pz_sign) not in (-1, 1):
        raise ValueError("pz_sign must be either -1 or +1")
    if not direction_source:
        raise ValueError("direction_source must be recorded")
    if min_abs_slope < 0 or min_axis_dominance_ratio < 1:
        raise ValueError("Invalid slope-selection thresholds")
    ax, ay = abs(sx), abs(sy)
    if ax == ay:
        raise ValueError("Ambiguous transform axis: equal absolute slopes")
    if ax > ay:
        axis, view, selected, smaller = "x", 1, sx, ay
    else:
        axis, view, selected, smaller = "y", 0, sy, ax
    if abs(selected) <= min_abs_slope:
        raise ValueError(
            f"Dominant absolute slope {abs(selected)} is not above {min_abs_slope}"
        )
    ratio = abs(selected) / smaller if smaller > 0 else inf
    if ratio < min_axis_dominance_ratio:
        raise ValueError(
            f"Axis dominance ratio {ratio} is below {min_axis_dominance_ratio}"
        )
    travel_slope = selected * int(pz_sign)
    if travel_slope == 0:
        raise ValueError("Selected travel slope is zero")
    return AxisDecision(
        slope_xz=sx,
        slope_yz=sy,
        selected_axis=axis,
        selected_view=view,
        selected_slope=selected,
        axis_dominance_ratio=ratio,
        pz_sign=int(pz_sign),
        direction_source=str(direction_source),
        travel_slope=travel_slope,
        entry_side="low" if travel_slope > 0 else "high",
    )


def choose_target_edge(
    track_coordinate_cm: float,
    low_edge_coordinate_cm: float,
    high_edge_coordinate_cm: float,
    slope_entry_side: str,
    near_edge_threshold_cm: float,
) -> EdgeDecision:
    """Prefer the closest edge for a track already near an active edge.

    Edge coordinates are the centres of the first and last active channels,
    matching the destination used by the discrete detector-ID mapper.  Outside
    the configured proximity band, the signed-slope entry side is retained.
    """
    values = (
        float(track_coordinate_cm),
        float(low_edge_coordinate_cm),
        float(high_edge_coordinate_cm),
        float(near_edge_threshold_cm),
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("Edge-decision coordinates and threshold must be finite")
    track, low, high, threshold = values
    if low >= high:
        raise ValueError("Low edge coordinate must be below high edge coordinate")
    if slope_entry_side not in {"low", "high"}:
        raise ValueError(f"Unsupported slope entry side {slope_entry_side!r}")
    if threshold < 0:
        raise ValueError("Near-edge threshold must be non-negative")

    distance_low = abs(track - low)
    distance_high = abs(high - track)
    closest = "low" if distance_low <= distance_high else "high"
    closest_distance = min(distance_low, distance_high)
    override = closest_distance <= threshold
    target = closest if override else slope_entry_side
    return EdgeDecision(
        selection_policy=(
            "nearest_edge_within_threshold" if override else "slope_entry_side"
        ),
        slope_entry_side=slope_entry_side,
        closest_side=closest,
        target_side=target,
        track_coordinate_cm=track,
        low_edge_coordinate_cm=low,
        high_edge_coordinate_cm=high,
        distance_to_low_edge_cm=distance_low,
        distance_to_high_edge_cm=distance_high,
        closest_edge_distance_cm=closest_distance,
        near_edge_threshold_cm=threshold,
        near_edge_override=override,
    )


def choose_anchor(
    layout: ChannelLayout,
    hit_det_ids: Iterable[int],
    side: str,
    requested_shift_cm: float,
) -> ChannelShift:
    """Choose the hit predicted closest to the requested destination edge."""
    if side not in {"low", "high"}:
        raise ValueError(f"Unsupported side {side!r}")
    if not isfinite(requested_shift_cm):
        raise ValueError("Requested shift must be finite")
    by_id = layout.by_det_id
    unique_ids = sorted(set(int(value) for value in hit_det_ids))
    missing = [value for value in unique_ids if value not in by_id]
    if missing:
        raise KeyError(f"Hits absent from {layout.key}: {missing[:5]}")
    if not unique_ids:
        raise ValueError(f"No anchor candidates in {layout.key}")
    edge = layout.channels[0 if side == "low" else -1]
    anchor = min(
        (by_id[value] for value in unique_ids),
        key=lambda item: (
            abs(item.centre_cm + requested_shift_cm - edge.centre_cm),
            item.ordinal,
            item.det_id,
        ),
    )
    realized = edge.centre_cm - anchor.centre_cm
    return ChannelShift(
        side=side,
        anchor_det_id=anchor.det_id,
        edge_det_id=edge.det_id,
        anchor_ordinal=anchor.ordinal,
        edge_ordinal=edge.ordinal,
        offset=edge.ordinal - anchor.ordinal,
        requested_shift_cm=float(requested_shift_cm),
        realized_anchor_shift_cm=realized,
        anchor_error_cm=realized - float(requested_shift_cm),
    )


def choose_translation_offset(
    layout: ChannelLayout,
    hit_det_ids: Iterable[int],
    requested_shift_cm: float,
) -> TranslationShift:
    """Choose a constant channel offset closest to a physical translation.

    Unlike :func:`choose_anchor`, this policy never forces an observed hit to
    the local detector edge.  It selects an offset from the complete detector
    layout, independently of the event occupancy, and only then applies normal
    edge dropping to the observed hits.  This prevents sparse edge hits from
    biasing the translation toward an incorrect smaller offset.
    """
    if not isfinite(requested_shift_cm):
        raise ValueError("Requested shift must be finite")
    by_id = layout.by_det_id
    unique_ids = sorted(set(int(value) for value in hit_det_ids))
    missing = [value for value in unique_ids if value not in by_id]
    if missing:
        raise KeyError(f"Hits absent from {layout.key}: {missing[:5]}")
    if not unique_ids:
        raise ValueError(f"No translation candidates in {layout.key}")
    hits = [by_id[value] for value in unique_ids]
    size = len(layout.channels)
    if requested_shift_cm < 0:
        offsets = range(-(size - 1), 1)
    elif requested_shift_cm > 0:
        offsets = range(0, size)
    else:
        offsets = (0,)

    best = None
    for offset in offsets:
        pairs = []
        for source in layout.channels:
            destination_ordinal = source.ordinal + offset
            if 0 <= destination_ordinal < size:
                destination = layout.channels[destination_ordinal]
                realized = destination.centre_cm - source.centre_cm
                pairs.append((source, destination, realized))
        if not pairs:
            continue
        errors = [item[2] - requested_shift_cm for item in pairs]
        mean_squared_error = sum(value * value for value in errors) / len(errors)
        score = (mean_squared_error, abs(offset), offset)
        if best is None or score < best[0]:
            best = (score, offset, pairs, errors)
    if best is None:
        raise RuntimeError(f"No valid translation offset for {layout.key}")

    _, offset, layout_pairs, _ = best
    event_pairs = []
    for source in hits:
        destination_ordinal = source.ordinal + offset
        if 0 <= destination_ordinal < size:
            destination = layout.channels[destination_ordinal]
            realized = destination.centre_cm - source.centre_cm
            event_pairs.append((source, destination, realized))
    statistic_pairs = event_pairs if event_pairs else layout_pairs
    errors = [item[2] - requested_shift_cm for item in statistic_pairs]
    realized_shifts = [item[2] for item in statistic_pairs]
    reference_index = min(
        range(len(statistic_pairs)),
        key=lambda index: (
            abs(errors[index]), statistic_pairs[index][0].ordinal,
            statistic_pairs[index][0].det_id,
        ),
    )
    reference_source, reference_destination, _ = statistic_pairs[reference_index]
    mean_realized = sum(realized_shifts) / len(realized_shifts)
    spacings = [
        right.centre_cm - left.centre_cm
        for left, right in zip(layout.channels, layout.channels[1:])
    ]
    return TranslationShift(
        selection_policy="closest_physical_translation",
        offset=offset,
        requested_shift_cm=float(requested_shift_cm),
        reference_source_det_id=reference_source.det_id,
        reference_destination_det_id=reference_destination.det_id,
        realized_shift_mean_cm=mean_realized,
        realized_shift_min_cm=min(realized_shifts),
        realized_shift_max_cm=max(realized_shifts),
        translation_error_cm=mean_realized - float(requested_shift_cm),
        rms_error_cm=sqrt(sum(value * value for value in errors) / len(errors)),
        max_abs_error_cm=max(abs(value) for value in errors),
        quantization_tolerance_cm=0.5 * max(spacings) if spacings else 0.0,
        mapped_hits=len(event_pairs),
        dropped_hits=len(hits) - len(event_pairs),
    )
def map_detector_ids(
    layout: ChannelLayout,
    source_det_ids: Iterable[int],
    shift: ChannelShift | TranslationShift,
) -> list[MappedHit]:
    """Apply a constant ordinal offset without clamping or wrapping."""
    by_id = layout.by_det_id
    output = []
    destinations = set()
    sources = set()
    for raw_det_id in source_det_ids:
        det_id = int(raw_det_id)
        if det_id in sources:
            raise ValueError(f"Duplicate source detector ID {det_id} in {layout.key}")
        sources.add(det_id)
        if det_id not in by_id:
            raise KeyError(f"Detector ID {det_id} absent from {layout.key}")
        source = by_id[det_id]
        destination_ordinal = source.ordinal + shift.offset
        if destination_ordinal < 0:
            mapped = MappedHit(
                det_id, None, source.ordinal, destination_ordinal, "outside_low",
                source.centre_cm, None, None,
            )
        elif destination_ordinal >= len(layout.channels):
            mapped = MappedHit(
                det_id, None, source.ordinal, destination_ordinal, "outside_high",
                source.centre_cm, None, None,
            )
        else:
            destination = layout.channels[destination_ordinal]
            if destination.det_id in destinations:
                raise ValueError(
                    f"Destination collision at {destination.det_id} in {layout.key}"
                )
            destinations.add(destination.det_id)
            mapped = MappedHit(
                det_id, destination.det_id, source.ordinal, destination_ordinal,
                "mapped", source.centre_cm, destination.centre_cm,
                destination.centre_cm - source.centre_cm,
            )
        output.append(mapped)
    if len(output) != len(sources):
        raise AssertionError("Mapped-hit count closure failed")
    return output


def summarize_mapping(mapped: Iterable[MappedHit]) -> dict:
    values = list(mapped)
    counts = {name: sum(item.status == name for item in values) for name in (
        "mapped", "outside_low", "outside_high",
    )}
    shifts = [item.realized_shift_cm for item in values if item.realized_shift_cm is not None]
    return {
        "input_hits": len(values),
        "counts": counts,
        "count_closure": sum(counts.values()) == len(values),
        "realized_shift_min_cm": min(shifts) if shifts else None,
        "realized_shift_max_cm": max(shifts) if shifts else None,
        "realized_shift_mean_cm": sum(shifts) / len(shifts) if shifts else None,
        "realized_shift_spread_cm": max(shifts) - min(shifts) if shifts else None,
    }
