#!/usr/bin/env python3
"""SNDSW/ROOT adapters for the SciFi edge-transform validation."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import ROOT
import SndlhcGeo

from scifi_edge_transform import (
    ActiveInterval,
    HitCoordinate,
    PlaneGeometry,
    PlaneKey,
    fit_linear_projection,
)
from scifi_sipm_event_display import load_sipm_geometry
from sndsw_event_display_common import open_event_tree


ROOT.gROOT.SetBatch(True)


@dataclass(frozen=True, order=True)
class MuFilterPlaneKey:
    system: int
    plane: int
    view: int

    @property
    def orientation(self) -> str:
        return "vertical" if self.view == 1 else "horizontal"

    @property
    def coordinate(self) -> str:
        return "x" if self.view == 1 else "y"


@dataclass(frozen=True)
class MuFilterPlaneGeometry:
    key: MuFilterPlaneKey
    z_cm: float
    low_edge_cm: float
    high_edge_cm: float
    intervals: tuple[ActiveInterval, ...]
    z_spread_cm: float


@dataclass(frozen=True)
class MuFilterPlaneTransform:
    key: MuFilterPlaneKey
    z_cm: float
    side: str
    low_edge_cm: float
    high_edge_cm: float
    target_cm: float
    track_original_cm: float
    track_transformed_cm: float
    delta_cm: float


def encode_scifi_detid(station: int, view: int, mat: int, sipm: int, channel: int) -> int:
    return station * 1_000_000 + view * 100_000 + mat * 10_000 + sipm * 1_000 + channel


def decode_scifi_detid(detector_id: int) -> tuple[int, int, int, int, int]:
    detector_id = int(detector_id)
    return (
        detector_id // 1_000_000,
        (detector_id // 100_000) % 10,
        (detector_id // 10_000) % 10,
        (detector_id // 1_000) % 10,
        detector_id % 1_000,
    )


def event_identity(tree) -> tuple[int, int]:
    header = tree.EventHeader
    run = int(header.GetRunId()) if hasattr(header, "GetRunId") else -1
    event = -1
    for method in ("GetEventNumber", "GetMCEntryNumber"):
        if hasattr(header, method):
            event = int(getattr(header, method)())
            break
    return run, event


class GeometryAdapter:
    """One loaded geofile, reinitialized for each real-data event."""

    def __init__(self, geofile: str):
        self.geofile = str(geofile)
        self.interface = SndlhcGeo.GeoInterface(self.geofile)
        self.scifi = self.interface.modules["Scifi"]
        self.mufilter = self.interface.modules["MuFilter"]
        template, validation = load_sipm_geometry(self.scifi)
        self.half_width_cm = float(template[0].half_width_cm)
        self.template_validation = validation

    def initialize_event(self, event_header):
        self.scifi.InitEvent(event_header)
        self.mufilter.InitEvent(event_header)

    def build_planes(self) -> dict[PlaneKey, PlaneGeometry]:
        output = {}
        for station in range(1, 6):
            for view in (0, 1):
                key = PlaneKey(station, view)
                intervals = []
                z_values = []
                seen = set()
                for mat in range(3):
                    for sipm in range(4):
                        for channel in range(128):
                            detector_id = encode_scifi_detid(
                                station, view, mat, sipm, channel
                            )
                            decoded = decode_scifi_detid(detector_id)
                            expected = (station, view, mat, sipm, channel)
                            if decoded != expected:
                                raise RuntimeError(
                                    f"SciFi detector-ID round trip failed: {detector_id} "
                                    f"decoded={decoded}, expected={expected}"
                                )
                            point_a, point_b = ROOT.TVector3(), ROOT.TVector3()
                            self.scifi.GetSiPMPosition(detector_id, point_a, point_b)
                            measured = (
                                0.5 * (float(point_a.X()) + float(point_b.X()))
                                if view == 1
                                else 0.5 * (float(point_a.Y()) + float(point_b.Y()))
                            )
                            z_values.extend((float(point_a.Z()), float(point_b.Z())))
                            channel_key = (mat, sipm, channel)
                            if channel_key in seen:
                                raise RuntimeError(f"Duplicate channel in plane {key}: {channel_key}")
                            seen.add(channel_key)
                            intervals.append(ActiveInterval(
                                measured - self.half_width_cm,
                                measured + self.half_width_cm,
                                mat, sipm, channel,
                            ))
                intervals.sort(key=lambda item: item.low_cm)
                if len(intervals) != 3 * 4 * 128:
                    raise RuntimeError(f"Plane {key} has {len(intervals)} channels")
                z_cm = float(statistics.median(z_values))
                z_spread = max(z_values) - min(z_values)
                # The three mats in one station/view are physically staggered
                # by roughly 0.49 cm in the Run-8329 geometry.  The validation
                # transform is defined at the median plane z, so accept that
                # construction while still rejecting a multi-plane mixture.
                if z_spread > 1.0:
                    raise RuntimeError(
                        f"Plane {key} endpoint z spread is unexpectedly large: {z_spread} cm"
                    )
                output[key] = PlaneGeometry(
                    key=key,
                    z_cm=z_cm,
                    low_edge_cm=min(item.low_cm for item in intervals),
                    high_edge_cm=max(item.high_cm for item in intervals),
                    intervals=tuple(intervals),
                    z_spread_cm=z_spread,
                )
        return output

    def build_mufilter_planes(self) -> dict[MuFilterPlaneKey, MuFilterPlaneGeometry]:
        """Enumerate complete Veto/US/DS bar geometry by measured plane."""
        plane_counts = {
            1: int(self.mufilter.GetConfParI("MuFilter/NVetoPlanes")),
            2: int(self.mufilter.GetConfParI("MuFilter/NUpstreamPlanes")),
            3: int(self.mufilter.GetConfParI("MuFilter/NDownstreamPlanes")),
        }
        bar_counts = {
            1: int(self.mufilter.GetConfParI("MuFilter/NVetoBars")),
            2: int(self.mufilter.GetConfParI("MuFilter/NUpstreamBars")),
            3: int(self.mufilter.GetConfParI("MuFilter/NDownstreamBars")),
        }
        half_width = {
            (1, 0): 0.5 * float(self.mufilter.GetConfParF("MuFilter/VetoBarY")),
            (1, 1): 0.5 * float(self.mufilter.GetConfParF("MuFilter/Veto3BarX")),
            (2, 0): 0.5 * float(self.mufilter.GetConfParF("MuFilter/UpstreamBarY")),
            (3, 0): 0.5 * float(self.mufilter.GetConfParF("MuFilter/DownstreamBarY")),
            (3, 1): 0.5 * float(self.mufilter.GetConfParF("MuFilter/DownstreamBarX_ver")),
        }
        output = {}
        for system in (1, 2, 3):
            for physical_plane in range(plane_counts[system]):
                if system == 3:
                    views = (0, 1) if physical_plane < 3 else (1,)
                else:
                    # Veto plane 3 is vertical; all other Veto and US planes
                    # are horizontal in the 2024 geometry.
                    views = (1,) if system == 1 and physical_plane == 2 else (0,)
                for view in views:
                    logical_plane = (
                        2 * physical_plane
                        + (1 if view == 1 and physical_plane < 3 else 0)
                        if system == 3 else physical_plane
                    )
                    key = MuFilterPlaneKey(system, logical_plane, view)
                    intervals, z_values = [], []
                    for bar in range(bar_counts[system]):
                        encoded_bar = bar + 60 if system == 3 and view == 1 else bar
                        detector_id = system * 10_000 + physical_plane * 1_000 + encoded_bar
                        point_a, point_b = ROOT.TVector3(), ROOT.TVector3()
                        self.mufilter.GetPosition(detector_id, point_a, point_b)
                        measured = (
                            0.5 * (float(point_a.X()) + float(point_b.X()))
                            if view == 1
                            else 0.5 * (float(point_a.Y()) + float(point_b.Y()))
                        )
                        z_values.extend((float(point_a.Z()), float(point_b.Z())))
                        width = half_width[(system, view)]
                        intervals.append(ActiveInterval(
                            measured - width, measured + width,
                            system, logical_plane, bar,
                        ))
                    intervals.sort(key=lambda item: item.low_cm)
                    output[key] = MuFilterPlaneGeometry(
                        key=key,
                        z_cm=float(statistics.median(z_values)),
                        low_edge_cm=min(item.low_cm for item in intervals),
                        high_edge_cm=max(item.high_cm for item in intervals),
                        intervals=tuple(intervals),
                        z_spread_cm=max(z_values) - min(z_values),
                    )
        return output


def extract_digi_hits(tree, geometry: GeometryAdapter) -> tuple[list[HitCoordinate], list[dict]]:
    collection = getattr(tree, "Digi_ScifiHits", None)
    if collection is None:
        raise RuntimeError("Raw event tree has no Digi_ScifiHits branch")
    hits = []
    display = []
    observed_channels = set()
    for hit in collection:
        detector_id = int(hit.GetDetectorID())
        observed = (
            int(hit.GetStation()), int(bool(hit.isVertical())), int(hit.GetMat()),
            int(hit.GetSiPM()), int(hit.GetSiPMChan()),
        )
        decoded = decode_scifi_detid(detector_id)
        encoded = encode_scifi_detid(*observed)
        if decoded != observed or encoded != detector_id:
            raise RuntimeError(
                f"Observed SciFi detector-ID mismatch for {detector_id}: "
                f"decoded={decoded}, accessors={observed}, encoded={encoded}"
            )
        if observed in observed_channels:
            raise RuntimeError(f"Multiple Digi_ScifiHits use channel {observed}")
        observed_channels.add(observed)
        point_a, point_b = ROOT.TVector3(), ROOT.TVector3()
        geometry.scifi.GetSiPMPosition(detector_id, point_a, point_b)
        measured = (
            0.5 * (float(point_a.X()) + float(point_b.X()))
            if observed[1] == 1
            else 0.5 * (float(point_a.Y()) + float(point_b.Y()))
        )
        record = HitCoordinate(
            key=PlaneKey(observed[0], observed[1]),
            coordinate_cm=measured,
            detector_id=detector_id,
            qdc=float(hit.GetSignal()),
            time_ns=float(hit.GetTime()),
            valid=bool(hit.isValid()),
        )
        hits.append(record)
        display.append({
            **asdict(record),
            "station": observed[0], "view": observed[1],
            "mat": observed[2], "sipm": observed[3], "channel": observed[4],
            "x_a_cm": float(point_a.X()), "y_a_cm": float(point_a.Y()),
            "z_a_cm": float(point_a.Z()), "x_b_cm": float(point_b.X()),
            "y_b_cm": float(point_b.Y()), "z_b_cm": float(point_b.Z()),
            "x_mid_cm": 0.5 * (float(point_a.X()) + float(point_b.X())),
            "y_mid_cm": 0.5 * (float(point_a.Y()) + float(point_b.Y())),
            "z_mid_cm": 0.5 * (float(point_a.Z()) + float(point_b.Z())),
        })
    return hits, display


def extract_mufilter_hits(tree, geometry: GeometryAdapter) -> list[dict]:
    """Extract geometry positions for Veto, US, and DS digis."""
    collection = getattr(tree, "Digi_MuFilterHits", None)
    if collection is None:
        raise RuntimeError("Raw event tree has no Digi_MuFilterHits branch")
    output = []
    labels = {1: "Veto", 2: "US", 3: "DS"}
    for hit in collection:
        detector_id = int(hit.GetDetectorID())
        system = int(hit.GetSystem())
        if system not in labels:
            raise RuntimeError(f"Unknown MuFilter system {system} for detector ID {detector_id}")
        point_a, point_b = ROOT.TVector3(), ROOT.TVector3()
        geometry.mufilter.GetPosition(detector_id, point_a, point_b)
        qdc = 0.0
        try:
            qdc = sum(max(0.0, float(value)) for _, value in hit.GetAllSignals())
        except Exception:
            qdc = math.nan
        output.append({
            "detector_id": detector_id,
            "system": system,
            "system_label": labels[system],
            "station": int((detector_id // 1000) % 10 + 1),
            "vertical": bool(hit.isVertical()),
            "valid": bool(hit.isValid()),
            "qdc": qdc,
            "x_a_cm": float(point_a.X()), "y_a_cm": float(point_a.Y()),
            "z_a_cm": float(point_a.Z()), "x_b_cm": float(point_b.X()),
            "y_b_cm": float(point_b.Y()), "z_b_cm": float(point_b.Z()),
            "x_mid_cm": 0.5 * (float(point_a.X()) + float(point_b.X())),
            "y_mid_cm": 0.5 * (float(point_a.Y()) + float(point_b.Y())),
            "z_mid_cm": 0.5 * (float(point_a.Z()) + float(point_b.Z())),
        })
    return output


def mufilter_plane_key(hit: dict) -> MuFilterPlaneKey:
    system = int(hit["system"])
    physical_plane = (int(hit["detector_id"]) % 10_000) // 1_000
    view = int(bool(hit["vertical"]))
    logical_plane = (
        2 * physical_plane + (1 if view == 1 and physical_plane < 3 else 0)
        if system == 3 else physical_plane
    )
    return MuFilterPlaneKey(system, logical_plane, view)


def transform_mufilter_hits(
    hits: list[dict], planes: dict[MuFilterPlaneKey, MuFilterPlaneGeometry],
    projection_x, projection_y, side_by_view: dict[int, str],
    inward_margin_cm: float = 0.0, residual_tolerance_cm: float = 1.0e-9,
) -> tuple[list[dict], dict]:
    """Transform all MuFilter digis using the SciFi-track extrapolation."""
    transforms = {}
    for key, plane in planes.items():
        side = side_by_view[key.view]
        projection = projection_x if key.view == 1 else projection_y
        track_position = float(projection.at(plane.z_cm))
        target = (
            plane.low_edge_cm + inward_margin_cm
            if side == "low" else plane.high_edge_cm - inward_margin_cm
        )
        if target < plane.low_edge_cm or target > plane.high_edge_cm:
            raise ValueError(f"MuFilter target outside active plane {key}")
        delta = target - track_position
        transforms[key] = MuFilterPlaneTransform(
            key=key, z_cm=plane.z_cm, side=side,
            low_edge_cm=plane.low_edge_cm, high_edge_cm=plane.high_edge_cm,
            target_cm=target, track_original_cm=track_position,
            track_transformed_cm=track_position + delta, delta_cm=delta,
        )

    output = []
    location_counts = {"inside_active": 0, "inside_gap": 0, "outside_geometry": 0}
    maximum_residual = 0.0
    for hit in hits:
        key = mufilter_plane_key(hit)
        if key not in planes or key not in transforms:
            raise KeyError(f"No MuFilter geometry/transform for hit plane {key}")
        plane, transform = planes[key], transforms[key]
        coordinate = "x" if key.view == 1 else "y"
        original = float(hit[f"{coordinate}_mid_cm"])
        shifted = original + transform.delta_cm
        destination = None
        location = "outside_geometry"
        for interval in plane.intervals:
            if interval.low_cm - 1e-10 <= shifted <= interval.high_cm + 1e-10:
                location = "inside_active"
                destination = int(interval.channel)
                break
        if location != "inside_active" and (
            plane.low_edge_cm - 1e-10 <= shifted <= plane.high_edge_cm + 1e-10
        ):
            location = "inside_gap"
        location_counts[location] += 1
        after_residual = shifted - transform.track_transformed_cm
        before_residual = original - transform.track_original_cm
        residual_difference = abs(after_residual - before_residual)
        maximum_residual = max(maximum_residual, residual_difference)
        if residual_difference > residual_tolerance_cm:
            raise RuntimeError(
                f"MuFilter hit {hit['detector_id']} residual changed by "
                f"{residual_difference} cm"
            )
        converted = dict(hit)
        converted.update({
            "logical_plane": key.plane,
            "view": key.view,
            "orientation": key.orientation,
            "coordinate": coordinate,
            "side": transform.side,
            "low_edge_cm": transform.low_edge_cm,
            "high_edge_cm": transform.high_edge_cm,
            "target_cm": transform.target_cm,
            "track_original_cm": transform.track_original_cm,
            "track_transformed_cm": transform.track_transformed_cm,
            "delta_cm": transform.delta_cm,
            "original_coordinate_cm": original,
            "transformed_coordinate_cm": shifted,
            "location": location,
            "destination_bar": destination,
            "residual_difference_cm": residual_difference,
        })
        for suffix in ("a", "b", "mid"):
            converted[f"transformed_{coordinate}_{suffix}_cm"] = (
                float(hit[f"{coordinate}_{suffix}_cm"]) + transform.delta_cm
            )
        other = "y" if coordinate == "x" else "x"
        for suffix in ("a", "b", "mid"):
            converted[f"transformed_{other}_{suffix}_cm"] = float(
                hit[f"{other}_{suffix}_cm"]
            )
        output.append(converted)
    if sum(location_counts.values()) != len(hits):
        raise RuntimeError("MuFilter transformed hit counts do not close")
    audit = {
        "passed": True,
        "n_original_hits": len(hits),
        "n_transformed_hits": len(output),
        "locations": location_counts,
        "max_residual_difference_cm": maximum_residual,
        "planes": [
            {
                **asdict(transform),
                "key": asdict(transform.key),
                "z_spread_cm": planes[key].z_spread_cm,
            }
            for key, transform in sorted(transforms.items())
        ],
    }
    return output, audit


def open_tree(path: str, candidates: list[str]):
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise OSError(f"Cannot open ROOT file {path}")
    name, tree = open_event_tree(source, candidates)
    if not tree:
        source.Close()
        raise RuntimeError(f"No tree from {candidates} in {path}")
    return source, str(name), tree


def open_track_tree(path: str, tree_name: str = "sndData"):
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise OSError(f"Cannot open track input {path}")
    tree = source.Get(tree_name)
    if not tree:
        source.Close()
        raise RuntimeError(f"Track tree {tree_name!r} not found in {path}")
    if tree.BuildIndex("runId", "eventId") < 0:
        source.Close()
        raise RuntimeError(f"Could not build run/event index for {path}")
    return source, tree


def find_track_entry(tree, run_id: int, event_id: int) -> int:
    entry = int(tree.GetEntryNumberWithIndex(int(run_id), int(event_id)))
    if entry < 0:
        raise KeyError(f"Run/event {run_id}/{event_id} missing from track input")
    if tree.GetEntry(entry) <= 0:
        raise RuntimeError(f"Could not load track entry {entry}")
    observed = (int(tree.runId), int(tree.eventId))
    expected = (int(run_id), int(event_id))
    if observed != expected:
        raise RuntimeError(f"Track index identity mismatch: {observed} != {expected}")
    return entry


def _vector(tree, name: str) -> list:
    if not hasattr(tree, name):
        raise RuntimeError(f"Track input is missing branch {name}")
    return list(getattr(tree, name))


def select_track(
    tree, type_priority: list[int], min_ndf: int = 1,
    min_scifi_planes_per_projection: int = 2,
) -> tuple[dict, list[dict]]:
    track_type = [int(value) for value in _vector(tree, "reco_track_type")]
    flag = [int(value) for value in _vector(tree, "reco_track_flag")]
    ndf = [int(value) for value in _vector(tree, "reco_track_ndf")]
    chi2ndf = [float(value) for value in _vector(tree, "reco_track_chi2ndf")]
    momentum = {}
    for component in ("x", "y", "z"):
        name = f"reco_track_mom_{component}"
        if hasattr(tree, name):
            values = [float(value) for value in getattr(tree, name)]
            if len(values) != len(track_type):
                raise RuntimeError(f"Track momentum branch {name} is misaligned")
            momentum[component] = values
    offsets = [int(value) for value in _vector(tree, "reco_track_measurement_offset")]
    if len(offsets) != len(track_type) + 1:
        raise RuntimeError("Track measurement offsets do not close to track count")
    measurement = {
        name: _vector(tree, f"reco_track_measurement_{name}")
        for name in ("point_x", "point_y", "point_z", "subsystem", "station", "view")
    }
    lengths = {len(values) for values in measurement.values()}
    if len(lengths) != 1 or offsets[-1] != next(iter(lengths)):
        raise RuntimeError("Flattened track measurement vectors are misaligned")
    candidates = []
    rejected = []
    priority = {int(value): index for index, value in enumerate(type_priority)}
    for index, kind in enumerate(track_type):
        summary = {
            "track_index": index, "track_type": kind, "track_flag": flag[index],
            "track_ndf": ndf[index], "track_chi2ndf": chi2ndf[index],
        }
        for component, values in momentum.items():
            summary[f"track_mom_{component}"] = values[index]
        reasons = []
        if kind not in priority:
            reasons.append("track type is not enabled")
        if not flag[index]:
            reasons.append("track flag is false")
        if ndf[index] < min_ndf:
            reasons.append(f"NDF is below {min_ndf}")
        if not math.isfinite(chi2ndf[index]) or chi2ndf[index] < 0:
            reasons.append("chi2/NDF is not finite and non-negative")
        begin, end = offsets[index], offsets[index + 1]
        groups = {}
        for offset in range(begin, end):
            if int(measurement["subsystem"][offset]) != 1:
                continue
            station = int(measurement["station"][offset])
            view = int(measurement["view"][offset])
            if station not in range(1, 6) or view not in (0, 1):
                reasons.append("invalid SciFi station/view in measurements")
                continue
            values = (
                float(measurement["point_x"][offset]),
                float(measurement["point_y"][offset]),
                float(measurement["point_z"][offset]),
            )
            if not all(math.isfinite(value) for value in values):
                reasons.append("non-finite SciFi fitted point")
                continue
            groups.setdefault((station, view), []).append(values)
        collapsed = []
        for (station, view), values in sorted(groups.items()):
            collapsed.append({
                "station": station, "view": view,
                "x_cm": float(statistics.median(item[0] for item in values)),
                "y_cm": float(statistics.median(item[1] for item in values)),
                "z_cm": float(statistics.median(item[2] for item in values)),
                "n_measurements": len(values),
            })
        try:
            projection_x = fit_linear_projection(
                [(item["z_cm"], item["x_cm"]) for item in collapsed], "x",
                min_distinct_z=min_scifi_planes_per_projection,
            )
            projection_y = fit_linear_projection(
                [(item["z_cm"], item["y_cm"]) for item in collapsed], "y",
                min_distinct_z=min_scifi_planes_per_projection,
            )
        except ValueError as error:
            reasons.append(str(error))
            projection_x = projection_y = None
        summary["n_scifi_planes"] = len(collapsed)
        if reasons:
            rejected.append({**summary, "rejection_reasons": reasons})
            continue
        record = {
            **summary,
            "priority": priority[kind],
            "projection_x": projection_x,
            "projection_y": projection_y,
            "collapsed_scifi_points": collapsed,
        }
        candidates.append(record)
    if not candidates:
        detail = "; ".join(
            f"track {item['track_index']}: {', '.join(item['rejection_reasons'])}"
            for item in rejected
        ) or "event has no reconstructed tracks"
        raise RuntimeError(f"No usable SciFi track: {detail}")
    candidates.sort(key=lambda item: (
        item["priority"], item["track_chi2ndf"], item["track_index"],
    ))
    selected = candidates[0]
    eligible = []
    for item in candidates:
        eligible.append({
            key: value for key, value in item.items()
            if key not in ("projection_x", "projection_y")
        } | {
            "projection_x": asdict(item["projection_x"]),
            "projection_y": asdict(item["projection_y"]),
        })
    selected = dict(selected)
    return selected, eligible + rejected


def track_input_path(pattern: str, row: dict) -> str:
    return str(pattern).format(
        source_file_index=int(row["source_file_index"]),
        run=int(row.get("runId", 8329)),
    )


def serializable_track(track: dict) -> dict:
    output = dict(track)
    output["projection_x"] = asdict(output["projection_x"])
    output["projection_y"] = asdict(output["projection_y"])
    return output
