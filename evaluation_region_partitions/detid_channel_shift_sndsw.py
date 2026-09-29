#!/usr/bin/env python3
"""SNDSW geometry adapter for detector-ID channel-sequence shifting."""

from __future__ import annotations

from collections import Counter

import ROOT
import SndlhcGeo

from detid_channel_shift import ChannelAddress, ChannelLayout, PlaneKey
from scifi_sipm_event_display import load_sipm_geometry


ROOT.gROOT.SetBatch(True)


def encode_scifi_detid(station: int, view: int, mat: int, sipm: int, channel: int) -> int:
    return station * 1_000_000 + view * 100_000 + mat * 10_000 + sipm * 1_000 + channel


def decode_scifi_detid(det_id: int) -> tuple[int, int, int, int, int]:
    value = int(det_id)
    return (
        value // 1_000_000,
        (value // 100_000) % 10,
        (value // 10_000) % 10,
        (value // 1_000) % 10,
        value % 1_000,
    )


def event_identity(tree) -> tuple[int, int]:
    header = tree.EventHeader
    return int(header.GetRunId()), int(header.GetEventNumber())


def public_eos_path(path: str) -> str:
    value = str(path)
    if value.startswith("/eos/experiment/"):
        return "root://eospublic.cern.ch/" + value
    return value


def layout_matches_axis(key: PlaneKey, selected_axis: str) -> bool:
    """Return whether this physical readout view measures the selected axis."""
    if selected_axis not in {"x", "y"}:
        raise ValueError(f"Unsupported selected axis {selected_axis!r}")
    return key.coordinate == selected_axis


class SciFiChannelGeometry:
    def __init__(self, geofile: str):
        self.geofile = str(geofile)
        self.interface = SndlhcGeo.GeoInterface(self.geofile)
        self.scifi = self.interface.modules["Scifi"]
        self.mufilter = self.interface.modules["MuFilter"]
        template, self.template_validation = load_sipm_geometry(self.scifi)
        self.half_width_cm = float(template[0].half_width_cm)

    def initialize_event(self, event_header) -> None:
        self.scifi.InitEvent(event_header)
        self.mufilter.InitEvent(event_header)

    def build_layouts(self) -> dict[PlaneKey, ChannelLayout]:
        layouts = {}
        for station in range(1, 6):
            for view in (0, 1):
                key = PlaneKey("Scifi", 0, station, view)
                channels = []
                for mat in range(3):
                    for sipm in range(4):
                        for channel in range(128):
                            det_id = encode_scifi_detid(station, view, mat, sipm, channel)
                            if decode_scifi_detid(det_id) != (station, view, mat, sipm, channel):
                                raise RuntimeError(f"SciFi detector-ID round trip failed: {det_id}")
                            a, b = ROOT.TVector3(), ROOT.TVector3()
                            self.scifi.GetSiPMPosition(det_id, a, b)
                            centre = 0.5 * (
                                float(a.X()) + float(b.X()) if view == 1
                                else float(a.Y()) + float(b.Y())
                            )
                            z_cm = 0.5 * (float(a.Z()) + float(b.Z()))
                            channels.append(ChannelAddress(
                                det_id=det_id,
                                key=key,
                                ordinal=-1,
                                centre_cm=centre,
                                low_cm=centre - self.half_width_cm,
                                high_cm=centre + self.half_width_cm,
                                hierarchy=(mat, sipm, channel),
                                z_cm=z_cm,
                            ))
                layout = ChannelLayout.ordered(key, channels)
                if len(layout.channels) != 1536:
                    raise RuntimeError(f"Expected 1536 SciFi channels in {key}")
                layouts[key] = layout
        return layouts

    def build_mufilter_layouts(self) -> dict[PlaneKey, ChannelLayout]:
        """Build coordinate-ordered layouts for every Veto, US, and DS view."""
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
        layouts = {}
        for system in (1, 2, 3):
            for plane in range(plane_counts[system]):
                if system == 3:
                    views = (0, 1) if plane < 3 else (1,)
                else:
                    views = (1,) if system == 1 and plane == 2 else (0,)
                for view in views:
                    key = PlaneKey("MuFilter", system, plane, view)
                    channels = []
                    for bar in range(bar_counts[system]):
                        encoded_bar = bar + 60 if system == 3 and view == 1 else bar
                        det_id = system * 10_000 + plane * 1_000 + encoded_bar
                        a, b = ROOT.TVector3(), ROOT.TVector3()
                        self.mufilter.GetPosition(det_id, a, b)
                        centre = 0.5 * (
                            float(a.X()) + float(b.X()) if view == 1
                            else float(a.Y()) + float(b.Y())
                        )
                        z_cm = 0.5 * (float(a.Z()) + float(b.Z()))
                        width = half_width[(system, view)]
                        channels.append(ChannelAddress(
                            det_id=det_id,
                            key=key,
                            ordinal=-1,
                            centre_cm=centre,
                            low_cm=centre - width,
                            high_cm=centre + width,
                            hierarchy=(bar,),
                            z_cm=z_cm,
                        ))
                    layouts[key] = ChannelLayout.ordered(key, channels)
        return layouts


def gap_audit(layout: ChannelLayout) -> dict:
    categories = Counter()
    spacings = {}
    for left, right in zip(layout.channels, layout.channels[1:]):
        lmat, lsipm, lchannel = left.hierarchy
        rmat, rsipm, rchannel = right.hierarchy
        if lmat != rmat:
            category = "mat_boundary"
        elif lsipm != rsipm:
            category = "sipm_boundary"
        elif {lchannel, rchannel} == {63, 64}:
            category = "central_die"
        else:
            category = "ordinary"
        categories[category] += 1
        spacings.setdefault(category, []).append(right.centre_cm - left.centre_cm)
    return {
        "channel_count": len(layout.channels),
        "counts": dict(categories),
        "spacing_cm": {
            key: {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            for key, values in spacings.items()
        },
    }


def digi_payload(hit) -> dict:
    n_sipms = int(hit.GetnSiPMs())
    n_sides = int(hit.GetnSides())
    n_channels = max(1, n_sipms * max(1, n_sides))
    return {
        "det_id": int(hit.GetDetectorID()),
        "valid": bool(hit.isValid()),
        "n_sipms": n_sipms,
        "n_sides": n_sides,
        "signals": [float(hit.GetSignal(index)) for index in range(n_channels)],
        "times": [float(hit.GetTime(index)) for index in range(n_channels)],
        "daq": [
            {
                "raw_index": int(hit.GetRawHitIndex(index)),
                "board": int(hit.GetBoardID(index)),
                "tofpet": int(hit.GetTofpetID(index)),
                "channel": int(hit.Getchannel(index)),
            }
            for index in range(n_channels)
        ],
    }


def mufilter_hit_key(hit) -> PlaneKey:
    return PlaneKey(
        "MuFilter", int(hit.GetSystem()), int(hit.GetPlane()),
        int(bool(hit.isVertical())),
    )
