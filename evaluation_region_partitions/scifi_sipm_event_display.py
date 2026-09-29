#!/usr/bin/env python3
"""Render geometry-faithful, channel-level SciFi SiPM event-display pages.

Run this script inside an SNDSW environment.  The channel rectangles are read
from ``SiPMmapVol``; they are deliberately not put on an equally spaced grid.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import ROOT
import SndlhcGeo


ROOT.gROOT.SetBatch(True)

QDC_DISTRIBUTION_COLOR = "#1f77b4"


@dataclass(frozen=True)
class ChannelGeometry:
    mat: int
    sipm: int
    channel: int
    centre_cm: float
    half_width_cm: float


@dataclass(frozen=True)
class HitRecord:
    detector_id: int
    station: int
    orientation: str
    mat: int
    sipm: int
    channel: int
    qdc: float
    time_ns: float
    valid: bool
    measured_coordinate_cm: float


@dataclass
class SipmGeometryContext:
    """Loaded geometry objects reusable for multiple events with one geofile."""

    geofile: str
    interface: object
    scifi: object
    records: list[ChannelGeometry]
    validation: dict


def load_sipm_geometry(scifi) -> tuple[list[ChannelGeometry], dict]:
    """Build the 3-mat/12-SiPM/1536-channel map from TGeo nodes."""
    scifi.SiPMOverlap()
    sipm_map = ROOT.gGeoManager.FindVolumeFast("SiPMmapVol")
    channel_volume = ROOT.gGeoManager.FindVolumeFast("ChannelVol")
    if not sipm_map or not channel_volume:
        raise RuntimeError("SiPMOverlap() did not create SiPMmapVol/ChannelVol")
    shape = channel_volume.GetShape()
    half_width = float(shape.GetDY())
    records = []
    seen = set()
    for node in sipm_map.GetNodes():
        number = int(node.GetNumber())
        # Both sndScifiHit::GetMat() and the SiPMOverlap node number use the
        # native zero-based mat index (0, 1, 2).
        key = (number // 10000, (number % 10000) // 1000, number % 1000)
        if key in seen:
            raise RuntimeError(f"Duplicate SiPM geometry node {key}")
        seen.add(key)
        records.append(ChannelGeometry(*key, float(node.GetMatrix().GetTranslation()[1]), half_width))
    records.sort(key=lambda item: item.centre_cm)
    expected = 3 * 4 * 128
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} channel nodes, found {len(records)}")
    return records, validate_sipm_geometry(records)


def create_sipm_geometry_context(geofile: str) -> SipmGeometryContext:
    interface = SndlhcGeo.GeoInterface(geofile)
    scifi = interface.modules["Scifi"]
    records, validation = load_sipm_geometry(scifi)
    return SipmGeometryContext(geofile, interface, scifi, records, validation)


def validate_sipm_geometry(records: list[ChannelGeometry]) -> dict:
    """Measure all gap classes from adjacent geometry nodes."""
    spacing = defaultdict(list)
    for left, right in zip(records, records[1:]):
        if left.mat != right.mat:
            category = "mat_boundary"
        elif left.sipm != right.sipm:
            category = "sipm_boundary"
        elif left.channel == 63 and right.channel == 64:
            category = "central_die"
        else:
            category = "ordinary"
        spacing[category].append(right.centre_cm - left.centre_cm)
    width = 2.0 * records[0].half_width_cm
    result = {"channel_width_cm": width, "channel_count": len(records), "spacing": {}}
    for category, values in spacing.items():
        result["spacing"][category] = {
            "count": len(values),
            "min_cm": min(values),
            "median_cm": float(np.median(values)),
            "max_cm": max(values),
            "median_empty_gap_cm": float(np.median(values)) - width,
        }
    medians = [result["spacing"][key]["median_cm"] for key in ("ordinary", "central_die", "sipm_boundary")]
    if not medians[0] < medians[1] < medians[2]:
        raise RuntimeError(f"Geometry does not preserve the expected unequal gaps: {medians}")
    return result


def _event_header_identity(tree) -> dict:
    header = getattr(tree, "EventHeader", None)
    if header is None:
        return {}
    output = {}
    for key, methods in {
        "run_id": ("GetRunId", "GetRunID"),
        "event_id": ("GetEventNumber", "GetMCEntryNumber"),
    }.items():
        for method in methods:
            if hasattr(header, method):
                try:
                    output[key] = int(getattr(header, method)())
                    break
                except Exception:
                    pass
    return output


def extract_event_sipm_hits(tree, event_index: int, scifi) -> tuple[list[HitRecord], dict]:
    if event_index < 0 or event_index >= int(tree.GetEntries()):
        raise IndexError(f"event index {event_index} outside [0, {tree.GetEntries()})")
    if tree.GetEntry(event_index) <= 0:
        raise RuntimeError(f"Could not read event entry {event_index}")
    digi_hits = getattr(tree, "Digi_ScifiHits", None)
    if digi_hits is None:
        raise RuntimeError("Event tree has no Digi_ScifiHits branch")
    hits = []
    duplicate_keys = Counter()
    for hit in digi_hits:
        station = int(hit.GetStation())
        orientation = "vertical" if bool(hit.isVertical()) else "horizontal"
        mat, sipm, channel = int(hit.GetMat()), int(hit.GetSiPM()), int(hit.GetSiPMChan())
        detector_id = int(hit.GetDetectorID())
        point_a, point_b = ROOT.TVector3(), ROOT.TVector3()
        scifi.GetSiPMPosition(detector_id, point_a, point_b)
        measured = float(point_a.X() if orientation == "vertical" else point_a.Y())
        key = (station, orientation, mat, sipm, channel)
        duplicate_keys[key] += 1
        hits.append(HitRecord(
            detector_id, station, orientation, mat, sipm, channel,
            float(hit.GetSignal()), float(hit.GetTime()), bool(hit.isValid()), measured,
        ))
    duplicates = {str(key): value for key, value in duplicate_keys.items() if value > 1}
    if duplicates:
        raise RuntimeError(f"Multiple digis map to the same channel: {duplicates}")
    return hits, {**_event_header_identity(tree), "source_event_index": event_index}


def select_detail_station(hits: list[HitRecord], requested: str) -> int:
    if requested != "auto":
        return int(requested)
    counts = Counter(hit.station for hit in hits if hit.valid)
    if not counts:
        counts = Counter(hit.station for hit in hits)
    if not counts:
        raise RuntimeError("The event contains no SciFi digis")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _coordinate_transform(records: list[ChannelGeometry], hits: list[HitRecord]) -> dict:
    """Fit local SiPM-map coordinates to the measured global x/y coordinate."""
    geometry = {(g.mat, g.sipm, g.channel): g for g in records}
    pairs = [(geometry[(h.mat, h.sipm, h.channel)].centre_cm, h.measured_coordinate_cm)
             for h in hits if (h.mat, h.sipm, h.channel) in geometry]
    if not pairs:
        return {"sign": 1.0, "offset_cm": 0.0, "max_abs_residual_cm": None, "samples": 0}
    local = np.asarray([item[0] for item in pairs])
    measured = np.asarray([item[1] for item in pairs])
    candidates = []
    for sign in (-1.0, 1.0):
        offset = float(np.median(measured - sign * local))
        residual = measured - (sign * local + offset)
        candidates.append((float(np.max(np.abs(residual))), sign, offset))
    maximum, sign, offset = min(candidates)
    return {"sign": sign, "offset_cm": offset, "max_abs_residual_cm": maximum, "samples": len(pairs)}


def _display_records(records: list[ChannelGeometry], transform: dict) -> list[ChannelGeometry]:
    converted = [ChannelGeometry(g.mat, g.sipm, g.channel,
                                 transform["sign"] * g.centre_cm + transform["offset_cm"],
                                 g.half_width_cm) for g in records]
    return sorted(converted, key=lambda item: item.centre_cm)


def _normalizer(hits: list[HitRecord]) -> Normalize:
    values = [max(0.0, h.qdc) for h in hits if h.valid and math.isfinite(h.qdc)]
    return Normalize(vmin=0.0, vmax=max(values) if values and max(values) > 0 else 1.0)


def _separators(axis, records: list[ChannelGeometry], detail: bool = False):
    for left, right in zip(records, records[1:]):
        if detail and left.sipm == right.sipm and left.channel != 63:
            continue
        middle = (left.centre_cm + right.centre_cm) / 2.0
        if left.mat != right.mat:
            axis.axvline(middle, color="black", lw=1.8, alpha=.85)
        elif left.sipm != right.sipm:
            axis.axvline(middle, color="black", lw=.8, alpha=.55)
        elif left.channel == 63 and right.channel == 64:
            axis.axvline(middle, color="black", lw=.65, ls="--", alpha=.6)


def draw_orientation_overview(pdf: PdfPages, records: list[ChannelGeometry], hits: list[HitRecord],
                              orientation: str, active_stations: list[int], transforms: dict,
                              norm: Normalize, title: str, page_width: float,
                              png_output: Path | None = None, png_dpi: int = 300):
    """Draw every active plane with aligned channel-status and QDC axes."""
    if not active_stations:
        figure = plt.figure(figsize=(page_width, 5.0))
        figure.text(
            .5, .56, f"No recorded {orientation} SciFi hits",
            ha="center", va="center", fontsize=22,
        )
        figure.text(.5, .46, title, ha="center", va="center", fontsize=14)
        pdf.savefig(figure)
        if png_output is not None:
            png_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(png_output, dpi=png_dpi)
        plt.close(figure)
        return
    height_ratios = [ratio for _ in active_stations for ratio in (1.0, 2.0)]
    figure, axes = plt.subplots(
        2 * len(active_stations), 1,
        figsize=(page_width, max(7.0, 3.4 * len(active_stations) + 1.8)),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    cmap = plt.get_cmap("viridis")
    coordinate_label = (
        "SciFi channel position — global x [cm]"
        if orientation == "vertical"
        else "SciFi channel position — global y [cm]"
    )
    finite_event_qdc = [max(0.0, hit.qdc) for hit in hits if math.isfinite(hit.qdc)]
    qdc_max = max([float(norm.vmax), *finite_event_qdc, 1.0])
    for station_index, station in enumerate(active_stations):
        sipm_axis = axes[2 * station_index, 0]
        qdc_axis = axes[2 * station_index + 1, 0]
        shown = _display_records(records, transforms[station])
        hit_map = {(h.mat, h.sipm, h.channel): h for h in hits
                   if h.station == station and h.orientation == orientation}
        geometry_by_key = {
            (geometry.mat, geometry.sipm, geometry.channel): geometry
            for geometry in shown
        }
        patches, colors = [], []
        for geometry in shown:
            hit = hit_map.get((geometry.mat, geometry.sipm, geometry.channel))
            patches.append(Rectangle((geometry.centre_cm - geometry.half_width_cm, 0.0),
                                     2.0 * geometry.half_width_cm, 1.0))
            if hit is None:
                colors.append((1, 1, 1, 1))
            elif not hit.valid:
                colors.append((.8, .2, .2, 1))
            else:
                colors.append(cmap(norm(max(0.0, hit.qdc))))
        # At full-plane scale one channel is about one screen pixel.  Drawing
        # an edge on every rectangle would cover the QDC colour entirely;
        # physical gaps and the explicit die/SiPM/mat separators remain.
        collection = PatchCollection(patches, facecolors=colors, edgecolors="none",
                                     linewidths=0, antialiased=False)
        sipm_axis.add_collection(collection)
        xmin = shown[0].centre_cm - shown[0].half_width_cm
        xmax = shown[-1].centre_cm + shown[-1].half_width_cm
        for axis in (sipm_axis, qdc_axis):
            _separators(axis, shown)
            axis.set_xlim(xmin, xmax)
        sipm_axis.set_ylim(0, 1)
        sipm_axis.set_yticks([])
        sipm_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        count = len(hit_map)
        valid = sum(hit.valid for hit in hit_map.values())
        sipm_axis.set_title(
            f"Station {station} — {count} recorded hits, {valid} valid",
            loc="left",
        )
        for mat in (0, 1, 2):
            mat_records = [g for g in shown if g.mat == mat]
            sipm_axis.text(
                np.mean([g.centre_cm for g in mat_records]), .5, f"Mat {mat}",
                ha="center", va="center", fontsize=9, color="black", alpha=.42,
            )

        valid_hits = [
            hit for hit in hit_map.values()
            if hit.valid and math.isfinite(hit.qdc)
        ]
        invalid_hits = [
            hit for hit in hit_map.values()
            if not hit.valid and math.isfinite(hit.qdc)
        ]
        valid_x = [
            geometry_by_key[(hit.mat, hit.sipm, hit.channel)].centre_cm
            for hit in valid_hits
        ]
        valid_qdc = [max(0.0, hit.qdc) for hit in valid_hits]
        valid_widths = [
            2.0 * geometry_by_key[(hit.mat, hit.sipm, hit.channel)].half_width_cm
            for hit in valid_hits
        ]
        if valid_hits:
            qdc_axis.bar(
                valid_x, valid_qdc, width=valid_widths,
                color=QDC_DISTRIBUTION_COLOR,
                edgecolor="none", linewidth=0, align="center", zorder=2,
            )
        if invalid_hits:
            qdc_axis.scatter(
                [geometry_by_key[(hit.mat, hit.sipm, hit.channel)].centre_cm
                 for hit in invalid_hits],
                [max(0.0, hit.qdc) for hit in invalid_hits],
                color="#cc3333", marker="x", s=22, linewidths=.8, zorder=3,
            )
        qdc_axis.set_ylim(0, 1.05 * qdc_max)
        qdc_axis.set_ylabel("QDC")
        qdc_axis.set_xlabel(coordinate_label)
        qdc_axis.set_axisbelow(True)
        qdc_axis.grid(axis="y", color="#999999", alpha=.22, linewidth=.5)
    figure.suptitle(
        f"{title}\nAll active {orientation} SciFi planes — channel status and QDC, physical gaps retained",
        fontsize=14,
    )
    color_axis = figure.add_axes([.948, .17, .012, .66])
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, label="QDC")
    figure.legend(handles=[Patch(facecolor="white", edgecolor="#777", label="No recorded hit"),
                           Patch(facecolor="#cc3333", label="Recorded invalid hit")],
                  loc="lower center", ncol=2, frameon=False)
    figure.subplots_adjust(top=.89, bottom=.085, left=.06, hspace=.38, right=.925)
    pdf.savefig(figure)
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_output, dpi=png_dpi)
    plt.close(figure)


def _plane_state(records: list[ChannelGeometry], hits: list[HitRecord]) -> tuple[dict, dict, Normalize]:
    orientations = ["horizontal", "vertical"]
    active_stations = {
        orientation: sorted({hit.station for hit in hits if hit.orientation == orientation})
        for orientation in orientations
    }
    transforms = {
        orientation: {
            station: _coordinate_transform(
                records,
                [hit for hit in hits if hit.orientation == orientation and hit.station == station],
            )
            for station in active_stations[orientation]
        }
        for orientation in orientations
    }
    return active_stations, transforms, _normalizer(hits)


def _plane_audit(
    context: SipmGeometryContext, hits: list[HitRecord], event_identity: dict,
    event_source_path: str, tree_name: str, event_index: int,
    active_stations: dict, transforms: dict, output_pdf: str,
    plane_page_width: float,
) -> dict:
    counts = Counter((hit.station, hit.orientation) for hit in hits)
    valid_counts = Counter((hit.station, hit.orientation) for hit in hits if hit.valid)
    mapped_keys = {(item.mat, item.sipm, item.channel) for item in context.records}
    unmapped = [
        asdict(hit) for hit in hits
        if (hit.mat, hit.sipm, hit.channel) not in mapped_keys
    ]
    return {
        "event": event_identity,
        "inputs": {
            "event_file": event_source_path,
            "tree": tree_name,
            "geofile": context.geofile,
            "source_event_index": event_index,
        },
        "output_pdf": str(Path(output_pdf).resolve()),
        "page_types": ["scifi_horizontal", "scifi_vertical"],
        "plane_page_width_inches": plane_page_width,
        "active_stations_by_orientation": active_stations,
        "hit_counts": {
            f"station_{key[0]}_{key[1]}": value
            for key, value in sorted(counts.items())
        },
        "valid_hit_counts": {
            f"station_{key[0]}_{key[1]}": value
            for key, value in sorted(valid_counts.items())
        },
        "total_recorded_scifi_hits": len(hits),
        "total_valid_scifi_hits": sum(hit.valid for hit in hits),
        "unmapped_hits": unmapped,
        "geometry_validation": context.validation,
        "global_coordinate_transforms": transforms,
        "global_coordinate_validation": {
            orientation: {
                str(station): {
                    "tolerance_cm": 0.02,
                    "passed": (
                        transform["max_abs_residual_cm"] is not None
                        and transform["max_abs_residual_cm"] <= 0.02
                    ),
                }
                for station, transform in transforms[orientation].items()
            }
            for orientation in ("horizontal", "vertical")
        },
    }


def render_plane_pages(
    event_source_path: str, tree_name: str, event_index: int, geofile: str,
    output_pdf: str, title: str, plane_page_width: float = 66.0,
    geometry_context: SipmGeometryContext | None = None,
) -> dict:
    """Write exactly two vector pages: horizontal then vertical SciFi planes."""
    context = geometry_context or create_sipm_geometry_context(geofile)
    if context.geofile != geofile:
        raise ValueError(
            f"Geometry context is for {context.geofile}, requested {geofile}"
        )
    source = ROOT.TFile.Open(event_source_path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {event_source_path}")
    try:
        tree = source.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Tree {tree_name!r} not found in {event_source_path}")
        hits, event_identity = extract_event_sipm_hits(tree, event_index, context.scifi)
        active_stations, transforms, norm = _plane_state(context.records, hits)
        output = Path(output_pdf).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output) as pdf:
            for orientation in ("horizontal", "vertical"):
                draw_orientation_overview(
                    pdf, context.records, hits, orientation,
                    active_stations[orientation], transforms[orientation], norm,
                    title, plane_page_width,
                )
        return _plane_audit(
            context, hits, event_identity, event_source_path, tree_name,
            event_index, active_stations, transforms, str(output), plane_page_width,
        )
    finally:
        source.Close()


def draw_sipm_detail(pdf: PdfPages, records: list[ChannelGeometry], hits: list[HitRecord], station: int,
                     orientation: str, mat: int, sipm: int, transform: dict, norm: Normalize, title: str):
    shown = [g for g in _display_records(records, transform) if g.mat == mat and g.sipm == sipm]
    hit_map = {(h.mat, h.sipm, h.channel): h for h in hits
               if h.station == station and h.orientation == orientation and h.mat == mat and h.sipm == sipm}
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [1.2, 2.2, 2.2]})
    for geometry in shown:
        hit = hit_map.get((mat, sipm, geometry.channel))
        if hit is None:
            color, hatch = "white", None
        elif hit.valid:
            color, hatch = cmap(norm(max(0.0, hit.qdc))), None
        else:
            color, hatch = "#cc3333", "////"
        axes[0].add_patch(Rectangle((geometry.centre_cm - geometry.half_width_cm, 0),
                                    2 * geometry.half_width_cm, 1, facecolor=color,
                                    edgecolor="#666", linewidth=.25, hatch=hatch))
    _separators(axes[0], shown, detail=True)
    axes[0].set_xlim(shown[0].centre_cm - shown[0].half_width_cm,
                     shown[-1].centre_cm + shown[-1].half_width_cm)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xlabel("global x [cm]" if orientation == "vertical" else "global y [cm]")
    axes[0].set_title("128 physical channel rectangles (dashed line: channel 63/64 die gap)", loc="left")
    valid = [h for h in hit_map.values() if h.valid]
    invalid = [h for h in hit_map.values() if not h.valid]
    axes[1].scatter([h.channel for h in valid], [h.qdc for h in valid], c=[cmap(norm(max(0, h.qdc))) for h in valid],
                    s=18, edgecolor="none")
    axes[1].scatter([h.channel for h in invalid], [h.qdc for h in invalid], c="#cc3333", marker="x", s=28)
    axes[1].set_ylabel("QDC")
    finite_time = [h for h in hit_map.values() if math.isfinite(h.time_ns)]
    axes[2].scatter([h.channel for h in finite_time], [h.time_ns for h in finite_time],
                    c=[cmap(norm(max(0, h.qdc))) if h.valid else "#cc3333" for h in finite_time], s=18)
    axes[2].set_ylabel("hit time [ns]")
    axes[2].set_xlabel("SiPM channel number")
    for axis in axes[1:]:
        axis.set_xlim(-1, 128)
        axis.grid(alpha=.2)
        axis.axvline(63.5, color="black", lw=.65, ls="--", alpha=.6)
    figure.suptitle(f"{title}\nStation {station}, {orientation}, mat {mat}, SiPM {sipm} — "
                    f"{len(hit_map)} recorded / {len(valid)} valid hits", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .93))
    pdf.savefig(figure)
    plt.close(figure)


def _merge_pdfs(overview_pdf: str | None, detail_pdf: str, output_pdf: str, workdir: str):
    pages = []
    if overview_pdf:
        first_page = str(Path(workdir) / "overview_page_1.pdf")
        subprocess.run(["pdfseparate", "-f", "1", "-l", "1", overview_pdf, first_page], check=True)
        pages.append(first_page)
    pages.append(detail_pdf)
    if len(pages) == 1:
        shutil.copyfile(detail_pdf, output_pdf)
    else:
        subprocess.run(["pdfunite", *pages, output_pdf], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tree", default="cbmsim")
    parser.add_argument("--event-index", required=True, type=int)
    parser.add_argument("--geo", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--identity-output", required=True)
    parser.add_argument("--overview-pdf", help="Existing overview PDF; its first page is prepended")
    parser.add_argument("--station", default="auto")
    parser.add_argument("--orientation", choices=("both", "horizontal", "vertical"), default="both")
    parser.add_argument("--detail-mode", choices=("fired-sipms", "all-sipms"), default="fired-sipms")
    parser.add_argument(
        "--max-detail-pages", type=int, default=0,
        help="Number of optional individual-SiPM pages after the plane overviews (default: 0)",
    )
    parser.add_argument(
        "--plane-page-width", type=float, default=66.0,
        help="Width in inches of horizontal/vertical all-plane pages (default: 66, three times the original width)",
    )
    parser.add_argument(
        "--plane-png-dir", type=Path,
        help="Optionally also save horizontal/vertical plane pages as standalone PNG files",
    )
    parser.add_argument("--plane-png-dpi", type=int, default=300)
    parser.add_argument("--particle", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--partition-id", default="")
    parser.add_argument("--feature-entry", type=int)
    parser.add_argument("--original-entry", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    output_pdf = Path(args.output_pdf).resolve()
    identity_output = Path(args.identity_output).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    identity_output.parent.mkdir(parents=True, exist_ok=True)

    geo = SndlhcGeo.GeoInterface(args.geo)
    scifi = geo.modules["Scifi"]
    records, geometry_validation = load_sipm_geometry(scifi)
    source = ROOT.TFile.Open(args.input)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {args.input}")
    tree = source.Get(args.tree)
    if not tree:
        raise RuntimeError(f"Tree {args.tree!r} not found in {args.input}")
    hits, event_identity = extract_event_sipm_hits(tree, args.event_index, scifi)
    station = select_detail_station(hits, args.station)
    orientations = ["horizontal", "vertical"] if args.orientation == "both" else [args.orientation]
    station_hits = [h for h in hits if h.station == station and h.orientation in orientations]
    active_stations = {
        orientation: sorted({h.station for h in hits if h.orientation == orientation})
        for orientation in orientations
    }
    plane_transforms = {
        orientation: {
            active_station: _coordinate_transform(
                records,
                [h for h in hits if h.orientation == orientation and h.station == active_station],
            )
            for active_station in active_stations[orientation]
        }
        for orientation in orientations
    }
    detail_transforms = {
        orientation: plane_transforms[orientation].get(
            station, _coordinate_transform(records, [])
        )
        for orientation in orientations
    }
    # One event-wide scale makes QDC colours comparable between every plane.
    norm = _normalizer([h for h in hits if h.orientation in orientations])
    title_parts = [part for part in (args.particle, args.region, f"entry {args.event_index}") if part]
    title = " — ".join(title_parts)

    groups = Counter((h.orientation, h.mat, h.sipm) for h in station_hits)
    if args.detail_mode == "all-sipms":
        detail_groups = [(orientation, mat, sipm) for orientation in orientations
                         for mat in (0, 1, 2) for sipm in range(4)]
    else:
        detail_groups = sorted(groups, key=lambda key: (-groups[key], key))
    selected_groups = detail_groups[:max(0, args.max_detail_pages)]

    with tempfile.TemporaryDirectory(prefix="scifi_sipm_display_") as workdir:
        detail_pdf = str(Path(workdir) / "details.pdf")
        with PdfPages(detail_pdf) as pdf:
            for orientation in orientations:
                png_output = None
                if args.plane_png_dir is not None:
                    png_output = args.plane_png_dir / f"{orientation}_planes.png"
                draw_orientation_overview(
                    pdf, records, hits, orientation, active_stations[orientation],
                    plane_transforms[orientation], norm, title, args.plane_page_width,
                    png_output, args.plane_png_dpi,
                )
            for orientation, mat, sipm in selected_groups:
                draw_sipm_detail(pdf, records, hits, station, orientation, mat, sipm,
                                 detail_transforms[orientation], norm, title)
        _merge_pdfs(args.overview_pdf, detail_pdf, str(output_pdf), workdir)

    counts = Counter((h.station, h.orientation) for h in hits)
    valid_counts = Counter((h.station, h.orientation) for h in hits if h.valid)
    mapped_keys = {(g.mat, g.sipm, g.channel) for g in records}
    unmapped = [asdict(h) for h in hits if (h.mat, h.sipm, h.channel) not in mapped_keys]
    audit = {
        "event": {**event_identity, "particle": args.particle, "region": args.region,
                  "partition_id": args.partition_id, "feature_entry": args.feature_entry,
                  "original_entry": args.original_entry},
        "inputs": {"event_file": args.input, "tree": args.tree, "geofile": args.geo,
                   "overview_pdf": args.overview_pdf},
        "output_pdf": str(output_pdf),
        "selected_station": station,
        "selected_orientations": orientations,
        "plane_page_width_inches": args.plane_page_width,
        "plane_png": {
            "directory": str(args.plane_png_dir.resolve()) if args.plane_png_dir else None,
            "dpi": args.plane_png_dpi if args.plane_png_dir else None,
        },
        "active_stations_by_orientation": active_stations,
        "selected_detail_pages": [{"orientation": key[0], "mat": key[1], "sipm": key[2],
                                     "recorded_hits": groups[key]} for key in selected_groups],
        "omitted_detail_page_count": max(0, len(detail_groups) - len(selected_groups)),
        "hit_counts": {f"station_{key[0]}_{key[1]}": value for key, value in sorted(counts.items())},
        "valid_hit_counts": {f"station_{key[0]}_{key[1]}": value for key, value in sorted(valid_counts.items())},
        "total_recorded_scifi_hits": len(hits),
        "total_valid_scifi_hits": sum(h.valid for h in hits),
        "unmapped_hits": unmapped,
        "geometry_validation": geometry_validation,
        "global_coordinate_transforms": plane_transforms,
        "global_coordinate_validation": {
            orientation: {
                str(active_station): {
                    "tolerance_cm": 0.02,
                    "passed": (
                        transform["max_abs_residual_cm"] is not None
                        and transform["max_abs_residual_cm"] <= 0.02
                    ),
                }
                for active_station, transform in plane_transforms[orientation].items()
            }
            for orientation in orientations
        },
        "status_semantics": {"white": "no recorded digi (not a health claim)",
                             "viridis": "valid recorded digi, coloured by QDC",
                             "red_hatch": "recorded digi with isValid() false"},
    }
    identity_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source.Close()
    print(f"Wrote {output_pdf}")
    print(f"Wrote {identity_output}")


if __name__ == "__main__":
    main()
