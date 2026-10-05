#!/usr/bin/env python3
"""Cluster SciFi plane hits and colour their QDC distributions by cluster.

The clustering coordinate is the electronics-order channel number

    layer_channel = channel + 128 * sipm + 512 * mat

so the physical channel-63/64 die gap, SiPM boundaries, and mat boundaries do
not split an otherwise consecutive cluster.  A genuinely missing channel does.
Only valid hits with finite QDC enter clusters; invalid hits remain visible as
red crosses in the display and are counted in the JSON audit.

Run the rendering entry point inside the repository's SNDSW environment.  The
pure clustering functions intentionally have no ROOT dependency and can be
unit-tested in a regular Python process.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

CONVERT_DATA = Path(__file__).resolve().parents[1] / "convertData"
if str(CONVERT_DATA) not in sys.path:
    sys.path.insert(0, str(CONVERT_DATA))
from scifi_plane_clustering import cluster_scifi_hits

ORIENTATIONS = ("horizontal", "vertical")
STATIONS = (1, 2, 3, 4, 5)
CHANNELS_PER_SIPM = 128
SIPMS_PER_MAT = 4
CHANNELS_PER_MAT = CHANNELS_PER_SIPM * SIPMS_PER_MAT
LARGEST_QDC_CLUSTER_COLOR = "#ffbf00"
CHANNELS_PER_PLANE = 3 * CHANNELS_PER_MAT


def layer_channel(hit: Any) -> int:
    """Return a zero-based, electronics-order channel index for one plane."""
    mat = int(hit.mat)
    sipm = int(hit.sipm)
    channel = int(hit.channel)
    if not 0 <= mat < 3:
        raise ValueError(f"mat outside [0, 3): {mat}")
    if not 0 <= sipm < SIPMS_PER_MAT:
        raise ValueError(f"SiPM outside [0, {SIPMS_PER_MAT}): {sipm}")
    if not 0 <= channel < CHANNELS_PER_SIPM:
        raise ValueError(f"channel outside [0, {CHANNELS_PER_SIPM}): {channel}")
    return channel + CHANNELS_PER_SIPM * sipm + CHANNELS_PER_MAT * mat


@dataclass(frozen=True)
class PlaneCluster:
    """One maximal run of consecutive valid, finite-QDC plane hits."""

    cluster_id: int
    hits: tuple[Any, ...]
    channels: tuple[int, ...]
    qdc_sum: float
    qdc_mean: float
    qdc_max: float

    @property
    def first_channel(self) -> int:
        return self.channels[0]

    @property
    def last_channel(self) -> int:
        return self.channels[-1]

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def peak_fraction(self) -> float:
        return self.qdc_max / self.qdc_sum if self.qdc_sum > 0.0 else 0.0


@dataclass(frozen=True)
class PlaneClusterResult:
    station: int
    orientation: str
    recorded_hits: tuple[Any, ...]
    clusters: tuple[PlaneCluster, ...]
    invalid_hit_count: int
    nonfinite_valid_hit_count: int
    largest_qdc_cluster_id: int | None
    largest_hit_cluster_id: int | None

    @property
    def valid_finite_hit_count(self) -> int:
        return sum(cluster.hit_count for cluster in self.clusters)

    @property
    def valid_finite_qdc_sum(self) -> float:
        return sum(cluster.qdc_sum for cluster in self.clusters)

    def audit(self) -> dict:
        clusters = []
        for cluster in self.clusters:
            clusters.append({
                "cluster_id": cluster.cluster_id,
                "first_layer_channel": cluster.first_channel,
                "last_layer_channel": cluster.last_channel,
                "hit_count": cluster.hit_count,
                "qdc_sum": cluster.qdc_sum,
                "qdc_mean": cluster.qdc_mean,
                "qdc_max": cluster.qdc_max,
                "peak_fraction": cluster.peak_fraction,
                "is_largest_qdc": cluster.cluster_id == self.largest_qdc_cluster_id,
                "is_largest_hits": cluster.cluster_id == self.largest_hit_cluster_id,
            })
        clustered_hits = sum(item["hit_count"] for item in clusters)
        clustered_qdc = sum(item["qdc_sum"] for item in clusters)
        return {
            "station": self.station,
            "orientation": self.orientation,
            "recorded_hit_count": len(self.recorded_hits),
            "valid_finite_hit_count": self.valid_finite_hit_count,
            "invalid_hit_count": self.invalid_hit_count,
            "nonfinite_valid_hit_count": self.nonfinite_valid_hit_count,
            "cluster_count": len(clusters),
            "largest_qdc_cluster_id": self.largest_qdc_cluster_id,
            "largest_hit_cluster_id": self.largest_hit_cluster_id,
            "clusters": clusters,
            "closure": {
                "clustered_hit_count": clustered_hits,
                "hit_count_difference": self.valid_finite_hit_count - clustered_hits,
                "clustered_qdc_sum": clustered_qdc,
                "qdc_sum_difference": self.valid_finite_qdc_sum - clustered_qdc,
            },
        }


def cluster_plane_hits(
    hits: Iterable[Any], station: int, orientation: str,
) -> PlaneClusterResult:
    """Cluster one plane with the production clustering implementation."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unknown orientation {orientation!r}")
    plane_hits = tuple(
        hit for hit in hits
        if int(hit.station) == station and str(hit.orientation) == orientation
    )

    indexed: dict[int, Any] = {}
    for hit in plane_hits:
        index = layer_channel(hit)
        if index in indexed:
            raise ValueError(
                f"duplicate layer channel {index} in station {station} {orientation}"
            )
        indexed[index] = hit

    vertical = orientation == "vertical"
    usable = [
        {"station": station, "isVertical": vertical,
         "layer_channel": index, "qdc": float(hit.qdc)}
        for index, hit in indexed.items()
        if bool(hit.valid) and math.isfinite(float(hit.qdc))
    ]
    shared = cluster_scifi_hits(usable)["planes"][(station, vertical)]
    clusters = [
        PlaneCluster(
            cluster_id=cluster["spatial_id"],
            hits=tuple(indexed[channel] for channel in cluster["channels"]),
            channels=cluster["channels"],
            qdc_sum=cluster["qdc"],
            qdc_mean=cluster["qdc_per_hit"],
            qdc_max=max(cluster["qdcs"]),
        )
        for cluster in shared["clusters"]
    ]

    largest_qdc = max(
        clusters,
        key=lambda cluster: (cluster.qdc_sum, cluster.hit_count, -cluster.first_channel),
        default=None,
    )
    largest_hits = max(
        clusters,
        key=lambda cluster: (cluster.hit_count, cluster.qdc_sum, -cluster.first_channel),
        default=None,
    )
    return PlaneClusterResult(
        station=station,
        orientation=orientation,
        recorded_hits=plane_hits,
        clusters=tuple(clusters),
        invalid_hit_count=sum(not bool(hit.valid) for hit in plane_hits),
        nonfinite_valid_hit_count=sum(
            bool(hit.valid) and not math.isfinite(float(hit.qdc))
            for hit in plane_hits
        ),
        largest_qdc_cluster_id=(largest_qdc.cluster_id if largest_qdc else None),
        largest_hit_cluster_id=(largest_hits.cluster_id if largest_hits else None),
    )


def cluster_event_hits(
    hits: Sequence[Any], stations: Sequence[int] = STATIONS,
) -> dict[tuple[int, str], PlaneClusterResult]:
    """Return stable results for every requested station and orientation."""
    return {
        (station, orientation): cluster_plane_hits(hits, station, orientation)
        for orientation in ORIENTATIONS
        for station in stations
    }


def _draw_hardware_boundaries(axis: Any) -> None:
    for mat in range(3):
        for sipm in range(SIPMS_PER_MAT):
            axis.axvline(
                mat * CHANNELS_PER_MAT + sipm * CHANNELS_PER_SIPM + 63.5,
                color="#777777", linestyle=":", linewidth=0.45, alpha=0.45,
                zorder=0,
            )
        for sipm in range(1, SIPMS_PER_MAT):
            axis.axvline(
                mat * CHANNELS_PER_MAT + sipm * CHANNELS_PER_SIPM - 0.5,
                color="#555555", linestyle="--", linewidth=0.65, alpha=0.55,
                zorder=0,
            )
    for mat in range(1, 3):
        axis.axvline(
            mat * CHANNELS_PER_MAT - 0.5,
            color="#222222", linestyle="-", linewidth=1.0, alpha=0.7,
            zorder=0,
        )


def draw_cluster_pages(
    output_pdf: Path,
    results: dict[tuple[int, str], PlaneClusterResult],
    title: str,
) -> None:
    """Write horizontal and vertical five-plane QDC pages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for orientation in ORIENTATIONS:
            figure, axes = plt.subplots(
                len(STATIONS), 1, sharex=True, figsize=(24, 14), squeeze=False,
            )
            for row, station in enumerate(STATIONS):
                axis = axes[row, 0]
                result = results[(station, orientation)]
                colours = plt.get_cmap("tab20")
                for cluster in result.clusters:
                    is_largest = cluster.cluster_id == result.largest_qdc_cluster_id
                    colour = (
                        LARGEST_QDC_CLUSTER_COLOR
                        if is_largest
                        else colours(cluster.cluster_id % colours.N)
                    )
                    axis.bar(
                        cluster.channels,
                        [max(0.0, float(hit.qdc)) for hit in cluster.hits],
                        width=0.92,
                        color=colour,
                        edgecolor="black" if is_largest else "none",
                        linewidth=0.55 if is_largest else 0.0,
                        zorder=2,
                    )
                invalid = [
                    hit for hit in result.recorded_hits
                    if not bool(hit.valid) and math.isfinite(float(hit.qdc))
                ]
                if invalid:
                    axis.scatter(
                        [layer_channel(hit) for hit in invalid],
                        [max(0.0, float(hit.qdc)) for hit in invalid],
                        marker="x", color="#cc3333", s=24, linewidths=0.8,
                        zorder=3,
                    )
                _draw_hardware_boundaries(axis)
                axis.set_xlim(-1, CHANNELS_PER_PLANE)
                axis.set_ylabel(f"S{station}\nQDC")
                axis.grid(axis="y", alpha=0.2, linewidth=0.5)
                if result.largest_qdc_cluster_id is None:
                    summary = "no valid finite-QDC hits"
                else:
                    largest = result.clusters[result.largest_qdc_cluster_id]
                    summary = (
                        f"clusters={len(result.clusters)}; largest-QDC C{largest.cluster_id}: "
                        f"hits={largest.hit_count}, QDC={largest.qdc_sum:.1f}, "
                        f"channels={largest.first_channel}-{largest.last_channel}"
                    )
                axis.set_title(summary, loc="left", fontsize=9)
            axes[-1, 0].set_xlabel(
                "layer channel = channel + 128 × SiPM + 512 × mat"
            )
            figure.suptitle(
                f"{title}\n{orientation.capitalize()} SciFi planes — QDC coloured by contiguous cluster",
                fontsize=14,
            )
            figure.legend(
                handles=[
                    Patch(
                        facecolor=LARGEST_QDC_CLUSTER_COLOR,
                        edgecolor="black",
                        label="largest-QDC cluster (fixed amber)",
                    ),
                    Line2D([], [], color="#cc3333", marker="x", linestyle="None", label="invalid hit"),
                    Line2D([], [], color="#777777", linestyle=":", label="channel 63/64 die boundary"),
                    Line2D([], [], color="#555555", linestyle="--", label="SiPM boundary"),
                    Line2D([], [], color="#222222", linestyle="-", label="mat boundary"),
                ],
                loc="lower center", ncol=5, frameon=False,
            )
            figure.subplots_adjust(top=0.91, bottom=0.09, left=0.055, right=0.99, hspace=0.48)
            pdf.savefig(figure)
            plt.close(figure)


def render_cluster_event(
    event_source_path: str,
    tree_name: str,
    event_index: int,
    geofile: str,
    output_pdf: str,
    title: str,
) -> dict:
    """Read one event, write its cluster display, and return an audit object."""
    import ROOT

    from scifi_sipm_event_display import (
        create_sipm_geometry_context,
        extract_event_sipm_hits,
    )

    ROOT.gROOT.SetBatch(True)
    context = create_sipm_geometry_context(geofile)
    source = ROOT.TFile.Open(event_source_path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open {event_source_path}")
    try:
        tree = source.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Tree {tree_name!r} not found in {event_source_path}")
        hits, event_identity = extract_event_sipm_hits(tree, event_index, context.scifi)
        results = cluster_event_hits(hits)
        resolved_pdf = Path(output_pdf).resolve()
        draw_cluster_pages(resolved_pdf, results, title)
        plane_audits = [
            results[(station, orientation)].audit()
            for orientation in ORIENTATIONS
            for station in STATIONS
        ]
        return {
            "event": event_identity,
            "inputs": {
                "event_file": event_source_path,
                "tree": tree_name,
                "geofile": geofile,
                "source_event_index": event_index,
            },
            "output_pdf": str(resolved_pdf),
            "page_types": ["scifi_cluster_horizontal", "scifi_cluster_vertical"],
            "clustering_definition": {
                "coordinate": "channel + 128 * sipm + 512 * mat",
                "adjacency": "successive layer-channel indices differ by exactly one",
                "included_hits": "valid hits with finite QDC",
                "qdc_for_summaries": "max(0, qdc)",
                "largest_qdc_display_color": LARGEST_QDC_CLUSTER_COLOR,
                "largest_qdc_tie_break": "higher hit count, then lower first layer channel",
                "largest_hits_tie_break": "higher QDC sum, then lower first layer channel",
            },
            "geometry_validation": context.validation,
            "planes": plane_audits,
            "event_closure": {
                "recorded_hit_count": len(hits),
                "plane_recorded_hit_count": sum(item["recorded_hit_count"] for item in plane_audits),
                "valid_finite_hit_count": sum(item["valid_finite_hit_count"] for item in plane_audits),
                "clustered_hit_count": sum(item["closure"]["clustered_hit_count"] for item in plane_audits),
                "clustered_qdc_sum": sum(item["closure"]["clustered_qdc_sum"] for item in plane_audits),
            },
        }
    finally:
        source.Close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tree", default="cbmsim")
    parser.add_argument("--event-index", required=True, type=int)
    parser.add_argument("--geo", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--title", default="SciFi cluster diagnostic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = render_cluster_event(
        args.input,
        args.tree,
        args.event_index,
        args.geo,
        args.output_pdf,
        args.title,
    )
    audit_path = Path(args.output_audit).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {audit['output_pdf']}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
