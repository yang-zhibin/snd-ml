#!/usr/bin/env python3
"""Draw SciFi fiducial rectangle definitions in x-y position space."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def channel_to_position(channel, n_channels, pos_min, pos_max):
    return pos_min + channel * (pos_max - pos_min) / n_channels


def add_rect(ax, name, bounds, edgecolor, facecolor="none", alpha=1.0, linestyle="-"):
    x_min, x_max, y_min, y_max = bounds
    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=facecolor != "none",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.4,
        linestyle=linestyle,
        alpha=alpha,
        label=name,
    )
    ax.add_patch(rect)


def area(bounds):
    x_min, x_max, y_min, y_max = bounds
    return (x_max - x_min) * (y_max - y_min)


def main():
    n_scifi_channels = 1536

    # Full SciFi active position spans used by data_quality_check/cal_fiducial_pos.py.
    active_x = (-46.09, -6.99)
    active_y = (14.21, 53.86)
    active = (*active_x, *active_y)

    # Original sndsw AvgSFChan cut:
    # vertical channels -> x, horizontal channels -> y.
    original_x = (
        channel_to_position(200, n_scifi_channels, *active_x),
        channel_to_position(1200, n_scifi_channels, *active_x),
    )
    original_y = (
        channel_to_position(300, n_scifi_channels, *active_y),
        channel_to_position(1336, n_scifi_channels, *active_y),
    )
    original = (*original_x, *original_y)

    # Separate-plane position cuts from region_v1/evaluation_options_v1.
    separated_s12 = (-44.0, -10.0, 18.0, 52.0)
    separated_s345 = (-42.0, -12.0, 20.0, 50.0)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    add_rect(ax, "SciFi active area", active, "black", linestyle="--")
    add_rect(ax, "Original AvgSFChan converted", original, "#1f77b4", "#1f77b4", alpha=0.18)
    add_rect(ax, "Separated planes 1-2", separated_s12, "#d62728", "#d62728", alpha=0.13)
    add_rect(ax, "Separated planes 3-5", separated_s345, "#2ca02c", "#2ca02c", alpha=0.13)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position [cm]")
    ax.set_ylabel("y position [cm]")
    ax.set_title("SciFi Fiducial Definitions")
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)

    margin = 4.0
    ax.set_xlim(active_x[0] - margin, active_x[1] + margin)
    ax.set_ylim(active_y[0] - margin, active_y[1] + margin)

    text = (
        f"Areas [cm^2]\n"
        f"active: {area(active):.1f}\n"
        f"original: {area(original):.1f}\n"
        f"sep. S1-2: {area(separated_s12):.1f}\n"
        f"sep. S3-5: {area(separated_s345):.1f}"
    )
    ax.text(
        0.03,
        0.03,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )

    out_dir = Path("evaluation_region_partitions/scifi_fiducial_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"scifi_fiducial_comparison.{suffix}", bbox_inches="tight", dpi=180)

    print(f"active bounds: x={active_x}, y={active_y}, area={area(active):.3f} cm^2")
    print(f"original bounds: x={original_x}, y={original_y}, area={area(original):.3f} cm^2")
    print(f"separated S1-2 bounds: {separated_s12}, area={area(separated_s12):.3f} cm^2")
    print(f"separated S3-5 bounds: {separated_s345}, area={area(separated_s345):.3f} cm^2")
    print(f"wrote {out_dir / 'scifi_fiducial_comparison.png'}")
    print(f"wrote {out_dir / 'scifi_fiducial_comparison.pdf'}")


if __name__ == "__main__":
    main()
