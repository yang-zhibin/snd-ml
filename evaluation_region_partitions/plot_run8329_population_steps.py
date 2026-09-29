#!/usr/bin/env python3
"""Plot Run-8329 event populations and reconstructed-track retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_EVENTS = 298_285_666
PASSING_MUON_RECO_EVENTS = 82_145_120
PASSING_MUON_RECO_DQ_EVENTS = 11_025_227
LUMINOSITY_FB = 0.5323492746290897


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing the 299 sourceNNN.json v8 manifests.",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def manifest_counts(manifest_dir: Path) -> dict[str, int]:
    paths = sorted(manifest_dir.glob("source*.json"))
    if len(paths) != 299:
        raise RuntimeError(
            f"Expected 299 manifest summaries in {manifest_dir}, found {len(paths)}"
        )

    totals: dict[str, int] = {}
    sources: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = int(payload["source_file_index"])
        if source in sources:
            raise RuntimeError(f"Duplicate source index {source}")
        sources.add(source)
        for name, value in payload["counts"].items():
            totals[name] = totals.get(name, 0) + int(value)

    expected_sources = set(range(299))
    if sources != expected_sources:
        missing = sorted(expected_sources - sources)
        extra = sorted(sources - expected_sources)
        raise RuntimeError(f"Manifest source mismatch: missing={missing}, extra={extra}")
    return totals


def format_step_axis(
    ax: plt.Axes,
    labels: list[str],
    values: list[int],
    denominator: int,
    title: str,
    color: str,
) -> None:
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)

    ax.step(x, y, where="mid", linewidth=3.0, color=color)
    ax.scatter(x, y, s=75, color=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.fill_between(x, y, 1.0, step="mid", color=color, alpha=0.13)

    for index, value in enumerate(values):
        fraction = 100.0 * value / denominator
        ax.annotate(
            f"{value:,}\n{fraction:.3f}%",
            (index, value),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylim(5.0e6, 6.0e8)
    ax.set_ylabel("Events (log scale)")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)


def main() -> None:
    args = parse_args()
    counts = manifest_counts(args.manifest_dir)
    real_dq = counts["real_dq"]
    usable_reco = counts["reco_scifi_eligible"]

    if real_dq != 11_115_982:
        raise RuntimeError(f"Unexpected RealDataDQ total: {real_dq:,}")
    if usable_reco > PASSING_MUON_RECO_DQ_EVENTS:
        raise RuntimeError("Usable reconstructed-track count exceeds PassingMuonReco∩DQ")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), constrained_layout=True)

    format_step_axis(
        axes[0],
        ["Complete\nraw run", "PassingMuonReco", "RealDataDQ"],
        [RAW_EVENTS, PASSING_MUON_RECO_EVENTS, real_dq],
        RAW_EVENTS,
        "A. Populations derived from the raw run",
        "#2864b8",
    )
    axes[0].text(
        0.02,
        0.04,
        "PassingMuonReco and RealDataDQ are separate branches of the raw run.",
        transform=axes[0].transAxes,
        fontsize=9.5,
        color="#444444",
    )

    format_step_axis(
        axes[1],
        ["RealDataDQ", "PassingMuonReco\n∩ RealDataDQ", "Usable v8 reco\nSciFi track"],
        [real_dq, PASSING_MUON_RECO_DQ_EVENTS, usable_reco],
        real_dq,
        "B. Reconstructed-track retention inside RealDataDQ",
        "#c84630",
    )
    axes[1].set_ylim(5.0e6, 1.6e7)
    axes[1].text(
        0.02,
        0.04,
        f"Rejected by the v8 usable-track policy: {real_dq - usable_reco:,} events",
        transform=axes[1].transAxes,
        fontsize=9.5,
        color="#444444",
    )

    fig.suptitle(
        "Run 8329 event populations and reconstructed SciFi-track coverage",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        f"Run 8329 • 299 source files • integrated luminosity "
        f"{LUMINOSITY_FB:.6f} fb$^{{-1}}$",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "raw": RAW_EVENTS,
                "passing_muon_reco": PASSING_MUON_RECO_EVENTS,
                "real_dq": real_dq,
                "passing_muon_reco_intersect_dq": PASSING_MUON_RECO_DQ_EVENTS,
                "usable_v8_reco_scifi": usable_reco,
                "output_pdf": str(args.output_pdf),
                "output_png": str(args.output_png),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
