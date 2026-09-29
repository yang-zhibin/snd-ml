#!/usr/bin/env python3
"""Plot deterministic 3D SciFi-hit examples from a classifier training split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
TRAIN_SRC = REPO / "train_3d" / "src"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from train_3d.data.data_driven_fold_dataset import HierarchicalTrainingSampler
from train_3d.data.scifi_hit_event_dataset import SciFiHitEventDataset

try:
    from .plotting_common import CLASS_ORDER
except ImportError:
    from plotting_common import CLASS_ORDER


def select_examples(fold_views: pd.DataFrame, cv_fold: int, seed: int,
                    examples_per_class: int, reader_backend: str):
    """Use the training hierarchy and retain unique, readable, non-empty events."""
    sampler = HierarchicalTrainingSampler(
        fold_views,
        cv_fold=cv_fold,
        seed=seed,
        class_probabilities="uniform",
        domain_probabilities=None,
        muondis_density_probabilities="uniform",
    )
    dataset = SciFiHitEventDataset(
        references=None,
        reader_backend=reader_backend,
        tree_name="hit3D",
        max_open_files=8,
    )
    selected = {name: [] for name in CLASS_ORDER}
    seen = set()
    errors = []
    # Oversampling is bounded but generous enough to skip duplicate or empty events.
    references = sampler.sample_epoch(max(2000, examples_per_class * 200), epoch=0)
    try:
        for reference in references:
            class_name = str(reference.class_name)
            if class_name not in selected or len(selected[class_name]) >= examples_per_class:
                continue
            identity = (str(reference.hit3d_path), int(reference.source_entry))
            if identity in seen:
                continue
            seen.add(identity)
            try:
                event = dataset[reference]
            except Exception as exc:  # Preserve other usable examples and report failures.
                errors.append(f"{identity[0]}:{identity[1]}: {exc}")
                continue
            if int(event["n_candidates"]) <= 0:
                continue
            selected[class_name].append((reference, event))
            if all(len(selected[name]) >= examples_per_class for name in CLASS_ORDER):
                break
    finally:
        dataset.reader.close()

    missing = {
        name: examples_per_class - len(events)
        for name, events in selected.items()
        if len(events) < examples_per_class
    }
    if missing:
        detail = "\n".join(errors[:10])
        raise RuntimeError(f"Could not select requested examples: {missing}\n{detail}")
    return selected


def plot_limits(selected):
    positions = np.concatenate([
        event["position"]
        for events in selected.values()
        for _, event in events
    ])
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    return lower - 0.04 * span, upper + 0.04 * span


def qdc_style(selected):
    transformed = np.concatenate([
        np.log1p(event["qdc"])
        for events in selected.values()
        for _, event in events
    ])
    lower, upper = np.percentile(transformed, [1.0, 99.0])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        lower = float(np.nanmin(transformed))
        upper = max(float(np.nanmax(transformed)), lower + 1.0)
    return Normalize(vmin=float(lower), vmax=float(upper), clip=True)


def draw_event(ax, reference, event, limits, norm, title):
    position = np.asarray(event["position"], dtype=float)
    color = np.log1p(np.asarray(event["qdc"], dtype=float))
    scaled = norm(color)
    sizes = 5.0 + 15.0 * scaled
    ax.scatter(
        position[:, 0], position[:, 1], position[:, 2],
        c=color, cmap="viridis", norm=norm, s=sizes,
        alpha=0.78, linewidths=0, rasterized=True,
    )
    lower, upper = limits
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(np.maximum(upper - lower, 1.0))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x [cm]", labelpad=1)
    ax.set_ylabel("y [cm]", labelpad=1)
    ax.set_zlabel("z [cm]", labelpad=1)
    ax.tick_params(labelsize=6, pad=0)
    density = f" | {reference.density_bin}" if reference.density_bin else ""
    identity = f"run {event['run_id']} | event {event['event_id']} | entry {reference.source_entry}"
    ax.set_title(
        f"{title}\n{reference.domain}{density} | {event['n_candidates']:,} SciFi candidates\n{identity}",
        fontsize=8.2, pad=8,
    )


def add_colorbar(fig, axes, norm):
    scalar = ScalarMappable(norm=norm, cmap="viridis")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=np.asarray(axes).ravel().tolist(),
                            fraction=0.022, pad=0.025)
    colorbar.set_label("log1p(QDC), shared across panels")


def plot_overview(selected, output_dir: Path, cv_fold: int, limits, norm):
    fig = plt.figure(figsize=(20, 10.5))
    axes = []
    for index, class_name in enumerate(CLASS_ORDER):
        ax = fig.add_subplot(2, 4, index + 1, projection="3d")
        reference, event = selected[class_name][0]
        draw_event(ax, reference, event, limits, norm, class_name)
        axes.append(ax)
    note = fig.add_subplot(2, 4, 8)
    note.axis("off")
    note.text(
        0.04, 0.95,
        f"Training examples\ncv_fold={cv_fold}\n\n"
        "Selection\n• exact training fold view\n• uniform class sampling\n"
        "• natural domain composition\n• uniform muon-DIS density bins\n\n"
        "Display\n• model-input SciFi candidates only\n• raw detector coordinates\n"
        "• shared log1p(QDC) scale\n• identical axes and camera",
        va="top", fontsize=11, linespacing=1.35,
    )
    fig.suptitle(
        "Data-driven 3D classifier: representative training events",
        fontsize=17, fontweight="bold", y=0.98,
    )
    add_colorbar(fig, axes, norm)
    fig.subplots_adjust(left=0.03, right=0.93, bottom=0.05, top=0.91,
                        wspace=0.08, hspace=0.18)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"training_events_3d_cv{cv_fold}.{suffix}",
                    dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_class_examples(class_name, examples, output_dir, cv_fold, limits, norm):
    fig = plt.figure(figsize=(18, 6.2))
    axes = []
    for index, (reference, event) in enumerate(examples):
        ax = fig.add_subplot(1, len(examples), index + 1, projection="3d")
        draw_event(ax, reference, event, limits, norm, f"Example {index + 1}")
        axes.append(ax)
    fig.suptitle(f"{class_name}: training events from cv_fold={cv_fold}",
                 fontsize=16, fontweight="bold", y=0.98)
    add_colorbar(fig, axes, norm)
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.85, wspace=0.08)
    fig.savefig(output_dir / f"{class_name}_examples.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def write_manifest(selected, path, cv_fold, seed):
    rows = []
    for class_name in CLASS_ORDER:
        for example_index, (reference, event) in enumerate(selected[class_name], 1):
            rows.append({
                "cv_fold": int(cv_fold), "split": "train", "seed": int(seed),
                "class_name": class_name, "example_index": example_index,
                "domain": reference.domain, "density_bin": reference.density_bin,
                "dataset_file_uid": reference.dataset_file_uid,
                "hit3d_path": reference.hit3d_path,
                "source_entry": int(reference.source_entry),
                "run_id": int(event["run_id"]), "event_id": int(event["event_id"]),
                "event_index": int(event["event_index"]),
                "pdg_code": int(event["pdg_code"]), "energy": float(event["energy"]),
                "n_scifi_candidates": int(event["n_candidates"]),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def generate(fold_views_path, output_dir, cv_fold=0, seed=42,
             examples_per_class=3, reader_backend="pyroot"):
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = pd.read_csv(fold_views_path, keep_default_na=False)
    selected = select_examples(
        views, cv_fold, seed, examples_per_class, reader_backend
    )
    limits = plot_limits(selected)
    norm = qdc_style(selected)
    plot_overview(selected, output_dir, cv_fold, limits, norm)
    for class_name in CLASS_ORDER:
        plot_class_examples(
            class_name, selected[class_name], output_dir, cv_fold, limits, norm
        )
    write_manifest(selected, output_dir / "selected_events.csv", cv_fold, seed)
    print(f"Wrote {len(CLASS_ORDER) * examples_per_class} event displays to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-views", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-class", type=int, default=3)
    parser.add_argument("--reader-backend", choices=("pyroot", "uproot"), default="pyroot")
    args = parser.parse_args()
    generate(
        args.fold_views, args.output_dir, args.cv_fold, args.seed,
        args.examples_per_class, args.reader_backend,
    )


if __name__ == "__main__":
    main()
