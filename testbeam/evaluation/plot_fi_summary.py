#!/usr/bin/env python3
import re
import glob
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_energy_particle(path):
    """
    Parse energy + particle from filename parts like:
      ..._E50GeV_Ppi+_MC_importance.csv
    Returns (energy_GeV:int, particle:str) or (None, None) if not matched.
    """
    base = os.path.basename(path)
    m = re.search(r"_E(\d+)GeV_([^_]+)_", base)
    if not m:
        return None, None
    E = int(m.group(1))
    p = m.group(2)
    return E, p


def load_all(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")

    rows = []
    for f in files:
        E, p = parse_energy_particle(f)
        if E is None:
            continue
        df = pd.read_csv(f, index_col=0)

        # Expect these columns (from earlier code)
        # If your CSV columns differ, change here
        for metric in ["spearman_abs_rho", "mutual_info"]:
            if metric not in df.columns:
                raise RuntimeError(f"{f} missing column '{metric}'. Has: {list(df.columns)}")

        rows.append((E, p, df))
    if not rows:
        raise RuntimeError("Matched files but could not parse energy/particle from names.")
    return rows


def make_heatmap_matrix(rows, particle, metric, topk=25, normalize=False):
    """
    Build matrix: index=features, columns=energies, values=metric.
    We select topk features by mean importance across energies for this particle.
    """
    # collect per energy
    perE = {}
    for E, p, df in rows:
        if p != particle:
            continue
        perE[E] = df[metric]

    if not perE:
        return None

    mat = pd.DataFrame(perE).sort_index(axis=1)  # columns are energies
    # choose topk features by mean across energies
    top_features = mat.mean(axis=1).sort_values(ascending=False).head(topk).index
    mat = mat.loc[top_features]

    if normalize:
        # normalize each feature row to [0,1] across energies for shape comparison
        vmin = mat.min(axis=1)
        vmax = mat.max(axis=1)
        denom = (vmax - vmin).replace(0, np.nan)
        mat = (mat.sub(vmin, axis=0)).div(denom, axis=0).fillna(0.0)

    return mat


def plot_heatmaps_for_particle(rows, particle, outdir, topk=25, normalize=False):
    os.makedirs(outdir, exist_ok=True)

    mats = {}
    for metric in ["spearman_abs_rho", "mutual_info"]:
        mats[metric] = make_heatmap_matrix(rows, particle, metric, topk=topk, normalize=normalize)

    if mats["spearman_abs_rho"] is None and mats["mutual_info"] is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * topk)), constrained_layout=True)

    for ax, metric, title in [
        (axes[0], "spearman_abs_rho", "Spearman |ρ|"),
        (axes[1], "mutual_info", "Mutual Information"),
    ]:
        mat = mats[metric]
        if mat is None:
            ax.axis("off")
            ax.set_title(f"{title} (no data)")
            continue

        im = ax.imshow(mat.values, aspect="auto")
        ax.set_title(f"{particle} — {title}" + (" (row-normalized)" if normalize else ""))
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Feature")

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")

        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index.tolist())

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = os.path.join(outdir, f"FI_heatmaps_{particle}" + ("_norm" if normalize else "") + ".png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_topk_trends(rows, particle, metric, outdir, topk=8):
    """
    Line plot: importance vs energy for the topk features (by mean importance).
    """
    os.makedirs(outdir, exist_ok=True)

    mat = make_heatmap_matrix(rows, particle, metric, topk=max(topk, 25), normalize=False)
    if mat is None:
        return

    # choose topk for lines
    top_features = mat.mean(axis=1).sort_values(ascending=False).head(topk).index
    mat = mat.loc[top_features].sort_index(axis=1)

    plt.figure(figsize=(10, 5))
    for feat in mat.index:
        plt.plot(mat.columns.values, mat.loc[feat].values, marker="o", label=feat)

    plt.xlabel("Energy (GeV)")
    plt.ylabel(metric)
    plt.title(f"{particle} — top {topk} feature trends ({metric})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = os.path.join(outdir, f"FI_trends_{particle}_{metric}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for *_importance.csv files")
    ap.add_argument("--outdir", default="plots_FI_summary", help="Output directory")
    ap.add_argument("--topk-heatmap", type=int, default=25, help="Top-k features in heatmaps")
    ap.add_argument("--topk-trends", type=int, default=8, help="Top-k features in trend lines")
    ap.add_argument("--normalize", action="store_true", help="Row-normalize heatmaps to emphasize energy dependence shape")
    args = ap.parse_args()

    rows = load_all(args.glob)
    particles = sorted({p for _, p, _ in rows})
    print("Found particles:", particles)

    for p in particles:
        plot_heatmaps_for_particle(rows, p, args.outdir, topk=args.topk_heatmap, normalize=args.normalize)
        plot_topk_trends(rows, p, "spearman_abs_rho", args.outdir, topk=args.topk_trends)
        plot_topk_trends(rows, p, "mutual_info", args.outdir, topk=args.topk_trends)


if __name__ == "__main__":
    main()
    
