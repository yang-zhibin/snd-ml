#!/usr/bin/env python3

import os
import json
import math
import argparse
import re
import numpy as np

import ROOT
import cabinetry
import matplotlib.pyplot as plt

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


def find_default_config(folder: str) -> str:
    config_path = os.path.join(folder, "config.yml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find config.yml in {folder}")
    return os.path.abspath(config_path)


def resolve_path(path_str: str, workdir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return os.path.abspath(os.path.join(workdir, path_str))


def absolutize_config_paths(config: dict, workdir: str) -> dict:
    config = dict(config)
    config["General"] = dict(config["General"])

    input_path = config["General"]["InputPath"]
    hist_folder = config["General"]["HistogramFolder"]

    config["General"]["HistogramFolder"] = resolve_path(hist_folder, workdir)

    if ":" in input_path:
        file_part, hist_part = input_path.split(":", 1)
        file_part = resolve_path(file_part, workdir)
        config["General"]["InputPath"] = f"{file_part}:{hist_part}"
    else:
        config["General"]["InputPath"] = resolve_path(input_path, workdir)

    return config


def nested_sum(x) -> float:
    return float(np.asarray(x, dtype=float).sum())


def load_metadata(workdir: str):
    meta_path = os.path.join(workdir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_lumi(workdir: str):
    lumi_path = os.path.join(workdir, "lumi.json")
    if not os.path.isfile(lumi_path):
        return None
    with open(lumi_path) as f:
        return json.load(f)


def hist_cfg_from_metadata(metadata: dict):
    if metadata is None:
        return None, "feature"

    hcfg = metadata["hist_cfg"]
    nbins = hcfg["nbins"]
    xmin = hcfg["xmin"]
    xmax = hcfg["xmax"]
    bin_width = (xmax - xmin) / nbins
    hist_cfg = (
        bin_width,
        xmin,
        xmax,
        hcfg["title"],
        hcfg["logy"],
        hcfg["extra"],
    )
    return hist_cfg, metadata["feature"]


def save_bestfit_parameters(fit_results, outdir):
    os.makedirs(outdir, exist_ok=True)

    param_names = list(fit_results.labels)
    bestfit = list(fit_results.bestfit)
    uncertainty = list(fit_results.uncertainty)

    json_data = []
    for name, val, err in zip(param_names, bestfit, uncertainty):
        json_data.append({
            "name": name,
            "value": float(val),
            "uncertainty": float(err),
        })

    json_path = os.path.join(outdir, "bestfit_parameters.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    txt_path = os.path.join(outdir, "bestfit_parameters.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'Parameter':35s} {'Value':>12s} {'Uncertainty':>12s}\n")
        f.write("-" * 65 + "\n")
        for name, val, err in zip(param_names, bestfit, uncertainty):
            f.write(f"{name:35s} {val:12.6g} {err:12.6g}\n")

    print(f"[info] saved best-fit parameters:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {txt_path}")


def compress_legend_label(label: str) -> str:
    """
    Convert long cabinetry sample labels into shorter legend labels.
    Examples:
      MC_neutron_90_100GeV -> n 90-100 GeV
      MC_kaon_20_30GeV     -> K 20-30 GeV
      MC_NC_numu           -> NC νμ
      MC_CC_nue            -> CC νe
    """
    label = label.strip()

    if label == "Data":
        return "Data"
    if label == "Uncertainty":
        return "Unc."

    # neutrino labels
    label = label.replace("numu", r"$\nu_\mu$")
    label = label.replace("nue", r"$\nu_e$")

    # sample prefixes
    label = label.replace("MC_neutron_", "n ")
    label = label.replace("MC_kaon_", "K ")
    label = label.replace("MC_NC_", "NC ")
    label = label.replace("MC_CC_", "CC ")
    label = label.replace("MC_", "")

    # convert energy range formatting
    m = re.match(r"^(.*?)(\d+)_(\d+)GeV$", label)
    if m:
        prefix, lo, hi = m.groups()
        label = f"{prefix}{lo}-{hi} GeV"

    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def infer_plot_tag(labels):
    """
    Guess whether a plot is prefit or postfit based on legend content.
    """
    lowered = [lab.lower() for lab in labels]
    if any("post-fit" in lab or "postfit" in lab for lab in lowered):
        return "postfit"
    if any("pre-fit" in lab or "prefit" in lab for lab in lowered):
        return "prefit"
    return "plot"
def compress_legend_label(label: str) -> str:
    label = label.strip()

    if label.lower() == "data":
        return "Data"
    if "uncertainty" in label.lower():
        return "Unc."
    if "post-fit" in label.lower() or "postfit" in label.lower():
        return "post-fit"
    if "pre-fit" in label.lower() or "prefit" in label.lower():
        return "pre-fit"

    label = label.replace("MC_neutron_", "n ")
    label = label.replace("MC_kaon_", "K ")
    label = label.replace("MC_NC_", "NC ")
    label = label.replace("MC_CC_", "CC ")
    label = label.replace("numu", r"$\nu_\mu$")
    label = label.replace("nue", r"$\nu_e$")
    label = label.replace("GeV", " GeV")
    label = label.replace("_", "-")

    return label

def infer_plot_tag(labels):
    lowered = [lab.lower() for lab in labels]
    if any("post-fit" in lab or "postfit" in lab for lab in lowered):
        return "postfit"
    if any("pre-fit" in lab or "prefit" in lab for lab in lowered):
        return "prefit"
    return "plot"

def save_compact_data_mc_figures(
    figs,
    outdir,
    suffix="compact",
    figsize=(22, 12),
    legend_ncol=2,
    legend_fontsize=13,
    save_png=False,
    dpi=200,
):
    """
    Save cabinetry data/MC figures with a larger plot area and a dedicated
    legend panel on the right. This is more reliable than fig.legend(...) when
    many legend entries are present.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    for i, item in enumerate(figs):
        fig = item["figure"]

        if len(fig.axes) < 1:
            print(f"[warning] figure {i} has no axes, skipping")
            plt.close(fig)
            continue

        # cabinetry usually gives:
        #   fig.axes[0] = main panel
        #   fig.axes[1] = ratio panel
        ax_main = fig.axes[0]
        ax_ratio = fig.axes[1] if len(fig.axes) > 1 else None

        # read legend entries from the main axis
        handles, labels = ax_main.get_legend_handles_labels()

        # if that fails, try existing legend object
        if (not handles or not labels) and ax_main.get_legend() is not None:
            old_leg = ax_main.get_legend()
            handles = old_leg.legend_handles
            labels = [txt.get_text() for txt in old_leg.get_texts()]

        if not handles or not labels:
            print(f"[warning] figure {i} has no legend entries, saving raw figure")
            region = item.get("region", f"region_{i}")
            base = os.path.join(outdir, f"{region}_{suffix}")
            fig.set_size_inches(*figsize)
            fig.savefig(base + ".pdf")
            if save_png:
                fig.savefig(base + ".png", dpi=dpi)
            plt.close(fig)
            continue

        short_labels = [compress_legend_label(lab) for lab in labels]

        # remove old legend from main axis
        old_legend = ax_main.get_legend()
        if old_legend is not None:
            old_legend.remove()

        # enlarge full figure
        fig.set_size_inches(*figsize)

        # ---------------------------------------------------------
        # Force the plot axes to occupy the left part of the figure
        # ---------------------------------------------------------
        # [left, bottom, width, height] in figure coordinates

        # main panel
        ax_main.set_position([0.08, 0.33, 0.52, 0.58])

        # ratio panel
        if ax_ratio is not None:
            ax_ratio.set_position([0.08, 0.12, 0.52, 0.16])

        # ---------------------------------------------------------
        # Add a dedicated invisible axis for the legend on the right
        # ---------------------------------------------------------
        ax_leg = fig.add_axes([0.64, 0.10, 0.33, 0.80])
        ax_leg.axis("off")

        ax_leg.legend(
            handles,
            short_labels,
            loc="center left",
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.5,
            labelspacing=0.8,
            borderaxespad=0.0,
        )

        region = item.get("region", f"region_{i}")
        tag = infer_plot_tag(labels)
        base = os.path.join(outdir, f"{region}_{tag}_{suffix}")

        fig.savefig(base + ".pdf")
        if save_png:
            fig.savefig(base + ".png", dpi=dpi)

        print(f"[info] saved compact figure: {base}.pdf")
        if save_png:
            print(f"[info] saved compact figure: {base}.png")

        plt.close(fig)
        
        
        
def run_fit(config_path: str, workspace_path: str = None, figures_dir: str = None):
    config_path = os.path.abspath(config_path)
    workdir = os.path.dirname(config_path)

    if workspace_path is None:
        workspace_path = os.path.join(workdir, "workspace.json")

    if figures_dir is None:
        figures_dir = f"{workdir}/figure"
    else:
        figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    npz_dir = f"{workdir}/npz"
    os.makedirs(npz_dir, exist_ok=True)


    metadata = load_metadata(workdir)
    lumi_info = load_lumi(workdir)
    hist_cfg, feature_name = hist_cfg_from_metadata(metadata)

    data_lumi = None
    if lumi_info is not None:
        data_lumi = lumi_info.get("data_lumi", None)

    print(f"[info] loading config: {config_path}")
    config = cabinetry.configuration.load(config_path)
    config = absolutize_config_paths(config, workdir)

    print(f"[info] InputPath = {config['General']['InputPath']}")
    print(f"[info] HistogramFolder = {config['General']['HistogramFolder']}")

    print("[info] collecting ROOT histograms")
    cabinetry.templates.collect(config)

    print("[info] postprocessing templates")
    cabinetry.templates.postprocess(config)

    print("[info] building workspace")
    workspace = cabinetry.workspace.build(config)

    with open(workspace_path, "w") as f:
        json.dump(workspace, f, indent=2)
    print(f"[info] workspace saved to: {workspace_path}")

    print("[info] building model and data")
    model, data = cabinetry.model_utils.model_and_data(workspace)

    print("[info] running fit")
    fit_results = cabinetry.fit.fit(model, data)
    save_bestfit_parameters(fit_results, os.path.join(workdir, "fit_results"))

    print("\n=== Fit results ===")
    cabinetry.fit.print_results(fit_results)

    print("\n[info] computing predictions")
    prefit_prediction = cabinetry.model_utils.prediction(model)
    postfit_prediction = cabinetry.model_utils.prediction(model, fit_results=fit_results)

    print(f"[info] saving plots in: {figures_dir}")

    # Save default cabinetry figures as before
    cabinetry.visualize.data_mc(
        prefit_prediction,
        data,
        config=config,
        figure_folder=figures_dir,
        close_figure=True,
    )

    cabinetry.visualize.data_mc(
        postfit_prediction,
        data,
        config=config,
        figure_folder=figures_dir,
        close_figure=True,
    )

    cabinetry.visualize.pulls(
        fit_results,
        figure_folder=figures_dir,
        close_figure=True,
    )

    cabinetry.visualize.correlation_matrix(
        fit_results,
        figure_folder=figures_dir,
        close_figure=True,
    )

    # Save improved-layout copies
    compact_figures_dir = os.path.join(figures_dir, "compact")
    os.makedirs(compact_figures_dir, exist_ok=True)

    prefit_figs = cabinetry.visualize.data_mc(
        prefit_prediction,
        data,
        config=config,
        figure_folder=figures_dir,
        close_figure=False,
    )

    save_compact_data_mc_figures(
        prefit_figs,
        outdir=compact_figures_dir,
        suffix="compact",
        figsize=(22, 12),
        legend_ncol=2,   # change to 3 if you want
        legend_fontsize=13,
        save_png=False,
    )

    postfit_figs = cabinetry.visualize.data_mc(
        postfit_prediction,
        data,
        config=config,
        figure_folder=figures_dir,
        close_figure=False,
    )

    save_compact_data_mc_figures(
        postfit_figs,
        outdir=compact_figures_dir,
        suffix="compact",
        figsize=(22, 12),
        legend_ncol=2,
        legend_fontsize=13,
        save_png=False,
    )

    print(f"[info] saving npz in: {npz_dir}")

    param_names = np.array(fit_results.labels, dtype=object)
    bestfit = np.array(fit_results.bestfit, dtype=float)
    uncertainty = np.array(fit_results.uncertainty, dtype=float)

    corr = getattr(fit_results, "correlation_matrix", None)
    if corr is None:
        corr = np.zeros((len(bestfit), len(bestfit)))
    else:
        corr = np.array(corr, dtype=float)

    region_names = [r["Name"] for r in config["Regions"]]
    postfit_yields = np.array(
        [nested_sum(postfit_prediction.model_yields[i]) for i in range(len(region_names))],
        dtype=float,
    )

    npz_path = os.path.join(npz_dir, "fit_results.npz")
    np.savez(
        npz_path,
        param_names=param_names,
        bestfit=bestfit,
        uncertainty=uncertainty,
        correlation_matrix=corr,
        region_names=np.array(region_names, dtype=object),
        postfit_yields=postfit_yields,
    )
    print(f"[info] saved npz: {npz_path}")

    print("\n=== Best-fit parameters ===")
    for name, val, err in zip(param_names, bestfit, uncertainty):
        print(f"{name:35s} = {val: .6g} ± {err:.6g}")

    print("\n=== Post-fit total yields by region ===")
    for region_name, total_yield in zip(region_names, postfit_yields):
        print(f"{region_name:20s} total = {total_yield:.6g}")

    print("\n[done]")
    print(f"config         : {config_path}")
    print(f"workspace      : {workspace_path}")
    print(f"figures        : {figures_dir}")
    print(f"compact_figures: {compact_figures_dir}")
    print(f"npz            : {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a cabinetry fit, save default cabinetry plots, and make grouped ROOT-style post-fit plots."
    )

    parser.add_argument(
        "--folder",
        default="./processed_hists/density_sndsw_scifi__base-cut-2-3-4-5-8-9-10-15__combinedHadrons__binSize500_1000to6000",
        help="Folder containing config.yml and ROOT file",
    )

    parser.add_argument(
        "--workspace",
        default=None,
        help="Optional output workspace JSON path",
    )

    parser.add_argument(
        "--figures",
        default=None,
        help="Optional figure directory for cabinetry plots (default: <folder>/figure)",
    )

    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    config_path = find_default_config(folder)

    run_fit(
        config_path=config_path,
        workspace_path=args.workspace,
        figures_dir=args.figures,
    )


if __name__ == "__main__":
    main()