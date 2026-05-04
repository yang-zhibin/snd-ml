#!/usr/bin/env python3

import os
import re
import glob
import math
import json
import argparse
from collections import defaultdict

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


KAON_BINS = [
    "MC_kaon_5-10GeV",
    "MC_kaon_10-20GeV",
    "MC_kaon_20-30GeV",
    "MC_kaon_30-40GeV",
    "MC_kaon_50-60GeV",
    "MC_kaon_60-70GeV",
    "MC_kaon_70-80GeV",
    "MC_kaon_80-90GeV",
    "MC_kaon_90-100GeV",
    "MC_kaon_100-150GeV",
]

NEUTRON_BINS = [
    "MC_neutron_5-10GeV",
    "MC_neutron_10-20GeV",
    "MC_neutron_20-30GeV",
    "MC_neutron_30-40GeV",
    "MC_neutron_50-60GeV",
    "MC_neutron_60-70GeV",
    "MC_neutron_70-80GeV",
    "MC_neutron_80-90GeV",
    "MC_neutron_90-100GeV",
    "MC_neutron_100-150GeV",
]

NEUTRINO_CATS = [
    "MC_NC_nue",
    "MC_NC_numu",
    "MC_CC_nue",
    "MC_CC_numu",
]


def print_hist_summary(tag, hist):
    if hist is None:
        print(f"[{tag}] hist = None")
        return

    nb = hist.GetNbinsX()
    print(
        f"[{tag}] "
        f"entries={hist.GetEntries():.6g} "
        f"vis={hist.Integral():.6g} "
        f"all={hist.Integral(0, nb+1):.6g} "
        f"uf={hist.GetBinContent(0):.6g} "
        f"of={hist.GetBinContent(nb+1):.6g} "
        f"max={hist.GetMaximum():.6g}"
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def sanitize(s: str) -> str:
    s = str(s).strip()
    s = s.replace(">=", "ge").replace("<=", "le")
    s = s.replace(">", "gt").replace("<", "lt")
    s = s.replace("==", "eq").replace("!=", "neq")
    s = s.replace("&&", "_and_").replace("||", "_or_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def clone_hist(hist, new_name: str):
    h = hist.Clone(new_name)
    h.SetDirectory(0)
    return h


def get_total_lumi(root_file):
    obj = root_file.Get("total_lumi")
    if not obj:
        raise RuntimeError(f"Could not find 'total_lumi' in {root_file.GetName()}")
    return float(obj.GetVal())


def partition_from_filename(filepath):
    base = os.path.basename(filepath)
    m = re.match(r"^hist_(.+)\.root$", base)
    return m.group(1) if m else None


def classify_partition(partition):
    if partition.startswith("real_data"):
        return "real_data"

    if partition in set(NEUTRINO_CATS):
        return partition

    m = re.match(r"^(MC_(?:kaon|neutron)_\d+-\d+GeV)(?:_Max\d+-\d+)?$", partition)
    if m:
        return m.group(1)

    return None


def sanitize_expr(expr):
    if not expr or not expr.strip():
        return "nocut"

    expr = expr.strip().replace(" ", "")
    replacements = {
        "&&": "_",
        "||": "_",
        ">=": "ge",
        "<=": "le",
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        "<": "lt",
        "=": "",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "/": "_",
        "*": "_",
        "+": "_",
        "-": "m",
        ".": "p",
        ",": "_",
        ":": "_",
        ";": "_",
        '"': "",
        "'": "",
    }

    for old, new in replacements.items():
        expr = expr.replace(old, new)

    while "__" in expr:
        expr = expr.replace("__", "_")

    return expr.strip("_")


def sanitize_cut(base_cut="", extra_cut=""):
    parts = []

    if base_cut and base_cut.strip():
        base_tokens = base_cut.replace(",", " ").split()
        parts.append("base-cut-" + "-".join(base_tokens))

    if extra_cut and extra_cut.strip():
        parts.append("extra-cut-" + sanitize_expr(extra_cut))

    if not parts:
        return "nocut"

    return "__".join(parts)


def build_output_subdir(args, hist_cfg):
    bin_size, xmin, xmax, _, _, _ = hist_cfg
    feature_name = sanitize(args.feature)
    cut_tag = sanitize_cut(args.base_cut, args.extra_cut)
    binning = f"binSize{bin_size:g}_{xmin:g}to{xmax:g}"

    hadron_tag = "combinedHadrons" if args.combine_hadrons else "splitHadrons"
    return f"{feature_name}__{cut_tag}__{hadron_tag}__{binning}"


def add_underflow_to_first_bin(hist):
    if hist is None:
        return

    hist.SetBinContent(1, hist.GetBinContent(1) + hist.GetBinContent(0))
    hist.SetBinError(1, math.sqrt(hist.GetBinError(1) ** 2 + hist.GetBinError(0) ** 2))
    hist.SetBinContent(0, 0.0)
    hist.SetBinError(0, 0.0)


def add_overflow_to_last_bin(hist):
    if hist is None:
        return

    nb = hist.GetNbinsX()
    hist.SetBinContent(nb, hist.GetBinContent(nb) + hist.GetBinContent(nb + 1))
    hist.SetBinError(nb, math.sqrt(hist.GetBinError(nb) ** 2 + hist.GetBinError(nb + 1) ** 2))
    hist.SetBinContent(nb + 1, 0.0)
    hist.SetBinError(nb + 1, 0.0)


def zero_hist(hist):
    if hist is None:
        return
    hist.Reset("ICES")
    hist.SetDirectory(0)


def sanitize_hist_bins(hist):
    if hist is None:
        return None

    nb = hist.GetNbinsX()
    for ibin in range(0, nb + 2):
        content = hist.GetBinContent(ibin)
        error = hist.GetBinError(ibin)

        if not math.isfinite(content):
            hist.SetBinContent(ibin, 0.0)
        if not math.isfinite(error):
            hist.SetBinError(ibin, 0.0)

    return hist


def make_draw_clone(hist, name):
    if hist is None:
        return None
    cloned = hist.Clone(name)
    cloned.SetDirectory(0)
    sanitize_hist_bins(cloned)
    return cloned


def style_hist(hist, color, fill=True):
    if hist is None:
        return
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetLineWidth(2)
    if fill:
        hist.SetFillColor(color)
        hist.SetFillStyle(1001)


def style_data_hist(hist):
    if hist is None:
        return
    hist.SetLineColor(ROOT.kBlack)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(1.0)
    hist.SetLineWidth(2)


def collect_files_and_lumi(input_dir, tree_name):
    filepaths = sorted(glob.glob(os.path.join(input_dir, "hist_*.root")))
    if not filepaths:
        raise RuntimeError(f"No files matching hist_*.root found in {input_dir}")

    category_files = defaultdict(list)
    grouped_lumi = defaultdict(float)
    open_files = []

    for filepath in filepaths:
        partition = partition_from_filename(filepath)
        if partition is None:
            continue

        category = classify_partition(partition)
        if category is None:
            print(f"[skip] unrecognized partition: {partition}")
            continue

        root_file = ROOT.TFile.Open(filepath)
        if not root_file or root_file.IsZombie():
            print(f"[skip] could not open {filepath}")
            continue

        open_files.append(root_file)

        tree = root_file.Get(tree_name)
        if not tree:
            print(f"[skip] tree '{tree_name}' not found in {filepath}")
            continue

        try:
            lumi = get_total_lumi(root_file)
        except Exception as e:
            print(f"[skip] failed to read lumi from {filepath}: {e}")
            continue

        category_files[category].append(filepath)
        grouped_lumi[category] += lumi

        print(f"[ok] {partition:35s} -> {category:25s} lumi={lumi}")

    if "real_data" not in category_files:
        raise RuntimeError("No real_data files found.")

    return category_files, grouped_lumi, open_files


def build_chains(category_files, tree_name):
    chains = {}

    for category, files in category_files.items():
        chain = ROOT.TChain(tree_name)
        for filepath in files:
            chain.Add(filepath)

        chains[category] = chain
        print(f"[chain] {category:25s}: {len(files)} files, entries={chain.GetEntries()}")

    return chains


def get_category_selection(category, base_cut, extra_cut, cut_info):
    base_tokens = base_cut.replace(",", " ").split() if base_cut else []

    if "real_data" not in category:
        base_tokens = [tok for tok in base_tokens if tok not in {"2", "3", "4"}]

    category_base_cut = " ".join(base_tokens)
    selection, _ = combine_cuts(category_base_cut, extra_cut, cut_info)
    return selection


def combine_cuts(base_cut, extra_cut, cut_info):
    selected_exprs = []
    cut_names = []

    if base_cut:
        tokens = base_cut.replace(",", " ").split()
        for tok in tokens:
            idx = int(tok)
            if idx not in cut_info:
                raise KeyError(f"Cut index {idx} not found in cut_info")
            name, expr = cut_info[idx]
            cut_names.append(name)
            selected_exprs.append(f"({expr})")

    if extra_cut and extra_cut.strip():
        selected_exprs.append(f"({extra_cut.strip()})")
        cut_names.append("extra_cut")

    combined_cut = " && ".join(selected_exprs) if selected_exprs else ""
    return combined_cut, cut_names


def make_histograms(chains, feature, hist_cfg, base_cut="", extra_cut="", fold_underflow=False, fold_overflow=False):
    bin_size, x_min, x_max, _, _, feature_cut = hist_cfg
    n_bins = int(round((x_max - x_min) / bin_size))

    if not math.isclose(x_min + n_bins * bin_size, x_max, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"Invalid binning for {feature}: xmin={x_min}, xmax={x_max}, "
            f"bin_size={bin_size} does not divide the range exactly"
        )
    grouped_hists = {}

    if feature_cut != "":
        if extra_cut:
            extra_cut = extra_cut + " && " + feature_cut
        else:
            extra_cut = feature_cut

    for category, chain in chains.items():
        safe_cat = re.sub(r"[^A-Za-z0-9_]", "_", category)
        hname = f"h_{safe_cat}"

        selection = get_category_selection(category, base_cut, extra_cut, cut_info)

        draw_expr = f"{feature}>>{hname}({n_bins},{x_min},{x_max})"
        selected = chain.Draw(draw_expr, selection, "goff")

        print(f"[draw] {category:25s} expr={draw_expr} cut={selection!r} selected={selected}")

        if selected < 0:
            raise RuntimeError(f"TChain::Draw failed for category={category}")

        hist_tmp = chain.GetHistogram()
        if not hist_tmp:
            raise RuntimeError(f"GetHistogram() returned null for category={category}")

        hist = hist_tmp.Clone(f"{hname}_clone")
        hist.SetDirectory(0)

        if fold_underflow:
            add_underflow_to_first_bin(hist)
        if fold_overflow:
            add_overflow_to_last_bin(hist)

        sanitize_hist_bins(hist)
        grouped_hists[category] = hist

        nb = hist.GetNbinsX()
        print(
            f"[raw {category}] "
            f"entries={hist.GetEntries():.6g} "
            f"vis={hist.Integral():.6g} "
            f"all={hist.Integral(0, nb+1):.6g} "
            f"uf={hist.GetBinContent(0):.6g} "
            f"of={hist.GetBinContent(nb+1):.6g} "
            f"max={hist.GetMaximum():.6g}"
        )

    return grouped_hists


def scale_mc_to_data(grouped_hists, grouped_lumi, data_category="real_data"):
    data_lumi = grouped_lumi[data_category]

    if not math.isfinite(data_lumi) or data_lumi <= 0:
        raise RuntimeError(f"Invalid data lumi: {data_lumi}")

    print(f"\nTotal data lumi = {data_lumi:.6g}")

    for category, hist in grouped_hists.items():
        if category == data_category:
            continue

        mc_lumi = grouped_lumi.get(category, float("nan"))

        if not math.isfinite(mc_lumi) or mc_lumi <= 0:
            zero_hist(hist)
            print(f"[scale] {category:25s} : set to 0 (invalid lumi = {mc_lumi})")
            continue

        scale = data_lumi / mc_lumi

        if not math.isfinite(scale):
            zero_hist(hist)
            print(f"[scale] {category:25s} : set to 0 (invalid scale = {scale})")
            continue

        hist.Scale(scale)
        sanitize_hist_bins(hist)
        print(f"[scale] {category:25s} : x {scale:.6g}")

    return data_lumi


def apply_had_scale_factor(grouped_hists, had_scale_factor):
    if had_scale_factor is None or had_scale_factor == 1:
        return

    for category, hist in grouped_hists.items():
        if hist is None:
            continue

        if category in KAON_BINS or category in NEUTRON_BINS or category in ["kaon", "neutron"]:
            hist.Scale(had_scale_factor)
            sanitize_hist_bins(hist)
            print(f"[had-scale] {category:25s} : x {had_scale_factor:.6g}")


def save_metadata(outdir, args, hist_cfg, region_name):
    bin_size, xmin, xmax, title, logy, feature_cut = hist_cfg
    n_bins = int(round((xmax - xmin) / bin_size))

    metadata = {
        "feature": args.feature,
        "hist_cfg": {
            "bin_size": bin_size,
            "nbins": n_bins,
            "xmin": xmin,
            "xmax": xmax,
            "title": title,
            "logy": logy,
            "extra": feature_cut,
        },
        "base_cut": args.base_cut,
        "extra_cut": args.extra_cut,
        "cut_name": sanitize_cut(args.base_cut, args.extra_cut),
        "tree": args.tree,
        "region_name": region_name,
        "fold_overflow": args.fold_overflow,
        "fold_underflow": args.fold_underflow,
        "input_dir": args.input_dir,
        "had_scale_factor": args.had_scale_factor,
        "combine_hadrons": args.combine_hadrons,
        "plot_title": args.title,
    }
    save_json(metadata, os.path.join(outdir, "metadata.json"))


def save_lumi_info(outdir, grouped_lumi, data_category="real_data", lumi_unit="unknown"):
    lumi_info = {
        "data_category": data_category,
        "data_lumi": grouped_lumi.get(data_category),
        "lumi_unit": lumi_unit,
        "all_lumi": dict(grouped_lumi),
    }
    save_json(lumi_info, os.path.join(outdir, "lumi.json"))


def write_minimal_cabinetry_config(outdir, region_name, written_samples, poi_name=None, min_integral_for_normfactor=1e-12):
    sample_lines = []
    normfactor_lines = []

    templates_dir = os.path.join(outdir, "templates")
    os.makedirs(templates_dir, exist_ok=True)

    non_data_samples = [s for s in written_samples if not s["is_data"]]

    floating_samples = [
        s for s in non_data_samples
        if abs(float(s.get("integral", 0.0))) > min_integral_for_normfactor
    ]

    if poi_name is None:
        if floating_samples:
            poi_name = f"mu_{floating_samples[0]['sample_name']}"
        else:
            poi_name = "mu_signal"

    for s in written_samples:
        if s["is_data"]:
            sample_lines.append(
                '  - Name: "Data"\n'
                '    SamplePath: "data"\n'
                '    Data: true'
            )
        else:
            sample_name = s["sample_name"]
            sample_lines.append(
                f'  - Name: "{sample_name}"\n'
                f'    SamplePath: "{sample_name}"'
            )

    for s in floating_samples:
        sample_name = s["sample_name"]

        if sample_name.startswith("MC_CC") or sample_name.startswith("MC_NC"):
            bounds = "[0.999, 1.001]"
        else:
            bounds = "[0.0, 10.0]"

        normfactor_lines.append(
            f'  - Name: "mu_{sample_name}"\n'
            f'    Samples: "{sample_name}"\n'
            f'    Regions: ["{region_name}"]\n'
            f'    Nominal: 1.0\n'
            f'    Bounds: {bounds}'
        )

    normfactors_block = "\n".join(normfactor_lines) if normfactor_lines else "  []"

    cfg = (
        "General:\n"
        '  Measurement: "template_fit"\n'
        f'  POI: "{poi_name}"\n'
        '  InputPath: "{RegionPath}.root:{SamplePath}_{VariationPath}"\n'
        f'  HistogramFolder: "{templates_dir}"\n'
        '  VariationPath: "nominal"\n'
        "\n"
        "Regions:\n"
        f'  - Name: "{region_name}"\n'
        f'    RegionPath: "{region_name}"\n'
        "\n"
        "Samples:\n"
        f"{chr(10).join(sample_lines)}\n"
        "\n"
        "NormFactors:\n"
        f"{normfactors_block}\n"
    )

    skipped_samples = [
        s for s in non_data_samples
        if abs(float(s.get("integral", 0.0))) <= min_integral_for_normfactor
    ]

    if floating_samples:
        print("[config] floating samples:")
        for s in floating_samples:
            sample_name = s["sample_name"]
            if sample_name.startswith("MC_CC") or sample_name.startswith("MC_NC"):
                print(f"  {sample_name}: integral={s['integral']:.6g}, bounds=[0.999, 1.001]")
            else:
                print(f"  {sample_name}: integral={s['integral']:.6g}, bounds=[0.0, 10.0]")

    if skipped_samples:
        print("[config] samples kept fixed because integral is zero/tiny:")
        for s in skipped_samples:
            print(f"  {s['sample_name']}: integral={s['integral']:.6g}")

    with open(os.path.join(outdir, "config.yml"), "w") as f:
        f.write(cfg)


def save_histograms_to_cabinetry_root(
    grouped_hists,
    outdir,
    region_name,
    data_category="real_data",
    category_name_map=None,
    variation="nominal",
    write_yaml=True,
):
    os.makedirs(outdir, exist_ok=True)

    if category_name_map is None:
        category_name_map = {}

    root_path = os.path.join(outdir, f"{region_name}.root")
    fout = ROOT.TFile.Open(root_path, "RECREATE")
    if not fout or fout.IsZombie():
        raise OSError(f"Could not create output ROOT file: {root_path}")

    written_samples = []

    for category, hist in grouped_hists.items():
        if hist is None:
            continue

        sample_name = category_name_map.get(category, category)
        if category == data_category:
            sample_name = "data"

        sample_name = sanitize(sample_name)
        hist_name = f"{sample_name}_{variation}"

        h_out = clone_hist(hist, hist_name)
        fout.cd()
        h_out.Write(hist_name)

        integral = float(h_out.Integral())
        written_samples.append(
            {
                "category": category,
                "sample_name": sample_name,
                "hist_name": hist_name,
                "is_data": category == data_category,
                "integral": integral,
                "nbins": int(h_out.GetNbinsX()),
                "xmin": float(h_out.GetXaxis().GetXmin()),
                "xmax": float(h_out.GetXaxis().GetXmax()),
            }
        )

    fout.Close()

    manifest = {
        "region_name": region_name,
        "root_file": root_path,
        "variation": variation,
        "samples": written_samples,
    }
    save_json(manifest, os.path.join(outdir, f"{region_name}_manifest.json"))

    if write_yaml:
        write_minimal_cabinetry_config(
            outdir=outdir,
            region_name=region_name,
            written_samples=written_samples,
        )

    return root_path, written_samples


def combine_categories(grouped_hists, categories, output_name):
    combined = None

    for cat in categories:
        hist = grouped_hists.get(cat)
        if hist is None:
            continue

        sanitize_hist_bins(hist)

        if combined is None:
            combined = hist.Clone(output_name)
            combined.SetDirectory(0)
        else:
            combined.Add(hist)

    if combined is not None:
        sanitize_hist_bins(combined)
    return combined


def combine_hadron_backgrounds(grouped_hists):
    final_hists = {}

    data_hist = grouped_hists.get("real_data")
    if data_hist is None:
        raise RuntimeError("Missing real_data histogram")

    final_hists["real_data"] = data_hist.Clone("h_real_data")
    final_hists["real_data"].SetDirectory(0)

    kaon_hist = combine_categories(grouped_hists, KAON_BINS, "h_kaon")
    neutron_hist = combine_categories(grouped_hists, NEUTRON_BINS, "h_neutron")

    if kaon_hist is not None:
        final_hists["kaon"] = kaon_hist
    if neutron_hist is not None:
        final_hists["neutron"] = neutron_hist

    for cat in NEUTRINO_CATS:
        hist = grouped_hists.get(cat)
        if hist is None:
            continue

        cloned = hist.Clone(f"h_{cat}")
        cloned.SetDirectory(0)
        sanitize_hist_bins(cloned)
        final_hists[cat] = cloned

    return final_hists


def main(args):
    if args.feature not in hist_info:
        raise ValueError(f"{args.feature} not defined in hist_info")

    os.makedirs(args.outdir, exist_ok=True)
    hist_cfg = hist_info[args.feature]

    category_files, grouped_lumi, open_files = collect_files_and_lumi(args.input_dir, args.tree)
    chains = build_chains(category_files, args.tree)

    grouped_hists = make_histograms(
        chains=chains,
        feature=args.feature,
        hist_cfg=hist_cfg,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        fold_underflow=args.fold_underflow,
        fold_overflow=args.fold_overflow,
    )

    scale_mc_to_data(grouped_hists, grouped_lumi, data_category="real_data")

    if args.combine_hadrons:
        print("[info] combining kaon/neutron into single templates for cabinetry")
        grouped_hists = combine_hadron_backgrounds(grouped_hists)
    else:
        print("[info] keeping kaon/neutron bins separate for cabinetry")

    apply_had_scale_factor(grouped_hists, args.had_scale_factor)

    subfolder = build_output_subdir(args, hist_cfg)
    full_outdir = os.path.join(args.outdir, subfolder)
    os.makedirs(full_outdir, exist_ok=True)

    print("Grouped histograms:")
    for key, hist in grouped_hists.items():
        if hist is None:
            print(f"  {key}: None")
        else:
            print(f"  {key}: name={hist.GetName()}, integral={hist.Integral()}")

    region_name = sanitize(args.region_name)

    category_name_map = {
        "real_data": "data",
    }

    root_path, written_samples = save_histograms_to_cabinetry_root(
        grouped_hists=grouped_hists,
        outdir=full_outdir,
        region_name=region_name,
        data_category="real_data",
        category_name_map=category_name_map,
        variation="nominal",
        write_yaml=True,
    )

    save_metadata(
        outdir=full_outdir,
        args=args,
        hist_cfg=hist_cfg,
        region_name=region_name,
    )

    save_lumi_info(
        outdir=full_outdir,
        grouped_lumi=grouped_lumi,
        data_category="real_data",
        lumi_unit="unknown",
    )

    print(f"\nSaved cabinetry input to: {full_outdir}")
    print(f"ROOT file: {root_path}")
    print("Written samples:")
    for s in written_samples:
        print(
            f"  category={s['category']}, sample={s['sample_name']}, "
            f"hist={s['hist_name']}, integral={s['integral']:.6g}, "
            f"is_data={s['is_data']}"
        )

    for f in open_files:
        if f:
            f.Close()


hist_info = {
    # Convention:
    # (bin_size, xmin, xmax, title, logy, feature_cut)
    "density_sndsw_scifi": (500, 1000, 6000, "Sum of SciFi Density Weight (SNDSW)", False, ""),
}


def scifi_plane_cut(plane, xmin, xmax, ymin, ymax):
    return (
        f"(count_scifi{plane} <= 2 || "
        f"(avg_scifi{plane}_x > {xmin} && avg_scifi{plane}_x < {xmax} && "
        f"avg_scifi{plane}_y > {ymin} && avg_scifi{plane}_y < {ymax}))"
    )


scifi_cuts = [
    scifi_plane_cut(1, -44, -10, 18, 52),
    scifi_plane_cut(2, -44, -10, 18, 52),
    scifi_plane_cut(3, -42, -12, 20, 50),
    scifi_plane_cut(4, -42, -12, 20, 50),
    scifi_plane_cut(5, -42, -12, 20, 50),
]

cut_15_expr = "density_sndsw_scifi > 1000 && " + " && ".join(scifi_cuts)

cut_info = {
    2: ("StableBeams",           "cutFlowSummary_StableBeams == 1"),
    3: ("IP1BunchCrossing",      "cutFlowSummary_IP1 == 1"),
    4: ("PreEvtClockCycle100",   "cutFlowSummary_EventDeltat_1_100 == 1"),
    5: ("avgScifiFiducial",      "cutFlowSummary_AvgSFChan == 1"),
    6: ("USBarsVeto_Top",        "cutFlowSummary_USBarsVeto_0_2_000000_1_2_000000 == 1"),
    7: ("USBarsVeto_Bottom",     "cutFlowSummary_USBarsVeto_0_8_000000_1_8_000000 == 1"),
    8: ("noVetoHit",             "cutFlowSummary_NoVetoHits == 1"),
    9: ("consecutiveSciFiHits",  "cutFlowSummary_At_least_two_consecutive_SciFi_planes == 1"),
    10: ("SciFiContinuity",      "cutFlowSummary_SciFiContinuity == 1"),
    11: ("USPlaneHit_0_1",       "cutFlowSummary_USPlanesHit == 1"),
    12: ("SciFiHit35",           "cutFlowSummary_SciFiMinHits == 1"),
    13: ("USQDC_700MC_600Data",  "cutFlowSummary_USQDC == 1"),
    14: ("NoHitLastDS",          "cutFlowSummary_DSVetoCut == 1"),
}
cut_info[15] = ("separate_scifi_plane_fiducial", cut_15_expr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ROOT histograms for cabinetry fits.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/experiment/sndlhc/users/zhibin/nueAnalysis",
        help="Directory containing hist_{partition}.root files",
    )
    parser.add_argument(
        "--feature",
        default="density_sndsw_scifi",
        choices=hist_info.keys(),
        help="Branch/expression to plot",
    )
    parser.add_argument(
        "--extra_cut",
        default="",
        help='Selection cut, e.g. "density_scifi>0.1 && count_us>2"',
    )
    parser.add_argument(
        "--base_cut",
        default="2 3 4 5 8 9 10 15",
        help='Selection base cut, e.g. "1 2 3 11"',
    )
    parser.add_argument(
        "--tree",
        default="sndData",
        help="TTree name",
    )
    parser.add_argument(
        "--outdir",
        default="processed_hists",
        help="Output directory",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Plot title",
    )
    parser.add_argument(
        "--fold-overflow",
        action="store_true",
        default=False,
        help="Fold overflow into last bin",
    )
    parser.add_argument(
        "--fold-underflow",
        action="store_true",
        default=False,
        help="Fold underflow into first bin",
    )
    parser.add_argument(
        "--had-scale-factor",
        type=float,
        default=1,
        help="Neutral hadron scale factor applied after lumi normalization",
    )
    parser.add_argument(
        "--combine-hadrons",
        action="store_true",
        default=True,
        help="Combine kaon and neutron bins into single templates for cabinetry",
    )
    parser.add_argument(
        "--region-name",
        default="CR",
        help="Region name for cabinetry (e.g. SR, CR1, CR2)",
    )

    args = parser.parse_args()
    main(args)