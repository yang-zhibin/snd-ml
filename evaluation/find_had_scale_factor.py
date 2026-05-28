#!/usr/bin/env python3

import os
import re
import glob
import math
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

BACKGROUND_MODELS = ["neutral_hadrons", "muonDIS"]
SCALE_FACTOR_TARGETS = ["fixed", "data"]


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

def get_total_lumi(root_file):
    obj = root_file.Get("total_lumi")
    if not obj:
        raise RuntimeError(f"Could not find 'total_lumi' in {root_file.GetName()}")

    lumi = float(obj.GetVal())
    return lumi


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

    if re.match(r"^MC_muonDIS_Max\d+-\d+$", partition):
        return "muonDIS"

    if re.match(r"^MC_muon(?:_Max\d+-\d+)?$", partition):
        return "muon"

    return None

def sanitize_expr(expr):
    """Sanitize a cut expression into a filename-safe tag."""
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
    """
    Build a filename-safe cut label like:
    base-cut-1-2-3__extra-cut-count_scifi_gt_200
    """
    parts = []

    if base_cut and base_cut.strip():
        # allow commas or spaces
        base_tokens = base_cut.replace(",", " ").split()
        base_part = "base-cut-" + "-".join(base_tokens)
        parts.append(base_part)

    if extra_cut and extra_cut.strip():
        extra_part = "extra-cut-" + sanitize_expr(extra_cut)
        parts.append(extra_part)

    if not parts:
        return "nocut"

    return "__".join(parts)

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
    """
    Build selection string for one category.
    For non-real_data categories, drop cuts 2, 3, 4.
    """
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

def make_histograms(chains, feature, hist_cfg, base_cut="", extra_cut="" ,fold_underflow=False, fold_overflow=False):
    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins =int((x_max - x_min)/bin_width)
    grouped_hists = {}


    for category, chain in chains.items():
        safe_cat = re.sub(r"[^A-Za-z0-9_]", "_", category)
        hname = f"h_{safe_cat}"
        
        selection = get_category_selection(category, base_cut, extra_cut, cut_info)
        
        #print(category, selection)
        
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


def get_hist_config(feature, extra_cut_key=""):
    hist_cfg = hist_info[feature]
    override = hist_range_overrides.get(feature, {}).get(extra_cut_key)

    if override is None:
        return hist_cfg

    bin_width, x_min, x_max = override
    _, _, _, axis_title, logy = hist_cfg
    print(
        f"[hist config] override for feature={feature!r}, extra_cut_key={extra_cut_key!r}: "
        f"bin_width={bin_width}, range=[{x_min}, {x_max}]"
    )
    return (bin_width, x_min, x_max, axis_title, logy)


def scale_mc_to_data(grouped_hists, grouped_lumi, data_category="real_data", keep_unscaled_on_invalid_lumi=None):
    keep_unscaled_on_invalid_lumi = set(keep_unscaled_on_invalid_lumi or [])
    data_lumi = grouped_lumi[data_category]

    if not math.isfinite(data_lumi) or data_lumi <= 0:
        raise RuntimeError(f"Invalid data lumi: {data_lumi}")

    print(f"\nTotal data lumi = {data_lumi:.6g}")

    for category, hist in grouped_hists.items():
        if category == data_category:
            continue

        mc_lumi = grouped_lumi.get(category, float("nan"))

        if not math.isfinite(mc_lumi) or mc_lumi <= 0:
            if category in keep_unscaled_on_invalid_lumi:
                print(f"[scale] {category:25s} : kept unscaled (invalid lumi = {mc_lumi})")
                continue
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
        # print_hist_summary(f"scaled {category}", hist)

    return data_lumi



def combine_categories(grouped_hists, categories, output_name, NutralHadScaleFactor=1):
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
        combined.Scale(NutralHadScaleFactor)
    return combined


def calculate_background_scale_factor(background_integral, data_integral, scale_factor_target, fixed_target=8.21):
    if background_integral <= 0:
        raise RuntimeError(
            "Cannot calculate background scale factor: background integral is zero after selections. "
            "Check whether the selected background has entries after base/extra cuts, whether the plotted "
            "feature range catches those entries, and whether luminosity scaling zeroed the histogram."
        )

    if scale_factor_target == "fixed":
        return fixed_target / background_integral, fixed_target, f"fixed {fixed_target}"

    if scale_factor_target == "data":
        return data_integral / background_integral, data_integral, "data integral"

    raise ValueError(f"Unknown scale-factor target: {scale_factor_target}")


def combine_backgrounds(args, grouped_hists):
    final_hists = {}

    data_hist = grouped_hists.get("real_data")
    if data_hist is None:
        raise RuntimeError("Missing real_data histogram")

    final_hists["data"] = data_hist.Clone("h_data")
    final_hists["data"].SetDirectory(0)
    data_integral = final_hists["data"].Integral()

    background_scale_factor = None
    scale_target_value = None
    scale_target_label = ""

    if args.background_model == "neutral_hadrons":
        kaon_hist = combine_categories(grouped_hists, KAON_BINS, "h_kaon", NutralHadScaleFactor=args.had_scale_factor)
        neutron_hist = combine_categories(grouped_hists, NEUTRON_BINS, "h_neutron", NutralHadScaleFactor=args.had_scale_factor)

        kaon_integral = kaon_hist.Integral() if kaon_hist is not None else 0.0
        neutron_integral = neutron_hist.Integral() if neutron_hist is not None else 0.0
        NutralHadron_integral = kaon_integral + neutron_integral

        background_scale_factor, scale_target_value, scale_target_label = calculate_background_scale_factor(
            background_integral=NutralHadron_integral,
            data_integral=data_integral,
            scale_factor_target=args.scale_factor_target,
        )

        print("=== Neutral Hadron Scaling Info ===")
        print(f"Applied neutral hadron scale factor                          = {args.had_scale_factor}")
        print(f"Scale factor target mode                                     = {args.scale_factor_target}")
        print(f"Scale factor target ({scale_target_label})                   = {scale_target_value}")
        print(f"Data total                                                   = {data_integral}")
        print(f"MC Kaon total                                                = {kaon_integral}")
        print(f"MC Neutron total                                             = {neutron_integral}")
        print(f"MC Neutral total (MC Kaon + MC Neutron)                      = {NutralHadron_integral}")
        print(f"background_scale_factor = target / MC Neutral total          = {background_scale_factor}")

        if args.normalise:
            if kaon_hist is not None:
                kaon_hist.Scale(background_scale_factor)
            if neutron_hist is not None:
                neutron_hist.Scale(background_scale_factor)

        if kaon_hist is not None:
            final_hists["kaon"] = kaon_hist
            # print_hist_summary("final kaon", kaon_hist)
        if neutron_hist is not None:
            final_hists["neutron"] = neutron_hist
            # print_hist_summary("final neutron", neutron_hist)

    elif args.background_model == "muonDIS":
        muondis_hist = grouped_hists.get("muonDIS")
        if muondis_hist is None:
            raise RuntimeError(
                "No muonDIS histograms found. Expected files like "
                "hist_MC_muonDIS_Max10-1.root in the input directory."
            )

        muondis_hist = muondis_hist.Clone("h_muonDIS")
        muondis_hist.SetDirectory(0)
        sanitize_hist_bins(muondis_hist)

        print_hist_summary("muonDIS before calculated scaling", muondis_hist)
        muondis_integral = muondis_hist.Integral()
        background_scale_factor, scale_target_value, scale_target_label = calculate_background_scale_factor(
            background_integral=muondis_integral,
            data_integral=data_integral,
            scale_factor_target=args.scale_factor_target,
        )

        if args.normalise:
            muondis_hist.Scale(background_scale_factor)

        final_hists["muonDIS"] = muondis_hist

        print("=== Muon DIS Scaling Info ===")
        print(f"Scale factor target mode                                     = {args.scale_factor_target}")
        print(f"Scale factor target ({scale_target_label})                   = {scale_target_value}")
        print(f"Data total                                                   = {data_integral}")
        print(f"MC muonDIS total before calculated scaling                   = {muondis_integral}")
        print(f"background_scale_factor = target / MC muonDIS total          = {background_scale_factor}")
        if args.normalise:
            print(f"MC muonDIS total after calculated scaling                    = {muondis_hist.Integral()}")
    else:
        raise ValueError(f"Unknown background model: {args.background_model}")

    # print_hist_summary("final data", final_hists["data"])
    for cat in NEUTRINO_CATS:
        hist = grouped_hists.get(cat)
        if hist is None:
            continue

        cloned = hist.Clone(f"h_{cat}")
        cloned.SetDirectory(0)
        sanitize_hist_bins(cloned)
        final_hists[cat] = cloned
        # print_hist_summary(f"final {cat}", cloned)

    return final_hists, background_scale_factor, scale_target_label


def style_final_hists(final_hists):
    style_data_hist(final_hists.get("data"))
    style_hist(final_hists.get("kaon"), ROOT.kBlue + 2, fill=True)
    style_hist(final_hists.get("neutron"), ROOT.kBlue - 2, fill=True)
    style_hist(final_hists.get("muonDIS"), ROOT.kMagenta + 2, fill=True)
    style_hist(final_hists.get("MC_CC_numu"), ROOT.kRed + 1, fill=True)
    style_hist(final_hists.get("MC_CC_nue"), ROOT.kYellow, fill=True)
    style_hist(final_hists.get("MC_NC_numu"), ROOT.kGreen + 3, fill=True)
    style_hist(final_hists.get("MC_NC_nue"), ROOT.kGreen - 5, fill=True)


def build_stack_draw_hists(final_hists, background_model):
    if background_model == "muonDIS":
        stack_order = ["muonDIS", "MC_NC_nue", "MC_NC_numu", "MC_CC_nue", "MC_CC_numu"]
    else:
        stack_order = ["kaon", "neutron", "MC_NC_nue", "MC_NC_numu", "MC_CC_nue", "MC_CC_numu"]
    draw_hists = []

    for cat in stack_order:
        hist = final_hists.get(cat)
        if hist is None:
            continue

        hdraw = make_draw_clone(hist, f"{cat}_draw")
        if hdraw is None:
            continue

        if hdraw.Integral() == 0 and hdraw.GetEntries() == 0:
            continue

        draw_hists.append((cat, hdraw))

    return draw_hists


def build_mc_sum(stack_draw_hists):
    mc_sum = None

    for _, hist in stack_draw_hists:
        if hist is None:
            continue

        if mc_sum is None:
            mc_sum = hist.Clone("mc_sum")
            mc_sum.SetDirectory(0)
        else:
            mc_sum.Add(hist)

    if mc_sum is not None:
        sanitize_hist_bins(mc_sum)

    return mc_sum

def get_ratio_range(ratio_hist, default_min=0.5, default_max=1.5, padding=0.15):
    vals = []

    for ibin in range(1, ratio_hist.GetNbinsX() + 1):
        y = ratio_hist.GetBinContent(ibin)
        ey = ratio_hist.GetBinError(ibin)

        if y <= 0:
            continue

        vals.append(y - ey)
        vals.append(y + ey)

    if not vals:
        return default_min, default_max

    # avoid absurdly large outlier
    vals = sorted(vals)
    if len(vals) > 1:
        vals = vals[0:-1]   # drop highest
    
    ymin = min(vals)
    ymax = max(vals)

    # avoid absurdly tiny range
    if ymax - ymin < 0.2:
        center = 0.5 * (ymin + ymax)
        ymin = center - 0.1
        ymax = center + 0.1

    # add fractional padding
    span = ymax - ymin
    ymin -= padding * span
    ymax += padding * span

    # optional: keep 1 inside the frame
    ymin = min(ymin, 1.0)
    ymax = max(ymax, 1.0)

    return ymin, ymax

def draw_plot(data_lumi, final_hists, feature, hist_cfg, outdir, background_model, background_scale_factor, scale_factor_target, scale_target_label="", base_cut="", extra_cut="", title=""):
    ROOT.gStyle.SetOptStat(0)

    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins =int((x_max - x_min) / bin_width)

    cut_tag = sanitize_cut(base_cut, extra_cut)
    scale_tag = "none" if background_scale_factor is None else f"{background_scale_factor:.6g}"
    output_file = os.path.join(outdir, f"{background_model}__target-{scale_factor_target}__scale-factor-{scale_tag}__{feature}__{cut_tag}__binWidth{(bin_width)}__range{x_min}-{x_max}__logy{logy}.pdf")

    # --------------------------------------------------
    # Build safe draw copies and keep them alive
    # --------------------------------------------------
    raw_stack_hists = build_stack_draw_hists(final_hists, background_model)

    stack_draw_hists = []
    for i, (cat, hist) in enumerate(raw_stack_hists):
        if hist is None:
            continue
        h = hist.Clone(f"{feature}_{cut_tag}_{cat}_stack_{i}")
        h.SetDirectory(0)
        stack_draw_hists.append((cat, h))

    data_hist = final_hists.get("data")
    if data_hist is not None:
        data_hist = data_hist.Clone(f"{feature}_{cut_tag}_data")
        data_hist.SetDirectory(0)

    # Safe MC sum
    mc_sum = None
    for _, hist in stack_draw_hists:
        if hist is None:
            continue
        if mc_sum is None:
            mc_sum = hist.Clone(f"{feature}_{cut_tag}_mc_sum")
            mc_sum.SetDirectory(0)
        else:
            mc_sum.Add(hist)

    # --------------------------------------------------
    # Stack
    # --------------------------------------------------
    stack = ROOT.THStack(f"mc_stack_{feature}_{cut_tag}", "")
    for _, hist in stack_draw_hists:
        stack.Add(hist)

    max_data = data_hist.GetMaximum() if data_hist is not None else 0.0
    max_mc = mc_sum.GetMaximum() if mc_sum is not None else 0.0
    ymax_base = max(max_data, max_mc, 1.0)
    ymax = 20.0 * ymax_base if logy else 1.4 * ymax_base

    # --------------------------------------------------
    # Canvas and pads
    # --------------------------------------------------
    canvas = ROOT.TCanvas(f"c_{feature}_{cut_tag}", "", 900, 800)

    pad_top = ROOT.TPad(f"pad_top_{feature}_{cut_tag}", "", 0.0, 0.35, 1.0, 1.0)
    pad_bottom = ROOT.TPad(f"pad_bottom_{feature}_{cut_tag}", "", 0.0, 0.0, 1.0, 0.35)

    pad_top.SetLeftMargin(0.12)
    pad_top.SetRightMargin(0.07)
    pad_top.SetTopMargin(0.08)
    pad_top.SetBottomMargin(0.02)

    pad_bottom.SetLeftMargin(0.12)
    pad_bottom.SetRightMargin(0.07)
    pad_bottom.SetTopMargin(0.03)
    pad_bottom.SetBottomMargin(0.40)

    if logy:
        pad_top.SetLogy()

    pad_top.Draw()
    pad_bottom.Draw()

    # --------------------------------------------------
    # Top pad
    # --------------------------------------------------
    pad_top.cd()

    frame = ROOT.TH1D(f"frame_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
    frame.SetDirectory(0)
    frame.SetTitle(title if title else "")
    frame.GetXaxis().SetTitle("")
    frame.GetXaxis().SetLabelSize(0)
    frame.GetYaxis().SetTitle(f"Events / {bin_width:.3g}")
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetTitleOffset(1.1)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.SetMinimum(0.1 if logy else 0.0)
    frame.SetMaximum(ymax)
    frame.Draw()

    if len(stack_draw_hists) > 0:
        stack.Draw("HIST SAME")

    if data_hist is not None:
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetLineWidth(2)
        data_hist.Draw("E1 SAME")

    legend = ROOT.TLegend(0.62, 0.58, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    if data_hist is not None:
        data_integral = data_hist.Integral()
        legend.AddEntry(data_hist, f"Data ({data_integral:.1f})", "lep")    

    label_map = {
        "kaon": "MC kaon",
        "neutron": "MC neutron",
        "muonDIS": "MC muonDIS",
        "MC_NC_nue": "MC NC #nu_{e}",
        "MC_NC_numu": "MC NC #nu_{#mu}",
        "MC_CC_nue": "MC CC #nu_{e}",
        "MC_CC_numu": "MC CC #nu_{#mu}",
    }

    for cat, hist in stack_draw_hists:
        integral = hist.Integral()
        label = f"{label_map.get(cat, cat)} ({integral:.1f})"
        legend.AddEntry(hist, label, "f")

    legend.Draw()
        
    y0 = 0.88      # starting height (top)
    dy = 0.045     # vertical spacing

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextAlign(13)   # left-align
    text.SetTextSize(0.032)

    # 1. Cuts
    base_cut_text = base_cut.strip() if base_cut and base_cut.strip() else "none"
    extra_cut_text = extra_cut.strip() if extra_cut and extra_cut.strip() else "none"
    text.DrawLatex(0.15, y0, f"#bf{{Base cut}}: {base_cut_text}")
    text.DrawLatex(0.15, y0 - dy, f"#bf{{Extra cut}}: {extra_cut_text}")

    # 2. Background scale factor
    if background_model == "neutral_hadrons":
        scale_text = (
            f"Neutral scale factor ({scale_target_label}) = {background_scale_factor:.3g}"
            if background_scale_factor is not None else
            "Neutral scale factor = n/a"
        )
    else:
        scale_text = (
            f"Muon DIS scale factor ({scale_target_label}) = {background_scale_factor:.3g}"
            if background_scale_factor is not None else
            "Muon DIS scale factor = n/a"
        )
    text.DrawLatex(0.15, y0 - 2*dy, scale_text)

    # 3. Luminosity
    if data_lumi is not None:
        text.DrawLatex(0.15, y0 - 3*dy, f"#int #font[12]{{L}} dt = {data_lumi:.3f} fb^{{-1}}")

    # --------------------------------------------------
    # Bottom pad
    # --------------------------------------------------
    pad_bottom.cd()

    ratio_frame = ROOT.TH1D(f"ratio_frame_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
    ratio_frame.SetDirectory(0)
    ratio_frame.SetTitle("")
    ratio_frame.GetXaxis().SetTitle(axis_title)
    ratio_frame.GetYaxis().SetTitle("Data/MC")

    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetTitleOffset(1.2)
    ratio_frame.GetXaxis().SetLabelSize(0.10)

    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.5)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetYaxis().CenterTitle()
    ratio_frame.Draw()

    ratio = None
    if data_hist is not None and mc_sum is not None:
        ratio = data_hist.Clone(f"ratio_{feature}_{cut_tag}")
        ratio.SetDirectory(0)

        for ibin in range(1, ratio.GetNbinsX() + 1):
            mc_val = mc_sum.GetBinContent(ibin)
            data_val = data_hist.GetBinContent(ibin)
            data_err = data_hist.GetBinError(ibin)

            if mc_val > 0:
                ratio.SetBinContent(ibin, data_val / mc_val)
                ratio.SetBinError(ibin, data_err / mc_val)
            else:
                ratio.SetBinContent(ibin, 0.0)
                ratio.SetBinError(ibin, 0.0)

        ratio.SetMarkerStyle(20)
        ratio.SetMarkerSize(0.9)
        ratio.SetLineWidth(2)
        ratio.Draw("E1 SAME")
        
        ratio_ymin, ratio_ymax = get_ratio_range(ratio)
    else:
        ratio_ymin, ratio_ymax = 0.5, 1.5
    
    ratio_frame.SetMinimum(ratio_ymin)
    ratio_frame.SetMaximum(ratio_ymax)
    

    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    pad_bottom.RedrawAxis()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    canvas.cd()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(output_file)

    print(f"\nSaved plot to {output_file}")
    
def main(args):
    if args.feature not in hist_info:
        raise ValueError(f"{args.feature} not defined in hist_info")

    os.makedirs(args.outdir, exist_ok=True)
    hist_cfg = get_hist_config(args.feature, args.extra_cut_key)

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
    keep_unscaled_on_invalid_lumi = []
    if args.background_model == "muonDIS":
        keep_unscaled_on_invalid_lumi.append("muonDIS")

    scale_mc_to_data(
        grouped_hists,
        grouped_lumi,
        data_category="real_data",
        keep_unscaled_on_invalid_lumi=keep_unscaled_on_invalid_lumi,
    )
    final_hists, background_scale_factor, scale_target_label = combine_backgrounds(args, grouped_hists)
    style_final_hists(final_hists)
    
    # for key in ["MC_NC_nue", "MC_NC_numu", "MC_CC_nue", "MC_CC_numu"]:
    #     final_hists.pop(key, None)
    
    
    draw_plot(
        data_lumi = grouped_lumi["real_data"],
        final_hists=final_hists,
        feature=args.feature,
        hist_cfg=hist_cfg,
        outdir=args.outdir,
        background_model=args.background_model,
        scale_factor_target=args.scale_factor_target,
        scale_target_label=scale_target_label,
        base_cut=args.base_cut, 
        extra_cut=args.extra_cut,
        title=args.title,
        background_scale_factor=background_scale_factor
    )

    # keep ROOT files alive until the very end
    _ = open_files


hist_info = {
    "density_scifi": (5000, 1000, 1e5, "Sum of SciFi Density Weight", True),
    "density_scifi1": (1000, 0, 0.4e5, "Plane1 Sum of SciFi Density Weight", True),
    "density_scifi2": (1000, 0, 0.4e5, "Plane2 Sum of SciFi Density Weight", True),
    "density_scifi3": (1000, 0, 0.4e5, "Plane3 Sum of SciFi Density Weight", True),
    "density_scifi4": (1000, 0, 0.4e5, "Plane4 Sum of SciFi Density Weight", True),
    "density_scifi5": (1000, 0, 0.4e5, "Plane5 Sum of SciFi Density Weight", True),
    
    # "density_sndsw_scifi": (2500, 2000, 0.4e5, "Sum of SciFi Density Weight (SNDSW)", False),
    "density_sndsw_scifi": (2000, 100, 40000, "Sum of SciFi Density Weight (SNDSW)", True),
    "count_scifi":   (10, 0, 800, "SciFi Hit Total Count", True),
    "count_scifi1":   (10, 0, 500, "Plane1 SciFi Hit Total Count", True),
    "count_scifi2":   (10, 0, 500, "Plane2 SciFi Hit Total Count", True),
    "count_scifi3":   (10, 0, 500, "Plane3 SciFi Hit Total Count", True),
    "count_scifi4":   (10, 0, 500, "Plane4 SciFi Hit Total Count", True),
    "count_scifi5":   (10, 0, 500, "Plane5 SciFi Hit Total Count", True),
    
    "avg_scifi_x":    (1, -60, 0, "Average SciFi X Position (Vertical)", True),
    "avg_scifi1_x":   (1, -60, 0, "Plane1 Average SciFi X Position (Vertical)", True),
    "avg_scifi2_x":   (1, -60, 0, "Plane2 Average SciFi X Position (Vertical)", True),
    "avg_scifi3_x":   (1, -60, 0, "Plane3 Average SciFi X Position (Vertical)", True),
    "avg_scifi4_x":   (1, -60, 0, "Plane4 Average SciFi X Position (Vertical)", True),
    "avg_scifi5_x":   (1, -60, 0, "Plane5 Average SciFi X Position (Vertical)", True),
    
    "avg_scifi_y":    (1, 0, 60, "Average SciFi Y Position (Horizontal)", True),
    "avg_scifi1_y":   (1, 0, 60, "Plane1 Average SciFi Y Position (Horizontal)", True),
    "avg_scifi2_y":   (1, 0, 60, "Plane2 Average SciFi Y Position (Horizontal)", True),
    "avg_scifi3_y":   (1, 0, 60, "Plane3 Average SciFi Y Position (Horizontal)", True),
    "avg_scifi4_y":   (1, 0, 60, "Plane4 Average SciFi Y Position (Horizontal)", True),
    "avg_scifi5_y":   (1, 0, 60, "Plane5 Average SciFi Y Position (Horizontal)", True),

    "count_us":      (1, 0, 52, "US Hit Count", True),
    "count_us1":      (1, 0, 12, "US1 Hit Count", True),
    "count_us2":      (1, 0, 12, "US2 Hit Count", True),
    "count_us3":      (1, 0, 12, "US3 Hit Count", True),
    "count_us4":      (1, 0, 12, "US4 Hit Count", True),
    "count_us5":      (1, 0, 12, "US5 Hit Count", True),
    
    "count_ds":      (1, 0, 40, "DS Hit Count", True),
    "count_ds1":      (1, 0, 40, "DS1 Hit Count", True),
    "count_ds2":      (1, 0, 40, "DS2 Hit Count", True),
    "count_ds3":      (1, 0, 40, "DS3 Hit Count", True),
    "count_ds4":      (1, 0, 40, "DS4 Hit Count", True),
    
    
    "qdc_scifi":     (500, 0, 1.5e4, "Sum of SciFi QDC", True),
    "qdc_scifi1":     (500, 0, 1e4, "Plane1 Sum of SciFi QDC", True),
    "qdc_scifi2":     (500, 0, 2e4, "Plane2 Sum of SciFi QDC", True),
    "qdc_scifi3":     (500, 0, 3e4, "Plane3 Sum of SciFi QDC", True),
    "qdc_scifi4":     (500, 0, 4e4, "Plane4 Sum of SciFi QDC", True),
    "qdc_scifi5":     (500, 0, 5e4, "Plane5 Sum of SciFi QDC", True),
    
    
    
    "qdc_us":     (1000, 0, 4e4, "Sum of US QDC", True),
    "qdc_us1":     (1000, 0, 2e4, "US1 QDC", True),
    "qdc_us2":     (1000, 0, 2e4, "US2 QDC", True),
    "qdc_us3":     (1000, 0, 2e4, "US3 QDC", True),
    "qdc_us4":     (1000, 0, 2e4, "US4 QDC", True),
    "qdc_us5":     (1000, 0, 2e4, "US5 QDC", True),
}

hist_range_overrides = {
    "density_sndsw_scifi": {
        "density_sndsw_scifi_gt_1000__density_sndsw_scifi_lt_6000": (250, 1000, 6000),
        "density_sndsw_scifi_gt_2000__density_sndsw_scifi_lt_5000": (100, 2000, 5000),
        "density_sndsw_scifi_gt_1000": (500, 1000, 40000),
        "density_sndsw_scifi_gt_100": (500, 100, 40000),
        "density_sndsw_scifi_gt_2000": (500, 2000, 40000),
    },
}

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacked MC vs data plot from PyROOT files.")
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
        default="", #count_scifi>200 density_scifi > 1000 density_sndsw_scifi_second > 20 
        # density_sndsw_scifi > 11000 && density_sndsw_scifi_second > 20 
        #density_scifi > 10000 && consecutiveSciFiHits == 1 && SciFiContinuity==1 && SciFiHit35==1 && NoHitLastDS == 1
        help='Selection cut, e.g. "density_scifi>0.1 && count_us>2"',
    )
    parser.add_argument(
        "--extra-cut-key",
        default="",
        help="Named extra cut key used to choose feature-specific plotting range overrides",
    )
    
    parser.add_argument(
        "--base_cut",
        default="2 3 4 5 8 9 10", 
        help='Selection base cut, e.g. 1 2 3, 11',
    )
    
    parser.add_argument(
        "--tree",
        default="sndData",
        help="TTree name",
    )
    parser.add_argument(
        "--outdir",
        default="hist_plots",
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
        "--normalise",
        action="store_true",
        default=False,
        help="Apply the calculated background scale factor to the selected background model",
    )
    
    parser.add_argument(
        "--had-scale-factor",
        type=float,
        default=1,
        help="Neutral hadron background scale factor",
    )
    parser.add_argument(
        "--background-model",
        choices=BACKGROUND_MODELS,
        default="neutral_hadrons",
        help="Background model to draw: kaon+neutron neutral hadrons or muonDIS",
    )
    parser.add_argument(
        "--scale-factor-target",
        choices=SCALE_FACTOR_TARGETS,
        default="fixed",
        help="Target used to calculate background scale factor: fixed 8.21 or selected data integral",
    )

    args = parser.parse_args()
    main(args)
