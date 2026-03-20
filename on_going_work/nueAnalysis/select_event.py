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

    return None


def sanitize_cut(cut):
    if not cut.strip():
        return "nocut"

    cut = cut.replace(" ", "")
    cut = cut.replace("&&", "_")
    cut = cut.replace("||", "_")

    replacements = {
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
        cut = cut.replace(old, new)

    while "__" in cut:
        cut = cut.replace("__", "_")

    return cut.strip("_")


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
def make_histograms(chains, feature, hist_cfg, cut="", fold_underflow=False, fold_overflow=False):
    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins =int((x_max - x_min)/bin_width)
    grouped_hists = {}
    selection = cut.strip()

    for category, chain in chains.items():
        if not (category == "real_data"):
            continue
        safe_cat = re.sub(r"[^A-Za-z0-9_]", "_", category)
        hname = f"h_{safe_cat}"

        draw_expr = f"{feature}>>{hname}({n_bins},{x_min},{x_max})"
        selected = chain.Draw(draw_expr, selection, "goff")
        
        print(category)
        expr = "runId:eventId:eventIndex:density_scifi"
        selected = chain.Draw(expr, cut, "goff")

        if selected < 0:
            raise RuntimeError("Draw failed")

        v1 = chain.GetV1()  # runId
        v2 = chain.GetV2()  # eventId
        v3 = chain.GetV3()  # eventIndex
        v4 = chain.GetV4()  # density_scifi

        for i in range(selected):
            runId = int(v1[i])
            eventId = int(v2[i])
            partition = int(eventId//1e6)
            eventIndex = int(v3[i])
            density_scifi  = v4[i]
            
            print(
                f"runId={int(runId):8d}, "
                f"eventId={int(eventId):10d}, "
                f"partition={int(partition):3d}, "
                f"eventIndex={int(eventIndex):10d}, "
                f"density_scifi={density_scifi:10.4f}"
            )


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
        # print_hist_summary(f"scaled {category}", hist)

    return data_lumi


def combine_categories(grouped_hists, categories, output_name, NutralKaonScaleFactor=1):
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
        combined.Scale(NutralKaonScaleFactor)
    return combined


def combine_backgrounds(grouped_hists):
    final_hists = {}

    data_hist = grouped_hists.get("real_data")
    if data_hist is None:
        raise RuntimeError("Missing real_data histogram")

    final_hists["data"] = data_hist.Clone("h_data")
    final_hists["data"].SetDirectory(0)

    kaon_hist = combine_categories(grouped_hists, KAON_BINS, "h_kaon", NutralKaonScaleFactor=8.21)
    neutron_hist = combine_categories(grouped_hists, NEUTRON_BINS, "h_neutron",NutralKaonScaleFactor=8.21)

    if kaon_hist is not None:
        final_hists["kaon"] = kaon_hist
        # print_hist_summary("final kaon", kaon_hist)
    if neutron_hist is not None:
        final_hists["neutron"] = neutron_hist
        # print_hist_summary("final neutron", neutron_hist)

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

    return final_hists


def style_final_hists(final_hists):
    style_data_hist(final_hists.get("data"))
    style_hist(final_hists.get("kaon"), ROOT.kBlue + 2, fill=True)
    style_hist(final_hists.get("neutron"), ROOT.kBlue - 2, fill=True)
    style_hist(final_hists.get("MC_CC_numu"), ROOT.kRed + 1, fill=True)
    style_hist(final_hists.get("MC_CC_nue"), ROOT.kYellow, fill=True)
    style_hist(final_hists.get("MC_NC_numu"), ROOT.kGreen + 3, fill=True)
    style_hist(final_hists.get("MC_NC_nue"), ROOT.kGreen - 5, fill=True)


def build_stack_draw_hists(final_hists):
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

def draw_plot(final_hists, feature, hist_cfg, outdir, cut="", title=""):
    ROOT.gStyle.SetOptStat(0)

    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins =int((x_max - x_min) / bin_width)

    cut_tag = sanitize_cut(cut)
    output_file = os.path.join(outdir, f"{feature}__{cut_tag}__binWidth{(bin_width)}__range{x_min}-{x_max}__logy{logy}.pdf")

    # --------------------------------------------------
    # Build safe draw copies and keep them alive
    # --------------------------------------------------
    raw_stack_hists = build_stack_draw_hists(final_hists)

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
    pad_top.SetRightMargin(0.05)
    pad_top.SetTopMargin(0.08)
    pad_top.SetBottomMargin(0.02)

    pad_bottom.SetLeftMargin(0.12)
    pad_bottom.SetRightMargin(0.05)
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
        legend.AddEntry(data_hist, "Data", "lep")

    label_map = {
        "kaon": "MC kaon",
        "neutron": "MC neutron",
        "MC_NC_nue": "MC NC #nu_{e}",
        "MC_NC_numu": "MC NC #nu_{#mu}",
        "MC_CC_nue": "MC CC #nu_{e}",
        "MC_CC_numu": "MC CC #nu_{#mu}",
    }

    for cat, hist in stack_draw_hists:
        legend.AddEntry(hist, label_map.get(cat, cat), "f")

    legend.Draw()
    pad_top.RedrawAxis()

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
    
    ratio_frame.SetMinimum(0)
    ratio_frame.SetMaximum(15)
    

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
    hist_cfg = hist_info[args.feature]

    category_files, grouped_lumi, open_files = collect_files_and_lumi(args.input_dir, args.tree)
    chains = build_chains(category_files, args.tree)
    grouped_hists = make_histograms(
        chains=chains,
        feature=args.feature,
        hist_cfg=hist_cfg,
        cut=args.cut,
        fold_underflow=args.fold_underflow,
        fold_overflow=args.fold_overflow,
    )


hist_info = {
    "density_scifi": (1000, 1000, 1e5, "Sum of SciFi Density Weight", True),
    "count_scifi":   (10, 0, 800, "SciFi Hit Total Count", True),
    "count_us":      (1, 0, 52, "US Hit Count", True),
    "count_us1":      (1, 0, 12, "US1 Hit Count", True),
    "qdc_scifi":     (500, 0, 1.5e4, "Sum of SciFi QDC", False),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacked MC vs data plot from PyROOT files.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/user/z/zhibin/nueAnalysis",
        help="Directory containing hist_{partition}.root files",
    )
    parser.add_argument(
        "--feature",
        default="density_scifi",
        choices=hist_info.keys(),
        help="Branch/expression to plot",
    )
    parser.add_argument(
        "--cut",
        default="density_scifi > 50000 && consecutiveSciFiHits == 1 && SciFiContinuity==1 && SciFiHit35==1 && NoHitLastDS == 1", #count_scifi>200
        #density_scifi > 10000 && consecutiveSciFiHits == 1 && SciFiContinuity==1 && SciFiHit35==1 && NoHitLastDS == 1
        help='Selection cut, e.g. "density_scifi>0.1 && count_us>2"',
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

    args = parser.parse_args()
    main(args)