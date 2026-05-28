#!/usr/bin/env python3

import argparse
import math
import os
import re
from collections import defaultdict

import ROOT

import find_had_scale_factor as common

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


DATASET_GROUPS = {
    "data": ["real_data"],
    "muon": ["muon"],
    "muonDIS": ["muonDIS"],
    "neutral_hadrons": common.KAON_BINS + common.NEUTRON_BINS,
    "neutrinos": common.NEUTRINO_CATS,
}

GROUP_LABELS = {
    "data": "Data",
    "muon": "Muon MC",
    "muonDIS": "muonDIS MC",
    "neutral_hadrons": "Kaon+neutron MC",
    "neutrinos": "Neutrino MC",
}

GROUP_COLORS = {
    "data": ROOT.kBlack,
    "muon": ROOT.kOrange + 7,
    "muonDIS": ROOT.kMagenta + 2,
    "neutral_hadrons": ROOT.kAzure + 1,
    "neutrinos": ROOT.kGreen + 2,
}

REGIONS = {
    "pass": lambda expr: f"({expr})",
    "fail": lambda expr: f"!({expr})",
}


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")


def combine_selection(*parts):
    selected = [f"({part})" for part in parts if part and str(part).strip()]
    return " && ".join(selected) if selected else "1"


def matching_cut_indices(cut_expr):
    matches = []
    stripped = cut_expr.strip()
    for index, (_, expr) in common.cut_info.items():
        if expr.strip() == stripped:
            matches.append(str(index))
    return matches


def remove_sideband_cut_from_base(base_cut, sideband_cut):
    tokens = base_cut.replace(",", " ").split() if base_cut else []
    if not tokens:
        return base_cut

    matches = set(matching_cut_indices(sideband_cut))
    if not matches:
        return base_cut

    kept = [token for token in tokens if token not in matches]
    removed = [token for token in tokens if token in matches]
    if removed:
        print(
            "[INFO] Removed sideband splitter cut index "
            f"{' '.join(removed)} from base_cut to keep pass/fail regions complementary"
        )
    return " ".join(kept)


def get_region_selection(category, base_cut, extra_cut, sideband_cut, region):
    base_without_splitter = remove_sideband_cut_from_base(base_cut, sideband_cut)
    common_selection = common.get_category_selection(
        category,
        base_without_splitter,
        extra_cut,
        common.cut_info,
    )
    region_cut = REGIONS[region](sideband_cut)
    return combine_selection(common_selection, region_cut)


def has_valid_lumi(lumi):
    return math.isfinite(lumi) and lumi > 0


def histogram_model(feature, category, region, hist_cfg):
    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    hist_name = f"h_{safe_name(category)}_{region}_{safe_name(feature)}"
    return hist_name, n_bins, x_min, x_max


def build_category_histogram(
    category_files,
    category,
    tree_name,
    feature,
    hist_cfg,
    selection,
    region,
    fold_underflow=False,
    fold_overflow=False,
):
    files = category_files.get(category, [])
    if not files:
        print(f"[skip] no files for {category}")
        return None, 0

    chain = ROOT.TChain(tree_name)
    for filepath in files:
        chain.Add(filepath)

    hist_name, n_bins, x_min, x_max = histogram_model(feature, category, region, hist_cfg)
    draw_expr = f"{feature}>>{hist_name}({n_bins},{x_min},{x_max})"
    selected = chain.Draw(draw_expr, selection, "goff")
    print(f"[draw] {category:25s} region={region:4s} expr={draw_expr} cut={selection!r} selected={selected}")

    if selected < 0:
        raise RuntimeError(f"TChain::Draw failed for category={category}, region={region}")

    hist_tmp = chain.GetHistogram()
    if hist_tmp is None:
        raise RuntimeError(f"GetHistogram() returned null for category={category}, region={region}")

    hist = hist_tmp.Clone(f"{hist_name}_clone")
    hist.SetDirectory(0)

    if fold_underflow:
        common.add_underflow_to_first_bin(hist)
    if fold_overflow:
        common.add_overflow_to_last_bin(hist)

    common.sanitize_hist_bins(hist)
    common.print_hist_summary(f"raw {category} {region}", hist)
    return hist, selected


def scaled_category_histogram(category, hist, grouped_lumi, target_lumi):
    if hist is None:
        return None, float("nan"), float("nan")

    if category == "real_data":
        return hist, grouped_lumi.get(category, float("nan")), 1.0

    sample_lumi = grouped_lumi.get(category, float("nan"))
    if not has_valid_lumi(sample_lumi):
        print(f"[skip] invalid luminosity for {category}: {sample_lumi}")
        return None, sample_lumi, float("nan")

    scale = target_lumi / sample_lumi
    scaled = hist.Clone(f"{hist.GetName()}_scaled")
    scaled.SetDirectory(0)
    scaled.Scale(scale)
    common.sanitize_hist_bins(scaled)
    print(f"[scale] {category:25s}: lumi={sample_lumi:.6g}, target={target_lumi:.6g}, x {scale:.6g}")
    return scaled, sample_lumi, scale


def add_to_sum(total, hist, name):
    if hist is None:
        return total
    if total is None:
        total = hist.Clone(name)
        total.SetDirectory(0)
    else:
        total.Add(hist)
    common.sanitize_hist_bins(total)
    return total


def build_all_histograms(
    category_files,
    grouped_lumi,
    tree_name,
    feature,
    hist_cfg,
    base_cut,
    extra_cut,
    sideband_cut,
    target_lumi,
    fold_underflow=False,
    fold_overflow=False,
):
    category_hists = defaultdict(dict)
    group_hists = defaultdict(dict)
    yields = []
    selections = {}

    for region in REGIONS:
        for group, categories in DATASET_GROUPS.items():
            group_total = None
            for category in categories:
                selection = get_region_selection(category, base_cut, extra_cut, sideband_cut, region)
                selections[(category, region)] = selection
                raw_hist, selected = build_category_histogram(
                    category_files=category_files,
                    category=category,
                    tree_name=tree_name,
                    feature=feature,
                    hist_cfg=hist_cfg,
                    selection=selection,
                    region=region,
                    fold_underflow=fold_underflow,
                    fold_overflow=fold_overflow,
                )

                scaled_hist, lumi, scale = scaled_category_histogram(
                    category,
                    raw_hist,
                    grouped_lumi,
                    target_lumi,
                )
                category_hists[(category, region)] = scaled_hist
                group_total = add_to_sum(
                    group_total,
                    scaled_hist,
                    f"h_group_{group}_{region}_{safe_name(feature)}",
                )

                raw_integral = raw_hist.Integral() if raw_hist is not None else 0.0
                scaled_integral = scaled_hist.Integral() if scaled_hist is not None else 0.0
                yields.append(
                    {
                        "group": group,
                        "category": category,
                        "region": region,
                        "selected": selected,
                        "raw_integral": raw_integral,
                        "scaled_integral": scaled_integral,
                        "lumi": lumi,
                        "scale": scale,
                        "selection": selection,
                    }
                )

            if group_total is not None:
                group_hists[group][region] = group_total
                common.print_hist_summary(f"scaled group {group} {region}", group_total)

    return group_hists, category_hists, yields, selections


def style_hist(hist, group, region=None):
    if hist is None:
        return
    color = GROUP_COLORS.get(group, ROOT.kGray + 1)
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetLineWidth(3 if group == "muonDIS" else 2)
    hist.SetFillStyle(0)
    if group == "data":
        hist.SetMarkerStyle(20)
        hist.SetMarkerSize(0.9)
    else:
        hist.SetMarkerStyle(24 if region == "fail" else 20)
        hist.SetMarkerSize(0.8)


def ratio_hist(numerator, denominator, name):
    ratio = numerator.Clone(name)
    ratio.SetDirectory(0)
    for ibin in range(1, ratio.GetNbinsX() + 1):
        num = numerator.GetBinContent(ibin)
        den = denominator.GetBinContent(ibin)
        err = numerator.GetBinError(ibin)
        if den > 0:
            ratio.SetBinContent(ibin, num / den)
            ratio.SetBinError(ibin, err / den)
        else:
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)
    common.sanitize_hist_bins(ratio)
    return ratio


def ratio_range(ratios, default=(0.0, 2.5)):
    values = []
    for hist in ratios:
        for ibin in range(1, hist.GetNbinsX() + 1):
            y = hist.GetBinContent(ibin)
            ey = hist.GetBinError(ibin)
            if y > 0:
                values.extend([y - ey, y + ey])
    if not values:
        return default
    ymin = max(0.0, min(values))
    ymax = max(values)
    if ymax - ymin < 0.2:
        center = 0.5 * (ymin + ymax)
        ymin, ymax = center - 0.1, center + 0.1
    padding = 0.2 * (ymax - ymin)
    return max(0.0, ymin - padding), max(1.2, ymax + padding)


def draw_dataset_comparison(group_hists, region, feature, hist_cfg, outdir, base_cut, extra_cut, sideband_key, sideband_cut):
    os.makedirs(outdir, exist_ok=True)
    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    cut_tag = common.sanitize_cut(base_cut, extra_cut)
    output_file = os.path.join(
        outdir,
        f"compare_datasets__region-{region}__{feature}__{cut_tag}__sideband-{safe_name(sideband_key)}.pdf",
    )

    canvas = ROOT.TCanvas(f"c_datasets_{region}_{safe_name(feature)}", "", 900, 800)
    top = ROOT.TPad(f"top_datasets_{region}_{safe_name(feature)}", "", 0, 0.35, 1, 1)
    bottom = ROOT.TPad(f"bottom_datasets_{region}_{safe_name(feature)}", "", 0, 0, 1, 0.35)
    top.SetLeftMargin(0.12)
    top.SetRightMargin(0.07)
    top.SetTopMargin(0.08)
    top.SetBottomMargin(0.02)
    bottom.SetLeftMargin(0.12)
    bottom.SetRightMargin(0.07)
    bottom.SetTopMargin(0.03)
    bottom.SetBottomMargin(0.40)
    if logy:
        top.SetLogy()
    top.Draw()
    bottom.Draw()

    present = [(group, group_hists[group].get(region)) for group in DATASET_GROUPS if group_hists.get(group, {}).get(region)]
    if not present:
        print(f"[skip] no group histograms for region={region}")
        return

    ymax_base = max(hist.GetMaximum() for _, hist in present)
    ymax = 20.0 * max(ymax_base, 1.0) if logy else 1.45 * max(ymax_base, 1.0)

    top.cd()
    frame = ROOT.TH1D(f"frame_datasets_{region}_{safe_name(feature)}", "", n_bins, x_min, x_max)
    frame.SetDirectory(0)
    frame.GetXaxis().SetLabelSize(0)
    frame.GetYaxis().SetTitle(f"Events / {bin_width:.3g}")
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetTitleOffset(1.1)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.SetMinimum(0.1 if logy else 0.0)
    frame.SetMaximum(ymax)
    frame.Draw()

    legend = ROOT.TLegend(0.58, 0.56, 0.89, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    for group, hist in present:
        style_hist(hist, group, region)
        option = "E1 SAME" if group == "data" else "HIST SAME"
        hist.Draw(option)
        if group != "data":
            hist.Draw("E1 SAME")
        legend.AddEntry(hist, f"{GROUP_LABELS[group]} ({hist.Integral():.3g})", "lep")
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.030)
    text.SetTextAlign(13)
    text.DrawLatex(0.15, 0.88, f"#bf{{Region}}: {region}, {sideband_key}")
    text.DrawLatex(0.15, 0.835, f"#bf{{Sideband cut}}: {sideband_cut}")
    text.DrawLatex(0.15, 0.790, f"#bf{{Base cut}}: {base_cut or 'none'}")
    text.DrawLatex(0.15, 0.745, f"#bf{{Extra cut}}: {extra_cut or 'none'}")

    bottom.cd()
    ratio_frame = ROOT.TH1D(f"ratio_frame_datasets_{region}_{safe_name(feature)}", "", n_bins, x_min, x_max)
    ratio_frame.SetDirectory(0)
    ratio_frame.GetXaxis().SetTitle(axis_title)
    ratio_frame.GetYaxis().SetTitle("group/data")
    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetTitleOffset(1.2)
    ratio_frame.GetXaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.5)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetYaxis().CenterTitle()
    ratio_frame.Draw()

    ratios = []
    data_hist = group_hists.get("data", {}).get(region)
    if data_hist:
        for group, hist in present:
            if group == "data":
                continue
            ratio = ratio_hist(hist, data_hist, f"ratio_{group}_{region}_{safe_name(feature)}")
            style_hist(ratio, group, region)
            ratio.Draw("E1 SAME")
            ratios.append(ratio)

    ymin, ymax_ratio = ratio_range(ratios)
    ratio_frame.SetMinimum(ymin)
    ratio_frame.SetMaximum(ymax_ratio)
    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")
    bottom.RedrawAxis()

    canvas.SaveAs(output_file)
    print(f"[save] {output_file}")


def normalized_clone(hist, name, mode):
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    if mode == "shape":
        integral = clone.Integral()
        if integral > 0:
            clone.Scale(1.0 / integral)
    common.sanitize_hist_bins(clone)
    return clone


def draw_region_comparison(group_hists, group, feature, hist_cfg, outdir, base_cut, extra_cut, sideband_key, sideband_cut, normalization):
    pass_hist = group_hists.get(group, {}).get("pass")
    fail_hist = group_hists.get(group, {}).get("fail")
    if not pass_hist or not fail_hist:
        print(f"[skip] missing pass/fail histogram for group={group}")
        return

    os.makedirs(outdir, exist_ok=True)
    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    cut_tag = common.sanitize_cut(base_cut, extra_cut)
    output_file = os.path.join(
        outdir,
        (
            f"compare_regions__{group}__{feature}__{cut_tag}"
            f"__sideband-{safe_name(sideband_key)}__norm-{normalization}.pdf"
        ),
    )

    pass_draw = normalized_clone(pass_hist, f"{pass_hist.GetName()}_draw_{normalization}", normalization)
    fail_draw = normalized_clone(fail_hist, f"{fail_hist.GetName()}_draw_{normalization}", normalization)
    pass_draw.SetLineColor(ROOT.kBlue + 1)
    pass_draw.SetMarkerColor(ROOT.kBlue + 1)
    pass_draw.SetMarkerStyle(20)
    pass_draw.SetLineWidth(3)
    fail_draw.SetLineColor(ROOT.kRed + 1)
    fail_draw.SetMarkerColor(ROOT.kRed + 1)
    fail_draw.SetMarkerStyle(24)
    fail_draw.SetLineWidth(3)

    canvas = ROOT.TCanvas(f"c_regions_{group}_{safe_name(feature)}", "", 900, 800)
    top = ROOT.TPad(f"top_regions_{group}_{safe_name(feature)}", "", 0, 0.35, 1, 1)
    bottom = ROOT.TPad(f"bottom_regions_{group}_{safe_name(feature)}", "", 0, 0, 1, 0.35)
    top.SetLeftMargin(0.12)
    top.SetRightMargin(0.07)
    top.SetTopMargin(0.08)
    top.SetBottomMargin(0.02)
    bottom.SetLeftMargin(0.12)
    bottom.SetRightMargin(0.07)
    bottom.SetTopMargin(0.03)
    bottom.SetBottomMargin(0.40)
    if logy:
        top.SetLogy()
    top.Draw()
    bottom.Draw()

    top.cd()
    ymax_base = max(pass_draw.GetMaximum(), fail_draw.GetMaximum(), 1.0)
    ymax = 20.0 * ymax_base if logy else 1.45 * ymax_base
    frame = ROOT.TH1D(f"frame_regions_{group}_{safe_name(feature)}", "", n_bins, x_min, x_max)
    frame.SetDirectory(0)
    frame.GetXaxis().SetLabelSize(0)
    y_title = "Shape-normalized events" if normalization == "shape" else f"Events / {bin_width:.3g}"
    frame.GetYaxis().SetTitle(y_title)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetTitleOffset(1.1)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.SetMinimum(0.1 if logy else 0.0)
    frame.SetMaximum(ymax)
    frame.Draw()
    pass_draw.Draw("E1 SAME")
    fail_draw.Draw("E1 SAME")

    legend = ROOT.TLegend(0.58, 0.64, 0.89, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(pass_draw, f"pass ({pass_hist.Integral():.3g})", "lep")
    legend.AddEntry(fail_draw, f"fail ({fail_hist.Integral():.3g})", "lep")
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.030)
    text.SetTextAlign(13)
    text.DrawLatex(0.15, 0.88, f"#bf{{Group}}: {GROUP_LABELS[group]}")
    text.DrawLatex(0.15, 0.835, f"#bf{{Sideband cut}}: {sideband_cut}")
    text.DrawLatex(0.15, 0.790, f"#bf{{Normalization}}: {normalization}")

    bottom.cd()
    ratio_frame = ROOT.TH1D(f"ratio_frame_regions_{group}_{safe_name(feature)}", "", n_bins, x_min, x_max)
    ratio_frame.SetDirectory(0)
    ratio_frame.GetXaxis().SetTitle(axis_title)
    ratio_frame.GetYaxis().SetTitle("pass/fail")
    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetTitleOffset(1.2)
    ratio_frame.GetXaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.5)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetYaxis().CenterTitle()
    ratio_frame.Draw()

    ratio = ratio_hist(pass_draw, fail_draw, f"ratio_pass_fail_{group}_{safe_name(feature)}")
    ratio.SetLineColor(ROOT.kBlack)
    ratio.SetMarkerColor(ROOT.kBlack)
    ratio.SetMarkerStyle(20)
    ratio.SetLineWidth(2)
    ratio.Draw("E1 SAME")
    ymin, ymax_ratio = ratio_range([ratio])
    ratio_frame.SetMinimum(ymin)
    ratio_frame.SetMaximum(ymax_ratio)
    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")
    bottom.RedrawAxis()

    canvas.SaveAs(output_file)
    print(f"[save] {output_file}")


def write_summary(outdir, feature, base_cut, extra_cut, sideband_key, sideband_cut, yields, selections):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"sideband_summary__{feature}__sideband-{safe_name(sideband_key)}.txt")
    with open(path, "w") as fout:
        fout.write(f"feature: {feature}\n")
        fout.write(f"base_cut: {base_cut}\n")
        fout.write(f"extra_cut: {extra_cut}\n")
        fout.write(f"sideband_key: {sideband_key}\n")
        fout.write(f"sideband_cut: {sideband_cut}\n")
        fout.write("regions:\n")
        fout.write(f"  pass: ({sideband_cut})\n")
        fout.write(f"  fail: !({sideband_cut})\n\n")

        fout.write("Selections:\n")
        for (category, region), selection in sorted(selections.items()):
            fout.write(f"  {category:25s} {region:4s}: {selection}\n")

        fout.write("\nYields:\n")
        fout.write(
            f"{'group':<18} {'category':<25} {'region':<6} {'selected':>12} "
            f"{'raw_integral':>16} {'scaled_integral':>18} {'lumi':>14} {'scale':>14}\n"
        )
        fout.write("-" * 130 + "\n")
        for row in yields:
            fout.write(
                f"{row['group']:<18} {row['category']:<25} {row['region']:<6} "
                f"{row['selected']:12d} {row['raw_integral']:16.8g} "
                f"{row['scaled_integral']:18.8g} {row['lumi']:14.8g} {row['scale']:14.8g}\n"
            )
    print(f"[save] {path}")


def main(args):
    if args.feature not in common.hist_info:
        raise ValueError(f"{args.feature} not defined in hist_info")
    if not args.sideband_cut.strip():
        raise ValueError("--sideband-cut must be a non-empty selection expression")
    if args.normalization not in {"lumi", "shape"}:
        raise ValueError("--normalization must be lumi or shape")

    ROOT.gStyle.SetOptStat(0)
    os.makedirs(args.outdir, exist_ok=True)

    hist_cfg = common.get_hist_config(args.feature, args.extra_cut_key)
    category_files, grouped_lumi, open_files = common.collect_files_and_lumi(args.input_dir, args.tree)

    target_lumi = args.target_lumi if args.target_lumi is not None else grouped_lumi.get("real_data", float("nan"))
    if not has_valid_lumi(target_lumi):
        raise RuntimeError(f"Invalid target luminosity: {target_lumi}")

    print(f"[INFO] target_lumi = {target_lumi:.6g}")
    print(f"[INFO] sideband {args.sideband_key}: {args.sideband_cut}")

    group_hists, _, yields, selections = build_all_histograms(
        category_files=category_files,
        grouped_lumi=grouped_lumi,
        tree_name=args.tree,
        feature=args.feature,
        hist_cfg=hist_cfg,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        sideband_cut=args.sideband_cut,
        target_lumi=target_lumi,
        fold_underflow=args.fold_underflow,
        fold_overflow=args.fold_overflow,
    )

    for region in REGIONS:
        draw_dataset_comparison(
            group_hists=group_hists,
            region=region,
            feature=args.feature,
            hist_cfg=hist_cfg,
            outdir=args.outdir,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            sideband_key=args.sideband_key,
            sideband_cut=args.sideband_cut,
        )

    for group in DATASET_GROUPS:
        draw_region_comparison(
            group_hists=group_hists,
            group=group,
            feature=args.feature,
            hist_cfg=hist_cfg,
            outdir=args.outdir,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            sideband_key=args.sideband_key,
            sideband_cut=args.sideband_cut,
            normalization=args.normalization,
        )

    write_summary(
        outdir=args.outdir,
        feature=args.feature,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        sideband_key=args.sideband_key,
        sideband_cut=args.sideband_cut,
        yields=yields,
        selections=selections,
    )

    _ = open_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare dataset families in complementary sideband regions."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/experiment/sndlhc/users/zhibin/nueAnalysis",
        help="Directory containing hist_*.root files",
    )
    parser.add_argument("--feature", default="density_sndsw_scifi", choices=list(common.hist_info.keys()))
    parser.add_argument("--base-cut", "--base_cut", dest="base_cut", default="2 3 4 8")
    parser.add_argument("--extra-cut", "--extra_cut", dest="extra_cut", default="")
    parser.add_argument("--extra-cut-key", default="")
    parser.add_argument("--sideband-key", default="sideband")
    parser.add_argument("--sideband-cut", required=True, help='Expression defining the pass region, e.g. "count_scifi > 2"')
    parser.add_argument("--tree", default="sndData")
    parser.add_argument("--outdir", default="sideband_comparison")
    parser.add_argument("--target-lumi", type=float, default=None)
    parser.add_argument("--normalization", choices=["lumi", "shape"], default="lumi")
    parser.add_argument("--fold-underflow", action="store_true")
    parser.add_argument("--fold-overflow", action="store_true")

    main(parser.parse_args())
