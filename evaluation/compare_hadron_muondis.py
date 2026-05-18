#!/usr/bin/env python3

import argparse
import math
import os
import re
from collections import Counter

import ROOT

import find_had_scale_factor as common

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


PDG_SELECTIONS = {
    "all": [],
    "neutral_kaon": [130, 310],
    "charged_kaon": [321, -321],
    "kaon_all": [130, 310, 321, -321],
    "neutron": [2112],
    "antineutron": [-2112],
    "neutron_all": [2112, -2112],
    "gamma": [22],
    "proton": [2212],
    "pion_charged": [211, -211],
}

PDG_TABLE_FEATURES = ["secondary_pdg", "secondary_pdg_sel"]

REFERENCE_BINS = {
    "kaon": common.KAON_BINS,
    "neutron": common.NEUTRON_BINS,
}

REFERENCE_LABELS = {
    "kaon": "MC kaon",
    "neutron": "MC neutron",
    "data": "Data",
}

REFERENCE_COLORS = {
    "kaon": [
        ROOT.kAzure - 9,
        ROOT.kAzure - 4,
        ROOT.kAzure + 1,
        ROOT.kBlue + 1,
        ROOT.kBlue + 3,
        ROOT.kCyan + 1,
        ROOT.kTeal - 6,
        ROOT.kTeal + 2,
        ROOT.kGreen + 2,
        ROOT.kGreen + 4,
    ],
    "neutron": [
        ROOT.kOrange - 9,
        ROOT.kOrange - 4,
        ROOT.kOrange + 1,
        ROOT.kOrange + 7,
        ROOT.kRed - 7,
        ROOT.kRed - 3,
        ROOT.kRed + 1,
        ROOT.kPink - 4,
        ROOT.kMagenta - 7,
        ROOT.kMagenta - 4,
    ],
}


def to_root_string_vector(strings):
    vec = ROOT.std.vector("string")()
    for item in strings:
        vec.push_back(str(item))
    return vec


def declare_rdf_helpers():
    ROOT.gInterpreter.Declare(
        r"""
        #include <vector>
        #include "ROOT/RVec.hxx"

        bool has_any_pdg(const ROOT::VecOps::RVec<int>& pdgs,
                         const ROOT::VecOps::RVec<int>& targets) {
            for (const auto pdg : pdgs) {
                for (const auto target : targets) {
                    if (pdg == target) return true;
                }
            }
            return false;
        }

        bool has_any_pdg(const std::vector<int>& pdgs,
                         const ROOT::VecOps::RVec<int>& targets) {
            for (const auto pdg : pdgs) {
                for (const auto target : targets) {
                    if (pdg == target) return true;
                }
            }
            return false;
        }
        """
    )


def parse_secondary_pdgs(args):
    if args.secondary_pdgs:
        pdgs = []
        for token in args.secondary_pdgs.replace(",", " ").split():
            pdgs.append(int(token))
        if not pdgs:
            raise ValueError("--secondary-pdgs was provided but no PDG codes were parsed")
        return pdgs, "custom_" + "_".join(str(pdg) for pdg in pdgs)

    if args.muondis_secondary not in PDG_SELECTIONS:
        raise ValueError(f"Unknown muonDIS secondary selection: {args.muondis_secondary}")

    return list(PDG_SELECTIONS[args.muondis_secondary]), args.muondis_secondary


def pdg_vector_expr(pdgs):
    values = ", ".join(str(int(pdg)) for pdg in pdgs)
    return f"ROOT::VecOps::RVec<int>{{{values}}}"


def pdg_code_to_name(pdg_code):
    db = ROOT.TDatabasePDG.Instance()
    particle = db.GetParticle(int(pdg_code))
    if particle:
        return str(particle.GetName())
    return f"unknown_{int(pdg_code)}"


def build_muondis_selection(base_cut, extra_cut, secondary_pdgs):
    selection = common.get_category_selection("muonDIS", base_cut, extra_cut, common.cut_info)

    if secondary_pdgs:
        pdg_expr = pdg_vector_expr(secondary_pdgs)
        secondary_cut = f"has_any_pdg(secondary_pdg, {pdg_expr})"
        return f"({selection}) && ({secondary_cut})" if selection else secondary_cut

    return selection if selection else "1"


def make_pdg_table(df, selection):
    node = df.Filter(selection)
    arr = node.Take["ROOT::RVec<int>"]("secondary_pdg").GetValue()

    counter = Counter()
    total = 0
    selected_events = 0

    for event_pdgs in arr:
        selected_events += 1
        for pdg in event_pdgs:
            counter[int(pdg)] += 1
            total += 1

    rows = []
    for pdg_code, count in counter.items():
        fraction = count / total if total > 0 else 0.0
        rows.append((pdg_code, pdg_code_to_name(pdg_code), count, fraction))

    rows.sort(key=lambda row: row[3], reverse=True)
    return rows, selected_events, total


def save_pdg_table(rows, selected_events, total_pdgs, outpath, selection):
    with open(outpath, "w") as fout:
        fout.write(f"selection: {selection}\n")
        fout.write(f"selected_events: {selected_events}\n")
        fout.write(f"total_secondary_pdgs: {total_pdgs}\n\n")
        fout.write(f"{'pdgcode':>12}  {'pdg_name':<24}  {'count':>12}  {'fraction':>14}\n")
        fout.write("-" * 70 + "\n")

        for pdg_code, name, count, fraction in rows:
            fout.write(f"{pdg_code:12d}  {name:<24}  {count:12d}  {fraction:14.8e}\n")

    print(f"[INFO] Saved PDG table: {outpath}")


def scale_hist_to_lumi(hist, sample_lumi, target_lumi, label):
    if not math.isfinite(sample_lumi) or sample_lumi <= 0:
        raise RuntimeError(f"Invalid luminosity for {label}: {sample_lumi}")

    scale = target_lumi / sample_lumi
    if not math.isfinite(scale):
        raise RuntimeError(f"Invalid scale for {label}: {scale}")

    hist.Scale(scale)
    common.sanitize_hist_bins(hist)
    print(f"[scale] {label:25s}: lumi={sample_lumi:.6g}, target={target_lumi:.6g}, x {scale:.6g}")
    return scale


def has_valid_lumi(sample_lumi):
    return math.isfinite(sample_lumi) and sample_lumi > 0


def make_reference_histograms(
    category_files,
    grouped_lumi,
    reference,
    tree_name,
    feature,
    hist_cfg,
    base_cut,
    extra_cut,
    target_lumi,
    fold_underflow=False,
    fold_overflow=False,
):
    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    reference_hists = []

    for category in REFERENCE_BINS[reference]:
        files = category_files.get(category, [])
        if not files:
            print(f"[skip] no files for {category}")
            continue

        sample_lumi = grouped_lumi.get(category, float("nan"))
        if not has_valid_lumi(sample_lumi):
            print(f"[skip] invalid luminosity for {category}: {sample_lumi}")
            continue

        chain = ROOT.TChain(tree_name)
        for filepath in files:
            chain.Add(filepath)

        selection = common.get_category_selection(category, base_cut, extra_cut, common.cut_info)
        safe_cat = re.sub(r"[^A-Za-z0-9_]", "_", category)
        hname = f"h_ref_{safe_cat}"
        draw_expr = f"{feature}>>{hname}({n_bins},{x_min},{x_max})"
        selected = chain.Draw(draw_expr, selection, "goff")
        print(f"[draw] {category:25s} expr={draw_expr} cut={selection!r} selected={selected}")

        if selected < 0:
            raise RuntimeError(f"TChain::Draw failed for reference category {category}")

        hist_tmp = chain.GetHistogram()
        if hist_tmp is None:
            raise RuntimeError(f"GetHistogram() returned null for reference category {category}")

        hist = hist_tmp.Clone(f"{hname}_clone")
        hist.SetDirectory(0)

        if fold_underflow:
            common.add_underflow_to_first_bin(hist)
        if fold_overflow:
            common.add_overflow_to_last_bin(hist)

        common.sanitize_hist_bins(hist)
        common.print_hist_summary(f"raw {category}", hist)

        scale_hist_to_lumi(hist, sample_lumi, target_lumi, category)
        reference_hists.append((category, hist))

    if not reference_hists:
        raise RuntimeError(f"No reference histograms were built for reference={reference}")

    return reference_hists


def make_muondis_histogram(
    category_files,
    grouped_lumi,
    tree_name,
    feature,
    hist_cfg,
    base_cut,
    extra_cut,
    target_lumi,
    secondary_pdgs,
    secondary_label,
    fold_underflow=False,
    fold_overflow=False,
):
    files = category_files.get("muonDIS", [])
    if not files:
        raise RuntimeError(
            "No muonDIS files found. Expected files like hist_MC_muonDIS_Max10-1.root "
            "in the input directory."
        )

    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    full_selection = build_muondis_selection(base_cut, extra_cut, secondary_pdgs)

    files_vec = to_root_string_vector(files)
    df = ROOT.RDataFrame(tree_name, files_vec)

    hist_feature = feature
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", feature):
        hist_feature = "feature_to_plot"
        df = df.Define(hist_feature, feature)

    hist_model = ROOT.RDF.TH1DModel(
        f"h_muonDIS_{secondary_label}",
        "",
        n_bins,
        float(x_min),
        float(x_max),
    )

    print(f"[rdf] muonDIS files={len(files)} cut={full_selection!r}")
    hist_ptr = df.Filter(full_selection).Histo1D(hist_model, hist_feature)
    hist = hist_ptr.GetValue().Clone(f"h_muonDIS_{secondary_label}_clone")
    hist.SetDirectory(0)

    if fold_underflow:
        common.add_underflow_to_first_bin(hist)
    if fold_overflow:
        common.add_overflow_to_last_bin(hist)

    common.sanitize_hist_bins(hist)
    common.print_hist_summary(f"raw muonDIS {secondary_label}", hist)

    scale_hist_to_lumi(hist, grouped_lumi.get("muonDIS", float("nan")), target_lumi, "muonDIS")
    return hist


def make_data_histogram(
    category_files,
    tree_name,
    feature,
    hist_cfg,
    base_cut,
    extra_cut,
    fold_underflow=False,
    fold_overflow=False,
):
    files = category_files.get("real_data", [])
    if not files:
        raise RuntimeError("No real_data files found for data comparison")

    bin_width, x_min, x_max, _, _ = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    chain = ROOT.TChain(tree_name)
    for filepath in files:
        chain.Add(filepath)

    selection = common.get_category_selection("real_data", base_cut, extra_cut, common.cut_info)
    hname = "h_data_reference"
    draw_expr = f"{feature}>>{hname}({n_bins},{x_min},{x_max})"
    selected = chain.Draw(draw_expr, selection, "goff")
    print(f"[draw] {'real_data':25s} expr={draw_expr} cut={selection!r} selected={selected}")

    if selected < 0:
        raise RuntimeError("TChain::Draw failed for real_data")

    hist_tmp = chain.GetHistogram()
    if hist_tmp is None:
        raise RuntimeError("GetHistogram() returned null for real_data")

    hist = hist_tmp.Clone(f"{hname}_clone")
    hist.SetDirectory(0)

    if fold_underflow:
        common.add_underflow_to_first_bin(hist)
    if fold_overflow:
        common.add_overflow_to_last_bin(hist)

    common.sanitize_hist_bins(hist)
    common.print_hist_summary("raw real_data", hist)
    return hist


def sum_histograms(named_hists, name):
    total = None
    for _, hist in named_hists:
        if total is None:
            total = hist.Clone(name)
            total.SetDirectory(0)
        else:
            total.Add(hist)

    if total is not None:
        common.sanitize_hist_bins(total)

    return total


def style_reference_histograms(reference_hists, reference):
    colors = REFERENCE_COLORS[reference]
    for i, (_, hist) in enumerate(reference_hists):
        color = colors[i % len(colors)]
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        hist.SetFillColor(color)
        hist.SetFillStyle(1001)


def style_muondis_hist(hist):
    hist.SetLineColor(ROOT.kMagenta + 2)
    hist.SetMarkerColor(ROOT.kMagenta + 2)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.9)
    hist.SetLineWidth(3)
    hist.SetFillStyle(0)


def style_data_reference_hist(hist):
    hist.SetLineColor(ROOT.kBlack)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.9)
    hist.SetLineWidth(2)
    hist.SetFillStyle(0)


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

    vals = sorted(vals)
    if len(vals) > 2:
        vals = vals[1:-1]

    ymin = min(vals)
    ymax = max(vals)
    if ymax - ymin < 0.2:
        center = 0.5 * (ymin + ymax)
        ymin = center - 0.1
        ymax = center + 0.1

    span = ymax - ymin
    ymin -= padding * span
    ymax += padding * span
    ymin = min(ymin, 1.0)
    ymax = max(ymax, 1.0)
    return ymin, ymax


def make_ratio(numerator, denominator, name):
    ratio = numerator.Clone(name)
    ratio.SetDirectory(0)

    for ibin in range(1, ratio.GetNbinsX() + 1):
        num = numerator.GetBinContent(ibin)
        den = denominator.GetBinContent(ibin)
        num_err = numerator.GetBinError(ibin)

        if den > 0:
            ratio.SetBinContent(ibin, num / den)
            ratio.SetBinError(ibin, num_err / den)
        else:
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)

    common.sanitize_hist_bins(ratio)
    return ratio


def draw_comparison(
    reference_hists,
    reference_sum,
    muondis_hist,
    reference,
    secondary_label,
    secondary_pdgs,
    feature,
    hist_cfg,
    target_lumi,
    outdir,
    base_cut,
    extra_cut,
    title="",
):
    ROOT.gStyle.SetOptStat(0)
    os.makedirs(outdir, exist_ok=True)

    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    cut_tag = common.sanitize_cut(base_cut, extra_cut)
    pdg_tag = "_".join(str(pdg) for pdg in secondary_pdgs)
    output_file = os.path.join(
        outdir,
        (
            f"compare_{reference}_stack_vs_muonDIS_{secondary_label}_{pdg_tag}"
            f"__{feature}__{cut_tag}__binWidth{bin_width}__range{x_min}-{x_max}__logy{logy}.pdf"
        ),
    )

    stack = ROOT.THStack(f"stack_{reference}_{feature}_{cut_tag}", "")
    for _, hist in reference_hists:
        stack.Add(hist)

    ymax_base = max(reference_sum.GetMaximum(), muondis_hist.GetMaximum(), 1.0)
    ymax = 20.0 * ymax_base if logy else 1.45 * ymax_base

    canvas = ROOT.TCanvas(f"c_compare_{reference}_{feature}_{cut_tag}", "", 900, 800)
    pad_top = ROOT.TPad(f"pad_top_{reference}_{feature}_{cut_tag}", "", 0.0, 0.35, 1.0, 1.0)
    pad_bottom = ROOT.TPad(f"pad_bottom_{reference}_{feature}_{cut_tag}", "", 0.0, 0.0, 1.0, 0.35)

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

    pad_top.cd()
    frame = ROOT.TH1D(f"frame_{reference}_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
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

    stack.Draw("HIST SAME")
    muondis_hist.Draw("E1 SAME")

    legend = ROOT.TLegend(0.58, 0.47, 0.89, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    legend.AddEntry(
        muondis_hist,
        f"muonDIS {secondary_label} ({muondis_hist.Integral():.2g})",
        "lep",
    )
    for category, hist in reversed(reference_hists):
        label = category.replace("MC_", "").replace("_", " ")
        legend.AddEntry(hist, f"{label} ({hist.Integral():.2g})", "f")

    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextAlign(13)
    text.SetTextSize(0.030)
    base_cut_text = base_cut.strip() if base_cut and base_cut.strip() else "none"
    extra_cut_text = extra_cut.strip() if extra_cut and extra_cut.strip() else "none"
    text.DrawLatex(0.15, 0.88, f"#bf{{Base cut}}: {base_cut_text}")
    text.DrawLatex(0.15, 0.835, f"#bf{{Extra cut}}: {extra_cut_text}")
    text.DrawLatex(
        0.15,
        0.790,
        f"#bf{{muonDIS secondary PDG}}: {', '.join(str(pdg) for pdg in secondary_pdgs)}",
    )
    text.DrawLatex(0.15, 0.745, f"#int #font[12]{{L}} dt = {target_lumi:.3f} fb^{{-1}}")

    pad_bottom.cd()
    ratio_frame = ROOT.TH1D(f"ratio_frame_{reference}_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
    ratio_frame.SetDirectory(0)
    ratio_frame.GetXaxis().SetTitle(axis_title)
    ratio_frame.GetYaxis().SetTitle("muonDIS/ref")
    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetTitleOffset(1.2)
    ratio_frame.GetXaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.5)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetYaxis().CenterTitle()
    ratio_frame.Draw()

    ratio = make_ratio(muondis_hist, reference_sum, f"ratio_{reference}_{feature}_{cut_tag}")
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.9)
    ratio.SetLineColor(ROOT.kMagenta + 2)
    ratio.SetMarkerColor(ROOT.kMagenta + 2)
    ratio.SetLineWidth(2)
    ratio.Draw("E1 SAME")

    ymin, ymax_ratio = get_ratio_range(ratio)
    ratio_frame.SetMinimum(ymin)
    ratio_frame.SetMaximum(ymax_ratio)

    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    pad_bottom.RedrawAxis()

    canvas.cd()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(output_file)
    print(f"\nSaved plot to {output_file}")


def draw_data_muondis_comparison(
    data_hist,
    muondis_hist,
    secondary_label,
    secondary_pdgs,
    feature,
    hist_cfg,
    target_lumi,
    outdir,
    base_cut,
    extra_cut,
    title="",
):
    ROOT.gStyle.SetOptStat(0)
    os.makedirs(outdir, exist_ok=True)

    bin_width, x_min, x_max, axis_title, logy = hist_cfg
    n_bins = int((x_max - x_min) / bin_width)
    cut_tag = common.sanitize_cut(base_cut, extra_cut)
    pdg_tag = "_".join(str(pdg) for pdg in secondary_pdgs) if secondary_pdgs else "all"
    output_file = os.path.join(
        outdir,
        (
            f"compare_data_vs_muonDIS_{secondary_label}_{pdg_tag}"
            f"__{feature}__{cut_tag}__binWidth{bin_width}__range{x_min}-{x_max}__logy{logy}.pdf"
        ),
    )

    ymax_base = max(data_hist.GetMaximum(), muondis_hist.GetMaximum(), 1.0)
    ymax = 20.0 * ymax_base if logy else 1.45 * ymax_base

    canvas = ROOT.TCanvas(f"c_compare_data_{feature}_{cut_tag}", "", 900, 800)
    pad_top = ROOT.TPad(f"pad_top_data_{feature}_{cut_tag}", "", 0.0, 0.35, 1.0, 1.0)
    pad_bottom = ROOT.TPad(f"pad_bottom_data_{feature}_{cut_tag}", "", 0.0, 0.0, 1.0, 0.35)

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

    pad_top.cd()
    frame = ROOT.TH1D(f"frame_data_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
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

    data_hist.Draw("E1 SAME")
    muondis_hist.Draw("HIST SAME")
    muondis_hist.Draw("E1 SAME")

    legend = ROOT.TLegend(0.58, 0.62, 0.89, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(data_hist, f"data ({data_hist.Integral():.2g})", "lep")
    legend.AddEntry(muondis_hist, f"muonDIS {secondary_label} ({muondis_hist.Integral():.2g})", "lep")
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextAlign(13)
    text.SetTextSize(0.030)
    base_cut_text = base_cut.strip() if base_cut and base_cut.strip() else "none"
    extra_cut_text = extra_cut.strip() if extra_cut and extra_cut.strip() else "none"
    text.DrawLatex(0.15, 0.88, f"#bf{{Base cut}}: {base_cut_text}")
    text.DrawLatex(0.15, 0.835, f"#bf{{Extra cut}}: {extra_cut_text}")
    text.DrawLatex(0.15, 0.790, f"#int #font[12]{{L}} dt = {target_lumi:.3f} fb^{{-1}}")

    pad_bottom.cd()
    ratio_frame = ROOT.TH1D(f"ratio_frame_data_{feature}_{cut_tag}", "", n_bins, x_min, x_max)
    ratio_frame.SetDirectory(0)
    ratio_frame.GetXaxis().SetTitle(axis_title)
    ratio_frame.GetYaxis().SetTitle("muonDIS/data")
    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetTitleOffset(1.2)
    ratio_frame.GetXaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetTitleSize(0.10)
    ratio_frame.GetYaxis().SetTitleOffset(0.5)
    ratio_frame.GetYaxis().SetLabelSize(0.08)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.GetYaxis().CenterTitle()
    ratio_frame.Draw()

    ratio = make_ratio(muondis_hist, data_hist, f"ratio_data_{feature}_{cut_tag}")
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.9)
    ratio.SetLineColor(ROOT.kMagenta + 2)
    ratio.SetMarkerColor(ROOT.kMagenta + 2)
    ratio.SetLineWidth(2)
    ratio.Draw("E1 SAME")

    ymin, ymax_ratio = get_ratio_range(ratio)
    ratio_frame.SetMinimum(ymin)
    ratio_frame.SetMaximum(ymax_ratio)

    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    pad_bottom.RedrawAxis()
    canvas.cd()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(output_file)
    print(f"\nSaved plot to {output_file}")


def write_muondis_pdg_table(
    category_files,
    tree_name,
    outdir,
    feature,
    base_cut,
    extra_cut,
    secondary_pdgs,
    secondary_label,
):
    files = category_files.get("muonDIS", [])
    if not files:
        raise RuntimeError(
            "No muonDIS files found. Expected files like hist_MC_muonDIS_Max10-1.root "
            "in the input directory."
        )

    os.makedirs(outdir, exist_ok=True)
    files_vec = to_root_string_vector(files)
    df = ROOT.RDataFrame(tree_name, files_vec)
    selection = build_muondis_selection(base_cut, extra_cut, secondary_pdgs)

    rows, selected_events, total_pdgs = make_pdg_table(df, selection)
    cut_tag = common.sanitize_cut(base_cut, extra_cut)
    pdg_tag = "_".join(str(pdg) for pdg in secondary_pdgs) if secondary_pdgs else "all"
    outpath = os.path.join(
        outdir,
        f"{feature}__muonDIS_{secondary_label}_{pdg_tag}__{cut_tag}.txt",
    )
    save_pdg_table(rows, selected_events, total_pdgs, outpath, selection)


def main(args):
    is_pdg_feature = "pdg" in args.feature.lower()
    if not is_pdg_feature and args.feature not in common.hist_info:
        raise ValueError(f"{args.feature} not defined in hist_info")

    if args.reference != "data" and args.reference not in REFERENCE_BINS:
        raise ValueError(f"Unknown reference sample: {args.reference}")

    secondary_pdgs, secondary_label = parse_secondary_pdgs(args)
    declare_rdf_helpers()

    category_files, grouped_lumi, open_files = common.collect_files_and_lumi(args.input_dir, args.tree)

    if is_pdg_feature:
        write_muondis_pdg_table(
            category_files=category_files,
            tree_name=args.tree,
            outdir=args.outdir,
            feature=args.feature,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            secondary_pdgs=secondary_pdgs,
            secondary_label=secondary_label,
        )
        _ = open_files
        return

    hist_cfg = common.get_hist_config(args.feature, args.extra_cut_key)

    target_lumi = grouped_lumi.get("real_data", float("nan"))
    if args.target_lumi is not None:
        target_lumi = args.target_lumi

    if not math.isfinite(target_lumi) or target_lumi <= 0:
        raise RuntimeError(f"Invalid target luminosity: {target_lumi}")

    print(f"\nTarget lumi = {target_lumi:.6g}")
    print(f"Reference sample = {args.reference}")
    print(f"muonDIS secondary selection = {secondary_label}: {secondary_pdgs}")

    if args.reference == "data":
        data_hist = make_data_histogram(
            category_files=category_files,
            tree_name=args.tree,
            feature=args.feature,
            hist_cfg=hist_cfg,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            fold_underflow=args.fold_underflow,
            fold_overflow=args.fold_overflow,
        )

        muondis_hist = make_muondis_histogram(
            category_files=category_files,
            grouped_lumi=grouped_lumi,
            tree_name=args.tree,
            feature=args.feature,
            hist_cfg=hist_cfg,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            target_lumi=target_lumi,
            secondary_pdgs=secondary_pdgs,
            secondary_label=secondary_label,
            fold_underflow=args.fold_underflow,
            fold_overflow=args.fold_overflow,
        )

        style_data_reference_hist(data_hist)
        style_muondis_hist(muondis_hist)

        print("\n=== Comparison Summary ===")
        print(f"data integral               = {data_hist.Integral():.6g}")
        print(f"muonDIS {secondary_label} integral       = {muondis_hist.Integral():.6g}")

        draw_data_muondis_comparison(
            data_hist=data_hist,
            muondis_hist=muondis_hist,
            secondary_label=secondary_label,
            secondary_pdgs=secondary_pdgs,
            feature=args.feature,
            hist_cfg=hist_cfg,
            target_lumi=target_lumi,
            outdir=args.outdir,
            base_cut=args.base_cut,
            extra_cut=args.extra_cut,
            title=args.title,
        )

        _ = open_files
        return

    reference_hists = make_reference_histograms(
        category_files=category_files,
        grouped_lumi=grouped_lumi,
        reference=args.reference,
        tree_name=args.tree,
        feature=args.feature,
        hist_cfg=hist_cfg,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        target_lumi=target_lumi,
        fold_underflow=args.fold_underflow,
        fold_overflow=args.fold_overflow,
    )

    muondis_hist = make_muondis_histogram(
        category_files=category_files,
        grouped_lumi=grouped_lumi,
        tree_name=args.tree,
        feature=args.feature,
        hist_cfg=hist_cfg,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        target_lumi=target_lumi,
        secondary_pdgs=secondary_pdgs,
        secondary_label=secondary_label,
        fold_underflow=args.fold_underflow,
        fold_overflow=args.fold_overflow,
    )

    reference_sum = sum_histograms(reference_hists, f"h_{args.reference}_sum")
    if reference_sum is None:
        raise RuntimeError("Reference sum histogram is empty")

    style_reference_histograms(reference_hists, args.reference)
    style_muondis_hist(muondis_hist)

    print("\n=== Comparison Summary ===")
    print(f"{REFERENCE_LABELS[args.reference]} stack integral = {reference_sum.Integral():.6g}")
    print(f"muonDIS {secondary_label} integral       = {muondis_hist.Integral():.6g}")

    draw_comparison(
        reference_hists=reference_hists,
        reference_sum=reference_sum,
        muondis_hist=muondis_hist,
        reference=args.reference,
        secondary_label=secondary_label,
        secondary_pdgs=secondary_pdgs,
        feature=args.feature,
        hist_cfg=hist_cfg,
        target_lumi=target_lumi,
        outdir=args.outdir,
        base_cut=args.base_cut,
        extra_cut=args.extra_cut,
        title=args.title,
    )

    _ = open_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare luminosity-normalized kaon/neutron energy-bin stacks "
            "against muonDIS events selected by secondary PDG."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/experiment/sndlhc/users/zhibin/nueAnalysis",
        help="Directory containing hist_{partition}.root files",
    )
    parser.add_argument(
        "--reference",
        choices=sorted(list(REFERENCE_BINS.keys()) + ["data"]),
        default="kaon",
        help="Reference to compare against muonDIS: neutral-hadron MC family or real data",
    )
    parser.add_argument(
        "--muondis-secondary",
        choices=sorted(PDG_SELECTIONS.keys()),
        default="neutral_kaon",
        help="Predefined secondary-PDG selection for muonDIS",
    )
    parser.add_argument(
        "--secondary-pdgs",
        default="",
        help='Override predefined secondary selection, e.g. "130,310"',
    )
    parser.add_argument(
        "--feature",
        default="density_sndsw_scifi",
        choices=list(common.hist_info.keys()) + PDG_TABLE_FEATURES,
        help="Branch/expression to plot",
    )
    parser.add_argument(
        "--extra_cut",
        default="",
        help='Selection cut, e.g. "density_sndsw_scifi > 1000 && density_sndsw_scifi < 6000"',
    )
    parser.add_argument(
        "--extra-cut-key",
        default="",
        help="Named extra cut key used to choose feature-specific plotting range overrides",
    )
    parser.add_argument(
        "--base_cut",
        default="2 3 4 5 8 9 10",
        help='Selection base cut indices, e.g. "5 8 9 10"',
    )
    parser.add_argument(
        "--tree",
        default="sndData",
        help="TTree name",
    )
    parser.add_argument(
        "--outdir",
        default="hadron_muondis_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--target-lumi",
        type=float,
        default=None,
        help="Target luminosity. Defaults to summed real_data luminosity from input files.",
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

    main(parser.parse_args())
