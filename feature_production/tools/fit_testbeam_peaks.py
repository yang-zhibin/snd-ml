#!/usr/bin/env python3
"""Gaussian peak position of testbeam feature distributions versus beam energy.

For every curve (particle x dataset x energy) of a finished testbeam production
run, fit a Gaussian around the distribution peak and record the fitted mean.
Then plot peak position versus beam energy, separately for MC and for data,
with electrons and pions overlaid. Default features: the cluster-1 density
fractions (expected energy independent) and the cluster-1 densities.

Detector setups are never mixed: tungsten (2024, W walls) and iron (2023 H8,
Fe walls) are fitted and plotted separately (--setups, default W Fe).

Fit: start at the highest bin above xmin + f*(xmax-xmin) (f = --min-x-fraction,
default per feature: 0.04 for fractions, i.e. x > 0.04, skipping the near-zero
spike of diffuse hadronic showers; 0.02 = the first bin for densities), then iterate a Gaussian fit in
[mean - K*sigma, mean + K*sigma] (K = --window, default 1.5) until it is stable.
Curves whose maximum sits in the first/last bin are fitted the same way but
flagged in peaks.csv (edge_peak = 1). Curves without a peak above --min-x (fitted
mean pinned to the window edge, has_peak = 0) are kept in peaks.csv but left out
of the fit pages' curves and of the peak-vs-energy plots.

Output (default <run>/peak-vs-energy/<selection>/<feature>/), tungsten:
  fits.pdf                one page per energy: histograms + fitted Gaussians
  peak_vs_energy_mc.pdf   e and pi MC: fitted peak vs E
  peak_vs_energy_data.pdf e and pi data: fitted peak vs E
iron: the same names with suffix _Fe (only files with curves are written);
  peaks.csv in the output root: fit results (mean, sigma, errors, chi2/ndf, flags)

Run inside the LCG environment (node with working CVMFS):
  source snakemake/envs/env_lcg.sh
  python3 feature_production/tools/fit_testbeam_peaks.py <run-dir>
      [--features silvia__scifi_horizontal_cluster1_density_fraction ...]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_testbeam_by_energy as tb  # noqa: E402  (shared loading, labels, styles)

ROOT = tb.ROOT

DEFAULT_FEATURES = [
    'silvia__scifi_horizontal_cluster1_density_fraction',
    'silvia__scifi_vertical_cluster1_density_fraction',
    'silvia__scifi_horizontal_cluster1_density',
    'silvia__scifi_vertical_cluster1_density',
]
SETUP_SUFFIX = {'W': '', 'Fe': '_Fe'}
# Lowest fraction of the x axis ignored in peak search/fit: fractions skip x < 0.04;
# densities (50 bins of ~750) skip only the first bin, which holds no-shower events.
MIN_X_FRACTION = {'_density_fraction': 0.04, '_density': 0.02}


def default_min_x_fraction(feature):
    for ending, value in MIN_X_FRACTION.items():
        if feature.endswith(ending):
            return value
    return 0.04
# Known problematic data run, annotated on the data plot (see task notes 2026-11-26).
FLAGGED = {('testbeam_data_2024', 11, 100): 'e^{-} 100 GeV data: run 100890 (75% empty, not a clean e^{-} sample)'}
SERIES_MARKERS = {('mc', 11): 20, ('mc', 211): 21, ('mc', -211): 25,
                  ('data', 11): 20, ('data', 211): 21, ('data', -211): 25}


def fit_peak(hist, window, name, min_x_fraction):
    """Iterative Gaussian fit around the peak above xmin + f*(xmax - xmin). None if empty."""
    axis = hist.GetXaxis()
    nbins = axis.GetNbins()
    min_x = axis.GetXmin() + min_x_fraction * (axis.GetXmax() - axis.GetXmin())
    first = max(1, axis.FindBin(min_x + 1e-9)) if min_x is not None else 1
    if axis.GetBinLowEdge(first) < min_x if min_x is not None else False:
        first += 1
    if hist.Integral(first, nbins) <= 0:
        return None
    peak_bin = max(range(first, nbins + 1), key=hist.GetBinContent)
    mean = axis.GetBinCenter(peak_bin)
    sigma = max(hist.GetStdDev(), 2 * axis.GetBinWidth(1)) / 2
    func = None
    for _ in range(6):
        xmin = axis.GetBinLowEdge(first)
        low = max(xmin, mean - window * sigma)
        high = min(axis.GetXmax(), mean + window * sigma)
        # keep at least 5 bins in the fit window
        while axis.FindBin(high) - axis.FindBin(low) < 4 and (low > xmin or high < axis.GetXmax()):
            low = max(xmin, low - axis.GetBinWidth(1))
            high = min(axis.GetXmax(), high + axis.GetBinWidth(1))
        func = ROOT.TF1(name, 'gaus', low, high)
        func.SetParameters(hist.GetBinContent(peak_bin), mean, sigma)
        func.SetParLimits(1, low, high)
        func.SetParLimits(2, 0.25 * axis.GetBinWidth(1), axis.GetXmax() - axis.GetXmin())
        result = hist.Fit(func, 'QNRS')
        new_mean, new_sigma = func.GetParameter(1), abs(func.GetParameter(2))
        stable = abs(new_mean - mean) < 0.05 * axis.GetBinWidth(1) and abs(new_sigma - sigma) < 0.05 * sigma
        mean, sigma = new_mean, new_sigma
        if stable:
            break
    ndf = func.GetNDF()
    fitted = func.GetParameter(1)
    tolerance = 0.25 * axis.GetBinWidth(1)
    # A mean pinned to the fit-window edge means the curve has no peak above min_x.
    has_peak = int(result is not None and int(result.Status()) == 0 and
                   low + tolerance < fitted < high - tolerance and peak_bin != first)
    return dict(has_peak=has_peak, func=func, mean=func.GetParameter(1), mean_err=func.GetParError(1),
                sigma=abs(func.GetParameter(2)), sigma_err=func.GetParError(2),
                chi2_ndf=func.GetChisquare() / ndf if ndf > 0 else float('nan'),
                fit_low=low, fit_high=high, status=int(result.Status()) if result else -1,
                edge_peak=int(peak_bin in (first, nbins)))


def draw_fit_page(canvas, curves, energy, feature, axis_title, keep):
    """Histograms (unit area) with their fitted Gaussians for one energy."""
    tb.draw_energy(canvas, curves, energy, feature, axis_title, False, keep)
    for curve in curves:
        fit = curve.get('fit')
        if not fit or not fit['has_peak']:
            continue
        func = fit['func']
        func.SetLineColor(curve['plotted'].GetLineColor())
        func.SetLineWidth(2)
        func.SetLineStyle(2 if curve['kind'] == 'mc' else 1)
        func.SetNpx(400)
        func.Draw('SAME')
        keep.append(func)


def series_key(curve):
    return (curve['kind'], curve['pdg'], curve['domain'])


def draw_trend(path, curves, kind, feature, axis_title, keep):
    """Fitted peak vs beam energy for one kind ('mc' or 'data'), e and pi overlaid."""
    series = {}
    for curve in curves:
        if curve['kind'] != kind or not curve.get('fit') or not curve['fit']['has_peak']:
            continue
        series.setdefault(series_key(curve), []).append(curve)
    canvas = ROOT.TCanvas('trend_%s_%s' % (feature, kind), '', 900, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.08)
    keep.append(canvas)
    if not series:
        return None
    values = [c['fit']['mean'] for group in series.values() for c in group]
    span = max(values) - min(values)
    ylo = max(0.0, min(values) - 0.4 * max(span, 0.05))
    yhi = max(values) + (0.9 + 0.12 * len(series)) * max(span, 0.05)
    frame = canvas.DrawFrame(0, ylo, 330, yhi)
    frame.GetXaxis().SetTitle('Beam energy (GeV)')
    frame.GetYaxis().SetTitle('Fitted Gaussian peak')
    frame.GetYaxis().SetTitleOffset(1.3)
    legend = ROOT.TLegend(0.15, 0.91 - 0.055 * len(series), 0.96, 0.91)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.028)
    for key in sorted(series, key=lambda k: (abs(k[1]) != 11, k[1], k[2])):
        group = sorted(series[key], key=lambda c: c['energy'])
        graph = ROOT.TGraphErrors(len(group))
        for i, curve in enumerate(group):
            graph.SetPoint(i, curve['energy'], curve['fit']['mean'])
            graph.SetPointError(i, 0.0, curve['fit']['mean_err'])
        color = tb.COLORS.get((key[1], key[0]), ROOT.kGray + 2)
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        marker = tb.DATA_MARKERS.get(key[2], SERIES_MARKERS.get((key[0], key[1]), 20)) if kind == 'data' \
            else SERIES_MARKERS.get((key[0], key[1]), 20)
        if kind == 'data' and key[1] == -211:
            marker = 25 if key[2] == 'testbeam_data_2024' else 26
        graph.SetMarkerStyle(marker)
        graph.SetMarkerSize(1.3)
        graph.SetLineWidth(2)
        graph.Draw('LP SAME')
        first = group[0]
        text = '%s %s, %s' % (tb.PARTICLES.get(key[1], str(key[1])), tb.DOMAIN_LABELS.get(key[2], key[2]),
                              tb.geometry_label(first['geometries']))
        legend.AddEntry(graph, text, 'lp')
        keep.append(graph)
    legend.Draw()
    keep.append(legend)
    notes = [text for (domain, pdg, energy), text in FLAGGED.items()
             if kind == 'data' and any(c['domain'] == domain and c['pdg'] == pdg and round(c['energy']) == energy
                                       for c in curves if c.get('fit') and c['fit']['has_peak'])]
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.024)
    for i, note in enumerate(notes):
        latex.DrawLatex(0.15, 0.14 + 0.035 * i, 'Note: ' + note)
    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextSize(0.04)
    title.SetTextSize(0.034)
    title.DrawLatex(0.14, 0.94, ('MC' if kind == 'mc' else 'Data') + ': ' + feature.replace('silvia__', '').replace('_', ' '))
    keep.extend([latex, title])
    canvas.SaveAs(str(path))
    return canvas


def setup_of(curve):
    """'W', 'Fe', ... from the curve geometry ids (first match)."""
    for geometry in curve['geometries']:
        match = tb.GEOMETRY_RE.match(str(geometry))
        if match:
            return match['material']
    return 'unknown'


def load_all_setups(selection_dir):
    """Load curves of every setup (iron included), grouped by setup then energy."""
    saved = set(tb.EXCLUDED_MATERIALS)
    tb.EXCLUDED_MATERIALS.clear()
    try:
        by_energy = tb.load_curves(selection_dir)
    finally:
        tb.EXCLUDED_MATERIALS.update(saved)
    grouped = {}
    for energy, curves in by_energy.items():
        for curve in curves:
            grouped.setdefault(setup_of(curve), {}).setdefault(energy, []).append(curve)
    return grouped


def process_setup(by_energy, out_dir, suffix, feature, selection, window, min_x_fraction, writer, keep):
    cfg = next(iter(by_energy.values()))[0]['config']
    axis_title = cfg.get('axis_title') or feature
    if len(axis_title) > 60:
        axis_title = feature
    all_curves, nfit, pages = [], 0, []
    energies = sorted(by_energy)
    for energy in energies:
        curves = sorted(by_energy[energy], key=tb.sort_key)
        for curve in curves:
            plotted = tb.normalised(curve, energy, feature)
            name = 'g_%s_%s_%d_%d' % (feature, curve['domain'], curve['pdg'], energy)
            curve['fit'] = fit_peak(plotted, window, name.replace('-', 'm'), min_x_fraction)
            nfit += curve['fit'] is not None
            all_curves.append(curve)
            fit = curve['fit'] or {}
            writer.writerow(dict(feature=feature, selection=selection, energy_gev=energy, group=curve['group'],
                                 particle_pdg=curve['pdg'], kind=curve['kind'], domain=curve['domain'],
                                 geometry=tb.geometry_label(curve['geometries']), valid_entries=curve['valid'],
                                 **{k: fit.get(k, '') for k in ('mean', 'mean_err', 'sigma', 'sigma_err', 'chi2_ndf',
                                                               'fit_low', 'fit_high', 'status', 'edge_peak', 'has_peak')}))
        canvas = ROOT.TCanvas('fit_%s%s_%d' % (feature, suffix, energy), '', 900, 700)
        draw_fit_page(canvas, curves, energy, feature, axis_title, keep)
        pages.append(canvas)
    fits_pdf = str(out_dir / ('fits%s.pdf' % suffix))
    for i, canvas in enumerate(pages):
        suffix_print = '' if len(pages) == 1 else ('(' if i == 0 else (')' if i == len(pages) - 1 else ''))
        canvas.Print(fits_pdf + suffix_print, 'Title:%d GeV' % energies[i])
    for kind in ('mc', 'data'):
        draw_trend(out_dir / ('peak_vs_energy_%s%s.pdf' % (kind, suffix)), all_curves, kind, feature + suffix,
                   axis_title, keep)
    return nfit


def process(selection_dir, out_dir, feature, selection, window, min_x_fraction, setups, writer):
    grouped = load_all_setups(selection_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep, counts = [], {}
    for setup in setups:
        if setup not in grouped:
            continue
        counts[setup] = process_setup(grouped[setup], out_dir, SETUP_SUFFIX.get(setup, '_' + setup), feature,
                                      selection, window, min_x_fraction, writer, keep)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Testbeam production run directory (contains histograms/)')
    parser.add_argument('--out', type=Path, help='Output directory (default <run>/peak-vs-energy)')
    parser.add_argument('--features', nargs='*', default=DEFAULT_FEATURES)
    parser.add_argument('--selection', default='selected')
    parser.add_argument('--window', type=float, default=1.5, help='Fit window half-width in sigmas (default 1.5)')
    parser.add_argument('--setups', nargs='+', default=['W', 'Fe'],
                        help='Detector setups to analyse, each separately (default: W Fe)')
    parser.add_argument('--min-x-fraction', type=float, default=None,
                        help='Ignore the lowest fraction of the x axis in peak search and fit '
                             '(default per feature: 0.04 for fractions, 0.02 = first bin for densities)')
    args = parser.parse_args()
    out_root = args.out or args.run / 'peak-vs-energy'
    out_root.mkdir(parents=True, exist_ok=True)
    fields = ['feature', 'selection', 'energy_gev', 'group', 'particle_pdg', 'kind', 'domain', 'geometry',
              'valid_entries', 'mean', 'mean_err', 'sigma', 'sigma_err', 'chi2_ndf', 'fit_low', 'fit_high',
              'status', 'edge_peak', 'has_peak']
    with open(out_root / 'peaks.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature in args.features:
            selection_dir = args.run / 'histograms' / '1d' / feature / args.selection
            if not selection_dir.is_dir():
                print('missing', selection_dir)
                continue
            counts = process(selection_dir, out_root / args.selection / feature, feature, args.selection,
                             args.window, args.min_x_fraction if args.min_x_fraction is not None
                             else default_min_x_fraction(feature), args.setups, writer)
            print('%-55s fits per setup: %s' % (feature, counts))
    print('Output:', out_root)


if __name__ == '__main__':
    main()
