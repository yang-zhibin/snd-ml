#!/usr/bin/env python3
"""Split testbeam feature distributions into one plot per beam energy.

Reads the per-group histograms already produced by a testbeam production run
(<run>/histograms/1d/<feature>/<selection>/<group>.root) and, for every feature,
draws one canvas per beam energy overlaying electrons vs pions and data vs MC,
all normalised to unit area.

Output (default <run>/plots-by-energy/<selection>/<feature>/):
  overview.pdf/.png   all energies as pads of one canvas
  E<energy>GeV.pdf/.png  one canvas per energy
  <feature>.pdf       multi-page PDF: overview + one page per energy
  summary.csv         entries, mean, std per curve

Run inside the LCG environment (on a node with working CVMFS):
  source snakemake/envs/env_lcg.sh
  python3 feature_production/tools/plot_testbeam_by_energy.py <run-dir> [--features f1 f2]
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

GROUP_RE = re.compile(r'^(?P<domain>[^:]+):pdg(?P<pdg>-?\d+):(?P<energy>[\d.]+)GeV$')
PARTICLES = {11: 'e^{-}', -11: 'e^{+}', 211: '#pi^{+}', -211: '#pi^{-}'}
DOMAIN_LABELS = {'testbeam_data_2024': 'data 2024', 'testbeam_mc_2024': 'MC 2024',
                 'testbeam_data_2023_h8': 'data 2023 H8'}
# Electrons (EM showers, signal-like) in blue tones, pions (hadronic, background-like) in red tones.
COLORS = {(11, 'data'): ROOT.kBlue + 1, (11, 'mc'): ROOT.kAzure + 7,
          (211, 'data'): ROOT.kRed + 1, (211, 'mc'): ROOT.kOrange + 7,
          (-211, 'data'): ROOT.kMagenta + 2, (-211, 'mc'): ROOT.kPink + 6,
          (-11, 'data'): ROOT.kCyan + 2, (-11, 'mc'): ROOT.kTeal + 3}
DATA_MARKERS = {'testbeam_data_2024': 20, 'testbeam_data_2023_h8': 24}


def parse_group(name):
    match = GROUP_RE.match(name)
    if not match:
        raise ValueError('Unexpected testbeam group label: ' + name)
    domain = match['domain']
    return dict(domain=domain, pdg=int(match['pdg']), energy=float(match['energy']),
                kind='mc' if '_mc_' in domain else 'data')


def load_curves(selection_dir):
    """Return {energy: [curve,...]} for one feature/selection directory."""
    by_energy = defaultdict(list)
    for path in sorted(selection_dir.glob('*.root')):
        f = ROOT.TFile.Open(str(path))
        meta = json.loads(str(f.Get('production_histogram_json').GetString()))
        hist = f.Get('hist').Clone('h_' + path.stem)
        hist.SetDirectory(0)
        f.Close()
        info = parse_group(meta['group'])
        info.update(hist=hist, group=meta['group'], config=meta.get('config', {}),
                    geometries=meta.get('geometries') or [meta.get('geometry', '')],
                    valid=int(meta.get('valid_entries', hist.GetEntries())),
                    selected=int(meta.get('selected_entries', 0)))
        by_energy[round(info['energy'])].append(info)
    return by_energy


def sort_key(curve):
    # electrons first, data before MC, 2024 before 2023
    return (abs(curve['pdg']) != 11, curve['pdg'], curve['kind'] != 'data', curve['domain'])


GEOMETRY_RE = re.compile(r'^(?P<material>[A-Za-z]+)_(?P<walls>\d+)walls')
MATERIALS = {'W': 'W', 'Fe': 'Fe'}


def geometry_label(geometries):
    """'W_2walls_28plates' -> '2 W walls'; several setups are joined with '+'."""
    labels = []
    for geometry in geometries:
        match = GEOMETRY_RE.match(str(geometry))
        text = ('%s %s walls' % (match['walls'], MATERIALS.get(match['material'], match['material']))
                if match else str(geometry))
        if text not in labels:
            labels.append(text)
    return ' + '.join(labels)


def label(curve):
    particle = PARTICLES.get(curve['pdg'], 'pdg' + str(curve['pdg']))
    return '%s %s, %g GeV, %s' % (particle, DOMAIN_LABELS.get(curve['domain'], curve['domain']),
                                  curve['energy'], geometry_label(curve['geometries']))


def style(curve):
    hist = curve['hist']
    color = COLORS.get((curve['pdg'], curve['kind']), ROOT.kGray + 2)
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetLineWidth(2)
    if curve['kind'] == 'data':
        hist.SetMarkerStyle(DATA_MARKERS.get(curve['domain'], 21))
        hist.SetMarkerSize(0.9)
        return 'E1'
    hist.SetMarkerSize(0)
    hist.SetLineStyle(1)
    return 'HIST'


def normalised(curve, energy, feature):
    hist = curve['hist'].Clone('n_%s_%s_%d' % (feature, curve['group'].replace(':', '_').replace('.', 'p'), energy))
    hist.SetDirectory(0)
    integral = hist.Integral()
    if integral > 0:
        hist.Scale(1.0 / integral)
    curve['plotted'] = hist
    return hist


def draw_energy(pad, curves, energy, feature, axis_title, logy, keep, compact=False):
    pad.cd()
    pad.SetLogy(bool(logy))
    pad.SetLeftMargin(0.13)
    pad.SetRightMargin(0.04)
    pad.SetTopMargin(0.08)
    curves = sorted(curves, key=sort_key)
    hists = []
    for curve in curves:
        hist = normalised(curve, energy, feature)
        curve['option'] = style(curve)
        hist.SetLineColor(curve['hist'].GetLineColor())
        hist.SetMarkerColor(curve['hist'].GetMarkerColor())
        hist.SetMarkerStyle(curve['hist'].GetMarkerStyle())
        hist.SetMarkerSize(curve['hist'].GetMarkerSize())
        hist.SetLineWidth(2)
        hists.append(hist)
    ymax = max([h.GetMaximum() for h in hists] + [1e-6])
    first = True
    for curve, hist in zip(curves, hists):
        hist.GetXaxis().SetTitle(axis_title)
        hist.GetYaxis().SetTitle('Normalised to unit area')
        hist.GetYaxis().SetTitleOffset(1.3)
        if logy:
            hist.SetMinimum(1e-4)
            hist.SetMaximum(ymax * 10 ** (1 + 0.5 * len(curves)))
        else:
            hist.SetMinimum(0)
            hist.SetMaximum(ymax * (1.0 + 0.11 * len(curves) + 0.2))
        hist.Draw(curve['option'] + ('' if first else ' SAME'))
        first = False
    # Full-width legend above the curves (the y range leaves headroom for it).
    legend = ROOT.TLegend(0.15, 0.91 - 0.055 * len(curves), 0.96, 0.91)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.036 if compact else 0.026)
    for curve, hist in zip(curves, hists):
        # Entries, mean and std live in summary.csv; the legend keeps only the label.
        text = label(curve)
        legend.AddEntry(hist, text, 'lep' if curve['kind'] == 'data' else 'l')
    legend.Draw()
    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextSize(0.045)
    title.DrawLatex(0.14, 0.94, '%d GeV' % energy)
    keep.extend(hists + [legend, title])


def plot_feature(selection_dir, out_dir, feature, selection, writer):
    by_energy = load_curves(selection_dir)
    if not by_energy:
        return 0
    any_curve = next(iter(by_energy.values()))[0]
    cfg = any_curve['config']
    axis_title = cfg.get('axis_title') or feature
    if len(axis_title) > 60:
        axis_title = feature
    logy = cfg.get('logy', False)
    out_dir.mkdir(parents=True, exist_ok=True)
    energies = sorted(by_energy)
    keep = []

    # Overview: all energies on one canvas.
    ncol = 3
    nrow = math.ceil(len(energies) / ncol)
    overview = ROOT.TCanvas('overview_' + feature, '', 1500, 500 * nrow)
    overview.Divide(ncol, nrow)
    for index, energy in enumerate(energies):
        draw_energy(overview.cd(index + 1), by_energy[energy], energy, feature, axis_title, logy, keep, compact=True)
    overview.SaveAs(str(out_dir / 'overview.pdf'))
    overview.SaveAs(str(out_dir / 'overview.png'))
    canvases = []
    for energy in energies:
        canvas = ROOT.TCanvas('c_%s_%d' % (feature, energy), '', 900, 700)
        draw_energy(canvas, by_energy[energy], energy, feature, axis_title, logy, keep)
        canvas.SaveAs(str(out_dir / ('E%dGeV.pdf' % energy)))
        canvas.SaveAs(str(out_dir / ('E%dGeV.png' % energy)))
        for curve in sorted(by_energy[energy], key=sort_key):
            writer.writerow(dict(feature=feature, selection=selection, energy_gev=energy, group=curve['group'],
                                 particle_pdg=curve['pdg'], kind=curve['kind'], domain=curve['domain'],
                                 geometry=geometry_label(curve['geometries']),
                                 valid_entries=curve['valid'], selected_entries=curve['selected'],
                                 mean=curve['hist'].GetMean(), std=curve['hist'].GetStdDev()))
        canvases.append(canvas)
    # Multi-page PDF written only after all single-file saves (interleaving confuses TPDF).
    multipage = str(out_dir / (feature + '.pdf'))
    pages = [overview] + canvases
    for i, canvas in enumerate(pages):
        suffix = '(' if i == 0 else (')' if i == len(pages) - 1 else '')
        canvas.Print(multipage + suffix, 'Title:' + (feature if i == 0 else '%d GeV' % energies[i - 1]))
    for canvas in pages:
        canvas.Close()
    return len(energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Testbeam production run directory (contains histograms/)')
    parser.add_argument('--out', type=Path, help='Output directory (default <run>/plots-by-energy)')
    parser.add_argument('--features', nargs='*', help='Only these features (default: all 1D features)')
    parser.add_argument('--selection', default=None, help='Selection name (default: all)')
    args = parser.parse_args()
    hist_root = args.run / 'histograms' / '1d'
    if not hist_root.is_dir():
        raise SystemExit('No histograms/1d under ' + str(args.run))
    out_root = args.out or args.run / 'plots-by-energy'
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['feature', 'selection', 'energy_gev', 'group', 'particle_pdg',
                                                    'kind', 'domain', 'geometry', 'valid_entries', 'selected_entries', 'mean', 'std'])
        writer.writeheader()
        for feature_dir in sorted(p for p in hist_root.iterdir() if p.is_dir()):
            feature = feature_dir.name
            if args.features and feature not in args.features:
                continue
            for selection_dir in sorted(p for p in feature_dir.iterdir() if p.is_dir()):
                if args.selection and selection_dir.name != args.selection:
                    continue
                n = plot_feature(selection_dir, out_root / selection_dir.name / feature, feature,
                                 selection_dir.name, writer)
                print('%-55s %-10s %d energies' % (feature, selection_dir.name, n))
    print('Output:', out_root)


if __name__ == '__main__':
    main()
