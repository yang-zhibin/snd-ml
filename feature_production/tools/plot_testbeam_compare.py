#!/usr/bin/env python3
"""Overlay hand-picked testbeam groups (any energies) for every feature.

Default comparison: 2024 tungsten data, electrons at 150 GeV versus pions
(pi+ and pi-) at 180 GeV. In the 2024 W data pions exist only at 180 GeV, so
this is the closest-energy e/pi data pair.

Groups are given as <domain>:<pdg>:<energy>, e.g. testbeam_data_2024:11:150.
Curves are normalised to unit area; iron setups are excluded as in the other
testbeam tools.

Output (default <run>/compare/<name>/<selection>/):
  <feature>.pdf   one plot per feature
  all_features.pdf  all features, one page each
  summary.csv     entries, mean and std per curve

Run inside the LCG environment (node with working CVMFS):
  source snakemake/envs/env_lcg.sh
  python3 feature_production/tools/plot_testbeam_compare.py <run-dir>
      [--groups testbeam_data_2024:11:150 testbeam_data_2024:211:180 ...] [--name e150_vs_pi180_data]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_testbeam_by_energy as tb  # noqa: E402

ROOT = tb.ROOT
DEFAULT_GROUPS = ['testbeam_data_2024:11:150', 'testbeam_data_2024:211:180', 'testbeam_data_2024:-211:180']
DEFAULT_NAME = 'e150_vs_pi180_data_W'


def parse_spec(spec):
    domain, pdg, energy = spec.split(':')
    return domain, int(pdg), round(float(energy))


def pick(by_energy, specs):
    chosen = []
    for domain, pdg, energy in specs:
        match = [c for c in by_energy.get(energy, []) if c['domain'] == domain and c['pdg'] == pdg]
        chosen.extend(match)
    return chosen


def draw(canvas, curves, feature, axis_title, logy, keep):
    canvas.cd()
    canvas.SetLogy(bool(logy))
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.08)
    curves = sorted(curves, key=tb.sort_key)
    hists = []
    for curve in curves:
        hist = tb.normalised(curve, round(curve['energy']), feature + '_cmp')
        curve['option'] = tb.style(curve)
        for attr in ('LineColor', 'MarkerColor', 'MarkerStyle', 'MarkerSize'):
            getattr(hist, 'Set' + attr)(getattr(curve['hist'], 'Get' + attr)())
        hist.SetLineWidth(2)
        hists.append(hist)
    ymax = max([h.GetMaximum() for h in hists] + [1e-6])
    xlo, xhi = tb.x_window(hists)
    for i, (curve, hist) in enumerate(zip(curves, hists)):
        hist.GetXaxis().SetRangeUser(xlo, xhi)
        hist.GetXaxis().SetTitle(axis_title)
        hist.GetYaxis().SetTitle('Normalised to unit area')
        hist.GetYaxis().SetTitleOffset(1.3)
        if logy:
            hist.SetMinimum(1e-4)
            hist.SetMaximum(ymax * 10 ** (1 + 0.5 * len(curves)))
        else:
            hist.SetMinimum(0)
            hist.SetMaximum(ymax * (1.0 + 0.11 * len(curves) + 0.2))
        hist.Draw(curve['option'] + (' SAME' if i else ''))
    legend = ROOT.TLegend(0.15, 0.91 - 0.055 * len(curves), 0.96, 0.91)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.028)
    for curve, hist in zip(curves, hists):
        legend.AddEntry(hist, tb.label(curve), 'lep' if curve['kind'] == 'data' else 'l')
    legend.Draw()
    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextSize(0.034)
    title.DrawLatex(0.14, 0.94, feature.replace('silvia__', '').replace('_', ' '))
    keep.extend(hists + [legend, title])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Testbeam production run directory (contains histograms/)')
    parser.add_argument('--groups', nargs='+', default=DEFAULT_GROUPS, help='<domain>:<pdg>:<energy> entries')
    parser.add_argument('--name', default=DEFAULT_NAME, help='Output sub-directory name')
    parser.add_argument('--out', type=Path, help='Output directory (default <run>/compare/<name>)')
    parser.add_argument('--features', nargs='*', help='Only these features (default: all 1D features)')
    parser.add_argument('--selection', default='selected')
    args = parser.parse_args()
    specs = [parse_spec(s) for s in args.groups]
    hist_root = args.run / 'histograms' / '1d'
    out_dir = (args.out or args.run / 'compare' / args.name) / args.selection
    out_dir.mkdir(parents=True, exist_ok=True)
    keep, pages = [], []
    with open(out_dir / 'summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['feature', 'group', 'particle_pdg', 'kind', 'domain', 'energy_gev',
                                                    'geometry', 'valid_entries', 'mean', 'std'])
        writer.writeheader()
        for feature_dir in sorted(p for p in hist_root.iterdir() if p.is_dir()):
            feature = feature_dir.name
            if args.features and feature not in args.features:
                continue
            selection_dir = feature_dir / args.selection
            if not selection_dir.is_dir():
                continue
            curves = pick(tb.load_curves(selection_dir), specs)
            if not curves:
                continue
            cfg = curves[0]['config']
            axis_title = cfg.get('axis_title') or feature
            if len(axis_title) > 60:
                axis_title = feature
            canvas = ROOT.TCanvas('cmp_' + feature, '', 900, 700)
            draw(canvas, curves, feature, axis_title, cfg.get('logy', False), keep)
            canvas.SaveAs(str(out_dir / (feature + '.pdf')))
            pages.append((feature, canvas))
            for curve in curves:
                writer.writerow(dict(feature=feature, group=curve['group'], particle_pdg=curve['pdg'],
                                     kind=curve['kind'], domain=curve['domain'], energy_gev=round(curve['energy']),
                                     geometry=tb.geometry_label(curve['geometries']), valid_entries=curve['valid'],
                                     mean=curve['hist'].GetMean(), std=curve['hist'].GetStdDev()))
    multipage = str(out_dir / 'all_features.pdf')
    for i, (feature, canvas) in enumerate(pages):
        suffix = '' if len(pages) == 1 else ('(' if i == 0 else (')' if i == len(pages) - 1 else ''))
        canvas.Print(multipage + suffix, 'Title:' + feature)
    print('%d features, groups: %s' % (len(pages), ', '.join(args.groups)))
    print('Output:', out_dir)


if __name__ == '__main__':
    main()
