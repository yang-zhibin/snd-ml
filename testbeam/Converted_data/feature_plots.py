import argparse
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from array import array
import csv
import pandas as pd
import glob
from tqdm import tqdm
import yaml
import difflib
import math
import re
from collections import defaultdict


ROOT.gStyle.SetOptStat(0)


def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo


def open_root_file(file_path, tree_name='cbmsim', mode='READ'):
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        print(f"⚠️  Skipping corrupted or missing file: {file_path}")
        return None, None
    file = ROOT.TFile(file_path, mode)
    if not file or file.IsZombie():
        print(f"[Warning] Could not open ROOT file: {file_path}")
        return None, None

    tree = file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        print(f"[Warning] TTree '{tree_name}' not found in {file_path}")
        file.Close()
        return file, None

    return file, tree


def hist_setup(name, bins=100, m=0, M=3000, color=ROOT.kBlue):
    hist = ROOT.TH1F(name, name, bins, m, M)
    # hist.GetXaxis().SetTitle(x_title)
    # hist.GetYaxis().SetTitle(y_title)
    # hist.GetXaxis().SetTitleSize(title_size)
    # hist.GetYaxis().SetTitleSize(title_size)
    # hist.GetXaxis().SetLabelSize(label_size)
    # hist.GetYaxis().SetLabelSize(label_size)
    # hist.GetYaxis().SetTitleOffset(1.2)
    hist.SetLineColor(color)
    hist.SetLineWidth(2)
    hist.SetDirectory(0)
    return hist


# def make_plot_from_hist(real_hist, MC_hist, save_name, title="Number of hits per event", x_title="Number of hits", y_title="Event count"):
#     if real_hist.Integral() > 0:
#         real_hist.Scale(1.0 / real_hist.Integral())
#     if MC_hist.Integral() > 0:
#         MC_hist.Scale(1.0 / MC_hist.Integral())

#     c = ROOT.TCanvas("c","c", 800, 600)

#      # Ajustement du Y max
#     max_val = max(real_hist.GetMaximum(), MC_hist.GetMaximum())
#     real_hist.SetMaximum(1.2 * max_val)
#     real_hist.SetMinimum(0)

#     # Titres des axes
#     real_hist.SetTitle(title)
#     real_hist.GetXaxis().SetTitle(x_title)
#     real_hist.GetYaxis().SetTitle(y_title)
#     real_hist.GetXaxis().SetTitleSize(0.045)
#     real_hist.GetYaxis().SetTitleSize(0.045)
#     real_hist.GetXaxis().CenterTitle(True)
#     real_hist.GetYaxis().CenterTitle(True)
    
#     # Draw both histograms on the same canvas
#     real_hist.Draw("HIST")
#     MC_hist.Draw("HIST SAME")

#     # Add legend
#     legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
#     legend.AddEntry(real_hist, "Real data", "l")
#     legend.AddEntry(MC_hist, "MC data", "l")
#     legend.Draw()
    
#     c.SaveAs(f"./tmp/plots/{save_name}.pdf")
    
    
def make_plots_from_histos(histos, energy, beam_type, year, output_dir="./tmp/plots"):
    """
    Compare les histogrammes real/MC pour chaque feature et crée un plot par feature.

    Args:
        histos (dict): Dictionnaire { feature_name: {"MC": TH1F, "real": TH1F} }.
        energy (str): énergie du faisceau.
        beam_type (str): type de faisceau.
        year (str): année des données.
        output_dir (str): dossier de sauvegarde des PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    for feature, hpair in histos.items():
        real_hist = hpair["real"]
        MC_hist = hpair["MC"]

        # Normalisation si possible
        if real_hist.Integral() > 0:
            real_hist.Scale(1.0 / real_hist.Integral())
        if MC_hist.Integral() > 0:
            MC_hist.Scale(1.0 / MC_hist.Integral())

        c = ROOT.TCanvas(f"c_{feature}", f"Canvas for {feature}", 800, 600)

        # Ajustement des bornes Y
        max_val = max(real_hist.GetMaximum(), MC_hist.GetMaximum())
        real_hist.SetMaximum(1.2 * max_val)
        real_hist.SetMinimum(0)

        # Titres
        real_hist.GetXaxis().SetTitle(feature)
        real_hist.GetYaxis().SetTitle("Normalized event count")
        real_hist.GetXaxis().SetTitleSize(0.045)
        real_hist.GetYaxis().SetTitleSize(0.045)
        real_hist.GetXaxis().CenterTitle(True)
        real_hist.GetYaxis().CenterTitle(True)

        # Dessin
        real_hist.Draw("HIST")
        MC_hist.Draw("HIST SAME")

        # Légende
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(real_hist, "Real data", "l")
        legend.AddEntry(MC_hist, "MC data", "l")
        legend.Draw()

        save_name = f"{output_dir}/comparison_{feature}_{energy}_{beam_type}_{year}.pdf"
        c.SaveAs(save_name)

        # Libère le canvas
        c.Close()

    
    
def create_histograms(hist_info):
    histos = {}
    for feature, (xmin, xmax, title) in hist_info.items():
        mc_hist = hist_setup(f"{title}_MC", 100, xmin, xmax, ROOT.kBlue)

        real_hist = hist_setup(title, 100, xmin, xmax, ROOT.kRed)

        histos[feature] = {"MC": mc_hist, "real": real_hist}

    return histos


def get_features(feature_path, histos, data_type="MC"):
    raw_data, raw_tree = open_root_file(feature_path, tree_name='sndData')
    
    if raw_data is None or raw_tree is None:
        print(f"[Warning] Could not load ROOT data from: {feature_path}")
        return
    
    # count=0
    for event in raw_tree:
        # if count>100:
        #     break
        # count+=1
        
        for feature, hpair in histos.items():
            if not hasattr(event, feature):
                # Si la variable n'existe pas dans cet event, on ignore
                continue
            value = getattr(event, feature)
            try:
                hpair[data_type].Fill(value)
            except TypeError:
                # Si la variable n'est pas un float/int, on ignore
                continue
        # # --- SciFi hits ---
        # stations = {
        #     1: {"total": event.count_scifi1},
        #     2: {"total": event.count_scifi2},
        #     3: {"total": event.count_scifi3},
        #     4: {"total": event.count_scifi4},
        # }
        # total_hits = event.count_scifi

        # hits.Fill(total_hits)
        
        # hits_station_1.Fill(stations[1]["total"])
        # hits_station_2.Fill(stations[2]["total"])
        # hits_station_3.Fill(stations[3]["total"])
        # hits_station_4.Fill(stations[4]["total"])
        
    raw_data.Close()


def main():
    parser = argparse.ArgumentParser(description="Test script with plot generation.")
    # parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    parser.add_argument("-b", "--beam_type", dest="beam_type", required=True, help="Type of beam (e.g., proton, kaon)")
    parser.add_argument("-e", "--energy", dest="energy", required=True, help="Beam energy in GeV")
    parser.add_argument("-y", "--year", dest="year", help="year", required=True)

    args = parser.parse_args()

    beam_type = args.beam_type
    energy = args.energy
    year = args.year

    print(f"Beam type: {beam_type}")
    print(f"Energy: {energy}")
    
    
    if year == "2023":
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/real_data_testbeam_June2023_H8_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/MC_data_testbeam2023_metadata.csv"
        MC_name = os.path.basename(MC_file)
    elif year == "2024":
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/real_data_testbeam_24_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/MC_data_testbeam2024_metadata.csv"
        MC_name = os.path.basename(MC_file)
    else:
        print('Unrecognized year')
        return 0
    
    file_exists = os.path.isfile(real_file)
    
    if (not file_exists):
        print(f'{real_file} does not exist, please first generate it.')
        return 1
    
    file_exists = os.path.isfile(MC_file)
    if (not file_exists):
        print(f'{MC_file} does not exist, please first generate it.')
        return 1
    
    # data_type, particle_subfolder = extract_info(csv_name)
    # print(data_type, particle_subfolder)
    
    real_df = pd.read_csv(real_file)
    MC_df = pd.read_csv(MC_file)

    real_group = real_df[(real_df["beam_energy"] == energy) & (real_df["beam_type"] == beam_type)]
    MC_group = MC_df[(MC_df["beam_energy"] == energy) & (MC_df["beam_type"] == beam_type)]

    # print(f'Number of real files for {year} {energy} {beam_type}: {len(real_group)}')
    # print(f'Number of MC files for {year} {energy} {beam_type}: {len(MC_group)}')
    
    # return
    
    hist_info = {
        "px": (0, 2, f"{year} px distribution for {energy} {beam_type}"),
        "py": (0, 2, f"{year} py distribution for {energy} {beam_type}"),
        "pz": (180, 182, f"{year} pz distribution for {energy} {beam_type}"),
        "x": (-45, -31, f"{year} x distribution for {energy} {beam_type}"),
        "y": (37, 51, f"{year} y distribution for {energy} {beam_type}"),
        "z": (300, 450, f"{year} z distribution for {energy} {beam_type}"),
        'count_scifi1':  (0, 1000, f"{year} SciFi hits in station 1 for {energy} {beam_type}"),
        'count_scifi2':  (0, 1000, f"{year} SciFi hits in station 2 for {energy} {beam_type}"),
        'count_scifi3':  (0, 1000, f"{year} SciFi hits in station 3 for {energy} {beam_type}"),
        'count_scifi4':  (0, 1000, f"{year} SciFi hits in station 4 for {energy} {beam_type}"),
        'count_scifi':  (0, 3000, f"{year} SciFi hits for {energy} {beam_type}"),
        'avg_scifi1_x': (-45, -31, f"{year} average x in station 1 for {energy} {beam_type}"),
        'avg_scifi1_y': (37, 51, f"{year} average y in station 1 for {energy} {beam_type}"),
        'avg_scifi2_x': (-45, -31, f"{year} average x in station 2 for {energy} {beam_type}"),
        'avg_scifi2_y': (37, 51, f"{year} average y in station 2 for {energy} {beam_type}"),
        'avg_scifi3_x': (-45, -31, f"{year} average x in station 3 for {energy} {beam_type}"),
        'avg_scifi3_y': (37, 51, f"{year} average y in station 3 for {energy} {beam_type}"),
        'avg_scifi4_x': (-45, -31, f"{year} average x in station 4 for {energy} {beam_type}"),
        'avg_scifi4_y': (37, 51, f"{year} average y in station 4 for {energy} {beam_type}"),
        'centroid_scifi1_x': (-100, 0, f"{year} centroid x in station 1 for {energy} {beam_type}"),
        'centroid_scifi1_y': (0, 200, f"{year} centroid y in station 1 for {energy} {beam_type}"),
        'centroid_scifi2_x': (-100, 0, f"{year} centroid x in station 2 for {energy} {beam_type}"),
        'centroid_scifi2_y': (0, 200, f"{year} centroid y in station 2 for {energy} {beam_type}"),
        'centroid_scifi3_x': (-100, 0, f"{year} centroid x in station 3 for {energy} {beam_type}"),
        'centroid_scifi3_y': (0, 200, f"{year} centroid y in station 3 for {energy} {beam_type}"),
        'centroid_scifi4_x': (-100, 0, f"{year} centroid x in station 4 for {energy} {beam_type}"),
        'centroid_scifi4_y': (0, 200, f"{year} centroid y in station 4 for {energy} {beam_type}"),
        'density_scifi1': (0, 2000, f"{year} hit density in station 1 for {energy} {beam_type}"),
        'density_scifi2': (0, 2000, f"{year} hit density in station 2 for {energy} {beam_type}"),
        'density_scifi3': (0, 2000, f"{year} hit density in station 3 for {energy} {beam_type}"),
        'density_scifi4': (0, 2000, f"{year} hit density in station 4 for {energy} {beam_type}"),
        'showerTagged': (0, 3, f"{year} hit density in station 1 for {energy} {beam_type}"),
        'showerStartStation': (-1, 5, f"{year} hit density in station 1 for {energy} {beam_type}"),
        "avgPos_slope_x": (-100, 100, f"{year} average x position of the slope for {energy} {beam_type}"),
        "avgPos_slope_y": (-100, 100, f"{year} average y position of the slope for {energy} {beam_type}"),
        "centroid_slope_y": (-100, 100, f"{year} x centroid of the slope for {energy} {beam_type}"),
        "centroid_slope_x": (-100, 100, f"{year} y centroid of the slope for {energy} {beam_type}"),
    }
    
    histos = create_histograms(hist_info)

    for file in MC_group['feature_path']:
        get_features(file, histos, data_type="MC")

    print(f'MC data done, now real for {year} {energy} {beam_type}')
    
    
    # count = 0
    for file in real_group['feature_path']:
        # count += 1
        get_features(file, histos, data_type="real")
        # if count >= 5:
        #     break

    print(f'real data done, now plotting for {year} {energy} {beam_type}')


    make_plots_from_histos(histos, energy, beam_type, year)


if __name__ == "__main__":
    main()