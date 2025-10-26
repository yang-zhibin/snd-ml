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


def make_plot_from_hist(real_hist, MC_hist, save_name, title="Number of hits per event", x_title="Number of hits", y_title="Event count"):
    if real_hist.Integral() > 0:
        real_hist.Scale(1.0 / real_hist.Integral())
    if MC_hist.Integral() > 0:
        MC_hist.Scale(1.0 / MC_hist.Integral())

    c = ROOT.TCanvas("c","c", 800, 600)

     # Ajustement du Y max
    max_val = max(real_hist.GetMaximum(), MC_hist.GetMaximum())
    real_hist.SetMaximum(1.2 * max_val)
    real_hist.SetMinimum(0)

    # Titres des axes
    real_hist.SetTitle(title)
    real_hist.GetXaxis().SetTitle(x_title)
    real_hist.GetYaxis().SetTitle(y_title)
    real_hist.GetXaxis().SetTitleSize(0.045)
    real_hist.GetYaxis().SetTitleSize(0.045)
    real_hist.GetXaxis().CenterTitle(True)
    real_hist.GetYaxis().CenterTitle(True)
    
    # Draw both histograms on the same canvas
    real_hist.Draw("HIST")
    MC_hist.Draw("HIST SAME")

    # Add legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hist, "Real data", "l")
    legend.AddEntry(MC_hist, "MC data", "l")
    legend.Draw()
    
    c.SaveAs(f"./tmp/plots/{save_name}.pdf")


def get_features(feature_path, hits, hits_station_1, hits_station_2, hits_station_3, hits_station_4):
    raw_data, raw_tree = open_root_file(feature_path, tree_name='sndData')
    
    if raw_data is None or raw_tree is None:
        print(f"[Warning] Could not load ROOT data from: {feature_path}")
        return
    
    # count=0
    for event in raw_tree:
        # if count>100:
        #     break
        # count+=1
        # --- SciFi hits ---
        stations = {
            1: {"total": event.count_scifi1},
            2: {"total": event.count_scifi2},
            3: {"total": event.count_scifi3},
            4: {"total": event.count_scifi4},
        }
        total_hits = event.count_scifi

        hits.Fill(total_hits)
        
        hits_station_1.Fill(stations[1]["total"])
        hits_station_2.Fill(stations[2]["total"])
        hits_station_3.Fill(stations[3]["total"])
        hits_station_4.Fill(stations[4]["total"])
        
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
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/real_data_testbeam_June2023_H8_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/MC_data_testbeam2023_updated_metadata.csv"
        MC_name = os.path.basename(MC_file)
    elif year == "2024":
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/real_data_testbeam_24_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/metadata/updated/MC_data_testbeam2024_updated_metadata.csv"
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

    real_hits = hist_setup(f'{year} Real data SciFi hits for {energy} {beam_type}', 100, 0, 3000, ROOT.kRed)
    MC_hits = hist_setup(f'{year} MC data SciFi hits for {energy} {beam_type}', 100, 0, 3000, ROOT.kBlue)

    real_hit_station_1 = hist_setup(f'{year} Real data hit in station 1 for {energy} {beam_type}', 100, 0, 1000, ROOT.kRed)
    MC_hit_station_1 = hist_setup(f'{year} MC data hit in station 1 for {energy} {beam_type}', 100, 0, 1000, ROOT.kBlue)

    real_hit_station_2 = hist_setup(f'{year} Real data hit in station 2 for {energy} {beam_type}', 100, 0, 1000, ROOT.kRed)
    MC_hit_station_2 = hist_setup(f'{year} MC data hit in station 2 for {energy} {beam_type}', 100, 0, 1000, ROOT.kBlue)    

    real_hit_station_3 = hist_setup(f'{year} Real data hit in station 3 for {energy} {beam_type}', 100, 0, 1000, ROOT.kRed)
    MC_hit_station_3 = hist_setup(f'{year} MC data hit in station 3 for {energy} {beam_type}', 100, 0, 1000, ROOT.kBlue)

    real_hit_station_4 = hist_setup(f'{year} Real data hit in station 4 for {energy} {beam_type}', 100, 0, 1000, ROOT.kRed)
    MC_hit_station_4 = hist_setup(f'{year} MC data hit in station 4 for {energy} {beam_type}', 100, 0, 1000, ROOT.kBlue)


    for file in MC_group['feature_path']:
        get_features(file, MC_hits, MC_hit_station_1, MC_hit_station_2, MC_hit_station_3, MC_hit_station_4)

    print(f'MC data done, now real for {year} {energy} {beam_type}')
    
    
    # count = 0
    for file in real_group['feature_path']:
        # count += 1
        get_features(file, real_hits, real_hit_station_1, real_hit_station_2, real_hit_station_3, real_hit_station_4)
        # if count >= 5:
        #     break

    print(f'real data done, now plotting for {year} {energy} {beam_type}')


    make_plot_from_hist(real_hits, MC_hits, title=f"{year} SciFi hits for {energy} {beam_type}", x_title="Number of hits", y_title="Normalized event count", save_name=f"comparison_hits_{energy}_{beam_type}_{year}")

    make_plot_from_hist(real_hit_station_1, MC_hit_station_1, title=f"{year} SciFi hits in station 1 for {energy} {beam_type}", x_title="Number of hits in station 1", y_title="Normalized event count", save_name=f"comparison_hits_station_1_{energy}_{beam_type}_{year}")
    
    make_plot_from_hist(real_hit_station_2, MC_hit_station_2, title=f"{year} SciFi hits in station 2 for {energy} {beam_type}", x_title="Number of hits in station 2", y_title="Normalized event count", save_name=f"comparison_hits_station_2_{energy}_{beam_type}_{year}")
    
    make_plot_from_hist(real_hit_station_3, MC_hit_station_3, title=f"{year} SciFi hits in station 3 for {energy} {beam_type}", x_title="Number of hits in station 3", y_title="Normalized event count", save_name=f"comparison_hits_station_3_{energy}_{beam_type}_{year}")
    
    make_plot_from_hist(real_hit_station_4, MC_hit_station_4, title=f"{year} SciFi hits in station 4 for {energy} {beam_type}", x_title="Number of hits in station 4", y_title="Normalized event count", save_name=f"comparison_hits_station_4_{energy}_{beam_type}_{year}")


if __name__ == "__main__":
    main()