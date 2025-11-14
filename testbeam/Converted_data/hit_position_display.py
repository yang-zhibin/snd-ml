import argparse
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import os
from argparse import ArgumentParser
# import SndlhcGeo
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
    parser.add_argument("-i", "--hit_input", dest="hit_input", help="hit input", required=True)


    args = parser.parse_args()

    hit_path = args.hit_input
    
    ROOT.gROOT.ProcessLine(".L /afs/.cern.ch/user/s/sfrankha/work/snd-ml/testbeam/Converted_data/EventClass.h+")

    raw_data, raw_tree = open_root_file(hit_path, tree_name="sndData")
    
    # Define branches (assuming branch setup functions are defined)

    # ids = ROOT.Id()
    # hits = ROOT.TClonesArray("Hit")
    # branch = raw_tree.GetBranch("Hits")
    # print("Branch class name:", branch.GetClassName())

    # Create histograms for hit positions
    h_x1_horizontal = hist_setup("h_x1_horizontal", bins=100, m=-45, M=-30)
    h_x2_horizontal = hist_setup("h_x2_horizontal", bins=100, m=-45, M=-30)
    h_y1_horizontal = hist_setup("h_y1_horizontal", bins=100, m=37, M=52)
    h_y2_horizontal = hist_setup("h_y2_horizontal", bins=100, m=37, M=52)
    h_z1_horizontal = hist_setup("h_z1_horizontal", bins=100, m=300, M=400)
    h_z2_horizontal = hist_setup("h_z2_horizontal", bins=100, m=300, M=400)

    h_x1_vertical = hist_setup("h_x1_vertical", bins=100, m=-45, M=-30)
    h_x2_vertical = hist_setup("h_x2_vertical", bins=100, m=-45, M=-30)
    h_y1_vertical = hist_setup("h_y1_vertical", bins=100, m=37, M=52)
    h_y2_vertical = hist_setup("h_y2_vertical", bins=100, m=37, M=52)
    h_z1_vertical = hist_setup("h_z1_vertical", bins=100, m=300, M=400)
    h_z2_vertical = hist_setup("h_z2_vertical", bins=100, m=300, M=400)
    
    b = raw_tree.GetBranch("Hits")
    b.Print()
    
    hits_arr = ROOT.TClonesArray("Hit")
    raw_tree.SetBranchAddress("Hits", hits_arr)

    for event in raw_tree:

        for i_hit in range(hits_arr.GetEntries()):
            
            hit = hits_arr.At(i_hit)

            if hit.orientation == 1:  # Vertical hits
                h_x1_vertical.Fill(hit.x1)
                h_x2_vertical.Fill(hit.x2)
                h_y1_vertical.Fill(hit.y1)
                h_y2_vertical.Fill(hit.y2)
                h_z1_vertical.Fill(hit.z1)
                h_z2_vertical.Fill(hit.z2)
            else:  # Horizontal hits
                h_x1_horizontal.Fill(hit.x1)
                h_x2_horizontal.Fill(hit.x2)
                h_y1_horizontal.Fill(hit.y1)
                h_y2_horizontal.Fill(hit.y2)
                h_z1_horizontal.Fill(hit.z1)
                h_z2_horizontal.Fill(hit.z2)

    # Create canvas and save plots
    c = ROOT.TCanvas("c", "c", 800, 600)
    for h, name in [
        (h_x1_horizontal, "x1_horizontal"), (h_x2_horizontal, "x2_horizontal"),
        (h_y1_horizontal, "y1_horizontal"), (h_y2_horizontal, "y2_horizontal"),
        (h_z1_horizontal, "z1_horizontal"), (h_z2_horizontal, "z2_horizontal"),
        (h_x1_vertical, "x1_vertical"), (h_x2_vertical, "x2_vertical"),
        (h_y1_vertical, "y1_vertical"), (h_y2_vertical, "y2_vertical"),
        (h_z1_vertical, "z1_vertical"), (h_z2_vertical, "z2_vertical")
    ]:
        h.Draw("HIST")
        c.SaveAs(f"./tmp/plots/pos_{name}.pdf")




if __name__ == "__main__":
    main()