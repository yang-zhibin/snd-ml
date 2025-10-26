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


# def hist_setup(hist, x_title, y_title, title_size=0.05, label_size=0.04, color):
#     hist.GetXaxis().SetTitle(x_title)
#     hist.GetYaxis().SetTitle(y_title)
#     hist.GetXaxis().SetTitleSize(title_size)
#     hist.GetYaxis().SetTitleSize(title_size)
#     hist.GetXaxis().SetLabelSize(label_size)
#     hist.GetYaxis().SetLabelSize(label_size)
#     hist.GetYaxis().SetTitleOffset(1.2)
#     hist.SetLineColor(color)
#     hist.SetLineWidth(2)
#     hist.SetDirectory(0)
    
    
def canva_setup(can, logx=False, logy=False):
    can.SetLogx(logx)
    can.SetLogy(logy)
    can.SetGridx()
    can.SetGridy()
    can.SetLeftMargin(0.15)
    can.SetRightMargin(0.05)
    can.SetBottomMargin(0.15)
    can.SetTopMargin(0.1)
    can.SetTickx()
    can.SetTicky()
    return can

def get_features(digi_path, geo_path, hits_hist, hit_time_hist, hit_qdc_hist, hits_horizontal_hist, hits_vertical_hist, hits_station_1_horizontal, hits_station_1_vertical, hits_station_2_horizontal, hits_station_2_vertical, hits_station_3_horizontal, hits_station_3_vertical, hits_station_4_horizontal, hits_station_4_vertical, hits_station_1, hits_station_2, hits_station_3, hits_station_4):
    snd_geo = setup_geometry(geo_path)
    raw_data, raw_tree = open_root_file(digi_path)
    
    if raw_data is None or raw_tree is None:
        print(f"[Warning] Could not load ROOT data from: {digi_path}")
        return
    
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    count=0
    for event in raw_tree:
        # if count>100:
        #     break
        # count+=1
        # --- SciFi hits ---
        vertical_hits = 0
        horizontal_hits = 0
        stations = {
            1: {"total": 0, "vertical": 0, "horizontal": 0},
            2: {"total": 0, "vertical": 0, "horizontal": 0},
            3: {"total": 0, "vertical": 0, "horizontal": 0},
            4: {"total": 0, "vertical": 0, "horizontal": 0},
        }
        for aHit in event.Digi_ScifiHits:
            if not aHit.isValid():
                continue
            detID = aHit.GetDetectorID()
            station = (detID // 1000000)
            # print(station)
            Scifi.GetSiPMPosition(detID, A, B)
            
            stations[station]["total"] += 1
            
            if aHit.isVertical():
                vertical_hits += 1
                stations[station]["vertical"] += 1
            else:
                horizontal_hits += 1
                stations[station]["horizontal"] += 1
            # qdc = sum(aHit.GetSignal(i) for i in range(aHit.GetnSiPMs()))
            time = aHit.GetTime()

            # extraire le plan à partir de detID :
            # plane = detID // 1000  # dépend de ta géométrie exacte
            # plane_hits[plane] = plane_hits.get(plane, 0) + 1
            
            max_QDC = 200 * 16
            this_qdc = 0
            ns = max(1,aHit.GetnSides())
            for side in range(ns):
                for m in  range(aHit.GetnSiPMs()):
                    qdc = aHit.GetSignal(m+side*aHit.GetnSiPMs())
                    if not qdc < 0:
                        this_qdc += qdc
            if this_qdc > max_QDC :
                this_qdc = max_QDC

            hit_qdc_hist.Fill(this_qdc)
            hit_time_hist.Fill(time)

        hits_hist.Fill(vertical_hits + horizontal_hits)
            
        hits_horizontal_hist.Fill(horizontal_hits)
        hits_vertical_hist.Fill(vertical_hits)
        
        hits_station_1.Fill(stations[1]["total"])
        hits_station_2.Fill(stations[2]["total"])
        hits_station_3.Fill(stations[3]["total"])
        hits_station_4.Fill(stations[4]["total"])
        hits_station_1_vertical.Fill(stations[1]["vertical"])
        hits_station_2_vertical.Fill(stations[2]["vertical"])
        hits_station_3_vertical.Fill(stations[3]["vertical"])
        hits_station_4_vertical.Fill(stations[4]["vertical"])
        hits_station_1_horizontal.Fill(stations[1]["horizontal"])
        hits_station_2_horizontal.Fill(stations[2]["horizontal"])
        hits_station_3_horizontal.Fill(stations[3]["horizontal"])
        hits_station_4_horizontal.Fill(stations[4]["horizontal"])
        
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
    print(f"Energy: {energy} GeV")
    
    
    if year == "2023":
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/snakemake/metadata/updated/real_data_testbeam_June2023_H8_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/snakemake/metadata/updated/MC_data_testbeam2023_updated_metadata.csv"
        MC_name = os.path.basename(MC_file)
    elif year == "2024":
        real_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/snakemake/metadata/updated/real_data_testbeam_24_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "/afs/cern.ch/work/s/sfrankha/snd-ml/testbeam/snakemake/metadata/updated/MC_data_testbeam2024_updated_metadata.csv"
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

    real_hits = ROOT.TH1F(f'{year} Real data number of hits for {energy} {beam_type}', f'{year} Real data number of hits for {energy} {beam_type}', 100, 0, 3000)
    MC_hits = ROOT.TH1F(f'{year} MC data number of hits for {energy} {beam_type}', f'{year} MC data number of hits for {energy} {beam_type}', 100, 0, 3000)
    MC_hits.SetLineColor(ROOT.kBlue)   
    MC_hits.SetLineWidth(2) 
    real_hits.SetLineColor(ROOT.kRed)  
    real_hits.SetLineWidth(2)
    real_hits.SetDirectory(0)
    MC_hits.SetDirectory(0)

    real_hits_horizontal = ROOT.TH1F(f'{year} Real data number of horizontal hits for {energy} {beam_type}', f'{year} Real data number of horizontal hits for {energy} {beam_type}', 100, 0, 1500)
    MC_hits_horizontal = ROOT.TH1F(f'{year} MC data number of horizontal hits for {energy} {beam_type}', f'{year} MC data number of horizontal hits for {energy} {beam_type}', 100, 0, 1500)
    MC_hits_horizontal.SetLineColor(ROOT.kBlue)   
    MC_hits_horizontal.SetLineWidth(2) 
    real_hits_horizontal.SetLineColor(ROOT.kRed)  
    real_hits_horizontal.SetLineWidth(2)
    real_hits_horizontal.SetDirectory(0)
    MC_hits_horizontal.SetDirectory(0)
    
    real_hits_vertical = ROOT.TH1F(f'{year} Real data number of vertical hits for {energy} {beam_type}', f'{year} Real data number of vertical hits for {energy} {beam_type}', 100, 0, 1500)
    MC_hits_vertical = ROOT.TH1F(f'{year} MC data number of vertical hits for {energy} {beam_type}', f'{year} MC data number of vertical hits for {energy} {beam_type}', 100, 0, 1500)
    MC_hits_vertical.SetLineColor(ROOT.kGreen+2)   
    MC_hits_vertical.SetLineWidth(2)
    real_hits_vertical.SetLineColor(ROOT.kOrange+7)  
    real_hits_vertical.SetLineWidth(2)
    real_hits_vertical.SetDirectory(0)
    MC_hits_vertical.SetDirectory(0)        

    real_hit_time = ROOT.TH1F(f'{year} Real data hit times for {energy} {beam_type}', f'{year} Real data hit times for {energy} {beam_type}', 100, 0, 50)
    MC_hit_time = ROOT.TH1F(f'{year} MC data hit times for {energy} {beam_type}', f'{year} MC data hit times for {energy} {beam_type}', 100, 0, 50)
    MC_hit_time.SetLineColor(ROOT.kBlue)   
    MC_hit_time.SetLineWidth(2) 
    real_hit_time.SetLineColor(ROOT.kRed)
    real_hit_time.SetLineWidth(2)
    real_hit_time.SetDirectory(0)
    MC_hit_time.SetDirectory(0)
    
    real_hit_qdc = ROOT.TH1F(f'{year} Real data hit QDC for {energy} {beam_type}', f'{year} Real data hit QDC for {energy} {beam_type}', 100, 0, 100)
    MC_hit_qdc = ROOT.TH1F(f'{year} MC data hit QDC for {energy} {beam_type}', f'{year} MC data hit QDC for {energy} {beam_type}', 100, 0, 100)
    MC_hit_qdc.SetLineColor(ROOT.kBlue)
    MC_hit_qdc.SetLineWidth(2) 
    real_hit_qdc.SetLineColor(ROOT.kRed)  
    real_hit_qdc.SetLineWidth(2)
    real_hit_qdc.SetDirectory(0)
    MC_hit_qdc.SetDirectory(0)

    real_hit_station_1 = ROOT.TH1F(f'{year} Real data hit in station 1 for {energy} {beam_type}', f'{year} Real data hit in station 1 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1 = ROOT.TH1F(f'{year} MC data hit in station 1 for {energy} {beam_type}', f'{year} MC data hit in station 1 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1.SetLineColor(ROOT.kBlue)
    MC_hit_station_1.SetLineWidth(2) 
    real_hit_station_1.SetLineColor(ROOT.kRed)  
    real_hit_station_1.SetLineWidth(2)
    real_hit_station_1.SetDirectory(0)
    MC_hit_station_1.SetDirectory(0)

    real_hit_station_2 = ROOT.TH1F(f'{year} Real data hit in station 2 for {energy} {beam_type}', f'{year} Real data hit in station 2 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2 = ROOT.TH1F(f'{year} MC data hit in station 2 for {energy} {beam_type}', f'{year} MC data hit in station 2 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2.SetLineColor(ROOT.kBlue)
    MC_hit_station_2.SetLineWidth(2)
    real_hit_station_2.SetLineColor(ROOT.kRed)
    real_hit_station_2.SetLineWidth(2)
    real_hit_station_2.SetDirectory(0)
    MC_hit_station_2.SetDirectory(0)

    real_hit_station_3 = ROOT.TH1F(f'{year} Real data hit in station 3 for {energy} {beam_type}', f'{year} Real data hit in station 3 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3 = ROOT.TH1F(f'{year} MC data hit in station 3 for {energy} {beam_type}', f'{year} MC data hit in station 3 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3.SetLineColor(ROOT.kBlue)
    MC_hit_station_3.SetLineWidth(2)
    real_hit_station_3.SetLineColor(ROOT.kRed)
    real_hit_station_3.SetLineWidth(2)
    real_hit_station_3.SetDirectory(0)
    MC_hit_station_3.SetDirectory(0)

    real_hit_station_4 = ROOT.TH1F(f'{year} Real data hit in station 4 for {energy} {beam_type}', f'{year} Real data hit in station 4 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4 = ROOT.TH1F(f'{year} MC data hit in station 4 for {energy} {beam_type}', f'{year} MC data hit in station 4 for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4.SetLineColor(ROOT.kBlue)
    MC_hit_station_4.SetLineWidth(2)
    real_hit_station_4.SetLineColor(ROOT.kRed)
    real_hit_station_4.SetLineWidth(2)
    real_hit_station_4.SetDirectory(0)
    MC_hit_station_4.SetDirectory(0)

    real_hit_station_1_vertical = ROOT.TH1F(f'{year} Real data hit in station 1 vertical plane for {energy} {beam_type}', f'{year} Real data hit in station 1 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1_vertical = ROOT.TH1F(f'{year} MC data hit in station 1 vertical plane for {energy} {beam_type}', f'{year} MC data hit in station 1 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1_vertical.SetLineColor(ROOT.kBlue)
    MC_hit_station_1_vertical.SetLineWidth(2)
    real_hit_station_1_vertical.SetLineColor(ROOT.kRed)
    real_hit_station_1_vertical.SetLineWidth(2)
    real_hit_station_1_vertical.SetDirectory(0)
    MC_hit_station_1_vertical.SetDirectory(0)

    real_hit_station_2_vertical = ROOT.TH1F(f'{year} Real data hit in station 2 vertical plane for {energy} {beam_type}', f'{year} Real data hit in station 2 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2_vertical = ROOT.TH1F(f'{year} MC data hit in station 2 vertical plane for {energy} {beam_type}', f'{year} MC data hit in station 2 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2_vertical.SetLineColor(ROOT.kBlue)
    MC_hit_station_2_vertical.SetLineWidth(2)
    real_hit_station_2_vertical.SetLineColor(ROOT.kRed)
    real_hit_station_2_vertical.SetLineWidth(2)
    real_hit_station_2_vertical.SetDirectory(0)
    MC_hit_station_2_vertical.SetDirectory(0)

    real_hit_station_3_vertical = ROOT.TH1F(f'{year} Real data hit in station 3 vertical plane for {energy} {beam_type}', f'{year} Real data hit in station 3 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3_vertical = ROOT.TH1F(f'{year} MC data hit in station 3 vertical plane for {energy} {beam_type}', f'{year} MC data hit in station 3 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3_vertical.SetLineColor(ROOT.kBlue)
    MC_hit_station_3_vertical.SetLineWidth(2)
    real_hit_station_3_vertical.SetLineColor(ROOT.kRed)
    real_hit_station_3_vertical.SetLineWidth(2)
    real_hit_station_3_vertical.SetDirectory(0)
    MC_hit_station_3_vertical.SetDirectory(0)

    real_hit_station_4_vertical = ROOT.TH1F(f'{year} Real data hit in station 4 vertical plane for {energy} {beam_type}', f'{year} Real data hit in station 4 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4_vertical = ROOT.TH1F(f'{year} MC data hit in station 4 vertical plane for {energy} {beam_type}', f'{year} MC data hit in station 4 vertical plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4_vertical.SetLineColor(ROOT.kBlue)
    MC_hit_station_4_vertical.SetLineWidth(2)
    real_hit_station_4_vertical.SetLineColor(ROOT.kRed)
    real_hit_station_4_vertical.SetLineWidth(2)
    real_hit_station_4_vertical.SetDirectory(0)
    MC_hit_station_4_vertical.SetDirectory(0)

    real_hit_station_1_horizontal = ROOT.TH1F(f'{year} Real data hit in station 1 horizontal plane for {energy} {beam_type}', f'{year} Real data hit in station 1 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1_horizontal = ROOT.TH1F(f'{year} MC data hit in station 1 horizontal plane for {energy} {beam_type}', f'{year} MC data hit in station 1 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_1_horizontal.SetLineColor(ROOT.kBlue)
    MC_hit_station_1_horizontal.SetLineWidth(2)
    real_hit_station_1_horizontal.SetLineColor(ROOT.kRed)
    real_hit_station_1_horizontal.SetLineWidth(2)
    real_hit_station_1_horizontal.SetDirectory(0)
    MC_hit_station_1_horizontal.SetDirectory(0)

    real_hit_station_2_horizontal = ROOT.TH1F(f'{year} Real data hit in station 2 horizontal plane for {energy} {beam_type}', f'{year} Real data hit in station 2 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2_horizontal = ROOT.TH1F(f'{year} MC data hit in station 2 horizontal plane for {energy} {beam_type}', f'{year} MC data hit in station 2 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_2_horizontal.SetLineColor(ROOT.kBlue)
    MC_hit_station_2_horizontal.SetLineWidth(2)
    real_hit_station_2_horizontal.SetLineColor(ROOT.kRed)
    real_hit_station_2_horizontal.SetLineWidth(2)
    real_hit_station_2_horizontal.SetDirectory(0)
    MC_hit_station_2_horizontal.SetDirectory(0)

    real_hit_station_3_horizontal = ROOT.TH1F(f'{year} Real data hit in station 3 horizontal plane for {energy} {beam_type}', f'{year} Real data hit in station 3 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3_horizontal = ROOT.TH1F(f'{year} MC data hit in station 3 horizontal plane for {energy} {beam_type}', f'{year} MC data hit in station 3 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_3_horizontal.SetLineColor(ROOT.kBlue)
    MC_hit_station_3_horizontal.SetLineWidth(2)
    real_hit_station_3_horizontal.SetLineColor(ROOT.kRed)
    real_hit_station_3_horizontal.SetLineWidth(2)
    real_hit_station_3_horizontal.SetDirectory(0)
    MC_hit_station_3_horizontal.SetDirectory(0)

    real_hit_station_4_horizontal = ROOT.TH1F(f'{year} Real data hit in station 4 horizontal plane for {energy} {beam_type}', f'{year} Real data hit in station 4 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4_horizontal = ROOT.TH1F(f'{year} MC data hit in station 4 horizontal plane for {energy} {beam_type}', f'{year} MC data hit in station 4 horizontal plane for {energy} {beam_type}', 100, 0, 1000)
    MC_hit_station_4_horizontal.SetLineColor(ROOT.kBlue)
    MC_hit_station_4_horizontal.SetLineWidth(2)
    real_hit_station_4_horizontal.SetLineColor(ROOT.kRed)
    real_hit_station_4_horizontal.SetLineWidth(2)
    real_hit_station_4_horizontal.SetDirectory(0)
    MC_hit_station_4_horizontal.SetDirectory(0)


    for file, geo_file in zip(MC_group['digi_path'], MC_group['geo_path']):
        
        get_features(file, geo_file, MC_hits, MC_hit_time, MC_hit_qdc, MC_hits_horizontal, MC_hits_vertical, MC_hit_station_1_horizontal, MC_hit_station_1_vertical, MC_hit_station_2_horizontal, MC_hit_station_2_vertical, MC_hit_station_3_horizontal, MC_hit_station_3_vertical, MC_hit_station_4_horizontal, MC_hit_station_4_vertical, MC_hit_station_1, MC_hit_station_2, MC_hit_station_3, MC_hit_station_4)

    print(f'MC data done, now real for {year} {energy} {beam_type}')
    
    
    count = 0
    for file, geo_file in zip(real_group['digi_path'], real_group['geo_path']):
        count += 1
        get_features(file, geo_file, real_hits, real_hit_time, real_hit_qdc, real_hits_horizontal, real_hits_vertical, real_hit_station_1_horizontal, real_hit_station_1_vertical, real_hit_station_2_horizontal, real_hit_station_2_vertical, real_hit_station_3_horizontal, real_hit_station_3_vertical, real_hit_station_4_horizontal, real_hit_station_4_vertical, real_hit_station_1, real_hit_station_2, real_hit_station_3, real_hit_station_4)
        if count >= 5:
            break

    print(f'real data done, now plotting for {year} {energy} {beam_type}')

    # comparison.append({
    # "energy": energy,
    # "type": beam_type,
    # "real_hits": np.concatenate(real_hits),
    # "mc_hits": np.concatenate(MC_hits),
    # })
    
    # # Exemple : un run réel vs un fichier MC
    # real_file = "path/to/real.root"
    # mc_file   = "path/to/mc.root"

    # h_real = make_hist_from_file(real_file, "Real Data", ROOT.kRed)
    # h_mc   = make_hist_from_file(mc_file,   "MC Data",   ROOT.kBlue)
    
    # Normalisation (aire = 1)
    if real_hits.Integral() > 0:
        real_hits.Scale(1.0 / real_hits.Integral())
    if MC_hits.Integral() > 0:
        MC_hits.Scale(1.0 / MC_hits.Integral())

    c = ROOT.TCanvas(f"{year} SciFi hits for {energy} {beam_type}", f"{year} SciFi hits for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hits.Draw("HIST")
    MC_hits.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hits, "Real hits", "l")
    legend.AddEntry(MC_hits, "MC hits", "l")
    legend.Draw()

    c.SaveAs(f"./tmp/plots/comparison_hits_{energy}_{beam_type}_{year}.pdf")
    
    
    if real_hits_horizontal.Integral() > 0:
        real_hits_horizontal.Scale(1.0 / real_hits_horizontal.Integral())
    if MC_hits_horizontal.Integral() > 0:
        MC_hits_horizontal.Scale(1.0 / MC_hits_horizontal.Integral())
        
    if real_hits_vertical.Integral() > 0:
        real_hits_vertical.Scale(1.0 / real_hits_vertical.Integral())
    if MC_hits_vertical.Integral() > 0:
        MC_hits_vertical.Scale(1.0 / MC_hits_vertical.Integral())

    c_pol = ROOT.TCanvas(f"{year} SciFi hits per plane for {energy} {beam_type}", f"{year} SciFi hits per plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hits_vertical.Draw("HIST")
    MC_hits_vertical.Draw("HIST SAME")
    real_hits_horizontal.Draw("HIST SAME")
    MC_hits_horizontal.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hits_vertical, "Real hits vertical", "l")
    legend.AddEntry(MC_hits_vertical, "MC hits vertical", "l")
    legend.AddEntry(real_hits_horizontal, "Real hits horizontal", "l")
    legend.AddEntry(MC_hits_horizontal, "MC hits horizontal", "l")
    legend.Draw()

    c_pol.SaveAs(f"./tmp/plots/comparison_hits_per_plane_{energy}_{beam_type}_{year}.pdf")
    
    
    if real_hit_time.Integral() > 0:
        real_hit_time.Scale(1.0 / real_hit_time.Integral())
    if MC_hit_time.Integral() > 0:
        MC_hit_time.Scale(1.0 / MC_hit_time.Integral())

    c_time = ROOT.TCanvas(f"{year} SciFi hit times for {energy} {beam_type}", f"{year} SciFi hit times for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_time.Draw("HIST")
    MC_hit_time.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_time, "Real hit time", "l")
    legend.AddEntry(MC_hit_time, "MC hit time", "l")
    legend.Draw()

    c_time.SaveAs(f"./tmp/plots/comparison_hit_times_{energy}_{beam_type}_{year}.pdf")

    if real_hit_qdc.Integral() > 0:
        real_hit_qdc.Scale(1.0 / real_hit_qdc.Integral())
    if MC_hit_qdc.Integral() > 0:
        MC_hit_qdc.Scale(1.0 / MC_hit_qdc.Integral())

    c_qdc = ROOT.TCanvas(f"{year} SciFi hit QDC for {energy} {beam_type}", f"{year} SciFi hit QDC for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_qdc.Draw("HIST")
    MC_hit_qdc.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_qdc, "Real hit QDC", "l")
    legend.AddEntry(MC_hit_qdc, "MC hit QDC", "l")
    legend.Draw()

    c_qdc.SaveAs(f"./tmp/plots/comparison_hit_qdc_{energy}_{beam_type}_{year}.pdf")
    
    
    if real_hit_station_1.Integral() > 0:
        real_hit_station_1.Scale(1.0 / real_hit_station_1.Integral())
    if MC_hit_station_1.Integral() > 0:
        MC_hit_station_1.Scale(1.0 / MC_hit_station_1.Integral())

    c_station_1 = ROOT.TCanvas(f"{year} SciFi hits in station 1 for {energy} {beam_type}", f"{year} SciFi hits in station 1 for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_1.Draw("HIST")
    MC_hit_station_1.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_1, "Real hits", "l")
    legend.AddEntry(MC_hit_station_1, "MC hits", "l")
    legend.Draw()

    c_station_1.SaveAs(f"./tmp/plots/comparison_hits_station_1_{energy}_{beam_type}_{year}.pdf")
    
    #do the same for station 2, 3, 4 and horizontal/vertical planes
    if real_hit_station_2.Integral() > 0:
        real_hit_station_2.Scale(1.0 / real_hit_station_2.Integral())
    if MC_hit_station_2.Integral() > 0:
        MC_hit_station_2.Scale(1.0 / MC_hit_station_2.Integral())

    c_station_2 = ROOT.TCanvas(f"{year} SciFi hits in station 2 for {energy} {beam_type}", f"{year} SciFi hits in station 2 for {energy} {beam_type}", 800, 600)
    real_hit_station_2.Draw("HIST")
    MC_hit_station_2.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_2 , "Real hits", "l")
    legend.AddEntry(MC_hit_station_2, "MC hits", "l")
    legend.Draw()

    c_station_2.SaveAs(f"./tmp/plots/comparison_hits_station_2_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_3.Integral() > 0:
        real_hit_station_3.Scale(1.0 / real_hit_station_3.Integral())
    if MC_hit_station_3.Integral() > 0:
        MC_hit_station_3.Scale(1.0 / MC_hit_station_3.Integral())

    c_station_3 = ROOT.TCanvas(f"{year} SciFi hits in station 3 for {energy} {beam_type}", f"{year} SciFi hits in station 3 for {energy} {beam_type}", 800, 600)
    real_hit_station_3.Draw("HIST")
    MC_hit_station_3.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_3 , "Real hits", "l")
    legend.AddEntry(MC_hit_station_3, "MC hits", "l")
    legend.Draw()

    c_station_3.SaveAs(f"./tmp/plots/comparison_hits_station_3_{energy}_{beam_type}_{year}.pdf")
    
    if real_hit_station_4.Integral() > 0:
        real_hit_station_4.Scale(1.0 / real_hit_station_4.Integral())
    if MC_hit_station_4.Integral() > 0:
        MC_hit_station_4.Scale(1.0 / MC_hit_station_4.Integral())

    c_station_4 = ROOT.TCanvas(f"{year} SciFi hits in station 4 for {energy} {beam_type}", f"{year} SciFi hits in station 4 for {energy} {beam_type}", 800, 600)
    real_hit_station_4.Draw("HIST")
    MC_hit_station_4.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_4 , "Real hits", "l")
    legend.AddEntry(MC_hit_station_4, "MC hits", "l")
    legend.Draw()

    c_station_4.SaveAs(f"./tmp/plots/comparison_hits_station_4_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_1_vertical.Integral() > 0:
        real_hit_station_1_vertical.Scale(1.0 / real_hit_station_1_vertical.Integral())
    if MC_hit_station_1_vertical.Integral() > 0:
        MC_hit_station_1_vertical.Scale(1.0 / MC_hit_station_1_vertical.Integral())

    c_station_1_vertical = ROOT.TCanvas(f"{year} SciFi hits in station 1 vertical plane for {energy} {beam_type}", f"{year} SciFi hits in station 1 vertical plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_1_vertical.Draw("HIST")
    MC_hit_station_1_vertical.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_1_vertical, "Real hits vertical", "l")
    legend.AddEntry(MC_hit_station_1_vertical, "MC hits vertical", "l")
    legend.Draw()

    c_station_1_vertical.SaveAs(f"./tmp/plots/comparison_hits_station_1_vertical_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_2_vertical.Integral() > 0:
        real_hit_station_2_vertical.Scale(1.0 / real_hit_station_2_vertical.Integral())
    if MC_hit_station_2_vertical.Integral() > 0:
        MC_hit_station_2_vertical.Scale(1.0 / MC_hit_station_2_vertical.Integral())

    c_station_2_vertical = ROOT.TCanvas(f"{year} SciFi hits in station 2 vertical plane for {energy} {beam_type}", f"{year} SciFi hits in station 2 vertical plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_2_vertical.Draw("HIST")
    MC_hit_station_2_vertical.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_2_vertical, "Real hits vertical", "l")
    legend.AddEntry(MC_hit_station_2_vertical, "MC hits vertical", "l")
    legend.Draw()

    c_station_2_vertical.SaveAs(f"./tmp/plots/comparison_hits_station_2_vertical_{energy}_{beam_type}_{year}.pdf")
    
    if real_hit_station_3_vertical.Integral() > 0:
        real_hit_station_3_vertical.Scale(1.0 / real_hit_station_3_vertical.Integral())
    if MC_hit_station_3_vertical.Integral() > 0:
        MC_hit_station_3_vertical.Scale(1.0 / MC_hit_station_3_vertical.Integral())

    c_station_3_vertical = ROOT.TCanvas(f"{year} SciFi hits in station 3 vertical plane for {energy} {beam_type}", f"{year} SciFi hits in station 3 vertical plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_3_vertical.Draw("HIST")
    MC_hit_station_3_vertical.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_3_vertical, "Real hits vertical", "l")
    legend.AddEntry(MC_hit_station_3_vertical, "MC hits vertical", "l")
    legend.Draw()

    c_station_3_vertical.SaveAs(f"./tmp/plots/comparison_hits_station_3_vertical_{energy}_{beam_type}_{year}.pdf")
    
    if real_hit_station_4_vertical.Integral() > 0:
        real_hit_station_4_vertical.Scale(1.0 / real_hit_station_4_vertical.Integral())
    if MC_hit_station_4_vertical.Integral() > 0:
        MC_hit_station_4_vertical.Scale(1.0 / MC_hit_station_4_vertical.Integral())

    c_station_4_vertical = ROOT.TCanvas(f"{year} SciFi hits in station 4 vertical plane for {energy} {beam_type}", f"{year} SciFi hits in station 4 vertical plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_4_vertical.Draw("HIST")
    MC_hit_station_4_vertical.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_4_vertical, "Real hits vertical", "l")
    legend.AddEntry(MC_hit_station_4_vertical, "MC hits vertical", "l")
    legend.Draw()

    c_station_4_vertical.SaveAs(f"./tmp/plots/comparison_hits_station_4_vertical_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_1_horizontal.Integral() > 0:
        real_hit_station_1_horizontal.Scale(1.0 / real_hit_station_1_horizontal.Integral())
    if MC_hit_station_1_horizontal.Integral() > 0:
        MC_hit_station_1_horizontal.Scale(1.0 / MC_hit_station_1_horizontal.Integral())

    c_station_1_horizontal = ROOT.TCanvas(f"{year} SciFi hits in station 1 horizontal plane for {energy} {beam_type}", f"{year} SciFi hits in station 1 horizontal plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_1_horizontal.Draw("HIST")
    MC_hit_station_1_horizontal.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_1_horizontal, "Real hits horizontal", "l")
    legend.AddEntry(MC_hit_station_1_horizontal, "MC hits horizontal", "l")
    legend.Draw()

    c_station_1_horizontal.SaveAs(f"./tmp/plots/comparison_hits_station_1_horizontal_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_2_horizontal.Integral() > 0:
        real_hit_station_2_horizontal.Scale(1.0 / real_hit_station_2_horizontal.Integral())
    if MC_hit_station_2_horizontal.Integral() > 0:
        MC_hit_station_2_horizontal.Scale(1.0 / MC_hit_station_2_horizontal.Integral())

    c_station_2_horizontal = ROOT.TCanvas(f"{year} SciFi hits in station 2 horizontal plane for {energy} {beam_type}", f"{year} SciFi hits in station 2 horizontal plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_2_horizontal.Draw("HIST")
    MC_hit_station_2_horizontal.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_2_horizontal, "Real hits horizontal", "l")
    legend.AddEntry(MC_hit_station_2_horizontal, "MC hits horizontal", "l")
    legend.Draw()

    c_station_2_horizontal.SaveAs(f"./tmp/plots/comparison_hits_station_2_horizontal_{energy}_{beam_type}_{year}.pdf")

    if real_hit_station_3_horizontal.Integral() > 0:
        real_hit_station_3_horizontal.Scale(1.0 / real_hit_station_3_horizontal.Integral())
    if MC_hit_station_3_horizontal.Integral() > 0:
        MC_hit_station_3_horizontal.Scale(1.0 / MC_hit_station_3_horizontal.Integral())

    c_station_3_horizontal = ROOT.TCanvas(f"{year} SciFi hits in station 3 horizontal plane for {energy} {beam_type}", f"{year} SciFi hits in station 3 horizontal plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_3_horizontal.Draw("HIST")
    MC_hit_station_3_horizontal.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_3_horizontal, "Real hits horizontal", "l")
    legend.AddEntry(MC_hit_station_3_horizontal, "MC hits horizontal", "l")
    legend.Draw()

    c_station_3_horizontal.SaveAs(f"./tmp/plots/comparison_hits_station_3_horizontal_{energy}_{beam_type}_{year}.pdf")
    
    if real_hit_station_4_horizontal.Integral() > 0:
        real_hit_station_4_horizontal.Scale(1.0 / real_hit_station_4_horizontal.Integral())
    if MC_hit_station_4_horizontal.Integral() > 0:
        MC_hit_station_4_horizontal.Scale(1.0 / MC_hit_station_4_horizontal.Integral())

    c_station_4_horizontal = ROOT.TCanvas(f"{year} SciFi hits in station 4 horizontal plane for {energy} {beam_type}", f"{year} SciFi hits in station 4 horizontal plane for {energy} {beam_type}", 800, 600)

    # Dessiner les deux histos sur le même canevas
    real_hit_station_4_horizontal.Draw("HIST")
    MC_hit_station_4_horizontal.Draw("HIST SAME")

    # Ajouter une légende
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(real_hit_station_4_horizontal, "Real hits horizontal", "l")
    legend.AddEntry(MC_hit_station_4_horizontal, "MC hits horizontal", "l")
    legend.Draw()

    c_station_4_horizontal.SaveAs(f"./tmp/plots/comparison_hits_station_4_horizontal_{energy}_{beam_type}_{year}.pdf")
# df_comp = pd.DataFrame(comparison)

# for i in range(len(df_comp)):
#     plt

    # # Simulate some dummy detector layer data
    # layers = np.arange(1, 11)  # 10 layers
    # # Fake energy deposition: depends on input energy + some randomness
    # energy_deposit = np.random.normal(loc=energy / 10, scale=0.5, size=len(layers))

    # # Plot
    # plt.figure()
    # plt.bar(layers, energy_deposit)
    # plt.xlabel("Detector Layer")
    # plt.ylabel("Energy Deposited (a.u.)")
    # plt.title(f"{beam} beam at {energy} GeV")
    # plt.tight_layout()

    # # Save plot
    # plot_filename = f"./plots/test_plot_{beam}_{energy}.png"
    # plt.savefig(plot_filename)
    # print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    main()