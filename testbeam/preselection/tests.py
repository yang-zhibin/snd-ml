import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib


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


# f = ROOT.TFile.Open("/afs/.cern.ch/user/s/sfrankha/eos/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg37.93_Y_43.12_Z_315/feature_sndLHC.PG_11-TGeant4_digCPP.root")
# f.ls()

# tree = f.Get("sndData")
# for br in tree.GetListOfBranches():
#     print(br.GetName())
# tree.Print()
raw_data, raw_tree = open_root_file("/afs/.cern.ch/user/s/sfrankha/eos/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg37.93_Y_43.12_Z_315/feature_sndLHC.PG_11-TGeant4_digCPP.root", tree_name='sndData')
    
if raw_data is None or raw_tree is None:
    print(f"[Warning] Could not load ROOT data from: {feature_path}")

count=0
for event in raw_tree:
    if count>10:
        break
    count+=1
    # --- SciFi hits ---
    print(event.count_scifi)
    print(event.count_scifi1)
    # stations = {
    #     1: {"total": event.count_scifi1},
    #     2: {"total": event.count_scifi2},
    #     3: {"total": event.count_scifi3},
    #     4: {"total": event.count_scifi4},
    # }
    # total_hits = event.count_scifi

    # hits.Fill(total_hits)
# h_all = ROOT.TH1F("h_all", "Nombre total de hits", 100, 0, 1000000)
# tree.Draw("Digi_ScifiHits.fDetectorID >> h_all", "", "goff")