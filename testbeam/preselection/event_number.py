import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
from tqdm import tqdm
import difflib
import matplotlib.pyplot as plt
import math
import numpy as np
import re
from collections import defaultdict


ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)


def open_root_file(file_path, tree_name='cbmsim', mode='read'):
    file = ROOT.TFile(file_path, mode)
    tree = file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        raise RuntimeError(f"TTree '{tree_name}' not found in {file_path}")
    return file, tree


def make_hist_from_file(root_file, hist, color, zombie_files, tree_missing_files, n_events, positive_files):
    # Crée un histogramme vide
    
    # hist = ROOT.TH1F(hist_name, hist_name, 100, 0, 1000)  # binning à adapter
    # try:
    #     with uproot.open(file_path) as f:
    #         if "cbmsim" not in f:   # exemple: vérifier si le tree attendu existe
    #             print(f"Wrong tree in {file_path}")
    #             bad_files.append(file_path)
    # except Exception as e:
    #     print(f"Zombie or unreadable file: {file_path} ({e})")
    #     bad_files.append(file_path)
    try:
        f = ROOT.TFile(root_file, 'read')
        tree = f.Get('cbmsim')
        # f = ROOT.TFile.Open(root_file)
        if not f or f.IsZombie():
            print(f"[SKIP] Fichier illisible ou zombie :{root_file}")
            # zombie_files.append(root_file)
            return hist
        
        if not tree or not isinstance(tree, ROOT.TTree):
            print(f"Wrong tree in {root_file}")
            # tree_missing_files.append(root_file)
            return hist


        # scifi_hits = ROOT.TClonesArray("sndScifiHit")  #why this step and whats a "sndScifiHit" object ?
        # tree.SetBranchAddress("Digi_ScifiHits", scifi_hits)
        # Remplir avec les hits SciFi
        # n_events.append(tree.GetEntries())
        for i in tqdm(range(tree.GetEntries()), desc="Processing events"):
            tree.GetEntry(i)
            # print(dir(tree.Digi_ScifiHits))
            if i>2:
                break
            # if hasattr(tree, "Digi_ScifiHits"):
            #     # nhits = tree.Digi_ScifiHits
            #     # nhits = scifi_hits.GetEntries()
            nhits = tree.Digi_ScifiHits.GetEntries()
            print(f"Event {i}: {nhits} SciFi hits")
                
            #     # if nhits > 0:
            #     #     print(f"Event {i}: {nhits} SciFi hits")
            #     #     positive_files.append(root_file)
            #     hist.Fill(nhits)
            # else:
            #     print(f"⚠️ Branche 'Digi_ScifiHits' non trouvée pour {root_file}")

        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        f.Close()
        return hist

    except OSError as e:
        print(f"[WARNING] Impossible d'ouvrir {root_file}: {e}")
        zombie_files.append(root_file)
        return hist


# def count_scifi_hits(root_file):
#     with ROOT.TFile.Open(root_file) as f:
#         tree = f.Get("cbmsim")  # à adapter selon la structure de tes fichiers
#         # Supposons que les hits SciFi soient stockés dans une branche spécifique
#         # Exemple : "ScifiPoint"
#         hits_per_event = []
#         for i in range(tree.GetEntries()):
#             tree.GetEntry(i)
#             hits_per_event.append(tree.Digi_ScifiHits.GetEntries())
#         return hits_per_event
    

def main(args):
    # if args.year == "2023":
    #     real_file = "../snakemake/metadata/updated/real_data_testbeam_June2023_H8_updated_metadata.csv"
    #     real_name = os.path.basename(real_file)
    #     MC_file = "../snakemake/metadata/updated/MC_data_testbeam2023_updated_metadata.csv"
    #     MC_name = os.path.basename(MC_file)
    # elif args.year == "2024":
    real_file = "../snakemake/metadata/updated/real_data_testbeam_24_updated_metadata.csv"
    real_name = os.path.basename(real_file)
    MC_file = "../snakemake/metadata/updated/MC_data_testbeam2024_updated_metadata.csv"
    MC_name = os.path.basename(MC_file)
    # else:
    #     print('Unrecognized year')
    #     return 0
    
    zombie_files = []
    tree_missing_files = []
    n_events_MC = []
    n_events_real = []
    positive_files_MC = []
    positive_files_real = []
    
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
    
    real_groups = real_df.groupby(["beam_energy", "beam_type"])
    MC_groups   = MC_df.groupby(["beam_energy", "beam_type"])
    
    # comparison = []
    
    for (energy, btype), group in real_groups:
        if not ((energy == '150GeV') and (btype == 'e-')):
            print(f'Skipping {energy} beam of {btype}')
            continue 
        if (energy, btype) not in MC_groups.groups:
            print(f'No equivalent for {energy} beam of {btype} in MC data, skipping.')
            continue
        
        hist_real = ROOT.TH1F(f'Real data {energy} {btype}', f'Real data {energy} {btype}', 100, 0, 3000)
        hist_MC = ROOT.TH1F(f'MC data {energy} {btype}', f'MC data {energy} {btype}', 100, 0, 3000)
        # real_hits = []
        # MC_hits = []
        
        # for file in MC_groups.get_group((energy, btype))['digi_path']:
        #     hist_MC = make_hist_from_file(file, hist_MC, ROOT.kBlue, zombie_files, tree_missing_files, n_events_MC, positive_files_MC)
        
        hist_real = make_hist_from_file(group['digi_path'].values[6], hist_real, ROOT.kRed, zombie_files, tree_missing_files, n_events_real, positive_files_real)            
        
        print(f'real data done, now MC for {energy} {btype}')
        
        hist_MC = make_hist_from_file(MC_groups.get_group((energy, btype))['digi_path'].values[0], hist_MC, ROOT.kBlue, zombie_files, tree_missing_files, n_events_MC, positive_files_MC)

        print(f'MC data done, now plotting for {energy} {btype}')
        
        # if zombie_files:
        #     with open(f"zombie_files_{energy}_{btype}.txt", "w") as fout:
        #         for bf in zombie_files:
        #             fout.write(bf + "\n")
        #     print(f"⚠️ {len(zombie_files)} zombie files saved in zombie_files_{energy}_{btype}.txt")
            
        # if tree_missing_files:
        #     with open(f"tree_missing_files.txt_{energy}_{btype}", "w") as fout:
        #         for bf in tree_missing_files:
        #             fout.write(bf + "\n")
        #     print(f"⚠️ {len(tree_missing_files)} tree missing files saved in tree_missing_files_{energy}_{btype}.txt")
            
        # if positive_files_MC:
        #     with open(f"positive_files_MC.txt_{energy}_{btype}", "w") as fout:
        #         for bf in positive_files_MC:
            #         fout.write(bf + "\n")
            # print(f"⚠️ {len(positive_files_MC)} positive files saved in positive_files_MC_{energy}_{btype}.txt")
        
        # for file in group['digi_path']:
        #     hist_real = make_hist_from_file(file, hist_real, ROOT.kRed, zombie_files, tree_missing_files, n_events_real, positive_files_real)

        
        # Normalisation (aire = 1)
        if hist_real.Integral() > 0:
            hist_real.Scale(1.0 / hist_real.Integral())
        if hist_MC.Integral() > 0:
            hist_MC.Scale(1.0 / hist_MC.Integral())

        c = ROOT.TCanvas(f"SciFi hits for {energy} {btype}", f"SciFi hits for {energy} {btype}", 800, 600)

        # Dessiner les deux histos sur le même canevas
        hist_real.Draw("HIST")
        hist_MC.Draw("HIST SAME")
        # hist_MC.Draw("HIST")

        # Ajouter une légende
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(hist_real, "Real Data", "l")
        legend.AddEntry(hist_MC, "MC Data", "l")
        legend.Draw()

        c.SaveAs(f"comparison_hits_{energy}_{btype}_2024_test.png")
        # c.SaveAs(f"MC_hits_{energy}_{btype}_2024_test.png")

        
        print(f'number of events in MC for {energy} {btype}: {sum(n_events_MC)}')

        
    # df_comp = pd.DataFrame(comparison)
    
    # for i in range(len(df_comp)):
    #     plt
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    # parser.add_argument("-y", "--year", dest="year", help="year", required=True)
    args = parser.parse_args()
    main(args)
    
# MC_data,150GeV_11,X_neg38_Y_45_Z_315/0,0,/eos/experiment/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg38_Y_45_Z_315/0/sndLHC.PG_11-TGeant4_digCPP.root,/eos/experiment/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg38_Y_45_Z_315/0/geofile_full.PG_11-TGeant4.root,150GeV,e-


# SciFi hits
    # for aHit in event.Digi_ScifiHits:
    #     if not aHit.isValid():
    #         continue
    #     detID = aHit.GetDetectorID()
    #     station = detID // 1000000

    #     Scifi.GetSiPMPosition(detID, A, 😎

    #     max_QDC = 200 * 16
    #     this_qdc = 0
    #     ns = max(1,aHit.GetnSides())
    #     for side in range(ns):
    #         for m in  range(aHit.GetnSiPMs()):
    #             qdc = aHit.GetSignal(m+side*aHit.GetnSiPMs())
    #             if not qdc < 0:
    #                 this_qdc += qdc
    #     if this_qdc > max_QDC :
    #         this_qdc = max_QDC
    #     hit_time = aHit.GetTime()
        
    #     all_hits.append({
    #         "detType": 0,
    #         "station": station,
    #         "isVertical": aHit.isVertical(),
    #         "x":A.x(),
    #         "y":A.y(),
    #         "z":A.z(),
    #         "qdc":this_qdc,
    #         "hit_time": hit_time
    #     })