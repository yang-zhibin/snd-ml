import os
import csv
import ROOT
import pandas as pd
import glob
from tqdm import tqdm
import yaml
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


def make_hist_from_file(root_file, hist, color):
    # Crée un histogramme vide
    # hist = ROOT.TH1F(hist_name, hist_name, 100, 0, 1000)  # binning à adapter
    try:
        # f = ROOT.TFile.Open(root_file)
        f = ROOT.TFile(root_file, 'read')
        tree = f.Get('cbmsim')
        if not f or f.IsZombie():
            print(f"[SKIP] Fichier illisible ou zombie :{root_file}")
            return hist
        
        # tree = f.Get("cbmsim")
        
        if not tree:
            print(f"[SKIP] Tree 'cbmsim' not found in file {root_file}")
            return hist

        scifi_hits = ROOT.TClonesArray("sndScifiHit")  #why this step and whats a "sndScifiHit" object ?
        tree.SetBranchAddress("Digi_ScifiHits", scifi_hits)
        # Remplir avec les hits SciFi
        for i in tqdm(range(tree.GetEntries()), desc="Processing events"):
            tree.GetEntry(i)
            if hasattr(tree, "Digi_ScifiHits"):
                # nhits = tree.Digi_ScifiHits
                nhits = scifi_hits.GetEntries()
                if nhits > 0:
                    print(f"Event {i}: {nhits} SciFi hits")
                hist.Fill(nhits)
            else:
                print(f"⚠️ Branche 'Digi_ScifiHits' non trouvée pour {root_file}")

        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        f.Close()
        return hist

    except OSError as e:
        print(f"[WARNING] Impossible d'ouvrir {root_file}: {e}")
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
    if args.year == "2023":
        real_file = "../snakemake/metadata/updated/real_data_testbeam_June2023_H8_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "../snakemake/metadata/updated/MC_data_testbeam2023_updated_metadata.csv"
        MC_name = os.path.basename(MC_file)
    elif args.year == "2024":
        real_file = "../snakemake/metadata/updated/real_data_testbeam_24_updated_metadata.csv"
        real_name = os.path.basename(real_file)
        MC_file = "../snakemake/metadata/updated/MC_data_testbeam2024_updated_metadata.csv"
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
    
    real_groups = real_df.groupby(["beam_energy", "beam_type"])
    MC_groups   = MC_df.groupby(["beam_energy", "beam_type"])
    
    # comparison = []
    
    for (energy, btype), group in real_groups:
        if (energy, btype) not in MC_groups.groups:
            print(f'No equivalent for {energy} beam of {btype} in MC data, skipping.')
            continue
        
        hist_real = ROOT.TH1F(f'Real data {energy} {btype}', f'Real data {energy} {btype}', 100, 0, 1000)
        hist_MC = ROOT.TH1F(f'MC data {energy} {btype}', f'MC data {energy} {btype}', 100, 0, 1000)
        # real_hits = []
        # MC_hits = []
        
        for file in MC_groups.get_group((energy, btype))['digi_path']:
            hist_MC = make_hist_from_file(file, hist_MC, ROOT.kBlue)
            # MC_hits.append(count_scifi_hits(file))

        print(f'MC data done, now real for {energy} {btype}')
        
        
        for file in group['digi_path']:
            hist_real = make_hist_from_file(file, hist_real, ROOT.kRed)
            # real_hits.append(count_scifi_hits(file))
            
        print(f'real data done, now plotting for {energy} {btype}')

        # comparison.append({
        # "energy": energy,
        # "type": btype,
        # "real_hits": np.concatenate(real_hits),
        # "mc_hits": np.concatenate(MC_hits),
        # })
        
        # # Exemple : un run réel vs un fichier MC
        # real_file = "path/to/real.root"
        # mc_file   = "path/to/mc.root"

        # h_real = make_hist_from_file(real_file, "Real Data", ROOT.kRed)
        # h_mc   = make_hist_from_file(mc_file,   "MC Data",   ROOT.kBlue)
        
        # Normalisation (aire = 1)
        if hist_real.Integral() > 0:
            hist_real.Scale(1.0 / hist_real.Integral())
        if hist_MC.Integral() > 0:
            hist_MC.Scale(1.0 / hist_MC.Integral())

        c = ROOT.TCanvas(f"SciFi hits for {energy} {btype}", f"SciFi hits for {energy} {btype}", 800, 600)

        # Dessiner les deux histos sur le même canevas
        hist_real.Draw("HIST")
        hist_MC.Draw("HIST SAME")

        # Ajouter une légende
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(hist_real, "Real Data", "l")
        legend.AddEntry(hist_MC, "MC Data", "l")
        legend.Draw()

        c.SaveAs(f"comparison_hits_{energy}_{btype}_{args.year}.png")

        
    # df_comp = pd.DataFrame(comparison)
    
    # for i in range(len(df_comp)):
    #     plt
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    parser.add_argument("-y", "--year", dest="year", help="year", required=True)
    args = parser.parse_args()
    main(args)
    
# MC_data,150GeV_11,X_neg38_Y_45_Z_315/0,0,/eos/experiment/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg38_Y_45_Z_315/0/sndLHC.PG_11-TGeant4_digCPP.root,/eos/experiment/sndlhc/MonteCarlo/testbeam2024/150GeV_11/X_neg38_Y_45_Z_315/0/geofile_full.PG_11-TGeant4.root,150GeV,e-
