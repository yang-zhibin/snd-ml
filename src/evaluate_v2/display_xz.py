import ROOT
import os
import argparse
import pandas as pd
import math 
import numpy as np
import pandas as pd
from scipy import optimize
from time import time
from datetime import datetime
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

def plot_avg_channel(rdf, out_dir):
    # Create 2D histograms for scifi and DS variables
    hist_scifi = rdf.Histo2D(("hist_scifi", "SciFi Avg Ver vs Hor", 100, 0, 1600, 100, 0, 1600), "scifi_avg_ver", "scifi_avg_hor")
    hist_ds = rdf.Histo2D(("hist_ds", "DS Avg Ver vs Hor", 100, 60, 120, 100, 0, 60), "DS_avg_ver", "DS_avg_hor")

    # Create canvases and save the histograms
    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    hist_scifi.Draw("COLZ")
    c1.SaveAs(f"{out_dir}/scifi_avg_ver_vs_hor.png")

    c2 = ROOT.TCanvas("c2", "Canvas", 800, 600)
    hist_ds.Draw("COLZ")
    c2.SaveAs(f"{out_dir}/DS_avg_ver_vs_hor.png")

def plot_yz(rdf, out_dir):
    hist_yz = rdf.Histo2D(("hist_yz", "Hit YZ distribution", 1000, 280, 360, 100, 0, 60,), "Hits.z1", "Hits.y1")

    # Create canvases and save the histograms
    c1 = ROOT.TCanvas("c1", "Canvas", 1200, 600)
    hist_yz.Draw("COLZ")
    c1.SaveAs(f"{out_dir}/yz.png")

def plot_xy(rdf, out_dir):
    rdf = rdf.Filter('ROOT::VecOps::All(Hits.detType == 1)')
    hist_xy = rdf.Histo2D(("hist_xy", "Hit XY distribution", 100, -60, 0, 100, 0, 60,), "Hits.x2", "Hits.y2")

    # Create canvases and save the histograms
    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    c1.SetLeftMargin(0.15)   # Left margin (default is 0.1)
    c1.SetRightMargin(0.15)  # Right margin (default is 0.1)
    c1.SetTopMargin(0.1)     # Top margin (default is 0.1)
    c1.SetBottomMargin(0.15)
    hist_xy.Draw("COLZ")
    c1.SaveAs(f"{out_dir}/xy.png")
    

def process(df):
    #rdf = df.Filter(f'PredClass==0')
    rdf = df
    print(rdf.GetColumnNames())

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot_muon/'
    
    #plot_avg_channel(rdf, out_dir)
    plot_xy(rdf, out_dir)
    return 0


def main():
    #model_list = ['baseline', 'weight','normalized_weight', 'intRate_weight','intRate_weightX100','intRate_weightX100^2']
    
    
    list_path = '/eos/user/z/zhibin/sndData/converted/real_muon/real_muon_evt_list.csv'
    list_df = pd.read_csv(list_path)
    input_path = [path for path in list_df['file']]
    print("reading data from", input_path)

    chain = ROOT.TChain("cbmsim")
    file_count = 0
    for filename in input_path:
        #print(filename)
        #if filename.endswith(".root") and (filename.startswith("test_3_") or filename.startswith("test_0_neutrino_output")) :
        if filename.endswith(".root"):
            print(filename, ' read')
            chain.Add(filename)
        #if (file_count>100):
        #    break
        file_count+=1

    df = ROOT.RDataFrame(chain)

    process(df)



if __name__ == "__main__":
    main()

