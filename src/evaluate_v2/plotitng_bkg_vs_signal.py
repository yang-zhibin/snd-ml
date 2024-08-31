import ROOT
import os
import argparse
import pandas as pd
import math 
import numpy as np
import pandas as pd
from scipy import optimize
from time import time


ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)


def plot_eff_yield(out_dir, eff_yield_df, signal):
    groups = eff_yield_df.groupby('model')

    # Define colors and markers for different models
    colors = [
    ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan,
    ROOT.kYellow, ROOT.kOrange, ROOT.kViolet, ROOT.kPink, ROOT.kSpring
    ]

    markers = [
        21, 22, 23, 24,
        25, 26, 27, 28, 29
    ]

    # Loop over groups to create graphs
    multi_graph_eff = ROOT.TMultiGraph()
    multi_graph_yield = ROOT.TMultiGraph()
    for i, (name, group) in enumerate(groups):
        n_points = len(group)

        # Efficiency graph
        signal_eff = group['signal_eff'].to_numpy()
        bkg_eff = group['bkg_eff'].to_numpy()
        graph_eff = ROOT.TGraph(n_points, signal_eff, bkg_eff) 
        graph_eff.SetTitle(name)

        # Yield graph
        signal_yield = group['signal_yield'].to_numpy()
        if signal == 've':
            signal_yield = group['signal_yield_ve_2'].to_numpy()
        bkg_yield = group['bkg_yield'].to_numpy()
        graph_yield = ROOT.TGraph(n_points, signal_yield, bkg_yield)
        graph_yield.SetTitle(name)
    
        # Apply remaining markers and colors
        graph_eff.SetMarkerStyle(markers[i % len(markers)])
        graph_eff.SetMarkerColor(colors[i % len(colors)])
        graph_eff.SetLineColor(colors[i % len(colors)])
        
        graph_yield.SetMarkerStyle(markers[i % len(markers)])
        graph_yield.SetMarkerColor(colors[i % len(colors)])
        graph_yield.SetLineColor(colors[i % len(colors)])
        
        # Add graphs to the multigraphs
        multi_graph_eff.Add(graph_eff)
        multi_graph_yield.Add(graph_yield)
    
    if signal == 'vm':
        cutbase = {
        'model': 'cutbase',
        'bkg_eff': 1.1097486512757378e-05,
        'signal_eff': 0.04606038063356271,
        'bkg_yield': 0.12389623629215483,
        'signal_yield': 7.23147975946934
        }
        cutbase_df = pd.DataFrame([cutbase])
        n_points = len(cutbase_df)
        name = 'cut-based'
        # Convert Pandas Series to numpy arrays
        signal_eff = cutbase_df['signal_eff'].to_numpy()
        bkg_eff = cutbase_df['bkg_eff'].to_numpy()
        signal_yield = cutbase_df['signal_yield'].to_numpy()
        bkg_yield = cutbase_df['bkg_yield'].to_numpy()

        # Efficiency graph
        graph_eff = ROOT.TGraph(n_points, signal_eff, bkg_eff)
        graph_eff.SetTitle(name)

        # Yield graph
        graph_yield = ROOT.TGraph(n_points, signal_yield, bkg_yield)
        graph_yield.SetTitle(name)
    
        # Apply remaining markers and colors
        graph_eff.SetMarkerStyle(20)
        graph_eff.SetMarkerColor(ROOT.kRed)
        graph_eff.SetLineColor(ROOT.kRed)
        
        graph_yield.SetMarkerStyle(20)
        graph_yield.SetMarkerColor(ROOT.kRed)
        graph_yield.SetLineColor(ROOT.kRed)
        
        # Add graphs to the multigraphs
        multi_graph_eff.Add(graph_eff)
        multi_graph_yield.Add(graph_yield)
        

    # Create and save the efficiency plot
    canvas_eff = ROOT.TCanvas("c_eff", "Signal vs Background Efficiency", 800, 600)
    canvas_eff.SetLogy()
    canvas_eff.SetLeftMargin(0.15)
    #multi_graph_eff.SetTitle(f'{signal}')
    multi_graph_eff.Draw("APL")
    multi_graph_eff.GetXaxis().SetTitle("Signal Efficiency")
    multi_graph_eff.GetYaxis().SetTitle("Background Efficiency")
    multi_graph_eff.SetTitle("Background Efficiency vs Signal Efficiency")

    legend_eff = canvas_eff.BuildLegend()
    legend_eff.SetHeader("Models", "C")
    legend_eff.SetBorderSize(0)

    canvas_eff.Update()
    canvas_eff.Draw()
    canvas_eff.SaveAs(f"{out_dir}/{signal}_efficiency_compare_plot.pdf")

    # Create and save the yield plot
    canvas_yield = ROOT.TCanvas("c_yield", "Signal vs Background Yield", 800, 600)
    canvas_yield.SetLogy()
    canvas_yield.SetLeftMargin(0.15)
    #multi_graph_yield.SetTitle(f'{signal}')
    multi_graph_yield.Draw("APL")
    multi_graph_yield.GetXaxis().SetTitle("Signal Yield")
    multi_graph_yield.GetYaxis().SetTitle("Background Yield")
    multi_graph_yield.SetTitle("Background Yield vs Signal Yield")

    #legend_yield = canvas_yield.BuildLegend(0.5, 0.2, 0.8, 0.5)
    legend_yield = canvas_yield.BuildLegend()
    legend_yield.SetHeader("Models", "C")
    legend_yield.SetBorderSize(0)

    canvas_yield.Update()
    canvas_yield.Draw()
    canvas_yield.SaveAs(f"{out_dir}/{signal}_yield_compare_plot.pdf")

def plot_real_muon(out_dir, eff_yield_df, signal):
    groups = eff_yield_df.groupby('model')

    # Define colors and markers for different models
    colors = [
    ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan,
    ROOT.kYellow, ROOT.kOrange, ROOT.kViolet, ROOT.kPink, ROOT.kSpring
    ]

    markers = [
        21, 22, 23, 24,
        25, 26, 27, 28, 29
    ]

    # Loop over groups to create graphs
    multi_graph_eff = ROOT.TMultiGraph()
    multi_graph_yield = ROOT.TMultiGraph()
    for i, (name, group) in enumerate(groups):
        n_points = len(group)

        # Efficiency graph
        signal_eff = group[f'{signal}_eff'].to_numpy()
        bkg_eff = group['muon_eff'].to_numpy()
        graph_eff = ROOT.TGraph(n_points, signal_eff, bkg_eff) 
        graph_eff.SetTitle(name)

        # Yield graph
        signal_yield = group[f'{signal}_yield'].to_numpy()
        bkg_yield = group['muon_yield'].to_numpy()
        graph_yield = ROOT.TGraph(n_points, signal_yield, bkg_yield)
        graph_yield.SetTitle(name)
    
        # Apply remaining markers and colors
        graph_eff.SetMarkerStyle(markers[i % len(markers)])
        graph_eff.SetMarkerColor(colors[i % len(colors)])
        graph_eff.SetLineColor(colors[i % len(colors)])
        
        graph_yield.SetMarkerStyle(markers[i % len(markers)])
        graph_yield.SetMarkerColor(colors[i % len(colors)])
        graph_yield.SetLineColor(colors[i % len(colors)])
        
        # Add graphs to the multigraphs
        multi_graph_eff.Add(graph_eff)
        multi_graph_yield.Add(graph_yield)
    
    if signal == 'vm':
        cutbase = {
        'model': 'cutbase',
        'bkg_eff': 1.1097486512757378e-05,
        'signal_eff': 0.04606038063356271,
        'bkg_yield': 0.12389623629215483,
        'signal_yield': 7.23147975946934
        }
        cutbase_df = pd.DataFrame([cutbase])
        n_points = len(cutbase_df)
        name = 'cut-based'
        # Convert Pandas Series to numpy arrays
        signal_eff = cutbase_df['signal_eff'].to_numpy()
        bkg_eff = cutbase_df['bkg_eff'].to_numpy()
        signal_yield = cutbase_df['signal_yield'].to_numpy()
        bkg_yield = cutbase_df['bkg_yield'].to_numpy()

        # Efficiency graph
        graph_eff = ROOT.TGraph(n_points, signal_eff, bkg_eff)
        graph_eff.SetTitle(name)

        # Yield graph
        graph_yield = ROOT.TGraph(n_points, signal_yield, bkg_yield)
        graph_yield.SetTitle(name)
    
        # Apply remaining markers and colors
        graph_eff.SetMarkerStyle(20)
        graph_eff.SetMarkerColor(ROOT.kRed)
        graph_eff.SetLineColor(ROOT.kRed)
        
        graph_yield.SetMarkerStyle(20)
        graph_yield.SetMarkerColor(ROOT.kRed)
        graph_yield.SetLineColor(ROOT.kRed)
        
        # Add graphs to the multigraphs
        multi_graph_eff.Add(graph_eff)
        multi_graph_yield.Add(graph_yield)
        

    # Create and save the efficiency plot
    canvas_eff = ROOT.TCanvas("c_eff", "Signal vs Background Efficiency", 800, 600)
    canvas_eff.SetLogy()
    canvas_eff.SetLeftMargin(0.15)
    #multi_graph_eff.SetTitle(f'{signal}')
    multi_graph_eff.Draw("APL")
    multi_graph_eff.GetXaxis().SetTitle("Signal Efficiency")
    multi_graph_eff.GetYaxis().SetTitle("Muon Efficiency")
    multi_graph_eff.SetTitle("Muon Efficiency vs Signal Efficiency")

    legend_eff = canvas_eff.BuildLegend()
    legend_eff.SetHeader("Models", "C")
    legend_eff.SetBorderSize(0)

    canvas_eff.Update()
    canvas_eff.Draw()
    canvas_eff.SaveAs(f"{out_dir}/{signal}_efficiency_real_muon.png")

    # Create and save the yield plot
    canvas_yield = ROOT.TCanvas("c_yield", "Signal vs Background Yield", 800, 600)
    canvas_yield.SetLogy()
    canvas_yield.SetLeftMargin(0.15)
    #multi_graph_yield.SetTitle(f'{signal}')
    multi_graph_yield.Draw("APL")
    multi_graph_yield.GetXaxis().SetTitle("Signal Yield")
    multi_graph_yield.GetYaxis().SetTitle("Muon Yield")
    multi_graph_yield.SetTitle("Muon Yield vs Signal Yield")

    legend_yield = canvas_yield.BuildLegend(0.5, 0.2, 0.8, 0.5)
    legend_yield.SetHeader("Models", "C")
    legend_yield.SetBorderSize(0)

    canvas_yield.Update()
    canvas_yield.Draw()
    canvas_yield.SaveAs(f"{out_dir}/{signal}_yield_real_muon.pdf")


def plot(signal):

    # model = 'baseline'
    # signal = 'vm'
    # model_output_path = f'/eos/user/z/zhibin/sndData/converted/pt/output/{model}/'
    # chain = ROOT.TChain("tree")
    # file_count = 0
    # for filename in os.listdir(model_output_path):
        
    #     if filename.endswith(".root"):
    #         print(os.path.join(model_output_path, filename), ' read')
    #         chain.Add(os.path.join(model_output_path, filename))
    #     #if (file_count>2):
    #     #    break
    #     file_count+=1
    # df = ROOT.RDataFrame(chain)
    # df = predict_class(df)

    # signal_eff, bkg_eff, signal_yield, bkg_yield = cutbase_eff_yield(df, signal)
    '''   
        cal cutbase eff yield
    neutron ((5, 10))GeV: eff:0.0, yield:0.0
    neutron ((10, 20))GeV: eff:1.3817097552853853e-06, yield:0.010487177042616074
    neutron ((20, 30))GeV: eff:1.3418568077763286e-05, yield:0.015833910331760678
    neutron ((30, 40))GeV: eff:2.2562922349702733e-05, yield:0.01195834884534245
    neutron ((40, 50))GeV: eff:2.9073063515921863e-05, yield:0.013548047598419589
    neutron ((50, 60))GeV: eff:3.6879502372581317e-05, yield:0.0009588670616871143
    neutron ((60, 70))GeV: eff:7.575279338425604e-05, yield:0.0013635502809166088
    neutron ((70, 80))GeV: eff:5.8280718795531814e-05, yield:0.0004942204953861098
    neutron ((80, 90))GeV: eff:8.749671887304226e-05, yield:0.0007419721760433984
    kaon ((5, 10))GeV: eff:4.809771917409162e-07, yield:0.012072527512696997
    kaon ((10, 20))GeV: eff:4.4676863416127e-06, yield:0.025555165874024646
    kaon ((20, 30))GeV: eff:1.6778636098228846e-05, yield:0.014312176591789206
    kaon ((30, 40))GeV: eff:4.22654268808115e-05, yield:0.004649196956889264
    kaon ((40, 50))GeV: eff:4.675204759358445e-05, yield:0.004385342064278221
    kaon ((50, 60))GeV: eff:4.599216600105782e-05, yield:0.0029802923568685467
    kaon ((60, 70))GeV: eff:9.087781477803094e-05, yield:0.0008996903663025063
    kaon ((70, 80))GeV: eff:9.253339614388102e-05, yield:0.0021467747905380397
    kaon ((80, 90))GeV: eff:0.00013121529970394548, yield:0.0015089759465953731
    cutbase method, signal:vm, signal eff:0.04606038063356271, bkg eff:1.1097486512757378e-05,signal yield:7.231479759469345, bkg yield:0.12389623629215483
        '''
    
    
    #csv_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/csv/intRate_weightX100^2_eff_yield_df.csv'
    #eff_yield_df = pd.read_csv(csv_path)

    df_list = []
    directory = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_v2/'
    print('signal:', signal)
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(signal) and filename.endswith('.csv'):
            # Read each CSV file into a dataframe
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            #df = df.sort_values(by=f'{signal}_eff')
            
            # Append the dataframe to the list
            df_list.append(df)

    # Concatenate all dataframes into one
    eff_yield_df = pd.concat(df_list, ignore_index=True)
    #eff_yield_df['signal_yield']

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/'
    plot_eff_yield(out_dir, eff_yield_df, signal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    args = parser.parse_args()
    plot(args.signal)