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

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6
}

def predict_class(df):
    print("predicting results")
    df = df.Define("ParticleType", """
        if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
        else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
        else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
        else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
        else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
        else if (PdgCode == 2112) return std::string("neutron");
        else if (PdgCode == 13 || PdgCode == -13) return std::string("muon");
        else return std::string("unknown");
        """)
    argmax_expr = """
        double vals[6] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5};
        int idx = 0;
        double max_val = vals[0];
        for (int i = 1; i < 6; ++i) {
            if (vals[i] > max_val) {
                max_val = vals[i];
                idx = i;
            }
        }
        return idx;
        """
    df = df.Define("PredClass", argmax_expr)


    return df

def predict_signal(df, signal, score=0):
    signal_class = particle_2_class[signal]
    if_signal = f'PredClass == {signal_class} && Prediction_{signal_class} > {str(score)}'
    
    pre_df = df.Define('signal', if_signal)

    #print("predicting signal with score ", score, pre_df.Filter(f'signal==1 && ParticleType=="{signal}"').Count().GetValue())
    return pre_df

def cutbase_eff_yield(pre_df,signal):
    print('cal cutbase eff yield')
    #print("in calcualtion pred_df", pre_df)
    signal_pass = pre_df.Filter(f'stage2==1 && ParticleType=="{signal}"').Count().GetValue()
    total_signal = pre_df.Filter(f'ParticleType=="{signal}"').Count().GetValue()
    signal_eff = signal_pass / total_signal if total_signal > 0 else 0

    bkg_pass = pre_df.Filter(f'stage2==1 && ParticleType!="{signal}"').Count().GetValue()
    total_bkg = pre_df.Filter(f'ParticleType!="{signal}"').Count().GetValue()
    bkg_eff = bkg_pass / total_bkg if total_bkg > 0 else 0

    signal_yield = signal_eff *  157

    bkg_yield = 0
    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1]
    }
    bins = [5,10,20,30,40,50,60,70,80,90]

    for bkg in bkgs:
        bkg_df = pre_df.Filter(f'ParticleType=="{bkg}"')
        bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')

        for index in range(len(bins)-1):
            lower = bins[index]
            upper = bins[index+1]
            filtered_df = bkg_df.Filter(f'(energy>{lower}) && (energy<={upper})')

            total_event = filtered_df.Count().GetValue()

            if total_event ==0:
                continue

            pass_event = filtered_df.Filter(f'stage2==1').Count().GetValue()
            eff = pass_event/total_event if total_event > 0 else 0
            yeild = eff * intRate[bkg][index]
            bkg_yield +=yeild

            print(f'{bkg} {lower, upper}GeV: eff:{eff}, yield:{yeild}')

    print(f"cutbase method, signal:{signal}, signal eff:{signal_eff}, bkg eff:{bkg_eff},signal yield:{signal_yield}, bkg yield:{bkg_yield}")

    return signal_eff, bkg_eff, signal_yield, bkg_yield


def eff_yield_from_model(df,signal, score, concern_particles):
    effs = {}
    yields = {}
    for particle in concern_particle:
        effs[particle] = []
        yields[particle] = []

    particle_inteRate = {
        've': 157,    # Example value
        'vm': 157 / 0.72 * 0.23,    # Example value
        'muon': 5.48e5,  # Example value
    }


    for particle in concern_particle:
        if particle in ['kaon', 'neutron']:
            continue  # Special handling for kaon and neutron

        event_pass = pre_df.Filter(f'signal==1 && ParticleType=="{particle}"').Count().GetValue()
        total_event = pre_df.Filter(f'ParticleType=="{particle}"').Count().GetValue()
        particle_eff = event_pass / total_event if total_event > 0 else 0

        particle_yield = particle_eff * particle_inteRate[particle]  # Assuming some base factor (like luminosity)

        effs[particle].append(signal_eff)
        yields[particle].append(signal_yield)

    #print("in calcualtion df", df)
    pre_df = predict_signal(df, signal, score)
    #print("in calcualtion pred_df", pre_df)
    signal_pass = pre_df.Filter(f'signal==1 && ParticleType=="{signal}"').Count().GetValue()
    total_signal = pre_df.Filter(f'ParticleType=="{signal}"').Count().GetValue()
    signal_eff = signal_pass / total_signal if total_signal > 0 else 0

    bkg_pass = pre_df.Filter(f'signal==1 && ParticleType!="{signal}"').Count().GetValue()
    total_bkg = pre_df.Filter(f'ParticleType!="{signal}"').Count().GetValue()
    bkg_eff = bkg_pass / total_bkg if total_bkg > 0 else 0

    signal_yield = signal_eff *  157

    #kaon_eff 


    muon_inRate = 5.48e5
    muon_pass = pre_df.Filter(f'signal==1 && ParticleType=="muon"').Count().GetValue()
    total_muon = pre_df.Filter(f'ParticleType=="muon"').Count().GetValue()
    muon_eff = muon_pass / total_muon
    muon_yield = muon_eff * muon_inRate

    bkg_yield = 0
    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1],
    }
    bins = [5,10,20,30,40,50,60,70,80,90]
    
    for bkg in bkgs:
        bkg_df = pre_df.Filter(f'ParticleType=="{bkg}"')
        bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')

        for index in range(len(bins)-1):
            lower = bins[index]
            upper = bins[index+1]
            filtered_df = bkg_df.Filter(f'(energy>{lower}) && (energy<={upper})')

            total_event = filtered_df.Count().GetValue()

            if total_event ==0:
                continue

            pass_event = filtered_df.Filter(f'signal==1').Count().GetValue()
            eff = pass_event/total_event if total_event > 0 else 0
            yeild = eff * intRate[bkg][index]
            bkg_yield +=yeild
    
    print(f"time:{time()}, signal:{signal}, score:{score}, signal eff:{signal_eff}, bkg eff:{bkg_eff},signal yield:{signal_yield}, bkg yield:{bkg_yield}")

    return signal_eff, bkg_eff, signal_yield, bkg_yield, muon_eff, muon_yield



    

def adaptive_step_search(df,signal, concern_particles, start_value=0, end_value=1, initial_step=0.1, tolerance=0.02):
    print('start adaptive_step_search')
    input_value = start_value
    step_size = initial_step

    eff_list = []
    yield_list = []


    previous_signal_eff, effs, yields= eff_yield_from_model(df,signal,input_value, concern_particles)
    input_value += step_size

    while previous_signal_eff > 1e-6:
        #print("in step",df)
        signal_eff, effs, yields = eff_yield_from_model(df,signal,input_value, concern_particles)
        

        # Adjust step size based on the rate of change of x
        if (abs(signal_eff - previous_signal_eff) > tolerance and abs(signal_eff - previous_signal_eff) < (tolerance*5)):
            step_size = max(step_size / 8, 1e-6)  # Decrease step size if change is rapid
            input_value += step_size
        elif abs(signal_eff - previous_signal_eff) > (tolerance*5):
            input_value -= step_size
            step_size = max(step_size / 10, 1e-6) 
            input_value += step_size
            signal_eff, effs, yields = eff_yield_from_model(df,signal,input_value, concern_particles)

            input_value += step_size
        else:
            step_size = min(step_size * 1.4, 0.1)  # Increase step size if change is slow
            input_value += step_size
        #print(input_value)
        eff_list.append(effs)
        yield_list.append(yields)

        previous_signal_eff = signal_eff

        

    return signal_eff_list, bkg_eff_list, signal_yield_list, bkg_yield_list

def cal_eff_yield(model_output_path, signal, concern_particles):
    print("reading output from", model_output_path)

    chain = ROOT.TChain("tree")
    file_count = 0
    for filename in os.listdir(model_output_path):
        
        if filename.endswith(".root"):
            print(os.path.join(model_output_path, filename), ' read')
            chain.Add(os.path.join(model_output_path, filename))
        #if (file_count>2):
        #    break
        file_count+=1
    df = ROOT.RDataFrame(chain)

    df = predict_class(df)
    signal_class = particle_2_class[signal]
    lowest_score = df.Filter(f'PredClass == {signal_class}').Min(f'Prediction_{signal_class}').GetValue()

    effs, yields = adaptive_step_search(df, signal, concern_particles, start_value=lowest_score)


    return effs, yields

def process(model_list, version, signal):

    concern_particles = ['ve', 'vm', 'kaon', 'neutron', 'muon']

    eff_yield_df = {
        'model': []
    }
    for particle in particles:
        eff_yield_df[f'{particle}_eff'] = []
        eff_yield_df[f'{particle}_yield'] = []


    for model in model_list:
        model_output_path = f'/eos/user/z/zhibin/sndData/converted/pt/output/{model}/'
        effs, yields = cal_eff_yield(model_output_path, signal, concern_particles)


        eff_yield_df['model'].append(model)
    
        # Assuming cal_eff_yield returns dictionaries for each particle's eff and yield
        for particle in concern_particles:
            eff_yield_df[f'{particle}_eff'].append(effs.get(particle, 0))
            eff_yield_df[f'{particle}_yield'].append(yields.get(particle, 0))



    return eff_yield_df

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
    canvas_eff.SaveAs(f"{out_dir}/{signal}_efficiency_compare_plot.png")

    # Create and save the yield plot
    canvas_yield = ROOT.TCanvas("c_yield", "Signal vs Background Yield", 800, 600)
    canvas_yield.SetLogy()
    canvas_yield.SetLeftMargin(0.15)
    #multi_graph_yield.SetTitle(f'{signal}')
    multi_graph_yield.Draw("APL")
    multi_graph_yield.GetXaxis().SetTitle("Signal Yield")
    multi_graph_yield.GetYaxis().SetTitle("Background Yield")
    multi_graph_yield.SetTitle("Background Yield vs Signal Yield")

    legend_yield = canvas_yield.BuildLegend(0.5, 0.2, 0.8, 0.5)
    legend_yield.SetHeader("Models", "C")
    legend_yield.SetBorderSize(0)

    canvas_yield.Update()
    canvas_yield.Draw()
    canvas_yield.SaveAs(f"{out_dir}/{signal}_yield_compare_plot.eps")

def main(model, signal):

    #model_list = ['baseline', 'weight','normalized_weight', 'intRate_weight','intRate_weightX100','intRate_weightX100^2']
    model_list = [model]
    version = '8'

    eff_yield_df = process(model_list, version, signal)

    df = pd.DataFrame(eff_yield_df)

    csv_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv/{signal}_{model}_eff_yield_df.csv'
    df.to_csv(csv_path, index=False)

    #out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/'
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
    directory = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/show_model/'
    print('signal:', signal)
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(signal) and filename.endswith('.csv'):
            # Read each CSV file into a dataframe
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            
            # Append the dataframe to the list
            df_list.append(df)

    # Concatenate all dataframes into one
    eff_yield_df = pd.concat(df_list, ignore_index=True)

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/'
    plot_eff_yield(out_dir, eff_yield_df, signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='weight')
    parser.add_argument("-s", "--signal", dest="signal", default='vm')
    args = parser.parse_args()
    main(args.model, args.signal)
    #plot(args.signal)

