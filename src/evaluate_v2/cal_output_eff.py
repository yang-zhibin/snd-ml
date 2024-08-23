import ROOT
import os
import argparse
import pandas as pd
import math 
import numpy as np
import pandas as pd
from scipy import optimize

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

def print_col(df):
    column_names = df.GetColumnNames()

    print("Columns in the DataFrame:")
    for name in column_names:
        print(name)

def get_rdf_info(rdf):
    # Get the number of entries in the RDataFrame
    n_entries = rdf.Count().GetValue()

    # Convert RDataFrame to Pandas DataFrame to estimate memory size
    rdf_pd = pd.DataFrame(rdf.AsNumpy())

    # Get the memory size of the Pandas DataFrame
    mem_size_bytes = rdf_pd.memory_usage(index=True, deep=True).sum()
    mem_size_mb = mem_size_bytes / (1024 * 1024)
    
    info = f"Number of entries: {n_entries}, memory:{mem_size_mb} MB"
    print(rdf)
    print(info)


def predict_class(df, signal):
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

def predict_signal(df, signal,score):
    signal_class = particle_2_class[signal]
    if_signal = f'PredClass == {signal_class} && Prediction_{signal_class} > {str(score)}'
    
    df = df.Define('signal', if_signal)

    return df


def cal_eff(df, signal):
    print("calculating efficiency")
    efficiencies={}
    for particle, class_num in particle_2_class.items():
        signal_count = df.Filter(f'ParticleType == "{particle}" && signal == 1').Count().GetValue()
        total_count = df.Filter(f'ParticleType == "{particle}"').Count().GetValue()
        efficiency = signal_count / total_count if total_count > 0 else 0
        efficiencies[particle] = efficiency



    #print("Efficiencies for each particle type:")
    #for particle_type, efficiency in efficiencies.items():
    #   print(f"{particle_type}: {efficiency:.4f}")

def find_score(score, df, target_eff, signal):
    df = predict_signal(df, signal, score)
    signal_pass = df.Filter(f'signal==1 && ParticleType=="{signal}"').Count().GetValue()
    total = df.Filter(f'ParticleType=="{signal}"').Count().GetValue()
    eff = signal_pass/total

    print(score, eff)
    return eff - target_eff



def cal_partition_eff(df, signal, model):
    df = predict_class(df, signal)

    eff = 0.05
    sol = optimize.root_scalar(find_score, args=(df,eff,signal), bracket=[0, 1], xtol=1e-06, method='brentq')

    eff_score =sol.root

    df = predict_signal(df, signal, eff_score)

    total = df.Count().GetValue()
    total_kaon = df.Filter(f'ParticleType=="kaon').Count().GetValue()
    

    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1]
    }
    bins = [5,10,20,30,40,50,60,70,80,90]

    
    for bkg in bkgs:
        print('drawing ', bkg)
        bkg_df = df.Filter(f'ParticleType=="{bkg}"')
        total_bakg = bkg_df.Count().GetValue()
        print(f"bkg total {total_bakg}")
        bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')
        x =[]
        y =[]
        y_cutbase = []
        ex=[]
        ey=[]
        total_yield = 0
        toral_cutbase_yield = 0
        for index in range(len(bins)-1):
            lower = bins[index]
            upper = bins[index+1]
            print(f"processing bin({lower},{upper})")
            filtered_df = bkg_df.Filter(f'(energy>{lower}) && (energy<={upper})')
            x_mean = filtered_df.Mean('energy').GetValue()
            total_event = filtered_df.Count().GetValue()

            if total_event ==0:
                continue

            pass_event = filtered_df.Filter(f'signal==1').Count().GetValue()
            cutbase_passevt = filtered_df.Filter(f'stage2==1').Count().GetValue()

            print(pass_event, cutbase_passevt)

            eff = pass_event/total_event if total_event > 0 else 0
            cutbase_eff = cutbase_passevt/total_event if total_event > 0 else 0

            yeild = eff * intRate[bkg][index]
            cutbase_yield = cutbase_eff * intRate[bkg][index]
            toral_cutbase_yield += cutbase_yield

            yeild_error = yeild * (1/math.sqrt(total_event))
            x_error = x_mean * (1/math.sqrt(total_event))

            total_yield +=yeild

            x.append(x_mean)
            y.append(yeild)
            y_cutbase.append(cutbase_yield)
            ex.append(x_error)
            ey.append(yeild_error)
        # Convert lists to numpy arrays
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='float64')
        ex = np.array(ex, dtype='float64')
        ey = np.array(ey, dtype='float64')
        y_cutbase = np.array(y_cutbase, dtype='float64')
        #print(x,y,ex,ey)

        mg = ROOT.TMultiGraph()
        #mg.SetTitle("yield")

        print(f"{bkg} total yield: {total_yield:.2e}, cutbase: {toral_cutbase_yield:.2e}")

        canvas = ROOT.TCanvas("Canvas", "eff curve", 400, 400)
        canvas.SetLogy()

        out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot'
        out_path = f"{out_dir}/eff_curve_model={model}_signal={signal}_bkg={bkg}.png"
        gr1 = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        gr2 = ROOT.TGraph(len(x), x, y_cutbase)
        gr1.SetLineColor(4)
        gr1.SetMarkerColor(4)

        mg.Add(gr1, "L")
        mg.Add(gr2, "L")
        mg.Draw("ALP")

        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(gr1, "ML-base", "l")
        legend.AddEntry(gr2, "cut-base", "l")
        legend.Draw()

        mg.GetXaxis().SetTitle("Energy (GeV)")
        mg.GetYaxis().SetTitle("Yield")

        canvas.SaveAs(out_path)






    ## define energy
    ##

    #print(f"Total:{total} , pass:{bkg_pass}, eff:{eff:.2e}")


def plotting_eff_curve(eff_df, out_path):
    print("plotting eff curve")
    n_points = len(eff_df)
    graph = ROOT.TGraph(n_points)

    print('debug')
    
    for i in range(n_points):
        graph.SetPoint(i, eff_df.iloc[i]['signal_eff'], eff_df.iloc[i]['bkg_eff'])


    canvas = ROOT.TCanvas("Canvas", "eff curve",2000, 2000)
    #graph.SetTitle("Efficiency Curve")
    graph.Draw("ALP")  

    canvas.SaveAs(out_path)

def plotting_eff_curve(eff_df, out_path, pointer_x, pointer_y):
    """
    Plots an efficiency curve using ROOT from a DataFrame containing signal and background efficiencies,
    and adds a pointer at specified coordinates.
    
    Parameters:
    eff_df (pd.DataFrame): DataFrame with columns 'signal_eff' and 'bkg_eff'.
    out_path (str): Path to save the resulting plot.
    pointer_x (float): X coordinate for the pointer.
    pointer_y (float): Y coordinate for the pointer.
    """
    print("plotting eff curve")
    
    # Number of points in the DataFrame
    n_points = len(eff_df)
    
    # Create a TGraph to hold the efficiency points
    graph = ROOT.TGraph(n_points)
    
    # Set the points for the TGraph
    for i in range(n_points):
        graph.SetPoint(i, eff_df.iloc[i]['signal_eff'], eff_df.iloc[i]['bkg_eff'])
    
    # Create a canvas to draw the graph
    canvas = ROOT.TCanvas("Canvas", "Efficiency Curve", 1600, 1200)
    canvas.SetLogy()
    
    # Set the title of the graph
    graph.SetTitle("Efficiency Curve")
    # Draw the graph with axis and lines
    graph.Draw("ALP")
    
    # Create a TMarker to add a pointer
    marker = ROOT.TMarker(pointer_x, pointer_y, 20)  # 20 is the marker style (full circle)
    marker.SetMarkerColor(ROOT.kRed)
    marker.SetMarkerSize(2)  # Adjust size as needed
    
    # Draw the marker
    marker.Draw()

    graph.SetLineColor(4)


    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(graph, "ML-base", "l")
    legend.AddEntry(marker, "cut-base", "l")
    legend.Draw()

    graph.GetXaxis().SetTitle("Signal Efficiency")
    graph.GetYaxis().SetTitle("BKG Efficiency")

    
    # Save the canvas as an image file
    canvas.SaveAs(out_path)

def plot_eff_curve(df, signal, model_name):
    print("process eff curve")
    df = predict_class(df, signal)
    signal_class = particle_2_class[signal]
    lowest_score = df.Filter(f'ParticleType == "{signal}"').Min(f'Prediction_{signal_class}').GetValue()


    total_bkg = df.Filter(f'ParticleType != "{signal}"').Count().GetValue()
    total_signal = df.Filter(f'ParticleType == "{signal}"').Count().GetValue()
    print("total signal: ", total_signal,"total bkg: ", total_bkg)


    eff_df = pd.DataFrame(columns=['bkg_eff', 'signal_eff'])

    score = lowest_score
    print(score)
    while score < 1.0:
        
        tmp_df = predict_signal(df, signal, score)

        signal_count = tmp_df.Filter(f'ParticleType == "{signal}" && signal == 1').Count().GetValue()
        bkg_count = tmp_df.Filter(f'ParticleType != "{signal}" && signal == 1').Count().GetValue()
        print(score,signal_count, bkg_count)
        
        signal_eff = signal_count / total_signal if total_signal > 0 else 0
        bkg_eff = bkg_count / total_bkg if total_bkg > 0 else 0


        eff_df.loc[len(eff_df.index)] = [bkg_eff,signal_eff]
        if score<0.8:
            score += 0.1
        elif score<0.9:
            score+=0.01
        elif score<0.97:
            score+=0.01
        else:
            score+=0.001

    pd.set_option('display.float_format', '{:.2e}'.format)


    signal_count = df.Filter(f'ParticleType == "{signal}" && stage2 == 1').Count().GetValue()
    bkg_count = df.Filter(f'ParticleType != "{signal}" && stage2 == 1').Count().GetValue()
        

    signal_eff = signal_count / total_signal if total_signal > 0 else 0
    bkg_eff = bkg_count / total_bkg if total_bkg > 0 else 0

    print(eff_df)
    out_path = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/{model_name}_eff_curve.png"
    plotting_eff_curve(eff_df, out_path, signal_eff,bkg_eff)

    #print(lowest_score)

def plotting_yield_curve(eff_df, out_path):
    
    pass

def cal_yield():

    pass

def cal_output_eff(args):
    #read output
    output_path = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{args.model}/output"
    print("reading output from", args)

    chain = ROOT.TChain("tree")
    file_count = 0
    for filename in os.listdir(output_path):
        if filename.endswith(".root"):
            chain.Add(os.path.join(output_path, filename))
        #if (file_count>0):
        #    break
        file_count+=1
    df = ROOT.RDataFrame(chain)
    #print_col(df)
    #cal model prediction class
    #df = prediction_results(df, args.signal)
    #get_rdf_info(df)
    
    #cal_eff(df, signal)
    cal_partition_eff(df, args.signal, args.model)
    #plot_eff_curve(df, args.signal, args.model)

    #eff = 0.1
    #df = predict_class(df, args.signal)
    #sol = optimize.root_scalar(find_score, args=(df,eff,args.signal), bracket=[0, 1], xtol=1e-06, method='brentq')

    #print(sol.root, sol.iterations, sol.function_calls)





    #print_col(df)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='v4_intRate_weight_recoTrack')
    parser.add_argument("-s", "--signal", dest="signal", default='vm')

    args = parser.parse_args()
    
    cal_output_eff(args)