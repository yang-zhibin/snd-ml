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

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
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
        else if (PdgCode == 13 || PdgCode == -13 || PdgCode == 0) return std::string("muon");
        else return std::string("unknown");
        """)
    argmax_expr = """
        double vals[7] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5, Prediction_6};
        int idx = 0;
        double max_val = vals[0];
        for (int i = 1; i < 7; ++i) {
            if (vals[i] > max_val) {
                max_val = vals[i];
                idx = i;
            }
        }
        return idx;
        """
    df = df.Define("PredClass", argmax_expr)


    return df

def predict_class_with_score(df, signal, score):
    print("predicting results")
    df = df.Define("ParticleType", """
        if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
        else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
        else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
        else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
        else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
        else if (PdgCode == 2112) return std::string("neutron");
        else if (PdgCode == 13 || PdgCode == -13 || PdgCode == 0) return std::string("muon");
        else return std::string("others");
        """)
        #
    signal_class = particle_2_class[signal]
    argmax_expr = f"""
        double vals[7] = {{Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5, Prediction_6}};
        int idx = {signal_class};
        double max_val = vals[{signal_class}];
        if (max_val > {score} ) return idx;
        else{{
            max_val = 0;
            for (int i = 0; i < 7; ++i) {{
                if (i=={signal_class}) continue;
                if (vals[i] > max_val) {{
                    max_val = vals[i];
                    idx = i;
                }}
            }}
            return idx;
        }}
        
        """

    df = df.Define("PredClass", argmax_expr)
    # pred_particle_expr = """
    #     if (PredClass == 0) return std::string("ve");
    #     else if (PredClass == 1) return std::string("vm");
    #     else if (PredClass == 2) return std::string("vt");
    #     else if (PredClass == 3) return std::string("NC");
    #     else if (PredClass == 4) return std::string("kaon");
    #     else if (PredClass == 5) return std::string("neutron");
    #     else if (PredClass == 6) return std::string("muon");
    #     else return std::string("others");
    # """

    # df = df.Define("PredParticles", pred_particle_expr)

    return df

def predict_signal(df, signal, score=0):
    signal_class = particle_2_class[signal]
    if_signal = f'PredClass == {signal_class} && Prediction_{signal_class} > {str(score)}'
    
    pre_df = df.Define('signal', if_signal)

    #print("predicting signal with score ", score, pre_df.Filter(f'signal==1 && ParticleType=="{signal}"').Count().GetValue())
    return pre_df

def compute_matrix(df):
    labels = ["ve", "vm", "vt", "NC", "kaon", "neutron", "muon"]
    confusion_matrix = pd.DataFrame(0, index=labels, columns=labels)
    #confusion_matrix.index = ['true_' + label for label in labels]
    #confusion_matrix.columns = ['pred_' + label for label in labels]

    for ture_p in particle_2_class:
        for pred_p in particle_2_class:
            pred_p_class = particle_2_class[pred_p]
            #print(ture_p,pred_p_class)
            count = df.Filter(f'ParticleType=="{ture_p}" && PredClass == {pred_p_class}').Count().GetValue()
            confusion_matrix.at[ture_p, pred_p] = count
    print(confusion_matrix)
    return confusion_matrix

def compute_matrix_with_sklearn(df):
    
    column_data = {}
    column_data['ParticleType'] = list(df.Take['string']('ParticleType').GetValue())
    column_data['PredClass'] = list(df.Take['int']('PredClass').GetValue())

    #print(column_data)
    data = pd.DataFrame(column_data)
    true_labels = data['ParticleType'].map(particle_2_class)
    predicted_labels = data["PredClass"]
    #pdg = data['PdgCode']

    print(f"true:{true_labels.value_counts()},sum:{true_labels.value_counts().sum()}, pred:{predicted_labels.value_counts()}, sum{predicted_labels.value_counts().sum()}")
    # conf_matrix = pd.crosstab(true_labels, predicted_labels)
    conf_matrix = sklearn_confusion_matrix(true_labels, predicted_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=particle_2_class.keys(), 
                              columns=particle_2_class.keys())
    print(conf_matrix)

    return conf_matrix_df

def cal_expected(df_pred,intRate,bins,true_count):
    df_pred = df_pred.Define('energy', 'sqrt(px*px + py*py + pz*pz)')
    total_expected = 0
    for index in range(len(bins)-1):
        lower = bins[index]
        upper = bins[index+1]
        filtered_df = df_pred.Filter(f'(energy>{lower}) && (energy<={upper})')
        pass_count = filtered_df.Count().GetValue()
        eff = pass_count/true_count
        expected = eff*intRate[index]
        total_expected+=expected
    return total_expected
    # bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')

    # for index in range(len(bins)-1):
    #     lower = bins[index]
    #     upper = bins[index+1]
    #     energy = (lower+upper) / 2

    #     filtered_df = bkg_df.Filter(f'(energy>{lower}) && (energy<={upper})')

    #     kaon_total = filtered_df.Filter(f'ParticleType=="kaon"').Count().GetValue()
    #     kaon_pass = filtered_df.Filter(f'signal==1 && ParticleType=="kaon"').Count().GetValue()
    #     kaon_eff = kaon_pass/kaon_total if kaon_total > 0 else 0
    #     kaon_yield = kaon_eff * intRate['kaon'][index]
    #     kaon_error_low, kaon_error_upp = proportion_confint(kaon_pass, kaon_total, alpha = 1-0.68,method='beta')
    #     kaon_error_low *= intRate['kaon'][index]
    #     kaon_error_upp *= intRate['kaon'][index] 

    #     neutron_total = filtered_df.Filter(f'ParticleType=="kaon"').Count().GetValue()
    #     neutron_pass = filtered_df.Filter(f'signal==1 && ParticleType=="kaon"').Count().GetValue()
    #     neutron_eff = neutron_pass/neutron_total if neutron_total > 0 else 0
    #     neutron_yield = neutron_eff * intRate['neutron'][index]
    #     neutron_error_low, neutron_error_upp = proportion_confint(neutron_pass,neutron_total, alpha = 1-0.68, method='beta')
    #     neutron_error_low *= intRate['neutron'][index]
    #     neutron_error_upp *= intRate['neutron'][index]

    #     new_row = [energy, neutron_total, neutron_pass, neutron_eff, neutron_yield, neutron_error_low, neutron_error_upp, 
    #                         kaon_total, kaon_pass, kaon_eff, kaon_yield, kaon_error_low, kaon_error_upp]
    #     bkg_energy_df.loc[len(bkg_energy_df)] = new_row

    #     print(new_row)

def compute_matrix_with_expected(df):
    #ve:55690, vm:181185, vt:3298,NC:77658,
    particle_inteRate = {
        'vm': 157,    
        've': 157 / 181185 * 55690,   
        'vt': 157 / 181185 * 3298,
        'NC': 157 / 181185 * 77658,
        'muon': 5.48e5,  
    }

    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1],
    }
    bins = [5,10,20,30,40,50,60,70,80,90]

    labels = ["ve", "vm", "vt", "NC", "kaon", "neutron", "muon"]
    confusion_matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    #confusion_matrix.index = ['true_' + label for label in labels]
    #confusion_matrix.columns = ['pred_' + label for label in labels]

    for ture_p in particle_2_class:
        df_true = df.Filter(f'ParticleType=="{ture_p}"')
        true_count = df_true.Count().GetValue()
        if true_count == 0:
            print(ture_p,0)
            continue
        for pred_p in particle_2_class:
            pred_p_class = particle_2_class[pred_p]
            df_pred = df_true.Filter(f'PredClass == {pred_p_class}')
            if(ture_p=='neutron'):
                expected = cal_expected(df_pred,intRate['neutron'],bins, true_count)
                confusion_matrix.at[ture_p, pred_p] = expected
            elif(ture_p=='kaon'):
                expected = cal_expected(df_pred,intRate['kaon'],bins, true_count)
                confusion_matrix.at[ture_p, pred_p] = expected
            else:
                count = df_pred.Count().GetValue()
                confusion_matrix.at[ture_p, pred_p] = count/true_count * particle_inteRate[ture_p]

    print(confusion_matrix)
    return confusion_matrix

    # bkg_df = pre_df.Filter(f'ParticleType=="kaon" || ParticleType=="neutron"')
    
def plot_avg_channel(rdf, out_dir):
    # Create 2D histograms for scifi and DS variables
    hist_scifi = rdf.Histo2D(("hist_scifi", "SciFi Avg Ver vs Hor", 100, 0, 1600, 100, 0, 1600), "scifi_avg_ver", "scifi_avg_hor")
    hist_ds = rdf.Histo2D(("hist_ds", "DS Avg Ver vs Hor", 100, 60, 120, 100, 0, 60), "DS_avg_ver", "DS_avg_hor")

    # Create canvases and save the histograms
    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    hist_scifi.Draw("COLZ")
    c1.SaveAs(f"{out_dir}/scifi_avg_ver_vs_hor_2.png")

    c2 = ROOT.TCanvas("c2", "Canvas", 800, 600)
    hist_ds.Draw("COLZ")
    c2.SaveAs(f"{out_dir}/DS_avg_ver_vs_hor_2.png")

def plot_xy(rdf, out_dir):
    hist_xy = rdf.Histo2D(("hist_xy", "Hit XY distribution", 100, 0, 1600, 100, 0, 1600), "Hits.x", "Hits.y")

    # Create canvases and save the histograms
    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    hist_xy.Draw("COLZ")
    c1.SaveAs(f"{out_dir}/xy.png")

def process(df, signal, score):
    #rdf = df.Filter(f'PredClass==0')
    rdf = df
    print(rdf.GetColumnNames())

    #rdf = rdf.Filter('PredClass == 1')

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot_muon/'
    
    #plot_avg_channel(rdf, out_dir)
    plot_xy(rdf, out_dir)
        
    return 0



def main(model, signal, score):
    #model_list = ['baseline', 'weight','normalized_weight', 'intRate_weight','intRate_weightX100','intRate_weightX100^2']
    
    
    model_output_path = f'/eos/user/z/zhibin/sndData/converted/pt/output/{model}/'

    print("reading output from", model_output_path)

    chain = ROOT.TChain("tree")
    file_count = 0
    for filename in os.listdir(model_output_path):
        #print(filename)
        #if filename.endswith(".root") and (filename.startswith("test_3_") or filename.startswith("test_0_neutrino_output")) :
        if filename.endswith("real_muon_2_output.root"):
            print(os.path.join(model_output_path, filename), ' read')
            chain.Add(os.path.join(model_output_path, filename))
        #if (file_count>0):
        #    break
        file_count+=1

    df = ROOT.RDataFrame(chain)

    df = predict_class(df)
    #df = predict_class_with_score(df, signal,score)

    process(df, signal, score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline_muon')
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    parser.add_argument("-t", "--threshold", dest="threshold", default=0.99661)
    args = parser.parse_args()
    main(args.model, args.signal, args.threshold)

