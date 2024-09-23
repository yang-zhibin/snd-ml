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
        else if (PdgCode == 13 || PdgCode == -13 || PdgCode == 0) return std::string("muon");
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

def process(df, signal, score):

    columns = ['energy','neutron_total','neutron_pass' ,'neutron_eff', 'neutron_yield', 'neutron_error_low','neutron_error_upp','kaon_total','kaon_pass','kaon_eff', 'kaon_yield', 'kaon_error_low','kaon_error_upp']
    bkg_energy_df = pd.DataFrame(columns=columns)

    particle_inteRate = {
        'vm': 157,    
        've': 157 / 0.72 * 0.23,   
        'muon': 5.48e5,  
    }

    pre_df = predict_signal(df, signal, score)

    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1],
    }
    bins = [5,10,20,30,40,50,60,70,80,90]

    bkg_df = pre_df.Filter(f'ParticleType=="kaon" || ParticleType=="neutron"')
    bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')

    for index in range(len(bins)-1):
        lower = bins[index]
        upper = bins[index+1]
        energy = (lower+upper) / 2

        filtered_df = bkg_df.Filter(f'(energy>{lower}) && (energy<={upper})')

        kaon_total = filtered_df.Filter(f'ParticleType=="kaon"').Count().GetValue()
        kaon_pass = filtered_df.Filter(f'signal==1 && ParticleType=="kaon"').Count().GetValue()
        kaon_eff = kaon_pass/kaon_total if kaon_total > 0 else 0
        kaon_yield = kaon_eff * intRate['kaon'][index]
        kaon_error_low, kaon_error_upp = proportion_confint(kaon_pass, kaon_total, alpha = 1-0.68,method='beta')
        kaon_error_low *= intRate['kaon'][index]
        kaon_error_upp *= intRate['kaon'][index] 

        neutron_total = filtered_df.Filter(f'ParticleType=="kaon"').Count().GetValue()
        neutron_pass = filtered_df.Filter(f'signal==1 && ParticleType=="kaon"').Count().GetValue()
        neutron_eff = neutron_pass/neutron_total if neutron_total > 0 else 0
        neutron_yield = neutron_eff * intRate['neutron'][index]
        neutron_error_low, neutron_error_upp = proportion_confint(neutron_pass,neutron_total, alpha = 1-0.68, method='beta')
        neutron_error_low *= intRate['neutron'][index]
        neutron_error_upp *= intRate['neutron'][index]

        new_row = [energy, neutron_total, neutron_pass, neutron_eff, neutron_yield, neutron_error_low, neutron_error_upp, 
                            kaon_total, kaon_pass, kaon_eff, kaon_yield, kaon_error_low, kaon_error_upp]
        bkg_energy_df.loc[len(bkg_energy_df)] = new_row

        print(new_row)

    
    return bkg_energy_df



def main(model, signal, score):

    #model_list = ['baseline', 'weight','normalized_weight', 'intRate_weight','intRate_weightX100','intRate_weightX100^2']
    
    
    model_output_path = f'/eos/user/z/zhibin/sndData/converted/pt/output/{model}/'

    print("reading output from", model_output_path)

    chain = ROOT.TChain("tree")
    file_count = 0
    for filename in os.listdir(model_output_path):
        #print(filename)
        #if filename.endswith(".root") and (filename.startswith("test_3_") or filename.startswith("test_0_neutrino_output")) :
        if filename.endswith(".root"):
            print(os.path.join(model_output_path, filename), ' read')
            chain.Add(os.path.join(model_output_path, filename))
        #if (file_count>2):
        #    break
        file_count+=1
    df = ROOT.RDataFrame(chain)

    df = predict_class(df)

    bkg_energy = process(df, signal, score)

    print("RDataFrame run times: ",df.GetNRuns())

    out_df = pd.DataFrame(bkg_energy)
    print(out_df)
    csv_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_bkg_energy/{signal}_{model}_bkg_energy_df.csv'
    out_df.to_csv(csv_path, index=False)

    #out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/'

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline')
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    parser.add_argument("-t", "--threshold", dest="threshold", default=0.994)
    args = parser.parse_args()
    main(args.model, args.signal, args.threshold)

