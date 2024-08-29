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


def eff_yield_from_model(df,model,signal, score, eff_yield_df, concern_particles):

    particle_inteRate = {
        'vm': 157,    
        've': 157 / 0.72 * 0.23,   
        'muon': 5.48e5,  
    }

    pre_df = predict_signal(df, signal, score)
    eff_yield_df['model'].append(model)
    eff_yield_df['score'].append(score)
    eff_yield_df['signal'].append(signal)
    for particle in concern_particles:
        if particle in ['kaon', 'neutron']:
            continue  # Special handling for kaon and neutron

        event_pass = pre_df.Filter(f'signal==1 && ParticleType=="{particle}"').Count().GetValue()
        total_event = pre_df.Filter(f'ParticleType=="{particle}"').Count().GetValue()
        particle_eff = event_pass / total_event if total_event > 0 else 0

        particle_yield = particle_eff * particle_inteRate[particle]  
        #print(f'{particle}, {particle_inteRate[particle]}')

        eff_yield_df[f'{particle}_eff'].append(particle_eff)
        eff_yield_df[f'{particle}_yield'].append(particle_yield)

    bkgs = ['neutron','kaon']
    intRate = {
    "neutron": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "kaon": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1],
    }
    bins = [5,10,20,30,40,50,60,70,80,90]
    
    for bkg in bkgs:
        bkg_yield = 0
        event_pass = pre_df.Filter(f'signal==1 && ParticleType=="{bkg}"').Count().GetValue()
        total_event = pre_df.Filter(f'ParticleType=="{bkg}"').Count().GetValue()
        bkg_eff = event_pass / total_event if total_event > 0 else 0

        bkg_df = pre_df.Filter(f'ParticleType=="{bkg}"')
        bkg_df = bkg_df.Define('energy', 'sqrt(px*px + py*py + pz*pz)')

        #print(f"{bkg}, {event_pass}, {total_event},{bkg_eff}")

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

        eff_yield_df[f'{bkg}_eff'].append(bkg_eff)
        eff_yield_df[f'{bkg}_yield'].append(bkg_yield)

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    output = []
    for key, value_list in eff_yield_df.items():
        if value_list:  # Check if the list is not empty
            last_element = value_list[-1]
            output.append(f"{key}: {last_element}")
        else:
            output.append(f"{key}: empty")

    print(f"{current_time}, {output}")

    signal_eff =  eff_yield_df[f'{signal}_eff'][-1]
    return signal_eff



    

def adaptive_step_search(df,model, signal, eff_yield_df, concern_particles,start_value=0, end_value=1, initial_step=0.1, tolerance=0.02):
    print('start adaptive_step_search')
    input_value = start_value
    step_size = initial_step

    eff_list = []
    yield_list = []


    previous_signal_eff= eff_yield_from_model(df,model,signal,input_value, eff_yield_df,concern_particles)
    input_value += step_size

    while previous_signal_eff > 1e-6:
        #print("in step",df)
        signal_eff = eff_yield_from_model(df,model,signal,input_value, eff_yield_df,concern_particles)
        

        # Adjust step size based on the rate of change of x
        if (abs(signal_eff - previous_signal_eff) > tolerance and abs(signal_eff - previous_signal_eff) < (tolerance*5)):
            step_size = max(step_size / 8, 1e-6)  # Decrease step size if change is rapid
            input_value += step_size
        elif abs(signal_eff - previous_signal_eff) > (tolerance*5):
            input_value -= step_size
            step_size = max(step_size / 10, 1e-6) 
            input_value += step_size
            signal_eff = eff_yield_from_model(df,model,signal,input_value, eff_yield_df,concern_particles)

            input_value += step_size
        else:
            step_size = min(step_size * 1.4, 0.1)  # Increase step size if change is slow
            input_value += step_size
        #print(input_value)

        previous_signal_eff = signal_eff

        

    return 0

def cal_eff_yield(model_output_path, model, signal, eff_yield_df, concern_particles):
    print("reading output from", model_output_path)

    chain = ROOT.TChain("tree")
    file_count = 0
    for filename in os.listdir(model_output_path):
        #print(filename)
        #if filename.endswith(".root") and (filename.startswith("test_muon_real") or filename.startswith("test_0_neutrino_output")) :
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

    adaptive_step_search(df, model, signal, eff_yield_df, concern_particles, start_value=lowest_score)


    return 0

def process(model_list, version, signal):

    concern_particles = ['ve', 'vm', 'kaon', 'neutron', 'muon']

    eff_yield_df = {
        'model': [],
        'signal': [],
        'score': []
    }
    for particle in concern_particles:
        eff_yield_df[f'{particle}_eff'] = []
        eff_yield_df[f'{particle}_yield'] = []

    print(eff_yield_df)
    for model in model_list:
        model_output_path = f'/eos/user/z/zhibin/sndData/converted/pt/output/{model}/'
        cal_eff_yield(model_output_path, model, signal, eff_yield_df, concern_particles)


    #print(eff_yield_df)
    return eff_yield_df


def main(model, signal):

    #model_list = ['baseline', 'weight','normalized_weight', 'intRate_weight','intRate_weightX100','intRate_weightX100^2']
    model_list = [model]
    version = ''

    eff_yield_df = process(model_list, version, signal)

    df = pd.DataFrame(eff_yield_df)
    print(df)
    csv_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_v2/{signal}_{model}_eff_yield_df.csv'
    df.to_csv(csv_path, index=False)

    #out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline')
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    args = parser.parse_args()
    main(args.model, args.signal)
    #plot(args.signal)

