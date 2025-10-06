import ROOT
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import argparse
import re
from typing import Tuple

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}
particle_to_target = {
        12: 0, -12: 0,
        14: 1, -14: 1,
        16: 2, -16: 2,
        112: 3, -112: 3, 114: 3, -114: 3, 116: 3, -116: 3,
        130: 4, 310: 4,
        2112: 5,
        13:6, -13:6,
        0:6
}
class_2_particle = {v: k for k, v in particle_2_class.items()}

def save_to_csv(matrices, outpath):
    combined_df_list = []

    for label, matrix in matrices.items():
        # Add a 'cut' column to identify the matrix this row came from
        matrix_with_cut = matrix.copy()
        matrix_with_cut.insert(0, 'cut', label)
        combined_df_list.append(matrix_with_cut)

    # Concatenate all matrices into one DataFrame
    combined_df = pd.concat(combined_df_list)
    combined_df.to_csv(outpath, index=True)
    print(f"matrices save to {outpath}")

def _class_to_pdg_list(t_class):
    """t_class can be either label ('ve') or numeric class id (0..6)."""
    if isinstance(t_class, str):
        cls_id = particle_2_class[t_class]
    else:
        cls_id = int(t_class)
    return [pdg for pdg, c in particle_to_target.items() if c == cls_id]

def _pdg_filter_clause(pdg_list, field="pdgCode"):
    """Build ROOT RDataFrame filter for a list of PDG codes."""
    if not pdg_list:
        # No PDGs for this class -> impossible match
        return "0"
    # e.g. "(pdgCode==12 || pdgCode==-12 || ...)"
    parts = [f"{field}=={pdg}" for pdg in pdg_list]
    return "(" + " || ".join(parts) + ")"

def cal_matrix(rdf, true_class):
    
    ## pre selection:
        # if (branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] ==1 and branch_vars["cut_G_has_consecutive_scifi_hits"][0] == 1 and branch_vars["scifi"][0]>5 and branch_vars["veto"][0] == 0):
        # branch_vars["preSelect_vetoFree"][0] = 1
        
    # a. raw file event count
    # b. non_vetoHit > 0
    
    # 1. scifi>200 
    # 2. veto==0

    
    cut_expr = [
        "",  # a: raw
        "At_least_1_non_veto_hit",  # b
        "At_least_1_non_veto_hit && scifi>200",  # 1
        "At_least_1_non_veto_hit && scifi>200 && veto==0",  # 2
    ]
    labels = ['a_raw', 'b_non_veto', '1_scifi>200',  '2_veto0']

    matrices = {}

    for label, base_expr in zip(labels, cut_expr):
        df = pd.DataFrame(0, index=true_class, columns=['count'], dtype=float)

        for t_class in true_class:
            if "data" in t_class:
                pdg_list = [0]
            else:
                pdg_list = _class_to_pdg_list(t_class)
            pdg_clause = _pdg_filter_clause(pdg_list)

            print(f"{t_class=}, {pdg_list=}, {pdg_clause=}")

            # Combine cut + PDG clause safely
            if base_expr:
                filter_expr = f"({base_expr}) && {pdg_clause}"
                count = rdf.Filter(filter_expr).Count().GetValue()
            else:
                # raw: only PDG selection
                count = rdf.Filter(pdg_clause).Count().GetValue()

            df.at[t_class, 'count'] = float(count)

        matrices[label] = df

    print(matrices)
    return matrices
            
        
        
        
    

def get_data_type(feature_path):
    print(feature_path)
    if 'MC_kaon' in feature_path:
        data_type = 'MC_kaon'
    elif 'MC_neutron' in feature_path:
        data_type = 'MC_neutron'
    elif 'MC_muon' in feature_path:
        data_type = 'MC_muon'
    elif 'MC_neutrino' in feature_path:
        data_type = 'MC_neutrino'
    elif 'real_data' in feature_path:
            data_type = 'data'
    else:
        data_type = 'unknown'

    return data_type


def process(preSelect_path, outpath, data_type):

    if 'kaon' in data_type:
        true_class = ['kaon']
    elif 'neutron' in data_type:
        true_class = ['neutron']
    elif 'neutrino' in data_type:
        true_class = ["ve", "vm", "vt", "NC"]
    elif 'muon' in data_type:
        true_class = ['muon']
    elif "data" in data_type:
        true_class = ["data"]
        
    
    preSelect_chain = ROOT.TChain("sndData")
    preSelect_chain.Add(preSelect_path)

    rdf = ROOT.RDataFrame(preSelect_chain)
    matrices = cal_matrix(rdf, true_class)


    save_to_csv(matrices, outpath)
    print("RDataFrame run times: ",rdf.GetNRuns())

   


def main(args):
    preSelect_path = args.preSelect
    outpath = args.output    
    data_type = get_data_type(preSelect_path)
    
    process(preSelect_path, outpath, data_type)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preSelect", dest="preSelect", help='preSelect file')
    parser.add_argument("-o", "--output", dest="output", help='output path')
    args = parser.parse_args()
    main(args)


#python cal_matrix.py -f {params.feature_path} -e {params.eval_path} -o "${{tmp_output}}"
#python cal_matrix.py -f /eos/experiment/sndlhc/users/zhibin/real_data/run_241/run_008285/vetoTagged_feature_real_data_run_241_run_008285_sndsw_raw-0000.root -e /eos/experiment/sndlhc/users/zhibin/real_data/run_241/run_008285/vetoTagged_eval_GravNet_v2_output_real_data_run_241_run_008285_sndsw_raw-0000.root -o /eos/experiment/sndlhc/users/zhibin/real_data/run_241/run_008285/vetoTagged_matrix_baseline_muon_output_real_data_run_241_run_008285_sndsw_raw-0000.csv