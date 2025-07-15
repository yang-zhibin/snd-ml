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



def split_prediction_clause(expr: str) -> Tuple[int, float, str]:
    """
    Extracts (prediction_class, prediction_cut, other_cuts) from a boolean expression.

    If the expression contains no 'Prediction_<class> > <value>' clause,
    it returns (0, 0.0, expr) unchanged.
    """
    if expr is None:
        return 0, 0.0, ""
    # 1  Find the Prediction clause (if any)
    m = re.search(r'\bPrediction_(\d+)\s*>\s*([0-9]*\.?[0-9]+)', expr)

    if not m:
        # --- default return when clause is absent ---
        return 0, 0.0, expr.strip()

    # 2  Normal extraction
    prediction_class = int(m.group(1))
    prediction_cut   = float(m.group(2))

    # 3  Remove the clause and tidy leftovers
    start, end = m.span()
    other_cuts_raw = (expr[:start] + expr[end:]).strip()

    # collapse duplicated ampersands (& clean edges)
    other_cuts = re.sub(r'(\s*&{2}\s*){2,}', ' && ', other_cuts_raw).strip(' &')

    return prediction_class, prediction_cut, other_cuts


def cal_matrix(rdf, true_class, cuts):
    pred_class = ["ve", "vm", "vt", "NC", "kaon", "neutron", "muon"]
    matrices = {}  # Store confusion matrix for each cut

    #print(list(rdf.GetColumnNames()))

    for cut in cuts:
        prediction_score_class, prediction_score_cut, other_cuts = split_prediction_clause(cut)
        print(f"other_cuts: {other_cuts}, prediction_score_class: {prediction_score_class}, prediction_score_cut: {prediction_score_cut}")
        if cut is None:
            rdf_cut = rdf
            label = 'no_cut'
        else:
            if other_cuts == "":
                rdf_cut = rdf
            else:
                rdf_cut = rdf.Filter(other_cuts)
            label = cut

        confusion_matrix = pd.DataFrame(0.0, index=true_class, columns=pred_class)

        for t_class in true_class:
            rdf_class = rdf_cut
            #print(f"t_class: {t_class}")
            if t_class == "data_veto_tagged":
                rdf_class = rdf_class.Filter("(veto1 + veto2 + veto3) > 0") #
                t_class_ParticleType = 'real_data'
            elif t_class == "data_zero_veto":
                rdf_class = rdf_class.Filter("(veto1 + veto2 + veto3) == 0")
                t_class_ParticleType = 'real_data'
            else:
                t_class_ParticleType = t_class

            for p_class in pred_class:
                
                p_class_number = particle_2_class[p_class]
                
                if p_class_number == prediction_score_class:
                    filter_expr = (
                        f'(ParticleType == "{t_class_ParticleType}" && '
                        f'pred_class_first == {p_class_number} && '
                        f'(Prediction_{prediction_score_class} > {prediction_score_cut}))'
                    )
                else:
                    filter_expr = (
                        f'(ParticleType == "{t_class_ParticleType}" && '
                        f'pred_class_first == {p_class_number}) || '
                        f'(ParticleType == "{t_class_ParticleType}" && '
                        f'pred_class_second == {p_class_number} && '
                        f'pred_class_first == {prediction_score_class} && '
                        f'Prediction_{prediction_score_class} <= {prediction_score_cut})'
                    )
                    

                print(f"filter_expr : {filter_expr}")
                if not isinstance(filter_expr, str):
                    raise ValueError(f"Invalid filter expression: {filter_expr}")
                if (t_class == "data_zero_veto" and (p_class=='ve' or p_class=='vm' or p_class=='vt' or p_class=='NC')):
                    pred_count = np.nan
                else:
                    pred_count = rdf_class.Filter(filter_expr).Count().GetValue()
                #print(f"    p_class: {p_class}, count: {pred_count}")
                confusion_matrix.at[t_class, p_class] = pred_count
        #print(f"cut: {label}",confusion_matrix)
        matrices[label] = confusion_matrix
    print(matrices)
    #save_to_csv(matrices)
    return matrices

def process(eval_path,feature_path,pred_path, data_type, cuts, outpath):
    if "real_data" in data_type:
        true_class = ['data_veto_tagged', 'data_zero_veto']

        eval_chain = ROOT.TChain("snddata")
        feature_chain = ROOT.TChain("snddata")
        pred_chain = ROOT.TChain("snddata")

        eval_chain.Add(eval_path)
        feature_chain.Add(feature_path)
        pred_chain.Add(pred_path)

        eval_chain.AddFriend(feature_chain, 'featureTree')
        eval_chain.AddFriend(pred_chain, 'predTree')
        rdf = ROOT.RDataFrame(eval_chain)

        
        rdf = ROOT.RDataFrame(eval_chain)
        matrices = cal_matrix(rdf, true_class, cuts)
    else:
        if 'kaon' in data_type:
            true_class = ['kaon']
        elif 'neutron' in data_type:
            true_class = ['neutron']
        elif 'neutrino' in data_type:
            true_class = ["ve", "vm", "vt", "NC"]
        elif 'muon' in data_type:
            true_class = ['muon']
        eval_chain = ROOT.TChain("snddata")
        pred_chain = ROOT.TChain("snddata")
        eval_chain.Add(eval_path)
        pred_chain.Add(pred_path)

        eval_chain.AddFriend(pred_chain, 'predTree')
        rdf = ROOT.RDataFrame(eval_chain)
        
        matrices = cal_matrix(rdf, true_class, cuts)

    save_to_csv(matrices, outpath)
    print("RDataFrame run times: ",rdf.GetNRuns())

def get_data_type(feature_path):
    if 'MC_kaon' in feature_path:
        data_type = 'MC_kaon'
    elif 'MC_neutron' in feature_path:
        data_type = 'MC_neutron'
    elif 'MC_muon' in feature_path:
        data_type = 'MC_muon'
    elif 'MC_neutrino' in feature_path:
        data_type = 'MC_neutrino'
    elif 'real_data' in feature_path:
        data_type = 'real_data'
    else:
        data_type = 'unknown'

    return data_type


def main(args):
    pred_path = args.pred
    feature_path = args.feature
    eval_path = args.eval
    outpath = args.output


    data_type = get_data_type(feature_path)

    cuts = [None]
    
    for threshold in np.arange(0.5, 0.99, 0.02):
        cut_name = f"Prediction_0 > {threshold}"
        cuts.append(cut_name)
        
    for threshold in range(0, 601, 50):
        cut_name = f"scifi_gt_{threshold}"
        cuts.append(cut_name)
        
    cuts += [
        "ds4_eq_0",
        "ds34_eq_0",
        "ds234_eq_0",
        "ds1234_eq_0",
        "us5_eq_0",
        "us45_eq_0",
        "us345_eq_0",
        "us2345_eq_0",
        "us12345_eq_0"
    ]
    
    hit_pairs = [
        (200, 1), (250, 1), (300, 1), (350, 1), (400, 1),
        (200, 2), (250, 2), (300, 2), (350, 2), (400, 2),
    ]
    
    for x, y in hit_pairs:
        cut_name = f"scifi_gt_{x}_us1_gt_{y}"
        cuts.append(cut_name)

    cuts += ["scifi_gt_300_us1_gt_2 && Prediction_0 > 0.92"]
    print(cuts)

    process(eval_path, feature_path, pred_path, data_type, cuts, outpath)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred", dest="pred", help="prediction output")
    parser.add_argument("-f", "--feature", dest="feature", help='feature file')
    parser.add_argument("-e", "--eval", dest="eval", help="eval output")
    parser.add_argument("-o", "--output", dest="output", help='output path')
    args = parser.parse_args()
    main(args)


#python cal_matrix.py -p {params.model_output} -f {params.feature_path} -e {params.eval_path} -o "${{tmp_output}}"