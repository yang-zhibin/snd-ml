import ROOT
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import argparse

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



def cal_matrix(rdf, true_class, cuts):
    pred_class = ["ve", "vm", "vt", "NC", "kaon", "neutron", "muon"]
    matrices = {}  # Store confusion matrix for each cut

    #print(list(rdf.GetColumnNames()))

    for cut in cuts:
        if cut is None:
            is_score_cut = False
            rdf_cut = rdf
            label = 'no_cut'
        else:
            is_score_cut = 'Prediction_' in cut
            rdf_cut = rdf if is_score_cut else rdf.Filter(cut)
            label = cut

        # Apply cut filter if not a score cut
        # if is_score_cut or (cut is None):
        #     rdf_cut = rdf
        # else:
        #     rdf_cut = rdf.Filter(cut)
        #     #print(f"cut {cut}, before filter count: {rdf.Count().GetValue()}, after filter count: {rdf_cut.Count().GetValue()},")

        confusion_matrix = pd.DataFrame(0.0, index=true_class, columns=pred_class)

        for t_class in true_class:
            rdf_class = rdf_cut
            #print(f"t_class: {t_class}")
            if t_class == "veto_inverted":
                rdf_class = rdf_class.Filter("(veto1 + veto2 + veto3) > 0") #
                t_class_ParticleType = 'real_data'
            elif t_class == "signal_region":
                rdf_class = rdf_class.Filter("(veto1 + veto2 + veto3) == 0")
                t_class_ParticleType = 'real_data'
            else:
                t_class_ParticleType = t_class

            for p_class in pred_class:
                
                p_class_number = particle_2_class[p_class]

                if p_class == 've' and is_score_cut:
                    filter_expr = f'ParticleType == "{t_class_ParticleType}" && PredClass == {p_class_number} && {cut}'
                    #print(filter_expr)
                else:
                    filter_expr = f'ParticleType == "{t_class_ParticleType}" && PredClass == {p_class_number}'

                if (t_class == "signal_region" and (p_class=='ve' or p_class=='vm' or p_class=='vt' or p_class=='NC')):
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
        true_class = ['veto_inverted', 'signal_region']

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
        elif 'muon' in [data_type]:
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

    cuts = [
            None,
            'Prediction_0 > 0.85',
            'Prediction_0 > 0.9',
            'Prediction_0 > 0.95',
            'fiducial_0',
            'fiducial_tl_1',
            'fiducial_tl_2',
            'fiducial_tl_3',
            'fiducial_tl_4',
            'fiducial_tl_5',
            'fiducial_tl_6',
            'fiducial_br_1',
            'fiducial_br_2',
            'fiducial_br_3',
            'fiducial_br_4',
            'fiducial_br_5',
            'fiducial_br_6',
            'fiducial_br_7',
            'fiducial_br_8',
            'fiducial_br_9',
        ]

    process(eval_path, feature_path, pred_path, data_type, cuts, outpath)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred", dest="pred", help="prediction output")
    parser.add_argument("-f", "--feature", dest="feature", help='feature file')
    parser.add_argument("-e", "--eval", dest="eval", help="eval output")
    parser.add_argument("-o", "--output", dest="output", help='output path')
    args = parser.parse_args()
    main(args)
