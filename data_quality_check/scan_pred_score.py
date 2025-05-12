import ROOT
import pandas as pd
import numpy as np
import os

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}


def predict_class_with_score(df):
    print("predicting results")
    df = df.Define("ParticleType", """
        if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
        else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
        else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
        else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
        else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
        else if (PdgCode == 2112) return std::string("neutron");
        else if (PdgCode == 13 || PdgCode == -13) return std::string("muon");
        else if (PdgCode == 0 ) return std::string("real_data");
        else return std::string("others");
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

def read_exist_output(dir_data, metadata_data_df):
    def file_exists(row):
        file_path = row['model_baseline_muon_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def read_metadata():
    metadata_data_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    metadata_data_df = pd.read_csv(metadata_data_path)
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'

    metadata_data_df = read_exist_output(dir_MC, metadata_data_df)

    return metadata_data_df

def cal_pred_results(df, score):
    pred_class = ["ve", "vm", "vt", "NC"]
    confusion_matrix = pd.DataFrame(0.0, index=pred_class, columns=pred_class)
    for t_class in pred_class:
        df_true = df.Filter(f'ParticleType=="{t_class}"')
        for p_class in pred_class:
            p_class_number = particle_2_class[p_class]
            if (p_class=='ve'):
                df_pred = df_true.Filter(f'PredClass=={p_class_number} && Prediction_{p_class_number} > {score}')
            else:
                df_pred = df_true.Filter(f'PredClass=={p_class_number}')
            pred_count = df_pred.Count().GetValue()
            confusion_matrix.at[t_class, p_class] = pred_count

    print(confusion_matrix)
    return confusion_matrix

def scan_pred_score(df):

    results = []
    for score in np.arange(0.9, 1.01, 0.01):
        result = cal_pred_results(df, score)
        results.append(result)
    
        #break

def main():
    metadata_df = read_metadata()
    model_name = 'baseline_muon'


    prediction_chain = ROOT.TChain("snddata")

    count = 0
    for index, row in metadata_df.iterrows():
        pred_path = row[f'model_{model_name}_output_path']
        prediction_chain.Add(pred_path)

        #if count>311:
        #    break
        count+=1


    rdf = ROOT.RDataFrame(prediction_chain)
    rdf = predict_class_with_score(rdf)

    scan_pred_score(rdf)

if __name__ == "__main__":
    main()