import ROOT
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}


def read_metadata():
    metadata_paths = {
        "neutrino": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv',
        #"kaon": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_kaon_FTFP_BERT_metadata.csv',
        #"neutron": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_neutron_FTFP_BERT_metadata.csv',
        #"real_data_2024": '/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/real_data_2024_metadata.csv'
    }

    cleaned_metadata = {}
    path_column = "eval_baseline_muon_output_path" 

    for key, file_path in metadata_paths.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, nrows = 10) # debug
            print(df)
            if path_column in df.columns:
                df = df[df[path_column].apply(os.path.exists)]
                cleaned_metadata[key] = df
            else:
                print(f"Column '{path_column}' not found in {file_path}")
        else:
            print(f"Metadata file not found: {file_path}")
    
    return cleaned_metadata

def save_to_csv(matrices):
    combined_csv_path = './csv/all_confusion_matrices.csv'
    combined_df_list = []

    for label, matrix in matrices.items():
        # Add a 'cut' column to identify the matrix this row came from
        matrix_with_cut = matrix.copy()
        matrix_with_cut.insert(0, 'cut', label)
        combined_df_list.append(matrix_with_cut)

    # Concatenate all matrices into one DataFrame
    combined_df = pd.concat(combined_df_list)
    combined_df.to_csv(combined_csv_path, index=True)


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
    save_to_csv(matrices)
    return matrices


def process_row(row, cuts):
    eval_path = row['eval_baseline_muon_output_path']
    feature_path = row['feature_path']
    lumi_per_file = row['lumi_per_file']
    data_type = row['data_type']
    pred_path = row['model_baseline_muon_output_path']

    
    if "real_data" in data_type:
        veto_ineff = row.get('veto_ineff', None)
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
        cal_matrix(rdf, true_class, cuts)
    else:
        if 'kaon' in data_type:
            true_class = ['kaon']
        elif 'neutron' in data_type:
            true_class = ['neutron']
        elif 'neutrino' in data_type:
            true_class = ["ve", "vm", "vt", "NC"]
        eval_chain = ROOT.TChain("snddata")
        pred_chain = ROOT.TChain("snddata")
        eval_chain.Add(eval_path)
        pred_chain.Add(pred_path)

        eval_chain.AddFriend(pred_chain, 'predTree')
        rdf = ROOT.RDataFrame(eval_chain)

        cal_matrix(rdf, true_class, cuts)

    print("RDataFrame run times: ",rdf.GetNRuns())



def process_exist_metadata(exist_metadata):
    for name, df in exist_metadata.items():
        print(f"Dataset: {name}, Entries: {len(df)}")

        matrix_path = f"./csv/matrix_{name}.csv"

        # Load existing matrix if it exists
        if os.path.exists(matrix_path):
            matrix_df = pd.read_csv(matrix_path, index_col=0)
            existing_ids = set(matrix_df.index)
        else:
            matrix_df = pd.DataFrame()
            existing_ids = set()

        
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


        updated_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            #if idx in existing_ids:
            #    continue  # Skip if already processed
            
            results = process_row(row, cuts)
            print(results)
            # Store the result with index = idx
            updated_rows.append((idx, results))

            break

        if updated_rows:
            updated_df = pd.DataFrame(
                data=[r[1] for r in updated_rows],
                index=[r[0] for r in updated_rows]
            )
            matrix_df = pd.concat([matrix_df, updated_df])
            matrix_df.to_csv(matrix_path)
            print(f"Updated matrix saved: {matrix_path}")
        else:
            print(f"No new entries to process for {name}")

def main():
    # plot exist data
    # loop over file to compute matrixs and save to csv, do not change index, keep it for checking if the matrix is saved before compute the matrixs

    exist_metadata = read_metadata()
    process_exist_metadata(exist_metadata)



if __name__ == "__main__": 
    main()
