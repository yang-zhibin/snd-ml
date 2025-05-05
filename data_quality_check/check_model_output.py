import ROOT
import pandas as pd
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

def predict_class_with_score(df, signal, score):
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

def main():
    corrupt_file = ['/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004449/model_baseline_output_real_data_2022_run_004449_sndsw_raw-0007.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004449/model_baseline_output_real_data_2022_run_004449_sndsw_raw-0009.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004503/model_baseline_output_real_data_2022_run_004503_sndsw_raw-0003.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004504/model_baseline_output_real_data_2022_run_004504_sndsw_raw-0003.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004504/model_baseline_output_real_data_2022_run_004504_sndsw_raw-0004.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004515/model_baseline_output_real_data_2022_run_004515_sndsw_raw-0000.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004527/model_baseline_output_real_data_2022_run_004527_sndsw_raw-0001.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004527/model_baseline_output_real_data_2022_run_004527_sndsw_raw-0011.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004527/model_baseline_output_real_data_2022_run_004527_sndsw_raw-0017.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004532/model_baseline_output_real_data_2022_run_004532_sndsw_raw-0003.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004532/model_baseline_output_real_data_2022_run_004532_sndsw_raw-0007.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0005.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0006.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0007.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0014.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0020.root',
                    '/eos/experiment/sndlhc/users/zhibin/real_data/2022/run_004541/model_baseline_output_real_data_2022_run_004541_sndsw_raw-0021.root',
                    '',
                    '',
                    '',
                    '',
                    ]
    metadata_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2022_metadata.csv'

    metadata_df = pd.read_csv(metadata_path)
    model_name = 'baseline'
    start=124
    for idx in range(start,len(metadata_df)):
        pred_path = metadata_df.loc[idx, f'model_{model_name}_output_path']
        if pred_path in corrupt_file:
            continue
        print(f"Processing {idx}: {pred_path}")
        rdf = ROOT.RDataFrame("snddata",pred_path)
        rdf = predict_class_with_score(rdf, 've', 0.99662)

        columns = rdf.GetColumnNames()
        print("Columns in RDataFrame:", [str(c) for c in columns])
         # Save histogram to file
        hist = rdf.Histo1D(("pred_hist", "Predicted Class;Class ID;Counts", 100, 0, 7), "PredClass")
        canvas = ROOT.TCanvas()
        hist.Draw()
        canvas.SaveAs("plot/pred_class_distribution.png")
        canvas.Clear()

        if idx>400:
            break


def delete_file():
    corrupt_file = ['/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/3/model_baseline_output_MC_neutrino_volTarget_100fb-1_3.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/23/model_baseline_output_MC_neutrino_volTarget_100fb-1_23.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/63/model_baseline_output_MC_neutrino_volTarget_100fb-1_63.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/64/model_baseline_output_MC_neutrino_volTarget_100fb-1_64.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/72/model_baseline_output_MC_neutrino_volTarget_100fb-1_72.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/83/model_baseline_output_MC_neutrino_volTarget_100fb-1_83.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/174/model_baseline_output_MC_neutrino_volTarget_100fb-1_174.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/178/model_baseline_output_MC_neutrino_volTarget_100fb-1_178.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/188/model_baseline_output_MC_neutrino_volTarget_100fb-1_188.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/194/model_baseline_output_MC_neutrino_volTarget_100fb-1_194.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/211/model_baseline_output_MC_neutrino_volTarget_100fb-1_211.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/215/model_baseline_output_MC_neutrino_volTarget_100fb-1_215.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/220/model_baseline_output_MC_neutrino_volTarget_100fb-1_220.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/221/model_baseline_output_MC_neutrino_volTarget_100fb-1_221.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/306/model_baseline_output_MC_neutrino_volTarget_100fb-1_306.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/315/model_baseline_output_MC_neutrino_volTarget_100fb-1_315.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/325/model_baseline_output_MC_neutrino_volTarget_100fb-1_325.root',
                    '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/354/model_baseline_output_MC_neutrino_volTarget_100fb-1_354.root',
                    ]

    for file_path in corrupt_file:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    main()
    #delete_file()