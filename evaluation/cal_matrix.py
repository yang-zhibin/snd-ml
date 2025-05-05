import pandas as pd
import argparse
import os
import ROOT

import logging

# Configure logging
logging.basicConfig(
    filename="cal_matrix.log",  # Log file name
    level=logging.INFO,   # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
)

#ROOT.ROOT.DisableImplicitMT()
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

def check_ids(feature_chain):

    for i in range(feature_chain.GetEntries()):
        
        feature_chain.GetEntry(i)
        evt_id = getattr(feature_chain, "eventId")
        run_id = getattr(feature_chain, "runId")
        evt_id2 = getattr(feature_chain, "predTree.EventId")
        run_id2 = getattr(feature_chain, "predTree.RunId")

        if (evt_id !=evt_id2) or (run_id!=run_id2):
            print(f"id not the same, before evtid:{evt_id}, runid:{run_id}, after evtid:{evt_id2}, runid:{run_id2}")
        #else:
            #print(f"id are the same, before evtid:{evt_id}, runid:{run_id}, after evtid:{evt_id2}, runid:{run_id2}")

        #break


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
    # argmax_expr = f"""
    #     double vals[7] = {{Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5, Prediction_6}};
    #     int idx = {signal_class};
    #     double max_val = vals[{signal_class}];
    #     if (max_val > {score} ) return idx;
    #     else{{
    #         max_val = 0;
    #         for (int i = 0; i < 7; ++i) {{
    #             if (i=={signal_class}) continue;
    #             if (vals[i] > max_val) {{
    #                 max_val = vals[i];
    #                 idx = i;
    #             }}
    #         }}
    #         return idx;
    #     }}
    #     """
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


def get_pred_list(pred_prefix):
    prefix, max_chunk = pred_prefix.rsplit("_chunk_", 1)
    max_chunk = int(max_chunk.split(".")[0])
    pred_list = [f"{prefix}_chunk_{i}.root" for i in range(max_chunk + 1)]
    return pred_list


def cal_matrix(df, true_class):
    print("predicting", true_class)
    pred_class = ["ve", "vm", "vt", "NC", "kaon", "neutron", "muon"]
    confusion_matrix = pd.DataFrame(0.0, index=true_class, columns=pred_class)
    for t_class in true_class:
        df_true = df.Filter(f'ParticleType=="{t_class}"')
        true_count = df_true.Count().GetValue()
        if (t_class=='vt'):
            print(f'vt true count:{true_count}')

        if true_count == 0:
            print("skipping",t_class,0)
            continue
        for p_class in pred_class:
            p_class_number = particle_2_class[p_class]
            #print(p_class_number)
            if (p_class=='ve'):
                df_pred = df_true.Filter(f'PredClass=={p_class_number} && Prediction_{p_class_number} > 0.99662')
            else:
                df_pred = df_true.Filter(f'PredClass=={p_class_number}')
            pred_count = df_pred.Count().GetValue()
            if (p_class=='vt'):
                print(f'mc {t_class} predecting vt, pred_count:{pred_count}')
            confusion_matrix.at[t_class, p_class] = pred_count
    return confusion_matrix


def process(rdf, real_data_lumi, mc_neutrino_lumi):
    #filter rdf (veto-inverted, cut flow)
    mc_neutrino = rdf.Filter("isMC == 1")
    veto_inverted = rdf.Filter("(isMC == 0)&&((veto1+veto2+veto3)>0)")
    signal_region = rdf.Filter("(isMC == 0)&&((veto1+veto2+veto3)<1)")
    print("veto_inverted",veto_inverted.Count().GetValue())
    #print("signal_region",signal_region.Count().GetValue())
    print("mc_neutrino",mc_neutrino.Count().GetValue()/mc_neutrino_lumi*real_data_lumi)
    #cal matrix
    mc_neutrino_matrix = cal_matrix(mc_neutrino, ["ve", "vm", "vt", "NC"])
    veto_inverted_matrix = cal_matrix(veto_inverted, ['real_data'])
    signal_region_matrix = cal_matrix(signal_region, ['real_data'])
    
    signal_region_matrix.at["real_data", "ve"] = None
    signal_region_matrix.at["real_data", "vm"] = None
    signal_region_matrix.at["real_data", "vt"] = None
    signal_region_matrix.at["real_data", "NC"] = None

    veto_inverted_matrix=veto_inverted_matrix.rename(index={'real_data': 'veto_inverted'})
    signal_region_matrix = signal_region_matrix.rename(index={'real_data': 'signal_region'})
    mc_neutrino_matrix.index = ["MC_" + str(idx) for idx in mc_neutrino_matrix.index]

    
    mc_neutrino_matrix = mc_neutrino_matrix/mc_neutrino_lumi*real_data_lumi
    veto_inverted_matrix = veto_inverted_matrix
    
    signal_region_matrix = signal_region_matrix
    #print(mc_neutrino_matrix)
    #print(veto_inverted_matrix)
    #print(signal_region_matrix)

    return pd.concat([mc_neutrino_matrix, veto_inverted_matrix,signal_region_matrix], axis = 0)

def cut_flow(rdf):
    rdf_cut = rdf.Filter(
        'DS_avg_ver >=70 && DS_avg_ver <=105 && DS_avg_hor >=10 && DS_avg_hor<=50 &&'
        'scifi_avg_ver >=200 && scifi_avg_ver <=1200 && scifi_avg_hor >=300 && scifi_avg_hor<=1336'
        )

    return rdf_cut




def read_exist_output(dir_data, metadata_data_df):
    def file_exists(row):
        file_path = row['model_baseline_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df


def read_metadata():
    metadata_data_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2022_metadata.csv'
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    #metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volMuFilter_20fb-1_metadata.csv'

    metadata_data_df = pd.read_csv(metadata_data_path)
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path)

    dir_data = '/eos/experiment/sndlhc/users/zhibin/real_data/'
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'

    metadata_data_df = read_exist_output(dir_data, metadata_data_df)
    metadata_mc_neutrino_df = read_exist_output(dir_MC, metadata_mc_neutrino_df)


    
    return metadata_data_df, metadata_mc_neutrino_df



def main(args):
    # read feature and prediction to chains seperated
       
    metadata_data_df, metadata_mc_neutrino_df= read_metadata()
    #print(metadata_mc_neutrino_df['model_baseline_output_path'])

    metadata_df = pd.concat([metadata_mc_neutrino_df], ignore_index=True)
    #print(metadata_df)
    model_name = args.model

    feature_chain = ROOT.TChain("snddata")
    prediction_chain = ROOT.TChain("snddata")

    #print(metadata_df)
    count = 0
    for index, row in metadata_df.iterrows():
        
        feature_path = row['feature_path']
        pred_path = row[f'model_{model_name}_output_path']
        #print(pred_path, feature_path)

        if not(os.path.isfile(pred_path)) or not((os.path.isfile(feature_path))):
            continue
        feature_chain.Add(feature_path)
        prediction_chain.Add(pred_path)

    
        #if count>311:
        #    break
        count+=1

    feature_chain.AddFriend(prediction_chain, 'predTree')

    rdf = ROOT.RDataFrame(feature_chain)
    rdf = predict_class_with_score(rdf, args.signal, args.threshold)

    # hist = rdf.Histo1D(("pred_hist", "Predicted Class;Class ID;Counts", 100, 0, 7), "PredClass")

    # # Save histogram to file
    # canvas = ROOT.TCanvas()
    # hist.Draw()
    # canvas.SaveAs("plot/pred_class_distribution.png")


    columns = rdf.GetColumnNames()
    print("Columns in RDataFrame:", [str(c) for c in columns])



    real_data_lumi = 1.871e+02
    # = metadata_data_df['lumi_per_file'].sum()
    print(F'lumi use in this matrix: {real_data_lumi:.4e}fb-1')
    mc_neutrino_lumi = 100*len(metadata_mc_neutrino_df)
    veto_ineff = 5.0e-7


    # no cut
    print('not cut')
    confusion_matrix = process(rdf,real_data_lumi, mc_neutrino_lumi)
    confusion_matrix.loc['veto_inverted'] = confusion_matrix.loc['veto_inverted'] * veto_ineff

    print(confusion_matrix)

    #with fiduil cut
    rdf = cut_flow(rdf)

    print()
    print()
    print('after fiducial cut')
    confusion_matrix2 = process(rdf,real_data_lumi, mc_neutrino_lumi)
    confusion_matrix2.loc['veto_inverted'] = confusion_matrix2.loc['veto_inverted'] * veto_ineff
    print(confusion_matrix2)


    print("RDataFrame run times: ",rdf.GetNRuns())






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", help='model name', default='baseline')
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    parser.add_argument("-t", "--threshold", dest="threshold", default=0.99662)
    args = parser.parse_args()
    main(args)