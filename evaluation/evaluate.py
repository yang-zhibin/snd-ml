
import ROOT
import argparse


# mode 
#0. check output cooruption
#1.physics cuts, cut name list
#2.score cuts, score range of each class 

# load metadata

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}
def pdg_2_particle(rdf):
    rdf = rdf.Define("ParticleType", """
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

    rdf = rdf.Define("ParticleClass", """
        if (ParticleType == "ve") return 0;
        else if (ParticleType == "vm") return 1;
        else if (ParticleType == "vt") return 2;
        else if (ParticleType == "NC") return 3;
        else if (ParticleType == "kaon") return 4;
        else if (ParticleType == "neutron") return 5;
        else if (ParticleType == "muon") return 6;
        else return -1;  // e.g. for "real_data" or "others"
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
    rdf = rdf.Define("PredClass", argmax_expr)
    return rdf

def main(args):

    print('evaluating prediction results...')
    feature_path = args.feature
    pred_path = args.pred
    out_file = args.output


    feature_chain = ROOT.TChain("snddata")
    prediction_chain = ROOT.TChain("snddata")

    feature_chain.Add(feature_path)
    prediction_chain.Add(pred_path)
    feature_chain.AddFriend(prediction_chain, 'predTree')

    rdf = ROOT.RDataFrame(feature_chain)
    #fiducial cuts
    #
    cuts = {
        "scifi_gt_10": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 10",
        "scifi_gt_30": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 30",
        "scifi_gt_50": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 50",
        "scifi_gt_70": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 70",
        "scifi_gt_90": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 90",
        "ds_fudicial": "DS_avg_ver >=70 && DS_avg_ver <=105 && DS_avg_hor >=10 && DS_avg_hor<=50",
        "scifi_fudicial": "scifi_avg_ver >=200 && scifi_avg_ver <=1200 && scifi_avg_hor >=300 && scifi_avg_hor<=1336",
        "no_hit_scifi1": "scifi1==0",
        "no_hit_scifi2": "scifi2==0",
    }

    score_cuts = {}

    columns_to_keep = ["ParticleType", "ParticleClass", "eventId", "runId", "pdgCode"]

    rdf = pdg_2_particle(rdf)

    for cut_name, cut_expr in cuts.items():
        print(f'defining {cut_name}')
        rdf = rdf.Define(f"{cut_name}", cut_expr)
        columns_to_keep.append(cut_name)
    
    print('saving output file...')
    rdf.Snapshot('snddata', out_file, columns_to_keep)
    print(f'saved to out_file')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred", dest="pred", help="prediction output")
    parser.add_argument("-f", "--feature", dest="feature", help='feature file')
    parser.add_argument("-o", "--output", dest="output", help='output path')
    args = parser.parse_args()
    main(args)

#python evaluate.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/model_baseline_output_MC_neutrino_volTarget_100fb-1_0.root -f /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/feature_MC_neutrino_volTarget_100fb-1_0.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/eval_baseline_MC_neutrino_volTarget_100fb-1_0.root




