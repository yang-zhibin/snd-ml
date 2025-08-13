
import ROOT
import argparse
import pandas as pd
import numpy as np


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


    rdf = rdf.Define("pred_sorted_indices", """
        std::vector<int> sorted_indices = [&]() {
            std::vector<std::pair<int, double>> preds = {
                {0, Prediction_0}, {1, Prediction_1}, {2, Prediction_2},
                {3, Prediction_3}, {4, Prediction_4}, {5, Prediction_5},
                {6, Prediction_6}
            };
            std::sort(preds.begin(), preds.end(),
                [](const auto &a, const auto &b) { return a.second > b.second; });
            std::vector<int> indices;
            for (auto &p : preds) indices.push_back(p.first);
            return indices;
        }();
        return sorted_indices;
    """)

    for i, name in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]):
        rdf = rdf.Define(f"pred_class_{name}", f"pred_sorted_indices[{i}]")

    return rdf

def process_eval(args):

    print('evaluating prediction results...')
    pred_path = args.pred
    out_file = args.eval_output

    prediction_chain = ROOT.TChain("sndData")
    prediction_chain.Add(pred_path)
    rdf = ROOT.RDataFrame(prediction_chain)
    
    columns_to_keep = [
        "ParticleType", "ParticleClass", "EventId", "RunId", "PdgCode"
    ]

    # Add the new prediction rank columns
    rank_names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
    columns_to_keep += [f"pred_class_{name}" for name in rank_names]

    # Add prediction score columns
    num_classes = 7 
    columns_to_keep += [f"Prediction_{i}" for i in range(num_classes)]

    rdf = pdg_2_particle(rdf)
    
    print('saving output file...')
    rdf.Snapshot('snddata', out_file, columns_to_keep)
    print(f'saved to {out_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred", dest="pred", help="prediction output")
    parser.add_argument("-o", "--eval_output", dest="eval_output", help='eval output path')
    args = parser.parse_args()
    
    process_eval(args)

#python evaluate.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_prediction_baseline_muon_output_MC_neutrino_volTarget_100fb-1_0.root -e /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_eval_baseline_muon_output_MC_neutrino_volTarget_100fb-1_0.root




