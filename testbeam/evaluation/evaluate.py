
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
    'e-': 0,
    'pi': 1,
    # 'pi-': 2,
    # 'muon': 3,
}


def pdg_2_particle(rdf):
    rdf = rdf.Define("ParticleType", """
        if (PdgCode == 11 || PdgCode == -11) return std::string("e-");
        else if (PdgCode == 211 || PdgCode == -211) return std::string("pi");
        else if (PdgCode == 13 || PdgCode == -13) return std::string("muon");
        else if (PdgCode == 0 ) return std::string("real_data");
        else return std::string("others");
    """)

    rdf = rdf.Define("ParticleClass", """
        if (ParticleType == "e-") return 0;
        else if (ParticleType == "pi") return 1;
        else if (ParticleType == "pi-") return 2;
        else if (ParticleType == "muon") return 3;
        else return -1;  // e.g. for "real_data" or "others"
    """)


    rdf = rdf.Define("pred_sorted_indices", """
        std::vector<int> sorted_indices = [&]() {
            std::vector<std::pair<int, double>> preds = {
                {0, Prediction_0}, {1, Prediction_1}
            };
            std::sort(preds.begin(), preds.end(),
                [](const auto &a, const auto &b) { return a.second > b.second; });
            std::vector<int> indices;
            for (auto &p : preds) indices.push_back(p.first);
            return indices;
        }();
        return sorted_indices;
    """)

    for i, name in enumerate(["first", "second"]):
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
    rank_names = ["first", "second"]
    columns_to_keep += [f"pred_class_{name}" for name in rank_names]

    # Add prediction score columns
    num_classes = 2
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




