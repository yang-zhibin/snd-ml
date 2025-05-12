
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

def cal_scan_fiducial_area():

    scifi_tl_xs = list(range(-46, -40, 1)) 
    scifi_tl_ys = list(range(53, 47, -1))  
    DS_tl_xs = list(range(-61, -45, 3))    
    DS_tl_ys = list(range(67, 56, -2))     

    scifi_br_xs = list(range(-7, -16, -1)) 
    scifi_br_ys = list(range(14, 23))      
    DS_br_xs = list(np.arange(1, -9.1, -1.25).tolist())      
    DS_br_ys = list(np.arange(8, 18.1, 1.25).tolist())      
    
    fiducial_tl_exprs = []
    for i in range(len(scifi_tl_xs)):
        scifi_tl_x = scifi_tl_xs[i]
        scifi_tl_y = scifi_tl_ys[i]
        DS_tl_x = DS_tl_xs[i]
        DS_tl_y = DS_tl_ys[i]
        fiducial_tl_expr = f'scifi_avg_x_pos >={scifi_tl_x} && scifi_avg_y_pos <= {scifi_tl_y} && DS_avg_x_pos >={DS_tl_x} && DS_avg_y_pos <= {DS_tl_y} '
        fiducial_tl_exprs.append(fiducial_tl_expr)
    
    fiducial_br_exprs = []
    for i in range(len(scifi_br_xs)):
        scifi_br_x = scifi_br_xs[i]
        scifi_br_y = scifi_br_ys[i]
        DS_br_x = DS_br_xs[i]
        DS_br_y = DS_br_ys[i]
        fiducial_br_expr = f'scifi_avg_x_pos <={scifi_br_x} && scifi_avg_y_pos >= {scifi_br_y} && DS_avg_x_pos <={DS_br_x} && DS_avg_y_pos >= {DS_br_y} '
        fiducial_br_exprs.append(fiducial_br_expr)

    return fiducial_tl_exprs, fiducial_br_exprs

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
        "scifi_gt_100": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 100",
        "scifi_gt_300": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 300",
        "scifi_gt_500": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 500",
        "scifi_gt_700": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 700",
        "scifi_gt_900": "(scifi1 + scifi2 + scifi3 + scifi4 + scifi5) > 900",
        "fiducial_0": "DS_avg_ver >=70 && DS_avg_ver <=105 && DS_avg_hor >=10 && DS_avg_hor<=50 && scifi_avg_ver >=200 && scifi_avg_ver <=1200 && scifi_avg_hor >=300 && scifi_avg_hor<=1336",
    }   

    fiducial_tl_exprs, fiducial_br_exprs = cal_scan_fiducial_area()

    fiducial_tl_cuts = {
        f"fiducial_tl_{i}": expr
        for i, expr in enumerate(fiducial_tl_exprs, 1)
    }

    fiducial_br_cuts = {
        f"fiducial_br_{i}": expr
        for i, expr in enumerate(fiducial_br_exprs, 1)
    }

    cuts.update(fiducial_tl_cuts)
    cuts.update(fiducial_br_cuts)

    columns_to_keep = ["ParticleType", "ParticleClass", "eventId", "runId", "pdgCode", "PredClass"]

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




