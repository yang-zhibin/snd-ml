import ROOT
import os
import argparse
import tqdm
import pandas as pd
from scipy import optimize

# Run in batch mode to avoid issues with canvas creation
ROOT.gROOT.SetBatch(True)

def create_histograms_by_pdgcode(cut_score, tree, out_dir, image_prefix):
    particle_mapping = {
        12: 've', -12: 've', 14: 'vm', -14: 'vm',
        16: 'vt', -16: 'vt', 2112: 'neutron', 2212: 'proton',
        130: 'kaon', 310: 'kaon0', 321: 'kaon+', -321: 'kaon-',
        112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC',
        116: 'NC', -116: 'NC', 230: '230'
    }

    histograms = {}
    histograms_cut = {}
    prediction_min = 0
    prediction_max = 1
    cut_value = cut_score

    

    for entry in tree:
        pdg_code = entry.PdgCode
        prediction = entry.Prediction
        particle = particle_mapping.get(pdg_code)

        if particle is None:
            print(f"{pdg_code} not in particle_mapping")
            continue

        if particle not in histograms:
            print(particle)
            histograms[particle] = ROOT.TH1F(f"hist_{image_prefix}", f"Prediction for {image_prefix}", 100, prediction_min, prediction_max)
            histograms_cut[particle] = ROOT.TH1F(f"hist_cut_{image_prefix}", f"Prediction for {image_prefix} with cut", 100, cut_value, prediction_max)

            # Set X and Y axis titles
            histograms[particle].GetXaxis().SetTitle("Score")
            histograms[particle].GetYaxis().SetTitle("Events")

            histograms_cut[particle].GetXaxis().SetTitle("Score")
            histograms_cut[particle].GetYaxis().SetTitle("Events")    

        histograms[particle].Fill(prediction)
        if prediction > cut_value:
            histograms_cut[particle].Fill(prediction)

    # Create and save the canvas
    canvas = ROOT.TCanvas("canvas", "Prediction Histograms by Particle", 800, 600)
    legend = ROOT.TLegend(0.75, 0.75, 0.9, 0.9)
    color_index = 2

    data_records = []

    for particle in histograms:
        hist = histograms[particle]
        hist_cut = histograms_cut[particle]
        #hist.SetStats(False)
        #hist_cut.SetStats(False)

        hist.GetXaxis().SetTitle("Score")
        hist_cut.GetYaxis().SetTitle("Events")
        efficiency = hist_cut.GetEntries() / hist.GetEntries() if hist.GetEntries() > 0 else 0
        eff_line = f"Efficiency for {particle}: {efficiency:.2E}, total:{hist.GetEntries()}, pass score cut: {hist_cut.GetEntries()}\n"
        print(eff_line)
        with open(f'{out_dir}/log.txt','a') as f:
            f.write(eff_line)
        hist.SetLineColor(color_index)
        hist.Draw("SAME" if color_index > 2 else "")
        legend.AddEntry(hist, f"{particle} ({efficiency:.2E})", "l")
        color_index += 1

                # Store data in a list of dictionaries
        data_records.append({
            "Partition": image_prefix,
            "Particle": particle,
            "Efficiency": efficiency,
            "Total Entries": hist.GetEntries(),
            "Passed Cut": hist_cut.GetEntries()
        })

    legend.Draw()
    image_name = f"{image_prefix}_prediction_histograms_by_pdgcode.png"
    image_file_path = os.path.join(out_dir, image_name)
    canvas.SaveAs(image_file_path)
    print("Histograms saved to", image_file_path)

    # Saving histograms with cuts
    canvas_cut = ROOT.TCanvas("canvas_cut", "Prediction Histograms with Cut by Particle", 800, 600)
    canvas_cut.Divide(2, 2)  # Adjust layout similarly
    i = 1
    for particle, histogram in histograms_cut.items():
        canvas_cut.cd(i)
        histogram.Draw()
        i += 1
    image2_path = os.path.join(out_dir, f"{image_prefix}_predictions_above_{cut_value}.png")
    canvas_cut.SaveAs(image2_path)
    print("Histograms saved to", image2_path)


    return data_records



def cal_all_eff(score, model, signal, outfile):

    pred_results = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model}/pred_output'
    particle_ranges = ['Neutrinos',
    "kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50",
    "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100",
    "neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50",
    "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90", "neutrons_90_100"
    ] 
    out_dir = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model}/eff_score{score}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    df_records = []
    for partition in particle_ranges:
        if 'kaon' in partition:
            particle = 'kaon'
        elif 'neutron' in partition:
            particle = 'neutron'
        else :
            particle = signal
        print(partition, pred_results)
        list_files = [
                os.path.join(pred_results, filename) 
                for filename in os.listdir(pred_results) 
                if partition in filename
            ]
        #print(list_files)
        print(list_files)
        df = ROOT.RDataFrame("tree",list_files)
        df = df.Define("ParticleType", """
        if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
        else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
        else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
        else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
        else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
        else if (PdgCode == 2112) return std::string("neutron");
        else if (PdgCode == 13 || PdgCode == -13) return std::string("muon");
        else return std::string("unknown");
        """)

        argmax_expr = """
        double vals[7] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5,Prediction_6};
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
        df = df.Define("predClass", argmax_expr)

        
        if particle == 'vm':
            select_class = 1
        elif particle == 've':
            select_class = 0
        # Filtering for "vm" particles where prediction is above the threshold
        filter_expr = f'ParticleType == "{particle}" && predClass == {select_class} && Prediction_{select_class} > ' + str(score)
        filtered_df = df.Filter(filter_expr)

    
        # Count entries matching the conditions
        total = df.Filter(f'ParticleType == "{particle}"').Count()
        filter_count = filtered_df.Count()


        # Calculate efficiency, making sure to access values properly
        eff = filter_count.GetValue() / total.GetValue() if total.GetValue() > 0 else 0
        
        df_records.append({
            "partition": partition,
            "particle": particle,
            "total_entries": total.GetValue(),
            "passed_cut": filter_count.GetValue(),
            "efficiency": eff,
        })
        print("{}, pass:{}, total:{} {:.2e}".format(partition,filter_count.GetValue(), total.GetValue(),eff))
        del df
    df = pd.DataFrame(df_records)
    print(df)
    df.to_csv(outfile, index=False)
    print(f"Efficiency data saved to CSV file {outfile}.")

particle_mapping = {
    12: 've', -12: 've',
    14: 'vm', -14: 'vm',
    16: 'vt', -16: 'vt',
    112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC', 116: 'NC', -116: 'NC',
    130: 'kaon', 310: 'kaon0',
    2112: 'neutron'
}
def map_particle_type(pdg_code):
    return particle_mapping.get(pdg_code, "unknown")

def save_filter_df(score, filter_outpath, model, signal):

    pred_results = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model}/pred_output'
    partition = 'Neutrinos'

    print("using reuslts from", pred_results)
    print("partition", partition)
    list_files = [
        os.path.join(pred_results, filename) 
        for filename in os.listdir(pred_results) 
        if partition in filename
    ]
    print('reading RDataFrame')
    df = ROOT.RDataFrame("tree",list_files)

    argmax_expr = """
    double vals[6] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5};
    int idx = 0;
    double max_val = vals[0];
    for (int i = 1; i < 6; ++i) {
        if (vals[i] > max_val) {
            max_val = vals[i];
            idx = i;
        }
    }
    return idx;
    """
    df = df.Define("predClass", argmax_expr)


    df = df.Define("ParticleType", """
    if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
    else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
    else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
    else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
    else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
    else if (PdgCode == 2112) return std::string("neutron");
    else return std::string("unknown");
    """)


    if signal == 'vm':
        select_class = 1
    elif signal == 've':
        select_class = 0
    # Filtering for "vm" particles where prediction is above the threshold
    filter_expr = f'ParticleType == "{signal}" && predClass == {select_class} && Prediction_{select_class} > {str(score)}' 
    filtered_df = df.Filter(filter_expr)

   
    # Count entries matching the conditions
    total = df.Filter(f'ParticleType == "{signal}"').Count()
    filter_count = filtered_df.Count()


    # Calculate efficiency, making sure to access values properly
    eff = filter_count.GetValue() / total.GetValue() if total.GetValue() > 0 else 0


    filtered_df.Snapshot("tree", filter_outpath)

def find_score(score=0.9):
    model = 'multiClass_weight_recoMuon_classWeight'
    signal = 'vm'

    pred_results = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model}/pred_output'
    partition = 'Neutrinos'

    print("using reuslts from", pred_results)
    print("partition", partition)
    list_files = [
        os.path.join(pred_results, filename) 
        for filename in os.listdir(pred_results) 
        if partition in filename
    ]
    print('reading RDataFrame')
    df = ROOT.RDataFrame("tree",list_files)

    argmax_expr = """
    double vals[6] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5};
    int idx = 0;
    double max_val = vals[0];
    for (int i = 1; i < 6; ++i) {
        if (vals[i] > max_val) {
            max_val = vals[i];
            idx = i;
        }
    }
    return idx;
    """
    df = df.Define("predClass", argmax_expr)


    df = df.Define("ParticleType", """
    if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
    else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
    else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
    else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
    else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
    else if (PdgCode == 2112) return std::string("neutron");
    else return std::string("unknown");
    """)


    if signal == 'vm':
        select_class = 1
    elif signal == 've':
        select_class = 0
        
    # Filtering particles where prediction is above the threshold
    filter_expr = f'ParticleType == "{signal}" && predClass == {select_class} && Prediction_{select_class} > ' + str(score)
    filtered_df = df.Filter(filter_expr)

   
    # Count entries matching the conditions
    total = df.Filter(f'ParticleType == "{signal}"').Count()
    filter_count = filtered_df.Count()



    # Calculate efficiency, making sure to access values properly
    eff = filter_count.GetValue() / total.GetValue() if total.GetValue() > 0 else 0

    print(score, eff)
    return eff - 0.10

def main():
    print("start process...")
    model = 'multiClass_weight_recoMuon_classWeight'
    signal = 'vm'
    sol = optimize.root_scalar(find_score, bracket=[0, 1], xtol = 1e-06, method='brentq')

    print(sol.root, sol.iterations, sol.function_calls)
    eff = 10 #%
    filter_outpath = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/score_cut_output/{signal}_{model}_{eff}.root"
    save_filter_df(sol.root, filter_outpath, model, signal)
    

    outfile = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/output/eff_{signal}_{model}_{eff}.csv'
    cal_all_eff(sol.root, model=model, signal=signal, outfile=outfile)

if __name__ == "__main__":
    #main()
    model = 'multi_new'
    signal = 'vm'
    eff = 10
    outfile = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/output/eff_{signal}_{model}_{eff}.csv'
    cal_all_eff(0.989, model=model, signal=signal, outfile=outfile)



    

    