import ROOT
import os
import argparse
import tqdm
import pandas as pd

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



def main(args):
    pred_results = '/afs/cern.ch/user/z/zhibin/work/snd-ml/outputs/root_output_v2/'
    #partition = args.partition
    particle_ranges = ['Neutrinos',
    "kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50",
    "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100",
    "neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50",
    "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90", "neutrons_90_100"
    ] 
    #partition = 'kaons_5_10'
    cut_score = 0.989
    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/outputs/plots_v2/plots_{}/'.format(cut_score)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    df_records = []
    for partition in particle_ranges:
        print(partition)
        with open(f'{out_dir}/log.txt','a') as f:
            f.write(f"processing {partition}\n")

        list_files = [
                os.path.join(pred_results, filename) 
                for filename in os.listdir(pred_results) 
                if partition in filename
            ]
        #print(list_files)
        tchain = ROOT.TChain("tree")
        for file_path in list_files:
            tchain.Add(file_path) 

        
        image_prefix = partition

        records = create_histograms_by_pdgcode(cut_score,tchain,out_dir,image_prefix)
        df_records.extend(records)
    df = pd.DataFrame(df_records)
    print(df)
    df.to_csv(os.path.join(out_dir, "efficiency_summary.csv"), index=False)
    print("Efficiency data saved to CSV file.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-p", "--partition", dest="partition", default='Neutrinos')
    parser.add_argument("-s", "--score", dest="score", default=0.9)
    args = parser.parse_args()
    main(args)
    