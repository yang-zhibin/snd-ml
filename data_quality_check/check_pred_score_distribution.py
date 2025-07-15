import ROOT
import pandas as pd
import os
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
class_2_particle = {v: k for k, v in particle_2_class.items()}

colors = {
    0: ROOT.kRed,
    1: ROOT.kBlue,
    2: ROOT.kGreen+2,
    3: ROOT.kMagenta,
    4: ROOT.kOrange+7,
    5: ROOT.kCyan,
    6: ROOT.kBlack,
}

def plot_hist(rdf, output_file_path):
    canvas = ROOT.TCanvas("canvas", "Prediction Scores", 800, 600)
    legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    any_drawn = False
    hist_proxies = []

    for label, class_id in particle_2_class.items():
        # Filter for true class
        df_filtered = rdf.Filter(f"ParticleClass == {class_id}")
        n_events = df_filtered.Count().GetValue()
        if n_events == 0:
            continue

        # Histogram of prediction score for this class
        hist_proxy = df_filtered.Histo1D(
            (f"hist_{label}", f"Prediction Score for {label}", 50, 0, 1),
            "Prediction_0"
        )
        hist = hist_proxy.GetValue()

        hist.SetStats(False)
        hist_proxies.append(hist_proxy)
        hist.SetMarkerStyle(20)
        hist.SetMarkerSize(0.5)
        hist.SetMarkerColor(colors[class_id])
        hist.Scale(1.0 / hist.Integral())
        hist.SetLineColorAlpha(colors[class_id], 0.9)
        hist.SetLineWidth(2)

        hist.SetFillColorAlpha(colors[class_id], 0.1)

        hist.GetXaxis().SetTitle("Prediction Score")
        hist.GetYaxis().SetTitle("Probability Density")
        hist.Draw("HIST E1 SAME" if any_drawn else "HIST E1")

        legend.AddEntry(hist, f'{label} ({n_events} events)', "l")
        any_drawn = True

    if any_drawn:
        legend.Draw()
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        canvas.SaveAs(output_file_path)
        print(f'File saved in: {output_file_path}')
    else:
        print("No valid histograms to draw. Skipping canvas save.")



def plot_hist_wrong_classify(rdf, output_file_path):
    canvas = ROOT.TCanvas("canvas", "Prediction Scores", 800, 600)
    legend = ROOT.TLegend(0.5, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    any_drawn = False
    hist_proxies = []

    for label, class_id in particle_2_class.items():
        # Filter for true class
        #df_filtered = rdf.Filter(f"ParticleClass == {class_id}")
        #n_events = df_filtered.Count().GetValue()
        #if n_events == 0:
        #    continue

        # Histogram of prediction score for this class
        hist_proxy = rdf.Histo1D(
            (f"hist_{class_id}", f"Prediction Score", 50, 0, 1),
            f"Prediction_{class_id}"
        )
        hist = hist_proxy.GetValue()

        #hist.SetStats(False)
        hist_proxies.append(hist_proxy)
        hist.SetMarkerStyle(20)
        hist.SetMarkerSize(0.5)
        hist.SetMarkerColor(colors[class_id])
        hist.Scale(1.0 / hist.Integral())
        hist.SetLineColorAlpha(colors[class_id], 0.9)
        hist.SetLineWidth(2)

        hist.SetFillColorAlpha(colors[class_id], 0.1)

        hist.GetXaxis().SetTitle("Prediction Score")
        hist.GetYaxis().SetTitle("Probability Density")
        hist.GetYaxis().SetRangeUser(0, 0.2)
        hist.Draw("HIST E1 SAME" if any_drawn else "HIST E1")

        legend.AddEntry(hist, f"Prediction_{class_id} ({label} score)", "l")
        any_drawn = True

    if any_drawn:
        legend.Draw()
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        canvas.SaveAs(output_file_path)
        print(f'File saved in: {output_file_path}')
    else:
        print("No valid histograms to draw. Skipping canvas save.")



def plot_hist_gap(rdf, output_file_path):
    canvas = ROOT.TCanvas("canvas", "Prediction Scores", 800, 600)
    legend = ROOT.TLegend(0.5, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    any_drawn = False
    hist_proxies = []

    for label, class_id in particle_2_class.items():
        # Filter for true class
        if class_id == 0: 
            continue
        df_filtered = rdf.Filter(f"ParticleClass == {class_id}")
        n_events = df_filtered.Count().GetValue()
        if n_events == 0:
            continue

        # Histogram of prediction score for this class
        df_filtered = df_filtered.Define('score_gap', f"Prediction_0 - Prediction_{class_id}")
        hist_proxy = df_filtered.Histo1D(
            (f"hist_{class_id}", f"Prediction Score", 50, 0, 1),
            f"score_gap"
        )
        hist = hist_proxy.GetValue()
        if hist.Integral() == 0:
            continue
        hist.SetStats(False)
        hist_proxies.append(hist_proxy)
        hist.SetMarkerStyle(20)
        hist.SetMarkerSize(0.5)
        hist.SetMarkerColor(colors[class_id])

        
        hist.Scale(1.0 / hist.Integral())
        hist.SetLineColorAlpha(colors[class_id], 0.9)
        hist.SetLineWidth(2)

        hist.SetFillColorAlpha(colors[class_id], 0.1)

        hist.GetXaxis().SetTitle("Prediction Score Gap")
        hist.GetYaxis().SetTitle("Probability Density")
        hist.GetYaxis().SetRangeUser(0, 0.1)
        hist.Draw("HIST E1 SAME" if any_drawn else "HIST E1")

        legend.AddEntry(hist, f"Score gap of {label} ({n_events} events)", "l")
        any_drawn = True

    if any_drawn:
        legend.Draw()
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        canvas.SaveAs(output_file_path)
        print(f'File saved in: {output_file_path}')
    else:
        print("No valid histograms to draw. Skipping canvas save.")

def print_event(chain):
    n_entries = chain.GetEntries()

    # Loop through entries
    for i in range(n_entries):
        chain.GetEntry(i)
        pdg = chain.pdgCode
        particle = chain.ParticleType
        
        # Access branches directly as attributes
        pred_class_first = chain.pred_class_first
        ParticleClass = chain.ParticleClass
        pred_particle_first = class_2_particle.get(pred_class_first, 'Unknown')

        #print(f"event index: {i}, particle: {particle}, pdg: {pdg}, predict as {pred_particle_first}, RunId: {chain.runId}, EventId: {chain.eventId}")
        if pred_class_first == 0 and ParticleClass == 1:
            print(f"event index: {i}, particle: {particle}, pdg: {pdg}, predict as {pred_particle_first}, RunId: {chain.runId}, EventId: {chain.eventId}, orignal index: {chain.eventId-1}")
            print(f"    RunId: {chain.runId}, EventId: {chain.eventId}")

        
def main():
    neutrino_csv_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    df = pd.read_csv(neutrino_csv_path)  

    # 2. Select feature paths
    paths = df["eval_baseline_muon_output_path"].tolist()


    # 3. Create TChain
    chain = ROOT.TChain("snddata")
    for path in paths:
        chain.Add(path)
        print(path)
        break

    # 4. Read into RDataFrame
    rdf = ROOT.RDataFrame(chain)
    #chain.GetListOfBranches().Print()
    #pred_ve_rdf = rdf.Filter("pred_class_first == 0")
    #plot_hist(pred_ve_rdf, f"plot/score_hist_pred_ve.pdf")

    print_event(chain)

    # vm_pred_ve_rdf = rdf.Filter("pred_class_first == 0 && ParticleClass == 1 ")
    # plot_hist_wrong_classify(vm_pred_ve_rdf, f"plot/score_hist_vm_pred_ve.pdf")
    
    #ve_pred_ve_rdf = rdf.Filter("pred_class_first == 0 && pred_class_first == 0 ")
    #plot_hist_wrong_classify(ve_pred_ve_rdf, f"plot/score_hist_ve_pred_ve.pdf")

    # NC_pred_ve_rdf = rdf.Filter("pred_class_first == 0 && ParticleClass == 3 ")
    # plot_hist_wrong_classify(NC_pred_ve_rdf, f"plot/score_hist_NC_pred_ve.pdf")

    # pred_ve_rdf = rdf.Filter("pred_class_first == 0")
    # plot_hist_gap(pred_ve_rdf, f"plot/score_hist_pred_ve_gap.pdf")


if __name__ == "__main__": 
    main()