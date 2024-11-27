import ROOT
import os
import pandas as pd

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

def draw_detId(df, hitType='Digi_MuFilterHits'):
    min_value = df.Min(f"{hitType}.fDetectorID").GetValue()
    max_value = df.Max(f"{hitType}.fDetectorID").GetValue()
    histogram = df.Histo1D(("hist_detector_id", f"Distribution of {hitType}.fDetectorID;fDetectorID;Counts", 1000, min_value, max_value),f"{hitType}.fDetectorID")

    # Draw the histogram
    canvas = ROOT.TCanvas("c1", f"{hitType}.fDetectorID Distribution", 800, 600)
    histogram.Draw()
    canvas.SaveAs(f"plots/{hitType}_fDetectorID_Distribution.png")

def draw_hit(df):
    df = df.Define("n_mu_filter_hits", "Digi_MuFilterHits.GetEntries()")
    df = df.Define("n_scifi_hits", "Digi_ScifiHits.GetEntries()")

    # Create histograms for the distribution of MuFilterHits and Digi_scifiHits
    hist_mu_filter = df.Histo1D(("hist_mu_filter_hits", "Number of Hits per Event;Number of Hits;Counts", 30, 0, 30), "n_mu_filter_hits")
    hist_scifi = df.Histo1D(("hist_scifi_hits", "Number of Hits per Event;Number of Hits;Counts", 30, 0, 30), "n_scifi_hits")

    # Draw both histograms on the same canvas
    canvas = ROOT.TCanvas("c1", "MuFilterHits and SciFiHits Distribution", 800, 600)
    hist_mu_filter.SetLineColor(ROOT.kRed)  # Set color for MuFilterHits histogram
    hist_scifi.SetLineColor(ROOT.kBlue)     # Set color for SciFiHits histogram

    # Draw histograms
    hist_mu_filter.Draw("HIST")
    hist_scifi.Draw("HIST SAME")

    # Add a legend
    legend = ROOT.TLegend(0.6, 0.7, 0.8, 0.9)
    legend.AddEntry(hist_mu_filter.GetPtr(), "MuFilterHits", "l")
    legend.AddEntry(hist_scifi.GetPtr(), "SciFiHits", "l")
    legend.Draw()

    # Save the canvas as an image
    canvas.SaveAs("plots/MuFilterHits_and_SciFiHits_Distribution.png")

def no_veto_eff(df):
    total_events = df.Count().GetValue()
    print(f'total: {total_events}')
    

    no_veto1 = df.Filter("All(Digi_MuFilterHits.fDetectorID > 10500)")
    no_veto1_events = no_veto1.Count().GetValue()
    print(f'No veto1: {no_veto1_events}, ratio: {no_veto1_events / total_events:.5f}')
    
    
    no_veto2 = df.Filter("All(Digi_MuFilterHits.fDetectorID > 11500 || Digi_MuFilterHits.fDetectorID <10500 )")
    no_veto2_events = no_veto2.Count().GetValue()
    print(f'No veto2: {no_veto2_events}, ratio: {no_veto2_events / total_events:.5f}')

    no_veto = df.Filter("All(Digi_MuFilterHits.fDetectorID > 11500)")
    no_veto_events = no_veto.Count().GetValue()
    print(f'No veto: {no_veto_events}, ratio: {no_veto_events / total_events:.5f}')

def veto_eff(df):
    hitType = 'Digi_ScifiHits'
    min_value = df.Min(f"{hitType}.fDetectorID").GetValue()
    max_value = df.Max(f"{hitType}.fDetectorID").GetValue()
    print(min_value, max_value)
    total_events = df.Count().GetValue()
    print(f'total: {total_events}')
    df = df.Define("n_mu_filter_hits", "Digi_MuFilterHits.GetEntries()")
    df = df.Define("n_scifi_hits", "Digi_ScifiHits.GetEntries()")

    df = df.Filter("n_scifi_hits>0")
    scifiHit_1_events = df.Count().GetValue()
    print(f'more than 0 scifi hit: {scifiHit_1_events}, ratio: {scifiHit_1_events/total_events:.5f}')

    #total_events = scifiHit_1_events
    #print(f'total: {total_events}')


    has_veto1 = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 10500)")
    has_veto1_events = has_veto1.Count().GetValue()
    print(f'has veto1: {has_veto1_events}, ratio: {has_veto1_events / total_events:.5f}')
    
    
    has_veto2 = df.Filter("Any(Digi_MuFilterHits.fDetectorID > 10500 && Digi_MuFilterHits.fDetectorID < 11500 )")
    has_veto2_events = has_veto2.Count().GetValue()
    print(f'has veto2: {has_veto2_events}, ratio: {has_veto2_events / total_events:.5f}')

    has_veto = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 11500)")
    has_veto_events = has_veto.Count().GetValue()
    print(f'has veto: {has_veto_events}, ratio: {has_veto_events / total_events:.5f}')

    no_scifi1 = df.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    no_scifi1_events = no_scifi1.Count().GetValue()
    print(f'No scifi1: {no_scifi1_events}, ratio: {no_scifi1_events / total_events:.5f}')

    no_scifi2 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000 )")
    no_scifi2_events = no_scifi2.Count().GetValue()
    print(f'No scifi2: {no_scifi2_events}, ratio: {no_scifi2_events / total_events:.5f}')

    no_scifi_12 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000)")
    no_scifi_12_events = no_scifi_12.Count().GetValue()
    print(f'No scifi 1 and 2: {no_scifi_12_events}, ratio: {no_scifi_12_events / total_events:.5f}')

    has_veto_no_scifi1 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    has_veto_no_scifi1_events = has_veto_no_scifi1.Count().GetValue()
    print(f'Has veto No scifi 1: {has_veto_no_scifi1_events}, ratio: {has_veto_no_scifi1_events / total_events:.5f}')

    has_veto_no_scifi2 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000 )")
    has_veto_no_scifi2_events = has_veto_no_scifi2.Count().GetValue()
    print(f'Has veto No scifi 2: {has_veto_no_scifi2_events}, ratio: {has_veto_no_scifi2_events / total_events:.5f}')

def scifi_eff(df):
    total_events = df.Count().GetValue()
    print(f'total: {total_events}')

    df = df.Define("n_mu_filter_hits", "Digi_MuFilterHits.GetEntries()")
    df = df.Define("n_scifi_hits", "Digi_ScifiHits.GetEntries()")

    n_scifi = 1
    n_mu = 1
    df = df.Filter(f"n_scifi_hits>{n_scifi} && n_mu_filter_hits > {n_mu}")
    scifiHit_1_events = df.Count().GetValue()
    print(f'scifi hit > {n_scifi} and muFilter hit > {n_mu}: {scifiHit_1_events}, ratio: {scifiHit_1_events/total_events:.5f}')

    total_events = df.Count().GetValue()
    print(f'total: {total_events}')

    has_veto1 = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 10500)")
    has_veto1_events = has_veto1.Count().GetValue()
    print(f'has veto1: {has_veto1_events}, ratio: {has_veto1_events / total_events:.5f}')
    
    
    has_veto2 = df.Filter("Any(Digi_MuFilterHits.fDetectorID > 10500 && Digi_MuFilterHits.fDetectorID < 11500 )")
    has_veto2_events = has_veto2.Count().GetValue()
    print(f'has veto2: {has_veto2_events}, ratio: {has_veto2_events / total_events:.5f}')

    has_veto = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 11500)")
    has_veto_events = has_veto.Count().GetValue()
    print(f'has veto: {has_veto_events}, ratio: {has_veto_events / total_events:.5f}')

    no_scifi1 = df.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    no_scifi1_events = no_scifi1.Count().GetValue()
    print(f'No scifi1: {no_scifi1_events}, ratio: {no_scifi1_events / total_events:.5f}')

    no_scifi2 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000 )")
    no_scifi2_events = no_scifi2.Count().GetValue()
    print(f'No scifi2: {no_scifi2_events}, ratio: {no_scifi2_events / total_events:.5f}')

    no_scifi_12 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000)")
    no_scifi_12_events = no_scifi_12.Count().GetValue()
    print(f'No scifi 1 and 2: {no_scifi_12_events}, ratio: {no_scifi_12_events / total_events:.5f}')

    has_veto_no_scifi1 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    has_veto_no_scifi1_events = has_veto_no_scifi1.Count().GetValue()
    print(f'Has veto No scifi 1: {has_veto_no_scifi1_events}, ratio: {has_veto_no_scifi1_events / total_events:.5f}')

    has_veto_no_scifi2 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000 )")
    has_veto_no_scifi2_events = has_veto_no_scifi2.Count().GetValue()
    print(f'Has veto No scifi 2: {has_veto_no_scifi2_events}, ratio: {has_veto_no_scifi2_events / total_events:.5f}')

def analyze_events(df, n_scifi=1, n_mu=1):
    results = {}

    original_total_events = df.Count().GetValue()
    results['original_total_events'] = original_total_events
    print(f'original total events: {original_total_events:3e}')

    # Define new columns for hit counts
    df = df.Define("n_mu_filter_hits", "Digi_MuFilterHits.GetEntries()")
    df = df.Define("n_scifi_hits", "Digi_ScifiHits.GetEntries()")

    # Apply filter for scifi hits and muFilter hits
    df = df.Filter(f"n_scifi_hits > {n_scifi} && n_mu_filter_hits > {n_mu}")
    scifi_mu_filter_events = df.Count().GetValue()
    results['scifi_mu_filter_events'] = scifi_mu_filter_events
    results['scifi_mu_filter_events'] = scifi_mu_filter_events / original_total_events
    print(f'Scifi hit > {n_scifi} and muFilter hit > {n_mu}: {scifi_mu_filter_events}, ratio: {scifi_mu_filter_events / original_total_events:.5f}')
    total_events = df.Count().GetValue()
    results['total_events'] = total_events
    print(f'total events: {total_events}')

    # Filters related to veto conditions
    has_veto1 = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 10500)")
    has_veto1_events = has_veto1.Count().GetValue()
    results['has_veto1_events'] = has_veto1_events
    results['has_veto1_ratio'] = has_veto1_events / total_events
    print(f'Has veto1: {has_veto1_events}, ratio: {has_veto1_events / total_events:.5f}')

    has_veto2 = df.Filter("Any(Digi_MuFilterHits.fDetectorID > 10500 && Digi_MuFilterHits.fDetectorID < 11500)")
    has_veto2_events = has_veto2.Count().GetValue()
    results['has_veto2_events'] = has_veto2_events
    results['has_veto2_ratio'] = has_veto2_events / total_events
    print(f'Has veto2: {has_veto2_events}, ratio: {has_veto2_events / total_events:.5f}')

    has_veto = df.Filter("Any(Digi_MuFilterHits.fDetectorID < 11500)")
    has_veto_events = has_veto.Count().GetValue()
    results['has_veto_events'] = has_veto_events
    results['has_veto_ratio'] = has_veto_events / total_events
    print(f'Has veto: {has_veto_events}, ratio: {has_veto_events / total_events:.5f}')

    # Filters for scifi hits
    no_scifi1 = df.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    no_scifi1_events = no_scifi1.Count().GetValue()
    results['no_scifi1_events'] = no_scifi1_events
    results['no_scifi1_ratio'] = no_scifi1_events / total_events
    print(f'No scifi1: {no_scifi1_events}, ratio: {no_scifi1_events / total_events:.5f}')

    no_scifi2 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000)")
    no_scifi2_events = no_scifi2.Count().GetValue()
    results['no_scifi2_events'] = no_scifi2_events
    results['no_scifi2_ratio'] = no_scifi2_events / total_events
    print(f'No scifi2: {no_scifi2_events}, ratio: {no_scifi2_events / total_events:.5f}')

    no_scifi_12 = df.Filter("All(Digi_ScifiHits.fDetectorID > 2500000)")
    no_scifi_12_events = no_scifi_12.Count().GetValue()
    results['no_scifi_12_events'] = no_scifi_12_events
    results['no_scifi_12_ratio'] = no_scifi_12_events / total_events
    print(f'No scifi 1 and 2: {no_scifi_12_events}, ratio: {no_scifi_12_events / total_events:.5f}')

    # Filters combining veto and scifi conditions
    has_veto_no_scifi1 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 1500000)")
    has_veto_no_scifi1_events = has_veto_no_scifi1.Count().GetValue()
    results['has_veto_no_scifi1_events'] = has_veto_no_scifi1_events
    results['has_veto_no_scifi1_ratio'] = has_veto_no_scifi1_events / total_events
    print(f'Has veto No scifi 1: {has_veto_no_scifi1_events}, ratio: {has_veto_no_scifi1_events / total_events:.5f}')

    has_veto_no_scifi2 = has_veto.Filter("All(Digi_ScifiHits.fDetectorID > 2500000 || Digi_ScifiHits.fDetectorID < 1500000)")
    has_veto_no_scifi2_events = has_veto_no_scifi2.Count().GetValue()
    results['has_veto_no_scifi2_events'] = has_veto_no_scifi2_events
    results['has_veto_no_scifi2_ratio'] = has_veto_no_scifi2_events / total_events
    print(f'Has veto No scifi 2: {has_veto_no_scifi2_events}, ratio: {has_veto_no_scifi2_events / total_events:.5f}')

    return results

def loop_analyze_mu(df, n_scifi=10, n_mu_range=range(1, 40, 2)):
    results_list = []

    for n_mu in n_mu_range:
        results = analyze_events(df, n_scifi=n_scifi, n_mu=n_mu)
        results['n_mu'] = n_mu
        results_list.append(results)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('csv/scifi_ineff_loop_n_mu.csv', index=False)
    return results_df

def loop_analyze_scifi(df, n_mu=1, n_scifi_range=range(1, 20, 2)):
    results_list = []

    for n_scifi in n_scifi_range:
        results = analyze_events(df, n_scifi=n_scifi, n_mu=n_mu)
        results['n_scifi'] = n_scifi
        results_list.append(results)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('csv/scifi_ineff_loop_n_scifi.csv', index=False)
    return results_df

def main():

    # read data from folder
    dir_path = '/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/'

    file_list = []
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
    
        # Check if the current path is a directory
        if os.path.isdir(folder_path):
            # Loop over all files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".root"):
                    file_path = os.path.join(folder_path, file_name)
                    print(file_path)
                    file_list.append(file_path)

        if(len(file_list)>100):
            break



    df = ROOT.RDataFrame("cbmsim", file_list)
    loop_analyze_mu(df)
    


    #veto_eff(df)
    #scifi_eff(df)
    
    # file = ROOT.TFile(file_list[0])
    # tree = file.Get("cbmsim")

    # # Access the first entry in the tree to examine Digi_scifiHits
    # tree.GetEntry(0)

    # # Assuming Digi_scifiHits is a branch in the tree
    # digi_hits = getattr(tree, "Digi_ScifiHits")
    # print(type(digi_hits))
    #scifi_eff(df)
    #veto_eff(df)
    #draw_detId(df, hitType='Digi_ScifiHits')
    #draw_hit(df)
    
    
    #select veto and muon system hits
    

if __name__ == "__main__":
    main()