import ROOT
import pandas as pd
import os
ROOT.gROOT.SetBatch(True)


def plot_hist(rdf, output_file_path, hist_name = 'n_scifi_hits'):
    # 6. Plot histograms grouped by class_id
    class_labels = {
        0: "ve", 1: "vm", 2: "vt", 3: "NC"
    }
    colors = [2, 4, 8, 6, 46, 38, 28] 


    canvas = ROOT.TCanvas("c", f"{hist_name} by class", 800, 600)
    legend = ROOT.TLegend(0.7, 0.6, 0.9, 0.9)

    any_drawn = False
    hist_proxies = []
    for class_id, label in class_labels.items():
        print(class_id, label)
        n_event = rdf.Filter(f"class_id == {class_id}").Count().GetValue()
        print(f'    n events : {rdf.Filter(f"class_id == {class_id}").Count().GetValue()}')
        hist_proxy = rdf.Filter(f"class_id == {class_id}").Histo1D(
            (f"hist_{label}", f"{hist_name} for {label}", 60, 0, 60),
            f"{hist_name}"
        )
        hist = hist_proxy.GetValue()
        hist_proxies.append(hist_proxy)
        print(hist)
        if hist.Integral() == 0:
            continue

        hist.Scale(1.0 / hist.Integral())
        hist.SetLineColor(colors[class_id])
        hist.SetLineWidth(2)

        hist.GetXaxis().SetTitle(f"{hist_name}")
        hist.GetYaxis().SetTitle("Density")
        hist.GetYaxis().SetRangeUser(0, 0.1)
        hist.Draw("SAME" if any_drawn else "HIST")
        legend.AddEntry(hist, f'{label} ({n_event} events)', "l")
        any_drawn = True

        #break

    legend.Draw()
    if any_drawn:
        os.makedirs("plot", exist_ok=True)
        canvas.Update()
        canvas.SaveAs(output_file_path)
        print(f'File saved in: {output_file_path}')
    else:
        print("No valid histograms to draw. Skipping canvas save.")




def main():
    neutrino_csv_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    df = pd.read_csv(neutrino_csv_path)  

    # 2. Select feature paths
    paths = df["feature_path"].tolist()

    # 3. Create TChain
    chain = ROOT.TChain("snddata")
    for path in paths:
        chain.Add(path)
        print(path)
        #break

    # 4. Read into RDataFrame
    rdf = ROOT.RDataFrame(chain)
    #chain.GetListOfBranches().Print()

    # 5. Define n_scifi_hits (example: count nonzero energy or hits in scifi1..scifi5)
    # You can adapt this expression depending on data type (array, float, object)
    rdf = rdf.Define("n_scifi_hits", "scifi1 + scifi2 + scifi3 + scifi4 + scifi5")
    rdf = rdf.Define("n_us_hits", "us1 + us2 + us3 + us4 + us5")
    rdf = rdf.Define("n_ds_hits", "ds1 + ds2 + ds3 + ds4")

    print("Number of entries:", rdf.Count().GetValue())
    rdf = rdf.Define("class_id", """
            if (pdgCode == 12 || pdgCode == -12) return 0; // ve
            else if (pdgCode == 14 || pdgCode == -14) return 1; // vm
            else if (pdgCode == 16 || pdgCode == -16) return 2; // vt
            else if (pdgCode == 112 || pdgCode == -112 || pdgCode == 114 || pdgCode == -114 || pdgCode == 116 || pdgCode == -116) return 3; // NC
            else if (pdgCode == 130 || pdgCode == 310) return 4; // kaon
            else if (pdgCode == 2112) return 5; // neutron
            else if (pdgCode == 13 || pdgCode == -13) return 6; // muon
            else if (pdgCode == 0) return 6; // real_data (same class as muon)
            else return -1; // others

    """)
    

    filters = {
        "s1": "scifi1 != 0",
        "s2": "scifi1 == 0 && scifi2 != 0",
        "s3": "scifi1 == 0 && scifi2 == 0 && scifi3 != 0",
        "s4": "scifi1 == 0 && scifi2 == 0 && scifi3 == 0 && scifi4 != 0",
        "s5": "scifi1 == 0 && scifi2 == 0 && scifi3 == 0 && scifi4 == 0 && scifi5 != 0",
    }

    #for station, condition in filters.items():
    #    rdf_filtered = rdf.Filter(condition)
    #    plot_plot_hist(rdf_filtered, f"plot/n_scifi_hits_by_class_{station}.pdf", hist_name = 'n_scifi_hits')

    #plot_hist(rdf, f"plot/n_ds_hits_by_class.pdf", hist_name = 'n_ds_hits')
    plot_hist(rdf, f"plot/n_us_hits_by_class.pdf", hist_name = 'n_us_hits')



if __name__ == "__main__": 
    main()