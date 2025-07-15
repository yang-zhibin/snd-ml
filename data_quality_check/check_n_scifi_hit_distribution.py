import ROOT
import pandas as pd
import os
ROOT.gROOT.SetBatch(True)


def plot_hist(rdf, output_file_path, hist_name="n_scifi_hits"):
    # ── 1. Labels and palette ────────────────────────────────────────────
    class_labels = {0: "ve", 1: "vm", 2: "vt", 3: "NC"}

    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen + 2,
        3: ROOT.kMagenta,
        4: ROOT.kOrange + 7,
        5: ROOT.kCyan,
        6: ROOT.kBlack,
    }

    # ── 2. Canvas & legend ───────────────────────────────────────────────
    canvas = ROOT.TCanvas("c", f"{hist_name} by class", 800, 600)
    legend = ROOT.TLegend(0.70, 0.60, 0.90, 0.90)

    any_drawn   = False
    hist_proxies = []                     # keep RDF proxies alive

    # ── 3. Loop over classes ─────────────────────────────────────────────
    for class_id, label in class_labels.items():
        rdf_f   = rdf.Filter(f"class_id == {class_id}")
        n_event = rdf_f.Count().GetValue()
        if n_event == 0:
            continue

        # build and fill histogram
        h_proxy = rdf_f.Histo1D(
            (f"h_{hist_name}_{label}",
             f"{hist_name} for {label};{hist_name};Probability density",
             120, 0, 1200),                    # adjust bins/range if needed
            hist_name
        )
        h = h_proxy.GetValue()
        hist_proxies.append(h_proxy)

        if h.Integral() == 0:
            continue

        # ── 3a. Styling identical to plot_hist_each_plane ───────────────
        color = colors.get(class_id, ROOT.kBlack)

        h.Scale(1.0 / h.Integral())
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.5)
        h.SetMarkerColor(color)
        h.SetLineColorAlpha(color, 0.9)
        h.SetLineWidth(2)
        h.SetStats(0)
        h.GetYaxis().SetRangeUser(0, 0.10)

        h.Draw("SAME" if any_drawn else "HIST")
        legend.AddEntry(h, f"{label} ({n_event})", "l")
        any_drawn = True

    # ── 4. Finish up ─────────────────────────────────────────────────────
    legend.Draw()
    if any_drawn:
        os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)
        canvas.SaveAs(output_file_path)
        print(f"Saved: {output_file_path}")
    else:
        print("[plot_hist] No valid histograms – skipped.")

def plot_hist_old(rdf, output_file_path, hist_name = 'n_scifi_hits'):
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

def plot_hist_each_plane(rdf):
    # ── 1. Per-plane binning ─────────────────────────────────────────────
    plane_binning = {
        'scifi1': (120, 0, 1200),
        'scifi2': (120, 0, 1200),
        'scifi3': (120, 0, 1200),
        'scifi4': (120, 0, 1200),
        'scifi5': (120, 0, 1200),
        'us1':    (13, 0, 13),
        'us2':    (13, 0, 13),
        'us3':    (13, 0, 13),
        'us4':    (13, 0, 13),
        'us5':    (13, 0, 13),
        'ds1':   (40, 0, 40),
        'ds2':   (40, 0, 40),
        'ds3':   (40, 0, 40),
        'ds4':   (40, 0, 40),
    }

    # ── 2. Class labels and colours ──────────────────────────────────────
    class_labels = {0: "ve", 1: "vm", 2: "vt", 3: "NC"}

    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen + 2,
        3: ROOT.kMagenta,
        4: ROOT.kOrange + 7,
        5: ROOT.kCyan,
        6: ROOT.kBlack,
    }

    # ── 3. Loop over each plane and generate plots ───────────────────────
    for plane, (n_bins, x_min, x_max) in plane_binning.items():
        canvas  = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        legend  = ROOT.TLegend(0.70, 0.60, 0.90, 0.90)
        out_pdf = f"plot/n_hits_{plane}.pdf"

        any_drawn = False
        hist_proxies = []

        for class_id, label in class_labels.items():
            rdf_f = rdf.Filter(f"class_id == {class_id}")
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                continue

            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f"N hits in {plane};N hits;Probability density",
                 n_bins, x_min, x_max),
                plane
            )
            h = h_proxy.GetValue()
            hist_proxies.append(h_proxy)

            if h.Integral() == 0:
                continue

            h.Scale(1.0 / h.Integral())
            color = colors.get(class_id, ROOT.kBlack)

            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.5)
            h.SetMarkerColor(color)
            h.SetLineColorAlpha(color, 0.9)
            h.SetLineWidth(2)
            h.SetStats(0)

            h.GetYaxis().SetRangeUser(0, 1)
            h.Draw("SAME" if any_drawn else "HIST")
            legend.AddEntry(h, f"{label} ({n_evt})", "l")
            any_drawn = True

        legend.Draw()
        if any_drawn:
            os.makedirs("plot", exist_ok=True)
            canvas.SaveAs(out_pdf)
            print(f"Saved: {out_pdf}")
        else:
            print(f"[{plane}] no valid histograms – skipped.")


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
    
    rdf = rdf.Define(
        "max_scifi_hits_in_one_plane",
        "ROOT::VecOps::Max(ROOT::VecOps::RVec<int>{scifi1, scifi2, scifi3, scifi4, scifi5})"
    )
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
    #plot_hist(rdf, f"plot/n_us_hits_by_class.pdf", hist_name = 'n_us_hits')
    
    #plot_hist_each_plane(rdf)

    plot_hist(rdf, f"plot/max_n_scifi_hits_in_one_plane.pdf", hist_name = 'max_scifi_hits_in_one_plane')


if __name__ == "__main__": 
    main()