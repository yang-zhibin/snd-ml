
import pandas as pd
import argparse
import os
import ROOT
import SndlhcGeo
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import numpy as np
from scipy.stats import binned_statistic
from array import array

pdg_db = ROOT.TDatabasePDG.Instance()


def drop_missing_files(df: pd.DataFrame, column_name: str, metadata_name: str = "") -> pd.DataFrame:
    """Drop rows where the file in column_name does not exist. Print summary per metadata."""
    exists_mask = df[column_name].apply(lambda path: os.path.exists(path))
    missing_count = (~exists_mask).sum()

    if metadata_name:
        print(f"{metadata_name}: {missing_count} missing files in '{column_name}'")
    else:
        print(f"{missing_count} missing files in '{column_name}'")

    return df[exists_mask].reset_index(drop=True)

def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data


def read_chain(metadata_df, metadata_name, min_file = 1e10):
    
    digi_chain = ROOT.TChain("cbmsim")
    count=0
    for index, row in metadata_df.iterrows():
        digi_path = row['digi_path']
        digi_chain.Add(digi_path)
        
        # f1 = ROOT.TFile.Open(digi_path)
        # if not f1 or f1.IsZombie() or not f1.Get("cbmsim"):
        #     print(f"Warning: {feature_path} is not valid.")
        #     continue

        count+=1
        if count> min_file:
           break

    print(f"{count} files read from {metadata_name}")
    return digi_chain

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo




def process_pass_muon_hits_data():
    mc_files = [
        # "MC_kaon_FTFP_BERT_metadata.csv",
        # "MC_neutron_FTFP_BERT_metadata.csv",
         "MC_muon_down_metadata.csv",
         "MC_muon_horizontal_metadata.csv",
         "MC_muon_up_metadata.csv",
        #"MC_neutrino_volTarget_100fb-1_metadata.csv",
        # "real_data_2024_metadata.csv",
    ]

    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'

    metadata_vars = load_metadata_files(mc_files, root_path)
    MC_muon_horizontal = metadata_vars["MC_muon_horizontal"]
    MC_muon_down = metadata_vars["MC_muon_down"]
    MC_muon_up = metadata_vars["MC_muon_up"]
    MC_muon = pd.concat([MC_muon_horizontal, MC_muon_down, MC_muon_up], ignore_index=True)

    
    MC_muon = drop_missing_files(MC_muon, "digi_path", "MC_muon")
    digi_chain = read_chain(MC_muon, "MC neutrino")
    
    
    # === Prepare output ===
    output_dir = "check_veto_hits_plots"
    os.makedirs(output_dir, exist_ok=True)
    out_file_name =  "pass_muon_veto_hits.root"
    out_file = ROOT.TFile(os.path.join(output_dir,out_file_name), "RECREATE")
    tree = ROOT.TTree("veto_hits", "Veto hit info")

    # === Define branches ===
    hit_time = array('f', [0])
    pdg = array('i', [0])
    energy_loss = array('f', [0])
    start_z = array('f', [0])
    time_category = array('i', [0])  # 0: early (≤25ns), 1: late (>25ns)
    veto1 = array('i', [0])
    veto2 = array('i', [0])

    tree.Branch("hit_time", hit_time, "hit_time/F")
    tree.Branch("pdg", pdg, "pdg/I")
    tree.Branch("energy_loss", energy_loss, "energy_loss/F")
    tree.Branch("start_z", start_z, "start_z/F")
    tree.Branch("time_category", time_category, "time_category/I")
    tree.Branch("veto1", veto1, "veto1/I")
    tree.Branch("veto2", veto2, "veto2/I")


    # create a hist of number veto hits
    
    # === Process events ===
    count= 0
    n_entries = digi_chain.GetEntries()
    for i in tqdm(range(n_entries), desc="Processing events"):
        digi_chain.GetEntry(i)
        hit2MC = digi_chain.Digi_MuFilterHits2MCPoints[0]

        n_veto_hit = 0
        for aHit in digi_chain.Digi_MuFilterHits:
            if not aHit.isValid() or aHit.GetSystem() != 1:  # Only Veto
                continue
            detID = aHit.GetDetectorID()
            station = (detID // 1000) % 10
            n_veto_hit += 1
            hit_time[0] = aHit.GetTime()
            time_category[0] = 0 if hit_time[0] <= 25 else 1
            detID = aHit.GetDetectorID()
            linksToMCPoints = hit2MC.wList(detID)

            for mc_point_i, _ in linksToMCPoints:
                if mc_point_i >= digi_chain.ScifiPoint.GetEntries():
                    continue
                scifi_point = digi_chain.ScifiPoint[mc_point_i]

                pdg[0] = scifi_point.PdgCode()
                energy_loss[0] = scifi_point.GetEnergyLoss()
                start_z[0] = (
                    digi_chain.MCTrack[1].GetStartZ()
                    if digi_chain.MCTrack.GetEntries() > 1
                    else -9999
                )
                count+=1
                tree.Fill()

        if count > (2000*1000):
            break

    tree.Write()
    out_file.Close()
    print(f"Saved output to {os.path.join(output_dir, out_file_name)}")
    


def process_veto_hits_data():
    mc_files = [
        # "MC_kaon_FTFP_BERT_metadata.csv",
        # "MC_neutron_FTFP_BERT_metadata.csv",
        # "MC_muon_down_metadata.csv",
        # "MC_muon_horizontal_metadata.csv",
        # "MC_muon_up_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
        # "real_data_2024_metadata.csv",
    ]

    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'

    metadata_vars = load_metadata_files(mc_files, root_path)
    MC_neutrino_metadata = metadata_vars["MC_neutrino_volTarget_100fb_1"]
    
    MC_neutrino_metadata = drop_missing_files(MC_neutrino_metadata, "digi_path", "MC_neutrino_volTarget_100fb_1")
    digi_chain = read_chain(MC_neutrino_metadata, "MC neutrino")
    
    
    # === Prepare output ===
    output_dir = "check_veto_hits_plots"
    os.makedirs(output_dir, exist_ok=True)
    out_file_name =  "veto_hits_test.root"
    out_file = ROOT.TFile(os.path.join(output_dir,out_file_name), "RECREATE")
    tree = ROOT.TTree("veto_hits", "Veto hit info")

    # === Define branches ===
    index = array('i', [0])
    hit_time = array('f', [0])
    neutrino_type = array('i', [0])
    pdg = array('i', [0])
    energy_loss = array('f', [0])
    start_z = array('f', [0])
    time_category = array('i', [0])  # 0: early (≤25ns), 1: late (>25ns)
    veto_plane = array('i', [0]) #veto1, veto2
    count_veto1 = array('i', [0])
    count_veto2 = array('i', [0])
    
    
    tree.Branch("index", index, "index/I")
    tree.Branch("hit_time", hit_time, "hit_time/F")
    tree.Branch("neutrino_type", neutrino_type, "neutrino_type/I")
    tree.Branch("pdg", pdg, "pdg/I")
    tree.Branch("energy_loss", energy_loss, "energy_loss/F")
    tree.Branch("start_z", start_z, "start_z/F")
    tree.Branch("time_category", time_category, "time_category/I")
    tree.Branch("veto_plane", veto_plane, "veto_plane/I")
    tree.Branch("count_veto1", count_veto1, "count_veto1/I")
    tree.Branch("count_veto2", count_veto2, "count_veto2/I")


    # create a hist of number veto hits
    h_n_veto_hits = ROOT.TH1I("h_n_veto_hits", "Number of Veto Hits per Event;N Veto Hits;Entries", 20, 0, 20)
    
    # === Process events ===
    n_entries = digi_chain.GetEntries()
    for i in tqdm(range(n_entries), desc="Processing events"):
        digi_chain.GetEntry(i)
        hit2MC = digi_chain.Digi_MuFilterHits2MCPoints[0]

        n_veto_hit = 0
        for aHit in digi_chain.Digi_MuFilterHits:
            if not aHit.isValid() or aHit.GetSystem() != 1:  # Only Veto
                continue
            n_veto_hit += 1
            hit_time[0] = aHit.GetTime()
            time_category[0] = 0 if hit_time[0] <= 25 else 1
            detID = aHit.GetDetectorID()
            linksToMCPoints = hit2MC.wList(detID)

            for mc_point_i, _ in linksToMCPoints:
                if mc_point_i >= digi_chain.ScifiPoint.GetEntries():
                    continue
                scifi_point = digi_chain.ScifiPoint[mc_point_i]

                pdg[0] = scifi_point.PdgCode()
                energy_loss[0] = scifi_point.GetEnergyLoss()
                start_z[0] = (
                    digi_chain.MCTrack[1].GetStartZ()
                    if digi_chain.MCTrack.GetEntries() > 1
                    else -9999
                )

                tree.Fill()

        h_n_veto_hits.Fill(n_veto_hit)
        if i > 100000:
            break
        
    total_events = h_n_veto_hits.GetEntries()
    nonzero_events = sum(h_n_veto_hits.GetBinContent(i) for i in range(2, h_n_veto_hits.GetNbinsX() + 1))  # bin 2+ → n > 0
    percent_nonzero = 100.0 * nonzero_events / total_events if total_events > 0 else 0.0

    
    print(f"Total events: {int(total_events)}")
    print(f"Events with n_veto_hit > 0: {int(nonzero_events)}")
    print(f"Percentage of events with n_veto_hit > 0: {percent_nonzero:.2f}%")
    
    # integral = h_n_veto_hits.Integral("width")
    # if integral > 0:
    #     h_n_veto_hits.Scale(1.0 / integral)
    # else:
    #     print("[warning] Histogram integral is zero; skipping normalization")
        
    c1 = ROOT.TCanvas("c1", "Veto Hit Distribution", 800, 600)
    ROOT.gStyle.SetOptStat(1110)
    h_n_veto_hits.Draw("hist")
    
    label = ROOT.TLatex()
    label.SetNDC(True)  # Normalized device coordinates
    label.SetTextSize(0.04)
    label.DrawLatex(0.4, 0.65, f"Events with (Veto Hits > 0): {percent_nonzero:.2f}%")
    
    c1.SaveAs(os.path.join(output_dir, "veto_hit_distribution.pdf"))
    
    tree.Write()
    out_file.Close()
    print(f"Saved output to {os.path.join(output_dir, out_file_name)}")
    
    
ROOT.ROOT.EnableImplicitMT()
ROOT.gInterpreter.Declare("""
    std::string get_particle_name(int pdg) {
        auto* pdg_db = TDatabasePDG::Instance();
        auto* particle = pdg_db->GetParticle(pdg);
        if (!particle) return "unknown";

        std::string name = particle->GetName();
        // Remove trailing + or - if present
        if (!name.empty() && (name.back() == '+' || name.back() == '-')) {
            name.pop_back();
        }

        return name;
        }
    """)

ROOT.gInterpreter.Declare("""
    std::string get_particle_name_with_sign(int pdg) {
        auto* pdg_db = TDatabasePDG::Instance();
        auto* particle = pdg_db->GetParticle(pdg);
        if (!particle) return "unknown";

        std::string name = particle->GetName();
        return name;
        }
    """)

def plot_pdg(df, output_dir = "check_veto_hits_plots"):
    # Declare C++ function for mapping PDG to particle name
    

    # Define particle name column
    #df = df.Define("particle", "get_particle_name(pdg)")
    
    # Filter by hit_time
    df_early = df.Filter("hit_time < 25")
    df_late  = df.Filter("hit_time >= 25")
    
    # plot stack bar plot of each particle

    # Get list of all unique particles
    particle_array = df.AsNumpy(["particle"])["particle"]
    particle_counts = Counter(particle_array)

    # Create stacked histograms
    stack = ROOT.THStack("stack", "Particle Type Distribution;Particle;Entries")

    # Sort particles by total count (descending)
    particle_list = [p for p, _ in particle_counts.most_common()]
    
    # Create a map from particle name to bin number
    particle_to_bin = {name: i+1 for i, name in enumerate(particle_list)}  # bins start at 1

    # Helper function to make a histogram
    def make_hist(df_filtered, hist_name):
        hist = ROOT.TH1F(hist_name, "", len(particle_list), 0.5, len(particle_list) + 0.5)
        hist.GetXaxis().SetTitle("Particle")
        hist.GetYaxis().SetTitle("Entries")
        # Fill histogram manually
        arr = df_filtered.AsNumpy(["particle"])["particle"]
        for p in arr:
            hist.Fill(particle_to_bin.get(p, 0))  # 0 will be underflow if unknown
        # Label x-axis
        for i, name in enumerate(particle_list):
            hist.GetXaxis().SetBinLabel(i + 1, str(name))
        return hist

    # Make histograms
    hist_early = make_hist(df_early, "hist_early")
    hist_early.SetFillColorAlpha(ROOT.kBlue, 0.8)
    hist_early.SetTitle("")

    hist_late = make_hist(df_late, "hist_late")
    hist_late.SetFillColorAlpha(ROOT.kRed, 0.8)
    hist_late.SetTitle("")

    # Add to stack
    stack.Add(hist_late)
    stack.Add(hist_early)
    

    # Draw using TCanvas
    ROOT.TGaxis.SetMaxDigits(3)
    c = ROOT.TCanvas("c", "Particle Type Stacked Histogram", 1000, 600)
    stack.Draw("hist")
    stack.GetXaxis().LabelsOption("d")
    stack.SetTitle("Stacked Particle Distribution by Hit Time;Particle;Entries")
    c.SetBottomMargin(0.3)
    
    # Legend
    legend = ROOT.TLegend(0.7, 0.75, 0.88, 0.88)
    legend.AddEntry(hist_early, "hit_time < 25 ns", "f")
    legend.AddEntry(hist_late, "hit_time #geq 25 ns", "f")
    legend.Draw()
    
    output_path = os.path.join(output_dir, "stacked_particles_by_hit_time.pdf")
    c.SaveAs(output_path)
    
    
def plot_hit_time(df, output_dir = "check_veto_hits_plots"):
    df = df.Define("particle", "get_particle_name(pdg)")
    #pass_muon_df = pass_muon_df.Define("particle", "get_particle_name(pdg)")
    
    
    
    expected_particles = ["e", "mu", "proton"] # "neutron", "K", "unknown", "gamma",  "pi",
    
    # Create canvas and legend
    ROOT.TGaxis.SetMaxDigits(3)
    c = ROOT.TCanvas("c", "Hit Time Hist", 1000, 600)
    leg = ROOT.TLegend(0.65, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kOrange + 7, ROOT.kMagenta,
              ROOT.kCyan + 1, ROOT.kViolet, ROOT.kTeal + 2, ROOT.kPink + 4,
              ROOT.kAzure - 3, ROOT.kGray + 2, ROOT.kSpring - 2, ROOT.kYellow + 2,
              ROOT.kOrange + 1, ROOT.kRed + 3, ROOT.kBlue + 3]

    first_drawn = False
    hist_proxies = []
    for i, particle in enumerate(expected_particles):
        df_particle = df.Filter(f'particle == "{particle}"')
        if df_particle.Count().GetValue() == 0:
            continue

        h_proxy = df_particle.Histo1D(
            (f"h_{particle}", f"Hit Time Distribution ;Hit time [ns];Entries", 200, 0, 20),
            "hit_time"
        )
        h_proxy.SetDirectory(0)
        hist = h_proxy.GetValue()
        hist_proxies.append(h_proxy)
        # Normalize to unit area
        # integral = hist.Integral()
        # if integral > 0:
        #     hist.Scale(1.0 / integral)
        # else:
        #     continue  # skip empty histograms

        # Style
        hist.SetLineColorAlpha(colors[i % len(colors)], 0.6)
        hist.SetLineWidth(2)
        hist.SetStats(False)

        # Draw
        draw_option = "hist" if not first_drawn else "hist same"
        hist.Draw(draw_option)
        first_drawn = True

        leg.AddEntry(hist, particle, "lep")

    
    leg.Draw()
    c.SetGrid()
    c.SetLogy()
    outpath = os.path.join(output_dir, "pass_muon_hit_time_0-20ns.pdf")
    c.SaveAs(outpath)
    c.Close()
    
    


def plot_energy_loss(df, pass_muon_df, output_dir = "check_veto_hits_plots"):
    #df = df.Define("particle", "get_particle_name(pdg)")
    pass_muon_df = pass_muon_df.Define("particle", "get_particle_name_with_sign(pdg)")
    
    df = df.Define("particle", "get_particle_name_with_sign(pdg)")
    
    bin_min = 0
    bin_max = 0.0015  # in GeV
    n_bins =150
    bin_width_GeV = (bin_max - bin_min) / n_bins
    bin_width_keV = bin_width_GeV * 1e6  # convert to keV
    
    draw_pass_muon =True
    

    
    
    # ["mu+","mu-"]
    draw_particle = "Proton"
    expected_particles = ["proton","antiproton"]
    
    if draw_pass_muon:
        plot_name = f"{draw_particle}_energy_loss_hist_with_passMuon_0-{int(bin_max*1e6)}kev.pdf"
    else:
        plot_name = f"{draw_particle}_energy_loss_hist_0-{int(bin_max*1e6)}kev.pdf"
    
    # Create canvas and legend
    ROOT.TGaxis.SetMaxDigits(3)
    c = ROOT.TCanvas("c", "Energy Loss Hist", 1000, 600)
    leg = ROOT.TLegend(0.65, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kOrange + 7, ROOT.kMagenta,
              ROOT.kCyan + 1, ROOT.kViolet, ROOT.kTeal + 2, ROOT.kPink + 4,
              ROOT.kAzure - 3, ROOT.kGray + 2, ROOT.kSpring - 2, ROOT.kYellow + 2,
              ROOT.kOrange + 1, ROOT.kRed + 3, ROOT.kBlue + 3]


    
    first_drawn = False
    hist_proxies = []
    for i, particle in enumerate(expected_particles):
        df_particle = df.Filter(f'particle == "{particle}"')
        if df_particle.Count().GetValue() == 0:
            continue

        h_proxy = df_particle.Histo1D(
            (f"h_{particle}",
            f"Energy Loss Distribution ;Energy Loss [GeV];Probability Density / {bin_width_keV:.1f} keV",
            n_bins, bin_min, bin_max),
            "energy_loss"
        )
        h_proxy.SetDirectory(0)
        hist = h_proxy.GetValue()
        hist_proxies.append(h_proxy)
        #Normalize to unit area
        integral = hist.Integral()
        if integral > 0:
            hist.Scale(1.0 / integral)
        else:
            continue  # skip empty histograms

        # Style
        hist.SetLineColorAlpha(colors[i % len(colors)], 0.5)
        hist.SetLineWidth(2)
        
        hist.SetStats(False)

        # Draw
        draw_option = "hist" if not first_drawn else "hist same"
        hist.Draw(draw_option)
        #hist.SetMinimum(0.1)
        first_drawn = True

        leg.AddEntry(hist, particle, "lep")
    
    if draw_pass_muon:
        for i, particle in enumerate(expected_particles):
            df_particle = pass_muon_df.Filter(f'particle == "{particle}"')
            if df_particle.Count().GetValue() == 0:
                continue


            h_proxy = df_particle.Histo1D(
                (f"h_{particle}",
                f"Energy Loss Distribution ;Energy Loss [GeV];Probability Density / {bin_width_keV:.1f} keV",
                n_bins, bin_min, bin_max),
                "energy_loss"
            )
            h_proxy.SetDirectory(0)
            hist = h_proxy.GetValue()
            hist_proxies.append(h_proxy)
            # Normalize to unit area
            integral = hist.Integral()
            if integral > 0:
                hist.Scale(1.0 / integral)
            else:
                continue  # skip empty histograms

            # Style
            hist.SetLineColorAlpha(colors[(i + len(expected_particles)) % len(colors)], 0.5)
            hist.SetLineWidth(2)
            
            hist.SetStats(False)

            # Draw
            draw_option = "hist" if not first_drawn else "hist same"
            hist.Draw(draw_option)
            #hist.SetMinimum(0.1)
            first_drawn = True

            leg.AddEntry(hist, f"{particle} (from pass muon)", "lep")


    leg.Draw()
    c.SetGrid()
    c.SetLogy()
    outpath = os.path.join(output_dir, plot_name)
    c.SaveAs(outpath)
    c.Close()
    
    
    


def plot_start_z(df, output_dir = "check_veto_hits_plots"):
    df = df.Define("particle", "get_particle_name(pdg)")
    
    expected_particles = ["mu", "neutron", "proton", "pi", "e"]
    colors = [ROOT.kMagenta, ROOT.kOrange + 7, ROOT.kGreen + 2, ROOT.kBlue, ROOT.kRed]
    
    # Create canvas and legend
    ROOT.TGaxis.SetMaxDigits(3)
    c = ROOT.TCanvas("c", "Energy Loss Hist", 1000, 600)
    leg = ROOT.TLegend(0.65, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    

    first_drawn = False
    hist_proxies = []
    stack = ROOT.THStack("stack", "Vertex Z Position;Z [cm];Entries")
    for i, particle in enumerate(expected_particles):
        df_particle = df.Filter(f'particle == "{particle}"')
        if df_particle.Count().GetValue() == 0:
            continue

        h_proxy = df_particle.Histo1D(
            (f"h_{particle}", f"Vertex Z Position Distribution ;Z [cm];Entries", 80, 280, 360),
            "start_z"
        )
        h_proxy.SetDirectory(0)
        hist = h_proxy.GetValue()
        hist_proxies.append(h_proxy)
        # Normalize to unit area
        # integral = hist.Integral()
        # if integral > 0:
        #     hist.Scale(1.0 / integral)
        # else:
        #     continue  # skip empty histograms

        # Style
        # Set fill color and style
        hist.SetFillColorAlpha(colors[i % len(colors)], 0.6)
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        
        hist.SetStats(False)

        # Draw
        stack.Add(hist)

        leg.AddEntry(hist, particle, "f")
        
    stack.Draw("hist")
    # stack hist
    
    leg.Draw()
    c.SetGrid()
    c.SetLogy()
    outpath = os.path.join(output_dir, "vertex_hist_stack.pdf")
    c.SaveAs(outpath)
    c.Close()
    

def plot_time_vs_energy_loss(df, output_dir="check_veto_hits_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Create canvas
    c = ROOT.TCanvas("c", "Hit Time vs Energy Loss", 800, 600)
    c.SetLogz()  # Log color scale if values vary a lot

    # Create 2D histogram
    h2d_proxy = df.Histo2D(
        ("h2d", "Hit Time vs Energy Loss;Energy Loss [GeV];Hit Time [ns]", 
         1000, 0, 0.01,
         1000, 0, 200),      # Y-axis 
        "energy_loss", "hit_time"
    )
    h2d_proxy.SetDirectory(0)
    h2d = h2d_proxy.GetValue()

    # Style
    h2d.SetStats(False)
    h2d.Draw("COLZ")
    
    c.SetLeftMargin(0.15) 
    c.SetBottomMargin(0.1)

    # Save
    outpath = os.path.join(output_dir, "hit_time_vs_energy_loss_2d_logScale.pdf")
    c.SetLogx()  # Log scale on X-axis
    c.SetLogy()  # Log scale on Y-axis
    #c.SetLogz()  # Log scale on color (Z) axis
    c.SaveAs(outpath)
    c.Close()
    
def plot_time_vs_start_z(df, output_dir="check_veto_hits_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Create canvas
    c = ROOT.TCanvas("c", "Hit Time vs Vertex Z Position", 800, 600)
    c.SetLogz()  # Log color scale if values vary a lot

    # Create 2D histogram
    h2d_proxy = df.Histo2D(
        ("h2d", "Hit Time vs Vertex Z Position;Vertex Z Position [cm]; Hit Time [ns];", 
         80, 280, 360,
         1000, 0, 200),      # Y-axis: energy_loss
        "start_z", "hit_time"
    )
    h2d_proxy.SetDirectory(0)
    h2d = h2d_proxy.GetValue()

    # Style
    h2d.SetStats(False)
    h2d.Draw("COLZ")
    
    c.SetLeftMargin(0.15) 
    c.SetBottomMargin(0.1)

    # Save
    outpath = os.path.join(output_dir, "hit_time_vs_vertex_z_position_2d_logScale.pdf")
    #c.SetLogx()  # Log scale on X-axis
    c.SetLogy()  # Log scale on Y-axis
    #c.SetLogz()  # Log scale on color (Z) axis
    c.SaveAs(outpath)
    c.Close()
    

def plot_e_loss_vs_start_z(df, output_dir="check_veto_hits_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Create canvas
    c = ROOT.TCanvas("c", "Energy Loss vs Vertex Z Position", 800, 600)
    c.SetLogz()  # Log color scale if values vary a lot

    # Create 2D histogram
    h2d_proxy = df.Histo2D(
        ("h2d", "Energy Loss vs Vertex Z Position;Vertex Z Position [cm];Energy Loss [GeV]", 
         80, 280, 360,
         1000, 0, 0.01),     
        "start_z" ,"energy_loss"
    )
    h2d_proxy.SetDirectory(0)
    h2d = h2d_proxy.GetValue()

    # Style
    h2d.SetStats(False)
    h2d.Draw("COLZ")
    
    c.SetLeftMargin(0.15) 
    c.SetBottomMargin(0.1)

    # Save
    outpath = os.path.join(output_dir, "energy_loss_vs_vertex_z_position_2d.pdf")
    #c.SetLogx()  # Log scale on X-axis
    #c.SetLogy()  # Log scale on Y-axis
    #c.SetLogz()  # Log scale on color (Z) axis
    c.SaveAs(outpath)
    c.Close()
    
def plot_veto_hist():
    tree_name = "veto_hits"
    file_name = "check_veto_hits_plots/veto_hits.root"
    pass_muon_file_name = "check_veto_hits_plots/pass_muon_veto_hits.root"
    
    df = ROOT.RDataFrame(tree_name, file_name)
    pass_muon_df = ROOT.RDataFrame(tree_name, pass_muon_file_name)
    
    # particle distribution (pdg)
    #plot_pdg(df)
    

    # veto hit time for each particles 
    plot_hit_time(pass_muon_df)
    
    # energy loss for each particles 
    #plot_energy_loss(df, pass_muon_df)
    
    # stack hist of Start Z for each particles ((e, neutron, pi, proton, gamma))
    #plot_start_z(df)
    

    # veto hit time vs energy loss
    #plot_time_vs_energy_loss(df)
    
    # veto hit time vs stack hist of Start Z
    #plot_time_vs_start_z(df)
    
    # energy loss vs stack hist of Start Z
    #plot_e_loss_vs_start_z(df)
    
    # plot veto_hit_time distribution
    # plot veto_hit_pdg distribution
    # plot start_z distribution (with veto tagged, stack with veto_hit_time <25 and veto_hit_time>25), 

            
    
    

    


if __name__ == "__main__":
    
    #process_veto_hits_data()
    #process_pass_muon_hits_data()
    plot_veto_hist()
    
    
    