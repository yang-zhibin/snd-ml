
import pandas as pd
import argparse
import os
import ROOT
# import SndlhcGeo
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import numpy as np
from scipy.stats import binned_statistic
from array import array
from collections import defaultdict

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
    out_file_name =  "veto_hits.root"
    out_file = ROOT.TFile(os.path.join(output_dir,out_file_name), "RECREATE")
    tree = ROOT.TTree("veto_hits", "Veto hit info")

    # === Define arrays ===
    event_index       = array('i', [0])
    hit_time          = array('f', [0])
    neutrino_type     = array('i', [0])
    pdg               = array('i', [0])
    energy_loss       = array('f', [0])  # <-- was missing in your snippet
    start_z           = array('f', [0])
    veto_plane        = array('i', [0])  # 1 = veto1, 2 = veto2
    only_proton_flag  = array('i', [0])

    # === Define branches ===
    tree.Branch("event_index", event_index, "event_index/I")
    tree.Branch("hit_time", hit_time, "hit_time/F")
    tree.Branch("neutrino_type", neutrino_type, "neutrino_type/I")
    tree.Branch("pdg", pdg, "pdg/I")
    tree.Branch("energy_loss", energy_loss, "energy_loss/F")
    tree.Branch("start_z", start_z, "start_z/F")
    tree.Branch("veto_plane", veto_plane, "veto_plane/I")
    tree.Branch("only_proton_flag", only_proton_flag, "only_proton_flag/I")

    
    particles = ["mu", "e", "neutron", "pi", "proton", "others"]
    mcpoint_count = {p: array('i', [0]) for p in particles}
    mcpoint_eloss = {p: array('f', [0]) for p in particles}
    for p in particles:
        tree.Branch(f"MC_point_{p}_count", mcpoint_count[p], f"MC_point_{p}_count/I")
        tree.Branch(f"MC_point_{p}_total_ELoss", mcpoint_eloss[p], f"MC_point_{p}_total_ELoss/F")


    # create a hist of number veto hits
    h_n_veto_hits = ROOT.TH1I("h_n_veto_hits", "Number of Veto Hits per Event;N Veto Hits;Entries", 20, 0, 20)

    # === Process events ===
    n_entries = digi_chain.GetEntries()
    for i in tqdm(range(n_entries), desc="Processing events"):
        digi_chain.GetEntry(i)
        hit2MC = digi_chain.Digi_MuFilterHits2MCPoints[0]

        event_index[0] = i
        event_pdg0 = digi_chain.MCTrack[0].GetPdgCode()
        event_pdg1 = digi_chain.MCTrack[1].GetPdgCode()

        neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
        if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
            neutrino_type[0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
        else:
            neutrino_type[0] = event_pdg0

        # cache MC start z if available
        start_z[0] = (
            digi_chain.MCTrack[1].GetStartZ()
            if hasattr(digi_chain, "MCTrack") and digi_chain.MCTrack.GetEntries() > 1
            else -999.0
        )

        n_veto_hit = 0

        for aHit in digi_chain.Digi_MuFilterHits:
            if not aHit.isValid() or aHit.GetSystem() != 1:  # Only Veto
                continue

            n_veto_hit += 1

            # Reset per-hit scalars
            hit_time[0] = float(aHit.GetTime())
            time_category[0] = 0 if hit_time[0] <= 25.0 else 1
            energy_loss[0] = 0.0
            only_proton_flag[0] = 0

            # Decode plane if possible; else -1
            try:
                veto_plane[0] = int(aHit.GetPlane())
            except Exception:
                veto_plane[0] = -1

            detID = aHit.GetDetectorID()
            linksToMCPoints = hit2MC.wList(detID)

            # Per-hit accumulation structures
            per_hit_count = {p: 0 for p in particles}
            per_hit_eloss = {p: 0.0 for p in particles}

            # Track e-loss per PDG to select a "dominant" PDG for this hit
            pdg_energy = defaultdict(float)

            # Loop over contributing MC points
            for mc_point_i, weight in linksToMCPoints:
                MC_point = digi_chain.MuFilterPoint[mc_point_i]

                pdg_code = int(MC_point.PdgCode())
                particle_type = get_particle_type(pdg_code)

                eloss = float(MC_point.GetEnergyLoss())
                # Optional momentum (computed but not stored; keep if you need later)
                # MC_track_momentum = (MC_point.GetPx()**2 + MC_point.GetPy()**2 + MC_point.GetPz()**2) ** 0.5

                # Weighted contributions (if weight encodes fraction); many setups use raw link weights ~[0,1]
                w_eloss = eloss * float(weight)

                # Accumulate totals
                energy_loss[0] += w_eloss
                per_hit_count[particle_type] += 1
                per_hit_eloss[particle_type] += w_eloss
                pdg_energy[pdg_code] += w_eloss

            # only_proton_flag: true iff there is at least one contributor and all are protons
            total_contrib = sum(per_hit_count[p] for p in particles)
            non_proton_contrib = total_contrib - per_hit_count["proton"]
            if total_contrib > 0 and non_proton_contrib == 0:
                only_proton_flag[0] = 1

            # Dominant PDG by summed e-loss (fallback -9999 if none)
            if pdg_energy:
                pdg[0] = max(pdg_energy.items(), key=lambda kv: kv[1])[0]
            else:
                pdg[0] = -9999

            # Push per-particle summaries into the branch arrays
            for p in particles:
                mcpoint_count[p][0] = int(per_hit_count[p])
                mcpoint_eloss[p][0] = float(per_hit_eloss[p])

            # Fill one TTree row per veto hit
            tree.Fill()

        # Fill per-event histogram once
        h_n_veto_hits.Fill(n_veto_hit)
        
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
    df = df.Define("particle", "get_particle_name(pdg)")
    
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
    
    df = df.Define("particle", "get_particle_name(pdg)")
    
    bin_min = 0
    bin_max = 0.005  # in GeV
    n_bins =100
    bin_width_GeV = (bin_max - bin_min) / n_bins
    bin_width_MeV = bin_width_GeV * 1e3  # convert to keV
    
    draw_pass_muon =False
    

    
    
    # ["mu+","mu-"]
    draw_particle = "All"
    #expected_particles = ["proton","antiproton"]
    expected_particles = ["mu", "neutron", "proton", "pi", "e"]
    
    if draw_pass_muon:
        plot_name = f"{draw_particle}_energy_loss_hist_with_passMuon_0-{int(bin_max*1e3)}Mev.pdf"
    else:
        plot_name = f"{draw_particle}_energy_loss_hist_0-{int(bin_max*1e3)}Mev.pdf"
    
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
            f"Energy Loss Distribution ;Energy Loss [GeV];Probability Density / {bin_width_MeV:.3f} MeV",
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
                f"Energy Loss Distribution ;Energy Loss [GeV];Probability Density / {bin_width_keV:.4f} keV",
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
    #c.SetLogy()
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
    #c.SetLogy()
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
    
def plot_momentum_vs_energyLoss(df, output_dir = "check_veto_hits_plots"):
    df = df.Define("particle", "get_particle_name(pdg)")
    
    expected_particles = ["mu", "e", "neutron", "pi", "proton"]# "e", "neutron", "pi", "proton"
    colors = [ROOT.kMagenta, ROOT.kOrange + 7, ROOT.kGreen + 2, ROOT.kBlue, ROOT.kRed]
    
    for i, particle in enumerate(expected_particles):
        df_particle = df.Filter(f'particle == "{particle}"')
        if df_particle.Count().GetValue() == 0:
            continue
        
        c = ROOT.TCanvas("c", "MCTrack Momentum vs Energy Loss", 800, 600)

        # Create 2D histogram
        h2d_proxy = df_particle.Histo2D(
            ("h2d", f"MCTrack Momentum ({particle}) vs Energy Loss;Energy Loss [GeV];MCTrack Momentum [GeV]", 
            100, 0, 0.01,
            100, 0, 1),      # Y-axis 
            "energy_loss", "MC_track_momentum"
        )
        h2d_proxy.SetDirectory(0)
        
        h2d = h2d_proxy.GetValue()

        # Style
        h2d.SetStats(False)
        h2d.Draw("COLZ")
        
        c.SetLeftMargin(0.15) 
        c.SetBottomMargin(0.1)

        # Save
        outpath = os.path.join(output_dir, f"MC_track_momentum_vs_energy_loss_2d_{particle}.pdf")
        #c.SetLogz()  # Log scale on color (Z) axis
        c.SaveAs(outpath)
        c.Close()
        
        
        # 1d momentum
        c = ROOT.TCanvas("c", f"MCTrack Momentum of {particle}", 800, 600)

        # Create 2D histogram
        h_proxy = df_particle.Histo1D(
                (f"h_{particle}",
                f"Momentum of {particle} Distribution ;Momentum ({particle}) [GeV];Entries",
                100, 0, 0.3),
                "MC_track_momentum"
            )
        h_proxy.SetDirectory(0)
        
        h = h_proxy.GetValue()
        h.Draw()

        # Style
        h.SetStats(False)
        
        c.SetLeftMargin(0.15) 
        c.SetBottomMargin(0.1)

        # Save
        outpath = os.path.join(output_dir, f"MC_track_momentum_1d_{particle}.pdf")
        #c.SetLogz()  # Log scale on color (Z) axis
        c.SaveAs(outpath)
        c.Close()
        
        
def plot_proton(
    df,
    output_dir="check_veto_hits_plots/lowEnergyProton",
    particles=("proton",),                 # extend if you want: ("e","neutron","pi","proton")
    bins_hitTime=(100, 0.0, 50),  
    bins_ELoss=(100, 0.0, 1e-6),
    bins_Momentum=(100, 0.0, 5e-3),
    bins_2d=((100, 0.0, 1e-6), (100, 0.0, 5e-3)),  # (x, y) binning for 2D
    set_batch=True
):
    """
    Plots (per particle):
      - 2D: MC_track_momentum vs energy_loss
      - 1D: MC_track_momentum
      - 1D: energy_loss
      - 1D: hit_time
    """
    
    if set_batch:
        ROOT.gROOT.SetBatch(True)

    os.makedirs(output_dir, exist_ok=True)

    # Resolve particle names once
    df_base = df.Define("particle", "get_particle_name(pdg)")

    # Colors (cycled)
    colors = [ROOT.kMagenta, ROOT.kOrange+7, ROOT.kGreen+2, ROOT.kBlue, ROOT.kRed]

    # Helper: book and draw a 1D histogram
    def draw_1d(df_node, var, title, xlab, filename,bins,  color=None, ):
        h_name = f"h_{var}_{title.replace(' ', '_')}"
        h_proxy = df_node.Histo1D((h_name, f"{title};{xlab};Entries", *bins), var)
        h_proxy.SetDirectory(0)
        h = h_proxy.GetValue()  # triggers (or reuses) event loop
        c = ROOT.TCanvas(f"c_{h_name}", h_name, 800, 600)
        h.SetStats(False)
        if color is not None:
            h.SetLineColor(color)
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.10)
        h.Draw("HIST")
        c.SaveAs(os.path.join(output_dir, filename))
        c.Close()

    # Helper: book and draw a 2D histogram
    def draw_2d(df_node, xvar, yvar, title, xlab, ylab, filename, bins_xy):
        (nx, xlow, xhigh), (ny, ylow, yhigh) = bins_xy
        h_name = f"h2d_{xvar}_vs_{yvar}_{title.replace(' ', '_')}"
        h_proxy = df_node.Histo2D(
            (h_name, f"{title};{xlab};{ylab}", nx, xlow, xhigh, ny, ylow, yhigh),
            xvar, yvar
        )
        h_proxy.SetDirectory(0)
        _ = h_proxy.GetValue()  # make sure it’s produced before drawing
        c = ROOT.TCanvas(f"c_{h_name}", h_name, 800, 600)
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.10)
        ROOT.gPad.SetRightMargin(0.18)
        h_proxy.Draw("COLZ")
        c.SaveAs(os.path.join(output_dir, filename))
        c.Close()

    for i, particle in enumerate(particles):
        color = colors[i % len(colors)]
        df_particle = df_base.Filter(f'particle == "{particle}"')
        
        max_mom = bins_Momentum[2]
        max_eloss = bins_ELoss[2]
        df_particle = df_particle.Filter(f'MC_track_momentum <= {max_mom} && energy_loss<= {max_eloss}' )

        # If no entries for that particle, skip quickly (cheap Count, only forces loop if not yet run)
        if df_particle.Count().GetValue() == 0:
            continue

        # 2D: momentum vs energy loss
        draw_2d(
            df_particle,
            xvar="energy_loss",
            yvar="MC_track_momentum",  # NOTE: fixed typo from "momentum"
            title=f"MCTrack Momentum ({particle}) vs Energy Loss",
            xlab="Energy Loss [GeV]",
            ylab="MCTrack Momentum [GeV]",
            filename=f"MC_track_mometum_vs_energy_loss_2d_lowEnergy_{particle}.pdf",
            bins_xy = bins_2d
        )

        # 1D: momentum
        draw_1d(
            df_particle,
            var="MC_track_momentum",   # NOTE: fixed typo from "momentum"
            title=f"Momentum of {particle} Distribution",
            xlab=f"Momentum ({particle}) [GeV]",
            filename=f"MC_track_momentum_1d_lowEnergy_{particle}.pdf",
            bins = bins_Momentum,
            color=color
        )

        # 1D: energy loss
        draw_1d(
            df_particle,
            var="energy_loss",
            title=f"Energy Loss of {particle} Distribution",
            xlab=f"Energy Loss ({particle}) [GeV]",
            filename=f"Energy_Loss_1d_lowEnergy_{particle}.pdf",
            bins = bins_ELoss,
            color=color
        )

        # 1D: hit time
        draw_1d(
            df_particle,
            var="hit_time",
            title=f"Hit Time of {particle} Distribution",
            xlab=f"Hit Time ({particle}) [ns]",  # unit likely ns; adjust if different
            filename=f"Hit_Time_1d_lowEnergy_{particle}.pdf",
            bins = bins_hitTime,
            color=color,
            # If your hit_time range is different, override bins here, e.g. bins=(200, 0, 200)
        )

        
    
    


    

def plot_veto_hist():
    tree_name = "veto_hits"
    file_name = "check_veto_hits_plots/veto_hits.root"
    pass_muon_file_name = "check_veto_hits_plots/pass_muon_veto_hits.root"
    
    df = ROOT.RDataFrame(tree_name, file_name)
    pass_muon_df = ROOT.RDataFrame(tree_name, pass_muon_file_name)
    
    # particle distribution (pdg)
    #plot_pdg(df)
    

    # veto hit time for each particles 
    #plot_hit_time(pass_muon_df)
    
    # energy loss for each particles 
    #plot_energy_loss(df, pass_muon_df)
    
    # plot momentum vs energy loss
    #plot_momentum_vs_energyLoss(df)
    
    # # stack hist of Start Z for each particles ((e, neutron, pi, proton, gamma))
    #plot_start_z(df)
    

    # # veto hit time vs energy loss
    # plot_time_vs_energy_loss(df)
    
    # # veto hit time vs stack hist of Start Z
    # plot_time_vs_start_z(df)
    
    # # energy loss vs stack hist of Start Z
    # plot_e_loss_vs_start_z(df)
    
    # plot low energy proton
    plot_proton(df)
    
    
    

    ## re-write the script
    ## each fill coorespond to each veto hit
    ## each veto hit, only record the total e-loss, hit time, N_type_mcPoint, MC_point_{particle}_count, MC_point_{particle}_total_ELoss,  particles = ["mu", "e", "neutron", "pi", "proton", "others"]
    ## 


if __name__ == "__main__":
    
    process_veto_hits_data()
    #process_pass_muon_hits_data()
    plot_veto_hist()
    
    
    