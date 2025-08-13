import ROOT
import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()

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


def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data
        


def drop_missing_files(df: pd.DataFrame, column_name: str, metadata_name: str = "") -> pd.DataFrame:
    """Drop rows where the file in column_name does not exist. Print summary per metadata."""
    exists_mask = df[column_name].apply(lambda path: os.path.exists(path))
    missing_count = (~exists_mask).sum()

    if metadata_name:
        print(f"{metadata_name}: {missing_count} missing files in '{column_name}'")
    else:
        print(f"{missing_count} missing files in '{column_name}'")

    return df[exists_mask].reset_index(drop=True)


def select_neutral_bkg(df, select_evt = 1e5):
    
    pat = re.compile(r"""
        (?P<model>[^/]+)/                    # mc_model_type
        (?P<particle>[^_]+)_                 # particle
        (?P<E_low>\d+\.?\d*)_                # E_low
        (?P<E_high>\d+\.?\d*)                # E_high
        (?:_.*)?                             # optional suffix
    """, re.VERBOSE)

    def _parse(row):
        m = pat.fullmatch(row["subfolder"])
        if m is None:
            raise ValueError(f"Unparsable subfolder: {row['subfolder']}")
        gd = m.groupdict()
        return pd.Series({
            "mc_model": gd["model"],
            "particle": gd["particle"],
            "E_low": float(gd["E_low"]),
            "E_high": float(gd["E_high"]),
        })

    df = df.join(df.apply(_parse, axis=1))

    high_energy = df[df["E_low"] >= 100].copy()
    low_energy = df[df["E_low"] < 100].copy()

    selected_low = []

    # Group by bin
    for _, group in low_energy.groupby(["mc_model", "particle", "E_low", "E_high"]):
        group = group.sort_values("n_event", ascending=False)
        total = 0
        selected = []
        for _, row in group.iterrows():
            #print(row)
            if total >= select_evt:
                break
            selected.append(row)
            total += row["n_event"]
        selected_low.extend(selected)

    
    return pd.concat([high_energy, pd.DataFrame(selected_low)], ignore_index=True)

def plot_avg_pos(
    rdf,
    name,
    detector="scifi",
    bin_width_x=1.0,
    bin_width_y=1.0,
    label_title="",
    bottom_title="",
    logz=False
    
):
    #ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptStat("emr")

    # Choose variables and defaults
    if detector == "scifi":
        x_col, y_col = "scifi_avg_x_pos", "scifi_avg_y_pos"
        default_label = "Scifi X vs Y"
        output_path = f"avg_pos_plot/scifi_avg_pos_{name}.pdf"
    elif detector == "DS":
        x_col, y_col = "DS_avg_x_pos", "DS_avg_y_pos"
        default_label = "DS X vs Y"
        output_path = f"avg_pos_plot/DS_avg_pos_{name}.pdf"
    else:
        raise ValueError(f"Unknown detector type: {detector}")

    if not label_title:
        label_title = default_label
    if not bottom_title:
        bottom_title = "X vs Y Position Map"

    # Filter invalid values
    rdf = rdf.Filter(f"{x_col} > -99")

    # Define binning and ranges
    x_min, x_max = -70, 10
    y_min, y_max = 0, 80
    nbins_x = int((x_max - x_min) / bin_width_x)
    nbins_y = int((y_max - y_min) / bin_width_y)

    # Create histogram
    hist2d = rdf.Histo2D(
        ("h2", f";X Position [cm];Y Position [cm]", nbins_x, x_min, x_max, nbins_y, y_min, y_max),
        x_col, y_col
    )

    # Canvas setup
    canvas = ROOT.TCanvas(name, "", 800, 600)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.12)
    canvas.SetBottomMargin(0.15)

        
    hist2d.SetTitle("")  # Disable default title
    hist2d.Draw("COLZ")
    
    canvas.Update()
    stat = hist2d.GetListOfFunctions().FindObject("stats")
    if stat:
        stat.SetX1NDC(0.75)  # Left edge of box
        stat.SetX2NDC(0.85)  # Right edge (avoid overlap with color bar)
        stat.SetY1NDC(0.77)  # Bottom edge
        stat.SetY2NDC(0.87) 
        stat.SetName("")  # remove title from stat box 
        canvas.Update()
        
        
    if logz:
        canvas.SetLogz()

    # Draw top-left label
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.05)
    label.DrawLatex(0.12, 0.92, label_title)

    # Draw bottom title
    bottom = ROOT.TLatex()
    bottom.SetNDC()
    bottom.SetTextAlign(21)  # Centered
    bottom.SetTextFont(42)
    bottom.SetTextSize(0.05)
    bottom.DrawLatex(0.5, 0.04, bottom_title)

    canvas.SaveAs(output_path)

def plot_for_detectors(rdf, name_suffix, label_title, scifi_bin_width=0.5):
    plot_avg_pos(rdf, f'{name_suffix}', detector="scifi", bin_width_x=scifi_bin_width, bin_width_y=0.5,
                     label_title=label_title, bottom_title="Average SciFi Position")
    plot_avg_pos(rdf, f'{name_suffix}', detector="DS", bin_width_x=1.5, bin_width_y=1.5,
                     label_title=label_title, bottom_title="Average DS Position")

def process_avg_pos(metadata_df, metadata_name, scifi_min=0, scifi_max=100):
    rdf, rdf_chain= read_rdf(metadata_df, metadata_name, 700)
    scifi_sum_expr = "(scifi)"
    scifi_cut = f"({scifi_sum_expr} > {scifi_min} && {scifi_sum_expr} < {scifi_max})"

    

    mc_particle_titles = {
        "kaon": ("Kaon [MC Simulation]", 0.5),
        "muon": ("Muon [MC Simulation]", 0.55),
        "neutron": ("Neutron [MC Simulation]", 0.5),
        "neutrino": ("Neutrino [MC Simulation]", 0.5),
    }

    for particle, (label_title, scifi_bin_width) in mc_particle_titles.items():
        if particle in metadata_name:
            filtered_rdf = rdf.Filter(scifi_cut) if particle != "neutrino" else rdf
            suffix = f'{metadata_name}_{scifi_min}<scifi<{scifi_max}' if particle != "neutrino" else metadata_name
            plot_for_detectors(filtered_rdf, suffix, label_title, scifi_bin_width)
            return

    if "real_data" in metadata_name:
        base_cut = f"((veto1 + veto2 + veto3) == 0) && {scifi_cut}"
        real_classes = {
            "muon_like": ("(pred_class_first == 6)", "Muon-like [Real Data]"),
            "kaon_like": ("(pred_class_first == 4)", "Kaon-like [Real Data]"),
            "neutron_like": ("(pred_class_first == 5)", "Neutron-like [Real Data]"),
            "veto_tagged": ("(veto1 + veto2 + veto3) > 0", "Veto-tagged [Real Data]"),
        }

        for tag, (class_cut, label_title) in real_classes.items():
            if tag == "veto_tagged":
                full_cut = f"{class_cut} && {scifi_cut}"
            else:
                full_cut = f"{class_cut} && {base_cut}"
            filtered_rdf = rdf.Filter(full_cut)
            suffix = f'real_data_{tag}_{scifi_min}<scifi<{scifi_max}'
            plot_for_detectors(filtered_rdf, suffix, label_title)

        # Full unfiltered real data
        plot_for_detectors(rdf, metadata_name, "SND@LHC Real Data")

def read_rdf(metadata_df, metadata_name, min_file = 1e10):
    feature_chain = ROOT.TChain("snddata")
    eval_chain = ROOT.TChain("snddata")

    count = 0

    for index, row in metadata_df.iterrows():
        feature_path = row['feature_path']
        eval_path = row['eval_baseline_muon_output_path']

        #print(f"feature_path: {feature_path}")
        #print(f"eval_path: {eval_path}")

        # sanity check before adding
        f1 = ROOT.TFile.Open(feature_path)
        if not f1 or f1.IsZombie() or not f1.Get("snddata"):
            print(f"Warning: {feature_path} is not valid.")
            continue

        f2 = ROOT.TFile.Open(eval_path)
        if not f2 or f2.IsZombie() or not f2.Get("snddata"):
            print(f"Warning: {eval_path} is not valid.")
            continue
        
        feature_chain.Add(feature_path)
        eval_chain.Add(eval_path)

        count+=1
        if count> min_file:
           break

    print(f"{count} files read from {metadata_name}")
    #feature_chain.AddFriend(eval_chain, 'eval')
    feature_chain.AddFriend(eval_chain, 'eval')
    rdf = ROOT.RDataFrame(feature_chain)
    
    #print(rdf.GetColumnNames())
    rdf = rdf.Define('scifi', 'scifi1 + scifi2 + scifi3 + scifi4 + scifi5')
    rdf = rdf.Define('us', 'us1 + us2 + us3 + us4 + us5')
    rdf = rdf.Define('ds', 'ds1 + ds2 + ds3 + ds4')
    return rdf, feature_chain
    


def process_n_hit(MC_neutrino,  MC_muon, MC_kaon, MC_neutron, real_data_2024):
    
    MC_neutrino_rdf, MC_neutrino_chain = read_rdf(MC_neutrino, "MC_neutrino")
    MC_muon_rdf, MC_muon_chain = read_rdf(MC_muon, "MC_muon")
    #print(MC_kaon)
    #print(MC_neutron)
    MC_kaon_rdf, MC_kaon_chain = read_rdf(MC_kaon, "MC_kaon")
    MC_neutron_rdf, MC_neutron_chain = read_rdf(MC_neutron, "MC_neutron")
    real_data_2024_rdf, real_data_2024_chain = read_rdf(real_data_2024, "real_data_2024", 700)
    
    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen,
        3: ROOT.kMagenta,
        4: ROOT.kOrange ,
        5: ROOT.kCyan,
        6: ROOT.kViolet,
    }

    #plane, (n_bins, x_min, x_max), plane name
    plane_binning = {
    #'scifi':  (120, 0, 1200, 'SciFi Total'),
    'us':     (13, 0, 13, 'US Total'),
    'ds':     (40, 0, 40, 'DS Total'),
    
    'us1':    (13, 0, 13, 'US Station 1'),
    
    'scifi1': (120, 0, 1200, 'SciFi Station 1'),
    'scifi2': (120, 0, 1200, 'SciFi Station 2'),
    'scifi3': (120, 0, 1200, 'SciFi Station 3'),
    'scifi4': (120, 0, 1200, 'SciFi Station 4'),
    'scifi5': (120, 0, 1200, 'SciFi Station 5'),

    
    'us2':    (13, 0, 13, 'US Station 2'),
    'us3':    (13, 0, 13, 'US Station 3'),
    'us4':    (13, 0, 13, 'US Station 4'),
    'us5':    (13, 0, 13, 'US Station 5'),

    'ds1':    (40, 0, 40, 'DS Station 1'),
    'ds2':    (40, 0, 40, 'DS Station 2'),
    'ds3':    (40, 0, 40, 'DS Station 3'),
    'ds4':    (40, 0, 40, 'DS Station 4'),
    }
    neutrino_label = {0: "ve", 1: "vm", 2: "vt", 3: "NC"}
    control_region_label = {4: "Kaon-like", 5: "Neutron-like", 6: "Muon-like"}
    MC_bkg_label = {4: "MC Kaon", 5: "MC Neutron", 6: "MC Muon"}
    
    print(real_data_2024_rdf.GetColumnNames())
    rdf = real_data_2024_rdf
    class_labels = control_region_label
    
    for plane, (n_bins, x_min, x_max, plane_name) in plane_binning.items():
        
        out_file = f"./n_hits/MC_and_data_no_preselection_n_hits_{plane}.pdf"
        canvas  = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        if ("scifi" in plane):
            canvas.SetLogy()
        legend  = ROOT.TLegend(0.55, 0.50, 0.90, 0.90)
        
        any_drawn = False
        hist_proxies = []
        
         
        for class_id, label in tqdm(MC_bkg_label.items(), desc="Processing bkg class labels"):
            print(f"class_id: {class_id}, {label} ")
            if (label == "MC Kaon"):
                rdf_f = MC_kaon_rdf
            elif (label == "MC Neutron"):
                rdf_f = MC_neutron_rdf
            elif (label == "MC Muon"):
                rdf_f = MC_muon_rdf
            
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                print(f"No event in {label}")
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f";Number of hits;Probability density",
                 n_bins, x_min, x_max),
                plane
            )
            h = h_proxy.GetValue()
            hist_proxies.append(h_proxy)

            if h.Integral() == 0:
                print(f"Integral == 0 in {label}, event: {n_evt}")
                continue

            h.Scale(1.0 / h.Integral())
            color = colors.get(class_id, ROOT.kBlack) - 4

            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.5)
            h.SetMarkerColor(color)
            h.SetLineColorAlpha(color, 0.9)
            h.SetLineWidth(2)
            h.SetStats(0)
            h.SetLineStyle(2)
            #h.GetYaxis().SetRangeUser(0, 0.4)

            #h.SetFillColorAlpha(color, 0.3)
            h.SetFillColorAlpha(color, 0.3)

            if ("scifi" in plane):
                h.GetYaxis().SetRangeUser(1e-5, 80)
            else:
                h.GetYaxis().SetRangeUser(0, 1)
            draw_option = "HIST SAME" if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({n_evt})", "fl")
            any_drawn = True
        

        for class_id, label in tqdm(control_region_label.items(), desc="Processing data-bkg class labels"):
            
            rdf_f = real_data_2024_rdf.Filter(f"pred_class_first == {class_id} && scifi >= 2")
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f";Number of hits;Probability density",
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
            #h.GetYaxis().SetRangeUser(0, 0.4)

            

            draw_option = "HIST SAME " if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({n_evt})", "l")
            any_drawn = True

                
        for class_id, label in tqdm(neutrino_label.items(), desc="Processing neutrino class labels"):
            
            rdf_f = MC_neutrino_rdf.Filter(f"ParticleClass == {class_id}")
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f";Number of hits;Probability density",
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

            
            draw_option = "SAME P E" if any_drawn else "HIST P E"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({n_evt})", "lep")
            any_drawn = True
           
            
        legend.Draw()
        if any_drawn:
            label = ROOT.TLatex()
            label.SetNDC()
            label.SetTextFont(42)
            label.SetTextSize(0.05)
            label.DrawLatex(0.12, 0.92, plane_name)
        
            os.makedirs("plot", exist_ok=True)
            
            canvas.SaveAs(out_file)
            
            print(f"Saved: {out_file}")
        else:
            print(f"[{plane}] no valid histograms – skipped.")
        canvas.Close()
        
        #break
   


def process_n_hit_stack_seperately(MC_neutrino,  MC_muon, MC_kaon, MC_neutron, real_data_2024):
    #MC_neutrino_rdf, MC_neutrino_chain = read_rdf(MC_neutrino, "MC_neutrino")
    #MC_muon_rdf, MC_muon_chain = read_rdf(MC_muon, "MC_muon")
    #MC_kaon_rdf, MC_kaon_chain = read_rdf(MC_kaon, "MC_kaon")
    #MC_neutron_rdf, MC_neutron_chain = read_rdf(MC_neutron, "MC_neutron")
    real_data_2024_rdf, real_data_2024_chain = read_rdf(real_data_2024, "real_data_2024", 7)
    int_lumi = real_data_2024['lumi_per_file'].iloc[:7].sum()
    
    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen,
        3: ROOT.kMagenta,
        4: ROOT.kOrange ,
        5: ROOT.kCyan,
        6: ROOT.kViolet,
    }

    #plane, (n_bins, x_min, x_max), plane name
    plane_binning = {
    'scifi':  (120, 0, 1200, 'SciFi Total'),
    'us':     (13, 0, 13, 'US Total'),
    'ds':     (40, 0, 40, 'DS Total'),
    
    'us1':    (13, 0, 13, 'US Station 1'),
    
    'scifi1': (120, 0, 1200, 'SciFi Station 1'),
    'scifi2': (120, 0, 1200, 'SciFi Station 2'),
    'scifi3': (120, 0, 1200, 'SciFi Station 3'),
    'scifi4': (120, 0, 1200, 'SciFi Station 4'),
    'scifi5': (120, 0, 1200, 'SciFi Station 5'),

    
    'us2':    (13, 0, 13, 'US Station 2'),
    'us3':    (13, 0, 13, 'US Station 3'),
    'us4':    (13, 0, 13, 'US Station 4'),
    'us5':    (13, 0, 13, 'US Station 5'),

    'ds1':    (40, 0, 40, 'DS Station 1'),
    'ds2':    (40, 0, 40, 'DS Station 2'),
    'ds3':    (40, 0, 40, 'DS Station 3'),
    'ds4':    (40, 0, 40, 'DS Station 4'),
    }
    neutrino_label = {0: "ve", 1: "vm", 2: "vt", 3: "NC"}
    control_region_label = {4: "Kaon-like", 5: "Neutron-like"}
    MC_bkg_label = {4: "MC Kaon", 5: "MC Neutron"}
    
    print(real_data_2024_rdf.GetColumnNames())
    rdf = real_data_2024_rdf
    class_labels = control_region_label
    
    all_pairs = pd.concat([
        MC_kaon[['E_low', 'E_high']],
        MC_neutron[['E_low', 'E_high']]
    ])
    unique_pairs = all_pairs.drop_duplicates().sort_values(by=['E_low', 'E_high'])

    # Step 2: Group both DataFrames
    kaon_groups = MC_kaon.groupby(['E_low', 'E_high'])
    neutron_groups = MC_neutron.groupby(['E_low', 'E_high'])
    
    
    for plane, (n_bins, x_min, x_max, plane_name) in plane_binning.items():
        for class_id, label in tqdm(MC_bkg_label.items(), desc="Processing data-bkg class labels"):
            if label == 'MC Kaon':
                out_file = f"./n_hits/Stack_kaon_n_hits_{plane}_scifi_150-350.pdf"
                groups = kaon_groups
                particle_label = 'Kaon'
            else:
                out_file = f"./n_hits/Stack_neutron_n_hits_{plane}.pdf"
                groups = neutron_groups
                particle_label = 'Neutron'
                
            canvas = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
            #if ("scifi" in plane):
            canvas.SetLogy()
            legend = ROOT.TLegend(0.55, 0.50, 0.90, 0.90)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
        
            any_drawn = False
            
            filter_expr = f"eval.pred_class_first == {class_id} && scifi >150 && scifi < 350"
            n_file_read = 2000
            y_min = 1e-6
            y_max = 1e6
            
            
            
            
            stack = ROOT.THStack("stack", f";Number of hits; Events")


            #int_lumi = 1

            # Bin (5.0, 10.0)
            group1 = groups.get_group((5.0, 10.0))
            rdf1, rdf1_chain = read_rdf(group1, "MC_kaon", n_file_read)
            rdf1 = rdf1.Filter(filter_expr)
            h1 = rdf1.Histo1D(("h1", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h1.SetDirectory(0)
            
            
            
            print(f"lumi (5, 20){group1['lumi_per_file'].sum()}, count:{rdf1.Count().GetValue()}")
            
            h1.Scale(int_lumi / group1['lumi_per_file'].sum())
            h1.SetFillColor(ROOT.kRed+1)
            h1.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h1)
            legend.AddEntry(h1, f"MC {particle_label} 5-10 GeV", "f")

            # Bin (10.0, 20.0)
            group2 = groups.get_group((10.0, 20.0))
            rdf2, rdf2_chain = read_rdf(group2, "MC_kaon", n_file_read)
            rdf2 = rdf2.Filter(filter_expr)
            h2 = rdf2.Histo1D(("h2", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h2.SetDirectory(0)
            h2.Scale(int_lumi / group2['lumi_per_file'].sum())
            h2.SetFillColor(ROOT.kRed+2)
            h2.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h2)
            legend.AddEntry(h2, f"MC {particle_label} 10-20 GeV", "f")

            # Bin (20.0, 30.0)
            group3 = groups.get_group((20.0, 30.0))
            rdf3, rdf3_chain = read_rdf(group3, "MC_kaon", n_file_read)
            rdf3 = rdf3.Filter(filter_expr)
            h3 = rdf3.Histo1D(("h3", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h3.SetDirectory(0)
            h3.Scale(int_lumi / group3['lumi_per_file'].sum())
            h3.SetFillColor(ROOT.kBlue+1)
            h3.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h3)
            legend.AddEntry(h3, f"MC {particle_label} 20-30 GeV", "f")

            # Bin (30.0, 40.0)
            group4 = groups.get_group((30.0, 40.0))
            rdf4, rdf4_chain = read_rdf(group4, "MC_kaon", n_file_read)
            rdf4 = rdf4.Filter(filter_expr)
            h4 = rdf4.Histo1D(("h4", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h4.SetDirectory(0)
            h4.Scale(int_lumi / group4['lumi_per_file'].sum())
            h4.SetFillColor(ROOT.kBlue+2)
            h4.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h4)
            legend.AddEntry(h4, f"MC {particle_label} 30-40 GeV", "f")

            # Bin (40.0, 50.0)
            group5 = groups.get_group((40.0, 50.0))
            rdf5, rdf5_chain = read_rdf(group5, "MC_kaon", n_file_read)
            rdf5 = rdf5.Filter(filter_expr)
            h5 = rdf5.Histo1D(("h5", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h5.SetDirectory(0)
            h5.Scale(int_lumi / group5['lumi_per_file'].sum())
            h5.SetFillColor(ROOT.kGreen+2)
            h5.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h5)
            legend.AddEntry(h5, f"MC {particle_label} 40-50 GeV", "f")

            # Bin (50.0, 60.0)
            group6 = groups.get_group((50.0, 60.0))
            rdf6, rdf6_chain = read_rdf(group6, "MC_kaon", n_file_read)
            rdf6 = rdf6.Filter(filter_expr)
            h6 = rdf6.Histo1D(("h6", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h6.SetDirectory(0)
            h6.Scale(int_lumi / group6['lumi_per_file'].sum())
            h6.SetFillColor(ROOT.kGreen+3)
            h6.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h6)
            legend.AddEntry(h6, f"MC {particle_label} 50-60 GeV", "f")

            # Bin (60.0, 70.0)
            group7 = groups.get_group((60.0, 70.0))
            rdf7, rdf7_chain = read_rdf(group7, "MC_kaon", n_file_read)
            rdf7 = rdf7.Filter(filter_expr)
            h7 = rdf7.Histo1D(("h7", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h7.SetDirectory(0)
            h7.Scale(int_lumi / group7['lumi_per_file'].sum())
            h7.SetFillColor(ROOT.kCyan+1)
            h7.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h7)
            legend.AddEntry(h7, f"MC {particle_label} 60-70 GeV", "f")

            # Bin (70.0, 80.0)
            group8 = groups.get_group((70.0, 80.0))
            rdf8, rdf8_chain = read_rdf(group8, "MC_kaon", n_file_read)
            rdf8 = rdf8.Filter(filter_expr)
            h8 = rdf8.Histo1D(("h8", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h8.SetDirectory(0)
            h8.Scale(int_lumi / group8['lumi_per_file'].sum())
            h8.SetFillColor(ROOT.kCyan+2)
            h8.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h8)
            legend.AddEntry(h8, f"MC {particle_label} 70-80 GeV", "f")

            # Bin (80.0, 90.0)
            group9 = groups.get_group((80.0, 90.0))
            rdf9, rdf9_chain = read_rdf(group9, "MC_kaon", n_file_read)
            rdf9 = rdf9.Filter(filter_expr)
            h9 = rdf9.Histo1D(("h9", "", n_bins, x_min, x_max), plane).GetValue().Clone()
            h9.SetDirectory(0)
            h9.Scale(int_lumi / group9['lumi_per_file'].sum())
            h9.SetFillColor(ROOT.kMagenta+1)
            h9.GetYaxis().SetRangeUser(y_min, y_max)
            stack.Add(h9)
            legend.AddEntry(h9, f"MC {particle_label} 80-90 GeV", "f")
            
            # Draw and save
            
            stack_sum = stack.GetStack().Last().Clone("stack_sum")
            total_integral = stack_sum.Integral()
            if total_integral > 0:
                for h in stack.GetHists():
                    h.Scale(1.0 / total_integral)
            
            stack.SetMinimum(y_min)
            stack.SetMaximum(y_max)
            stack.Draw("HIST")
            canvas.Update()
            
            canvas.Update()
            
            any_drawn = True
            
            
            rdf_f = real_data_2024_rdf.Filter(filter_expr)
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                    f";Number of hits;Probability density",
                    n_bins, x_min, x_max),
                plane
            )
            h = h_proxy.GetValue()

            if h.Integral() == 0:
                continue

            #h.Scale(1.0 / h.Integral())
            color = colors.get(class_id, ROOT.kBlack)

            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.5)
            h.SetMarkerColor(color)
            h.SetLineColorAlpha(color, 0.9)
            h.SetLineWidth(2)
            h.SetStats(0)
            h.GetYaxis().SetRangeUser(1e-6, 1e5)
            
            draw_option = "HIST SAME " if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{particle_label}-like (Data)", "l")
            any_drawn = True
            
            
            #if "scifi" in plane:
            #    stack.GetYaxis().SetRangeUser(1e-5, 80)
            #else:
            #  stack.GetYaxis().SetRangeUser(0, 1)
            
            



            
            #h.GetYaxis().SetRangeUser(0, 0.4)
            # if ("scifi" in plane):
            #     h.GetYaxis().SetRangeUser(1e-5, 80)
            # else:
            #     h.GetYaxis().SetRangeUser(0, 1)
            

            



        


            legend.Draw()
            if any_drawn:
                label = ROOT.TLatex()
                label.SetNDC()
                label.SetTextFont(42)
                label.SetTextSize(0.040)  # smaller than 0.045
                label.SetTextAlign(31)
                label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {int_lumi:.2f} fb^{{-1}}")
                label.DrawLatex(0.20, 0.94, plane_name)
            
                os.makedirs("plot", exist_ok=True)
                
                canvas.SaveAs(out_file)
                
                print(f"Saved: {out_file}")
            else:
                print(f"[{plane}] no valid histograms – skipped.")
            canvas.Close()
            
        
            
        #break


def process_n_hit_stack(MC_neutrino,  MC_muon, MC_kaon, MC_neutron, real_data_2024):
    #MC_neutrino_rdf, MC_neutrino_chain = read_rdf(MC_neutrino, "MC_neutrino")
    #MC_muon_rdf, MC_muon_chain = read_rdf(MC_muon, "MC_muon")
    #MC_kaon_rdf, MC_kaon_chain = read_rdf(MC_kaon, "MC_kaon")
    #MC_neutron_rdf, MC_neutron_chain = read_rdf(MC_neutron, "MC_neutron")
    real_data_2024_rdf, real_data_2024_chain = read_rdf(real_data_2024, "real_data_2024", 2)
    
    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen,
        3: ROOT.kMagenta,
        4: ROOT.kOrange ,
        5: ROOT.kCyan,
        6: ROOT.kViolet,
    }

    #plane, (n_bins, x_min, x_max), plane name
    plane_binning = {
    'scifi':  (24, 0, 1200, 'SciFi Total'),
    'us':     (13, 0, 13, 'US Total'),
    'ds':     (40, 0, 40, 'DS Total'),
    
    'us1':    (13, 0, 13, 'US Station 1'),
    
    'scifi1': (120, 0, 1200, 'SciFi Station 1'),
    'scifi2': (120, 0, 1200, 'SciFi Station 2'),
    'scifi3': (120, 0, 1200, 'SciFi Station 3'),
    'scifi4': (120, 0, 1200, 'SciFi Station 4'),
    'scifi5': (120, 0, 1200, 'SciFi Station 5'),

    
    'us2':    (13, 0, 13, 'US Station 2'),
    'us3':    (13, 0, 13, 'US Station 3'),
    'us4':    (13, 0, 13, 'US Station 4'),
    'us5':    (13, 0, 13, 'US Station 5'),

    'ds1':    (40, 0, 40, 'DS Station 1'),
    'ds2':    (40, 0, 40, 'DS Station 2'),
    'ds3':    (40, 0, 40, 'DS Station 3'),
    'ds4':    (40, 0, 40, 'DS Station 4'),
    }
    neutrino_label = {0: "ve", 1: "vm", 2: "vt", 3: "NC"}
    control_region_label = {4: "Kaon-like", 5: "Neutron-like"}
    MC_bkg_label = {4: "MC Kaon", 5: "MC Neutron"}
    
    print(real_data_2024_rdf.GetColumnNames())
    rdf = real_data_2024_rdf
    class_labels = control_region_label
    
    all_pairs = pd.concat([
        MC_kaon[['E_low', 'E_high']],
        MC_neutron[['E_low', 'E_high']]
    ])
    unique_pairs = all_pairs.drop_duplicates().sort_values(by=['E_low', 'E_high'])

    # Step 2: Group both DataFrames
    kaon_groups = MC_kaon.groupby(['E_low', 'E_high'])
    neutron_groups = MC_neutron.groupby(['E_low', 'E_high'])
    
    
    for plane, (n_bins, x_min, x_max, plane_name) in plane_binning.items():
        
        out_file = f"./n_hits/high_stats_Stack_neutral_bkg_n_hits_{plane}.pdf"
        canvas = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        if ("scifi" in plane):
            canvas.SetLogy()
        legend = ROOT.TLegend(0.55, 0.50, 0.90, 0.90)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        
        any_drawn = False
        hist_proxies = []

        h_total_kaon = None
        h_total_neutron = None
        
        for idx, (E_low, E_high) in enumerate(tqdm(unique_pairs.itertuples(index=False), total=len(unique_pairs))):

            kaon_group = kaon_groups.get_group((E_low, E_high)) if (E_low, E_high) in kaon_groups.groups else pd.DataFrame()
            neutron_group = neutron_groups.get_group((E_low, E_high)) if (E_low, E_high) in neutron_groups.groups else pd.DataFrame()
            kaon_lumi = kaon_group['lumi_per_file'].sum()
            neutron_lumi = neutron_group['lumi_per_file'].sum()
            
            MC_kaon_rdf, MC_kaon_chain = read_rdf(kaon_group, "MC_kaon", 2)
            MC_neutron_rdf, MC_neutron_chain = read_rdf(neutron_group, "MC_neutron", 2)

            MC_kaon_rdf = MC_kaon_rdf.Filter("pred_class_first == 4 && scifi >= 2")
            MC_neutron_rdf = MC_neutron_rdf.Filter("pred_class_first == 5 && scifi >= 2")

            # Kaon hist
            h_proxy_kaon = MC_kaon_rdf.Histo1D((f"h_{plane}_kaon_E{E_low}_{E_high}", "", n_bins, x_min, x_max), plane)
            h_kaon = h_proxy_kaon.GetValue()
            h_kaon.Scale(kaon_lumi)
            # add to kaon hist
            if h_total_kaon is None:
                h_total_kaon = h_kaon.Clone("h_total_kaon")
                h_total_kaon.SetDirectory(0)  # Important if you're not writing to file immediately
            else:
                h_total_kaon.Add(h_kaon)

            
            # Neutron hist
            h_proxy_neutron = MC_neutron_rdf.Histo1D((f"h_{plane}_neutron_E{E_low}_{E_high}", "", n_bins, x_min, x_max), plane)
            h_neutron = h_proxy_neutron.GetValue()
            h_neutron.Scale(neutron_lumi)
            # add to neutron hists
            if h_total_neutron is None:
                h_total_neutron = h_neutron.Clone("h_total_neutron")
                h_total_neutron.SetDirectory(0)
            else:
                h_total_neutron.Add(h_neutron)

        stack = ROOT.THStack("stack", f"{plane} stacked distribution;{plane};Entries")
        
        kaon_color = colors.get(4, ROOT.kBlack)- 4
        neutron_color = colors.get(5, ROOT.kBlack)- 4
        h_total_kaon.SetFillColorAlpha(kaon_color, 0.3)     
        h_total_neutron.SetFillColorAlpha(neutron_color, 0.3)  

        # Add histograms to stack
        
        h_total_kaon.Scale(1.0 / h_total_kaon.Integral())
        h_total_neutron.Scale(1.0 / h_total_neutron.Integral())
        
        
        stack.Add(h_total_kaon)
        stack.Add(h_total_neutron)
        
        legend.AddEntry(h_total_kaon, "MC Kaon", "f")
        legend.AddEntry(h_total_neutron, "MC Neutron", "f")
        # Draw the stack
        stack.Draw("HIST SAME")
        if ("scifi" in plane):
            stack.GetYaxis().SetRangeUser(1e-5, 80)
        else:
            stack.GetYaxis().SetRangeUser(0, 1)
        any_drawn = True
        
        
        
        for class_id, label in tqdm(control_region_label.items(), desc="Processing data-bkg class labels"):
            
            rdf_f = real_data_2024_rdf.Filter(f"pred_class_first == {class_id} && scifi >= 2")
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f";Number of hits;Probability density",
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
            #h.GetYaxis().SetRangeUser(0, 0.4)
            if ("scifi" in plane):
                h.GetYaxis().SetRangeUser(1e-5, 80)
            else:
                h.GetYaxis().SetRangeUser(0, 1)
            

            

            draw_option = "HIST SAME " if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({n_evt})", "l")
            any_drawn = True

        


        legend.Draw()
        if any_drawn:
            label = ROOT.TLatex()
            label.SetNDC()
            label.SetTextFont(42)
            label.SetTextSize(0.05)
            label.DrawLatex(0.12, 0.92, plane_name)
        
            os.makedirs("plot", exist_ok=True)
            
            canvas.SaveAs(out_file)
            
            print(f"Saved: {out_file}")
        else:
            print(f"[{plane}] no valid histograms – skipped.")
        canvas.Close()
        
        
        break

def select_eval_neutrion(MC_neutrino):
    train_csv = '/eos/user/z/zhibin/sndData/converted/combined_train.csv'
    train_df = pd.read_csv(train_csv)
    
    # Filter to Neutrinos
    train_df = train_df[train_df['partition'] == 'Neutrinos'].copy()
    
    # Extract partition
    train_df['partition'] = train_df['file'].str.extract(r'/Neutrinos/(\d+)/sndLHC')[0]
    train_df.dropna(subset=['partition'], inplace=True)

    # Ensure type consistency
    train_partitions = train_df['partition'].astype(str).unique()
    MC_neutrino['partition'] = MC_neutrino['partition'].astype(str)

    # Debug print of matching rows
    matching_rows = MC_neutrino[MC_neutrino['partition'].isin(train_partitions)]
    print("Dropping the following paths:")
    #print(matching_rows[['partition', 'digi_path']])

    # Filter out training partitions
    MC_neutrino = MC_neutrino[~MC_neutrino['partition'].isin(train_partitions)]
    #print(MC_neutrino)
    return MC_neutrino
    
    
def plot_n_hit_with_energy(MC_kaon, MC_neutron , E_low, E_high):
    MC_kaon_rdf,  MC_kaon_chain =  read_rdf(MC_kaon, "MC_kaon")
    MC_neutron_rdf,  MC_neutron_chain =  read_rdf(MC_neutron, "MC_neutron")
    E_low = int(E_low)
    E_high = int(E_high)
    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen,
        3: ROOT.kMagenta,
        4: ROOT.kOrange ,
        5: ROOT.kCyan,
        6: ROOT.kViolet,
    }

    #plane, (n_bins, x_min, x_max), plane name
    plane_binning = {
    'scifi':  (120, 0, 1200, 'SciFi Total'),
    'us':     (13, 0, 13, 'US Total'),
    'ds':     (40, 0, 40, 'DS Total'),
    
    'us1':    (13, 0, 13, 'US Station 1'),
    
    # 'scifi1': (120, 0, 1200, 'SciFi Station 1'),
    # 'scifi2': (120, 0, 1200, 'SciFi Station 2'),
    # 'scifi3': (120, 0, 1200, 'SciFi Station 3'),
    # 'scifi4': (120, 0, 1200, 'SciFi Station 4'),
    # 'scifi5': (120, 0, 1200, 'SciFi Station 5'),

    
    # 'us2':    (13, 0, 13, 'US Station 2'),
    # 'us3':    (13, 0, 13, 'US Station 3'),
    # 'us4':    (13, 0, 13, 'US Station 4'),
    # 'us5':    (13, 0, 13, 'US Station 5'),

    # 'ds1':    (40, 0, 40, 'DS Station 1'),
    # 'ds2':    (40, 0, 40, 'DS Station 2'),
    # 'ds3':    (40, 0, 40, 'DS Station 3'),
    # 'ds4':    (40, 0, 40, 'DS Station 4'),
    }
    MC_bkg_label = {4: "MC Kaon", 5: "MC Neutron"}
    

    
    for plane, (n_bins, x_min, x_max, plane_name) in plane_binning.items():
        
        out_file = f"./n_hits_MC_neutral/MC_and_data_no_preselection_n_hits_{plane}_E{E_low}_{E_high}.pdf"
        canvas  = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        if ("scifi" in plane):
            canvas.SetLogy()
        legend  = ROOT.TLegend(0.55, 0.50, 0.90, 0.90)
        
        any_drawn = False
        hist_proxies = []
        
         
        for class_id, label in tqdm(MC_bkg_label.items(), desc="Processing bkg class labels"):
            print(f"class_id: {class_id}, {label} ")
            if (label == "MC Kaon"):
                rdf_f = MC_kaon_rdf
            elif (label == "MC Neutron"):
                rdf_f = MC_neutron_rdf
            elif (label == "MC Muon"):
                rdf_f = MC_muon_rdf
            
            n_evt = rdf_f.Count().GetValue()
            if n_evt == 0:
                print(f"No event in {label}")
                continue
            title = f"{plane_name}"
            h_proxy = rdf_f.Histo1D(
                (f"h_{plane}_{label}",
                 f";Number of hits;Probability density",
                 n_bins, x_min, x_max),
                plane
            )
            h = h_proxy.GetValue()
            hist_proxies.append(h_proxy)

            if h.Integral() == 0:
                print(f"Integral == 0 in {label}, event: {n_evt}")
                continue

            h.Scale(1.0 / h.Integral())
            color = colors.get(class_id, ROOT.kBlack) - 4

            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.5)
            h.SetMarkerColor(color)
            h.SetLineColorAlpha(color, 0.9)
            h.SetLineWidth(2)
            h.SetStats(0)
            h.SetLineStyle(2)
            #h.GetYaxis().SetRangeUser(0, 0.4)

            #h.SetFillColorAlpha(color, 0.3)
            h.SetFillColorAlpha(color, 0.3)
            if ("scifi" in plane):
                h.GetYaxis().SetRangeUser(1e-5, 80)
            else:
                h.GetYaxis().SetRangeUser(0, 1)
            draw_option = "HIST SAME" if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({E_low}-{E_high} GeV)", "fl")
            any_drawn = True
    
           
            
        legend.Draw()
        if any_drawn:
            label = ROOT.TLatex()
            label.SetNDC()
            label.SetTextFont(42)
            label.SetTextSize(0.05)
            label.DrawLatex(0.12, 0.92, plane_name)
        
            os.makedirs("plot", exist_ok=True)
            
            canvas.SaveAs(out_file)
            
            print(f"Saved: {out_file}")
        else:
            print(f"[{plane}] no valid histograms – skipped.")
        canvas.Close()
        
        #break
    
    
    

def process_n_hits_in_diff_energy(MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT):
    MC_kaon_df = select_neutral_bkg(MC_kaon_FTFP_BERT, 1e5)
    MC_neutron_df = select_neutral_bkg(MC_neutron_FTFP_BERT, 1e5)
    
    # Step 1: Get all unique (E_low, E_high) pairs across both DataFrames
    all_pairs = pd.concat([
        MC_kaon_df[['E_low', 'E_high']],
        MC_neutron_df[['E_low', 'E_high']]
    ])
    unique_pairs = all_pairs.drop_duplicates().sort_values(by=['E_low', 'E_high'])

    # Step 2: Group both DataFrames
    kaon_groups = MC_kaon_df.groupby(['E_low', 'E_high'])
    neutron_groups = MC_neutron_df.groupby(['E_low', 'E_high'])

    # Step 3: Iterate over all unique energy pairs and access both groups
    for E_low, E_high in unique_pairs.itertuples(index=False):
        kaon_group = kaon_groups.get_group((E_low, E_high)) if (E_low, E_high) in kaon_groups.groups else pd.DataFrame()
        neutron_group = neutron_groups.get_group((E_low, E_high)) if (E_low, E_high) in neutron_groups.groups else pd.DataFrame()
        plot_n_hit_with_energy(kaon_group,neutron_group, E_low, E_high)
        
    pass
    # select 10^5 events for each energy bin
    # one plot one energy bin, two hist in one plot (kaon and neutron)
    
    

def main():
    mc_files = [
        "MC_kaon_FTFP_BERT_metadata.csv",
        "MC_neutron_FTFP_BERT_metadata.csv",
        "MC_muon_down_metadata.csv",
        "MC_muon_horizontal_metadata.csv",
        "MC_muon_up_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
        "real_data_2024_metadata.csv",
    ]

    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'

    metadata_vars = load_metadata_files(mc_files, root_path)

    MC_kaon_FTFP_BERT = metadata_vars["MC_kaon_FTFP_BERT"]
    MC_neutron_FTFP_BERT = metadata_vars["MC_neutron_FTFP_BERT"]
    MC_muon_down = metadata_vars["MC_muon_down"]
    MC_muon_horizontal = metadata_vars["MC_muon_horizontal"]
    MC_muon_up = metadata_vars["MC_muon_up"]
    MC_neutrino_volTarget_100fb_1 = metadata_vars["MC_neutrino_volTarget_100fb_1"]
    real_data_2024 = metadata_vars["real_data_2024"]
    
    # Drop missing 'feature_path' files
    MC_muon_down = drop_missing_files(MC_muon_down, "eval_baseline_muon_output_path", "MC_muon_down")
    MC_muon_horizontal = drop_missing_files(MC_muon_horizontal, "eval_baseline_muon_output_path", "MC_muon_horizontal")
    MC_muon_up = drop_missing_files(MC_muon_up, "eval_baseline_muon_output_path", "MC_muon_up")
    MC_muon = pd.concat([MC_muon_down, MC_muon_horizontal, MC_muon_up], ignore_index=True)
    
    
    # Drop missing 'eval_baseline_muon_output_path' files
    MC_kaon_FTFP_BERT = drop_missing_files(MC_kaon_FTFP_BERT, "eval_baseline_muon_output_path", "MC_kaon_FTFP_BERT")
    MC_neutron_FTFP_BERT = drop_missing_files(MC_neutron_FTFP_BERT, "eval_baseline_muon_output_path", "MC_neutron_FTFP_BERT")
    MC_neutrino_volTarget_100fb_1 = drop_missing_files(MC_neutrino_volTarget_100fb_1, "eval_baseline_muon_output_path", "MC_neutrino_volTarget_100fb_1")
    
    real_data_2024 = drop_missing_files(real_data_2024, "eval_baseline_muon_output_path", "real_data_2024")
    
    
    #process_n_hits_in_diff_energy(MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT)
    
    # select portion of kaon and neutron()
    MC_kaon_FTFP_BERT = select_neutral_bkg(MC_kaon_FTFP_BERT, 1e4)
    MC_neutron_FTFP_BERT = select_neutral_bkg(MC_neutron_FTFP_BERT)
    MC_neutrino_volTarget_100fb_1 = select_eval_neutrion(MC_neutrino_volTarget_100fb_1)
    
    
    #print(MC_kaon_FTFP_BERT.groupby("E_low")["n_event"].sum())
    #print(MC_muon_down)
    
    #process_avg_pos(MC_muon, 'MC_muon')
    #process_avg_pos(MC_kaon_FTFP_BERT, 'MC_kaon_FTFP_BERT')
    #process_avg_pos(MC_neutron_FTFP_BERT, 'MC_neutron_FTFP_BERT')
    #process_avg_pos(MC_neutrino_volTarget_100fb_1, 'MC_neutrino')
    
    #process_avg_pos(real_data_2024, 'real_data_2024')
    
    #process_n_hit(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    
    #process_cut_eff(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    
    #process_n_hit_stack(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    process_n_hit_stack_seperately(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    
    

if __name__ == "__main__":
    main()
    #cal_scan_fiducial_area()