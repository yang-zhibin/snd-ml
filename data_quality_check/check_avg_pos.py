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


def select_neutral_bkg(df):
    
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
            if total >= 100000:
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



def process_avg_pos(metadata_df, metadata_name):
    # read feature path and eval path into rdf
    # filter rdf is needed
    # plot
    
    feature_chain = ROOT.TChain("snddata")
    eval_chain = ROOT.TChain("snddata")

    count = 0
    for index, row in metadata_df.iterrows():
        feature_path = row['feature_path']
        eval_path = row['eval_baseline_muon_output_path']
        feature_chain.Add(feature_path)
        eval_chain.Add(eval_path)

        if count>700:
           break
        count+=1

    print(f"{count} files read from {metadata_name}")
    feature_chain.AddFriend(eval_chain, 'eval')
    rdf = ROOT.RDataFrame(feature_chain)
    
    if "kaon" in metadata_name:
        plot_avg_pos(
            rdf, metadata_name,
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Kaon [MC Simulation]",
            bottom_title="Average SciFi Position"
        )
        plot_avg_pos(
            rdf, metadata_name,
            detector="DS",
            bin_width_x=1.5, bin_width_y=1.5,
            label_title="Kaon [MC Simulation]",
            bottom_title="Average DS Position"
        )

    elif "neutron" in metadata_name:
        plot_avg_pos(
            rdf, metadata_name,
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Neutron [MC Simulation]",
            bottom_title="Average SciFi Position"
        )
        plot_avg_pos(
            rdf, metadata_name,
            detector="DS",
            bin_width_x=1.5, bin_width_y=1.5,
            label_title="Neutron [MC Simulation]",
            bottom_title="Average DS Position"
        )

    elif "muon" in metadata_name:
        plot_avg_pos(
            rdf, metadata_name,
            detector="scifi",
            bin_width_x=0.55, bin_width_y=0.5,
            label_title="Muon [MC Simulation]",
            bottom_title="Average SciFi Position"
        )
        plot_avg_pos(
            rdf, metadata_name,
            detector="DS",
            bin_width_x=1.5, bin_width_y=1.5,
            label_title="Muon [MC Simulation]",
            bottom_title="Average DS Position"
        )

    elif "neutrino" in metadata_name:
        plot_avg_pos(
            rdf, metadata_name,
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Neutrino [MC Simulation]",
            bottom_title="Average SciFi Position"
        )
        plot_avg_pos(
            rdf, metadata_name,
            detector="DS",
            bin_width_x=1.5, bin_width_y=1.5,
            label_title="Neutrino [MC Simulation]",
            bottom_title="Average DS Position"
        )

    elif "real_data" in metadata_name:
        # muon-like 
        muon_like_rdf = rdf.Filter("pred_class_first == 6 && (veto1+veto2+veto3 == 0) ")
        plot_avg_pos(
            muon_like_rdf, 'real_data_muon_like',
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Muon-like [Real Data]",
            bottom_title="Average SciFi Position"
        )
        # plot_avg_pos(
        #     muon_like_rdf, 'real_data_muon_like',
        #     detector="DS",
        #     bin_width_x=1.5, bin_width_y=1.5,
        #     label_title="Muon-like [Real Data]",
        #     bottom_title="Average DS Position"
        # )

        # kaon-like
        kaon_like_rdf = rdf.Filter("pred_class_first == 4 && (veto1+veto2+veto3 == 0) ")
        plot_avg_pos(
            kaon_like_rdf, 'real_data_kaon_like',
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Kaon-like [Real Data]",
            bottom_title="Average SciFi Position"
        )
        # plot_avg_pos(
        #     kaon_like_rdf, 'real_data_kaon_like',
        #     detector="DS",
        #     bin_width_x=1.5, bin_width_y=1.5,
        #     label_title="Kaon-like [Real Data]",
        #     bottom_title="Average DS Position"
        # )

        # neutron-like
        neutron_like_rdf = rdf.Filter("pred_class_first == 5 && (veto1+veto2+veto3 == 0) ")
        plot_avg_pos(
            neutron_like_rdf, 'real_data_neutron_like',
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Neutron-like [Real Data]",
            bottom_title="Average SciFi Position"
        )
        # plot_avg_pos(
        #     neutron_like_rdf, 'real_data_neutron_like',
        #     detector="DS",
        #     bin_width_x=1.5, bin_width_y=1.5,
        #     label_title="Neutron-like [Real Data]",
        #     bottom_title="Average DS Position"
        # )
        
        
        
        veto_tagged_rdf = rdf.Filter("(veto1+veto2+veto3 > 0) ")
        
        plot_avg_pos(
            veto_tagged_rdf, 'real_data_veto_tagged',
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="Veto-tagged [Real Data]",
            bottom_title="Average SciFi Position"
        )
        # plot_avg_pos(
        #     veto_tagged_rdf, 'real_data_veto_tagged',
        #     detector="DS",
        #     bin_width_x=1.5, bin_width_y=1.5,
        #     label_title="Veto-tagged [Real Data]",
        #     bottom_title="Average DS Position"
        # )

        
        
        
        plot_avg_pos(
            rdf, metadata_name,
            detector="scifi",
            bin_width_x=0.5, bin_width_y=0.5,
            label_title="SND@LHC Real Data",
            bottom_title="Average SciFi Position"
        )
        # plot_avg_pos(
        #     rdf, metadata_name,
        #     detector="DS",
        #     bin_width_x=1.5, bin_width_y=1.5,
        #     label_title="SND@LHC Real Data",
        #     bottom_title="Average DS Position"
        # )


def read_rdf(metadata_df, metadata_name):
    feature_chain = ROOT.TChain("snddata")
    eval_chain = ROOT.TChain("snddata")

    count = 0

    for index, row in metadata_df.iterrows():
        feature_path = row['feature_path']
        eval_path = row['eval_baseline_muon_output_path']

        print(f"feature_path: {feature_path}")
        print(f"eval_path: {eval_path}")

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
        if count> 400:
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
    MC_kaon_rdf, MC_kaon_chain = read_rdf(MC_kaon, "MC_kaon")
    real_data_2024_rdf, real_data_2024_chain = read_rdf(real_data_2024, "real_data_2024")
    
    colors = {
        0: ROOT.kRed,
        1: ROOT.kBlue,
        2: ROOT.kGreen + 2,
        3: ROOT.kMagenta,
        4: ROOT.kOrange + 7,
        5: ROOT.kCyan,
        6: ROOT.kBlack,
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
    control_region_label = {4: "Kaon-like", 5: "Neutron-like", 6: "Muon-like"}
    
    print(real_data_2024_rdf.GetColumnNames())
    rdf = real_data_2024_rdf
    class_labels = control_region_label
    
    for plane, (n_bins, x_min, x_max, plane_name) in plane_binning.items():
        
        out_file = f"./n_hits/neutrino_and_data_no_preselection_n_hits_{plane}.pdf"
        canvas  = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        canvas.SetLogy()
        legend  = ROOT.TLegend(0.55, 0.50, 0.90, 0.90)
        
        any_drawn = False
        hist_proxies = []
        

        for class_id, label in tqdm(control_region_label.items(), desc="Processing class labels"):
            
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

            h.SetFillColorAlpha(color, 0.3)

            h.GetYaxis().SetRangeUser(1e-5, 80)
            draw_option = "HIST SAME " if any_drawn else "HIST"
            h.Draw(draw_option)
            legend.AddEntry(h, f"{label} ({n_evt})", "fl")
            any_drawn = True

                
        for class_id, label in tqdm(neutrino_label.items(), desc="Processing class labels"):
            
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

            h.GetYaxis().SetRangeUser(1e-5, 80)
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
    # MC_kaon_FTFP_BERT = drop_missing_files(MC_kaon_FTFP_BERT, "eval_baseline_muon_output_path", "MC_kaon_FTFP_BERT")
    # MC_neutron_FTFP_BERT = drop_missing_files(MC_neutron_FTFP_BERT, "eval_baseline_muon_output_path", "MC_neutron_FTFP_BERT")
    # MC_neutrino_volTarget_100fb_1 = drop_missing_files(MC_neutrino_volTarget_100fb_1, "eval_baseline_muon_output_path", "MC_neutrino_volTarget_100fb_1")
    
    real_data_2024 = drop_missing_files(real_data_2024, "eval_baseline_muon_output_path", "real_data_2024")
        
    
    # select portion of kaon and neutron()
    MC_kaon_FTFP_BERT = select_neutral_bkg(MC_kaon_FTFP_BERT)
    MC_neutron_FTFP_BERT = select_neutral_bkg(MC_neutron_FTFP_BERT)
    MC_neutrino_volTarget_100fb_1 = select_eval_neutrion(MC_neutrino_volTarget_100fb_1)
    
    
    #print(MC_kaon_FTFP_BERT.groupby("E_low")["n_event"].sum())
    #print(MC_muon_down)
    
    # process_avg_pos(MC_muon, 'MC_muon')
    # process_avg_pos(MC_kaon_FTFP_BERT, 'MC_kaon_FTFP_BERT')
    # process_avg_pos(MC_neutron_FTFP_BERT, 'MC_neutron_FTFP_BERT')
    # process_avg_pos(MC_neutrino_volTarget_100fb_1, 'MC_neutrino')
    
    #process_avg_pos(real_data_2024, 'real_data_2024')
    
    process_n_hit(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    
    #process_cut_eff(MC_neutrino_volTarget_100fb_1, MC_muon, MC_kaon_FTFP_BERT, MC_neutron_FTFP_BERT, real_data_2024)
    
    
    

if __name__ == "__main__":
    main()
    #cal_scan_fiducial_area()