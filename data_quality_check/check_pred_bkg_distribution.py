
# read all the metadata
# check file exist
# read feature and prediciton at the same time to rdf
# filter rdf
# plot hist (n scifi hit, avg position )

import pandas as pd
import argparse
import os
import ROOT
from tqdm import tqdm

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()

def read_exist_output(dir_data, metadata_data_df):
    def file_exists(row):
        file_path = row['eval_baseline_muon_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def read_metadata():
    metadata_data_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_metadata.csv'
    metadata_data_df = pd.read_csv(metadata_data_path)
    dir_data = '/eos/experiment/sndlhc/users/zhibin/real_data/'
    metadata_data_df = read_exist_output(dir_data, metadata_data_df)

    return metadata_data_df


particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}


def pdg_2_particle(rdf):
    rdf = rdf.Define("ParticleType", """
        if (PdgCode == 12 || PdgCode == -12) return std::string("ve");
        else if (PdgCode == 14 || PdgCode == -14) return std::string("vm");
        else if (PdgCode == 16 || PdgCode == -16) return std::string("vt");
        else if (PdgCode == 112 || PdgCode == -112 || PdgCode == 114 || PdgCode == -114 || PdgCode == 116 || PdgCode == -116) return std::string("NC");
        else if (PdgCode == 130 || PdgCode == 310) return std::string("kaon");
        else if (PdgCode == 2112) return std::string("neutron");
        else if (PdgCode == 13 || PdgCode == -13) return std::string("muon");
        else if (PdgCode == 0 ) return std::string("real_data");
        else return std::string("others");
    """)

    rdf = rdf.Define("ParticleClass", """
        if (ParticleType == "ve") return 0;
        else if (ParticleType == "vm") return 1;
        else if (ParticleType == "vt") return 2;
        else if (ParticleType == "NC") return 3;
        else if (ParticleType == "kaon") return 4;
        else if (ParticleType == "neutron") return 5;
        else if (ParticleType == "muon") return 6;
        else return -1;  // e.g. for "real_data" or "others"
    """)

    argmax_expr = """
        double vals[7] = {Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5, Prediction_6};
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
    rdf = rdf.Define("PredClass", argmax_expr)
    return rdf
def plot_hits(rdf, name_prefix):
    # Define hit variables
    rdf = rdf.Define("n_scifi_hits", "scifi1 + scifi2 + scifi3 + scifi4 + scifi5")
    rdf = rdf.Define("n_us_hits", "us1 + us2 + us3 + us4 + us5")
    rdf = rdf.Define("n_ds_hits", "ds1 + ds2 + ds3 + ds4")

    rdf = rdf.Filter("n_scifi_hits > 2")

    class_labels = {
        4: "pred_kaon",
        5: "pred_neutron",
        6: "pred_muon"
    }

    colors = [2, 4, 8, 6, 46, 38, 28]
    markers = [20, 21, 22, 23, 24, 25, 33]

    hit_vars = [
        ("n_scifi_hits", 200, 0, 1000),
        ("n_us_hits", 50, 0, 50),
        ("n_ds_hits", 100, 0, 250)
    ]

    for hit_var, bins, x_min, x_max in hit_vars:
        canvas = ROOT.TCanvas(f"c_{hit_var}", f"{hit_var} by PredClass", 800, 600)
        ROOT.gStyle.SetOptStat(0)
        legend = ROOT.TLegend(0.65, 0.6, 0.88, 0.88)
        any_drawn = False
        hist_proxies = []

        for idx, (class_id, label) in enumerate(class_labels.items()):
            filtered = rdf.Filter(f"PredClass == {class_id}")
            n_events = filtered.Count().GetValue()
            if n_events == 0:
                continue

            hist_proxy = filtered.Histo1D(
                (f"hist_{hit_var}_{label}", f"{hit_var} for {label}", bins, x_min, x_max),
                hit_var
            )
            hist = hist_proxy.GetValue()
            hist_proxies.append(hist_proxy)

            if hist.Integral() == 0:
                continue

            hist.Scale(1.0 / hist.Integral())
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            hist.SetLineColor(color)
            hist.SetMarkerColor(color)
            hist.SetMarkerStyle(marker)
            hist.SetMarkerSize(0.5)
            hist.SetLineWidth(2)
            hist.SetFillColorAlpha(color, 0.35)

            hist.GetXaxis().SetTitle(hit_var)
            hist.GetYaxis().SetTitle("Density")
            hist.GetYaxis().SetRangeUser(0, 1)
            #hist.Draw("PE HIST SAME" if any_drawn else "PE HIST")
            hist.Draw("HIST SAME F" if any_drawn else "HIST F")
            legend.AddEntry(hist, f"{label} ({n_events} events)", "lep")
            any_drawn = True

        legend.Draw()

        if any_drawn:
            os.makedirs("plot", exist_ok=True)
            output_path = f"plot/{name_prefix}_{hit_var}.pdf"
            canvas.SaveAs(output_path)
            print(f"Saved: {output_path}")
        else:
            print(f"No valid histograms for {hit_var}")

def cal_pos(index, n_ch, pos_range):
    return pos_range[0] + (index) * (pos_range[1] - pos_range[0]) / (n_ch)


def check_fiducial_pos():
    scifi_n_ch = 1536
    scifi_hor_ch = [300, 1336]
    scifi_ver_ch = [200, 1200]
    scfit_hor_limit_pos = [14.21, 53.86]
    scfit_ver_limit_pos = [-46.09, -6.99]

    DS_n_bar = 60
    DS_hor_bar = [10, 50]
    DS_ver_bar = [15, 50]#DS_ver_bar = [70-60, 105-60]
    DS_hor_limit_pos = [7.61, 67.58]
    DS_ver_limit_pos = [-61.98, 1.72]

    scifi_hor_pos = []
    scifi_ver_pos = []
    DS_hor_pos = []
    DS_ver_pos = []

    scifi_hor_pos = [cal_pos(ch, scifi_n_ch, scfit_hor_limit_pos) for ch in scifi_hor_ch]
    scifi_ver_pos = [cal_pos(ch, scifi_n_ch, scfit_ver_limit_pos) for ch in scifi_ver_ch]

    DS_hor_pos = [cal_pos(bar, DS_n_bar, DS_hor_limit_pos) for bar in DS_hor_bar]
    DS_ver_pos = [cal_pos(bar, DS_n_bar, DS_ver_limit_pos) for bar in DS_ver_bar]

    return scifi_hor_pos, scifi_ver_pos, DS_hor_pos, DS_ver_pos


def plot_avg(df, output_pdf):
    os.makedirs("plot", exist_ok=True)

    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.Print(output_pdf + "[")

    scifi_hor_pos, scifi_ver_pos, DS_hor_pos, DS_ver_pos = check_fiducial_pos()
    scifi_hor_ch = [300, 1336]
    scifi_ver_ch = [200, 1200]
    DS_hor_bar = [10, 50]
    DS_ver_bar = [70, 105]

    

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.03)
    latex.SetTextFont(42)
    # Format: (x_col, y_col, title, x_min, x_max, y_min, y_max)
    plots = [
        ("scifi_avg_x_pos", "scifi_avg_y_pos", "Scifi X vs Y;X Position;Y Position", -90, 10, 0, 90),
        ("DS_avg_x_pos", "DS_avg_y_pos", "DS X vs Y;X Position;Y Position", -90, 10, 0, 90),

        ("scifi_avg_ver", "scifi_avg_hor", "Scifi Ver vs Hor;Vertical;Horizontal", 0, 1600, 0, 1600),
        ("DS_avg_ver", "DS_avg_hor", "DS Ver vs Hor;Vertical;Horizontal", 60, 120, 0, 60),
    ]

    for x_col, y_col, title, x_min, x_max, y_min, y_max in plots:
        hist = df.Histo2D(
            (f"h_{x_col}_{y_col}", title, 100, x_min, x_max, 100, y_min, y_max),
            x_col, y_col
        )
        df_filtered = df.Filter(f"{x_col} > -100 && {y_col} > -100")
        x_min_val = df_filtered.Min(x_col).GetValue()
        x_max_val = df_filtered.Max(x_col).GetValue()
        y_min_val = df_filtered.Min(y_col).GetValue()
        y_max_val = df_filtered.Max(y_col).GetValue()

        # draw fiducial box for pos plot with scifi_hor_pos, scifi_ver_pos, DS_hor_pos, DS_ver_pos, and ch/bar box with scifi_hor_ch, scifi_ver_ch, DS_hor_bar, DS_ver_bar

        #x_axis = hist.GetXaxis()
        #x_axis.SetLimits(x_max, x_min)
        #hist.GetXaxis().SetLimits(x_max, x_min)
        #hist.GetXaxis().SetRangeUser(x_max, x_min)
        hist.Draw("COLZ")
        #hist.GetXaxis().SetRangeUser(x_max, x_min)
        

        latex.DrawLatex(0.12, 0.85, f"{x_col}: min = {x_min_val:.2f}, max = {x_max_val:.2f}")
        latex.DrawLatex(0.12, 0.80, f"{y_col}: min = {y_min_val:.2f}, max = {y_max_val:.2f}")
        canvas.Print(output_pdf)

    canvas.Print(output_pdf + "]")

def plot_avg_pos(rdf):
    pred_kaon = rdf.Filter(f"PredClass == 4")
    plot_avg(pred_kaon, f"pred_kaon_avg_pos.pdf")

    pred_neutron = rdf.Filter(f"PredClass == 5")
    plot_avg(pred_neutron, f"pred_neutron_avg_pos.pdf")

    pred_muon = rdf.Filter(f"PredClass == 6")
    plot_avg(pred_muon, f"pred_muon_avg_pos.pdf")
    

def plot_hist_each_plane(rdf):
    # ── 1. Per-plane binning ─────────────────────────────────────────────
    plane_binning = {
        'scifi1': (100, 0, 200),
        'scifi2': (100, 0, 200),
        'scifi3': (100, 0, 200),
        'scifi4': (100, 0, 200),
        'scifi5': (100, 0, 200),
        # 'us1':    (13, 0, 13),
        # 'us2':    (13, 0, 13),
        # 'us3':    (13, 0, 13),
        # 'us4':    (13, 0, 13),
        # 'us5':    (13, 0, 13),
        # 'ds1':   (40, 0, 40),
        # 'ds2':   (40, 0, 40),
        # 'ds3':   (40, 0, 40),
        # 'ds4':   (40, 0, 40),
    }

    # ── 2. Class labels and colours ──────────────────────────────────────
    class_labels = {4: "pred_kaon", 5: "pred_neutron"}

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
    for plane, (n_bins, x_min, x_max) in tqdm(plane_binning.items(), desc="Processing planes"):
        canvas  = ROOT.TCanvas("c", f"Number of hits in {plane}", 800, 600)
        legend  = ROOT.TLegend(0.70, 0.60, 0.90, 0.90)
        out_pdf = f"plot/control_region_n_hits_{plane}.pdf"

        any_drawn = False
        hist_proxies = []

        for class_id, label in tqdm(class_labels.items(), desc="Processing class labels"):
            rdf_f = rdf.Filter(f"pred_class_first == {class_id} && scifi_gt_100")
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


def plot_rdf(rdf):
    #plot_hits(rdf, 'pred_bkg_gt2')
    #plot_avg_pos(rdf)
    plot_hist_each_plane(rdf)
    
    
def main():
    metadata_data_df = read_metadata()
    print(metadata_data_df)
    feature_chain = ROOT.TChain("snddata")
    prediction_chain = ROOT.TChain("snddata")

    count = 0
    for index, row in metadata_data_df.iterrows():
        
        feature_path = row['feature_path']
        pred_path = row['eval_baseline_muon_output_path']
        #pred_path = row[f'model_baseline_muon_output_path']
        print(pred_path, feature_path)

        if not(os.path.isfile(pred_path)) or not((os.path.isfile(feature_path))):
            continue
        feature_chain.Add(feature_path)
        prediction_chain.Add(pred_path)

    
        if count>8:
            break
        count+=1

    print(f"{count} files read...")
    feature_chain.AddFriend(prediction_chain, 'predTree')
    rdf = ROOT.RDataFrame(feature_chain)

    #rdf = pdg_2_particle(rdf)

    columns = rdf.GetColumnNames()
    print("Columns in RDataFrame:", [str(c) for c in columns])

    #pred_bkg = rdf.Filter("(pred_class_first == 4 || pred_class_first == 5 || pred_class_first == 6) && scifi_gt_100" )

    plot_rdf(rdf)

    # pred_muon
    # pred_kaon
    # pred_neutron

if __name__ == "__main__":
    main()