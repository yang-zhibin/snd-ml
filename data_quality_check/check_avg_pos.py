import ROOT
import pandas as pd
import os
import numpy as np

def cal_scan_fiducial_area():

    scifi_tl_xs = list(range(-46, -40, 1)) 
    scifi_tl_ys = list(range(53, 47, -1))  
    DS_tl_xs = list(range(-61, -45, 3))    
    DS_tl_ys = list(range(67, 56, -2))     

    scifi_br_xs = list(range(-7, -16, -1)) 
    scifi_br_ys = list(range(14, 23))      
    DS_br_xs = list(np.arange(1, -9.1, -1.25).tolist())      
    DS_br_ys = list(np.arange(8, 18.1, 1.25).tolist())      
    
    fiducial_tl_exprs = []
    for i in range(len(scifi_tl_xs)):
        scifi_tl_x = scifi_tl_xs[i]
        scifi_tl_y = scifi_tl_ys[i]
        DS_tl_x = DS_tl_xs[i]
        DS_tl_y = DS_tl_ys[i]
        fiducial_tl_expr = f'scifi_avg_x_pos >={scifi_tl_x} && scifi_avg_y_pos <= {scifi_tl_y} && DS_avg_x_pos >={DS_tl_x} && DS_avg_y_pos <= {DS_tl_y} '
        fiducial_tl_exprs.append(fiducial_tl_expr)
    
    fiducial_br_exprs = []
    for i in range(len(scifi_br_xs)):
        scifi_br_x = scifi_br_xs[i]
        scifi_br_y = scifi_br_ys[i]
        DS_br_x = DS_br_xs[i]
        DS_br_y = DS_br_ys[i]
        fiducial_br_expr = f'scifi_avg_x_pos <={scifi_br_x} && scifi_avg_y_pos >= {scifi_br_y} && DS_avg_x_pos <={DS_br_x} && DS_avg_y_pos >= {DS_br_y} '
        fiducial_br_exprs.append(fiducial_br_expr)

    print("Top-left fiducial expressions:")
    for i, expr in enumerate(fiducial_tl_exprs, 1):
        print(f"{i:2d}: {expr}")

    print("\nBottom-right fiducial expressions:")
    for i, expr in enumerate(fiducial_br_exprs, 1):
        print(f"{i:2d}: {expr}")
 
        
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
        

        # Add fiducial box depending on detector
        if "scifi" in x_col and "pos" in x_col:
            # Draw scifi position fiducial box
            box = ROOT.TBox(scifi_ver_pos[0],scifi_hor_pos[0], scifi_ver_pos[1], scifi_hor_pos[1])
            box.SetLineColor(ROOT.kRed)
            box.SetLineWidth(2)
            box.SetFillStyle(0)
            box.Draw()
            #hist.GetXaxis().SetRangeUser(x_max, x_min)

        elif "DS" in x_col and "pos" in x_col:
            # Draw DS position fiducial box
            box = ROOT.TBox(DS_ver_pos[0], DS_hor_pos[0], DS_ver_pos[1], DS_hor_pos[1])
            box.SetLineColor(ROOT.kRed)
            box.SetLineWidth(2)
            box.SetFillStyle(0)
            box.Draw()
            #hist.GetXaxis().SetRangeUser(x_max, x_min)

        elif "scifi" in x_col and "ver" in x_col:
            # Draw SciFi channel cut box
            box = ROOT.TBox(scifi_ver_ch[0], scifi_hor_ch[0], scifi_ver_ch[1], scifi_hor_ch[1])
            box.SetLineColor(ROOT.kRed)
            box.SetLineWidth(2)
            box.SetFillStyle(0)
            box.Draw()
            #hist.GetXaxis().SetRangeUser(x_max, x_min)

        elif "DS" in x_col and "ver" in x_col:
            # Draw DS bar range box
            box = ROOT.TBox(DS_ver_bar[0], DS_hor_bar[0], DS_ver_bar[1], DS_hor_bar[1])
            box.SetLineColor(ROOT.kRed)
            box.SetLineWidth(2)
            box.SetFillStyle(0)
            box.Draw()

        latex.DrawLatex(0.12, 0.85, f"{x_col}: min = {x_min_val:.2f}, max = {x_max_val:.2f}")
        latex.DrawLatex(0.12, 0.80, f"{y_col}: min = {y_min_val:.2f}, max = {y_max_val:.2f}")
        canvas.Print(output_pdf)

    canvas.Print(output_pdf + "]")


def read_exist_output(dir_data, metadata_data_df, file_column_name):
    def file_exists(row):
        file_path = row[file_column_name]
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

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

def main():
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'  
    #metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_metadata.csv'  
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path,nrows=100)
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'
    #dir_MC = '/eos/experiment/sndlhc/users/zhibin/real_data/2024'

    file_column_name = 'feature_path'
    metadata_mc_neutrino_df = read_exist_output(dir_MC, metadata_mc_neutrino_df, file_column_name)    

    t_chain = ROOT.TChain("snddata")

    for index, row in metadata_mc_neutrino_df.iterrows():
        file_path = row[file_column_name]
        print('reading:',file_path)
        t_chain.Add(file_path)

    rdf = ROOT.RDataFrame(t_chain)

    output_pdf = "plot/avg_pos.pdf"
    plot_avg(rdf,output_pdf)

if __name__ == "__main__":
    #main()
    cal_scan_fiducial_area()