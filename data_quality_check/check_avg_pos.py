import ROOT
import pandas as pd
import os

def plot_avg(df, output_pdf):
    os.makedirs("plot", exist_ok=True)

    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.Print(output_pdf + "[")

    # Format: (x_col, y_col, title, x_min, x_max, y_min, y_max)
    plots = [
        ("scifi_avg_x_pos", "scifi_avg_y_pos", "Scifi X vs Y;X Position;Y Position", -70, 10, 0, 70),
        ("DS_avg_x_pos", "DS_avg_y_pos", "DS X vs Y;X Position;Y Position", -70, 10, 0, 70),

        ("scifi_avg_ver", "scifi_avg_hor", "Scifi Ver vs Hor;Vertical;Horizontal", 0, 1600, 0, 1600),
        ("DS_avg_ver", "DS_avg_hor", "DS Ver vs Hor;Vertical;Horizontal", 60, 120, 0, 60),
    ]

    for x_col, y_col, title, x_min, x_max, y_min, y_max in plots:
        hist = df.Histo2D(
            (f"h_{x_col}_{y_col}", title, 100, x_min, x_max, 100, y_min, y_max),
            x_col, y_col
        )

        hist.Draw("COLZ")
        canvas.Print(output_pdf)

    canvas.Print(output_pdf + "]")


def read_exist_output(dir_data, metadata_data_df, file_column_name):
    def file_exists(row):
        file_path = row[file_column_name]
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def check_fiducial_pos():
    scifi_vert = [200, 1200] # x position
    scifi_hor  = [300, 1336] # y position
    DS_vert = [10, 50] # x position
    DS_hor = [70, 105] # y position

    geo_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root" 



def main():
    #metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'  
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_metadata.csv'  
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path,nrows=100)
    #dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/real_data/2024'

    file_column_name = 'feature_path'
    metadata_mc_neutrino_df = read_exist_output(dir_MC, metadata_mc_neutrino_df, file_column_name)    

    t_chain = ROOT.TChain("snddata")

    for index, row in metadata_mc_neutrino_df.iterrows():
        file_path = row[file_column_name]
        print('reading:',file_path)
        t_chain.Add(file_path)

    rdf = ROOT.RDataFrame(t_chain)

    output_pdf = "plot/avg_pos_2024.pdf"
    plot_avg(rdf,output_pdf)

if __name__ == "__main__":
    main()