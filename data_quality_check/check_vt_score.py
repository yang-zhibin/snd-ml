import ROOT
import pandas as pd
import os

particle_2_class = {
    've': 0,
    'vm': 1,
    'vt': 2,
    'NC': 3,
    'kaon': 4,
    'neutron': 5,
    'muon': 6,
}


def predict_class_with_score(df):
    print("predicting results")
    df = df.Define("ParticleType", """
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

    df = df.Define("PredClass", argmax_expr)

    return df

def read_exist_output(dir_data, metadata_data_df):
    def file_exists(row):
        file_path = row['model_baseline_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def plot_vt_score_by_ture_particle(df):
    particle_types = ["ve", "vm", "vt", "NC"]  # add or modify as needed

    # Create a canvas and legend
    canvas = ROOT.TCanvas("c", "Histograms", 1600, 1200)
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)

    # List to hold histograms to keep them in scope
    histograms = []

    # Loop over each particle type
    for i, ptype in enumerate(particle_types):
        # Filter dataframe
        df_filtered = df.Filter(f'ParticleType == "{ptype}"')

        # Create histogram
        model = ROOT.RDF.TH1DModel(f"hist_{ptype}", f"vt predict score for {ptype};vt_predict_score;PDF", 50, 0.99, 1)
        column_name = "Prediction_2"
        hist = df_filtered.Histo1D(model, column_name)

        if hist.Integral() != 0:
            hist.Scale(1.0 / hist.Integral())
        
        # Set line color for visibility
        hist.SetLineColor(i + 1)
        
        # Draw histograms (first with "", rest with "same")
        draw_option = "" if i == 0 else "same"
        hist.Draw(draw_option)

        # Add to legend and list
        legend.AddEntry(hist.GetPtr(), ptype)
        histograms.append(hist)

    legend.Draw()
    canvas.SaveAs("plot/vt_pred_score_by_particle_0.99.pdf")

def plot_vt_score_by_pred_particle(df):
    canvas = ROOT.TCanvas("c", "Histograms", 1600, 1200)
    model = ROOT.RDF.TH1DModel("h_predclass", "Prediction Class;Class;Entries", 7, 0, 7)
    column_name = "PredClass"
    hist = df.Histo1D(model, column_name)

    hist.Draw("HIST")
    canvas.SaveAs("plot/PredClass_hist.png")

    

def plot_high_vt_score(df):
    df_filtered = df.Filter(f'Prediction_2 > 0.99')

    canvas = ROOT.TCanvas("c", "Histograms", 1600, 1200)
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)

    # List to hold histograms to keep them in scope
    histograms = []

    for particle, class_id in particle_2_class.items():
        # Create histogram
        model = ROOT.RDF.TH1DModel(f"pred_{particle}", f"pred {particle};pred score;PDF", 100, 0.9, 1)
        column_name = f"Prediction_{class_id}"
        hist = df_filtered.Histo1D(model, column_name)

        if hist.Integral() != 0:
            hist.Scale(1.0 / hist.Integral())
        
        # Set line color for visibility
        hist.SetLineColor(class_id + 1)
        
        # Draw histograms (first with "", rest with "same")
        draw_option = "" if class_id == 0 else "same"
        hist.Draw(draw_option)

        # Add to legend and list
        legend.AddEntry(hist.GetPtr(), particle)
        histograms.append(hist)

    legend.Draw()
    canvas.SaveAs("plot/high_vt_pred_score_by_particle.pdf")

def plot_2d_vt_score(df):
    ROOT.gStyle.SetOptStat(1)
    canvas = ROOT.TCanvas()
    canvas.Divide(1, 1)

    # Open the PDF file
    canvas.Print("plot/2d_vt_score_2.pdf[")

    for name, idx in particle_2_class.items():
        df_filtered = df.Filter(f'ParticleType == "vt"')


        x_col = "Prediction_2"
        y_col = f"Prediction_{idx}"

        hist2d = df_filtered.Histo2D(
            (f"h2_{name}", f"{y_col} vs {x_col};{x_col};{y_col}", 100, 0, 1, 100, 0, 1),
            x_col,
            y_col
        )
        canvas.SetLogz()
        hist2d.Draw("COLZ")
        canvas.Update()

        # Move stats box to top-left
        stats = hist2d.GetListOfFunctions().FindObject("stats")
        if stats:
            stats.SetX1NDC(0.1)
            stats.SetX2NDC(0.3)
            stats.SetY1NDC(0.75)
            stats.SetY2NDC(0.9)

        canvas.Modified()
        canvas.Update()
        canvas.Print("plot/2d_vt_score_2.pdf")  # Save this page

    # Close the PDF file
    canvas.Print("plot/2d_vt_score_2.pdf]")

def plot_max_pred_score(df):
    df = df.Filter(f'ParticleType == "vt"')
    df = df.Define("max_pred_score", 
    "std::max({Prediction_0, Prediction_1, Prediction_2, Prediction_3, Prediction_4, Prediction_5, Prediction_6})")

    hist2d = df.Histo2D(
        ("h2", "Max Prediction Score vs Prediction_2;Prediction_2;max_pred_score", 
        100, 0, 1,   # X-axis: max_pred_score bins
        100, 0, 1),  # Y-axis: Prediction_2 bins
        "Prediction_2", "max_pred_score"
    )

    c = ROOT.TCanvas()
    c.SetLogz()
    hist2d.Draw("COLZ")  # COLZ for color map
    c.Update()
    stats = hist2d.GetListOfFunctions().FindObject("stats")
    if stats:
        stats.SetX1NDC(0.7)
        stats.SetX2NDC(0.9)
        stats.SetY1NDC(0.1)
        stats.SetY2NDC(0.3)
    c.Modified()
    c.Update()
    c.SaveAs("plot/max_vs_pred2_vt_true2.pdf")
def main():
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'  
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path)
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'
    metadata_mc_neutrino_df = read_exist_output(dir_MC, metadata_mc_neutrino_df)    

    prediction_chain = ROOT.TChain("snddata")

    model_name = 'baseline'
    for index, row in metadata_mc_neutrino_df.iterrows():
        pred_path = row[f'model_{model_name}_output_path']
        prediction_chain.Add(pred_path)

    rdf = ROOT.RDataFrame(prediction_chain)
    rdf = predict_class_with_score(rdf)

    #plot_vt_score(rdf)
    #plot_vt_score_by_pred_particle(rdf)
    #plot_high_vt_score(rdf)
    plot_2d_vt_score(rdf)
    plot_max_pred_score(rdf)




if __name__ == "__main__":
    main()