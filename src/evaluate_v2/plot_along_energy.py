import ROOT
import pandas as pd
import argparse
import os
import numpy as np

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

def plot_neutron(out_dir, bkg_energy_df, signal):
    energy = bkg_energy_df['energy'].values
    neutron_yield = bkg_energy_df['neutron_yield'].values
    neutron_error_low = bkg_energy_df['neutron_error_low'].values
    neutron_error_upp = bkg_energy_df['neutron_error_upp'].values
    neutron_pass = bkg_energy_df['neutron_pass'].values

    # Number of points
    n_points = len(energy)
    # Create the TGraphErrors object
    energy = np.array(energy, dtype=np.float64)
    neutron_yield = np.array(neutron_yield, dtype=np.float64)
    neutron_error_low = np.array(neutron_error_low, dtype=np.float64)
    neutron_error_upp = np.array(neutron_error_upp, dtype=np.float64)
    # Create the TGraphAsymmErrors object
    graph = ROOT.TGraphAsymmErrors(n_points, energy, neutron_yield, 
                                   np.zeros(n_points), np.zeros(n_points), 
                                   neutron_yield-neutron_error_low, neutron_error_upp-neutron_yield)
    graph.SetTitle("Neutron Yield vs Energy")
    graph.GetXaxis().SetTitle("Energy (GeV)")
    graph.GetYaxis().SetTitle("Neutron Yield")

    # Customize marker style
    graph.SetMarkerColor(4)
    graph.SetMarkerStyle(21) 
    graph.SetMarkerSize(1)

    # Create a canvas to draw the graph
    canvas = ROOT.TCanvas("canvas", "Neutron Yield Plot", 800, 600)

    # Draw the graph with error bars on the canvas
    graph.Draw("APL")
    y_min = 1e-6  # Example minimum value; adjust as needed
    y_max = 1 # Example maximum value; adjust as needed
    graph.GetYaxis().SetRangeUser(y_min, y_max)
    canvas.SetLogy()
    # Save the plot to a file
    outpath = f'{out_dir}/neutron_yield_vs_energy.pdf'
    canvas.SaveAs(outpath)
def plot_kaon(out_dir, bkg_energy_df):
    """
    Plots kaon yield vs. energy using ROOT and saves the plot as a PDF.

    Parameters:
    out_dir (str): Directory where the plot will be saved.
    bkg_energy_df (pandas.DataFrame): DataFrame containing columns 'energy', 'kaon_yield',
                                      'kaon_error_low', 'kaon_error_upp', and 'kaon_pass'.
    """
    # Extract data
    energy = bkg_energy_df['energy'].values
    kaon_yield = bkg_energy_df['kaon_yield'].values
    kaon_error_low = bkg_energy_df['kaon_error_low'].values
    kaon_error_upp = bkg_energy_df['kaon_error_upp'].values

    # Convert to numpy arrays
    energy = np.array(energy, dtype=np.float64)
    kaon_yield = np.array(kaon_yield, dtype=np.float64)
    kaon_error_low = np.array(kaon_error_low, dtype=np.float64)
    kaon_error_upp = np.array(kaon_error_upp, dtype=np.float64)

    # Number of points
    n_points = len(energy)

    # Create the TGraphAsymmErrors object
    graph = ROOT.TGraphAsymmErrors(n_points, energy, kaon_yield, 
                                   np.zeros(n_points), np.zeros(n_points), 
                                   kaon_yield-kaon_error_low, kaon_error_upp-kaon_yield)
    graph.SetTitle("Kaon Yield vs Energy")
    graph.GetXaxis().SetTitle("Energy (GeV)")
    graph.GetYaxis().SetTitle("Kaon Yield")

    # Customize marker style
    graph.SetMarkerColor(2)  # Red color
    graph.SetMarkerStyle(22) # Square marker
    graph.SetMarkerSize(1)

    # Create a canvas to draw the graph
    canvas = ROOT.TCanvas("canvas", "Kaon Yield Plot", 800, 600)

    # Draw the graph with error bars on the canvas
    graph.Draw("APL")

    # Set y-axis range and log scale
    y_min = 1e-6  # Example minimum value; adjust as needed
    y_max = 1     # Example maximum value; adjust as needed
    graph.GetYaxis().SetRangeUser(y_min, y_max)
    canvas.SetLogy()

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save the plot to a file
    outpath = os.path.join(out_dir, 'kaon_yield_vs_energy.pdf')
    canvas.SaveAs(outpath)

    print(f"Kaon plot saved to {outpath}")
def plot(signal):
    df_list = []
    directory = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_bkg_energy/'
    print('signal:', signal)
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(signal) and filename.endswith('.csv'):
            # Read each CSV file into a dataframe
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            #df = df.sort_values(by=f'{signal}_eff')
            model  = filename.split('_')[1]
            df['model'] = model
            # Append the dataframe to the list
            df_list.append(df)

    # Concatenate all dataframes into one
    bkg_energy_df = pd.concat(df_list, ignore_index=True)

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot_bkg/'
    #plot_eff_yield(out_dir, eff_yield_df, signal)
    plot_neutron(out_dir, bkg_energy_df, signal)
    plot_kaon(out_dir, bkg_energy_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--signal", dest="signal", default='ve')
    args = parser.parse_args()
    plot(args.signal)