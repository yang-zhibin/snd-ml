import ROOT
import os

ROOT.EnableImplicitMT()

def print_tree_structure(file_path, tree_name):
    # Open the ROOT file
    root_file = ROOT.TFile(file_path, "READ")
    
    # Attempt to get the tree from the file
    tree = root_file.Get(tree_name)
    
    if tree:
        # Print the structure of the tree
        tree.Print()
    else:
        # Inform the user if the tree is not found
        print(f"Tree '{tree_name}' not found in the file '{file_path}'.")
    
    # Close the file to free resources
    root_file.Close()

def create_histograms_by_pdgcode(file_path, out_dir, image_prefix):
    particle_mapping = {
        12:  've',
        -12: 've',
        14:  'vm',
        -14: 'vm',
        16:  'vt',
        -16: 'vt',
        2112: 'neutron',
        2212: 'proton',
        130: 'kaon',
        310: 'kaon0',
        321: 'kaon+',
        -321: 'kaon-',
        112:  'NC',
        -112: 'NC',
        114:  'NC',
        -114: 'NC',
        116:  'NC',
        -116: 'NC'
    }
    # Run in batch mode to avoid issues with canvas creation
    ROOT.gROOT.SetBatch(True)
    # Disable the statistics box
    ROOT.gStyle.SetOptStat(0)


    # Open the ROOT file
    root_file = ROOT.TFile(file_path)

    # Get the tree from the ROOT file
    tree = root_file.Get("tree")

    # Create a dictionary to store histograms for different particles
    histograms = {}

    # Get the range for Prediction
    prediction_min = 0
    prediction_max = 1

    #print(prediction_min, prediction_max)
    # Loop over the entries in the tree
    for entry in tree:

        pdg_code = entry.PdgCode
        #print(pdg_code)
        prediction = entry.Prediction

        # Map pdg_code to particle name
        particle = particle_mapping.get(pdg_code, None)
        if particle is None:
            print(pdg_code, 'not in particle_mapping')
            continue

        # Check if a histogram for this particle exists, if not create one
        if particle not in histograms:
            histograms[particle] = ROOT.TH1F(f"hist_{particle}", f"Prediction for {particle}", 100, prediction_min, prediction_max)

        # Fill the histogram
        histograms[particle].Fill(prediction)
    
    print("finish fill")
    #print(histograms)
    # Create a canvas
    canvas = ROOT.TCanvas("canvas", "Prediction Histograms by Particle", 800, 600)
    print("canvas created")

    # Draw the histograms on the same canvas
    first_histogram = True
    legend = ROOT.TLegend(0.75, 0.75, 0.9, 0.9)

    #print('dubug')
    color_index = 2  # Start with color index 2 to avoid black color
    for particle, histogram in histograms.items():
        print(particle,histogram )
        histogram.SetLineColor(color_index)
        if first_histogram:
            histogram.Draw()
            first_histogram = False
        else:
            histogram.Draw("SAME")
        legend.AddEntry(histogram, particle, "l")
        color_index += 1

    legend.Draw()

    # Save the canvas as an image
    image_name = "{}_prediction_histograms_by_pdgcode.png".format(image_prefix)
    image_file_path = os.path.join(out_dir,image_name)
    canvas.SaveAs(image_file_path)

    # Close the ROOT file
    root_file.Close()
def create_signal_background_histogram(file_path, out_dir, image_prefix):
    signal_pdg_codes = {14, -14}  # PDG codes for VM
    # Run in batch mode to avoid issues with canvas creation
    ROOT.gROOT.SetBatch(True)
    # Disable the statistics box
    ROOT.gStyle.SetOptStat(0)

    # Open the ROOT file
    root_file = ROOT.TFile(file_path)

    # Get the tree from the ROOT file
    tree = root_file.Get("tree")

    # Get the range for Prediction
    prediction_min = 0
    prediction_max = 1

    # Create histograms for signal and background
    signal_histogram = ROOT.TH1F("signal_hist", "Prediction for VM (Signal)", 100, prediction_min, prediction_max)
    background_histogram = ROOT.TH1F("background_hist", "Prediction for Background", 100, prediction_min, prediction_max)

    # Loop over the entries in the tree
    for entry in tree:
        pdg_code = entry.PdgCode
        prediction = entry.Prediction

        # Fill the appropriate histogram
        if pdg_code in signal_pdg_codes:
            signal_histogram.Fill(prediction)
        else:
            background_histogram.Fill(prediction)
    
    # Find the maximum bin content
    max_bin_content = max(signal_histogram.GetMaximum(), background_histogram.GetMaximum())

    # Set the y-axis maximum value to accommodate the highest bin content
    signal_histogram.SetMaximum(500)  # Add 10% padding
    background_histogram.SetMaximum(500)

    # Create a canvas
    canvas = ROOT.TCanvas("canvas", "Signal vs Background Prediction Histograms", 800, 600)

    # Draw the histograms on the same canvas
    signal_histogram.SetLineColor(ROOT.kRed)
    background_histogram.SetLineColor(ROOT.kBlue)

    signal_histogram.Draw()
    background_histogram.Draw("SAME")

    # Create a legend
    legend = ROOT.TLegend(0.75, 0.75, 0.9, 0.9)
    legend.AddEntry(signal_histogram, "VM (Signal)", "l")
    legend.AddEntry(background_histogram, "Background", "l")
    legend.Draw()

    # Save the canvas as an image
    image_name = "{}_signal_vs_background_histograms.png".format(image_prefix)
    image_file_path = os.path.join(out_dir,image_name)
    canvas.SaveAs(image_file_path)

    # Close the ROOT file
    root_file.Close()

def main():
    # Example usage
    #file_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/log/test_local/vm_predictions.root"
    file_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/outputs/root_output/Neutrinos_id0_predictions.root'
    #print_tree_structure(file_path, "tree")

    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/outputs/plots/'
    image_prefix = 'Neutrinos'
    create_histograms_by_pdgcode(file_path, out_dir, image_prefix)
    #create_signal_background_histogram(file_path,out_dir, image_prefix)




if __name__ == "__main__":

    main()