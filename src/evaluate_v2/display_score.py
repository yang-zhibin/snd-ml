import ROOT

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

if __name__ == "__main__":
    selection_type = 'neutrino'
    input_file = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/selected_data/{selection_type}_selection.root"
    tree_name = "tree"

    df = ROOT.RDataFrame(tree_name, input_file)
    # Assuming the tree is named "tree"
    ve = df.Filter('ParticleType == "ve"').Histo1D(("ve", "Ve", 50, 0.994, 1), "Prediction_0")
    vm = df.Filter('ParticleType == "kaon"|| ParticleType == "neutron"').Histo1D(("vm", "Neutral Hardron", 50, 0.994, 1), "Prediction_0")
    vt = df.Filter('ParticleType == "muon"').Histo1D(("vt", "Muon", 50, 0.994, 1), "Prediction_0")
    # Create a canvas to draw the histograms
    canvas = ROOT.TCanvas("canvas", "Prediction Score Distribution", 800, 600)

    ve.SetStats(0)
    
    # Draw the histogram for label 0
    ve.SetLineColor(ROOT.kRed)
    ve.SetFillColorAlpha(ROOT.kRed, 0.5)
    ve.Draw("hist")

    # Draw the histogram for label 1 on the same canvas
    vm.SetLineColor(ROOT.kBlue)
    vm.SetFillColorAlpha(ROOT.kBlue, 0.5)
    vm.Draw("hist same")

    vt.SetLineColor(ROOT.kGreen)
    vt.SetFillColorAlpha(ROOT.kGreen, 0.5)
    vt.Draw("hist same")

    canvas.SetLeftMargin(0.15)
    # Add a legend
    legend = canvas.BuildLegend(0.4, 0.8, 0.6, 0.6)
    #legend = canvas.BuildLegend()
    legend.SetHeader("Labels", "C")  # Optional: Set a header for the legend
    legend.SetBorderSize(0)
    y_axis = ve.GetYaxis()
    y_axis.SetRangeUser(0, 130)

    # Add titles and labels
    ve.SetTitle("Prediction Score Distribution by Label")
    ve.GetXaxis().SetTitle("Prediction Score")
    ve.GetYaxis().SetTitle("Count")

    # Update the canvas to show the plot
    canvas.Update()

    # Optionally, save the canvas as an image
    canvas.SaveAs("/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot_v2/prediction_score_distribution.pdf")

