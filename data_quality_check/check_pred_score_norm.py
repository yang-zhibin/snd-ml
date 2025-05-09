import ROOT

def main():
    # Open ROOT files
    file_sigmoid = ROOT.TFile("/afs/cern.ch/user/z/zhibin/work/snd-ml/data_quality_check/test_data/model_output_sigmoid.root")
    file_softmax = ROOT.TFile("/afs/cern.ch/user/z/zhibin/work/snd-ml/data_quality_check/test_data/model_output_softmax.root")

    # Get trees
    tree_sigmoid = file_sigmoid.Get("snddata")  # Replace "tree" with actual tree name
    tree_softmax = file_softmax.Get("snddata")  # Same here

    # Check both trees have same number of entries
    n_entries = min(tree_sigmoid.GetEntries(), tree_softmax.GetEntries())
    print(f"Processing {n_entries} events")

    # Loop over both trees simultaneously
    for i in range(n_entries):
        tree_sigmoid.GetEntry(i)
        tree_softmax.GetEntry(i)

        # Example of accessing branches – adjust names as needed
        sigmoid_preds = [getattr(tree_sigmoid, f"Prediction_{j}") for j in range(7)]
        softmax_preds = [getattr(tree_softmax, f"Prediction_{j}") for j in range(7)]

        sigmoid_str = [f"{x:.8f}" for x in sigmoid_preds]
        softmax_str = [f"{x:.8f}" for x in softmax_preds]

        print(f"Event {i}:")
        print(f"  Sigmoid: {sigmoid_str}")
        print(f"  Softmax: {softmax_str}")
        print()
        if i>5:
            break

if __name__ == "__main__":
    main()