import ROOT
import argparse

ROOT.gROOT.ProcessLine(".L EventClasses.h+")


def main(args):
    pred_file = args.pred_file
    raw_file = args.raw_file
    output_file = '/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/test_data/with_score.root'

    # Open the source and target files
    source_file = ROOT.TFile.Open(pred_file, "READ")
    target_file = ROOT.TFile.Open(raw_file, "READ")
    new_file = ROOT.TFile(output_file, "UPDATE")

    # Get the trees from both files
    source_tree = source_file.Get("tree")
    target_tree = target_file.Get("cbmsim")

    # Ensure the trees exist
    if not source_tree:
        raise ValueError("Source tree 'tree' not found in source file!")
    if not target_tree:
        raise ValueError("Target tree 'cbmsim' not found in target file!")
    
    # Disable all branches initially
    #source_tree.SetBranchStatus("*", 0)

    # Enable only the branches you need
    #branches_to_keep = ["Prediction_0", "Prediction_1", "EventId"]
    #for branch in branches_to_keep:
    #    source_tree.SetBranchStatus(branch, 1)

    #new_tree = source_tree.CloneTree(0)
    # Add the source tree as a friend to the target tree
    

    new_tree = target_tree.CloneTree()
    new_tree.AddFriend(source_tree, args.model)

    # Write the updated tree to the new file
    new_tree.Write()

    # Create an RDataFrame for the target tree
    # target_df = ROOT.RDataFrame(target_tree)
    # print(target_df.GetColumnNames())
    # df_filter = target_df.Filter(f"(Id.eventId != {args.model}.EventId) || (Id.runId != {args.model}.RunId)")
    # print(df_filter.Count().GetValue())



    # Limit to the first 1000 entries
    #limited_df = target_df.Range(1000)



    # Snapshot the updated tree to a new file
    #limited_df.Snapshot("cbmsim", target_file)

    

    # Close the files
    new_file.Close()
    source_file.Close()
    target_file.Close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline_muon')
    parser.add_argument("-p", "--pred_file", dest="pred_file", default = '/eos/user/z/zhibin/sndData/converted/pt/test/Neutrinos/output/baseline_muon/test_neutrino_partition_1_output.root')
    parser.add_argument("-r", "--raw_file", dest="raw_file", default = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/1/neutrinos_converted_sndLHC.Genie-TGeant4_20240126_digCPP.root')
    args = parser.parse_args()
    main(args)