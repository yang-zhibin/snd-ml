import ROOT
from argparse import ArgumentParser

def print_all_tree_structures(filename):
    # Step 1: Open the ROOT file
    file = ROOT.TFile(filename)

    # Step 2: Get list of keys (objects) in the file
    keys = file.GetListOfKeys()

    # Step 3: Iterate over the keys to find trees and print their structures
    found_tree = False
    for key in keys:
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TTree):
            print(f"Tree Name: {obj.GetName()}")
            obj.Print()
            found_tree = True

    # Step 4: Close the file
    file.Close()

    if not found_tree:
        print(f"No trees found in {filename}.")

def save_subset_events(input_filename, output_filename, events_to_save):
    # Step 1: Create RDataFrame from the input TTree
    rdf = ROOT.RDataFrame("cbmsim", input_filename)

    # Step 4: Use the Filter method with a selection range
    sub_rdf = rdf.Range(0, 5)

    # Step 5: Create a snapshot with selected events
    snapshot = sub_rdf.Snapshot("cbmsim", output_filename)


def main(args):
    input_file = ROOT.TFile(args.in_file, "READ")

    # Use RDataFrame to filter the tree
    muons_rdf = ROOT.RDataFrame("cbmsim", input_file)
    total_count = muons_rdf.Count()
    scifi_cut = f"(Digi_ScifiHits.GetEntriesFast() > {args.n_scifiHit}) && Digi_MuFilterHits.GetEntriesFast() > {args.n_scifiHit}"
    filtered_df = muons_rdf.Filter(scifi_cut)
    filtered_count = filtered_df.Count()

    total_events = total_count.GetValue()
    filtered_events = filtered_count.GetValue()
    ratio = filtered_events / total_events

    print(f"Total number of events: {total_events}")
    print(f"Number of events with more than {args.n_scifiHit} ScifiHits: {filtered_events}")
    print(f"Ratio of events with more than {args.n_scifiHit} ScifiHits to total events: {ratio:.4f}")

    # Snapshot the filtered data to a new ROOT file
    filtered_df.Snapshot("cbmsim", args.out_file)




    # Open the output file to copy non-TTree objects, check https://github.com/SND-LHC/sndsw/blob/ca94962ae7a67b30738f6dd6acf466becb0cb434/analysis/neutrinoFilterGoldenSample.cxx#L67
    output_file = ROOT.TFile(args.out_file, "UPDATE")

    # Write the BranchList with kSingleKey option
    branch_list = input_file.Get("BranchList")
    if branch_list:
        branch_list.Write("BranchList", ROOT.TObject.kSingleKey)
    else:
        print("BranchList not found in the original file.")

    # Write the TimeBasedBranchList with kSingleKey option
    time_based_branch_list = input_file.Get("TimeBasedBranchList")
    if time_based_branch_list:
        time_based_branch_list.Write("TimeBasedBranchList", ROOT.TObject.kSingleKey)
    else:
        print("TimeBasedBranchList not found in the original file, create an empty Tlist ")
        empty_list = ROOT.TList()
        empty_list.Write("TimeBasedBranchList", ROOT.TObject.kSingleKey)

    # Write the FileHeader if it exists
    file_header = input_file.Get("FileHeader")
    if file_header:
        file_header.Write()
    else:
        print("FileHeader not found in the original file.")

    # Write the FileHeaderHeader if it exists
    file_header_header = input_file.Get("FileHeaderHeader")
    if file_header_header:
        file_header_header.Write()
    else:
        print("FileHeaderHeader not found in the original file.")

    input_file.Close()
    output_file.Close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--in_file", dest="in_file", help="input file path", required=True)
    parser.add_argument("-o", "--out_file", dest="out_file", help="output file path", required=True)
    parser.add_argument("-n", "--n_scifiHit", dest="n_scifiHit", help="minimum number of scifi hits", type=int, default=0)
    args = parser.parse_args()
    
    main(args)