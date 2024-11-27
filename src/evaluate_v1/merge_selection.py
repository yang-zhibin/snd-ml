import pandas as pd
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
from collections import Counter
def check_duplicate(data):
    unique_pairs = set(data)

    # Checking if there are duplicates
    if len(unique_pairs) == len(data):
        print("No duplicates found.")
    else:
        print("Duplicates found.")

    # Optional: Identify the duplicates if needed
    if len(unique_pairs) != len(data):
        
        counts = Counter(data)
        duplicates = [item for item, count in counts.items() if count > 1]
        print("Duplicate pairs are:", duplicates)

def check_duplicate(path1, path2, output_file):
    try:
        # Open the first file and extract arrays
        with uproot.open(path1) as f1:
            tree1 = f1["cbmsim"]
            runId1 = tree1["runId"].array(library="np")
            eventId1 = tree1["eventId"].array(library="np")
            x1 = tree1["Label/x"].array(library="np")
            y1 = tree1["Label/y"].array(library="np")
            z1 = tree1["Label/z"].array(library="np")
            px1 = tree1["Label/px"].array(library="np")
            py1 = tree1["Label/py"].array(library="np")
            pz1 = tree1["Label/pz"].array(library="np")

        # Open the second file and extract arrays
        with uproot.open(path2) as f2:
            tree2 = f2["cbmsim"]
            runId2 = tree2["runId"].array(library="np")
            eventId2 = tree2["eventId"].array(library="np")
            x2 = tree2["Label/x"].array(library="np")
            y2 = tree2["Label/y"].array(library="np")
            z2 = tree2["Label/z"].array(library="np")
            px2 = tree2["Label/px"].array(library="np")
            py2 = tree2["Label/py"].array(library="np")
            pz2 = tree2["Label/pz"].array(library="np")

        # Create a mapping from (runId, eventId) to data arrays
        data1 = {(runId, eventId): np.array([x, y, z, px, py, pz])
                 for runId, eventId, x, y, z, px, py, pz in zip(runId1, eventId1, x1, y1, z1, px1, py1, pz1)}
        data2 = {(runId, eventId): np.array([x, y, z, px, py, pz])
                 for runId, eventId, x, y, z, px, py, pz in zip(runId2, eventId2, x2, y2, z2, px2, py2, pz2)}

        # Compare the data for the same (runId, eventId)
        differences = []
        for key in data1:
            if key in data2:
                if not np.array_equal(data1[key], data2[key]):
                    difference = (f"Duplicate runId {key[0]}, eventId {key[1]}:\n"
                                  f"File 1: {', '.join(f'{x:.4f}' for x in data1[key])}\n"
                                  f"File 2: {', '.join(f'{x:.4f}' for x in data2[key])}\n")
                    differences.append(difference)

        # Write differences to a file if any
        if differences:
            with open(output_file, 'w') as f:
                f.writelines(differences)
                print(f"Differences written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_branches(tree, branches):
    """Extract specified branches from a given tree and return as a dictionary."""
    data = {branch: tree[branch].array(library="np") for branch in branches}
    return data

def main():
    #list_path = '/eos/user/z/zhibin/sndData/converted/prediction_files.csv'
    list_path = '/eos/user/z/zhibin/sndData/converted/real_muon/real_muon_evt_list.csv'
    list_df = pd.read_csv(list_path)
    neutrino_path = [path for path in list_df['file']]
    print(neutrino_path)
    model = 'baseline_muon'
    output_dir = '/eos/user/z/zhibin/sndData/converted/pt/output/baseline_muon/'
    output_file_list = [f'{output_dir}/test_real_muon_2_output.root']
    
    # Define branches to process
    branches = [
        "runId", "eventId", "Label/x", "Label/y", 
        "VmSeclection/stage1", "VmSeclection/stage2", "Label/pdgCode",
        "Label/scifi_avg_ver", "Label/scifi_avg_hor", 
        "Label/DS_avg_ver", "Label/DS_avg_hor"
    ]
    simple_branch_names = [branch.split('/')[-1] for branch in branches]
    #print(simple_branch_names)

    # Load data from the output files and create a set of (runId, eventId) tuples
    combineId_set = set()
    for file in output_file_list:
        with uproot.open(file) as f:
            tree = f["tree"]
            data = extract_branches(tree, ["RunId", "EventId"])
            combineId_set.update(zip(data["RunId"], data["EventId"]))

    print(f"Total unique (runId, eventId) pairs: {len(combineId_set)}")

    # Initialize dictionaries to store filtered data
    filtered_data = {branch: [] for branch in simple_branch_names}

    for file in tqdm(neutrino_path):
        with uproot.open(file) as f:
            tree = f["cbmsim"]
            data = extract_branches(tree, branches)


            # Convert runId and eventId to tuples for fast filtering
            combineIds = list(zip(data["runId"], data["eventId"]))
            mask = np.array([combineId in combineId_set for combineId in combineIds], dtype=bool)  # Boolean array for filtering

            # Append the filtered data for each branch
            for branch, simple_name in zip(branches, simple_branch_names):
                #print(branch,simple_name )
                filtered_data[simple_name].extend(data[branch][mask])

    # Convert lists to arrays and store in a ROOT file
    with uproot.recreate(f"merge_output/{model}_merged_file.root") as f:
        tree_data = {simple_name: np.array(filtered_data[simple_name]) for simple_name in simple_branch_names}
        f["merged_tree"] = tree_data

if __name__ == "__main__":
    main()
    #path1 = '/eos/user/z/zhibin/sndData/converted/Neutrinos/221/sndLHC.Genie-TGeant4_20240126_digCPP_converted_00221.root'
    #path2 = '/eos/user/z/zhibin/sndData/converted/Neutrinos/222/sndLHC.Genie-TGeant4_20240126_digCPP_converted_00222.root'
    #outfile ='duplicate_runId.txt'
    #check_duplicate(path1,path2,outfile)