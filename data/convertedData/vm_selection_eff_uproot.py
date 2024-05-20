import os
import fnmatch
import uproot
import pandas as pd
import numpy as np

def print_tree_structure(file_path, tree_name):
    with uproot.open(file_path) as file:
        try:
            tree = file[tree_name]
            # Print the structure of the tree
            tree.show()
        except KeyError:
            # Inform the user if the tree is not found
            print(f"Tree '{tree_name}' not found in the file '{file_path}'.")

def find_files(root_path, pattern):
    matches = []
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            if filename.endswith('.root'):
                matches.append(os.path.join(root, filename))
    return matches

def cal_eff(root_path):
    print(root_path)
    pattern = '*converted*'
    files_list = find_files(root_path, pattern)
    
    if not files_list:
        print("No files found.")
        return

    print_tree_structure(files_list[0], "cbmsim")

    data_frames = []

    for f in files_list:
        with uproot.open(f) as file:
            tree = file["cbmsim"]
            arrays = tree.arrays(filter_name=["Id/runId", "Id/eventId", "Label/pdgCode", "Label/px", "Label/py", "Label/pz",
                                              "Hits/Hits.*", "ScifiCluster/ScifiCluster.*", "RecoMuon/RecoMuon.*", 
                                              "VmSeclection/stage1", "VmSeclection/stage2"], library="pd")
            data_frames.append(arrays)

    if data_frames:
        # Concatenate all dataframes into a single DataFrame
        data = pd.concat(data_frames, ignore_index=True)
        # Further data manipulation and analysis here

        # Example operation: count the entries
        total_count = len(data)
        print("Number of entries:", total_count)

    # Filter the DataFrame to select events where stage1 and stage2 are True
    stage1 = data[data['stage1'] == True]
    stage2 = data[data['stage2'] == True]
    count1 = len(stage1)
    count2 = len(stage2)

    eff1 = count1 / total_count
    eff2 = count2 / total_count

    print("stage1, count: {}, eff: {:.2e}".format(count1, eff1))
    print("stage2, count: {}, eff: {:.2e}".format(count2, eff2))

    #unique_run_ids = set(data['runId'])
    #print(unique_run_ids)

def main():
    neutrino_path = "/eos/user/z/zhibin/sndData/converted/Neutrinos/1/" 
    cal_eff(neutrino_path)

if __name__ == "__main__":
    main()
