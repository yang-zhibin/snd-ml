import os
import uproot
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import re
import gzip
from tqdm import tqdm
import awkward as ak

# convert root format to pt format

particle_to_target = {
        12: 0, -12: 0,
        14: 1, -14: 1,
        16: 2, -16: 2,
        112: 3, -112: 3, 114: 3, -114: 3, 116: 3, -116: 3,
        130: 4, 310: 4,
        2112: 5,
        13:6, -13:6,
        0:6
}

def main(args):
    hit_column_name = ['Hits/Hits.detType', 'Hits/Hits.orientation', 'Hits/Hits.x1', 'Hits/Hits.y1', 'Hits/Hits.z1', 'Hits/Hits.x2', 'Hits/Hits.y2', 'Hits/Hits.z2']
    feature_column_name = ['px', 'py', 'pz']
    id_column_name = ['pdgCode', 'eventId','runId']
    
    tree_name = 'snddata'
    chunk_size = int(args.chunk_size)
    hit_path = args.hit_path
    feature_path = args.feature_path

    out_file = args.out_file

    with uproot.open(hit_path) as hit_file, uproot.open(feature_path) as feature_file:
        print('Opened hit and feature files')
        
        if tree_name not in hit_file or tree_name not in feature_file:
            print(f"Tree {tree_name} not found in one of the files")
            return
        
        hit_tree = hit_file[tree_name]
        feature_tree = feature_file[tree_name]

        chunk_counter = 0
        total_events = min(hit_tree.num_entries, feature_tree.num_entries)
        print(f"Total events: {total_events}")
        event_data = []
        for start in range(0, total_events, chunk_size):
            
            
            end = min(start + chunk_size, total_events)
            print(f"Processing chunk {chunk_counter + 1}: Events {start} to {end}")
            
            # Load data from uproot (returns a dictionary of NumPy arrays)
            hit_array = hit_tree.arrays(hit_column_name, entry_start=start, entry_stop=end)
            feature_array = feature_tree.arrays(feature_column_name, entry_start=start, entry_stop=end)
            id_array = feature_tree.arrays(id_column_name, entry_start=start, entry_stop=end)
            
            num_events_in_chunk = len(feature_array)
            for idx in range(num_events_in_chunk):
                # For hit and feature data: gather values across all columns
                hit_tensors = torch.stack([
                    torch.tensor(hit_array[col][idx], dtype=torch.float32) for col in hit_column_name
                ])
                #print(hit_tensors)

                feature_tensors = torch.stack([
                    torch.tensor(feature_array[col][idx], dtype=torch.float32) for col in feature_column_name
                ])
                #print(feature_tensors)
                #print(id_array['pdgCode'][idx], id_array['runId'][idx], id_array['eventId'][idx])
                one_event_data = {
                    "pdgCode": id_array['pdgCode'][idx],
                    "runId": id_array['runId'][idx],
                    "eventId": id_array['eventId'][idx],
                    "hitFeature": hit_tensors,
                    "eventFeatures": feature_tensors,
                }
                event_data.append(one_event_data)
            
            chunk_counter += 1


        with gzip.open(out_file, 'wb') as f:
            torch.save(event_data, f)
            #torch.save(event_data, chunk_out_path)
        print(f"Saved data to {out_file}")
            
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hit", "--hitPath", dest="hit_path", help="hit data file path", required=True)
    parser.add_argument("-f", "--featurePath", dest="feature_path", help="feature path", required=True)
    parser.add_argument("-o", "--outfile", dest="out_file", help="last output chunk file", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-c", "--chunkSize", dest="chunk_size", help="max chunk size ", default=1e5)
    
    args = parser.parse_args()
    
    main(args)
