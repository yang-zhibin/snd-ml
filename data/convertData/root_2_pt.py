import uproot
import os
import csv
from itertools import islice
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import argparse
import awkward as ak
import numpy as np
import time
import shutil
import numba


def process_root_file(path, out_path, chunk_size=1e5):
    hit_features_name = ['Hits.detType', 'Hits.orientation', 'Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
    ids_name = ['pdgCode', 'runId', 'eventId', 'partitionId']

    tree_name = 'cbmsim'

    recreate = True


    events = []
    chunk_counter = 0

    with uproot.open(path) as file:
        print('Opened file')
        if tree_name not in file:
            print(f"Tree {tree_name} not found in file {path}")
            return

        tree = file[tree_name]
        total_events = tree.num_entries
        print(f"Total events: {total_events}")

        for start in range(0, total_events, int(chunk_size)):

            chunk_out_path = f"{out_path}_chunk_{chunk_counter}.pt"
            if os.path.exists(chunk_out_path) and recreate:
                os.remove(chunk_out_path)
                print(f"{chunk_out_path} removed.")
            
            end = min(start + int(chunk_size), total_events)
            print(f"Processing chunk {chunk_counter + 1}: Events {start} to {end}")

            # Load data for the chunk
            hit_features_df = tree.arrays(hit_features_name, library="pd", entry_start=start, entry_stop=end)
            event_features_df = tree.arrays(event_features_name, library="pd", entry_start=start, entry_stop=end)
            ids_df = tree.arrays(ids_name, library="pd", entry_start=start, entry_stop=end)

            weight = 1
            normalized_weight = 1
            intRate_weight = 1
            event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])

            for (idx1, hit_row), (idx2, id_row), (idx3, event_row) in zip(
                hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows()
            ):
                tensors_list = []
                detType = np.array(hit_row['Hits.detType'])
                mask = detType != 1
                for feature in hit_row.index:
                    f = np.array(hit_row[feature])
                    value = f[mask]
                    tensor = torch.tensor(value, dtype=torch.float32)
                    tensors_list.append(tensor)
                hit_tensors = torch.stack(tensors_list)

                event_tensors = torch.stack([torch.tensor(event_row[feature], dtype=torch.float32) for feature in event_row.index])

                data = Data(
                    pdgCode=id_row['pdgCode'],
                    runId=id_row['runId'],
                    eventId=id_row['eventId'],
                    partitionId=id_row['partitionId'],
                    fileId=path,
                    weight=weight,
                    normalized_weight=normalized_weight,
                    intRate_weight=intRate_weight,
                    hitFeature=hit_tensors,
                    eventFeatures=event_tensors,
                )
                events.append(data)

            # Save chunk
            
            torch.save(events, chunk_out_path)
            print(f"Saved chunk {chunk_counter + 1} to {chunk_out_path}")

            # Reset for the next chunk
            events = []
            chunk_counter += 1


def main(args):
    in_file = args.in_file
    out_path = args.out_path

    #in_file = '/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess/run_005865/real_muon_converted_sndsw_raw-0001.root'
    #out_path = '/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess/run_005865/pt/'
    chunk_size = 1e4
    process_root_file(in_file, out_path, chunk_size)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", dest="in_file")
    parser.add_argument("-o", "--out_path", dest="out_path")
    args = parser.parse_args()
    main(args)

