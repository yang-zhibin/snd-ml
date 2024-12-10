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



def process_root_file(path, out_path, chunk_size):
    #print(len(chunk))
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
    ids_name = ['pdgCode','runId','eventId', 'partitionId'] 
    #label_name = ['px', 'py', 'pz', 'x', 'y', 'z', 'stage2','scifi_avg_ver', 'scifi_avg_hor','DS_avg_ver','DS_avg_hor']

    tree_name = 'cbmsim'

    recreate = True
    if os.path.exists(out_path) and (recreate):
        os.remove(out_path)
        print(f"{out_path} removed.")

    events = []

    with uproot.open(path) as file:
        print('opened file')
        if tree_name not in file:
            print(f"Tree {tree_name} not found in file {path}")

        tree = file[tree_name]
        #print(tree.keys())
        iter_idx = 0

        hit_features_df = tree.arrays(hit_features_name, library="pd")
        event_features_df = tree.arrays(event_features_name, library="pd")
        ids_df = tree.arrays(ids_name, library="pd")
        #label_df = tree.arrays(label_name, library="pd")
        

        weight  = 1
        normalized_weight  =1
        intRate_weight  =1
        event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])

        

        for (idx1, hit_row), (idx2, id_row), (idx3, event_row) in zip(hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows()):
        
            #print(idx1)

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
            #label_tensors = torch.stack([torch.tensor(label_row[feature], dtype=torch.float32) for feature in label_row.index])
            #print(label_tensors)
            #print(event_tensors)
            #print(label_tensors)
            data = Data(
                pdgCode=id_row['pdgCode'], 
                runId=id_row['runId'], 
                eventId=id_row['eventId'],
                partitionId = id_row['partitionId'],
                fileId=path,
                weight = weight,
                normalized_weight = normalized_weight,
                intRate_weight = intRate_weight,
                hitFeature=hit_tensors, 
                eventFeatures=event_tensors
            )
            #print(data)
            #print(data)
            events.append(data)
    print(len(events))
    torch.save(events, out_path)
    print(f"Saved tensors to {out_path}")

def main(args):
    in_file = args.in_file
    out_path = args.out_path

    #in_file = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/1/neutrinos_converted_sndLHC.Genie-TGeant4_20240126_digCPP.root'
    #out_file = '/eos/user/z/zhibin/sndData/converted/pt/test/Neutrinos/test_neutrino_partition_1.pt'
    chunk_size = 1e5
    process_root_file(in_file, out_path, chunk_size)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", dest="in_file")
    parser.add_argument("-o", "--out_path", dest="out_path")
    args = parser.parse_args()
    main(args)

