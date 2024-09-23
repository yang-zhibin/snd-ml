import uproot
import os
import csv
from itertools import islice
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import argparse


def simple_cuts(hit_features_df, event_features_df, ids_df):
    num_hit_cut = 5

    # cut1: hits>5
    hitLengths = hit_features_df['Hits.detType'].apply(len)
    cut1 = hitLengths>num_hit_cut
    #print("cut1", hit_features_df[cut1])

    #cut2: no vetoHits
    cut2 = hit_features_df['Hits.detType'].apply(lambda x: all(hit != 1 for hit in x))
    cuts = cut1 & cut2

    #sum_false_cuts = (~cuts).sum()
    #print("Sum of False in cuts:", sum_false_cuts)

    return cuts

def process_root_file(path, out_path):
    #print(len(chunk))
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
    ids_name = ['pdgCode','runId','eventId'] 
    label_name = ['px', 'py', 'pz', 'x', 'y', 'z', 'stage2']

    tree_name = 'cbmsim'


    events = []

    with uproot.open(path) as file:
        if tree_name not in file:
            print(f"Tree {tree_name} not found in file {path}")

        
        tree = file[tree_name]
        #print(tree.keys())
        hit_features_df = tree.arrays(hit_features_name, library="pd")
        event_features_df = tree.arrays(event_features_name, library="pd")
        ids_df = tree.arrays(ids_name, library="pd")
        label_df = tree.arrays(label_name, library="pd")
        
        
        cut = simple_cuts(hit_features_df, event_features_df, ids_df)
        hit_features_df = hit_features_df[cut].reset_index(drop=True)
        event_features_df = event_features_df[cut].reset_index(drop=True)
        ids_df = ids_df[cut].reset_index(drop=True)
        label_df = label_df[cut].reset_index(drop=True)
        #print(type(ids_df))
        weight  =1 #row['weight']
        normalized_weight  =1 #row['normalized_weight']
        intRate_weight  =1 #row['intRate_weight']
        event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])
        for (idx1, hit_row), (idx2, id_row), (idx3, event_row), (idx4, label_row) in zip(hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows(), label_df.iterrows()):
            hit_tensors = torch.stack([torch.tensor(hit_row[feature], dtype=torch.float32) for feature in hit_row.index])
            event_tensors = torch.stack([torch.tensor(event_row[feature], dtype=torch.float32) for feature in event_row.index])
            #label_tensors = torch.stack([torch.tensor(label_row[feature], dtype=torch.float32) for feature in label_row.index])
            #print(label_tensors)
            #print(event_tensors)
            #print(label_tensors)
            data = Data(
                pdgCode=id_row['pdgCode'], 
                runId=id_row['runId'], 
                eventId=id_row['eventId'],
                fileId=path,
                px=label_row['px'],
                py=label_row['py'],
                pz=label_row['pz'],
                x=label_row['x'],
                y=label_row['y'],
                z=label_row['z'],
                stage2 = label_row['stage2'],
                
                weight = weight,
                normalized_weight = normalized_weight,
                intRate_weight = intRate_weight,
                hitFeature=hit_tensors, 
                eventFeatures=event_tensors
            )
            #print(data)
            events.append(data)
    
    torch.save(events, out_path)
    print(f"Saved tensors to {out_path}")
   


def process_root_chunk(chunk, out_path, MC=True):
    #print(len(chunk))
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
    ids_name = ['pdgCode','runId','eventId'] 
    label_name = ['px', 'py', 'pz', 'x', 'y', 'z', 'stage2']

    tree_name = 'cbmsim'


    events = []
    for idx, row in tqdm(chunk.iterrows(), total=len(chunk)):
        path = row['file']
        #print(path)
        try:
            with uproot.open(path) as file:
                if tree_name not in file:
                    print(f"Tree {tree_name} not found in file {path}")
                    continue

                tree = file[tree_name]
                #print(tree.keys())
                hit_features_df = tree.arrays(hit_features_name, library="pd")
                event_features_df = tree.arrays(event_features_name, library="pd")
                ids_df = tree.arrays(ids_name, library="pd")
                label_df = tree.arrays(label_name, library="pd")
                
                
                cut = simple_cuts(hit_features_df, event_features_df, ids_df)
                hit_features_df = hit_features_df[cut].reset_index(drop=True)
                event_features_df = event_features_df[cut].reset_index(drop=True)
                ids_df = ids_df[cut].reset_index(drop=True)
                label_df = label_df[cut].reset_index(drop=True)
                #print(type(ids_df))
                if MC:
                    weight  =row['weight']
                    normalized_weight  =row['normalized_weight']
                    intRate_weight  =row['intRate_weight']
                else:
                    weight  = 1
                    normalized_weight  =1
                    intRate_weight  =1


                event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])
                for (idx1, hit_row), (idx2, id_row), (idx3, event_row), (idx4, label_row) in zip(hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows(), label_df.iterrows()):
                    hit_tensors = torch.stack([torch.tensor(hit_row[feature], dtype=torch.float32) for feature in hit_row.index])
                    event_tensors = torch.stack([torch.tensor(event_row[feature], dtype=torch.float32) for feature in event_row.index])
                    #label_tensors = torch.stack([torch.tensor(label_row[feature], dtype=torch.float32) for feature in label_row.index])
                    #print(label_tensors)
                    #print(event_tensors)
                    #print(label_tensors)
                    data = Data(
                        pdgCode=id_row['pdgCode'], 
                        runId=id_row['runId'], 
                        eventId=id_row['eventId'],
                        fileId=path,

                        px=label_row['px'],
                        py=label_row['py'],
                        pz=label_row['pz'],
                        x=label_row['x'],
                        y=label_row['y'],
                        z=label_row['z'],
                        stage2 = label_row['stage2'],
                        
                        weight = weight,
                        normalized_weight = normalized_weight,
                        intRate_weight = intRate_weight,

                        hitFeature=hit_tensors, 
                        eventFeatures=event_tensors
                    )
                    #print(data)
                    events.append(data)
        
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue

    torch.save(events, out_path)
    print(f"Saved tensors to {out_path}")
    
def root_2_pt(root_path):
    partitions ={
        'train',
        'val',
        'test',
    }

    for partition in partitions:
        print(f"processing {partition}")
        csv_path = f"{root_path}/combined_{partition}.csv"
        event_list = pd.read_csv(csv_path)

        chunk_size = 400000
        cumsum = event_list['event'].cumsum()
        chunk_indices = cumsum // chunk_size
        print(chunk_indices)
        
        for chunk_index in tqdm(range(chunk_indices.max() + 1)):
            chunk = event_list[chunk_indices == chunk_index]
            # Process your chunk here
            #print(len(chunk))
            out_path = f"{root_path}/pt/{partition}_{chunk_index}.pt"
            if os.path.exists(out_path):
                continue
            process_root_chunk(chunk, out_path)

def condor_root_2_pt(root_path):
    #parser = argparse.ArgumentParser(description="Process some integers.")
    #parser.add_argument("-n", "--num", dest="num", type=int, required=True)
    #args = parser.parse_args()
    #chunk_index = args.num

    partition ='test'

    print(f"processing {partition}")
    csv_path = f"{root_path}/mixed_evt_list.csv"
    event_list = pd.read_csv(csv_path)

    chunk_size = 50 * 1e4
    cumsum = event_list['event'].cumsum()
    chunk_indices = cumsum // chunk_size

    print(chunk_indices)
    
    for chunk_index in tqdm(range(2,int(chunk_indices.max()) + 1)):
        chunk = event_list[chunk_indices == chunk_index]
        # Process your chunk here
        #print(len(chunk))
        out_path = f"{root_path}/pt/{partition}_{chunk_index}.pt"
        if os.path.exists(out_path):
            print(out_path)
            continue

        process_root_chunk(chunk, out_path)

def real_muon_root_2_pt(root_path):
    #parser = argparse.ArgumentParser(description="Process some integers.")
    #parser.add_argument("-n", "--num", dest="num", type=int, required=True)
    #args = parser.parse_args()
    #chunk_index = args.num

    partition ='test'

    print(f"processing {partition}")
    csv_path = f"{root_path}/muon_real/muon_real_evt_list.csv"
    event_list = pd.read_csv(csv_path)


    
    print(event_list)
    out_path = f"{root_path}/pt/{partition}_muon_real.pt"


    process_root_chunk(event_list, out_path, MC=False)


def neutrino_root_2_pt(root_path):
    splits = [0.5,  0.2, 0.3]
    csv_path = "/afs/cern.ch/user/z/zhibin/eos/sndData/converted/Neutrinos/neutrinos_evt_list.csv"
    event_list = pd.read_csv(csv_path)

    total_len = len(event_list)
    split1_len = int(total_len * splits[0])
    split2_len = int(total_len * splits[1])
    split3_len = total_len - split1_len - split2_len


    # Split the DataFrame
    train_list = event_list[:split1_len]
    val_list = event_list[split1_len:split1_len + split2_len]
    test_list = event_list[split1_len + split2_len:]

    print(len(train_list), len(val_list),len(test_list))
    
    train_outfile = f"{root_path}/pt/train_neutrino.pt"
    val_outfile = f"{root_path}/pt/val_neutrino.pt"
    test_outfile = f"{root_path}/pt/test_neutrino.pt"

    process_root_chunk(train_list, train_outfile)
    process_root_chunk(val_list, val_outfile)
    process_root_chunk(test_list, test_outfile)

def muon_root_2_pt(root_path):
    muon_list = '/eos/user/z/zhibin/sndData/converted/muons/muons_evt_list.csv'
    event_list = pd.read_csv(muon_list)

    for index, row in event_list.iterrows():
        path = row['file']
        partition = row['partition']
        out_path = f"{root_path}/pt/test_{partition}_{index}.pt"
        process_root_file(path, out_path)


if __name__ == "__main__":
    root_path = '/eos/user/z/zhibin/sndData/converted/'
    #neutrino_root_2_pt(root_path)
    #condor_root_2_pt(root_path)
    #muon_root_2_pt(root_path)
    real_muon_root_2_pt(root_path)

