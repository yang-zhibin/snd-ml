import uproot
import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse


def partition_list(file_list, ratios):
    #print(len(file_list))
    total = sum(ratios)
    lengths = [len(file_list) * r // total for r in ratios]
    # Adjust the last partition to account for any rounding errors.
    lengths[-1] += len(file_list) - sum(lengths)
    from itertools import accumulate
    start_indexes = [0] + list(accumulate(lengths[:-1]))
    return [file_list[start:start + length] for start, length in zip(start_indexes, lengths)]


def GetEventList(paths, partition, out_path):
    print("prepare event list for training")
    print(paths, partition) #partition = [8,1,1]

    train_Flist, validation_Flist, test_Flist = [], [], []
    
    for root_path in paths:
        print(f"Processing files in: {root_path}")
        file_list = []
        event_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.root') and "converted" in filename:
                    file_path = os.path.join(root, filename)
                    try:
                        with uproot.open(file_path) as Rfile:
                            tree = Rfile["cbmsim"]
                            n_evt = tree.num_entries
                        event_list.append(n_evt)
                        file_list.append(file_path)
                    except KeyError:
                        print(f"Error: 'cbmsim' tree not found in {filename}.")
                    except Exception as e:
                        print(f"Error opening {filename}: {e}")

        # Check if lists have files and events, then partition them
        if file_list:
            train_files, validation_files, test_files = partition_list(file_list, partition)
            train_evts, validation_evts, test_evts = partition_list(event_list, partition)
            train_Flist.extend(zip(train_files, train_evts))
            validation_Flist.extend(zip(validation_files, validation_evts))
            test_Flist.extend(zip(test_files, test_evts))

            print(f"Partition sizes - Train: {len(train_files)}, Validation: {len(validation_files)}, Test: {len(test_files)}")
            print(f"Event counts - Train: {len(train_evts)}, Validation: {len(validation_evts)}, Test: {len(test_evts)}")
        else:
            print("No valid files found in this directory.")        
            
    # Save the results to text files for each partition
    for category, data in zip(["train", "validation", "test"], [train_Flist, validation_Flist, test_Flist]):
        filename = f"{out_path}/{category}_files.csv"
        with open(filename, 'w') as file:
            for file_path, events in data:
                file.write(f"{file_path}, {events}\n")
        print(f"Saved {category} partition data to {filename}")

def remove_vetoHits(df):
    print("removing vetoHits")
    # Iterate over each row
    mask = df['Hits.detType'].apply(lambda x: np.array(x) == 1)
    
    # Iterate over each column and apply the mask
    for column in df.columns:
        #print(column)
        df[column] = df.apply(lambda row: [x for i, x in enumerate(row[column]) if not mask[row.name][i]], axis=1)
    
    return df

def simple_cuts(hit_features_df, event_features_df, ids_df):
    num_hit_cut = 5

    # cut1: hits>5
    hitLengths = hit_features_df['Hits.detType'].apply(len)
    cut1 = hitLengths>num_hit_cut
    #print("cut1", hit_features_df[cut1])

    #cut2: no vetoHits
    cut2 = hit_features_df['Hits.detType'].apply(lambda x: all(hit != 1 for hit in x))
    cuts = cut1 & cut2
    
    return cuts

def convert_to_tensor_and_save(hit_features_df, event_features_df, ids_df, out_file):
    #print(event_features_df)
    event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])
    events = []
    #print(ids_df)
    for (idx1, hit_row), (idx2, id_row), (idx3, event_row) in zip(hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows()):
        hit_tensors = torch.stack([torch.tensor(hit_row[feature], dtype=torch.float32) for feature in hit_row.index])
        event_tensors = torch.stack([torch.tensor(event_row[feature], dtype=torch.float32) for feature in event_row.index])
        #print(id_row)
        #print(event_row)

        #print (hit_tensors.shape)
        #print (event_tensors.shape)
        data = Data(pdgCode = id_row['pdgCode'], runId = id_row['runId'], eventId = id_row['eventId'],\
            hitFeature = hit_tensors, eventFeatures=event_tensors)
        events.append(data)
        

    torch.save(events, out_file)
    print(f"Saved tensors to {out_file}")



def preprocess(data_list_path, split, out_path):
    
    list_df = pd.read_csv(data_list_path, header=None)
    data_list = list_df.iloc[:,0]
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']

    ids_name = ['runId','eventId','partitionId','pdgCode'] 
    #print(list_df)
    #print(data_list)
    tree_name = 'cbmsim'
    for path in tqdm(data_list):
        folder = path.split('/')[-3]
        partition = path.split('/')[-2]
        out_file = '{}/{}_{}_{}.pt'.format(out_path, split, folder, partition)
        #if (os.path.exists(out_file)):
            #continue
        
        with uproot.open(path) as file:
            tree = file[tree_name]
            hit_features_df = tree.arrays(hit_features_name, library="pd")
            event_features_df = tree.arrays(event_features_name, library="pd")
            ids_df = tree.arrays(ids_name, library="pd")


            cut = simple_cuts(hit_features_df, event_features_df, ids_df)
            hit_features_df = hit_features_df[cut].reset_index(drop=True)
            event_features_df = event_features_df[cut].reset_index(drop=True)
            ids_df = ids_df[cut].reset_index(drop=True)

            if 'muon' in path:
                print(path)
                hit_features_df = remove_vetoHits(hit_features_df)
            
            convert_to_tensor_and_save(hit_features_df, event_features_df, ids_df, out_file)

    #print("hit",all_hit_features)
    #print("event",all_event_features)

def preprocess_pred(data_list_path, split, out_path, job_number):
    
    list_df = pd.read_csv(data_list_path, header=None)
    length_df = len(list_df)
    start = job_number*1000
    end = start+1000
    data_list = list_df.iloc[start:end,0]
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']

    ids_name = ['runId','eventId','pdgCode'] 
    #print(list_df)
    #print(data_list)
    tree_name = 'cbmsim'
    for path in tqdm(data_list):
        with uproot.open(path) as file:
            tree = file[tree_name]
            hit_features_df = tree.arrays(hit_features_name, library="pd")
            event_features_df = tree.arrays(event_features_name, library="pd")
            ids_df = tree.arrays(ids_name, library="pd")
            
            cut = simple_cuts(hit_features_df, event_features_df, ids_df)
            hit_features_df = hit_features_df[~cut].reindex()
            event_features_df = event_features_df[~cut].reindex()
            ids_df = ids_df[~cut].reindex()

            folder = path.split('/')[-3]
            partition = path.split('/')[-2]
            out_file = '{}/{}_{}_{}.pt'.format(out_path, split, folder, partition)

            if 'muon' in path:
                print(path)
                hit_features_df = remove_vetoHits(hit_features_df)
            
            convert_to_tensor_and_save(hit_features_df, event_features_df, ids_df, out_file)

    #print("hit",all_hit_features)
    #print("event",all_event_features)

def prepare_muon():
    pred_list = "/eos/user/z/zhibin/sndData/converted/muons/muons_up/prediction_files.csv"
    out_path = '/eos/user/z/zhibin/sndData/converted/pt_muon/'

    preprocess(pred_list, 'pred_muon', out_path)

def main():
    train_list = "/eos/user/z/zhibin/sndData/converted/train_files.csv"
    val_list = "/eos/user/z/zhibin/sndData/converted/validation_files.csv"
    test_list = "/eos/user/z/zhibin/sndData/converted/test_files.csv"
    pred_list = "/eos/user/z/zhibin/sndData/converted/prediction_files.csv"

    # List of data lists and corresponding splits
    lists_and_splits = [
        (val_list, 'val'),
        (train_list, 'train'),
        #(test_list, 'test')
        #(pred_list, 'pred')
    ]

    out_path = '/eos/user/z/zhibin/sndData/converted/pt_data/'
    if os.path.isdir(out_path):
        print(f"{out_path} Exists")
    else:
        print(f"Doesn't exists, create {out_path}")
        os.mkdir(out_path)
    
    #preprocess(test_list, 'test', out_path)
    # Loop over the data lists and splits to preprocess each one
    for data_list, split in lists_and_splits:
        print('processing', split)
        preprocess(data_list, split, out_path)

def prepare_pred(job_number):
    pred_list = "/eos/user/z/zhibin/sndData/converted/prediction_files.csv"

    # List of data lists and corresponding splits
    lists_and_splits = [
        (pred_list, 'pred')
    ]

    out_path = '/eos/user/z/zhibin/sndData/converted/pt_data/'
    
    #preprocess(test_list, 'test', out_path)
    # Loop over the data lists and splits to preprocess each one
    for data_list, split in lists_and_splits:
        print('processing', split)
        preprocess_pred(data_list, split, out_path, job_number)

def test():
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']

    ids_name = ['runId','eventId','pdgCode'] 
    path = '/eos/user/z/zhibin/sndData/converted/Neutrinos/400/sndLHC.Genie-TGeant4_20240126_digCPP_converted_00400.root'
    tree_name = 'cbmsim'
    with uproot.open(path) as file:
            tree = file[tree_name]
            hit_features_df = tree.arrays(hit_features_name, library="pd")
            event_features_df = tree.arrays(event_features_name, library="pd")
            ids_df = tree.arrays(ids_name, library="pd")

            cut = simple_cuts(hit_features_df, event_features_df, ids_df)
            hit_features_df = hit_features_df[~cut].reindex()
            event_features_df = event_features_df[~cut].reindex()
            ids_df = ids_df[~cut].reindex()


            out_file = 'test.pt'
            
            convert_to_tensor_and_save(hit_features_df, event_features_df, ids_df, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-n", "--num", dest="num", type=int, required=True)
    args = parser.parse_args()
    job_number = args.num

    prepare_pred(job_number)
    main()
    prepare_muon()