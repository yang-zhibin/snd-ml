import uproot
import os
# import csv
# from itertools import islice
# import pandas as pd
# from tqdm import tqdm
import torch
from torch_geometric.data import Data
import argparse
import awkward as ak
import numpy as np
import time
import shutil
import numba

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


def filter_lists(row, columns):
    mask = [val != 0 for val in row['Hits.detType']]  # Create a mask where 'Hits.detType' is not 0
    filtered_row = {}
    
    # Apply the mask to all list columns in the row
    for col in columns:
        filtered_row[col] = [x for x, m in zip(row[col], mask) if m]
    
    return pd.Series(filtered_row)

# @numba.jit
# def create_data(chunk, event_features_name, hit_features_name):
#     events = []
#     mask = chunk['Hits.detType'] != 1
#     for n in range(len(chunk)):
#         event_tensors = torch.stack([torch.tensor(chunk[feature][n], dtype=torch.float32) for feature in event_features_name])
#         hit_tensors = torch.stack([torch.tensor(chunk[feature][mask][n], dtype=torch.float32) for feature in hit_features_name])
#         data = Data(
#             # pdgCode=id_row['pdgCode'], 
#             # runId=id_row['runId'], 
#             # eventId=id_row['eventId'],
#             # partitionId = id_row['partitionId'],
#             # fileId=path,
#             # px=label_row['px'],
#             # py=label_row['py'],
#             # pz=label_row['pz'],
#             # x=label_row['x'],
#             # y=label_row['y'],
#             # z=label_row['z'],
#             # stage2 = label_row['stage2'],

#             # scifi_avg_ver = label_row['scifi_avg_ver'],
#             # scifi_avg_hor = label_row['scifi_avg_hor'],
#             # DS_avg_ver = label_row['DS_avg_ver'],
#             # DS_avg_hor = label_row['DS_avg_hor'],
            
#             # weight = weight,
#             # normalized_weight = normalized_weight,
#             # intRate_weight = intRate_weight,
#             hitFeature=hit_tensors, 
#             eventFeatures=event_tensors
#         )
#         events.append(data)
#     return events

def process_root_file(path, out_path):
    #print(len(chunk))
    hit_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
    event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
    ids_name = ['pdgCode','runId','eventId', 'partitionId'] 
    label_name = ['px', 'py', 'pz', 'x', 'y', 'z', 'stage2','scifi_avg_ver', 'scifi_avg_hor','DS_avg_ver','DS_avg_hor']

    tree_name = 'cbmsim'

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)


    with uproot.open(path) as file:
        print('opened file')
        if tree_name not in file:
            print(f"Tree {tree_name} not found in file {path}")

        tree = file[tree_name]
        #print(tree.keys())
        iter_idx = 0
        for chunk in tree.iterate(hit_features_name + event_features_name + ids_name + label_name, step_size="100 MB"):
            print(f"Processing chunk {iter_idx} ...")
            mask = chunk['Hits.detType'] != 1
            # print(f'mask lenght = {len(mask)}')
            # hit_inputs = {}
            # for feature in hit_features_name:
            #     x = chunk[feature]
            #     hit_inputs[feature](x[mask])
            # events = create_data(chunk, event_features_name, hit_features_name)
            # events = []
            event_tensors = torch.stack([torch.tensor(chunk[feature], dtype=torch.float32) for feature in event_features_name])
            hit_tensors = torch.stack([torch.tensor(chunk[feature][mask], dtype=torch.float32) for feature in hit_features_name])
            data = Data(
                # pdgCode=id_row['pdgCode'], 
                # runId=id_row['runId'], 
                # eventId=id_row['eventId'],
                # partitionId = id_row['partitionId'],
                # fileId=path,
                # px=label_row['px'],
                # py=label_row['py'],
                # pz=label_row['pz'],
                # x=label_row['x'],
                # y=label_row['y'],
                # z=label_row['z'],
                # stage2 = label_row['stage2'],

                # scifi_avg_ver = label_row['scifi_avg_ver'],
                # scifi_avg_hor = label_row['scifi_avg_hor'],
                # DS_avg_ver = label_row['DS_avg_ver'],
                # DS_avg_hor = label_row['DS_avg_hor'],
                
                # weight = weight,
                # normalized_weight = normalized_weight,
                # intRate_weight = intRate_weight,
                hitFeature=hit_tensors, 
                eventFeatures=event_tensors,
                hit_indices= 
            )
                events.append(data)
            torch.save(events, os.path.join(out_path, f"data_{iter_idx}.pt"))
            iter_idx += 1

        raise RuntimeError("stop")
        hit_features_df = tree.arrays(hit_features_name, library="pd")
        event_features_df = tree.arrays(event_features_name, library="pd")
        ids_df = tree.arrays(ids_name, library="pd")
        label_df = tree.arrays(label_name, library="pd")
        
        
        #hit_features_df = hit_features_df.apply(filter_lists, axis=1, columns=hit_features_df.columns)
        #print()
        #hit_features_df = hit_features_df[cut].reset_index(drop=True)
        #event_features_df = event_features_df[cut].reset_index(drop=True)
        #ids_df = ids_df[cut].reset_index(drop=True)
        #label_df = label_df[cut].reset_index(drop=True)
        #print(type(ids_df))
        weight  =1 #row['weight']
        normalized_weight  =1 #row['normalized_weight']
        intRate_weight  =1 #row['intRate_weight']
        event_features_df = event_features_df.map(lambda x: x if len(x) > 0 else [0])

        for (idx1, hit_row), (idx2, id_row), (idx3, event_row), (idx4, label_row) in zip(hit_features_df.iterrows(), ids_df.iterrows(), event_features_df.iterrows(), label_df.iterrows()):
        
            print(idx1)

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
                px=label_row['px'],
                py=label_row['py'],
                pz=label_row['pz'],
                x=label_row['x'],
                y=label_row['y'],
                z=label_row['z'],
                stage2 = label_row['stage2'],

                scifi_avg_ver = label_row['scifi_avg_ver'],
                scifi_avg_hor = label_row['scifi_avg_hor'],
                DS_avg_ver = label_row['DS_avg_ver'],
                DS_avg_hor = label_row['DS_avg_hor'],
                
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
    label_name = ['px', 'py', 'pz', 'x', 'y', 'z', 'stage2','scifi_avg_ver', 'scifi_avg_hor','DS_avg_ver','DS_avg_hor']

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
                
                
                #cut = simple_cuts(hit_features_df, event_features_df, ids_df)
                #hit_features_df = hit_features_df[cut].reset_index(drop=True)
                #event_features_df = event_features_df[cut].reset_index(drop=True)
                #ids_df = ids_df[cut].reset_index(drop=True)
                #label_df = label_df[cut].reset_index(drop=True)
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
                    
                    print(idx1)
                    tensors_list = []
                    mask = hit_row['Hits.detType'] != 1
                    for feature in hit_row.index:
                        value = hit_row[feature][mask]
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
                        fileId=path,

                        px=label_row['px'],
                        py=label_row['py'],
                        pz=label_row['pz'],
                        x=label_row['x'],
                        y=label_row['y'],
                        z=label_row['z'],
                        stage2 = label_row['stage2'],
                        scifi_avg_ver = label_row['scifi_avg_ver'],
                        scifi_avg_hor = label_row['scifi_avg_hor'],
                        DS_avg_ver = label_row['DS_avg_ver'],
                        DS_avg_hor = label_row['DS_avg_hor'],
                        
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
    csv_path = f"{root_path}/real_muon/real_muon_evt_list.csv"
    event_list = pd.read_csv(csv_path)

    chunk_size = 50 * 1e4
    cumsum = event_list['event'].cumsum()
    chunk_indices = cumsum // chunk_size

    print(chunk_indices)
    
    for chunk_index in tqdm(range(2,int(chunk_indices.max()) + 1)):
        chunk = event_list[chunk_indices == chunk_index]
        # Process your chunk here
        #print(len(chunk))
        out_path = f"{root_path}/pt/{partition}_real_muon_{chunk_index}.pt"
        if os.path.exists(out_path):
            print(out_path)
            continue

        process_root_chunk(chunk, out_path, MC=False)

def real_muon_root_2_pt(root_path):
    #parser = argparse.ArgumentParser(description="Process some integers.")
    #parser.add_argument("-n", "--num", dest="num", type=int, required=True)
    #args = parser.parse_args()
    #chunk_index = args.num

    partition ='test'

    print(f"processing {partition}")
    csv_path = f"{root_path}/real_muon/real_muon_evt_list.csv"
    event_list = pd.read_csv(csv_path)

    
    print(event_list)
    out_path = f"{root_path}/pt/{partition}_real_muon.pt"


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

def root_2_pt_after_selection():
    # neutrino_file_path = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/selection_tmp/scifi_area_margin5cm_selection.root'
    # neutrino_out_path = f'/eos/user/z/zhibin/sndData/converted/pt/test_neutrino_scifi_area_margin5cm_selection.pt'

    # print(f'processing {neutrino_file_path}')
    # process_root_file(neutrino_file_path, neutrino_out_path)

    # muon_file_path = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/scifi_area_margin5cm_selection.root'
    # muon_out_path = f'/eos/user/z/zhibin/sndData/converted/pt/test_muon_scifi_area_margin5cm_selection.pt'

    # print(f'processing {muon_file_path}')
    # process_root_file(muon_file_path, muon_out_path)

    # muon_2_file_path = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/outside_scifi_area_selection.root'
    # muon_2_out_path = f'/eos/user/z/zhibin/sndData/converted/pt/test_muon_outside_scifi_area_selection.pt'

    # print(f'processing {muon_2_file_path}')
    # process_root_file(muon_2_file_path, muon_2_out_path)

    muon_3_file_path = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/scifi_area_margin5cm_veto.root'
    muon_3_out_path = f'/eos/user/z/zhibin/sndData/converted/pt_tmp/'

    print(f'processing {muon_3_file_path}')
    process_root_file(muon_3_file_path, muon_3_out_path)

    # muon_4_file_path = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/scifi_area_margin5cm_no_veto.root'
    # muon_4_out_path = f'/eos/user/z/zhibin/sndData/converted/pt/test_muon_scifi_area_margin5cm_no_veto.pt'

    # print(f'processing {muon_4_file_path}')
    # process_root_file(muon_4_file_path, muon_4_out_path)


if __name__ == "__main__":
    root_path = '/eos/user/z/zhibin/sndData/converted/'
    #neutrino_root_2_pt(root_path)
    #condor_root_2_pt(root_path)
    #muon_root_2_pt(root_path)
    #real_muon_root_2_pt(root_path)
    root_2_pt_after_selection()

