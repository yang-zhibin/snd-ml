import os
import pandas as pd
import glob
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch
import gzip
import pickle

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


class PredGeoDataset(InMemoryDataset):
    def __init__(self,root, pt_file, use_event_feature=False, weight_type=None, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.root = root
        self.pt_file = pt_file
        out_dir, out_name = os.path.split(pt_file)
        self.out_name = out_name
        if force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'processed_{self.out_name}.pt']

    def process(self):
        pt_file = self.pt_file

        #print(file)
        with gzip.open(pt_file, 'rb') as f:
            all_events_flattened = torch.load(f)
        
            all_event = []
            for i, evt in enumerate(all_events_flattened):
                #print(evt)
                # Ensure x (hitFeature) has at least (1, N) shape
                if evt["hitFeature"].numel() > 0:
                    x = evt["hitFeature"].squeeze()
                    #print(x.shape)
                    if len(x.shape) == 1:
                        x = x.unsqueeze(-1)
                    x = x.mT
                    
                else:
                    x = torch.zeros((1,evt["hitFeature"].shape[0]))


                # Ensure event_feature is always 2D
                event_feature = evt["eventFeatures"].unsqueeze(1).mT

                y = torch.tensor([particle_to_target.get(evt["pdgCode"])]) 
                ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]])
                ids = ids.unsqueeze(1).mT
                all_event.append(Data(x=x, event_feature=event_feature, weights=torch.tensor(1), y=y, ids=ids))

                #print(f"Event {i}: x={x.shape}, event_feature={event_feature.shape}, y={y.shape}, ids={ids.shape}")

            print('in process', self.processed_paths[0])
            self.save(all_event, self.processed_paths[0])
        
       

class TrainGeoDataset(InMemoryDataset):
    def __init__(self,root, metadata_dir, split, use_event_feature=False, weight_type=None, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.root = root
        self.metadata_dir = metadata_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def process(self):
        metadata_files = glob.glob(os.path.join(self.metadata_dir, "test*_metadata.csv"))
        df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)
        
        pt_prefixs = df.loc[df['split'] == self.split, 'pt_prefix'].tolist()
        files = []
        for prefix in pt_prefixs:
            files.extend(glob.glob(f"{prefix}*"))

        all_events_flattened = [event for file in files for event in torch.load(file)]

        all_event = []
        for i, evt in enumerate(all_events_flattened):
            
            # Ensure x (hitFeature) has at least (1, N) shape
            if evt.hitFeature.numel() > 0:
                x = evt.hitFeature
                if len(x.shape) ==1:
                    x = x.unsqueeze(-1)
                x = x.mT
                
            else:
                x = torch.zeros((1,evt.hitFeature.shape[0]))
            
            # if (i == 894):
            #     print(evt.hitFeature)
            #     print(x)
            #     print(evt.hitFeature.shape)

            # Ensure event_feature is always 2D
            event_feature = evt.eventFeatures

            y = torch.tensor([particle_to_target.get(evt.pdgCode)]) 
            ids = torch.tensor([evt.pdgCode, evt.runId, evt.eventId])

            all_event.append(Data(x=x, event_feature=event_feature, weights=torch.tensor(1), y=y, ids=ids))
            

            #print(f"Event {i}: x={x.shape}, event_feature={event_feature.shape}, y={y.shape}, ids={ids.shape}")

        print('in process', self.processed_paths[0])
        self.save(all_event, self.processed_paths[0])


class PredGeoDataset_3D(Dataset):
    def __init__(self, root, pkl_file, split, use_event_feature=False, weight_type=None,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        self.pkl_file = pkl_file
        self.split = split
        self.use_event_feature = use_event_feature
        self.weight_type = weight_type
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        files = [f for f in os.listdir(self.processed_dir) if f.startswith(f"{self.split}_data_")]
        return files

    def process(self):

        pkl_hit_paths = [self.pkl_file]
        print('reading:', pkl_hit_paths)
        idx = 0
        file_count = 0
        for pkl_hit_file in pkl_hit_paths:
            with gzip.open(pkl_hit_file, 'rb') as f:
                all_events_flattened = pickle.load(f)

            for i, evt in enumerate(all_events_flattened):
                hits_staton = torch.tensor(evt["hits_staton"], dtype=torch.int).unsqueeze(-1)  # (n_hits, 1)
                hits_pos = torch.tensor(evt["hits_pos"], dtype=torch.float32)                  # (n_hits, 3)
                hits_label = torch.tensor(evt["hits_label"], dtype=torch.int)                 # (n_hits,)

                x = torch.cat([hits_staton, hits_pos], dim=1)  # shape: (n_hits, 4)
                y = hits_label
                ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]])
                n_hits = x.shape[0]

                data = Data(x=x, y=y, ids=ids, n_hits=n_hits)

                if self.use_event_feature:
                    data.event_feature = torch.tensor(1.0)  # Placeholder

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
                idx += 1

                if idx>10:  #debug
                    break
            file_count += 1 

    def len(self):
        return 4

    def get(self, idx):
        filename = f'{self.split}_data_{idx}.pt'
        data_path = os.path.join(self.processed_dir, filename)
        return torch.load(data_path)


class TrainGeoDataset_3D(Dataset):
    def __init__(self, root, metadata_dir, split, use_event_feature=False, weight_type=None,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        self.split = split
        self.metadata_dir = metadata_dir
        self.use_event_feature = use_event_feature
        self.weight_type = weight_type
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        files = [f for f in os.listdir(self.processed_dir) if f.startswith(f"{self.split}_data_")]
        return files

    def process(self):
        # 1. Load metadata
        metadata_files = glob.glob(os.path.join(self.metadata_dir, "MC_neutrino_volTarget_100fb-1_metadata.csv"))
        df = pd.concat((pd.read_csv(file) for file in metadata_files), ignore_index=True)

        # 2. Filter by split
        df = df[df['split_3d_1'] == self.split]
    
        # 3. Check file existence
        df['file_exists'] = df['pkl_hit_path'].apply(os.path.isfile)

        # Print dropped and kept files
        dropped_files = df[~df['file_exists']]['pkl_hit_path'].tolist()
        kept_files = df[df['file_exists']]['pkl_hit_path'].tolist()

        print(f"\nKept {self.split} {len(kept_files)} files:")
        for f in kept_files:
            print(f"  ✓ {f}")

        print(f"\nDropped {self.split} {len(dropped_files)} files (not found):")
        for f in dropped_files:
            print(f"  ✗ {f}")

        # 4. Use only existing files
        df = df[df['file_exists']]
        pkl_hit_paths = df['pkl_hit_path'].tolist()



        idx = 0
        file_count = 0
        for pkl_hit_file in pkl_hit_paths:
            with gzip.open(pkl_hit_file, 'rb') as f:
                all_events_flattened = pickle.load(f)

            for i, evt in enumerate(all_events_flattened):
                hits_staton = torch.tensor(evt["hits_staton"], dtype=torch.int).unsqueeze(-1)  # (n_hits, 1)
                hits_pos = torch.tensor(evt["hits_pos"], dtype=torch.float32)                  # (n_hits, 3)
                hits_label = torch.tensor(evt["hits_label"], dtype=torch.int)                 # (n_hits,)

                x = torch.cat([hits_staton, hits_pos], dim=1)  # shape: (n_hits, 4)
                y = hits_label
                ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]])
                n_hits = x.shape[0]

                data = Data(x=x, y=y, ids=ids, n_hits=n_hits)

                if self.use_event_feature:
                    data.event_feature = torch.tensor(1.0)  # Placeholder

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
                idx += 1
             # debug
            if (file_count>4 and self.split == 'train'):
                break
            elif (file_count>2 and self.split == 'valid'):
                break
            file_count += 1 

    def len(self):
        if self.split == 'train':
            return 2000
        elif self.split == 'valid':
            return 500
        #return len(self.processed_file_names)

    def get(self, idx):
        filename = f'{self.split}_data_{idx}.pt'
        data_path = os.path.join(self.processed_dir, filename)
        return torch.load(data_path)