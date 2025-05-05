import os
import pandas as pd
import glob
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
import gzip

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
