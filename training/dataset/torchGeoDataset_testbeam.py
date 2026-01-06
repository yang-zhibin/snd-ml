import os
import pandas as pd
import glob
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
import gzip

particle_to_target = {
        11: 0, -11: 0,
        211: 1, -211: 1,
        

}

class PredGeoDataset(InMemoryDataset):
    def __init__(self, root, pt_file, use_veto_hits=False, use_event_feature=False, weight_type=None,
                 selected_hit_columns=None, selected_veto_hit_columns=None, selected_event_columns=None,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.root = root
        self.pt_file = pt_file
        self.use_veto_hits = use_veto_hits
        self.use_event_feature = use_event_feature
        self.selected_hit_columns = selected_hit_columns if selected_hit_columns else []
        self.selected_veto_hit_columns = selected_veto_hit_columns if selected_veto_hit_columns else []
        self.selected_event_columns = selected_event_columns if selected_event_columns else []
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
        use_veto_hits = self.use_veto_hits

        with gzip.open(pt_file, 'rb') as f:
            all_events_flattened = torch.load(f)
        
        all_event = []
        for i, evt in enumerate(all_events_flattened):
            # Select only the columns specified by selected_hit_columns
            if self.selected_hit_columns:
                hit_feature = torch.stack([evt["hitFeature"][col] for col in self.selected_hit_columns]).squeeze().mT
            else:
                hit_feature = torch.stack([evt["hitFeature"][col] for col in evt["hitFeature"]]).squeeze().mT

            x = hit_feature

            # Select event features if specified
            if self.use_event_feature and self.selected_event_columns:
                event_feature = torch.stack([evt["eventFeatures"][col] for col in self.selected_event_columns]).unsqueeze(1).mT
            else:
                event_feature = torch.tensor([])  # Default empty tensor if no event columns are selected

            # Target values (e.g., particle ID mapping)

            y = torch.tensor([particle_to_target.get(evt["pdgCode"])]) 
            
            # Event identifiers (pdgCode, runId, eventId)
            ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]])
            ids = ids.unsqueeze(1).mT

            # Create Data object for each event and append to the list
            all_event.append(Data(x=x, event_feature=event_feature, weights=torch.tensor(1), y=y, ids=ids))

            # Print event shape details for debugging (optional)
            # print(f"Event {i}: x={x.shape}, y={y.shape}, ids={ids.shape}")

        print(f'Processed data saved to {self.processed_paths[0]}')
        self.save(all_event, self.processed_paths[0])



def read_pt_path_and_weight(split, metadata_dir, split_name):
    # Load the CSV files
    df = pd.read_csv(os.path.join(metadata_dir, f'{split_name}.csv'))

    return df[df["split"]==split]

class TrainGeoDataset(InMemoryDataset):
    def __init__(self, root, metadata_dir, split_name, split, 
                 use_veto_hits=False, use_event_feature=False, weight_type=None,
                 selected_hit_columns=None, selected_veto_hit_columns=None, selected_event_columns=None,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        
       
        self.root = root
        self.metadata_dir = metadata_dir
        self.split_name = split_name
        self.split = split
        
        self.use_veto_hits = use_veto_hits
        self.use_event_feature = use_event_feature
        self.selected_hit_columns = selected_hit_columns if selected_hit_columns else []
        self.selected_veto_hit_columns = selected_veto_hit_columns if selected_veto_hit_columns else []
        self.selected_event_columns = selected_event_columns if selected_event_columns else []
        
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.split}_{self.split_name}.pt']

    def process(self):
        use_veto_hits = self.use_veto_hits
        #read pt hit paths and weight
        pt_paths = read_pt_path_and_weight(self.split, self.metadata_dir, self.split_name)
        # print(pt_paths)
        all_event = []
        #loop over the pt_paths
        # 
        for idx, row in pt_paths.iterrows():
            pt_file = row["pt_hit_path"]
            weight = row['event_weight']
        
            with gzip.open(pt_file, 'rb') as f:
                all_events_flattened = torch.load(f)
            
            
            for i, evt in enumerate(all_events_flattened):
                # Select only the columns specified by selected_hit_columns
                if self.selected_hit_columns:
                    hit_feature = torch.stack([evt["hitFeature"][col] for col in self.selected_hit_columns]).squeeze().mT
                else:
                    hit_feature = torch.stack([evt["hitFeature"][col] for col in evt["hitFeature"]]).squeeze().mT

               

                x = hit_feature

                # Select event features if specified
                if self.use_event_feature and self.selected_event_columns:
                    event_feature = torch.stack([evt["eventFeatures"][col] for col in self.selected_event_columns]).unsqueeze(1).mT
                else:
                    event_feature = torch.tensor([])  # Default empty tensor if no event columns are selected

                # Target values (e.g., particle ID mapping)
                y = torch.tensor([particle_to_target.get(evt["pdgCode"])]) 
                
                # Event identifiers (pdgCode, runId, eventId)
                ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]])
                ids = ids.unsqueeze(1).mT

                # Create Data object for each event and append to the list
                all_event.append(Data(
                    x=x,
                    event_feature=event_feature,
                    weights=torch.tensor(weight, dtype=torch.float),
                    y=y,
                    ids=ids
                ))
        print(f'{self.split} dataset in process', self.processed_paths[0])
        self.save(all_event, self.processed_paths[0])
        
