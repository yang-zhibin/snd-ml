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

            # Select only the columns specified by selected_veto_hit_columns
            if self.selected_veto_hit_columns:
                veto_hit_feature = torch.stack([evt["vetoHitFeature"][col] for col in self.selected_veto_hit_columns]).squeeze()
            else:
                veto_hit_feature = torch.stack([evt["vetoHitFeature"][col] for col in evt["vetoHitFeature"]]).squeeze()

            # Ensure veto_hit_feature is 2D with shape (features, N)
            if veto_hit_feature.ndim == 1:
                veto_hit_feature = veto_hit_feature.unsqueeze(-1)
            veto_hit_feature = veto_hit_feature.mT

            # Combine hit and veto hit features if required
            if use_veto_hits:
                x = torch.cat([hit_feature, veto_hit_feature], dim=0)
            else:
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



def read_pt_path_and_weight(split, metadata_dir, model_name):
    # Load the CSV files
    neutrino_df = pd.read_csv(os.path.join(metadata_dir, f'{model_name}_neutrino_split.csv'))
    neutral_bkg_df = pd.read_csv(os.path.join(metadata_dir, f'{model_name}_neutral_bkg_split.csv'))
    muon_bkg_df = pd.read_csv(os.path.join(metadata_dir, f'{model_name}_muon_bkg_split.csv'))

    # Filter and select relevant columns, renaming pt_hit_path for consistency
    neutrino = neutrino_df[neutrino_df['split'] == split][['vetoFree_pt_hit_path', 'weight']].rename(
        columns={'vetoFree_pt_hit_path': 'pt_hit_path'}
    )
    neutral_bkg = neutral_bkg_df[neutral_bkg_df['split'] == split][['vetoFree_pt_hit_path', 'weight']].rename(
        columns={'vetoFree_pt_hit_path': 'pt_hit_path'}
    )
    muon_bkg_vetoFree = muon_bkg_df[muon_bkg_df['vetoFree_split'] == split][['vetoFree_pt_hit_path', 'weight']].rename(
        columns={'vetoFree_pt_hit_path': 'pt_hit_path'}
    )
    muon_bkg_vetoTagged = muon_bkg_df[muon_bkg_df['vetoTagged_split'] == split][['vetoTagged_pt_hit_path', 'weight']].rename(
        columns={'vetoTagged_pt_hit_path': 'pt_hit_path'}
    )

    # Concatenate all together
    combined_df = pd.concat([neutrino, neutral_bkg, muon_bkg_vetoFree, muon_bkg_vetoTagged], ignore_index=True)
    # check pt_hit_path file exist, drop if it doesn't exist
    combined_df = combined_df[combined_df['pt_hit_path'].apply(os.path.isfile)].reset_index(drop=True)
    
    #print(combined_df)
    return combined_df

class TrainGeoDataset(InMemoryDataset):
    def __init__(self, root, metadata_dir, model_name, split, 
                 use_veto_hits=False, use_event_feature=False, weight_type=None,
                 selected_hit_columns=None, selected_veto_hit_columns=None, selected_event_columns=None,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        
       
        self.root = root
        self.metadata_dir = metadata_dir
        self.model_name = model_name
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
        return [f'{self.split}.pt']

    def process(self):
        use_veto_hits = self.use_veto_hits
        #read pt hit paths and weight
        pt_paths = read_pt_path_and_weight(self.split, self.metadata_dir, self.model_name)

        all_event = []
        #loop over the pt_paths
        # 
        for idx, row in pt_paths.iterrows():
            pt_file = row["pt_hit_path"]
            weight = row["weight"]
        
            with gzip.open(pt_file, 'rb') as f:
                all_events_flattened = torch.load(f)
            
            
            for i, evt in enumerate(all_events_flattened):
                # Select only the columns specified by selected_hit_columns
                if self.selected_hit_columns:
                    hit_feature = torch.stack([evt["hitFeature"][col] for col in self.selected_hit_columns]).squeeze().mT
                else:
                    hit_feature = torch.stack([evt["hitFeature"][col] for col in evt["hitFeature"]]).squeeze().mT

                # Select only the columns specified by selected_veto_hit_columns
                if self.selected_veto_hit_columns:
                    veto_hit_feature = torch.stack([evt["vetoHitFeature"][col] for col in self.selected_veto_hit_columns]).squeeze()
                else:
                    veto_hit_feature = torch.stack([evt["vetoHitFeature"][col] for col in evt["vetoHitFeature"]]).squeeze()

                # Ensure veto_hit_feature is 2D with shape (features, N)
                if veto_hit_feature.ndim == 1:
                    veto_hit_feature = veto_hit_feature.unsqueeze(-1)
                veto_hit_feature = veto_hit_feature.mT

                # Combine hit and veto hit features if required
                if use_veto_hits:
                    x = torch.cat([hit_feature, veto_hit_feature], dim=0)
                else:
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
        
