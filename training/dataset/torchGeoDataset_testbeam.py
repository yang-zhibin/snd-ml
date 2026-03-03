import os
import pandas as pd
import glob
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch
import gzip

particle_to_target = {
        11: 0, -11: 0,
        211: 1, -211: 1,
        

}

class PredStreamingDataset(Dataset):
    def __init__(self, pt_file, selected_hit_columns=None, use_event_feature=False, selected_event_columns=None):
        with gzip.open(pt_file, "rb") as f:
            self.events = torch.load(f, map_location="cpu")

        self.selected_hit_columns = selected_hit_columns or []
        self.use_event_feature = use_event_feature
        self.selected_event_columns = selected_event_columns or []

        # infer hit columns from first event if not provided
        if not self.selected_hit_columns:
            self.selected_hit_columns = list(self.events[0]["hitFeature"].keys())
        self.n_feat = len(self.selected_hit_columns) if self.selected_hit_columns else 1
        self.dummy_x = torch.zeros((1, self.n_feat), dtype=torch.float)

        if self.use_event_feature and self.selected_event_columns:
            self.n_evt = len(self.selected_event_columns)
            self.dummy_event = torch.zeros((1, self.n_evt), dtype=torch.float)
        else:
            self.dummy_event = torch.zeros((1, 1), dtype=torch.float)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, i):
        evt = self.events[i]
        hit = evt["hitFeature"]
        cols = self.selected_hit_columns

        t0 = hit[cols[0]]
        if not torch.is_tensor(t0):
            t0 = torch.as_tensor(t0)
        t0 = t0.reshape(-1)
        n_hits = t0.numel()

        if n_hits == 0:
            x = self.dummy_x
            is_dummy = 1
        else:
            feat_cols = [t0]
            for c in cols[1:]:
                t = hit[c]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                feat_cols.append(t.reshape(-1))
            x = torch.stack(feat_cols, dim=1).to(torch.float)
            is_dummy = 0

        if self.use_event_feature and self.selected_event_columns:
            ev = evt["eventFeatures"]
            ef = []
            for c in self.selected_event_columns:
                t = ev[c]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                ef.append(t.reshape(1))
            event_feature = torch.cat(ef, dim=0).unsqueeze(0).to(torch.float)
        else:
            event_feature = self.dummy_event

        y = torch.tensor([particle_to_target.get(evt["pdgCode"])], dtype=torch.long)
        ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]], dtype=torch.long).unsqueeze(0)

        data = Data(
            x=x,
            event_feature=event_feature,
            weights=torch.tensor([1.0], dtype=torch.float),
            y=y,
            ids=ids,
        )
        data.is_dummy = torch.tensor([is_dummy], dtype=torch.long)
        return data
    
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
        with gzip.open(self.pt_file, "rb") as f:
            all_events_flattened = torch.load(f, map_location="cpu")

        all_event = []

        # Precompute columns once (assumes all events share the same feature keys)
        # If keys can differ, keep the per-event fallback below.
        first_evt = all_events_flattened[0]
        hit_dict0 = first_evt["hitFeature"]
        cols = self.selected_hit_columns if self.selected_hit_columns else list(hit_dict0.keys())
        n_feat = len(cols) if len(cols) > 0 else 1

        # Pre-allocate a dummy node feature (1 node, n_feat features)
        dummy_x = torch.zeros((1, n_feat), dtype=torch.float)

        # Event feature handling
        use_evt = bool(self.use_event_feature and self.selected_event_columns)
        if use_evt:
            n_evt_feat = len(self.selected_event_columns)
            dummy_event_feature = torch.zeros((1, n_evt_feat), dtype=torch.float)
        else:
            dummy_event_feature = torch.zeros((1, 1), dtype=torch.float)

        for evt in all_events_flattened:
            hit_dict = evt["hitFeature"]

            # If keys can differ per event, uncomment this fallback:
            # cols = self.selected_hit_columns if self.selected_hit_columns else list(hit_dict.keys())
            # n_feat = len(cols) if len(cols) > 0 else 1

            # Fast path: grab first column to get n_hits
            t0 = hit_dict[cols[0]]
            if not torch.is_tensor(t0):
                t0 = torch.as_tensor(t0)
            t0 = t0.reshape(-1)
            n_hits = t0.numel()

            if n_hits == 0:
                x = dummy_x  # already float, already 2-D
                is_dummy = 1
            else:
                # Build [n_hits, n_feat] by stacking columns as dim=1
                feat_cols = [t0]
                for c in cols[1:]:
                    t = hit_dict[c]
                    if not torch.is_tensor(t):
                        t = torch.as_tensor(t)
                    feat_cols.append(t.reshape(-1))
                x = torch.stack(feat_cols, dim=1).to(torch.float)
                is_dummy = 0

            # Event-level features
            if use_evt:
                # build [1, n_evt_feat]
                ef = []
                evdict = evt["eventFeatures"]
                for c in self.selected_event_columns:
                    t = evdict[c]
                    if not torch.is_tensor(t):
                        t = torch.as_tensor(t)
                    ef.append(t.reshape(1))
                event_feature = torch.cat(ef, dim=0).unsqueeze(0).to(torch.float)
            else:
                event_feature = dummy_event_feature

            y = torch.tensor([particle_to_target.get(evt["pdgCode"])], dtype=torch.long)
            ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]], dtype=torch.long).unsqueeze(0)

            data = Data(
                x=x,
                event_feature=event_feature,
                weights=torch.tensor([1.0], dtype=torch.float),
                y=y,
                ids=ids,
            )
            data.is_dummy = torch.tensor([is_dummy], dtype=torch.long)
            all_event.append(data)

        print(f"Processed data saved to {self.processed_paths[0]}")
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
                # --- Hit features ---
                hit_dict = evt["hitFeature"]
                cols = self.selected_hit_columns if self.selected_hit_columns else list(hit_dict.keys())

                feat_cols = []
                for c in cols:
                    t = torch.as_tensor(hit_dict[c])

                    # Make sure each feature is 1-D: [n_hits] (scalar -> [1])
                    t = t.reshape(-1)

                    feat_cols.append(t)

                # Handle empty events (optional but recommended)
                if len(feat_cols) == 0:
                    x = torch.empty((0, 0), dtype=torch.float)
                else:
                    # Sanity: all columns must have same n_hits
                    n_hits0 = feat_cols[0].numel()
                    for c, t in zip(cols, feat_cols):
                        if t.numel() != n_hits0:
                            raise RuntimeError(f"Column {c} has {t.numel()} hits, expected {n_hits0}")

                    # Stack columns into [n_hits, n_features]
                    x = torch.stack(feat_cols, dim=1).to(torch.float)
                # after x is created (shape [n_hits, n_features])
                if x.size(0) == 0:
                    continue  # skip events with no hits
                # Select event features if specified
                if self.use_event_feature and self.selected_event_columns:
                    ef = [torch.as_tensor(evt["eventFeatures"][c]).reshape(1) for c in self.selected_event_columns]
                    event_feature = torch.cat(ef, dim=0).unsqueeze(0).to(torch.float)  # [1, n_event_features]
                else:
                    event_feature = torch.empty((1, 0), dtype=torch.float)  # consistent 2-D

                # Target values (e.g., particle ID mapping)
                t = particle_to_target.get(evt["pdgCode"])
                if t is None:
                    print(f'event:{evt["pdgCode"]}, target return None')
                    continue  # or raise
                y = torch.tensor([t], dtype=torch.long)
                y = torch.tensor([particle_to_target.get(evt["pdgCode"])]) 
                
                # Event identifiers (pdgCode, runId, eventId)
                ids = torch.tensor([evt["pdgCode"], evt["runId"], evt["eventId"]], dtype=torch.long).unsqueeze(0)  # [1, 3]

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
        
