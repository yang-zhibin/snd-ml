import uproot
import pandas as pd
import torch
import numpy as np
import glob
from torch_geometric.data import Dataset, Data



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
particle_mapping = {
    12: 've', -12: 've',
    14: 'vm', -14: 'vm',
    16: 'vt', -16: 'vt',
    112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC', 116: 'NC', -116: 'NC',
    130: 'kaon', 310: 'kaon',
    2112: 'neutron',
    13:'muon', -13:'muon',
    0:'data'
}


def read_csv_from_dir(directory):
    
    csv_files = glob.glob(f"{directory}/test*.csv")  # Get all CSV files in the directory
    print("csv_files:",csv_files)
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the given directory.")

    dfs = [pd.read_csv(file) for file in csv_files]  # Read all CSVs into DataFrames
    return pd.concat(dfs, ignore_index=True)  # Merge them into one DataFrame

# Example usage:
# df = read_csv_from_dir("path/to/folder")
# print(df.head())



class ROOTGraphDataset(Dataset):
    def __init__(self, metadata_dir, split, use_event_feature, weight_type, transform=None, pre_transform=None):
        """
        Args:
            metadata_dir (str): Path to the directory containing metadata CSV.
            split (str): "train", "val", or "test".
        """

        self.metadata = read_csv_from_dir(metadata_dir)
        self.metadata = self.metadata[self.metadata["split"] == split]  # Filter by split
        self.hit_paths = self.metadata["hit_path"].tolist()
        self.feature_paths = self.metadata["feature_path"].tolist()  

        self.use_event_feature = use_event_feature
        self.weight_type = weight_type

        # Store index mapping: (hit file path, feature file path, event index)
        self.graph_list = []
        for file_idx, (hit_path, feature_path) in enumerate(zip(self.hit_paths, self.feature_paths)):
            with uproot.open(hit_path) as f:
                tree = f["snddata"]  # Ensure correct tree name
                num_events = tree["Id/eventId"].num_entries  # Count events
                for event_idx in range(num_events):
                    self.graph_list.append((hit_path, feature_path, event_idx))  

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        """Load a specific event's graph from the corresponding ROOT file."""
        hit_path, feature_path, event_idx = self.graph_list[idx]
        
        with uproot.open(hit_path) as f:
            tree = f["snddata"]
            
            pdg_code = tree["Id/pdgCode"].array()[event_idx]
            target = particle_to_target[pdg_code]

            # Extract the available branches correctly
            hit_branches = [
                "Hits/Hits.x1", "Hits/Hits.y1", "Hits/Hits.z1",
                "Hits/Hits.x2", "Hits/Hits.y2", "Hits/Hits.z2",
                "Hits/Hits.hitTime", "Hits/Hits.detType"
            ]

            try:
                hit_data = tree.arrays(hit_branches, library="np")

                x1 = hit_data["Hits/Hits.x1"][event_idx]
                y1 = hit_data["Hits/Hits.y1"][event_idx]
                z1 = hit_data["Hits/Hits.z1"][event_idx]
                x2 = hit_data["Hits/Hits.x2"][event_idx]
                y2 = hit_data["Hits/Hits.y2"][event_idx]
                z2 = hit_data["Hits/Hits.z2"][event_idx]
                hit_time = hit_data["Hits/Hits.hitTime"][event_idx]
                det_type = hit_data["Hits/Hits.detType"][event_idx]

            except KeyError:
                raise KeyError(f"Expected keys not found in {hit_path}. Available keys: {tree.keys()}")

        # Stack features into node attributes
        features = torch.tensor(np.column_stack([x1, y1, z1, x2, y2, z2, hit_time, det_type]), dtype=torch.float)
        data = Data(x=features, y=target, event_feature=0, weights=1, id=event_idx)
        return data
