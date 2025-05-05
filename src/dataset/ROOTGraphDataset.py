import uproot
import pandas as pd
import torch
import numpy as np
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

class ROOTGraphDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None, pre_transform=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            split (str): "train", "val", or "test".
        """
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata[self.metadata["split"] == split]  # Filter by split
        self.file_paths = self.metadata["input_path"].tolist()

        # Store index mapping: (file index, event index) for fast access
        self.graph_list = []
        for file_idx, file_path in enumerate(self.file_paths):
            with uproot.open(f"{file_path}:your_tree_name") as tree:  # Change tree name
                num_events = tree["Id/eventId"].num_entries  # Count events
                for event_idx in range(num_events):
                    self.graph_list.append((file_path, event_idx))  # Store (file path, event index)

        super().__init__(transform, pre_transform)

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        """Load a specific event's graph from the corresponding ROOT file."""
        file_path, event_idx = self.graph_list[idx]

        with uproot.open(f"{file_path}:your_tree_name") as tree:  # Change tree name
            # Load event-level data
            pdg_code = tree["Id/pdgCode"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]

            # Load hit-level data
            x1 = tree["Hits/x1"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            y1 = tree["Hits/y1"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            z1 = tree["Hits/z1"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            x2 = tree["Hits/x2"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            y2 = tree["Hits/y2"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            z2 = tree["Hits/z2"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            hit_time = tree["Hits/hitTime"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]
            det_type = tree["Hits/detType"].array(entry_start=event_idx, entry_stop=event_idx+1)[0]

        # Stack features into node attributes
        node_features = torch.tensor(np.column_stack([x1, y1, z1, x2, y2, z2, hit_time, det_type]), dtype=torch.float)
        num_nodes = node_features.shape[0]

        # Simple edge connections (can be improved)
        edge_index = torch.tensor([list(range(num_nodes-1)), list(range(1, num_nodes))], dtype=torch.long)

        # Create a PyG Data object
        return Data(x=node_features, edge_index=edge_index, y=torch.tensor([pdg_code], dtype=torch.long))
