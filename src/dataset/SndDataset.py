import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import uproot
import pandas as pd
import numpy as np

def pad_point_cloud(point_cloud, feature_len, max_points=4096):
    """Pad the point cloud to max_points with zero-vectors."""
    #print(point_cloud.dtype)
    padding = max_points - point_cloud.shape[1]
    if padding > 0:
        pad_tensor = torch.zeros((feature_len, padding), dtype=point_cloud.dtype)
        point_cloud = torch.cat([point_cloud, pad_tensor], dim=1)
    else:
        point_cloud = point_cloud[:,:max_points]
    return point_cloud

def preprocess():
    pass

class SndDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with directory paths and the number of images in each.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = pd.read_csv(file_list, header=None)  # Assuming no header
        self.transform = transform
        self.cumulative_counts = self.file_list.iloc[:, 1].cumsum()  # Cumulative sum of images counts

    def __len__(self):
        # The length of the dataset is the sum of the second column (total number of images)
        return self.file_list.iloc[:, 1].sum()

    def __getitem__(self, idx):
        # Find the correct folder and image index
        file_idx = self.cumulative_counts.searchsorted(idx + 1)
        #print("f_idx",file_idx)
        if file_idx == 0:
            evt_idx = idx
        else:
            evt_idx = idx - self.cumulative_counts[file_idx - 1]

        #print("e_idx", evt_idx)

        file_path = self.file_list.iloc[file_idx, 0]
       # print(file_path)

        with uproot.open(file_path) as Rfile:
            tree = Rfile['cbmsim']
            event_data = tree.arrays(entry_start=evt_idx, entry_stop=evt_idx + 1)

        #print(event_data.fields)
        # pdgCode -> y
        pdgCode = event_data['pdgCode'][0]
        signal = 'vm'
        if (signal == 'vm'):
            if (abs(pdgCode)==14):
                y = 1
            else:
                y = 0
        y=torch.tensor([y])

        # id ('runId', 'eventId')
        ids = [event_data['runId'][0],event_data['eventId'][0]]

        # features
        
        hits_features_name = ['Hits.detType','Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
        event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
        features = []
        for feature in hits_features_name:
            features.append(np.squeeze(event_data[feature].to_numpy()))
        
        pad_features = pad_point_cloud(torch.Tensor(np.array(features)), len(hits_features_name))
        

        use_event_feature = False
        if (use_event_feature):
            # Create a combined list of all feature names
            all_features_name = hits_features_name + event_features_name

            # Create a list of tensors from the event_data dictionary using a list comprehension for all features
            tensors = [torch.tensor(event_data[feature], dtype=torch.float32) for feature in all_features_name]
        else:
            tensors = [torch.tensor(event_data[feature], dtype=torch.float32) for feature in hits_features_name]

        x = torch.squeeze(torch.stack(tensors)).T

        #print("stack x", x.shape)
        #print("pad x",pad_features.T.shape)


        return Data(x=x, y=y, ids=ids)


# Example usage
if __name__ == "__main__":
    pass