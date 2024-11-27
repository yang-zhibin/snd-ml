import os.path as osp
import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import uproot


class GeoSndDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def process(self):
        file_list = pd.read_csv(self.root, header=None)
        cumulative_counts = file_list.iloc[:, 1].cumsum()  
        # Find the correct folder and image index
        for idx in range(file_list.iloc[:, 1].sum()):
            file_idx = cumulative_counts.searchsorted(idx + 1)
            #print("f_idx",file_idx)
            if file_idx == 0:
                evt_idx = idx
            else:
                evt_idx = idx - cumulative_counts[file_idx - 1]

            #print("e_idx", evt_idx)

            file_path = file_list.iloc[file_idx, 0]
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

            # features
            
            hits_features_name = ['Hits.orientation','Hits.x1', 'Hits.y1', 'Hits.z1', 'Hits.x2', 'Hits.y2', 'Hits.z2']
            event_features_name = ['RecoMuon.px', 'RecoMuon.py', 'RecoMuon.pz', 'RecoMuon.x', 'RecoMuon.y', 'RecoMuon.z']
            features = []
            for feature in hits_features_name:
                features.append(np.squeeze(event_data[feature].to_numpy()))

            use_event_feature = False
            if (use_event_feature):
                for feature in event_features_name:
                    features.append(np.squeeze(event_data[feature].to_numpy))

            # id ('runId', 'eventId')
            ids = [event_data['runId'][0],event_data['eventId'][0]]
            
            #print(id)
            #print(y)
            #print(np.asarray(features,dtype="object").shape)
            #print(features)
            # print(np.asarray(event_features).shape)
            # print(event_features)


            # features  = np.stack(hit_features, event_features)
            # print(np.asarray(features).shape)
            # print(features)
            #print(features)

            #pad_features = pad_point_cloud(torch.Tensor(np.array(features)))
            #pad_x = torch.stack(pad_features)
            #print(pad_x.shape)


            features = [torch.tensor(feature, dtype=torch.float) for feature in features]
            x = torch.stack(features)
            #print("x:",x.shape)
            y=torch.tensor([y])
            #print("y",y.shape)

            torch.save(Data(x=x, y=y, ids=ids), osp.join(self.processed_dir, f'data_{idx}.pt') )  # Cumulative sum of images counts


    def len(self):
        file_list = pd.read_csv(self.root, header=None)
        return file_list.iloc[:, 1].sum()

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    