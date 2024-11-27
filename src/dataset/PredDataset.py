import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import pandas as pd


particle_to_target = {
        12: 0, -12: 0,
        14: 1, -14: 1,
        16: 2, -16: 2,
        112: 3, -112: 3, 114: 3, -114: 3, 116: 3, -116: 3,
        130: 4, 310: 4,
        2112: 5,
        13:6, -13:6
}
particle_mapping = {
    12: 've', -12: 've',
    14: 'vm', -14: 'vm',
    16: 'vt', -16: 'vt',
    112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC', 116: 'NC', -116: 'NC',
    130: 'kaon', 310: 'kaon0',
    2112: 'neutron',
    13:'muon', -13:'muon'
}


def find_partition_weight(search_string, dataframe):
    # Iterate through each row in the DataFrame
    for index, row in dataframe.iterrows():
        #print(row)
        # Check if the 'partition' string is in the search_string
        if row['partition'] in search_string:
            # If found, return the corresponding weight
            return row['partition'], row['normalized_weight']
    # If no partition is found in the string, return None
    return None, 0.1

class PredDataset(InMemoryDataset):
    def __init__(self, root, save_dir,chunk, chunk_id, partition, split='pred', signal=None, weight=None, use_event_feature=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.signal = signal
        self.save_dir = save_dir
        self.partition = partition
        self.chunk =chunk
        self.chunk_id =chunk_id
        self.use_event_feature = use_event_feature
        self.weight = pd.read_csv('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/dataset/trianing_weight.csv')
        self.path = os.path.join(save_dir, '{}_{}_{}_data.pt'.format(self.split, partition, chunk_id))
        #print(self.path)
        # force_reload is not working, manually force rm old processed data
        super().__init__(root, transform, pre_transform, pre_filter)

        print("reading input",self.processed_paths[0])
        self.load(self.processed_paths[0])

        

    @property
    def raw_file_names(self):
        return self.chunk

    @property
    def processed_file_names(self):
        #return [path]
        return [self.path]
    
    def process(self):
        all_event = []
        for file in self.raw_file_names:
            print(file)
            
            #print(partition, weight)
            events = torch.load(file)
            for evt in events:
                #print(f"-------{evt}--------")
                hit_feature = evt.hitFeature
                event_feature = evt.eventFeatures
                pdgCode = evt.pdgCode
                runId = evt.runId
                eventId = evt.eventId
                
                particle = particle_mapping[pdgCode]
                event_weight = 1
                
                #prepare label
                if (self.signal == 'vm'):
                    if (abs(pdgCode)==14):
                        y = 1
                    else:
                        y = 0

                elif(self.signal == 've'):
                    if (abs(pdgCode)==12):
                        y = 1
                    else:
                        y = 0

                else:
                    y = particle_to_target[pdgCode]
                
                hit_feature = hit_feature.T
                event_feature = event_feature.T

                #print(particle, weight,event_weight)

                # if (self.use_event_feature is True):
                #     n_hits = hit_feature.shape[0]
                #     n_event_features = event_feature.shape[1]
                #     expaned_evt_feature = event_feature.expand(n_hits, n_event_features)
                #     hit_feature = torch.cat((hit_feature, expaned_evt_feature), dim=1)

                

                all_event.append(Data(x=hit_feature, event_feature=event_feature,  weights = torch.tensor(event_weight), y=torch.tensor(y), ids =[pdgCode, runId, eventId] ))
        #print(all_event.shape)

        self.save(all_event, self.processed_paths[0])
       
    
