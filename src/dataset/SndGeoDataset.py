import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os

class SndGeoDataset(InMemoryDataset):
    def __init__(self, root, split='train', signal='vm', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.signal = signal
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        list_files = [os.path.join(self.root,filename) for filename in os.listdir(self.root) if filename.startswith(self.split)]
        #print(list_files)
        return list_files

    @property
    def processed_file_names(self):
        path = os.path.join(self.root, '{}_data.pt'.format(self.split))

        #return [path]
        return [path]

    def process(self):
        all_event = []
        for file in self.raw_file_names:
            events = torch.load(file)
            for evt in events:
                #print(evt)
                hit_feature = evt.hitFeature
                event_feature = evt.eventFeatures
                pdgCode = evt.pdgCode
                runId = evt.runId
                eventId = evt.eventId

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
                y=torch.tensor(y)
                
                hit_feature = hit_feature.T
                event_feature = event_feature.T
                

                all_event.append(Data(x=hit_feature, x_e=event_feature, y=y, ids =[pdgCode, runId, eventId] ))
        #print(all_event.shape)

        self.save(all_event, self.processed_paths[0])
       

