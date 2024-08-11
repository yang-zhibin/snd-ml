import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import pandas as pd
import numpy as np
from pytorch_lightning import Callback
import awkward as ak
import uproot


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
    130: 'kaon', 310: 'kaon',
    2112: 'neutron',
    13:'muon', -13:'muon'
}

class RootSaver(Callback):
    def __init__(self, out_path, input_file):
        super().__init__()
        self.out_path = out_path
        self.input_file = input_file
        self.data = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):   
        # Convert prediction tensor to numpy and squeeze unnecessary dimensions
        outputs=outputs['outputs']
        predictions = torch.sigmoid(outputs)
        #print("pred",predictions.shape)
        predictions = predictions.detach().cpu().numpy().squeeze()
        
        # Extract runId and eventId from the batch
        # Assuming the IDs are passed as a tuple or list with the batch, accessible via `batch.ids`
        ids = batch.ids
        event_feature = batch.event_feature
        label = batch.label


        #print(outputs.shape,outputs.squeee().shape, predictions.size)
        #print(outputs)
        #print(outputs.squeeze())
        # Store each prediction with its corresponding runId and eventId
        

        #ToDo, fix output bug
        if (outputs.shape[0]<2):
            pass
        else:
            for i in range(predictions.shape[0]):  # Loop over the batch dimension
                particle, pdg_code, run_id, event_id, file_id  = ids[i]
                px,py,pz,pos_x,pos_y,pos_z, stage2= label[i]
                prediction_list = predictions[i].tolist() if predictions.ndim > 1 else [predictions[i]]
                data_entry = prediction_list + [particle, pdg_code, run_id, event_id, file_id] + [px,py,pz,pos_x,pos_y, pos_z,stage2]
                # Store all predictions for the current instance together with its identifiers
                self.data.append(data_entry)

    def on_test_epoch_end(self, trainer, pl_module):
        #print(self.data)
        columns = ['Prediction_{}'.format(i) for i in range(len(self.data[0]) - 12)] + ['particle','PdgCode', 'RunId', 'EventId', "FileId",'px', 'py', 'pz', 'pos_x', 'pos_y', 'pos_z','stage2']
        data_dict = {col: [] for col in columns}
        
        # Populate the dictionary
        for entry in self.data:
            for col, value in zip(columns, entry):
                data_dict[col].append(value)
        
        # Convert the dictionary to an Awkward Array
        ak_array = ak.Array(data_dict)

        
        file_name_with_ext = os.path.basename(self.input_file)
        file_name, _ = os.path.splitext(file_name_with_ext)
        # Save to ROOT file
        pred_path = f"{self.out_path}/{file_name}_output.root"
        with uproot.recreate(pred_path) as root_file:
            root_file["tree"] = {key: ak_array[key] for key in ak_array.fields}
        
        print(f"Predictions saved to '{pred_path}'.")
        # Optionally clear the list to save memory
        self.data.clear()

class SndGeoDataset(InMemoryDataset):
    def __init__(self, root, weight_type, split='train' ,file_path=None , use_event_feature=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.use_event_feature = use_event_feature
        self.file_path =file_path
        self.root = root
        self.weight_type = weight_type

        print("In dataset process",split, file_path)
        if file_path == None:
            self.path = os.path.join(root, 'input/{}_input.pt'.format(self.split))
        else:
            # for condor
            self.path = './input/{}'.format(os.path.basename(file_path))
            #self.path = os.path.join(root, 'input/{}'.format(os.path.basename(file_path)))
        # force_reload is not working, manually force rm old processed data
        
        if (force_reload):
            try:
                os.remove(self.path)
            except OSError:
                pass
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        if (self.split == 'train') or (self.split == 'val'):
            list_files = [os.path.join(self.root,filename) for filename in os.listdir(self.root) if filename.startswith(self.split)]
        else:
            list_files = [self.file_path]

        print(list_files)
        return list_files

    @property
    def processed_file_names(self):
        return [self.path]
    
    def process(self):
        all_event = []
        for file in self.raw_file_names:
            print(file)
            
            events = torch.load(file)
            for evt in events:
                #print(f"-------{evt}--------")
                
                #print(evt)
                pdgCode = evt.pdgCode
                y = particle_to_target[pdgCode]
                particle = particle_mapping[pdgCode]

                runId = evt.runId
                eventId = evt.eventId
                fileId = evt.fileId
                
                if (self.weight_type == 'weight'):
                    event_weight = evt.weight
                elif(self.weight_type == 'normalized_weight'):
                    event_weight = evt.normalized_weight
                elif(self.weight_type == 'intRate_weight'):
                    event_weight = evt.intRate_weight
                elif(self.weight_type == 'intRate_weightX100'):
                    event_weight = evt.intRate_weightX100
                elif(self.weight_type == 'intRate_weightX100^2'):
                    print('testing condor ')
                    event_weight = (evt.weight * 100) * (evt.weight * 100)


                px=evt.px
                py=evt.py
                pz=evt.pz
                pos_x =evt.x
                pos_y =evt.y
                pos_z =evt.z
                stage2 = evt.stage2
                
                hit_feature = evt.hitFeature
                event_feature = evt.eventFeatures
                hit_feature = hit_feature.T
                event_feature = event_feature.T      

                all_event.append(Data(x=hit_feature, event_feature=event_feature,  weights = torch.tensor(event_weight), y=torch.tensor(y), ids =[particle,pdgCode, runId, eventId, fileId],label=[px,py,pz,pos_x,pos_y,pos_z,stage2] ))
        #print(all_event.shape)

        self.save(all_event, self.processed_paths[0])
       
    
