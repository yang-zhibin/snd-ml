import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import pandas as pd
import numpy as np
from pytorch_lightning import Callback
import awkward as ak
import uproot

#print('check import ROOT')
#import ROOT


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

class RootSaver(Callback):
    def __init__(self, out_dir, input_file, model_name):
        super().__init__()
        self.out_dir = out_dir
        self.input_file = input_file
        self.model_name = model_name
        self.data = []
        self.data_column_names = []

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

        #ToDo, fix output bug
        if (outputs.shape[0]<2):
            return
        
        for i in range(predictions.shape[0]):  # Loop over the batch dimension
            particle, pdg_code, run_id, event_id, file_id  = ids[i]
            prediction_list = predictions[i].tolist() if predictions.ndim > 1 else [predictions[i]]
            data_entry = prediction_list + [particle, pdg_code, run_id, event_id, file_id]
            # Store all predictions for the current instance together with its identifiers
            self.data.append(data_entry)

        self.data_column_names = [f'Prediction_{i}' for i in range(len(predictions[0]))] + ['particle', 'PdgCode', 'RunId', 'EventId', 'FileId']


    def on_test_epoch_end(self, trainer, pl_module):
        if not self.data:
            print("No data to save.")
            return

        #print(self.data)
        columns = self.data_column_names
        data_dict = {col: [] for col in columns}
        
        # Populate the dictionary
        for entry in self.data:
            for col, value in zip(columns, entry):
                data_dict[col].append(value)
        
        # Convert the dictionary to an Awkward Array
        ak_array = ak.Array(data_dict)

        
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        # Save to ROOT file
        pred_path = f"{self.out_dir}/output_{self.model_name}_{base_name}.root"
        with uproot.recreate(pred_path) as root_file:
            root_file["tree"] = {key: ak_array[key] for key in ak_array.fields}
        
        print(f"Predictions saved to '{pred_path}'.")
        # Optionally clear the list to save memory
        self.data.clear()

class SndGeoDataset(InMemoryDataset):
    def __init__(self, root, raw_file, weight_type , use_event_feature=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.use_event_feature = use_event_feature
        self.root = root
        self.weight_type = weight_type
        self.raw_file = raw_file

        base_name = os.path.splitext(os.path.basename(raw_file))[0]
        self.path = os.path.join(root, 'processed_input_{}.pt'.format(base_name))

        #force_reload is not working, manually force rm old processed data
        if (force_reload):
            try:
                os.remove(self.path)
            except OSError:
                pass
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.path)

    @property
    def raw_file_names(self):
        print("raw_file_names:",self.raw_file)
        return [self.raw_file]

    @property
    def processed_file_names(self):
        print('processed_file_names',[self.path])
        return [self.path]
    
    def process(self):
        # Load all files in one go
        all_events_raw = [torch.load(file) for file in self.raw_file_names]
        all_events_flattened = [evt for events in all_events_raw for evt in events]

        # Map the weight calculation for each weight type
        weight_mapping = {
            'weight': lambda evt: evt.weight,
            'normalized_weight': lambda evt: evt.normalized_weight,
            'intRate_weight': lambda evt: evt.intRate_weight,
            'intRate_weightX100': lambda evt: evt.intRate_weight * 100,
            'intRate_weightX100^2': lambda evt: (evt.weight * 100) ** 2,
        }

        # Use the appropriate weight calculation function based on weight_type
        calculate_weight = weight_mapping.get(self.weight_type, lambda evt: None)

        # Vectorized processing
        all_event = [
            Data(
                # assign zero tensor for no hits event, otherwise raise an error 
                x=evt.hitFeature.T if evt.hitFeature.numel() > 0 else torch.zeros((evt.hitFeature.shape[0],1)).T, 
                event_feature=evt.eventFeatures.T,
                weights=torch.tensor(calculate_weight(evt)),
                y=torch.tensor(particle_to_target[evt.pdgCode]),
                ids=[
                    particle_mapping[evt.pdgCode],
                    evt.pdgCode,
                    evt.runId,
                    evt.eventId,
                    evt.fileId,
                ],
            )
            for evt in all_events_flattened
        ]


        print('in process', self.processed_paths[0])
        self.save(all_event, self.path)


class SndGeoDatasetTest(InMemoryDataset):
    def __init__(self, root, raw_file, weight_type , use_event_feature=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.raw_file = raw_file
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.raw_file)


    @property
    def processed_file_names(self):
        print('processed_file_names',[self.path])
        return [self.raw_file]
    
