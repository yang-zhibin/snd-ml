from pytorch_lightning import Callback
import uproot
import awkward as ak
import torch

class RootSaver(Callback):
    def __init__(self, pt_file, model_name, out_path):
        super().__init__()
        self.out_path = out_path
        self.model_name = model_name
        self.data = []
        self.data_column_names = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):   
        outputs=outputs['outputs']
        #print(f"output shape:", outputs.shape)
        predictions = torch.softmax(outputs,dim=1)
        predictions = predictions.detach().cpu().numpy().squeeze()
        
        # Extract runId and eventId from the batch
        ids = batch.ids

        #ToDo, fix output bug
        #if (outputs.shape[0]<2):
        #    return

        for i in range(predictions.shape[0]):  # Loop over the batch dimension
            pdg_code, run_id, event_id  = ids[i]
            prediction_list = predictions[i].tolist() if predictions.ndim > 1 else [predictions[i]]
            data_entry = prediction_list + [pdg_code, run_id, event_id]
            self.data.append(data_entry)

        self.data_column_names = [f'Prediction_{i}' for i in range(len(predictions[0]))] + ['PdgCode', 'RunId', 'EventId']


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

        

        with uproot.recreate(self.out_path) as root_file:
            root_file["snddata"] = {key: ak_array[key] for key in ak_array.fields}
        
        print(f"Predictions saved to '{self.out_path}'.")
        # Optionally clear the list to save memory
        self.data.clear()