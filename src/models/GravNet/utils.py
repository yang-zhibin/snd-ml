import warnings

import torch
from torch import nn
import uproot

warnings.simplefilter(action='ignore', category=FutureWarning)
from pytorch_lightning import Callback
import pandas as pd
import awkward as ak

class PredictionRootSaver(Callback):
    def __init__(self, out_path, data_type):
        super().__init__()
        self.out_path = out_path
        self.data_type = data_type
        self.data = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):   
        # Convert prediction tensor to numpy and squeeze unnecessary dimensions
        predictions = torch.sigmoid(outputs)
       #print("pred",predictions.shape)
        predictions = predictions.detach().cpu().numpy().squeeze()
        
        # Extract runId and eventId from the batch
        # Assuming the IDs are passed as a tuple or list with the batch, accessible via `batch.ids`
        ids = batch.ids

        #print(outputs.shape,outputs.squeee().shape, predictions.size)
        #print(outputs)
        #print(outputs.squeeze())
        # Store each prediction with its corresponding runId and eventId
        
        #ToDd, fix onr output bug
        
        if (outputs.shape[0]<2):
            pass
        else:
            for i in range(predictions.shape[0]):  # Loop over the batch dimension
                pdg_code, run_id, event_id = ids[i]
                prediction_list = predictions[i].tolist() if predictions.ndim > 1 else [predictions[i]]
                data_entry = prediction_list + [pdg_code, run_id, event_id]
                # Store all predictions for the current instance together with its identifiers
                self.data.append(data_entry)

    def on_predict_epoch_end(self, trainer, pl_module):
        columns = ['Prediction_{}'.format(i) for i in range(len(self.data[0]) - 3)] + ['PdgCode', 'RunId', 'EventId']
        data_dict = {col: [] for col in columns}
        
        # Populate the dictionary
        for entry in self.data:
            for col, value in zip(columns, entry):
                data_dict[col].append(value)
        
        # Convert the dictionary to an Awkward Array
        ak_array = ak.Array(data_dict)
        
        # Save to ROOT file
        pred_path = f"{self.out_path}/{self.data_type}_predictions.root"
        with uproot.recreate(pred_path) as root_file:
            root_file["tree"] = ak_array
        
        print(f"Predictions saved to '{pred_path}'.")
        # Optionally clear the list to save memory
        self.data.clear()

class PredictionSaver(Callback):
    def __init__(self, out_path, data_type):
        super().__init__()
        self.out_path = out_path
        self.data_type = data_type
        self.data = []
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):   
        # Convert prediction tensor to numpy and squeeze unnecessary dimensions
        predictions = torch.sigmoid(outputs)
        predictions = predictions.detach().cpu().numpy().squeeze()
        #print(predictions)

        # Extract runId and eventId from the batch
        # Assuming the IDs are passed as a tuple or list with the batch, accessible via `batch[1]`
        ids = batch.ids
        #print(ids)
        # Store each prediction with its corresponding runId and eventId
        for prediction, id in zip(predictions, ids):
            pdg_code, run_id, event_id   = id
            #print(prediction, run_id, event_id)
            self.data.append([prediction, pdg_code, run_id, event_id])

    def on_predict_epoch_end(self, trainer, pl_module):
        # Convert collected data to a DataFrame
        df = pd.DataFrame(self.data, columns=['Prediction', 'PdgCode', 'RunId', 'EventId'])

        print(df)
        # Save to CSV file
        pred_path = '{}/{}_predictions.csv'.format(self.out_path, self.data_type)
        df.to_csv(pred_path, index=False)
        print("Predictions saved to '{}'.".format(pred_path))

        # Optionally clear the list to save memory
        self.data.clear()

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=True,
    dropout=0.0,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers with dropout
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)

def get_fully_connected_edges(x):
    
    n_nodes = len(x)
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    
    return torch.cat([edges, edges.flip(0)], axis=1)
