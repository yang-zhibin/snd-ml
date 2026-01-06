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
        # Extract logits from Lightning output dict
        logits = outputs["outputs"]
        #print(f"output shape: {logits.shape}")

        # Convert logits -> probabilities depending on shape
        if logits.ndim == 1:
            # [batch] – single logit per sample (binary)
            probs = torch.sigmoid(logits)
        elif logits.ndim == 2 and logits.size(1) == 1:
            # [batch, 1] – also single logit per sample
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            # [batch, num_classes] – multi-class
            probs = torch.softmax(logits, dim=-1)

        predictions = probs.detach().cpu().numpy()
        #print("predictions", predictions.shape)

        # Extract runId and eventId from the batch
        ids = batch.ids
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()

        #print("ids", getattr(ids, "shape", "no shape attr"))

        batch_size = predictions.shape[0]

        for i in range(batch_size):  # Loop over the batch dimension
            pdg_code, run_id, event_id = ids[i]

            if predictions.ndim == 1:
                # Single probability per sample
                prediction_list = [float(predictions[i])]
            else:
                # Vector of probabilities per sample
                prediction_list = predictions[i].tolist()

            data_entry = prediction_list + [int(pdg_code), int(run_id), int(event_id)]
            self.data.append(data_entry)

        # Build column names according to prediction shape
        if predictions.ndim == 1:
            pred_cols = ["Prediction"]
        else:
            n_pred = predictions.shape[1]
            pred_cols = [f"Prediction_{i}" for i in range(n_pred)]

        self.data_column_names = pred_cols + ["PdgCode", "RunId", "EventId"]

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.data:
            print("No data to save.")
            return

        columns = self.data_column_names
        data_dict = {col: [] for col in columns}

        # Populate the dictionary
        for entry in self.data:
            for col, value in zip(columns, entry):
                data_dict[col].append(value)

        # Convert the dictionary to an Awkward Array
        ak_array = ak.Array(data_dict)

        with uproot.recreate(self.out_path) as root_file:
            root_file["sndData"] = {key: ak_array[key] for key in ak_array.fields}

        print(f"Predictions saved to '{self.out_path}'.")
        # Optionally clear the list to save memory
        self.data.clear()