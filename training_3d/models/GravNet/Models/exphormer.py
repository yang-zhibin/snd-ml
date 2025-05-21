import torch

from ..gnn_base import GNNBase
from .multi_model import MultiModel

class Exphormer(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------
        input_dim = len(self.hparams["feature_set"]) + len(self.hparams["global_feature_set"])
        output_dim = self.hparams["nb_classes"]
        # Encode input features to hidden features
        self.model = MultiModel(input_dim, output_dim, hparams)

    def forward(self, batch):
        # Adds global features to x
        batch.x = self.concat_feature_set(batch)

        # Add dummy edge_index (required for exphormer even if empty)
        batch.edge_index = torch.empty([2, 0], dtype=torch.long, device=batch.batch.device)
        pred = self.model.forward(batch)
        return pred
