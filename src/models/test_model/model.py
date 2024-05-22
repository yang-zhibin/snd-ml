import pytorch_lightning as pl
import torch
from torch import nn



class LinearNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7*1024, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Softmax(dim=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        target,inputs,ids = batch
        #print(inputs.shape)
        output = self(inputs)
        #print(output.shape)
        #print(output)
        #print(target.shape)
        #print(target)
        loss = self.loss_fn(torch.squeeze(output), target.float())
        values = {'loss': loss}
        self.log_dict(values)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.network.parameters(), lr=0.1)