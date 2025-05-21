import warnings

from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class GNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        if "graph_construction" not in self.hparams: self.hparams["graph_construction"] = None
        self.trainset, self.valset, self.testset = None, None, None

        self.validation_step_outputs = []
        self.test_step_outputs = []

    #def predict_step(self, batch, batch_idx, dataloader_idx=0):
        #output = self(batch)
        #prediction = torch.sigmoid(output)
        #return batch.ids, prediction

    def setup(self, stage="fit"):
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
            self.logger.experiment.define_metric("acc" , summary="max")
            for i in range(self.hparams['nb_classes']):
                self.logger.experiment.define_metric(f"acc{i}" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
        
    def concat_feature_set(self, batch):
        """
        Concatenates per-node features and possibly global features.
        - If `batch.x` does not exist already, it is created from the (per-node) feature_set
        - Items in the global_feature_set are stacked on top of batch.x

        """
        #print("concating....")
        all_features = []
        if not hasattr(batch, 'x') or batch.x is None:
            for feature in self.hparams["feature_set"]:
                all_features.append(batch[feature])
            batch.x = torch.stack(all_features).T
        global_features = batch.global_x
        for feature in self.hparams["global_feature_set"]:
            batch.x = torch.cat([batch.x, torch.unsqueeze(batch[feature][batch.batch], dim=1)], dim=1)

        #print(batch.x.shape)
        return batch.x

    def get_metrics(self, targets, output):
        
        prediction = torch.sigmoid(output)

        if self.hparams['nb_classes'] == 1:
            accs = {}
            tp = (prediction.round() == targets).sum().item()
            acc = tp / len(targets)
            try:
                auc = roc_auc_score(targets.bool().cpu().detach(), prediction.cpu().detach())
            except Exception:
                auc = 0
            for i in range(2):
                targets_i = targets[targets == i]
                #print((prediction.round() == i).sum().item())
                
                if not len(targets_i) == 0:
                    accs[i] = ((prediction.round() == i) & (prediction.round() == targets)).sum().item() / len(targets_i)
                else: 
                    accs[i] = 0.5
            return acc, auc, accs

        tp = (torch.argmax(prediction, dim=1) == targets).sum().item()
        acc = tp / len(targets)

        accs = {}
        for i in range(self.hparams["nb_classes"]):
            targets_i = targets[targets == i]
            if not len(targets_i) == 0:
                accs[i] =  (torch.argmax(prediction[targets == i], dim=1) == targets_i).sum().item() / len(targets_i)
            else: 
                accs[i] = 0.5
        auc_prediction = torch.softmax(output, dim=1)
        if len(torch.unique(targets)) != auc_prediction.shape[1]:
            auc_prediction = torch.softmax(torch.index_select(auc_prediction, 1, torch.unique(targets)), dim=1)
        try:
            auc = roc_auc_score(targets.cpu().detach(), auc_prediction.cpu().detach(), multi_class='ovr')
        except ValueError:
            #print('WARNING in gnn_base.py get_metrics, possibly inconsistent shapes', targets.shape, auc_prediction.shape)
            auc = 0.
    
        return acc, auc, accs
    
    def apply_loss_function(self, output, batch):
        #print(batch.weights.shape)
        #print(batch.weights)
       
        if (self.hparams['use_weights']):
            if self.hparams["nb_classes"] == 1 :
                return F.binary_cross_entropy_with_logits(output, batch.y.float(), weight = batch.y + 0.2) #, pos_weight=torch.tensor(self.hparams["pos_weight"]))
            else:
                return F.cross_entropy(output, batch.y, weight=torch.tensor(self.hparams["class_weights"], device=self.device)) # the version above doesn't work with multiclass and integer class labels
        else:
            if self.hparams["nb_classes"] == 1:
                return F.binary_cross_entropy_with_logits(output, batch.y.float()) #, pos_weight=torch.tensor(self.hparams["pos_weight"]))
            else:
                cross_entropy = F.cross_entropy(output, batch.y, weight=torch.tensor(self.hparams["class_weights"], device=self.device),reduction='none')
                weighted_cross_entropy = cross_entropy * batch.weights 

                return torch.mean(weighted_cross_entropy) # the version above doesn't work with multiclass and integer class labels
        

    def training_step(self, batch, batch_idx, **kwargs):
        output = self(batch).squeeze(-1)
        #print("output",output.shape)
        #print("y",batch.y.shape)
        loss = self.apply_loss_function(output, batch)
        
        acc, auc, accs = self.get_metrics(batch.y, output)
        if self.hparams['nb_classes'] == 1:
            save_n_acc = 2
        else:
            save_n_acc = self.hparams['nb_classes']
        log_dict = {"train_loss": loss, "train_auc": auc, 'train_acc': acc} | {f'train_acc{i}': accs[i] for i in range(save_n_acc)}
        #print(log_dict)
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=self.hparams["batch_size"]['train'])
        #self.log(on_epoch=True, logger=True, batch_size=batch_size,)
        torch.cuda.empty_cache()
        return loss        

    def shared_val_step(self, batch, test=True):

        output = self(batch).squeeze(-1)
        #print(output.shape)
        #print("y",batch.y.shape)

        loss = self.apply_loss_function(output, batch)

        acc, auc, accs = self.get_metrics(batch.y, output)
        
        opt = self.optimizers()
        
        current_lr = self.optimizers().param_groups[0]["lr"] if opt else 1.

        if self.hparams['nb_classes'] == 1:
            save_n_acc = 2
        else:
            save_n_acc = self.hparams['nb_classes']
        log_dict = {"val_acc":acc,"val_loss": loss, "current_lr": current_lr} | {f'val_acc{i}': accs[i] for i in range(save_n_acc)}
        
        self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=self.hparams["batch_size"]['val'])

        ret = {
            "loss": loss,
            "outputs": output,
            "targets": batch.y,
            "acc": acc,
            "auc": auc
        }
        
        ret = ret | {f'acc{i}': accs[i] for i in range(self.hparams['nb_classes'])}

        if test:
            self.test_step_outputs.append(ret)
        else:
            self.validation_step_outputs.append(ret)

        return ret

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch, False)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch)
        
    def shared_end_step(self, step_outputs):
        # Concatenate all predictions and targets
        preds = torch.cat([output["outputs"] for output in step_outputs])
        targets = torch.cat([output["targets"] for output in step_outputs])

        # Calculate the ROC curve
        #print(targets.shape, preds.shape)
        acc, auc, accs = self.get_metrics(targets, preds)

        self.log_dict({"val_acc": acc, "val_auc": auc} | {f'val_acc{i}': accs[i] for i in range(self.hparams['nb_classes'])})
    
    def on_validation_epoch_end(self):
        self.shared_end_step(self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_end_step(self.test_step_outputs)
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
        ]
        if "scheduler" not in self.hparams or self.hparams["scheduler"] is None or self.hparams["scheduler"] == "StepLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optimizer[0],
                        step_size=self.hparams["patience"],
                        gamma=self.hparams["factor"],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self.hparams["scheduler"] == "CosineWarmLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer[0],
                        T_0 = self.hparams["patience"], 
                        T_mult=2,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        return optimizer, scheduler
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        
