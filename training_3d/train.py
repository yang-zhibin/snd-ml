import yaml
import torch
import numpy as np
import os
import torchexplorer
import wandb
import pandas as pd
import argparse

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.torchGeoDataset import TrainGeoDataset
from torch_geometric.loader import DataLoader

from models.test_model.model import LinearNet
from models.GravNet.Models.gravnet import GravNet


def main(model_name):
    config_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/training/configs/{model_name}.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    wandb_log_dir = os.path.join(config['logger']['save_dir'], f"{config['logger']['name']}")
    
    if not os.path.exists(wandb_log_dir):
        os.makedirs(wandb_log_dir, exist_ok=True) 

    run = wandb.init(       
        dir = wandb_log_dir,
        project=config['logger']['project'],
        name = f"{config['logger']['name']}_v{config['logger']['version']}",
        entity = config['logger']['entity'],
        id = f"{config['logger']['name']}_v{config['logger']['version']}",
    )

    logger = WandbLogger(
        save_dir=config['logger']['save_dir'],
        log_model=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_log_dir,
        filename='best',
        monitor=config['ModelCheckpoint']['monitor'], 
        mode=config['ModelCheckpoint']['mode'], 
        save_top_k=config['ModelCheckpoint']['save_top_k'], 
        save_last=config['ModelCheckpoint']['save_last']
    )

    model = GravNet(config)

    ckpt_path = wandb_log_dir

    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        max_epochs=config['max_epochs'],
        check_val_every_n_epoch = config['check_val_every_n_epoch'],
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    train_data_root = '/eos/user/z/zhibin/sndData/train_data'
    train_data= torchGeoDataset(root=train_data_root, metadata_dir=config['metadata_dir'],split='train', use_event_feature=config['use_event_feature'],weight_type=config['weight_type'])
    val_data= torchGeoDataset(root=train_data_root, metadata_dir=config['metadata_dir'],split='validation', use_event_feature=config['use_event_feature'],weight_type=config['weight_type'])

    
    print("prepare dataloader")
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"]['train'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"]['val'], shuffle=False, num_workers=4)
    
    print("start training")
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.save(('{}/*ckpt*'.format(ckpt_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline')
    args = parser.parse_args()

    print("start training model:", args.model)
    main(args.model)