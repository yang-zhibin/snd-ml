import yaml
from torch_geometric.data import DataLoader as GeoDataLoader
from dataset.SndGeoDataset import SndGeoDataset

import uproot
from lightning.pytorch.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer

from models.test_model.model import LinearNet
from models.GravNet.Models.gravnet import GravNet

from models.GravNet.utils import PredictionSaver
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import os

import torchexplorer
import wandb
import pandas as pd
import argparse


def main(model_name):
    #torch.cuda.empty_cache()
    print("max_num_worker_suggest",len(os.sched_getaffinity(0)))
    print("cpu count", os.cpu_count())
    #reading config file
    config_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/{model_name}.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    #torchexplorer.setup()
    run = wandb.init(       
        dir = os.path.join(config['logger']['save_dir'], f"v{config['logger']['version']}_{config['logger']['name']}"),
        project=config['logger']['project'],
        name = f"v{config['logger']['version']}_{config['logger']['name']}",
        entity = config['logger']['entity'],
        id = f"{config['logger']['name']}_v{config['logger']['version']}",
        #resume = "never",
        #settings=wandb.Settings(code_dir=".")
    )

    run.save('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/train_model.py')
    run.save(config_path)
    run.save('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/dataset/SndGeoDataset.py')
    run.save('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/models/GravNet/*.py')
    run.save('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/models/GravNet/*/*.py')

    
    
    #initial wandb
    logger = WandbLogger(
        save_dir=config['logger']['save_dir'],
        log_model=True
    )



    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['logger']['save_dir'], config['logger']['name']),
        filename='best',
        monitor=config['ModelCheckpoint']['monitor'], 
        mode=config['ModelCheckpoint']['mode'], 
        save_top_k=config['ModelCheckpoint']['save_top_k'], 
        save_last=config['ModelCheckpoint']['save_last']
    )

    #model = LinearNet()
    model = GravNet(config)

    #torchexplorer.watch(model, backend='wandb')

    ckpt_path = os.path.join(config['logger']['save_dir'], config['logger']['name'])

    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        # devices=1,
        #num_nodes=config["nodes"],
        max_epochs=config['max_epochs'],
        #val_check_interval=config['val_check_interval'],
        check_val_every_n_epoch = config['check_val_every_n_epoch'],
        logger=logger,
        #strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        callbacks=[checkpoint_callback],
        #default_root_dir=log
    )

    print("prepare dataset")
    val_data= SndGeoDataset(root=config['data'],split='val', use_event_feature=config['use_event_feature'],weight_type=config['weight_type'])
    train_data= SndGeoDataset(root=config['data'],split='train', use_event_feature=config['use_event_feature'],weight_type=config['weight_type'])
    
    print("prepare dataloader")
    train_dataloader = GeoDataLoader(train_data, batch_size=config["batch_size"]['train'], shuffle=True, num_workers=4)
    val_dataloader = GeoDataLoader(val_data, batch_size=config["batch_size"]['val'], shuffle=False, num_workers=4)
    
    print("start training")
    # for event in train_data:
    #     print(event)
    #     #hit = event.x
    #     #recoMu = event.x_e

    #     break
    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.save(('{}/*ckpt*'.format(ckpt_path)))

    #build_dataset(config)
    #train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline')
    args = parser.parse_args()

    print(args.model)
    main(args.model)
