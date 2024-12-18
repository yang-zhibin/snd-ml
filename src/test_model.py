import yaml
from torch_geometric.data import DataLoader as GeoDataLoader
from dataset.SndGeoDataset import SndGeoDataset, SndGeoDatasetTest
from dataset.SndGeoDataset import RootSaver

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
import argparse

import torchexplorer
import wandb
import pandas as pd
from tqdm import tqdm

def test_model(args):
    model_name = args.model
    raw_file = args.in_file
    out_dir = args.out_dir

    print("start testing")
    #pt_path = '/eos/user/z/zhibin/sndData/converted/pt/test/Neutrinos/'
    split = 'test'
    config_path = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/{model_name}.yml'
    #config_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/GravNetConfig.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    #print(list_files)
    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'
    #version = 'v8'
    ckpt_root = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model_name}/'
    #ckpt_root = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/multiClass_weight_recoMuon/'
    ckpt_path = f"{ckpt_root}/best.ckpt"

    print(f'model :{model_name}')
    print('reading model...')
    model = GravNet.load_from_checkpoint(ckpt_path)

    test_data=SndGeoDataset(root=out_dir, raw_file=raw_file, use_event_feature=config['use_event_feature'], force_reload=True,weight_type=config['weight_type'])
    test_dataloader = GeoDataLoader(test_data, batch_size=config["batch_size"]['train'], shuffle=False, num_workers=4)

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        callbacks=[RootSaver(out_dir,raw_file, model_name)],
        )
    #print("model running test dataset")
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline_muon')
    parser.add_argument("-o", "--out_dir", dest="out_dir", default = '/eos/user/z/zhibin/sndData/converted/pt_tmp/2/')
    parser.add_argument("-i", "--in_file", dest="in_file", default = '/eos/user/z/zhibin/sndData/converted/pt_tmp/2/raw/test_neutrino_2.pt')
    args = parser.parse_args()

    test_model(args)