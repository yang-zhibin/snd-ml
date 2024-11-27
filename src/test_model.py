import yaml
from torch_geometric.data import DataLoader as GeoDataLoader
from dataset.SndGeoDataset import SndGeoDataset
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

def test_model(model_name):
    print("start testing")
    pt_path = '/eos/user/z/zhibin/sndData/converted/pt'
    split = 'test_muon_outside'
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
    out_path = f"{pt_path}/output/{model_name}/"
    #out_path = f"{pt_path}/output/multiClass_weight_recoMuon"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f'model :{model_name}')
    print('reading model...')
    model = GravNet.load_from_checkpoint(ckpt_path)

    list_files = [os.path.join(pt_path,filename) for filename in os.listdir(pt_path) if filename.startswith(split)]
    print(list_files)

    for file in tqdm(list_files):
        print(f"processing {file}")
        file_name_with_ext = os.path.basename(file)
        file_name, _ = os.path.splitext(file_name_with_ext)
        # Save to ROOT file
        pred_path = f"{out_path}/{file_name}_output.root"

        #if os.path.exists(pred_path):
        #    continue

        test_data=SndGeoDataset(root=pt_path,file_path=file, split=split, use_event_feature=config['use_event_feature'], force_reload=True,weight_type=config['weight_type'])
        test_dataloader = GeoDataLoader(test_data, batch_size=config["batch_size"]['train'], shuffle=False, num_workers=4)

        trainer = Trainer(
            accelerator = accelerator,
            devices="auto",
            callbacks=[RootSaver(out_path,file)],
            )
        #print("model running test dataset")
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='baseline_muon')
    args = parser.parse_args()

    test_model(args.model)