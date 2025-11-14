import yaml
import torch
import numpy as np
import tempfile

import pandas as pd
import argparse
import glob
import re
from pathlib import Path

from pytorch_lightning import Trainer


from dataset.torchGeoDataset import PredGeoDataset
from torch_geometric.loader import DataLoader
from dataset.CallbackSaver import RootSaver

from models.GravNet.Models.gravnet import GravNet

def extract_model_name(output):
    match = re.search(r'prediction_(.*?)_output', output)
    return match.group(1) if match else None


def main(args):
    print("start testing")
    out_path = args.output
    model_name = extract_model_name(out_path)
    pt_hit_path = args.input
    print("pt file: ",pt_hit_path)

    with open(args.models, 'r') as file:
        models = yaml.safe_load(file)

    print(f'model :{model_name}')
    for entry in models:
        if model_name in entry:
            config_path = entry[model_name][0]["config"]
            ckpt_path = entry[model_name][1]["ckpt"]
            break

    if config_path is None:
        raise ValueError(f"Model name '{model_name}' not found in models list.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f'ckpt:{ckpt_path}')
    print(f'config:{config_path}')

    #print(list_files)
    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'
    
    print('reading model...')
    if accelerator == 'cpu':
        model = GravNet.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    else:
        model = GravNet.load_from_checkpoint(ckpt_path)


    tmp_dir = args.tmpdir
    print("Temporary directory for dataloader:", tmp_dir)
    
    test_data= PredGeoDataset(root=tmp_dir, pt_file=pt_hit_path,use_veto_hits=config['use_veto_hits'], use_event_feature=config['use_event_feature'], weight_type=config['weight_type'],
                              selected_hit_columns=config['hit_feature_cols'], selected_veto_hit_columns=config['hit_feature_cols'], selected_event_columns=config['event_feature_cols'],
                              force_reload=True)
    
    print("preparing dataloader...")
    test_dataloader = DataLoader(test_data,  batch_size=config["batch_size"]['test'], shuffle=False, num_workers=4)

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        callbacks=[RootSaver(pt_hit_path, model_name, out_path)],
        )

    print("model running test dataset...")
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help='input pt hit file path')
    parser.add_argument("-o", "--output", dest="output", help='prediction output file path')
    parser.add_argument("-t", "--tmpdir", dest="tmpdir", help='tmpdir for dataloader')
    parser.add_argument("-m", "--models", dest="models", help='models config file path', default='/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/model_config.yaml')

    args = parser.parse_args()
    main(args)