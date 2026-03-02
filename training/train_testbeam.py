import yaml
import torch
import numpy as np
import os
import torchexplorer
import wandb
import pandas as pd
import argparse
import glob
import time
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.torchGeoDataset_testbeam import TrainGeoDataset
from torch_geometric.loader import DataLoader

from models.test_model.model import LinearNet
from models.GravNet.Models.gravnet import GravNet
import tempfile


def main(args):
    config_path = args.config
    # tmp_dir = args.tmp_dir

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_name = config['model_name']
    split_name = config['split_name']
    
    name = f"{config['model_name']}__{config['split_name']}__v{config['logger']['version']}"
    run_id = f"{name}__{int(time.time())}"   # unique
    wandb_log_dir = os.path.join(config['logger']['save_dir'], run_id)
    
    if not os.path.exists(wandb_log_dir):
        os.makedirs(wandb_log_dir, exist_ok=True) 

    logger = WandbLogger(
        project=config["logger"]["project"],
        entity=config["logger"]["entity"],
        name=run_id,         
        id=run_id,            
        save_dir=wandb_log_dir,
        log_model=True,      
        resume="allow", 
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_log_dir,
        filename='best',
        monitor=config['ModelCheckpoint']['monitor'], 
        mode=config['ModelCheckpoint']['mode'], 
        save_top_k=config['ModelCheckpoint']['save_top_k'], 
        save_last=config['ModelCheckpoint']['save_last'],
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
        precision="16-mixed",
        accumulate_grad_batches=config['accumulate_grad_batches'],
    )

    # make a tmp dir for tmp processed_pt_root
    tmp_processed_pt_root = tempfile.mkdtemp(prefix="processed_pt_")

    def maybe_copy_pt(split, split_name, src_root, dst_root):
        fname = f"{split}_{split_name}.pt"
        src = os.path.join(src_root, fname)
        dst = os.path.join(dst_root, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    processed_root = config["processed_pt_root"]

    # remember which files existed BEFORE processing
    train_existed = os.path.exists(
        os.path.join(processed_root, f"train_{split_name}.pt")
    )
    val_existed = os.path.exists(
        os.path.join(processed_root, f"val_{split_name}.pt")
    )

    # copy existing files into tmp
    maybe_copy_pt("train", split_name, processed_root, tmp_processed_pt_root)
    maybe_copy_pt("val",   split_name, processed_root, tmp_processed_pt_root)

    train_data = TrainGeoDataset(
        root=tmp_processed_pt_root,
        metadata_dir=config["metadata_dir"],
        split_name=split_name,
        split="train",
        use_veto_hits=config["use_veto_hits"],
        use_event_feature=config["use_event_feature"],
        weight_type=config["weight_type"],
        selected_hit_columns=config["hit_feature_cols"],
        selected_veto_hit_columns=config["hit_feature_cols"],
        selected_event_columns=config["event_feature_cols"],
        force_reload=True,
    )

    val_data = TrainGeoDataset(
        root=tmp_processed_pt_root,
        metadata_dir=config["metadata_dir"],
        split_name=split_name,
        split="val",
        use_veto_hits=config["use_veto_hits"],
        use_event_feature=config["use_event_feature"],
        weight_type=config["weight_type"],
        selected_hit_columns=config["hit_feature_cols"],
        selected_veto_hit_columns=config["hit_feature_cols"],
        selected_event_columns=config["event_feature_cols"],
        force_reload=True,
    )
    
    if not train_existed:
        maybe_copy_pt("train", split_name, tmp_processed_pt_root, processed_root)

    if not val_existed:
        maybe_copy_pt("val", split_name, tmp_processed_pt_root, processed_root)
        
    if "orientation_filter" in config and config["orientation_filter"] is not None:
        orientation_filter = config["orientation_filter"]
        print(f"Filtering dataset for orientation = {orientation_filter}")
        train_data = [d for d in train_data if getattr(d, "orientation", None) == orientation_filter]
        val_data = [d for d in val_data if getattr(d, "orientation", None) == orientation_filter]
    
    
    print("prepare dataloader")
    nw = min(16, os.cpu_count() or 4)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config["batch_size"]["train"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,              # needs num_workers>0
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=config["batch_size"]["val"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    print("start training")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.get("resume_from_checkpoint", None)) # 
    wandb.save(('{}/*ckpt*'.format(ckpt_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config")
    # parser.add_argument("-t", "--tmp_dir", dest="tmp_dir")
    args = parser.parse_args()

    main(args)