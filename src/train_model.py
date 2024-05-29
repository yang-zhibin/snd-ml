import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader
import uproot
from lightning.pytorch.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer
from dataset.prepareEvtList import GetEventList
from dataset.SndDataset import SndDataset
from dataset.GeoSndDataset import GeoSndDataset

from models.test_model.model import LinearNet
from models.GeometricAttention.Models.gravnet import GravNet

from models.GeometricAttention.utils import PredictionSaver

import numpy as np



def main():
    #reading config file
    with open('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/train_config.yml', 'r') as file:
        config = yaml.safe_load(file)
       
    with open('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/GravNetConfig.yml', 'r') as file:
        model_hparam = yaml.safe_load(file)
    
    

    #initial wandb
    logger = WandbLogger(
        name = 'test',
        save_dir='/afs/cern.ch/user/z/zhibin/work/snd-ml/log',
        version='v2',
        project='snd-ml',
        checkpoint_name='ckpt1'
    )

    #model = LinearNet()
    model = GravNet(model_hparam)

    accelerator = "gpu" if torch.cuda.is_available() else None

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        # devices=1,
        #num_nodes=config["nodes"],
        max_epochs=5,
        logger=logger,
        # strategy=CustomDDPPlugin(find_unused_parameters=False),
        #strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        callbacks=[PredictionSaver()],
        #default_root_dir=log
    )

    #GetEventList(config['datapath'], config['partition'], '/eos/user/z/zhibin/sndData/converted/toy_dataset/')

    #training_data = SndDataset('/eos/user/z/zhibin/sndData/converted/toy_dataset/train_files.csv')


    mode = 'predict'
    if (mode=="train"):
        training_data= SndDataset('/eos/user/z/zhibin/sndData/converted/toy_dataset/train_files.csv')
        train_dataloader = GeoDataLoader(training_data, batch_size=64, shuffle=True, num_workers=8)
        trainer.fit(model, train_dataloader)

    elif (mode=='predict'):
        train_model = GravNet.load_from_checkpoint("/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml/v1/checkpoints/epoch=3-step=5324.ckpt")
        pred_data = SndDataset('/eos/user/z/zhibin/sndData/converted/toy_dataset/test_files_0.csv')
        pred_dataloader = GeoDataLoader(pred_data, batch_size=4, shuffle=False, num_workers=1)
        predict_trainer = Trainer(
            accelerator = accelerator,
            devices="auto",
            max_epochs=1,
            callbacks=[PredictionSaver()],
            )
        trainer.predict(train_model, pred_dataloader)

    

    #build_dataset(config)
    #train(config)

if __name__ == "__main__":
    main()