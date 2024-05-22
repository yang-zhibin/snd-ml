import yaml
from torch.utils.data import DataLoader
from models.test_model.model import LinearNet
import uproot
from lightning.pytorch.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer
from dataset.prepareEvtList import GetEventList
from dataset.SndDataset import SndDataset

import numpy as np



def main():
    #reading config file
    with open('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/configs/train_config.yml', 'r') as file:
        config = yaml.safe_load(file)


    #initial wandb
    logger = WandbLogger(
        name = 'loggerName',
        save_dir='/afs/cern.ch/user/z/zhibin/work/snd-ml/log',
        version='v1',
        project='loggerProject',
        checkpoint_name='ckpt1'
    )

    model = LinearNet()

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
        #callbacks=[checkpoint_callback],
        #default_root_dir=log
    )

    #GetEventList(config['datapath'], config['partition'], '/eos/user/z/zhibin/sndData/converted/toy_dataset/')

    training_data = SndDataset('/eos/user/z/zhibin/sndData/converted/toy_dataset/train_files.csv')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4)

    # count = 0
    # for data in training_data:
    #     y, x, ids = data
    #     print(np.asarray(x).shape)
    #     count+=1
    #     if count>5:
    #         break

    trainer.fit(model, train_dataloader)

    

    #build_dataset(config)
    #train(config)

if __name__ == "__main__":
    main()