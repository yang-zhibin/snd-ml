from models.GravNet.Models.gravnet import GravNet
from torch_geometric.data import DataLoader as GeoDataLoader
from dataset.PredDataset import PredDataset

from pytorch_lightning import Trainer
from models.GravNet.utils import PredictionSaver
from models.GravNet.utils import PredictionRootSaver

import torch

from torchviz import make_dot
#import torchexplorer
import os
import argparse

def pred_csv():
    pred_data= SndGeoDataset(root='/eos/user/z/zhibin/sndData/converted/toy_dataset/pt_data/',split='pred')

    accelerator = "gpu" if torch.cuda.is_available() else None

    run_name = 'test_local'
    ckpt_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/log/{}/best.ckpt".format(run_name)
    train_model = GravNet.load_from_checkpoint(ckpt_path)

    out_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/log/{}/".format(run_name)
    pred_type = 'vm'

    pred_dataloader = GeoDataLoader(pred_data, batch_size=64, shuffle=False, num_workers=8)
   
    predict_trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        max_epochs=1,
        callbacks=[PredictionSaver(out_path,pred_type)],
        )
    
    predict_trainer.predict(train_model, pred_dataloader)

def prepare_prediction_input(partition=None):
    pt_path = '/eos/user/z/zhibin/sndData/converted/pt_data'
    save_dir = '/eos/user/z/zhibin/sndData/converted/pred_input/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    split = 'pred'
    list_files = [
            os.path.join(pt_path, filename) 
            for filename in os.listdir(pt_path) 
            if filename.startswith(split) and partition in filename
        ]
    gap = 100
    for i in range(0, len(list_files), gap):
        chunk = list_files[i:i+gap]
        pred_data= PredDataset(root=pt_path,save_dir=save_dir,chunk=chunk,chunk_id=i,partition=partition,split='pred')

    print("pred intput complete")

def pred_root_partition(partition=None):
    print("start prediction...")

    model_name = 'binary_vm'
    pt_path = '/eos/user/z/zhibin/sndData/converted/pt_data'
    save_dir = f'/eos/user/z/zhibin/sndData/converted/pred_input/'
    split = 'pred'

    list_files = [
            os.path.join(pt_path, filename) 
            for filename in os.listdir(pt_path) 
            if filename.startswith(split) and partition in filename
        ]

    #print(list_files)
    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'
    ckpt_root = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model_name}/'
    ckpt_path = f"{ckpt_root}/best.ckpt"
    out_path = f"{ckpt_root}/pred_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f'model :{model_name}, partition: {partition}')
    print('reading model...')
    train_model = GravNet.load_from_checkpoint(ckpt_path)
    gap = 100
    for i in range(0, len(list_files), gap):
        chunk = []

        pred_data= PredDataset(root=pt_path,save_dir=save_dir,chunk=chunk,chunk_id=i,partition=partition,split='pred')
        pred_type = "{}_id{}".format(partition, i)

        pred_dataloader = GeoDataLoader(pred_data, batch_size=32, shuffle=False, num_workers=4)
    
        predict_trainer = Trainer(
            accelerator = accelerator,
            devices="auto",
            max_epochs=1,
            callbacks=[PredictionRootSaver(out_path,pred_type)],
            )
        
        predict_trainer.predict(train_model, pred_dataloader)

    print(f"{partition} completed")

def pred_root_muon(model_name):
    print("start prediction...")

    pt_path = '/eos/user/z/zhibin/sndData/converted/pt_muon/'
    save_dir = f'/eos/user/z/zhibin/sndData/converted/pred_muon_input/'
    split = 'pred'



    #print(list_files)
    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'
    ckpt_root = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model_name}/'
    ckpt_path = f"{ckpt_root}/best.ckpt"
    out_path = f"{ckpt_root}/pred_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f'model :{model_name}')
    print('reading model...')
    train_model = GravNet.load_from_checkpoint(ckpt_path)

    #particle_ranges = ['Neutrinos',
    #"kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50",
    # "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100", 
    # "neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50",
    # "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90", "neutrons_90_100",
    # "kaons_5_10",
    # ] 
    particle_ranges = '2024'

    for partition in particle_ranges:
        print( f'partition: {partition}')
        list_files = [
                os.path.join(pt_path, filename) 
                for filename in os.listdir(pt_path) 
                if filename.startswith(split) and partition in filename
            ]
        gap = 100
        for i in range(0, len(list_files), gap):
            pred_type = "{}_id{}".format(partition, i)
            outfile = f"{out_path}/{pred_type}_predictions.root"
            if os.path.isfile(outfile):
                print("pass", outfile)
                continue
            chunk = list_files[i:i+gap]

            pred_data= PredDataset(root=pt_path,save_dir=save_dir,chunk=chunk,chunk_id=i,partition=partition,split='pred')
            

            pred_dataloader = GeoDataLoader(pred_data, batch_size=32, shuffle=False, num_workers=4)
        
            predict_trainer = Trainer(
                accelerator = accelerator,
                devices="auto",
                max_epochs=1,
                callbacks=[PredictionRootSaver(out_path,pred_type)],
                )
            
            predict_trainer.predict(train_model, pred_dataloader)

        print(f"{partition} completed")
    
    print('all completed')

def pred_root(model_name):
    print("start prediction...")

    pt_path = '/eos/user/z/zhibin/sndData/converted/pt_data'
    save_dir = f'/eos/user/z/zhibin/sndData/converted/pred_input/'
    split = 'pred'



    #print(list_files)
    accelerator = "gpu" if torch.cuda.is_available() else 'cpu'
    ckpt_root = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/log/snd-ml-GravNet/{model_name}/'
    ckpt_path = f"{ckpt_root}/best.ckpt"
    out_path = f"{ckpt_root}/pred_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f'model :{model_name}')
    print('reading model...')
    train_model = GravNet.load_from_checkpoint(ckpt_path)

    particle_ranges = ['Neutrinos',
    "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50",
    "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100", 
    "neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50",
    "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90", "neutrons_90_100",
    "kaons_5_10",
    ] 

    for partition in particle_ranges:
        print( f'partition: {partition}')
        list_files = [
                os.path.join(pt_path, filename) 
                for filename in os.listdir(pt_path) 
                if filename.startswith(split) and partition in filename
            ]
        gap = 100
        for i in range(0, len(list_files), gap):
            pred_type = "{}_id{}".format(partition, i)
            outfile = f"{out_path}/{pred_type}_predictions.root"
            if os.path.isfile(outfile):
                print("pass", outfile)
                continue
            chunk = list_files[i:i+gap]

            pred_data= PredDataset(root=pt_path,save_dir=save_dir,chunk=chunk,chunk_id=i,partition=partition,split='pred')
            

            pred_dataloader = GeoDataLoader(pred_data, batch_size=32, shuffle=False, num_workers=4)
        
            predict_trainer = Trainer(
                accelerator = accelerator,
                devices="auto",
                max_epochs=1,
                callbacks=[PredictionRootSaver(out_path,pred_type)],
                )
            
            predict_trainer.predict(train_model, pred_dataloader)

        print(f"{partition} completed")
    
    print('all completed')

def model_visualization():
    print("visualizing model...")
    run_name = 'test_local'
    ckpt_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/log/{}/best.ckpt".format(run_name)

    pred_data= SndGeoDataset(root='/eos/user/z/zhibin/sndData/converted/toy_dataset/pt_data/',split='pred')
    model = GravNet.load_from_checkpoint(ckpt_path)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    for event in pred_data:
        output = model(event)
        break

    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)


    # Save or display the graph
    dot.render("loaded_model_graph", format="png")  # Saves the graph as a PNG file
    dot.view()

def test():

    pred_data= SndGeoDataset(root='/eos/user/z/zhibin/sndData/converted/pt_data/',split='val', signal= 'vm')
    pred_dataloader = GeoDataLoader(pred_data, batch_size=1, shuffle=False, num_workers=4)
   
    for event in pred_data:
        print(event)
        y = event.y
        ids = event.ids
        print('y', y)
        print('ids', ids)
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default='multi_new')
    args = parser.parse_args()
    #pred_root(args.model)
    #pred_root_muon(args.model)
    #prepare_prediction_input(args.partition)
    #test()
    #model_visualization()