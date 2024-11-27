#!/bin/bash

Model=$1

# Login to WANDB

wandb login 5ddaf13209de9fb81832f9685712bcac58d6e607

# Use user CERNBOX as EOS instance
export EOS_MGM_URL=root://eosuser.cern.ch
# stage-in
# eos cp -r /eos/user/z/zhibin/sndData/converted/train_neutrinos/ ./input_data/neutrinos/
# eos cp -r /eos/user/z/zhibin/sndData/converted/train_neutrons/  ./input_data/neutrons/
# eos cp -r /eos/user/z/zhibin/sndData/converted/train_kaons/     ./input_data/kaons/

nvidia-smi
source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh

#Issue of pytorch lightning 
export CUDA_VISIBLE_DEVICES=0

# Start training
python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/train_model.py -m $Model