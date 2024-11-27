#!/bin/bash
Partition=$1

export CUDA_VISIBLE_DEVICES=0

source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh
python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/predict_model.py -m ${Partition}