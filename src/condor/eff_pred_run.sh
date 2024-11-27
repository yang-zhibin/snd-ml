#!/bin/bash
Partition=$1
source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh
conda activate pyroot
python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/predict_model.py -p ${Partition}