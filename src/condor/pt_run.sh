#!/bin/bash
JOB_NUM=$1
source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh
python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/dataset/preprocess.py -n ${JOB_NUM}