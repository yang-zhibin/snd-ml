#!/bin/bash
JOB_NUM=$1
source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh
conda activate pyroot
python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/read_stage2.py -n ${JOB_NUM}