#!/bin/bash

INPUT_DIR=$1


 # Set up pyroot environment
source /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/condor_scripts/env_conda.sh

python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/vm_selection_eff.py -r ${INPUT_DIR}