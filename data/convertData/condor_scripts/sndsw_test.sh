#!/bin/bash

# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/data_examination/check_MuonReco.py