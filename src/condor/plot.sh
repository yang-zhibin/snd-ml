#!/bin/bash
model=$1
signal=$2
initial=$3
begin=$4

#export CUDA_VISIBLE_DEVICES=0
export EOSSHIP=root://eosuser.cern.ch/
mkdir ./${model}
xrdcp -f /eos/user/z/zhibin/sndData/converted/pt/output/${model}/* ./${model}/.


source /afs/cern.ch/user/z/zhibin/work/snd-ml/src/condor/env_conda.sh
conda activate pyroot

python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/cal_bkg_vs_signal.py -m ${model} -s ${signal} -i ${initial} -b ${begin}

rm -r ./${model}