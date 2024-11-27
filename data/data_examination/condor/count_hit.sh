#!/bin/bash

#recoMuon


Partition=$1



# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

export EOSSHIP=root://eosuser.cern.ch/

mkdir ./convert_rawData/
tmp_dir="convert_rawData"


python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/data_examination/select_muon_like.py -p ${Partition} -o ./${tmp_dir}/

#check output directory
xrdcp -f ./${tmp_dir}/* /eos/user/z/zhibin/sndData/converted/veto_ineff/


rm -r ${tmp_dir}
