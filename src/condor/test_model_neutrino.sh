#!/bin/bash

INPUT_DIR=$1
BASE_OUT_DIR=$2
RUN_RANGE_START=$3
RUN_RANGE_END=$4
PARTICLE=$5


# Set up conda environment
source /afs/cern.ch/user/z/zhibin/env_conda.sh
export EOSSHIP=root://eosuser.cern.ch/
export CUDA_VISIBLE_DEVICES=0

model=baseline_muon


# create temporary working directory
root_dir=./data_root/
mkdir ${root_dir}

out_dir=${root_dir}/output/
mkdir ${out_dir}

for i_run in `seq ${RUN_RANGE_START} ${RUN_RANGE_END}`
do

    # find all .root files and save them to a list
    input_file=(${INPUT_DIR}/${i_run}/*.pt)

    mkdir -p ${BASE_OUT_DIR}/${i_run}/output/

    # check if there are any .root files in the directory
    if [ ${#input_file[@]} -eq 0 ]; then
        echo "No .pt files found in $file_dir. Skipping..."
        continue
    fi


    python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/test_model.py -r ${root_dir} -i ${input_file}

    xrdcp -f ${out_dir}/*.root ${BASE_OUT_DIR}/${i_run}/output/

    # remove temporary directory
    rm -r ${out_dir}/*
done
