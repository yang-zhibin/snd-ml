#!/bin/bash

INPUT_LIST_CSV=$1
BASE_OUT_DIR=$2
PARTITION=$3
PARTICLE=$4

# Set up conda environment
source /afs/cern.ch/user/z/zhibin/env_conda.sh
export EOSSHIP=root://eosuser.cern.ch/
export CUDA_VISIBLE_DEVICES=0

model=baseline_muon

# Read CSV file, filter by partition, and loop through each line
awk -F, -v partition="$PARTITION" 'NR > 1 && $3 == partition { print $0 }' "$INPUT_LIST_CSV" | while IFS=',' read -r path count partition run_number
do  
    echo ${i_run}

    

    raw_path=${path}
    i_run=${run_number}
    file_name=$(basename ${raw_path})
    input_files="${BASE_OUT_DIR}/run_00${i_run}/pt/${PARTICLE}_${i_run}_${file_name%.*}_chunk_*.pt"
    tmp_dir=./model_ouput/
    mkdir $tmp_dir
    eos_out_dir=${BASE_OUT_DIR}/run_00${i_run}/model_output/${model}/
    if [ ! -d "${eos_out_dir}" ]; then
        mkdir -p "${eos_out_dir}"
    fi


    # Loop through each file matching the pattern
    for file in $input_files; do
        outfile_name=output_${model}_${file%.*}.root
        eos_outfile_name=${eos_out_dir}/${outfile_name}

        # Check if the output file already exists
        if [ -e "$eos_outfile_name" ]; then
            echo "Output file already exists: $eos_outfile_name. Skipping..."
            continue
        fi

        # Check if the input file exists
        if [ -e "$file" ]; then
            echo "Processing file: $file"
            python /afs/cern.ch/user/z/zhibin/work/snd-ml/src/test_model.py -m ${model} -i ${file} -o ${tmp_dir}
        else
            echo "Input file does not exist or no files matched the pattern: $file"
            continue
        fi
    done

    ls -al $tmp_dir


    xrdcp -f ${tmp_dir}/*.root ${eos_out_dir}/.

    rm -r ${tmp_dir}

done
