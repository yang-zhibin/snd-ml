#!/bin/bash

INPUT_LIST_CSV=$1
BASE_OUT_DIR=$2
PARTITION=$3
PARTICLE=$4

# Set up conda environment
source /afs/cern.ch/user/z/zhibin/env_conda.sh
export EOSSHIP=root://eosuser.cern.ch/

mkdir pt/

# Read CSV file, filter by partition, and loop through each line
awk -F, -v partition="$PARTITION" 'NR > 1 && $3 == partition { print $0 }' "$INPUT_LIST_CSV" | while IFS=',' read -r path count partition run_number
do  
    raw_path=${path}
    i_run=${run_number}
    file_name=$(basename ${raw_path})
    input_file=${BASE_OUT_DIR}/run_00${i_run}/${PARTICLE}_converted_$(basename ${raw_path})
    tmp_outpath=pt/${PARTICLE}_${i_run}_${file_name%.*}

    eos_out_dir=${BASE_OUT_DIR}/run_00${i_run}/pt/
    if [ ! -d "${eos_out_dir}" ]; then
        mkdir -p "${eos_out_dir}"
    fi


    echo "Processing run: ${i_run}"
    echo "Input file: ${input_file}"
    echo "Temporary output file: ${tmp_outpath}"

    # Run the Python script for conversion
    python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/root_2_pt.py -i ${input_file} -o ${tmp_outpath} 

    # Copy the output file to the destination directory
    xrdcp -f pt/* ${eos_out_dir}/.

    # Remove the temporary output file
    rm -r pt/*
    
done
