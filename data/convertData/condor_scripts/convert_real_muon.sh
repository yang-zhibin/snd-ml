#!/bin/bash

INPUT_LIST_CSV=$1
GEO_FILE=$2
BASE_OUT_DIR=$3
PARTITION=$4
PARTICLE=$5

# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

export EOSSHIP=root://eosuser.cern.ch/

cp /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/EventClasses.h ./

data_type='real' 

# Read CSV file, filter by partition, and loop through each line
awk -F, -v partition="$PARTITION" 'NR > 1 && $3 == partition { print $0 }' "$INPUT_LIST_CSV" | while IFS=',' read -r path count partition run_number
do  
    input_file=${path}
    i_run=${run_number}
    tmp_outfile=${PARTICLE}_converted_$(basename ${input_file})

    eos_out_dir=${BASE_OUT_DIR}/run_00${i_run}/
    if [ ! -d "${eos_out_dir}" ]; then
        mkdir -p "${eos_out_dir}"
    fi

    # If ${eos_out_dir}/${tmp_outfile} exists, skip to the next iteration
    if [ -f "${eos_out_dir}/${tmp_outfile}" ]; then
        echo "File ${eos_out_dir}/${tmp_outfile} already exists. Skipping..."
        continue
    fi

    python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/convert_rawData.py -r ${input_file} -g ${GEO_FILE} -o ${tmp_outfile} -id ${i_run} -p ${PARTICLE} -t ${data_type}



    xrdcp -f ${tmp_outfile} ${eos_out_dir}/.

    rm -r ${tmp_outfile}
    
done
