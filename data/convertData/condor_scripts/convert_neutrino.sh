#!/bin/bash

INPUT_DIR=$1
BASE_OUT_DIR=$2
RUN_RANGE_START=$3
RUN_RANGE_END=$4
PARTICLE=$5
CUT_SET=0

# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

export EOSSHIP=root://eosuser.cern.ch/

cp /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/EventClasses.h ./

for i_run in `seq ${RUN_RANGE_START} ${RUN_RANGE_END}`
do
    # check if directory exists
    file_dir=${INPUT_DIR}/${i_run}/
    if [ ! -d "$file_dir" ]; then
        echo "Directory $file_dir does not exist. Skipping..."
        continue
    fi

    # find all .root files and save them to a list
    input_files=(${INPUT_DIR}/${i_run}/*_20240126_digCPP.root)

    #/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root

    # check if there are any .root files in the directory
    if [ ${#input_files[@]} -eq 0 ]; then
        echo "No .root files found in $file_dir. Skipping..."
        continue
    fi

    # define geo file
    geofile=(${INPUT_DIR}/${i_run}/geofile*)

    # create temporary working directory
    mkdir ./convert_rawData/
    tmp_dir="convert_rawData"

    # loop through each .root file and process it
    for input_file in "${input_files[@]}"
    do
        python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/convert_rawData.py -r ${input_file} -g ${geofile} -o ./${tmp_dir}/${PARTICLE}_converted_$(basename ${input_file}) -id ${i_run} -p ${PARTICLE} -t MC
    done

    # check output directory
    mkdir -p ${BASE_OUT_DIR}/${i_run}/
    xrdcp -f ./${tmp_dir}/* ${BASE_OUT_DIR}/${i_run}/

    # remove temporary directory
    rm -r ${tmp_dir}
done
