#!/bin/bash

#recoMuon
INPUT_DIR=$1
BASE_OUT_DIR=$2
RUN_RANGE_START=$3
RUN_RANGE_END=$4
CUT_SET=0

# INPUT_DIR="/eos/experiment/sndlhc/users/marssnd/PGsim/neutrons/neu_5_10_tgtarea/Ntuples/"
# BASE_OUT_DIR="/eos/user/z/zhibin/SND_analysis/test/"
# RUN_RANGE_START=1
# RUN_RANGE_END=5
# CUT_SET=4

# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

export EOSSHIP=root://eosuser.cern.ch/

cp /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/EventClasses.h ./

for i_run in `seq ${RUN_RANGE_START} ${RUN_RANGE_END}`
do
    digi_files=(${INPUT_DIR}/${i_run}/*20240126_digCPP.root)
    alternative_digi_files=(${INPUT_DIR}/${i_run}/*_digCPP.root)
    #check if digi file exist
    for file in "$digi_file" "${alternative_digi_files[@]}"; do
        if [ -f "$file" ]; then
            input_file="$file"
            break
        fi
    done

    if [ -z "$input_file" ]; then
        echo "No valid digitized file found in ${INPUT_DIR}/${i_run}. SKIPPING directory"
        continue
    fi

    #check if geo file exist 
    geo_file=$(find "${INPUT_DIR}/${i_run}" -maxdepth 1 -type f -name 'geofile*')

    if [ -n "$geofile" ]; then
        echo "Using geofile found in ${INPUT_DIR}/${i_run}/"
    else
        echo "Using provided geo_file: /eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"
        geofile="/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"
    fi

    #creat tmp working dir
    mkdir ./convert_rawData/
    tmp_dir="convert_rawData"

    #full recoMuon  outFileName = options.outPath+filename.replace('.root','_'+runN+'_muonReco.root')
    python $SNDSW_ROOT/shipLHC/run_muonRecoSND.py -f ${input_file} -g ${geo_file} -c passing_mu_DS -sc 1 -s ./${tmp_dir}/ -hf linearSlopeIntercept -o


    #stage1 
    # cut set (0: stage 1 selection, 1: no veto or scifi 1st layer, 2: FV sideband, 3: include walls 2 and 5, 4: nue filter (no DS selection, allow walls 1 to 4."
    neutrinoFilterGoldenSample ${input_file} ./${tmp_dir}/filtered_MC_00${i_run}_stage1.root $CUT_SET
    
    # stage1_reco
    python ${SNDSW_ROOT}/shipLHC/run_muonRecoSND.py -f ./${tmp_dir}/filtered_MC_00${i_run}_stage1.root -g ${geo_file} -c passing_mu_DS -sc 1 -s ./${tmp_dir}/ -hf linearSlopeIntercept -o

    #stage2
    python ${SNDSW_ROOT}/analysis/neutrinoFilterGoldenSample_stage2.py -f ./${tmp_dir}/filtered_MC_00${i_run}_stage1.root -t ./${tmp_dir}/filtered_MC_00${i_run}_stage1__muonReco.root -o ./${tmp_dir}/filtered_MC_00${i_run}_stage2.root -g ${geo_file};
	
    #convert

    filename=$(basename "$input_file")
    base_name="${filename%.root}"
    muon_out_file="${base_name}__muonReco.root"

    python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/convert_rawData.py -r ${input_file} -g ${geo_file} -m ./${tmp_dir}/${muon_out_file} -s1 ./${tmp_dir}/filtered_MC_00${i_run}_stage1.root -s2 ./${tmp_dir}/filtered_MC_00${i_run}_stage2.root -o ./${tmp_dir}/${base_name}_converted_00${i_run}.root -p ${i_run} 

    #check output directory
    mkdir -p ${BASE_OUT_DIR}/${i_run}/
    xrdcp -f ./${tmp_dir}/* ${BASE_OUT_DIR}/${i_run}/

    # convert

    rm -r ${tmp_dir}
done