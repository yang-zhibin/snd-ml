#!/bin/bash

#recoMuon
INPUT_FILE=$1
GEO_FILE=$2
OUT_DIR=$3
PARTITION=$4
CUT_SET=0

# Set up SND environment
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
eval `alienv load --no-refresh sndsw/latest`

export EOSSHIP=root://eosuser.cern.ch/

cp /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/EventClasses.h ./


digi_files=$INPUT_FILE
geo_file=$GEO_FILE

#creat tmp working dir
mkdir ./convert_rawData/
tmp_dir="convert_rawData"

filename=$(basename "$digi_files")
base_name="${filename%.root}"
muon_reco_file="${base_name}_target__muonReco.root"
muon_target_file="${base_name}_target.root"

python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/select_target_muon.py -i ${digi_files} -o ./${tmp_dir}/${muon_target_file}

#full recoMuon  outFileName = options.outPath+filename.replace('.root','_'+runN+'_muonReco.root')
python $SNDSW_ROOT/shipLHC/run_muonRecoSND.py -f ./${tmp_dir}/${muon_target_file} -g ${geo_file} -c passing_mu_DS -sc 1 -s ./${tmp_dir}/ -hf linearSlopeIntercept -o

#stage1 
# cut set (0: stage 1 selection, 1: no veto or scifi 1st layer, 2: FV sideband, 3: include walls 2 and 5, 4: nue filter (no DS selection, allow walls 1 to 4."
neutrinoFilterGoldenSample ./${tmp_dir}/${muon_target_file} ./${tmp_dir}/filtered_MC_00${PARTITION}_stage1.root $CUT_SET

# stage1_reco
python ${SNDSW_ROOT}/shipLHC/run_muonRecoSND.py -f ./${tmp_dir}/filtered_MC_00${PARTITION}_stage1.root -g ${geo_file} -c passing_mu_DS -sc 1 -s ./${tmp_dir}/ -hf linearSlopeIntercept -o

#stage2
python ${SNDSW_ROOT}/analysis/neutrinoFilterGoldenSample_stage2.py -f ./${tmp_dir}/filtered_MC_00${PARTITION}_stage1.root -t ./${tmp_dir}/filtered_MC_00${PARTITION}_stage1__muonReco.root -o ./${tmp_dir}/filtered_MC_00${PARTITION}_stage2.root -g ${geo_file};
#convert

ls ./convert_rawData/
python /afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertData/convert_rawData.py -r ./${tmp_dir}/${muon_target_file} -g ${geo_file} -m ./${tmp_dir}/${muon_reco_file} -s1 ./${tmp_dir}/filtered_MC_00${PARTITION}_stage1.root -s2 ./${tmp_dir}/filtered_MC_00${PARTITION}_stage2.root -o ./${tmp_dir}/${base_name}_converted_00${PARTITION}.root -p ${PARTITION} 

#check output directory
mkdir -p ${OUT_DIR}/${PARTITION}/
xrdcp -f ./${tmp_dir}/* ${OUT_DIR}/${PARTITION}/

# convert

rm -r ${tmp_dir}
