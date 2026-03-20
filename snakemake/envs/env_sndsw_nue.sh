#activate sndsw env
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/setUp.sh
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh
eval $(alienv load sndsw/latest-event_level_info_output-release --no-refresh)