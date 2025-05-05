#activate sndsw env
echo "Setting up SNDSW"
SNDLHC_mymaster=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/setUp.sh
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Jan30/setUp.sh
eval $(alienv load sndsw/latest-master-release --no-refresh)