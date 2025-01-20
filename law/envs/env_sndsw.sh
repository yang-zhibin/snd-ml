#activate sndsw env
SNDBUILD_DIR=/afs/cern.ch/work/z/zhibin/public/SndBuild
export ALIBUILD_WORK_DIR=$SNDBUILD_DIR/sw
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Aug30/setUp.sh
source /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/setUp.sh
#alienv load --no-refresh sndsw/latest
alienv enter sndsw/latest
