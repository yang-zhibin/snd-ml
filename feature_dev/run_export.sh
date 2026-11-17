#!/usr/bin/env bash
# Match the established unified-shower conversion environment, in a clean process.
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | grep -v '/cvmfs/sft.cern.ch/lcg/' | tr '\n' ':' | sed 's/:$//')
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
set +u
source "$root/snakemake/envs/env_sndsw_nue.sh"
export PYTHONPATH="$SNDSW_ROOT:$root/convertData:${PYTHONPATH:-}"
set -euo pipefail
hostname
pwd
#command -v python3
#python3 --version
converter_python=/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python
"$converter_python" -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'
exec "$converter_python" "$@"
