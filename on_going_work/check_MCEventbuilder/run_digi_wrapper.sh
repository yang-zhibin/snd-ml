#!/bin/bash
set -euo pipefail

/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python "$@"
rc=$?

if [[ $rc -eq 143 ]]; then
  echo "SIGTERM caught — mapping to success"
  exit 0
fi

exit $rc
