#!/bin/bash
# properties = {properties}

set -e

echo "hostname:"
hostname -f

# Cluster nodes do not consistently provide the small ``appdirs`` dependency
# required by the Snakemake bootstrap command.  Make the workflow's minimal,
# self-contained compatibility module visible before the bootstrap runs.
export PYTHONPATH="/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/vendor${{PYTHONPATH:+:$PYTHONPATH}}"

{exec_job}
