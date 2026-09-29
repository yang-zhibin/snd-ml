#!/bin/bash
set -euo pipefail

# HTCondor's container universe starts this command inside the training image,
# where /usr/bin/python3 owns the validated dependency set.  A normal local
# Snakemake run only sees container_image as a scheduling resource, so fall
# back to the shared LCG view when those dependencies are not present.
if PYTHONNOUSERSITE=1 /usr/bin/python3 -c \
    'import awkward, numpy, pandas, uproot' >/dev/null 2>&1
then
    echo "metadata Python: container /usr/bin/python3" >&2
    exec env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
        /usr/bin/python3 "$@"
fi

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source "${runner_dir}/envs/env_lcg.sh"
set -u
python_bin="$(command -v python3)"

# Fail here with a focused diagnostic rather than later in a worker script.
PYTHONNOUSERSITE=1 "${python_bin}" -c 'import awkward, numpy, pandas, uproot'
echo "metadata Python: local LCG ${python_bin}" >&2
exec env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${python_bin}" "$@"
