#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${FEATURE_DEV_PYTHON:-} ]]; then
    python_bin=$FEATURE_DEV_PYTHON
else
    set +u
    source "$root/snakemake/envs/env_lcg.sh" >/dev/null
    python_bin=$(command -v python3)
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:-} == _stage ]]; then
    shift
    exec "$python_bin" "$root/feature_dev/engine.py" "$@"
fi
exec "$python_bin" "$root/feature_dev/cli.py" "$@"
