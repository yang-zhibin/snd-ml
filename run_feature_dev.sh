#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == setup ]]; then
    shift
    set +u
    unset PYTHONHOME
    source "$root/snakemake/envs/env_lcg.sh" >/dev/null
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    exec python3 "$root/feature_dev/setup_env.py" "$@"
fi
# Local configuration is a plain path, never sourced as shell code.
if [[ -z ${FEATURE_DEV_SNAKEMAKE:-} && -f "$root/.feature-dev-snakemake" ]]; then
    IFS= read -r FEATURE_DEV_SNAKEMAKE < "$root/.feature-dev-snakemake"
    export FEATURE_DEV_SNAKEMAKE
fi
if [[ -n ${FEATURE_DEV_PYTHON:-} ]]; then
    python_bin=$FEATURE_DEV_PYTHON
else
    set +u
    source "$root/snakemake/envs/env_lcg.sh" >/dev/null
    python_bin=$(command -v python3)
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:-} == prepare-sample ]]; then
    shift
    exec "$python_bin" "$root/feature_dev/datasets.py" "$@"
fi
if [[ ${1:-} == _stage ]]; then
    shift
    exec "$python_bin" "$root/feature_dev/engine.py" "$@"
fi
exec "$python_bin" "$root/feature_dev/cli.py" "$@"
