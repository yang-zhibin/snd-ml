#!/usr/bin/env bash
# Production converter + optional coworker plugin; analysis stays in LCG.
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == setup ]]; then
    shift
    exec "$root/run_feature_dev.sh" setup "$@"
fi
if [[ -z ${FEATURE_DEV_SNAKEMAKE:-} && -f "$root/.feature-dev-snakemake" ]]; then
    IFS= read -r FEATURE_DEV_SNAKEMAKE < "$root/.feature-dev-snakemake"
    export FEATURE_DEV_SNAKEMAKE
fi
set +u
unset PYTHONHOME
source "$root/snakemake/envs/env_lcg.sh" >/dev/null
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:-} == _worker ]]; then
    shift
    exec python3 "$root/feature_production/worker.py" "$@"
fi
exec python3 "$root/feature_production/cli.py" "$@"
