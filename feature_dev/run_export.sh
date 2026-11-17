#!/usr/bin/env bash
# Activate the established converter stack without inheriting another Python.
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# setUp.sh captures sys.path from the first python3 on PATH. Filtering only LCG
# leaves LHCb/Conda interpreters behind and mixes their stdlib with the converter.
# Keep identity, credentials and scratch settings, but discard software stacks,
# module state, exported functions and Bash startup hooks before activation.
clean_env=(PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1)
for name in HOME USER LOGNAME KRB5CCNAME X509_USER_PROXY X509_CERT_DIR \
            BEARER_TOKEN_FILE TMPDIR TMP TEMP LANG LC_ALL TZ; do
    if [[ -v $name ]]; then
        clean_env+=("$name=${!name}")
    fi
done
exec /usr/bin/env -i "${clean_env[@]}" /bin/bash --noprofile --norc -c '
    set -eo pipefail
    root=$1
    shift
    set +u
    source "$root/snakemake/envs/env_sndsw_nue.sh"
    export PYTHONPATH="$SNDSW_ROOT:$root/convertData:${PYTHONPATH:-}"
    set -euo pipefail
    converter_python=/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python
    hostname
    pwd
    command -v "$converter_python"
    "$converter_python" --version
    "$converter_python" -c '\''import sys, io, ROOT, SndlhcGeo; print("converter executable:", sys.executable); print("stdlib:", io.__file__); print("ROOT:", ROOT.__version__)'\''
    exec "$converter_python" "$@"
' run-export "$root" "$@"
