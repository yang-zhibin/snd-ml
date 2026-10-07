#!/usr/bin/env bash
# Submit only missing unified shower-feature products through HTCondor.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest_root=/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/manifests
snakemake_bin=/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake
jobs=${1:-20}

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [positive-job-count]" >&2
    exit 2
fi

manifests=(
    "$manifest_root/mc_neutrino.csv"
    "$manifest_root/mc_muon_up.csv"
    "$manifest_root/testbeam_data_2024.csv"
)

for manifest in "${manifests[@]}"; do
    if [[ ! -r "$manifest" ]]; then
        echo "ERROR: manifest is not readable: $manifest" >&2
        exit 1
    fi
done

mapfile -t remaining_targets < <(
    /usr/bin/python3 -c '
import csv
import os
import sys

for manifest in sys.argv[1:]:
    with open(manifest, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = row["validation_record"]
            if not os.path.isfile(target):
                print(target)
' "${manifests[@]}"
)

printf 'Remaining canonical files: %d\n' "${#remaining_targets[@]}"
printf '%s\n' "${remaining_targets[@]}"

if ((${#remaining_targets[@]} == 0)); then
    echo "No missing feature files remain."
    exit 0
fi

cd "$repo_root"
# The target list contains only missing canonical products. Reuse existing
# validated shard markers even when implementation files have newer mtimes.
exec env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$snakemake_bin" \
    --snakefile snakemake/snakefile \
    --profile snakemake/.config/snakemake/htcondor \
    --jobs "$jobs" \
    "${remaining_targets[@]}" \
    --rerun-triggers input params \
    --consider-ancient unified_shower_feature_shard_v1=manifest \
    --rerun-incomplete \
    --latency-wait 120 \
    --keep-going \
    --drop-metadata
