#!/usr/bin/env bash
# Submit only missing unified shower-feature products through HTCondor.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest_root=/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/manifests
snakemake_bin=/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake
jobs=20
jobs_set=0
dry_run=0
keep_shards=0
exclude_source_uids=()
families=()

usage() {
    cat >&2 <<EOF
Usage: $0 [positive-job-count] [options]

Options:
  --jobs N                    Maximum concurrent jobs (default: 20)
  --family FAMILY             Select mc_neutrino, mc_muon_up, or
                              testbeam_data_2024; may be repeated
  --keep-shards               Validate complete shard sets without merging
  --exclude-source-uid UID    Skip one source UID; may be repeated
  --dry-run                   Resolve and print the filtered DAG only
  -h, --help                  Show this help
EOF
}

while (($#)); do
    case "$1" in
        --jobs)
            if (($# < 2)); then
                usage
                exit 2
            fi
            jobs=$2
            jobs_set=1
            shift 2
            ;;
        --exclude-source-uid)
            if (($# < 2)); then
                usage
                exit 2
            fi
            exclude_source_uids+=("$2")
            shift 2
            ;;
        --family)
            if (($# < 2)); then
                usage
                exit 2
            fi
            families+=("$2")
            shift 2
            ;;
        --keep-shards)
            keep_shards=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        [1-9][0-9]*|[1-9])
            if ((jobs_set)); then
                echo "ERROR: job count was specified more than once" >&2
                usage
                exit 2
            fi
            jobs=$1
            jobs_set=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ((${#families[@]} == 0)); then
    families=(mc_neutrino mc_muon_up testbeam_data_2024)
fi
for family in "${families[@]}"; do
    case "$family" in
        mc_neutrino|mc_muon_up|testbeam_data_2024) ;;
        *)
            echo "ERROR: unsupported family: $family" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --jobs must be a positive integer" >&2
    usage
    exit 2
fi

for source_uid in "${exclude_source_uids[@]}"; do
    if [[ ! "$source_uid" =~ ^(unified_src|tbfull_src)_[0-9a-f]{20}$ ]]; then
        echo "ERROR: invalid source UID: $source_uid" >&2
        exit 2
    fi
done

manifests=()
for family in "${families[@]}"; do
    manifests+=("$manifest_root/$family.csv")
done

for manifest in "${manifests[@]}"; do
    if [[ ! -r "$manifest" ]]; then
        echo "ERROR: manifest is not readable: $manifest" >&2
        exit 1
    fi
done

for source_uid in "${exclude_source_uids[@]}"; do
    if ! awk -F, -v uid="$source_uid" '
        NR > 1 && $3 == uid { found = 1 }
        END { exit !found }
    ' "${manifests[@]}"; then
        echo "ERROR: excluded source UID is absent from the input manifests: $source_uid" >&2
        exit 2
    fi
done

mapfile -t remaining_targets < <(
    /usr/bin/python3 -c '
import csv
import os
import sys

manifest_count = int(sys.argv[1])
manifests = sys.argv[2:2 + manifest_count]
keep_shards = sys.argv[2 + manifest_count] == "1"
exclude_count = int(sys.argv[3 + manifest_count])
excluded = set(sys.argv[4 + manifest_count:4 + manifest_count + exclude_count])
shard_root = sys.argv[4 + manifest_count + exclude_count]
shard_families = {"mc_muon_up", "testbeam_data_2024"}
threshold_input_events = 50000

for manifest in manifests:
    with open(manifest, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["source_uid"] in excluded:
                continue
            canonical_target = row["validation_record"]
            if os.path.isfile(canonical_target):
                continue
            family = row["family"]
            is_sharded = (
                family in shard_families
                and int(float(row["input_event_count"])) > threshold_input_events
            )
            if keep_shards and is_sharded:
                target = os.path.join(
                    shard_root, family, row["source_uid"], "shards.complete.json"
                )
            else:
                target = canonical_target
            if not os.path.isfile(target):
                print(target)
' "${#manifests[@]}" "${manifests[@]}" "$keep_shards" \
    "${#exclude_source_uids[@]}" "${exclude_source_uids[@]}" \
    /eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/shards
)

printf 'Selected families: %s\n' "${families[*]}"
printf 'Keep validated shards: %s\n' "$keep_shards"
printf 'Excluded source files: %d\n' "${#exclude_source_uids[@]}"
printf '  %s\n' "${exclude_source_uids[@]}"
printf 'Remaining ready targets: %d\n' "${#remaining_targets[@]}"
printf '%s\n' "${remaining_targets[@]}"

if ((${#remaining_targets[@]} == 0)); then
    echo "No missing feature files remain."
    exit 0
fi

cd "$repo_root"
# The target list contains only missing canonical or shard-completeness
# products. Reuse existing
# validated shard markers even when implementation files have newer mtimes.
snakemake_extra=()
if ((dry_run)); then
    snakemake_extra+=(--dry-run)
fi
exec env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$snakemake_bin" \
    --snakefile snakemake/snakefile \
    --profile snakemake/.config/snakemake/htcondor \
    --jobs "$jobs" \
    "${remaining_targets[@]}" \
    "${snakemake_extra[@]}" \
    --rerun-triggers input params \
    --consider-ancient unified_shower_feature_shard_v1=manifest \
    --rerun-incomplete \
    --latency-wait 120 \
    --keep-going \
    --drop-metadata
