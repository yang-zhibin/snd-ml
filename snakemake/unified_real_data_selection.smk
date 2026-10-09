"""Standalone two-phase entry point for 2024 real-data selection indexes."""

from pathlib import Path
import re
import sys


REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
CONFIG_PATH = (
    f"{REPO_ROOT}/snakemake/metadata/configs/"
    "unified_real_data_selection_v1.yaml"
)
METADATA_MODULE_DIR = f"{REPO_ROOT}/snakemake/metadata"
if METADATA_MODULE_DIR not in sys.path:
    sys.path.insert(0, METADATA_MODULE_DIR)

from prepare_unified_real_data_selection_manifest import (  # noqa: E402
    build_manifest,
    load_config,
    write_manifest,
)


SETTINGS = load_config(CONFIG_PATH)
SELECTION_MANIFEST = build_manifest(CONFIG_PATH)
MANIFEST_PATH = (
    str(SETTINGS["manifest_root"]).rstrip("/")
    + "/real_data_2024_selection_sources.csv"
)
RECORDS = {
    str(record["source_uid"]): record
    for record in SELECTION_MANIFEST.to_dict("records")
}
EXCLUDED_SOURCE_UIDS = frozenset(
    str(uid) for uid in SETTINGS.get("excluded_source_uids", [])
)
UNKNOWN_EXCLUDED_SOURCE_UIDS = EXCLUDED_SOURCE_UIDS - set(RECORDS)
if UNKNOWN_EXCLUDED_SOURCE_UIDS:
    raise ValueError(
        "unknown excluded real-data selection sources: "
        f"{sorted(UNKNOWN_EXCLUDED_SOURCE_UIDS)}"
    )
SOURCE_UID_PATTERN = "(?:" + "|".join(
    re.escape(uid) for uid in sorted(RECORDS)
) + ")"


def record(wildcards):
    uid = str(wildcards.source_uid)
    if uid not in RECORDS:
        raise ValueError(f"unknown real-data selection source {uid}")
    return RECORDS[uid]


def value(wildcards, column):
    return str(record(wildcards)[column])


def selection_targets(canary=False):
    frame = SELECTION_MANIFEST.loc[
        ~SELECTION_MANIFEST["source_uid"].astype(str).isin(EXCLUDED_SOURCE_UIDS)
    ]
    if canary:
        frame = frame.loc[frame["is_canary"].astype(int).eq(1)]
        if len(frame) != 1:
            raise ValueError(f"expected one selection canary, found {len(frame)}")
    return [str(path) for path in frame["validation_record"]]


def canary_record():
    selected = SELECTION_MANIFEST.loc[
        SELECTION_MANIFEST["is_canary"].astype(int).eq(1)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one selection canary, found {len(selected)}")
    return selected.iloc[0].to_dict()


localrules: unified_prepare_real_data_selection_manifest_v1


rule unified_prepare_real_data_selection_manifest_v1:
    input:
        config=CONFIG_PATH,
        script=(
            f"{REPO_ROOT}/snakemake/metadata/"
            "prepare_unified_real_data_selection_manifest.py"
        ),
    output:
        MANIFEST_PATH
    run:
        write_manifest(SELECTION_MANIFEST, output[0])


rule unified_real_data_selection_index_v1:
    input:
        manifest=MANIFEST_PATH,
        digi=lambda wc: value(wc, "digi_path"),
        cutflow=lambda wc: value(wc, "base_cutflow_path"),
        builder=f"{REPO_ROOT}/convertData/build_real_data_2024_selection_indices.py",
        validator=(
            f"{REPO_ROOT}/convertData/"
            "validate_real_data_2024_selection_indices.py"
        ),
        policy=f"{REPO_ROOT}/convertData/real_data_2024_selection.py",
        base_policy=f"{REPO_ROOT}/convertData/sample_selection.py",
    output:
        no_veto=(
            str(SETTINGS["index_root"]).rstrip("/")
            + "/{source_uid}/no_veto.root"
        ),
        veto_10pct=(
            str(SETTINGS["index_root"]).rstrip("/")
            + "/{source_uid}/veto_10pct.root"
        ),
        summary=(
            str(SETTINGS["index_root"]).rstrip("/")
            + "/{source_uid}/selection_summary.json"
        ),
        marker=(
            str(SETTINGS["index_root"]).rstrip("/")
            + "/{source_uid}/selection.validated.json"
        ),
    params:
        source_uid=lambda wc: value(wc, "source_uid"),
        input_tree=lambda wc: value(wc, "input_tree"),
    wildcard_constraints:
        source_uid=SOURCE_UID_PATTERN,
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=1500,
        disk_mb=1000,
        nvidia_gpu=0,
        eos_feature_read=1,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            -c 'import ROOT; print(ROOT.__version__)'

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/real_data_selection.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.builder} \
            --digi-path "{input.digi}" \
            --base-cutflow-path "{input.cutflow}" \
            --input-tree "{params.input_tree}" \
            --source-uid "{params.source_uid}" \
            --no-veto-output "$tmp_dir/no_veto.root" \
            --veto-10pct-output "$tmp_dir/veto_10pct.root" \
            --summary-output "$tmp_dir/selection_summary.json"

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            --digi-path "{input.digi}" \
            --input-tree "{params.input_tree}" \
            --source-uid "{params.source_uid}" \
            --no-veto-index "$tmp_dir/no_veto.root" \
            --veto-10pct-index "$tmp_dir/veto_10pct.root" \
            --summary "$tmp_dir/selection_summary.json" \
            --validation-record "$tmp_dir/selection.validated.json"

        mkdir -p "$(dirname "{output.no_veto}")"
        xrdcp -f "$tmp_dir/no_veto.root" "{output.no_veto}"
        xrdcp -f "$tmp_dir/veto_10pct.root" "{output.veto_10pct}"
        xrdcp -f "$tmp_dir/selection_summary.json" "{output.summary}"
        xrdcp -f "$tmp_dir/selection.validated.json" "{output.marker}"
        test -r "{output.no_veto}"
        test -r "{output.veto_10pct}"
        test -r "{output.summary}"
        test -r "{output.marker}"
        """


rule unified_real_data_selection_local_smoke_v1:
    input:
        manifest=MANIFEST_PATH,
        digi=lambda wc: str(canary_record()["digi_path"]),
        cutflow=lambda wc: str(canary_record()["base_cutflow_path"]),
        builder=f"{REPO_ROOT}/convertData/build_real_data_2024_selection_indices.py",
        validator=(
            f"{REPO_ROOT}/convertData/"
            "validate_real_data_2024_selection_indices.py"
        ),
        policy=f"{REPO_ROOT}/convertData/real_data_2024_selection.py",
        base_policy=f"{REPO_ROOT}/convertData/sample_selection.py",
    output:
        no_veto=(
            str(SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/no_veto.root"
        ),
        veto_10pct=(
            str(SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/veto_10pct.root"
        ),
        summary=(
            str(SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/selection_summary.json"
        ),
        marker=(
            str(SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/selection.validated.json"
        ),
    params:
        source_uid=str(canary_record()["source_uid"]),
        input_tree=str(canary_record()["input_tree"]),
        max_base_events=int(SETTINGS["local_smoke"]["max_base_events"]),
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=1500,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail
        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            -c 'import ROOT; print(ROOT.__version__)'
        mkdir -p "$(dirname "{output.no_veto}")"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.builder} \
            --digi-path "{input.digi}" \
            --base-cutflow-path "{input.cutflow}" \
            --input-tree "{params.input_tree}" \
            --source-uid "{params.source_uid}" \
            --max-base-events "{params.max_base_events}" \
            --no-veto-output "{output.no_veto}" \
            --veto-10pct-output "{output.veto_10pct}" \
            --summary-output "{output.summary}"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            --digi-path "{input.digi}" \
            --input-tree "{params.input_tree}" \
            --source-uid "{params.source_uid}" \
            --no-veto-index "{output.no_veto}" \
            --veto-10pct-index "{output.veto_10pct}" \
            --summary "{output.summary}" \
            --validation-record "{output.marker}"
        """


rule unified_real_data_selection_canary_v1:
    input:
        selection_targets(canary=True)


rule unified_real_data_selection_indexes_v1:
    input:
        selection_targets()
