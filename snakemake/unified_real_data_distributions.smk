"""2024 real-data feature production and two-cohort shape distributions."""

from pathlib import Path
from functools import lru_cache
import re
import shlex

import pandas as pd
import yaml


REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
CONFIG_PATH = (
    f"{REPO_ROOT}/snakemake/metadata/configs/"
    "unified_real_data_distributions_v1.yaml"
)
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle)

VERSION = str(SETTINGS["version"])
FEATURES = tuple(SETTINGS["features"])
PROFILES = tuple(SETTINGS["profiles"])
GROUPS = tuple(SETTINGS["groups"])
PROFILE_FAMILIES = {
    profile: str(settings["family"])
    for profile, settings in SETTINGS["profiles"].items()
}
FAMILY_PROFILES = {family: profile for profile, family in PROFILE_FAMILIES.items()}
METADATA_ROOT = str(SETTINGS["outputs"]["metadata_root"]).rstrip("/")
FEATURE_ROOT = str(SETTINGS["outputs"]["feature_root"]).rstrip("/")
HISTOGRAM_ROOT = str(SETTINGS["outputs"]["histogram_root"]).rstrip("/")
PLOT_ROOT = str(SETTINGS["outputs"]["plot_root"]).rstrip("/")
SNAPSHOT_ROOT = f"{METADATA_ROOT}/{VERSION}/feature_snapshots"
MANIFEST_BUILDER = (
    f"{REPO_ROOT}/snakemake/metadata/"
    "prepare_unified_real_data_feature_manifests.py"
)
SNAPSHOT_BUILDER = (
    f"{REPO_ROOT}/snakemake/metadata/build_unified_partial_feature_snapshots.py"
)
CONVERTER = f"{REPO_ROOT}/convertData/digi_2_features.py"
VALIDATOR = f"{REPO_ROOT}/convertData/validate_unified_shower_feature_file.py"
DISTRIBUTION_SCRIPT = (
    f"{REPO_ROOT}/evaluation_region_partitions/unified_feature_distributions.py"
)
SOURCE_UID_PATTERN = r"unified_src_[0-9a-f]{20}"


def source_manifest(profile):
    return str(SETTINGS["source_manifests"][PROFILE_FAMILIES[str(profile)]])


def feature_path(profile, source_uid, suffix=""):
    return (
        f"{FEATURE_ROOT}/{profile}/{source_uid}/"
        f"feature_scifi_cluster_v1_{source_uid}.root{suffix}"
    )


def snapshot_manifest(family):
    return f"{SNAPSHOT_ROOT}/{family}.csv"


def audit_path(family):
    return f"{METADATA_ROOT}/{VERSION}/feature_completeness/{family}.json"


def histogram_path(group, feature):
    return f"{HISTOGRAM_ROOT}/{VERSION}/{feature}/{group}.root"


def plot_path(feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/unit_area/{feature}.{suffix}"


def _prepared_outputs():
    return checkpoints.unified_prepare_real_data_distribution_manifests_v1.get().output


def _manifest_path_for_profile(profile):
    return str(getattr(_prepared_outputs(), str(profile)))


@lru_cache(maxsize=None)
def _manifest_frame(profile):
    frame = pd.read_csv(_manifest_path_for_profile(profile), keep_default_na=False)
    if frame.empty:
        raise ValueError(f"empty real-data feature manifest for {profile}")
    return frame


@lru_cache(maxsize=None)
def _manifest_records(profile):
    frame = _manifest_frame(profile)
    if frame["source_uid"].duplicated().any():
        raise ValueError(f"duplicate source UIDs in real-data feature manifest for {profile}")
    return {
        str(record["source_uid"]): record
        for record in frame.to_dict("records")
    }


def _record(wildcards):
    profile = str(wildcards.profile)
    uid = str(wildcards.source_uid)
    record = _manifest_records(profile).get(uid)
    if record is None:
        raise ValueError(f"no {profile} record for {uid}")
    expected = feature_path(profile, uid)
    if str(record["feature_scifi_cluster_v1_path"]) != expected:
        raise ValueError(
            f"feature output mismatch for {profile}/{uid}: "
            f"{record['feature_scifi_cluster_v1_path']} != {expected}"
        )
    return record


def _value(wildcards, column):
    return str(_record(wildcards).get(column, ""))


def _feature_targets(wildcards, canary=False):
    targets = []
    for profile in PROFILES:
        frame = _manifest_frame(profile)
        if canary:
            frame = frame.loc[pd.to_numeric(frame["is_canary"]).eq(1)]
            if len(frame) != 1:
                raise ValueError(
                    f"expected one {profile} canary row, found {len(frame)}"
                )
        targets.extend(
            feature_path(profile, str(uid), ".validated.json")
            for uid in frame["source_uid"]
        )
    return targets


def _feature_cli(wildcards):
    record = _record(wildcards)
    arguments = [
        "-p", str(record["cutflow_path"]),
        "-d", str(record["digi_path"]),
        "-g", str(record["geo_path"]),
        "-t", "real_data",
        "--tree-name", str(record["input_tree"]),
        "--selection-mode", "indexed_cutflow",
        "--source-domain-id", "3",
        "--source-uid", str(record["source_uid"]),
    ]
    return " ".join(shlex.quote(value) for value in arguments)


def _plot_hists(wildcards):
    return [histogram_path(group, wildcards.feature) for group in GROUPS]


def _plot_hist_args(wildcards, input):
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(GROUPS, input.hists)
    )


localrules: unified_prepare_real_data_distribution_manifests_v1, unified_real_data_distribution_snapshot_v1


checkpoint unified_prepare_real_data_distribution_manifests_v1:
    input:
        config=CONFIG_PATH,
        selection_manifest=str(SETTINGS["inputs"]["selection_manifest"]),
        selection_policy=str(SETTINGS["inputs"]["selection_policy"]),
        builder=MANIFEST_BUILDER,
    output:
        no_veto=source_manifest("no_veto"),
        veto_10pct=source_manifest("veto_10pct"),
        coverage_json=f"{METADATA_ROOT}/{VERSION}/source_coverage.json",
        coverage_csv=f"{METADATA_ROOT}/{VERSION}/source_coverage.csv",
    threads: 16
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=200,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        hostname
        pwd
        command -v python3
        python3 --version
        python3 -c 'import pandas, yaml; print(pandas.__version__)'
        python3 {input.builder} --config {input.config} \
            --coverage-json {output.coverage_json} \
            --coverage-csv {output.coverage_csv} --workers {threads}
        """


rule unified_real_data_distribution_feature_v1:
    input:
        manifest=lambda wc: _manifest_path_for_profile(wc.profile),
        digi=lambda wc: _value(wc, "digi_path"),
        geo=lambda wc: _value(wc, "geo_path"),
        index=lambda wc: _value(wc, "cutflow_path"),
        selection_validation=lambda wc: _value(
            wc, "selection_validation_record"
        ),
        converter=CONVERTER,
        validator=VALIDATOR,
        selection=f"{REPO_ROOT}/convertData/sample_selection.py",
        provenance=f"{REPO_ROOT}/convertData/source_provenance.py",
        clustering=f"{REPO_ROOT}/convertData/scifi_plane_clustering.py",
        config=CONFIG_PATH,
    output:
        feature=(
            f"{FEATURE_ROOT}/{{profile}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root"
        ),
        marker=(
            f"{FEATURE_ROOT}/{{profile}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root.validated.json"
        ),
    params:
        cli=_feature_cli,
        input_count=lambda wc: int(float(_value(wc, "input_event_count"))),
        selected_count=lambda wc: int(float(_value(wc, "selected_event_count"))),
        source_uid=lambda wc: _value(wc, "source_uid"),
        input_tree=lambda wc: _value(wc, "input_tree"),
    wildcard_constraints:
        profile="|".join(PROFILES),
        source_uid=SOURCE_UID_PATTERN,
    threads: 1
    resources:
        runtime=2 * 60 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0,
        eos_feature_read=1,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' | tr '\n' ':' | sed 's/:$//')
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
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python -c \
            'import ROOT, SndlhcGeo; print(ROOT.__version__)'
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_real_data_feature.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            {params.cli} -o "$tmp_dir/feature.root"
        converter_commit=$(git -C {REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_dir/feature.root" --manifest-version {VERSION} \
            --expected-source-uid {params.source_uid} \
            --expected-source-domain-id 3 \
            --expected-input-event-count {params.input_count} \
            --expected-selected-event-count {params.selected_count} \
            --sample-type real_data --selection-mode indexed_cutflow \
            --input-tree {params.input_tree} --expected-max-events 0 \
            --input-digi-path {input.digi} --input-cutflow-path {input.index} \
            --published-feature-path {output.feature} \
            --converter-commit "$converter_commit" \
            --validation-record "$tmp_dir/validated.json"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_dir/feature.root" "{output.feature}"
        xrdcp -f "$tmp_dir/validated.json" "{output.marker}"
        test -r "{output.feature}"
        test -r "{output.marker}"
        """


rule unified_real_data_distribution_feature_canaries_v1:
    input:
        lambda wc: _feature_targets(wc, canary=True)


rule unified_real_data_distribution_features_v1:
    input:
        _feature_targets


rule unified_real_data_distribution_snapshot_v1:
    input:
        manifests=lambda wc: [
            _manifest_path_for_profile(profile) for profile in PROFILES
        ],
        config=CONFIG_PATH,
        builder=SNAPSHOT_BUILDER,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
        selection_coverage=lambda wc: checkpoints.unified_prepare_real_data_distribution_manifests_v1.get().output.coverage_json,
    output:
        manifests=[snapshot_manifest(family) for family in GROUPS],
        coverage_json=f"{METADATA_ROOT}/{VERSION}/feature_coverage.json",
        coverage_csv=f"{METADATA_ROOT}/{VERSION}/feature_coverage.csv",
    threads: 16
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=200,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.builder} --config {input.config} \
            --output-root {SNAPSHOT_ROOT} \
            --coverage-json {output.coverage_json} \
            --coverage-csv {output.coverage_csv} --workers {threads}
        """


rule unified_real_data_distribution_feature_audit_v1:
    input:
        manifest=lambda wc: snapshot_manifest(str(wc.family)),
        coverage=rules.unified_real_data_distribution_snapshot_v1.output.coverage_json,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
    output:
        f"{METADATA_ROOT}/{VERSION}/feature_completeness/{{family}}.json"
    wildcard_constraints:
        family="|".join(GROUPS),
    threads: 1
    resources:
        runtime=15 * 60,
        mem_mb=1000,
        disk_mb=100,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} audit --config {input.config} \
            --manifest {input.manifest} --family {wildcards.family} \
            --output {output}
        """


rule unified_real_data_distribution_raw_hist_v1:
    input:
        manifest=lambda wc: snapshot_manifest(str(wc.group)),
        ready=lambda wc: audit_path(str(wc.group)),
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
    output:
        f"{HISTOGRAM_ROOT}/{VERSION}/{{feature}}/{{group}}.root"
    wildcard_constraints:
        feature="|".join(FEATURES),
        group="|".join(GROUPS),
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=2000,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_real_data_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} histogram --config {input.config} \
            --manifest {input.manifest} --group {wildcards.group} \
            --feature {wildcards.feature} --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/hist.root" "{output}"
        test -r "{output}"
        """


rule unified_real_data_distribution_raw_hists_v1:
    input:
        [histogram_path(group, feature) for feature in FEATURES for group in GROUPS]


rule unified_real_data_distribution_plot_v1:
    input:
        hists=_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/unit_area/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/unit_area/{{feature}}.csv",
    params:
        hist_args=_plot_hist_args,
    wildcard_constraints:
        feature="|".join(FEATURES),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_real_data_plot.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} plot --config {input.config} \
            --feature {wildcards.feature} --normalization unit_area \
            {params.hist_args} --output "$tmp_dir/plot.pdf" \
            --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" "{output.pdf}"
        mv "$tmp_dir/summary.csv" "{output.summary}"
        """


rule unified_real_data_distributions_v1:
    input:
        [plot_path(feature, suffix) for feature in FEATURES for suffix in ("pdf", "csv")]
