"""Standalone frozen-snapshot workflow for preliminary Stage-1/Stage-2 plots."""

from pathlib import Path
import shlex
import yaml

REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
CONFIG_PATH = f"{REPO_ROOT}/snakemake/metadata/configs/unified_partial_distributions_20261007_v1.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle)

VERSION = str(SETTINGS["version"])
FEATURES = tuple(SETTINGS["features"])
FAMILIES = tuple(SETTINGS["source_manifests"])
STAGE1_UNIT_GROUPS = tuple(SETTINGS["stages"]["stage1"]["unit_area_groups"])
STAGE1_EXPOSURE_GROUPS = tuple(SETTINGS["stages"]["stage1"]["common_exposure_groups"])
STAGE1_GROUPS = tuple(dict.fromkeys(STAGE1_UNIT_GROUPS + STAGE1_EXPOSURE_GROUPS))
STAGE2_COMPARISONS = {
    str(name): tuple(groups)
    for name, groups in SETTINGS["stages"]["stage2"]["comparisons"].items()
}
STAGE2_GROUPS = tuple(
    dict.fromkeys(group for groups in STAGE2_COMPARISONS.values() for group in groups)
)
METADATA_ROOT = str(SETTINGS["outputs"]["metadata_root"]).rstrip("/")
HISTOGRAM_ROOT = str(SETTINGS["outputs"]["histogram_root"]).rstrip("/")
PLOT_ROOT = str(SETTINGS["outputs"]["plot_root"]).rstrip("/")
SNAPSHOT_ROOT = f"{METADATA_ROOT}/{VERSION}/feature_snapshots"
SNAPSHOT_BUILDER = f"{REPO_ROOT}/snakemake/metadata/build_unified_partial_feature_snapshots.py"
DISTRIBUTION_SCRIPT = f"{REPO_ROOT}/evaluation_region_partitions/unified_feature_distributions.py"


def snapshot_manifest(family):
    return f"{SNAPSHOT_ROOT}/{family}.csv"


def audit_path(family):
    return f"{METADATA_ROOT}/{VERSION}/feature_completeness/{family}.json"


def group_family(group):
    return str(SETTINGS["groups"][str(group)]["family"])


def histogram_path(stage, group, feature):
    return f"{HISTOGRAM_ROOT}/{VERSION}/{stage}/{feature}/{group}.root"


def stage1_plot_path(normalization, feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/stage1/{normalization}/{feature}.{suffix}"


def stage2_plot_path(comparison, feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/stage2/{comparison}/unit_area/{feature}.{suffix}"


def stage1_plot_groups(normalization):
    if str(normalization) == "unit_area":
        return STAGE1_UNIT_GROUPS
    if str(normalization) == "common_exposure_19fb":
        return STAGE1_EXPOSURE_GROUPS
    raise ValueError(f"unknown Stage-1 normalization {normalization}")


def stage1_plot_hists(wildcards):
    return [histogram_path("stage1", group, wildcards.feature)
            for group in stage1_plot_groups(wildcards.normalization)]


def stage1_hist_args(wildcards, input):
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(stage1_plot_groups(wildcards.normalization), input.hists)
    )


def stage2_plot_hists(wildcards):
    return [histogram_path("stage2", group, wildcards.feature)
            for group in STAGE2_COMPARISONS[str(wildcards.comparison)]]


def stage2_hist_args(wildcards, input):
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(STAGE2_COMPARISONS[str(wildcards.comparison)], input.hists)
    )


localrules: unified_partial_feature_snapshot_v1


rule unified_partial_feature_snapshot_v1:
    input:
        config=CONFIG_PATH,
        builder=SNAPSHOT_BUILDER,
        source_manifests=list(SETTINGS["source_manifests"].values()),
    output:
        manifests=[snapshot_manifest(family) for family in FAMILIES],
        coverage_json=f"{METADATA_ROOT}/{VERSION}/coverage.json",
        coverage_csv=f"{METADATA_ROOT}/{VERSION}/coverage.csv",
    threads: 8
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
        hostname
        pwd
        command -v python3
        python3 --version
        python3 -c 'import pandas, yaml; print(pandas.__version__)'
        python3 {input.builder} \
            --config {input.config} \
            --output-root {SNAPSHOT_ROOT} \
            --coverage-json {output.coverage_json} \
            --coverage-csv {output.coverage_csv} \
            --workers {threads}
        """


rule unified_partial_feature_snapshots_v1:
    input:
        rules.unified_partial_feature_snapshot_v1.output


rule unified_partial_feature_audit_v1:
    input:
        manifest=lambda wc: snapshot_manifest(str(wc.family)),
        coverage=rules.unified_partial_feature_snapshot_v1.output.coverage_json,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        f"{METADATA_ROOT}/{VERSION}/feature_completeness/{{family}}.json"
    wildcard_constraints:
        family="|".join(FAMILIES),
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
            --manifest {input.manifest} --family {wildcards.family} --output {output}
        """


rule unified_partial_feature_audits_v1:
    input:
        [audit_path(family) for family in FAMILIES]


rule unified_partial_raw_hist_v1:
    input:
        manifest=lambda wc: snapshot_manifest(group_family(wc.group)),
        ready=lambda wc: audit_path(group_family(wc.group)),
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        f"{HISTOGRAM_ROOT}/{VERSION}/{{stage}}/{{feature}}/{{group}}.root"
    wildcard_constraints:
        stage="stage1|stage2",
        feature="|".join(FEATURES),
        group="|".join(STAGE1_GROUPS + STAGE2_GROUPS),
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_partial_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} histogram --config {input.config} \
            --manifest {input.manifest} --group {wildcards.group} \
            --feature {wildcards.feature} --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/hist.root" "{output}"
        test -r "{output}"
        """


rule unified_stage1_partial_raw_hists_v1:
    input:
        [histogram_path("stage1", group, feature)
         for feature in FEATURES for group in STAGE1_GROUPS]


rule unified_stage2_partial_raw_hists_v1:
    input:
        [histogram_path("stage2", group, feature)
         for feature in FEATURES for group in STAGE2_GROUPS]


rule unified_stage1_partial_plot_v1:
    input:
        hists=stage1_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/stage1/{{normalization}}/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/stage1/{{normalization}}/{{feature}}.csv",
    params:
        hist_args=stage1_hist_args,
    wildcard_constraints:
        normalization="unit_area|common_exposure_19fb",
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_partial_plot.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} plot --config {input.config} \
            --feature {wildcards.feature} --normalization {wildcards.normalization} \
            {params.hist_args} --output "$tmp_dir/plot.pdf" \
            --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" "{output.pdf}"
        mv "$tmp_dir/summary.csv" "{output.summary}"
        """


rule unified_stage2_partial_plot_v1:
    input:
        hists=stage2_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/stage2/{{comparison}}/unit_area/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/stage2/{{comparison}}/unit_area/{{feature}}.csv",
    params:
        hist_args=stage2_hist_args,
    wildcard_constraints:
        comparison="|".join(STAGE2_COMPARISONS),
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_partial_plot.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} plot --config {input.config} \
            --feature {wildcards.feature} --normalization unit_area \
            {params.hist_args} --output "$tmp_dir/plot.pdf" \
            --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" "{output.pdf}"
        mv "$tmp_dir/summary.csv" "{output.summary}"
        """


rule unified_stage1_partial_distributions_v1:
    input:
        [stage1_plot_path(normalization, feature, suffix)
         for normalization in ("unit_area", "common_exposure_19fb")
         for feature in FEATURES for suffix in ("pdf", "csv")]


rule unified_stage2_partial_distributions_v1:
    input:
        [stage2_plot_path(comparison, feature, suffix)
         for comparison in STAGE2_COMPARISONS
         for feature in FEATURES for suffix in ("pdf", "csv")]


rule unified_partial_distributions_v1:
    input:
        rules.unified_stage1_partial_distributions_v1.input,
        rules.unified_stage2_partial_distributions_v1.input,
