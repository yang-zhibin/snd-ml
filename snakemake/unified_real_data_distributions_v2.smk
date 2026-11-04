"""Standalone histogram-bundle and PDF-only workflow for 2024 real data."""

from pathlib import Path
import shlex

import yaml


REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
HIST_CONFIG = str(config.get(
    "real_data_histogram_config",
    f"{REPO_ROOT}/snakemake/metadata/configs/unified_real_data_histograms_v2.yaml",
))
PLOT_CONFIG = str(config.get(
    "real_data_plot_config",
    f"{REPO_ROOT}/snakemake/metadata/configs/unified_real_data_plots_v2.yaml",
))
with open(HIST_CONFIG, "r", encoding="utf-8") as handle:
    HIST_SETTINGS = yaml.safe_load(handle)
with open(PLOT_CONFIG, "r", encoding="utf-8") as handle:
    PLOT_SETTINGS = yaml.safe_load(handle)


def _repo_path(value):
    path = Path(str(value))
    return str(path if path.is_absolute() else Path(REPO_ROOT) / path)


BASE_CONFIG = _repo_path(HIST_SETTINGS["base_distribution_config"])
DEFINITIONS_CONFIG = _repo_path(HIST_SETTINGS["histogram_definitions_config"])
with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
    BASE_SETTINGS = yaml.safe_load(handle)
with open(DEFINITIONS_CONFIG, "r", encoding="utf-8") as handle:
    DEFINITIONS_SETTINGS = yaml.safe_load(handle)

GROUPS = {str(name): dict(value) for name, value in BASE_SETTINGS["groups"].items()}
COHORTS = {
    str(name): tuple(value["groups"])
    for name, value in PLOT_SETTINGS["cohorts"].items()
}
ACTIVE_GROUPS = tuple(dict.fromkeys(
    group for groups in COHORTS.values() for group in groups
))
if not set(ACTIVE_GROUPS).issubset(GROUPS):
    raise ValueError("V2 plot configuration names an unknown real-data group")

FAMILIES = tuple(dict.fromkeys(str(GROUPS[group]["family"]) for group in ACTIVE_GROUPS))
FEATURES_1D = tuple(DEFINITIONS_SETTINGS["features"])
FEATURES_2D = tuple(DEFINITIONS_SETTINGS["features2d"])
HIST_VERSION = str(HIST_SETTINGS["version"])
PLOT_VERSION = str(PLOT_SETTINGS["version"])
BUNDLE_ROOT = str(HIST_SETTINGS["outputs"]["bundle_root"]).rstrip("/")
METADATA_ROOT = str(HIST_SETTINGS["outputs"]["metadata_root"]).rstrip("/")
SOURCE_METADATA_ROOT = str(BASE_SETTINGS["outputs"]["metadata_root"]).rstrip("/")
SNAPSHOT_ROOT = f"{SOURCE_METADATA_ROOT}/{BASE_SETTINGS['version']}/feature_snapshots"
COVERAGE_JSON = f"{SOURCE_METADATA_ROOT}/{BASE_SETTINGS['version']}/feature_coverage.json"
PLOT_ROOT = str(PLOT_SETTINGS["output_root"]).rstrip("/")
BUNDLE_SCRIPT = f"{REPO_ROOT}/evaluation_region_partitions/unified_histogram_bundles.py"
PLOT_SCRIPT = f"{REPO_ROOT}/evaluation_region_partitions/plot_unified_histogram_bundles.py"
BUNDLE_IMPLEMENTATION = f"{REPO_ROOT}/evaluation_region_partitions/unified_testbeam_histogram_bundles.py"
PLOT_IMPLEMENTATION = f"{REPO_ROOT}/evaluation_region_partitions/plot_unified_testbeam_histogram_bundles.py"
COVERAGE_SCRIPT = f"{REPO_ROOT}/evaluation_region_partitions/validate_unified_feature_snapshot_coverage.py"
COVERAGE_RECEIPT = f"{METADATA_ROOT}/feature_snapshot_coverage.validated.json"
COVERAGE_SETTINGS = dict(PLOT_SETTINGS.get("coverage", {}))
COVERAGE_POLICY = str(COVERAGE_SETTINGS.get("policy", "complete"))
MINIMUM_AVAILABLE_SOURCES = int(
    COVERAGE_SETTINGS.get("minimum_available_sources", 1)
)
ALLOWED_UNAVAILABLE_STATUSES = tuple(
    str(value)
    for value in COVERAGE_SETTINGS.get("allowed_unavailable_statuses", [])
)


def _manifest(group):
    return f"{SNAPSHOT_ROOT}/{GROUPS[group]['family']}.csv"


def _group_manifest(wildcards):
    return _manifest(str(wildcards.group))


def _audit(group):
    return f"{METADATA_ROOT}/audits/{group}.json"


def _bundle(group):
    return f"{BUNDLE_ROOT}/{HIST_VERSION}/{group}.root"


def _receipt(group):
    return f"{METADATA_ROOT}/bundle_validation/{group}.validated.json"


def _cohort_bundles(wildcards):
    return [_bundle(group) for group in COHORTS[str(wildcards.cohort)]]


def _cohort_receipts(wildcards):
    return [_receipt(group) for group in COHORTS[str(wildcards.cohort)]]


def _bundle_args(wildcards, input):
    return " ".join(
        f"--bundle {shlex.quote(group + '=' + str(path))}"
        for group, path in zip(COHORTS[str(wildcards.cohort)], input.bundles)
    )


def _plot(cohort, kind, feature):
    return f"{PLOT_ROOT}/{PLOT_VERSION}/{cohort}/unit_area/{kind}/{feature}.pdf"


ALL_1D_PLOTS = [
    _plot(cohort, "faceted", feature)
    for cohort in COHORTS
    for feature in FEATURES_1D
]
ALL_2D_PLOTS = [
    _plot(cohort, "two_dimensional/faceted", feature)
    for cohort in COHORTS
    for feature in FEATURES_2D
]
ALL_PLOTS = ALL_1D_PLOTS + ALL_2D_PLOTS


localrules:
    unified_real_data_feature_snapshot_coverage_v2,
    unified_real_data_plot_manifest_v2


rule unified_real_data_feature_snapshot_coverage_v2:
    input:
        coverage=COVERAGE_JSON,
        sources=[str(BASE_SETTINGS["source_manifests"][family]) for family in FAMILIES],
        selection_coverage=str(BASE_SETTINGS["inputs"]["selection_coverage"]),
        selection_manifest=str(BASE_SETTINGS["inputs"]["selection_manifest"]),
        selection_policy=_repo_path(BASE_SETTINGS["inputs"]["selection_policy"]),
        contract=f"{REPO_ROOT}/convertData/snapshot_contract.py",
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
        manifests=[f"{SNAPSHOT_ROOT}/{family}.csv" for family in FAMILIES],
        script=COVERAGE_SCRIPT,
        environment=f"{REPO_ROOT}/snakemake/envs/env_lcg.sh",
    output:
        COVERAGE_RECEIPT
    params:
        families=" ".join(f"--family {shlex.quote(family)}" for family in FAMILIES),
        policy=shlex.quote(COVERAGE_POLICY),
        minimum=MINIMUM_AVAILABLE_SOURCES,
        allowed=" ".join(
            f"--allowed-unavailable-status {shlex.quote(status)}"
            for status in ALLOWED_UNAVAILABLE_STATUSES
        ),
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        python3 {input.script} --coverage {input.coverage} {params.families} \
            --policy {params.policy} \
            --minimum-available-sources {params.minimum} {params.allowed} \
            --output {output}
        """


rule unified_real_data_feature_branch_audit_v2:
    input:
        manifest=_group_manifest,
        coverage=COVERAGE_RECEIPT,
        config=HIST_CONFIG,
        base_config=BASE_CONFIG,
        definitions=DEFINITIONS_CONFIG,
        script=BUNDLE_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
        implementation=BUNDLE_IMPLEMENTATION,
    output:
        f"{METADATA_ROOT}/audits/{{group}}.json"
    wildcard_constraints:
        group="|".join(ACTIVE_GROUPS),
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, pandas, yaml'
        python3 {input.script} audit --config {input.config} \
            --manifest {input.manifest} --group {wildcards.group} --output {output}
        """


rule unified_real_data_histogram_bundle_v2:
    input:
        manifest=_group_manifest,
        audit=lambda wc: _audit(str(wc.group)),
        config=HIST_CONFIG,
        base_config=BASE_CONFIG,
        definitions=DEFINITIONS_CONFIG,
        script=BUNDLE_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
        implementation=BUNDLE_IMPLEMENTATION,
    output:
        f"{BUNDLE_ROOT}/{HIST_VERSION}/{{group}}.root"
    wildcard_constraints:
        group="|".join(ACTIVE_GROUPS),
    threads: 1
    resources:
        runtime=8 * 60 * 60,
        mem_mb=8000,
        disk_mb=6000,
        nvidia_gpu=0,
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, pandas, yaml'
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_real_bundle.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} build --config {input.config} \
            --manifest {input.manifest} --group {wildcards.group} \
            --output "$tmp_dir/bundle.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/bundle.root" {output}
        test -r {output}
        """


rule unified_real_data_validate_histogram_bundle_v2:
    input:
        bundle=lambda wc: _bundle(str(wc.group)),
        config=HIST_CONFIG,
        base_config=BASE_CONFIG,
        definitions=DEFINITIONS_CONFIG,
        script=BUNDLE_SCRIPT,
        integrity=f"{REPO_ROOT}/convertData/feature_integrity.py",
        implementation=BUNDLE_IMPLEMENTATION,
    output:
        f"{METADATA_ROOT}/bundle_validation/{{group}}.validated.json"
    wildcard_constraints:
        group="|".join(ACTIVE_GROUPS),
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        python3 {input.script} validate --config {input.config} \
            --bundle {input.bundle} --group {wildcards.group} --output {output}
        """


rule unified_real_data_histogram_bundles_v2:
    input:
        [_receipt(group) for group in ACTIVE_GROUPS]


rule unified_real_data_faceted_1d_plot_v2:
    input:
        bundles=_cohort_bundles,
        receipts=_cohort_receipts,
        config=PLOT_CONFIG,
        script=PLOT_SCRIPT,
        implementation=PLOT_IMPLEMENTATION,
    output:
        f"{PLOT_ROOT}/{PLOT_VERSION}/{{cohort}}/unit_area/faceted/{{feature}}.pdf"
    params:
        bundle_args=_bundle_args,
    wildcard_constraints:
        cohort="|".join(COHORTS),
        feature="|".join(FEATURES_1D),
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        python3 {input.script} faceted-1d --config {input.config} \
            --cohort {wildcards.cohort} --feature {wildcards.feature} \
            {params.bundle_args} --output {output}
        """


rule unified_real_data_faceted_2d_plot_v2:
    input:
        bundles=_cohort_bundles,
        receipts=_cohort_receipts,
        config=PLOT_CONFIG,
        script=PLOT_SCRIPT,
        implementation=PLOT_IMPLEMENTATION,
    output:
        f"{PLOT_ROOT}/{PLOT_VERSION}/{{cohort}}/unit_area/two_dimensional/faceted/{{feature}}.pdf"
    params:
        bundle_args=_bundle_args,
    wildcard_constraints:
        cohort="|".join(COHORTS),
        feature="|".join(FEATURES_2D),
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=3000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        python3 {input.script} faceted-2d --config {input.config} \
            --cohort {wildcards.cohort} --feature {wildcards.feature} \
            {params.bundle_args} --output {output}
        """


rule unified_real_data_plot_manifest_v2:
    input:
        plots=ALL_PLOTS,
        bundles=[_bundle(group) for group in ACTIVE_GROUPS],
        receipts=[_receipt(group) for group in ACTIVE_GROUPS],
        coverage=COVERAGE_RECEIPT,
        config=PLOT_CONFIG,
        script=PLOT_SCRIPT,
        implementation=PLOT_IMPLEMENTATION,
    output:
        f"{PLOT_ROOT}/{PLOT_VERSION}/plot_manifest.json"
    params:
        bundle_args=" ".join(
            f"--bundle {shlex.quote(group + '=' + _bundle(group))}"
            for group in ACTIVE_GROUPS
        ),
        plot_args=" ".join(f"--plot {shlex.quote(path)}" for path in ALL_PLOTS),
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        python3 {input.script} manifest --config {input.config} \
            {params.bundle_args} {params.plot_args} \
            --coverage-receipt {input.coverage} --output {output}
        """


rule unified_real_data_all_plots_v2:
    input:
        f"{PLOT_ROOT}/{PLOT_VERSION}/plot_manifest.json"
