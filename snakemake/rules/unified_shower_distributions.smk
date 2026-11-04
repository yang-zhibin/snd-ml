import shlex


UNIFIED_DISTRIBUTION_CONFIG = (
    f"{UNIFIED_REPO_ROOT}/snakemake/metadata/configs/"
    "unified_shower_distributions_v1.yaml"
)
with open(UNIFIED_DISTRIBUTION_CONFIG, "r", encoding="utf-8") as _handle:
    UNIFIED_DISTRIBUTIONS = yaml.safe_load(_handle)

UNIFIED_DISTRIBUTION_SCRIPT = (
    f"{UNIFIED_REPO_ROOT}/evaluation_region_partitions/"
    "unified_feature_distributions.py"
)
UNIFIED_DISTRIBUTION_VERSION = str(UNIFIED_DISTRIBUTIONS["version"])
UNIFIED_DISTRIBUTION_METADATA_ROOT = str(
    UNIFIED_DISTRIBUTIONS["outputs"]["metadata_root"]
).rstrip("/")
UNIFIED_DISTRIBUTION_HIST_ROOT = str(
    UNIFIED_DISTRIBUTIONS["outputs"]["histogram_root"]
).rstrip("/")
UNIFIED_DISTRIBUTION_PLOT_ROOT = str(
    UNIFIED_DISTRIBUTIONS["outputs"]["plot_root"]
).rstrip("/")
UNIFIED_DISTRIBUTION_FEATURES = tuple(UNIFIED_DISTRIBUTIONS["features"])
UNIFIED_STAGE1_UNIT_GROUPS = tuple(
    UNIFIED_DISTRIBUTIONS["stages"]["stage1"]["unit_area_groups"]
)
UNIFIED_STAGE1_EXPOSURE_GROUPS = tuple(
    UNIFIED_DISTRIBUTIONS["stages"]["stage1"]["common_exposure_groups"]
)
UNIFIED_STAGE1_GROUPS = tuple(
    dict.fromkeys(UNIFIED_STAGE1_UNIT_GROUPS + UNIFIED_STAGE1_EXPOSURE_GROUPS)
)


def _unified_distribution_family(group):
    return str(UNIFIED_DISTRIBUTIONS["groups"][str(group)]["family"])


def _unified_distribution_hist_path(group, feature):
    return (
        f"{UNIFIED_DISTRIBUTION_HIST_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
        f"stage1/{feature}/{group}.root"
    )


def _unified_distribution_audit_path(family):
    return (
        f"{UNIFIED_DISTRIBUTION_METADATA_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
        f"feature_completeness/{family}.json"
    )


def _unified_distribution_plot_path(normalization, feature, suffix):
    return (
        f"{UNIFIED_DISTRIBUTION_PLOT_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
        f"stage1/{normalization}/{feature}.{suffix}"
    )


def _unified_distribution_plot_groups(normalization):
    if str(normalization) == "unit_area":
        return UNIFIED_STAGE1_UNIT_GROUPS
    if str(normalization) == "common_exposure_19fb":
        return UNIFIED_STAGE1_EXPOSURE_GROUPS
    raise ValueError(f"Unknown Stage-1 normalization {normalization}")


def _unified_distribution_plot_hists(wildcards):
    return [
        _unified_distribution_hist_path(group, wildcards.feature)
        for group in _unified_distribution_plot_groups(wildcards.normalization)
    ]


def _unified_distribution_hist_args(wildcards, input):
    groups = _unified_distribution_plot_groups(wildcards.normalization)
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(groups, input.hists)
    )


rule unified_stage1_feature_audit_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(str(wc.family)),
        config=UNIFIED_DISTRIBUTION_CONFIG,
        script=UNIFIED_DISTRIBUTION_SCRIPT,
        integrity=f"{UNIFIED_REPO_ROOT}/convertData/feature_integrity.py",
    output:
        (
            f"{UNIFIED_DISTRIBUTION_METADATA_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
            "feature_completeness/{family}.json"
        )
    wildcard_constraints:
        family="mc_neutrino|mc_muon_up",
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
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} audit \
            --config {input.config} \
            --manifest {input.manifest} \
            --family {wildcards.family} \
            --output {output}
        """


rule unified_stage1_feature_audits_v1:
    input:
        [
            _unified_distribution_audit_path(family)
            for family in ("mc_neutrino", "mc_muon_up")
        ]


rule unified_stage1_raw_hist_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(
            _unified_distribution_family(wc.group)
        ),
        ready=lambda wc: _unified_distribution_audit_path(
            _unified_distribution_family(wc.group)
        ),
        config=UNIFIED_DISTRIBUTION_CONFIG,
        script=UNIFIED_DISTRIBUTION_SCRIPT,
        integrity=f"{UNIFIED_REPO_ROOT}/convertData/feature_integrity.py",
    output:
        (
            f"{UNIFIED_DISTRIBUTION_HIST_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
            "stage1/{feature}/{group}.root"
        )
    wildcard_constraints:
        feature="|".join(UNIFIED_DISTRIBUTION_FEATURES),
        group="|".join(UNIFIED_STAGE1_GROUPS),
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
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_distribution_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} histogram \
            --config {input.config} \
            --manifest {input.manifest} \
            --group {wildcards.group} \
            --feature {wildcards.feature} \
            --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/hist.root" "{output}"
        test -r "{output}"
        """


rule unified_stage1_raw_hists_v1:
    input:
        [
            _unified_distribution_hist_path(group, feature)
            for feature in UNIFIED_DISTRIBUTION_FEATURES
            for group in UNIFIED_STAGE1_GROUPS
        ]


rule unified_stage1_plot_v1:
    input:
        hists=_unified_distribution_plot_hists,
        config=UNIFIED_DISTRIBUTION_CONFIG,
        script=UNIFIED_DISTRIBUTION_SCRIPT,
        integrity=f"{UNIFIED_REPO_ROOT}/convertData/feature_integrity.py",
    output:
        pdf=(
            f"{UNIFIED_DISTRIBUTION_PLOT_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
            "stage1/{normalization}/{feature}.pdf"
        ),
        summary=(
            f"{UNIFIED_DISTRIBUTION_PLOT_ROOT}/{UNIFIED_DISTRIBUTION_VERSION}/"
            "stage1/{normalization}/{feature}.csv"
        ),
    params:
        hist_args=_unified_distribution_hist_args,
    wildcard_constraints:
        normalization="unit_area|common_exposure_19fb",
        feature="|".join(UNIFIED_DISTRIBUTION_FEATURES),
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
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_distribution_plot.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} plot \
            --config {input.config} \
            --feature {wildcards.feature} \
            --normalization {wildcards.normalization} \
            {params.hist_args} \
            --output "$tmp_dir/plot.pdf" \
            --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" "{output.pdf}"
        mv "$tmp_dir/summary.csv" "{output.summary}"
        """


rule unified_stage1_plots_unit_area_v1:
    input:
        [
            _unified_distribution_plot_path("unit_area", feature, "pdf")
            for feature in UNIFIED_DISTRIBUTION_FEATURES
        ],
        [
            _unified_distribution_plot_path("unit_area", feature, "csv")
            for feature in UNIFIED_DISTRIBUTION_FEATURES
        ],


rule unified_stage1_plots_exposure_v1:
    input:
        [
            _unified_distribution_plot_path(
                "common_exposure_19fb", feature, "pdf"
            )
            for feature in UNIFIED_DISTRIBUTION_FEATURES
        ],
        [
            _unified_distribution_plot_path(
                "common_exposure_19fb", feature, "csv"
            )
            for feature in UNIFIED_DISTRIBUTION_FEATURES
        ],


rule unified_stage1_distributions_v1:
    input:
        rules.unified_stage1_plots_unit_area_v1.input,
        rules.unified_stage1_plots_exposure_v1.input,
