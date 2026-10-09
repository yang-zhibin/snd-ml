"""Geometry-controlled 2023 preparation and testbeam distribution workflow."""

from pathlib import Path
import shlex

import pandas as pd
import yaml


REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
CONFIG_PATH = str(config.get(
    "testbeam_distribution_config",
    f"{REPO_ROOT}/snakemake/metadata/configs/unified_testbeam_distributions_v3.yaml",
))
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle)

VERSION = str(SETTINGS["version"])
FEATURES = tuple(SETTINGS["features"])
METADATA_ROOT = str(SETTINGS["outputs"]["metadata_root"]).rstrip("/")
SELECTION_ROOT = str(SETTINGS["outputs"]["selection_root"]).rstrip("/")
FEATURE_ROOT_2023 = str(SETTINGS["outputs"]["feature_root_2023"]).rstrip("/")
HISTOGRAM_ROOT = str(SETTINGS["outputs"]["histogram_root"]).rstrip("/")
PLOT_ROOT = str(SETTINGS["outputs"]["plot_root"]).rstrip("/")
MANIFEST_2023 = str(SETTINGS["source_manifests"]["testbeam_data_2023_h8"])
SUMMARY_2023 = f"{METADATA_ROOT}/testbeam_data_2023_h8_summary.json"
MANIFEST_BUILDER = f"{REPO_ROOT}/snakemake/metadata/prepare_unified_testbeam_2023_manifest.py"
INDEX_BUILDER = f"{REPO_ROOT}/convertData/build_indexed_cutflow.py"
CONVERTER = f"{REPO_ROOT}/convertData/digi_2_features.py"
VALIDATOR = f"{REPO_ROOT}/convertData/validate_unified_shower_feature_file.py"
DISTRIBUTION_SCRIPT = f"{REPO_ROOT}/evaluation_region_partitions/unified_feature_distributions.py"
SNAPSHOT_BUILDER = (
    f"{REPO_ROOT}/snakemake/metadata/build_unified_partial_feature_snapshots.py"
)
SNAPSHOT_ROOT = f"{METADATA_ROOT}/feature_snapshots"
SNAPSHOT_COVERAGE_JSON = f"{SNAPSHOT_ROOT}/coverage.json"
SNAPSHOT_COVERAGE_CSV = f"{SNAPSHOT_ROOT}/coverage.csv"


def expanded_groups():
    groups = {str(name): dict(value) for name, value in SETTINGS.get("groups", {}).items()}
    templates = SETTINGS.get("group_templates", {})
    for series_name, series in SETTINGS.get("group_series", {}).items():
        template = templates[str(series["template"])]
        for energy in series["energies"]:
            name = f"{series_name}_{int(energy)}"
            groups[name] = {
                "family": str(template["family"]),
                "geometry_setup_id": str(template["geometry_setup_id"]),
                "source_filters": {
                    "source_pdg_code": int(template["source_pdg_code"]),
                    "source_energy_gev": int(energy),
                },
            }
    return groups


GROUPS = expanded_groups()
PLOT_SETS = {
    str(name): tuple(value["groups"])
    for name, value in SETTINGS["plot_sets"].items()
}
PLOT_GEOMETRY_POLICIES = {
    str(name): str(value.get("geometry_policy", "same_geometry"))
    for name, value in SETTINGS["plot_sets"].items()
}
CROSS_GEOMETRY_PLOT_SETS = tuple(
    name
    for name, policy in PLOT_GEOMETRY_POLICIES.items()
    if policy == "cross_geometry_diagnostic"
)
COMBINED_PARTICLE_ENERGY_PLOT_SETS = (
    "data_combined_particle_energy",
    "mc_combined_particle_energy",
)
if not set(COMBINED_PARTICLE_ENERGY_PLOT_SETS).issubset(PLOT_SETS):
    raise ValueError(
        "combined particle-energy plot sets are missing from the configuration"
    )


def _manifest_2023_frame():
    checkpoint_result = checkpoints.unified_prepare_testbeam_2023_manifest_v1.get()
    return pd.read_csv(str(checkpoint_result.output.manifest), keep_default_na=False)


def _row(source_uid):
    frame = _manifest_2023_frame()
    rows = frame.loc[frame["source_uid"].astype(str).eq(str(source_uid))]
    if len(rows) != 1:
        raise ValueError(f"expected one 2023 manifest row for {source_uid}, got {len(rows)}")
    return rows.iloc[0]


def _value(wildcards, column):
    return _row(wildcards.source_uid)[column]


def _index_path(source_uid):
    return f"{SELECTION_ROOT}/{source_uid}/selected_entries.root"


def _index_summary_path(source_uid):
    return _index_path(source_uid) + ".validated.json"


def _feature_path(source_uid):
    return f"{FEATURE_ROOT_2023}/{source_uid}/feature_scifi_cluster_v1_{source_uid}.root"


def _feature_marker(source_uid):
    return _feature_path(source_uid) + ".validated.json"


def _all_2023_indexes(_wildcards):
    return [_index_summary_path(uid) for uid in _manifest_2023_frame()["source_uid"].astype(str)]


def _all_2023_features(_wildcards):
    return [_feature_marker(uid) for uid in _manifest_2023_frame()["source_uid"].astype(str)]


def _snapshot_manifest(family):
    return f"{SNAPSHOT_ROOT}/{family}.csv"


def _group_manifest(group):
    checkpoints.unified_prepare_testbeam_feature_snapshots_v3.get()
    family = str(GROUPS[str(group)]["family"])
    return _snapshot_manifest(family)


def _filtered_group_rows(group):
    family = str(GROUPS[str(group)]["family"])
    frame = pd.read_csv(_group_manifest(group), keep_default_na=False)
    frame = frame.loc[frame["family"].astype(str).eq(family)].copy()
    for column, expected in GROUPS[str(group)].get("source_filters", {}).items():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[numeric.eq(float(expected))].copy()
    if frame.empty:
        raise ValueError(f"no manifest sources match group {group}")
    return frame


def _group_markers(wildcards):
    return _filtered_group_rows(str(wildcards.group))["validation_record"].astype(str).tolist()


def _histogram_path(group, feature):
    return f"{HISTOGRAM_ROOT}/{VERSION}/{feature}/{group}.root"


def _plot_path(plot_set, feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/{plot_set}/unit_area/{feature}.{suffix}"


def _plot_hists(wildcards):
    return [_histogram_path(group, wildcards.feature)
            for group in PLOT_SETS[str(wildcards.plot_set)]]


def _plot_hist_args(wildcards, input):
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(PLOT_SETS[str(wildcards.plot_set)], input.hists)
    )


def _plot_geometry_policy(wildcards):
    return PLOT_GEOMETRY_POLICIES[str(wildcards.plot_set)]


localrules: unified_prepare_testbeam_2023_manifest_v1, unified_testbeam_2023_selection_index_v1, unified_prepare_testbeam_feature_snapshots_v3


checkpoint unified_prepare_testbeam_2023_manifest_v1:
    input:
        metadata=str(SETTINGS["inputs"]["testbeam_data_2023_metadata"]),
        config=CONFIG_PATH,
        builder=MANIFEST_BUILDER,
    output:
        manifest=MANIFEST_2023,
        summary=SUMMARY_2023,
    threads: 1
    resources:
        runtime=10 * 60,
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
        python3 {input.builder} --metadata {input.metadata} \
            --config {input.config} \
            --selection-root {SELECTION_ROOT} --feature-root {FEATURE_ROOT_2023} \
            --output-manifest {output.manifest} --output-summary {output.summary}
        """


rule unified_testbeam_2023_selection_index_v1:
    input:
        manifest=lambda wc: checkpoints.unified_prepare_testbeam_2023_manifest_v1.get().output.manifest,
        builder=INDEX_BUILDER,
    output:
        index=f"{SELECTION_ROOT}/{{source_uid}}/selected_entries.root",
        summary=f"{SELECTION_ROOT}/{{source_uid}}/selected_entries.root.validated.json",
    params:
        input_count=lambda wc: int(_value(wc, "input_event_count")),
        selected_count=lambda wc: int(_value(wc, "selected_event_count")),
    wildcard_constraints:
        source_uid=r"unified_tb23_src_[0-9a-f]{20}",
    threads: 1
    resources:
        runtime=5 * 60,
        mem_mb=1000,
        disk_mb=100,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 -c 'import ROOT; print(ROOT.__version__)'
        python3 {input.builder} --output {output.index} --summary {output.summary} \
            --source-uid {wildcards.source_uid} \
            --input-event-count {params.input_count} \
            --selected-event-count {params.selected_count}
        """


rule unified_testbeam_2023_selection_indexes_v1:
    input:
        _all_2023_indexes


rule unified_testbeam_2023_feature_v1:
    input:
        manifest=lambda wc: checkpoints.unified_prepare_testbeam_2023_manifest_v1.get().output.manifest,
        index=lambda wc: _index_path(wc.source_uid),
        index_summary=lambda wc: _index_summary_path(wc.source_uid),
        digi=lambda wc: str(_value(wc, "digi_path")),
        geo=lambda wc: str(_value(wc, "geo_path")),
        converter=CONVERTER,
        validator=VALIDATOR,
    output:
        feature=f"{FEATURE_ROOT_2023}/{{source_uid}}/feature_scifi_cluster_v1_{{source_uid}}.root",
        marker=f"{FEATURE_ROOT_2023}/{{source_uid}}/feature_scifi_cluster_v1_{{source_uid}}.root.validated.json",
    params:
        input_count=lambda wc: int(_value(wc, "input_event_count")),
        selected_count=lambda wc: int(_value(wc, "selected_event_count")),
        pdg_code=lambda wc: int(_value(wc, "source_pdg_code")),
        energy=lambda wc: int(_value(wc, "source_energy_gev")),
        manifest_version=lambda wc: str(_value(wc, "manifest_version")),
        geometry_id=lambda wc: str(_value(wc, "source_geometry_id")),
        geometry_checksum=lambda wc: str(_value(wc, "source_geometry_checksum")),
    wildcard_constraints:
        source_uid=r"unified_tb23_src_[0-9a-f]{20}",
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_tb23_feature.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            -p {input.index} -d {input.digi} -g {input.geo} -o "$tmp_dir/feature.root" \
            -t real_testbeam --tree-name cbmsim --selection-mode indexed_cutflow \
            --source-domain-id 6 --source-pdg-code {params.pdg_code} \
            --source-energy-gev {params.energy} --source-uid {wildcards.source_uid} \
            --source-geometry-id {params.geometry_id} \
            --source-geometry-checksum {params.geometry_checksum}
        converter_commit=$(git -C {REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_dir/feature.root" --manifest-version {params.manifest_version} \
            --expected-source-uid {wildcards.source_uid} --expected-source-domain-id 6 \
            --expected-source-pdg-code {params.pdg_code} \
            --expected-source-energy-gev {params.energy} \
            --expected-source-geometry-id {params.geometry_id} \
            --expected-source-geometry-checksum {params.geometry_checksum} \
            --expected-input-event-count {params.input_count} \
            --expected-selected-event-count {params.selected_count} \
            --sample-type real_testbeam --selection-mode indexed_cutflow \
            --input-tree cbmsim --expected-max-events 0 \
            --input-digi-path {input.digi} --input-cutflow-path {input.index} \
            --published-feature-path {output.feature} --converter-commit "$converter_commit" \
            --validation-record "$tmp_dir/validated.json"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_dir/feature.root" {output.feature}
        xrdcp -f "$tmp_dir/validated.json" {output.marker}
        test -r {output.feature}
        test -r {output.marker}
        """


rule unified_testbeam_2023_features_v1:
    input:
        _all_2023_features


checkpoint unified_prepare_testbeam_feature_snapshots_v3:
    input:
        config=CONFIG_PATH,
        builder=SNAPSHOT_BUILDER,
        source_manifests=[str(path) for path in SETTINGS["source_manifests"].values()],
        features_2023=_all_2023_features,
    output:
        manifests=[
            _snapshot_manifest(family)
            for family in SETTINGS["source_manifests"]
        ],
        coverage_json=SNAPSHOT_COVERAGE_JSON,
        coverage_csv=SNAPSHOT_COVERAGE_CSV,
    threads: 4
    resources:
        runtime=30 * 60,
        mem_mb=2000,
        disk_mb=500,
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


rule unified_testbeam_raw_hist_v3:
    input:
        manifest=lambda wc: _group_manifest(wc.group),
        validated=_group_markers,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_tb_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} histogram --config {input.config} \
            --manifest {input.manifest} --group {wildcards.group} \
            --feature {wildcards.feature} --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/hist.root" {output}
        test -r {output}
        """


rule unified_testbeam_raw_hists_v3:
    input:
        [_histogram_path(group, feature) for group in GROUPS for feature in FEATURES]


rule unified_testbeam_plot_v3:
    input:
        hists=_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/{{feature}}.csv",
    params:
        hist_args=_plot_hist_args,
        geometry_policy=_plot_geometry_policy,
    wildcard_constraints:
        plot_set="|".join(PLOT_SETS),
        feature="|".join(FEATURES),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=500,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDWRITEBYTECODE=1
        set +u
        source {REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_tb_plot.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} plot --config {input.config} \
            --plot-set {wildcards.plot_set} \
            --feature {wildcards.feature} --normalization unit_area \
            --geometry-policy {params.geometry_policy} \
            {params.hist_args} --output "$tmp_dir/plot.pdf" \
            --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" {output.pdf}
        mv "$tmp_dir/summary.csv" {output.summary}
        """


rule unified_testbeam_distributions_v3:
    input:
        [_plot_path(plot_set, feature, suffix)
         for plot_set in PLOT_SETS for feature in FEATURES for suffix in ("pdf", "csv")]


rule unified_testbeam_cross_geometry_distributions_v3:
    input:
        [_plot_path(plot_set, feature, suffix)
         for plot_set in CROSS_GEOMETRY_PLOT_SETS
         for feature in FEATURES for suffix in ("pdf", "csv")]


rule unified_testbeam_combined_particle_energy_plots_v3:
    input:
        [_plot_path(plot_set, feature, suffix)
         for plot_set in COMBINED_PARTICLE_ENERGY_PLOT_SETS
         for feature in FEATURES for suffix in ("pdf", "csv")]
