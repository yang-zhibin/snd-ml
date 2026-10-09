"""Build combined testbeam plots from frozen feature snapshots and histograms."""

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
HISTOGRAM_ROOT = str(SETTINGS["outputs"]["histogram_root"]).rstrip("/")
PLOT_ROOT = str(SETTINGS["outputs"]["plot_root"]).rstrip("/")
DISTRIBUTION_SCRIPT = (
    f"{REPO_ROOT}/evaluation_region_partitions/unified_feature_distributions.py"
)
SNAPSHOT_ROOT = f"{METADATA_ROOT}/feature_snapshots"
COMBINED_PLOT_SETS = (
    "data_combined_particle_energy",
    "mc_combined_particle_energy",
)


def expanded_groups():
    groups = {
        str(name): dict(value)
        for name, value in SETTINGS.get("groups", {}).items()
    }
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
    name: tuple(SETTINGS["plot_sets"][name]["groups"])
    for name in COMBINED_PLOT_SETS
}
PLOT_GEOMETRY_POLICIES = {
    name: str(SETTINGS["plot_sets"][name]["geometry_policy"])
    for name in COMBINED_PLOT_SETS
}
COMBINED_GROUPS = tuple(dict.fromkeys(
    group for plot_set in COMBINED_PLOT_SETS for group in PLOT_SETS[plot_set]
))
COMBINED_FAMILIES = {GROUPS[group]["family"] for group in COMBINED_GROUPS}
if not COMBINED_FAMILIES.issubset({
    "testbeam_mc_2024", "testbeam_data_2024", "testbeam_data_2023_h8",
}):
    raise ValueError(f"unexpected combined-plot families: {COMBINED_FAMILIES}")


def _snapshot_manifest(family):
    return f"{SNAPSHOT_ROOT}/{family}.csv"


def _filtered_group_rows(group):
    family = str(GROUPS[group]["family"])
    manifest = _snapshot_manifest(family)
    if not Path(manifest).is_file():
        raise FileNotFoundError(
            f"missing frozen feature snapshot {manifest}; run the preparation workflow first"
        )
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame = frame.loc[frame["family"].astype(str).eq(family)].copy()
    for column, expected in GROUPS[group].get("source_filters", {}).items():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[numeric.eq(float(expected))].copy()
    available = frame["feature_scifi_cluster_v1_path"].map(
        lambda value: Path(str(value)).is_file()
    ) & frame["validation_record"].map(
        lambda value: Path(str(value)).is_file()
    )
    frame = frame.loc[available].copy()
    if frame.empty:
        raise ValueError(f"no frozen snapshot sources match group {group}")
    return frame


def _group_manifest(wildcards):
    return _snapshot_manifest(str(GROUPS[str(wildcards.group)]["family"]))


def _group_markers(wildcards):
    return _filtered_group_rows(str(wildcards.group))[
        "validation_record"
    ].astype(str).tolist()


def _histogram_path(group, feature):
    return f"{HISTOGRAM_ROOT}/{VERSION}/{feature}/{group}.root"


def _plot_path(plot_set, feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/{plot_set}/unit_area/{feature}.{suffix}"


def _derived_plot_path(plot_set, kind, feature, suffix):
    return f"{PLOT_ROOT}/{VERSION}/{plot_set}/unit_area/{kind}/{feature}.{suffix}"


def _plot_hists(wildcards):
    return [
        _histogram_path(group, wildcards.feature)
        for group in PLOT_SETS[str(wildcards.plot_set)]
    ]


def _plot_hist_args(wildcards, input):
    return " ".join(
        f"--hist {shlex.quote(str(group) + '=' + str(path))}"
        for group, path in zip(
            PLOT_SETS[str(wildcards.plot_set)], input.hists
        )
    )


def _plot_geometry_policy(wildcards):
    return PLOT_GEOMETRY_POLICIES[str(wildcards.plot_set)]


rule unified_testbeam_combined_raw_hist_v3:
    input:
        manifest=_group_manifest,
        validated=_group_markers,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        f"{HISTOGRAM_ROOT}/{VERSION}/{{feature}}/{{group}}.root"
    wildcard_constraints:
        feature="|".join(FEATURES),
        group="|".join(COMBINED_GROUPS),
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
            --feature {wildcards.feature} --allow-missing-validated-sources \
            --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname "{output}")"
        xrdcp -f "$tmp_dir/hist.root" {output}
        test -r {output}
        """


rule unified_testbeam_combined_plot_v3:
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
        plot_set="|".join(COMBINED_PLOT_SETS),
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


rule unified_testbeam_combined_faceted_plot_v3:
    input:
        hists=_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/faceted/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/faceted/{{feature}}.csv",
    params:
        hist_args=_plot_hist_args,
        geometry_policy=_plot_geometry_policy,
    wildcard_constraints:
        plot_set="|".join(COMBINED_PLOT_SETS),
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_tb_facet.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} faceted-plot --config {input.config} \
            --plot-set {wildcards.plot_set} --feature {wildcards.feature} \
            --geometry-policy {params.geometry_policy} {params.hist_args} \
            --output "$tmp_dir/plot.pdf" --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" {output.pdf}
        mv "$tmp_dir/summary.csv" {output.summary}
        """


rule unified_testbeam_combined_energy_summary_plot_v3:
    input:
        hists=_plot_hists,
        config=CONFIG_PATH,
        script=DISTRIBUTION_SCRIPT,
    output:
        pdf=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/energy_summary/{{feature}}.pdf",
        summary=f"{PLOT_ROOT}/{VERSION}/{{plot_set}}/unit_area/energy_summary/{{feature}}.csv",
    params:
        hist_args=_plot_hist_args,
        geometry_policy=_plot_geometry_policy,
    wildcard_constraints:
        plot_set="|".join(COMBINED_PLOT_SETS),
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
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_tb_summary.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} energy-summary-plot --config {input.config} \
            --plot-set {wildcards.plot_set} --feature {wildcards.feature} \
            --geometry-policy {params.geometry_policy} {params.hist_args} \
            --output "$tmp_dir/plot.pdf" --summary "$tmp_dir/summary.csv"
        mkdir -p "$(dirname "{output.pdf}")"
        mv "$tmp_dir/plot.pdf" {output.pdf}
        mv "$tmp_dir/summary.csv" {output.summary}
        """



rule unified_testbeam_combined_particle_energy_plots_v3:
    input:
        [
            _plot_path(plot_set, feature, suffix)
            for plot_set in COMBINED_PLOT_SETS
            for feature in FEATURES
            for suffix in ("pdf", "csv")
        ]


rule unified_testbeam_combined_readable_plots_v3:
    input:
        [
            _derived_plot_path(plot_set, kind, feature, suffix)
            for plot_set in COMBINED_PLOT_SETS
            for kind in ("faceted", "energy_summary")
            for feature in FEATURES
            for suffix in ("pdf", "csv")
        ]


rule unified_testbeam_combined_scifi_cluster_qdc_per_hit_plot_v3:
    input:
        [
            _plot_path(plot_set, "scifi_cluster_qdc_per_hit", suffix)
            for plot_set in COMBINED_PLOT_SETS
            for suffix in ("pdf", "csv")
        ]
