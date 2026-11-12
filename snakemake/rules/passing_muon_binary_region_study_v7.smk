import sys
sys.path.insert(0, str(Path(workflow.basedir).resolve()))
from output_paths import resolve_output

# V7: the V6 detector study plus data-only flat-v2 reconstructed-track diagnostics.

PASSING_MUON_V7_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
with open(PASSING_MUON_V7_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V7 = yaml.safe_load(stream)

PASSING_MUON_V7_SELECTIONS = ["coverage_loose", "coverage_tight"]
PASSING_MUON_V7_RECO_PLOT_GROUPS = list(
    PASSING_MUON_V7["reco_track_diagnostics"]["plot_groups"]
)
PASSING_MUON_V7_COMPOSITION_TRACK_KEY = str(
    PASSING_MUON_V7["background_composition"]["track_definition"]["key"]
)
_pmv7_root = PASSING_MUON_V7["output"]["root_dir"].rstrip("/")
_pmv7_plot_root = (
    resolve_output(PASSING_MUON_V7['output']['plot_dir'].rstrip('/'), PERSONAL_WORK_SPACE)
)
_pmv7_track_input_root = PASSING_MUON_V7["output"]["track_input_root"].rstrip("/")
PASSING_MUON_V7_TRACK_MANIFEST = (
    resolve_output(PASSING_MUON_V7['output']['track_manifest'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_MC_UP_SOURCE_METADATA = (
    resolve_output(PASSING_MUON_V7['mc_muon_up']['source_metadata'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_MC_UP_METADATA = (
    resolve_output(PASSING_MUON_V7['mc_muon_up']['metadata_csv'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_MC_UP_VALIDATION = (
    resolve_output(PASSING_MUON_V7['mc_muon_up']['validation_csv'], PERSONAL_WORK_SPACE)
)


def _passing_muon_v7_mc_up_features():
    metadata = pd.read_csv(
        PASSING_MUON_V7_MC_UP_SOURCE_METADATA, keep_default_na=False
    )
    mc_config = PASSING_MUON_V7["mc_muon_up"]
    source_partitions = {
        str(definition["source_partition"])
        for definition in mc_config["productions"].values()
    }
    rows = metadata.loc[
        metadata["data_type"].astype(str).eq("MC_muon")
        & metadata["subfolder"].astype(str).eq(str(mc_config["subfolder"]))
        & metadata["partition"].astype(str).isin(source_partitions)
    ].copy()
    expected = int(mc_config["expected_source_files"])
    if len(rows) != expected:
        raise ValueError(
            f"V7 expected {expected} MC-muon-up rows, found {len(rows)}"
        )
    return sorted(
        os.path.join(str(row.output_base_path), str(row.feature_path))
        if not os.path.isabs(str(row.feature_path))
        else str(row.feature_path)
        for row in rows.itertuples()
    )


PASSING_MUON_V7_MC_UP_FEATURES = _passing_muon_v7_mc_up_features()
_pmv7_track_table = PASSING_MUON_TRACK_V2_TABLE.loc[
    pd.to_numeric(PASSING_MUON_TRACK_V2_TABLE["run"]) == int(PASSING_MUON_V7["run"])
].copy()
_pmv7_track_by_source = {
    f"{int(source):03d}": sorted(group["track_feature_full_path"].astype(str))
    for source, group in _pmv7_track_table.groupby("source_file_index")
}
if set(_pmv7_track_by_source) != set(PASSING_MUON_V2_SOURCES):
    raise ValueError(
        "V7 flat-v2 source coverage does not match the validated Run-8329 DQ sources"
    )
_pmv7_track_expected_by_source = {
    f"{int(source):03d}": int(pd.to_numeric(group["n_saved_event"]).sum())
    for source, group in _pmv7_track_table.groupby("source_file_index")
}
_pmv7_track_audit_dir = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
    "passing_muon_run8329_detector_selection_study_v7/track_input_audits"
)


def _pmv7_track_input(source):
    return (
        f"{_pmv7_track_input_root}/"
        f"passingMuonDQ_flat_v2_run008329_source{source}.root"
    )


def _pmv7_track_audit(source):
    return f"{_pmv7_track_audit_dir}/source{source}.csv"


PASSING_MUON_V7_TRACK_INPUTS = [
    _pmv7_track_input(source) for source in PASSING_MUON_V2_SOURCES
]
PASSING_MUON_V7_TRACK_AUDITS = [
    _pmv7_track_audit(source) for source in PASSING_MUON_V2_SOURCES
]
PASSING_MUON_V7_DETECTOR_HIST = (
    f"{_pmv7_root}/histograms/detector_selection_combined.root"
)
PASSING_MUON_V7_COUNT_HIST = (
    f"{_pmv7_root}/histograms/detector_selection_count_diagnostics.root"
)
PASSING_MUON_V7_RECO_HIST = (
    f"{_pmv7_root}/histograms/reco_track_diagnostics_flat_v2.root"
)
PASSING_MUON_V7_COMPOSITION_HIST = (
    f"{_pmv7_root}/histograms/background_composition_flat_v2.root"
)
PASSING_MUON_V7_DETECTOR_PLOTS = [
    f"{_pmv7_plot_root}/{selection}.pdf"
    for selection in PASSING_MUON_V7_SELECTIONS
]
PASSING_MUON_V7_COUNT_PLOTS = [
    f"{_pmv7_plot_root}/{selection}_counts.pdf"
    for selection in PASSING_MUON_V7_SELECTIONS
]
PASSING_MUON_V7_COVERAGE_STACKED_COUNT_PLOTS = [
    f"{_pmv7_plot_root}/{selection}_stacked_counts.pdf"
    for selection in PASSING_MUON_V7_SELECTIONS
]
PASSING_MUON_V7_RECO_PLOTS = [
    f"{_pmv7_plot_root}/{selection}_reco_{plot_group}.pdf"
    for selection in PASSING_MUON_V7_SELECTIONS
    for plot_group in PASSING_MUON_V7_RECO_PLOT_GROUPS
]
PASSING_MUON_V7_COMPOSITION_PLOTS = [
    f"{_pmv7_plot_root}/{selection}_composition_"
    f"{PASSING_MUON_V7_COMPOSITION_TRACK_KEY}.pdf"
    for selection in PASSING_MUON_V7_SELECTIONS
]


rule filter_passing_muon_flat_v2_by_real_dq_v7:
    input:
        tracks=lambda wc: _pmv7_track_by_source[wc.source],
        script=(
            f"{PERSONAL_WORK_SPACE}/convertData/"
            "filter_passing_muon_by_real_data_dq.py"
        ),
    output:
        root=_pmv7_track_input_root
        + "/passingMuonDQ_flat_v2_run008329_source{source}.root",
        audit=_pmv7_track_audit_dir + "/source{source}.csv",
    wildcard_constraints:
        source="|".join(PASSING_MUON_V2_SOURCES)
    params:
        # The validated V2 real-data DQ feature is an immutable reference for
        # this study. Keep it out of ``input`` so Snakemake does not connect it
        # to the protected historical production rule and try to rebuild it
        # merely because converter source code has a newer timestamp.
        real=lambda wc: _pmv2_real_feature(wc.source),
        source_index=lambda wc: int(wc.source),
        expected=lambda wc: _pmv7_track_expected_by_source[wc.source],
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=3500, disk_mb=5000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/passing_muon_flat_v2_dq.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --real-feature {params.real} \
          --passing-features {input.tracks:q} \
          --source-file-index {params.source_index} \
          --run {PASSING_MUON_V7[run]} \
          --expected-passing-events {params.expected} \
          --output-root "$tmp_dir/output.root" \
          --published-root {output.root} \
          --output-audit {output.audit}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule prepare_passing_muon_v7_track_manifest:
    input:
        metadata=PASSING_MUON_TRACK_V2_METADATA,
        roots=PASSING_MUON_V7_TRACK_INPUTS,
        audits=PASSING_MUON_V7_TRACK_AUDITS,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "prepare_passing_muon_v7_track_manifest.py"
        ),
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        manifest=PASSING_MUON_V7_TRACK_MANIFEST,
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=2000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --track-metadata {input.metadata} \
          --audits {input.audits:q} \
          --run {PASSING_MUON_V7[run]} \
          --output {output.manifest}
        """


rule prepare_mc_muon_up_reference_v7:
    input:
        source_metadata=PASSING_MUON_V7_MC_UP_SOURCE_METADATA,
        config=PASSING_MUON_V7_CONFIG,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "prepare_mc_muon_up_reference_v5.py"
        ),
        histogrammer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
    output:
        metadata=PASSING_MUON_V7_MC_UP_METADATA,
        validation=PASSING_MUON_V7_MC_UP_VALIDATION,
    threads: 4
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --source-metadata {input.source_metadata} \
          --study-config {input.config} \
          --hist-config {input.hist_config} \
          --output-metadata {output.metadata} \
          --output-validation {output.validation} \
          --threads {threads}
        """


rule make_passing_muon_binary_region_hist_v7:
    input:
        mc_muon_up_metadata=PASSING_MUON_V7_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V7_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v7.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V7_DETECTOR_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    params:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/detector_selection_hist_v7.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {params.partition_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule make_passing_muon_binary_region_count_hist_v7:
    input:
        mc_muon_up_metadata=PASSING_MUON_V7_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V7_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v7.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V7_COUNT_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    params:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/detector_count_hist_v7.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {params.partition_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule make_passing_muon_reco_diagnostics_v7:
    input:
        manifest=PASSING_MUON_V7_TRACK_MANIFEST,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_reco_diagnostics_v7.py"
        ),
        config_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_reco.py"
        ),
        detector_engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V7_RECO_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/reco_diagnostics_v7.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --track-manifest {input.manifest} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule make_passing_muon_background_composition_v7:
    input:
        manifest=PASSING_MUON_V7_TRACK_MANIFEST,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_background_composition_v7.py"
        ),
        config_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_background_composition_v7.py"
        ),
        reco_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_reco.py"
        ),
        reco_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_reco_diagnostics_v7.py"
        ),
        detector_engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V7_COMPOSITION_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/background_composition_v7.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --track-manifest {input.manifest} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule plot_passing_muon_binary_region_study_v7:
    input:
        root=PASSING_MUON_V7_DETECTOR_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_binary_region_study_v6.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv7_plot_root + "/{selection}.pdf",
    wildcard_constraints:
        selection="coverage_loose|coverage_tight"
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=3000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --selection {wildcards.selection} \
          --output-pdf {output.pdf}
        """


rule plot_passing_muon_reco_diagnostics_v7:
    input:
        root=PASSING_MUON_V7_RECO_HIST,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_reco_diagnostics_v7.py"
        ),
        config_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_reco.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv7_plot_root + "/{selection}_reco_{plot_group}.pdf",
    wildcard_constraints:
        selection="coverage_loose|coverage_tight",
        plot_group="simple|hough"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --region-config {input.region_config} \
          --selection {wildcards.selection} \
          --plot-group {wildcards.plot_group} \
          --output-pdf {output.pdf}
        """


rule plot_passing_muon_binary_region_counts_v7:
    input:
        root=PASSING_MUON_V7_COUNT_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_binary_region_counts_v7.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv7_plot_root + "/{selection}_counts.pdf",
    wildcard_constraints:
        selection="coverage_loose|coverage_tight"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --selection {wildcards.selection} \
          --output-pdf {output.pdf}
        """


rule plot_passing_muon_coverage_stacked_counts_v7:
    input:
        root=PASSING_MUON_V7_DETECTOR_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_coverage_stacked_counts_v7.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv7_plot_root + "/{selection}_stacked_counts.pdf",
    wildcard_constraints:
        selection="coverage_loose|coverage_tight"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --selection {wildcards.selection} \
          --output-pdf {output.pdf}
        """


rule plot_passing_muon_background_composition_v7:
    input:
        root=PASSING_MUON_V7_COMPOSITION_HIST,
        region_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_background_composition_v7.py"
        ),
        config_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_background_composition_v7.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=(
            _pmv7_plot_root + "/{selection}_composition_"
            + PASSING_MUON_V7_COMPOSITION_TRACK_KEY + ".pdf"
        ),
    wildcard_constraints:
        selection="coverage_loose|coverage_tight"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --region-config {input.region_config} \
          --selection {wildcards.selection} \
          --output-pdf {output.pdf}
        """


rule passing_muon_run8329_binary_region_study_hist_v7:
    input:
        detector=PASSING_MUON_V7_DETECTOR_HIST,
        reco=PASSING_MUON_V7_RECO_HIST,


rule passing_muon_run8329_binary_region_study_count_plots_v7:
    input:
        counts=PASSING_MUON_V7_COUNT_PLOTS,


rule passing_muon_run8329_coverage_stacked_count_plots_v7:
    input:
        PASSING_MUON_V7_COVERAGE_STACKED_COUNT_PLOTS


rule passing_muon_run8329_background_composition_hist_v7:
    input:
        PASSING_MUON_V7_COMPOSITION_HIST


rule passing_muon_run8329_background_composition_plots_v7:
    input:
        PASSING_MUON_V7_COMPOSITION_PLOTS


rule passing_muon_run8329_background_composition_v7:
    input:
        histogram=PASSING_MUON_V7_COMPOSITION_HIST,
        plots=PASSING_MUON_V7_COMPOSITION_PLOTS,


rule passing_muon_run8329_binary_region_study_plots_v7:
    input:
        detector=PASSING_MUON_V7_DETECTOR_PLOTS,
        counts=PASSING_MUON_V7_COUNT_PLOTS,
        stacked_counts=PASSING_MUON_V7_COVERAGE_STACKED_COUNT_PLOTS,
        reco=PASSING_MUON_V7_RECO_PLOTS,


rule passing_muon_run8329_detector_selection_study_v7:
    input:
        manifest=PASSING_MUON_V7_TRACK_MANIFEST,
        detector_histogram=PASSING_MUON_V7_DETECTOR_HIST,
        reco_histogram=PASSING_MUON_V7_RECO_HIST,
        detector_plots=PASSING_MUON_V7_DETECTOR_PLOTS,
        count_plots=PASSING_MUON_V7_COUNT_PLOTS,
        stacked_count_plots=PASSING_MUON_V7_COVERAGE_STACKED_COUNT_PLOTS,
        reco_plots=PASSING_MUON_V7_RECO_PLOTS,


rule passing_muon_run8329_binary_region_study_v7:
    input:
        rules.passing_muon_run8329_detector_selection_study_v7.input


# Data-only loose event displays are deliberately isolated from the main V7
# aggregate so routine detector/track plotting never launches SNDSW pages.
PASSING_MUON_V7_EVENT_DISPLAY_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_v7.yaml"
)
with open(PASSING_MUON_V7_EVENT_DISPLAY_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V7_EVENT_DISPLAY = yaml.safe_load(stream)

_pmv7_event_samples = list(PASSING_MUON_V7_EVENT_DISPLAY["samples"])
if _pmv7_event_samples != ["run8329"]:
    raise ValueError("V7 loose event displays require samples: [run8329]")
_pmv7_event_veto_categories = list(
    PASSING_MUON_V7_EVENT_DISPLAY["selection"]["failure_veto_categories"]
)
_pmv7_event_additional_criteria = list(
    PASSING_MUON_V7_EVENT_DISPLAY["selection"].get(
        "additional_failure_criteria", {}
    )
)
PASSING_MUON_V7_EVENT_BASE_CATEGORIES = [
    "run8329__candidate",
    *(
        f"run8329__failure__{veto}"
        for veto in _pmv7_event_veto_categories
    ),
]
PASSING_MUON_V7_EVENT_ADDITIONAL_CATEGORIES = [
    *(
        f"run8329__failure__{veto}__{criterion}"
        for criterion in _pmv7_event_additional_criteria
        for veto in _pmv7_event_veto_categories
    ),
]
PASSING_MUON_V7_EVENT_CATEGORIES = [
    *PASSING_MUON_V7_EVENT_BASE_CATEGORIES,
    *PASSING_MUON_V7_EVENT_ADDITIONAL_CATEGORIES,
]
_pmv7_event_root = (
    resolve_output(PASSING_MUON_V7_EVENT_DISPLAY['output']['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_EVENT_MANIFEST = (
    f"{_pmv7_event_root}/"
    f"{PASSING_MUON_V7_EVENT_DISPLAY['output']['selected_manifest']}"
)
PASSING_MUON_V7_EVENT_COUNTS = (
    f"{_pmv7_event_root}/"
    f"{PASSING_MUON_V7_EVENT_DISPLAY['output']['category_counts']}"
)
PASSING_MUON_V7_RENDERED_MANIFEST = (
    f"{_pmv7_event_root}/"
    f"{PASSING_MUON_V7_EVENT_DISPLAY['output']['rendered_manifest']}"
)
PASSING_MUON_V7_EVENT_PDFS = [
    f"{_pmv7_event_root}/pdf/{category}.pdf"
    for category in PASSING_MUON_V7_EVENT_BASE_CATEGORIES
]
PASSING_MUON_V7_EVENT_PDFS += [
    f"{_pmv7_event_root}/{criterion}/pdf/"
    f"run8329__failure__{veto}.pdf"
    for criterion in _pmv7_event_additional_criteria
    for veto in _pmv7_event_veto_categories
]
PASSING_MUON_V7_EVENT_SUMMARIES = [
    f"{_pmv7_event_root}/summaries/{category}.csv"
    for category in PASSING_MUON_V7_EVENT_BASE_CATEGORIES
]
PASSING_MUON_V7_EVENT_SUMMARIES += [
    f"{_pmv7_event_root}/{criterion}/summaries/"
    f"run8329__failure__{veto}.csv"
    for criterion in _pmv7_event_additional_criteria
    for veto in _pmv7_event_veto_categories
]


rule select_passing_muon_v7_loose_event_display_events:
    input:
        config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        study_config=PASSING_MUON_V7_CONFIG,
        reference_histogram=PASSING_MUON_V7_DETECTOR_HIST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v7_event_display_events.py"
        ),
        config_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_event_display_config.py"
        ),
        v6_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v6_event_display_events.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        manifest=PASSING_MUON_V7_EVENT_MANIFEST,
        counts=PASSING_MUON_V7_EVENT_COUNTS,
    threads: 1
    resources:
        runtime=8 * 60 * 60, mem_mb=6000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --config {input.config} \
          --output-manifest {output.manifest} \
          --output-counts {output.counts} \
          --repo-root {PERSONAL_WORK_SPACE}
        """


rule plot_passing_muon_v7_loose_event_display:
    input:
        config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_V7_EVENT_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
        common=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=f"{_pmv7_event_root}/pdf/{{category}}.pdf",
        summary=f"{_pmv7_event_root}/summaries/{{category}}.csv",
    wildcard_constraints:
        category="|".join(PASSING_MUON_V7_EVENT_BASE_CATEGORIES)
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {input.manifest} \
          --category {wildcards.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary}
        """


rule plot_passing_muon_v7_loose_event_display_additional:
    input:
        config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_V7_EVENT_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
        common=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=(
            f"{_pmv7_event_root}/{{criterion}}/pdf/"
            "run8329__failure__{veto}.pdf"
        ),
        summary=(
            f"{_pmv7_event_root}/{{criterion}}/summaries/"
            "run8329__failure__{veto}.csv"
        ),
    wildcard_constraints:
        criterion="|".join(_pmv7_event_additional_criteria),
        veto="|".join(_pmv7_event_veto_categories)
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=4000, nvidia_gpu=0
    params:
        category=lambda wildcards: (
            f"run8329__failure__{wildcards.veto}__{wildcards.criterion}"
        ),
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {input.manifest} \
          --category {params.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary}
        """


rule merge_passing_muon_v7_loose_event_display_summaries:
    input:
        summaries=PASSING_MUON_V7_EVENT_SUMMARIES,
        config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "merge_passing_muon_v7_event_display_summaries.py"
        ),
        config_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_event_display_config.py"
        ),
    output:
        rendered=PASSING_MUON_V7_RENDERED_MANIFEST,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    params:
        summaries=" ".join(PASSING_MUON_V7_EVENT_SUMMARIES),
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --output {output.rendered} \
          {params.summaries}
        """


rule passing_muon_run8329_loose_event_displays_data_v7:
    input:
        manifest=PASSING_MUON_V7_EVENT_MANIFEST,
        counts=PASSING_MUON_V7_EVENT_COUNTS,
        pdfs=PASSING_MUON_V7_EVENT_PDFS,
        rendered=PASSING_MUON_V7_RENDERED_MANIFEST,


# Run-8329 event identities rendered into a separate three-page-per-event
# SNDSW + SciFi-channel product.  The source summaries are parameters, rather
# than tracked inputs, so Snakemake cannot follow their old producer rules and
# rebuild or replace the original V7 display set.
PASSING_MUON_V7_SCIFI_EVENT_DISPLAY_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_scifi_v7.yaml"
)
with open(PASSING_MUON_V7_SCIFI_EVENT_DISPLAY_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V7_SCIFI_EVENT_DISPLAY = yaml.safe_load(stream)

_pmv7_scifi_frozen = PASSING_MUON_V7_SCIFI_EVENT_DISPLAY["frozen_input"]
PASSING_MUON_V7_SCIFI_EVENT_CATEGORIES = list(_pmv7_scifi_frozen["categories"])
_pmv7_scifi_base_categories = list(_pmv7_scifi_frozen["base_categories"])
_pmv7_scifi_additional_criteria = list(_pmv7_scifi_frozen["additional_criteria"])
_pmv7_scifi_veto_categories = list(_pmv7_scifi_frozen["veto_categories"])
_pmv7_scifi_additional_categories = [
    f"run8329__failure__{veto}__{criterion}"
    for criterion in _pmv7_scifi_additional_criteria
    for veto in _pmv7_scifi_veto_categories
]
if PASSING_MUON_V7_SCIFI_EVENT_CATEGORIES != [
    *_pmv7_scifi_base_categories,
    *_pmv7_scifi_additional_categories,
]:
    raise ValueError(
        "V7 SciFi categories must be the four base categories followed by "
        "each configured criterion's three veto categories"
    )
_pmv7_scifi_root = (
    resolve_output(PASSING_MUON_V7_SCIFI_EVENT_DISPLAY['output']['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
)
_pmv7_scifi_source_root = (
    resolve_output(_pmv7_scifi_frozen['source_summary_root'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_SCIFI_SELECTION_MANIFEST = (
    resolve_output(_pmv7_scifi_frozen['selection_manifest'], PERSONAL_WORK_SPACE)
)
_pmv7_scifi_base_source_summaries = [
    f"{_pmv7_scifi_source_root}/summaries/{category}.csv"
    for category in _pmv7_scifi_base_categories
]
_pmv7_scifi_additional_source_summaries = [
    f"{_pmv7_scifi_source_root}/{criterion}/summaries/"
    f"run8329__failure__{veto}.csv"
    for criterion in _pmv7_scifi_additional_criteria
    for veto in _pmv7_scifi_veto_categories
]
PASSING_MUON_V7_SCIFI_SOURCE_SUMMARIES = [
    *_pmv7_scifi_base_source_summaries,
    *_pmv7_scifi_additional_source_summaries,
]
_pmv7_scifi_base_pdfs = [
    f"{_pmv7_scifi_root}/pdf/{category}.pdf"
    for category in _pmv7_scifi_base_categories
]
_pmv7_scifi_additional_pdfs = [
    f"{_pmv7_scifi_root}/{criterion}/pdf/run8329__failure__{veto}.pdf"
    for criterion in _pmv7_scifi_additional_criteria
    for veto in _pmv7_scifi_veto_categories
]
PASSING_MUON_V7_SCIFI_EVENT_PDFS = [
    *_pmv7_scifi_base_pdfs,
    *_pmv7_scifi_additional_pdfs,
]
_pmv7_scifi_base_summaries = [
    f"{_pmv7_scifi_root}/summaries/{category}.csv"
    for category in _pmv7_scifi_base_categories
]
_pmv7_scifi_additional_summaries = [
    f"{_pmv7_scifi_root}/{criterion}/summaries/run8329__failure__{veto}.csv"
    for criterion in _pmv7_scifi_additional_criteria
    for veto in _pmv7_scifi_veto_categories
]
PASSING_MUON_V7_SCIFI_EVENT_SUMMARIES = [
    *_pmv7_scifi_base_summaries,
    *_pmv7_scifi_additional_summaries,
]
_pmv7_scifi_base_audits = [
    f"{_pmv7_scifi_root}/audits/{category}.json"
    for category in _pmv7_scifi_base_categories
]
_pmv7_scifi_additional_audits = [
    f"{_pmv7_scifi_root}/{criterion}/audits/run8329__failure__{veto}.json"
    for criterion in _pmv7_scifi_additional_criteria
    for veto in _pmv7_scifi_veto_categories
]
PASSING_MUON_V7_SCIFI_EVENT_AUDITS = [
    *_pmv7_scifi_base_audits,
    *_pmv7_scifi_additional_audits,
]
PASSING_MUON_V7_SCIFI_RENDERED_MANIFEST = (
    f"{_pmv7_scifi_root}/"
    f"{PASSING_MUON_V7_SCIFI_EVENT_DISPLAY['output']['rendered_manifest']}"
)


rule plot_passing_muon_v7_scifi_channel_event_display_base:
    input:
        config=PASSING_MUON_V7_SCIFI_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v7_scifi_channel_event_displays.py"
        ),
        scifi_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "scifi_sipm_event_display.py"
        ),
        sndsw_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
        v6_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
    params:
        manifest=lambda wildcards: (
            f"{_pmv7_scifi_source_root}/summaries/{wildcards.category}.csv"
        ),
        selection_manifest=PASSING_MUON_V7_SCIFI_SELECTION_MANIFEST,
        category=lambda wildcards: wildcards.category,
    output:
        pdf=f"{_pmv7_scifi_root}/pdf/{{category}}.pdf",
        summary=f"{_pmv7_scifi_root}/summaries/{{category}}.csv",
        audit=f"{_pmv7_scifi_root}/audits/{{category}}.json",
    wildcard_constraints:
        category="|".join(_pmv7_scifi_base_categories)
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=6000, disk_mb=8000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {params.manifest} \
          --selection-manifest {params.selection_manifest} \
          --category {params.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary} \
          --output-audit {output.audit}
        """


rule plot_passing_muon_v7_scifi_channel_event_display_additional:
    input:
        config=PASSING_MUON_V7_SCIFI_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v7_scifi_channel_event_displays.py"
        ),
        scifi_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "scifi_sipm_event_display.py"
        ),
        sndsw_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
        v6_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
    params:
        manifest=lambda wildcards: (
            f"{_pmv7_scifi_source_root}/{wildcards.criterion}/summaries/"
            f"run8329__failure__{wildcards.veto}.csv"
        ),
        selection_manifest=PASSING_MUON_V7_SCIFI_SELECTION_MANIFEST,
        category=lambda wildcards: (
            f"run8329__failure__{wildcards.veto}__{wildcards.criterion}"
        ),
    output:
        pdf=(
            f"{_pmv7_scifi_root}/{{criterion}}/pdf/"
            "run8329__failure__{veto}.pdf"
        ),
        summary=(
            f"{_pmv7_scifi_root}/{{criterion}}/summaries/"
            "run8329__failure__{veto}.csv"
        ),
        audit=(
            f"{_pmv7_scifi_root}/{{criterion}}/audits/"
            "run8329__failure__{veto}.json"
        ),
    wildcard_constraints:
        criterion="|".join(_pmv7_scifi_additional_criteria),
        veto="|".join(_pmv7_scifi_veto_categories),
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=6000, disk_mb=8000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {params.manifest} \
          --selection-manifest {params.selection_manifest} \
          --category {params.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary} \
          --output-audit {output.audit}
        """


rule merge_passing_muon_v7_scifi_channel_event_display_summaries:
    input:
        config=PASSING_MUON_V7_SCIFI_EVENT_DISPLAY_CONFIG,
        summaries=PASSING_MUON_V7_SCIFI_EVENT_SUMMARIES,
        audits=PASSING_MUON_V7_SCIFI_EVENT_AUDITS,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "merge_passing_muon_v7_scifi_event_display_summaries.py"
        ),
        renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v7_scifi_channel_event_displays.py"
        ),
    output:
        rendered=PASSING_MUON_V7_SCIFI_RENDERED_MANIFEST,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    params:
        manifests=" ".join(PASSING_MUON_V7_SCIFI_SOURCE_SUMMARIES),
        summaries=" ".join(PASSING_MUON_V7_SCIFI_EVENT_SUMMARIES),
        audits=" ".join(PASSING_MUON_V7_SCIFI_EVENT_AUDITS),
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --manifests {params.manifests} \
          --output {output.rendered} \
          --summaries {params.summaries} \
          --audits {params.audits}
        """


# Additive zero-Veto samples use a fresh reserve manifest.  They deliberately
# do not depend on the loose pass/fail summaries, so the four density samples
# contain every event satisfying count_veto == 0 regardless of working point.
PASSING_MUON_V7_VETO0_SCIFI_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_scifi_veto0_v7.yaml"
)
with open(PASSING_MUON_V7_VETO0_SCIFI_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V7_VETO0_SCIFI = yaml.safe_load(stream)

_pmv7_veto0_selection = PASSING_MUON_V7_VETO0_SCIFI["selection"]
_pmv7_veto0_criteria = list(_pmv7_veto0_selection["inclusive_criteria"])
_pmv7_veto0_veto_categories = list(
    _pmv7_veto0_selection["inclusive_veto_categories"]
)
if _pmv7_veto0_veto_categories != ["veto_eq0"]:
    raise ValueError("V7 additive SciFi displays require veto_eq0 only")
PASSING_MUON_V7_VETO0_SCIFI_CATEGORIES = [
    f"run8329__inclusive__veto_eq0__{criterion}"
    for criterion in _pmv7_veto0_criteria
]
if PASSING_MUON_V7_VETO0_SCIFI_CATEGORIES != list(
    PASSING_MUON_V7_VETO0_SCIFI["frozen_input"]["categories"]
):
    raise ValueError("V7 zero-Veto SciFi configured categories are inconsistent")
_pmv7_veto0_scifi_root = (
    resolve_output(PASSING_MUON_V7_VETO0_SCIFI['output']['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_V7_VETO0_SCIFI_MANIFEST = (
    f"{_pmv7_veto0_scifi_root}/"
    f"{PASSING_MUON_V7_VETO0_SCIFI['output']['selected_manifest']}"
)
PASSING_MUON_V7_VETO0_SCIFI_COUNTS = (
    f"{_pmv7_veto0_scifi_root}/"
    f"{PASSING_MUON_V7_VETO0_SCIFI['output']['category_counts']}"
)
PASSING_MUON_V7_VETO0_SCIFI_PDFS = [
    f"{_pmv7_veto0_scifi_root}/{criterion}/pdf/run8329__veto_eq0.pdf"
    for criterion in _pmv7_veto0_criteria
]
PASSING_MUON_V7_VETO0_SCIFI_SUMMARIES = [
    f"{_pmv7_veto0_scifi_root}/{criterion}/summaries/run8329__veto_eq0.csv"
    for criterion in _pmv7_veto0_criteria
]
PASSING_MUON_V7_VETO0_SCIFI_AUDITS = [
    f"{_pmv7_veto0_scifi_root}/{criterion}/audits/run8329__veto_eq0.json"
    for criterion in _pmv7_veto0_criteria
]
PASSING_MUON_V7_VETO0_SCIFI_RENDERED = (
    f"{_pmv7_veto0_scifi_root}/"
    f"{PASSING_MUON_V7_VETO0_SCIFI['output']['rendered_manifest']}"
)


rule select_passing_muon_v7_veto0_scifi_event_display_events:
    input:
        config=PASSING_MUON_V7_VETO0_SCIFI_CONFIG,
        study_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v7_event_display_events.py"
        ),
        config_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_event_display_config.py"
        ),
        v6_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v6_event_display_events.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        manifest=PASSING_MUON_V7_VETO0_SCIFI_MANIFEST,
        counts=PASSING_MUON_V7_VETO0_SCIFI_COUNTS,
    threads: 1
    resources:
        runtime=8 * 60 * 60, mem_mb=6000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --config {input.config} \
          --output-manifest {output.manifest} \
          --output-counts {output.counts} \
          --repo-root {PERSONAL_WORK_SPACE}
        """


rule plot_passing_muon_v7_veto0_scifi_channel_event_display:
    input:
        config=PASSING_MUON_V7_VETO0_SCIFI_CONFIG,
        selection_manifest=PASSING_MUON_V7_VETO0_SCIFI_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v7_scifi_channel_event_displays.py"
        ),
        scifi_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "scifi_sipm_event_display.py"
        ),
        sndsw_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
        v6_renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
    output:
        pdf=(
            f"{_pmv7_veto0_scifi_root}/{{criterion}}/pdf/"
            "run8329__veto_eq0.pdf"
        ),
        summary=(
            f"{_pmv7_veto0_scifi_root}/{{criterion}}/summaries/"
            "run8329__veto_eq0.csv"
        ),
        audit=(
            f"{_pmv7_veto0_scifi_root}/{{criterion}}/audits/"
            "run8329__veto_eq0.json"
        ),
    wildcard_constraints:
        criterion="|".join(_pmv7_veto0_criteria),
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=6000, disk_mb=8000, nvidia_gpu=0
    params:
        category=lambda wildcards: (
            f"run8329__inclusive__veto_eq0__{wildcards.criterion}"
        ),
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --selection-manifest {input.selection_manifest} \
          --direct-selection-manifest \
          --category {params.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary} \
          --output-audit {output.audit}
        """


rule merge_passing_muon_v7_veto0_scifi_event_display_summaries:
    input:
        config=PASSING_MUON_V7_VETO0_SCIFI_CONFIG,
        manifest=PASSING_MUON_V7_VETO0_SCIFI_MANIFEST,
        summaries=PASSING_MUON_V7_VETO0_SCIFI_SUMMARIES,
        audits=PASSING_MUON_V7_VETO0_SCIFI_AUDITS,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "merge_passing_muon_v7_scifi_event_display_summaries.py"
        ),
        renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v7_scifi_channel_event_displays.py"
        ),
    output:
        rendered=PASSING_MUON_V7_VETO0_SCIFI_RENDERED,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    params:
        summaries=" ".join(PASSING_MUON_V7_VETO0_SCIFI_SUMMARIES),
        audits=" ".join(PASSING_MUON_V7_VETO0_SCIFI_AUDITS),
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --manifests {input.manifest} \
          --output {output.rendered} \
          --summaries {params.summaries} \
          --audits {params.audits}
        """


rule passing_muon_run8329_veto0_scifi_event_displays_v7:
    input:
        manifest=PASSING_MUON_V7_VETO0_SCIFI_MANIFEST,
        counts=PASSING_MUON_V7_VETO0_SCIFI_COUNTS,
        pdfs=PASSING_MUON_V7_VETO0_SCIFI_PDFS,
        audits=PASSING_MUON_V7_VETO0_SCIFI_AUDITS,
        rendered=PASSING_MUON_V7_VETO0_SCIFI_RENDERED,


rule passing_muon_run8329_loose_event_displays_scifi_v7:
    input:
        pdfs=PASSING_MUON_V7_SCIFI_EVENT_PDFS,
        audits=PASSING_MUON_V7_SCIFI_EVENT_AUDITS,
        rendered=PASSING_MUON_V7_SCIFI_RENDERED_MANIFEST,


# Pooled upward-muon MC event displays use the same category definitions as
# data but remain isolated so running this target cannot invalidate data PDFs.
PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_mc_v7.yaml"
)
with open(PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_MC_V7_EVENT_DISPLAY = yaml.safe_load(stream)

_pmmcv7_samples = list(PASSING_MUON_MC_V7_EVENT_DISPLAY["samples"])
if _pmmcv7_samples != ["mc_muon_up"]:
    raise ValueError("V7 MC event displays require samples: [mc_muon_up]")
_pmmcv7_veto_categories = list(
    PASSING_MUON_MC_V7_EVENT_DISPLAY["selection"]["failure_veto_categories"]
)
_pmmcv7_additional_criteria = list(
    PASSING_MUON_MC_V7_EVENT_DISPLAY["selection"].get(
        "additional_failure_criteria", {}
    )
)
PASSING_MUON_MC_V7_EVENT_BASE_CATEGORIES = [
    "mc_muon_up__candidate",
    *(
        f"mc_muon_up__failure__{veto}"
        for veto in _pmmcv7_veto_categories
    ),
]
PASSING_MUON_MC_V7_EVENT_ADDITIONAL_CATEGORIES = [
    *(
        f"mc_muon_up__failure__{veto}__{criterion}"
        for criterion in _pmmcv7_additional_criteria
        for veto in _pmmcv7_veto_categories
    ),
]
PASSING_MUON_MC_V7_EVENT_CATEGORIES = [
    *PASSING_MUON_MC_V7_EVENT_BASE_CATEGORIES,
    *PASSING_MUON_MC_V7_EVENT_ADDITIONAL_CATEGORIES,
]
_pmmcv7_event_root = (
    resolve_output(PASSING_MUON_MC_V7_EVENT_DISPLAY['output']['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_MC_V7_EVENT_MANIFEST = (
    f"{_pmmcv7_event_root}/"
    f"{PASSING_MUON_MC_V7_EVENT_DISPLAY['output']['selected_manifest']}"
)
PASSING_MUON_MC_V7_EVENT_COUNTS = (
    f"{_pmmcv7_event_root}/"
    f"{PASSING_MUON_MC_V7_EVENT_DISPLAY['output']['category_counts']}"
)
PASSING_MUON_MC_V7_RENDERED_MANIFEST = (
    f"{_pmmcv7_event_root}/"
    f"{PASSING_MUON_MC_V7_EVENT_DISPLAY['output']['rendered_manifest']}"
)
PASSING_MUON_MC_V7_EVENT_PDFS = [
    f"{_pmmcv7_event_root}/pdf/{category}.pdf"
    for category in PASSING_MUON_MC_V7_EVENT_BASE_CATEGORIES
]
PASSING_MUON_MC_V7_EVENT_PDFS += [
    f"{_pmmcv7_event_root}/{criterion}/pdf/"
    f"mc_muon_up__failure__{veto}.pdf"
    for criterion in _pmmcv7_additional_criteria
    for veto in _pmmcv7_veto_categories
]
PASSING_MUON_MC_V7_EVENT_SUMMARIES = [
    f"{_pmmcv7_event_root}/summaries/{category}.csv"
    for category in PASSING_MUON_MC_V7_EVENT_BASE_CATEGORIES
]
PASSING_MUON_MC_V7_EVENT_SUMMARIES += [
    f"{_pmmcv7_event_root}/{criterion}/summaries/"
    f"mc_muon_up__failure__{veto}.csv"
    for criterion in _pmmcv7_additional_criteria
    for veto in _pmmcv7_veto_categories
]


rule select_passing_muon_mc_v7_event_display_events:
    input:
        config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        study_config=PASSING_MUON_V7_CONFIG,
        reference_histogram=PASSING_MUON_V7_DETECTOR_HIST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v7_event_display_events.py"
        ),
        config_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_event_display_config.py"
        ),
        v6_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v6_event_display_events.py"
        ),
        engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        manifest=PASSING_MUON_MC_V7_EVENT_MANIFEST,
        counts=PASSING_MUON_MC_V7_EVENT_COUNTS,
    threads: 1
    resources:
        runtime=8 * 60 * 60, mem_mb=6000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --config {input.config} \
          --output-manifest {output.manifest} \
          --output-counts {output.counts} \
          --repo-root {PERSONAL_WORK_SPACE}
        """


rule plot_passing_muon_mc_v7_event_display:
    input:
        config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_MC_V7_EVENT_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
        common=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=f"{_pmmcv7_event_root}/pdf/{{category}}.pdf",
        summary=f"{_pmmcv7_event_root}/summaries/{{category}}.csv",
    wildcard_constraints:
        category="|".join(PASSING_MUON_MC_V7_EVENT_BASE_CATEGORIES)
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=8000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {input.manifest} \
          --category {wildcards.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary}
        """


rule plot_passing_muon_mc_v7_event_display_additional:
    input:
        config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_MC_V7_EVENT_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
        common=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=(
            f"{_pmmcv7_event_root}/{{criterion}}/pdf/"
            "mc_muon_up__failure__{veto}.pdf"
        ),
        summary=(
            f"{_pmmcv7_event_root}/{{criterion}}/summaries/"
            "mc_muon_up__failure__{veto}.csv"
        ),
    wildcard_constraints:
        criterion="|".join(_pmmcv7_additional_criteria),
        veto="|".join(_pmmcv7_veto_categories)
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=8000, nvidia_gpu=0
    params:
        category=lambda wildcards: (
            f"mc_muon_up__failure__{wildcards.veto}__{wildcards.criterion}"
        ),
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --manifest {input.manifest} \
          --category {params.category} \
          --output-pdf {output.pdf} \
          --output-summary {output.summary}
        """


rule merge_passing_muon_mc_v7_event_display_summaries:
    input:
        summaries=PASSING_MUON_MC_V7_EVENT_SUMMARIES,
        config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "merge_passing_muon_v7_event_display_summaries.py"
        ),
        config_helpers=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_event_display_config.py"
        ),
    output:
        rendered=PASSING_MUON_MC_V7_RENDERED_MANIFEST,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    params:
        summaries=" ".join(PASSING_MUON_MC_V7_EVENT_SUMMARIES),
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --output {output.rendered} \
          {params.summaries}
        """


rule passing_muon_up_mc_loose_event_displays_v7:
    input:
        manifest=PASSING_MUON_MC_V7_EVENT_MANIFEST,
        counts=PASSING_MUON_MC_V7_EVENT_COUNTS,
        pdfs=PASSING_MUON_MC_V7_EVENT_PDFS,
        rendered=PASSING_MUON_MC_V7_RENDERED_MANIFEST,


# Full-population distributions corresponding to the cumulative SciFi-density
# event-display subsets. These are independent of the sampled/rendered pages.
_pmv7_density_data_selection = PASSING_MUON_V7_EVENT_DISPLAY["selection"]
_pmv7_density_mc_selection = PASSING_MUON_MC_V7_EVENT_DISPLAY["selection"]
if _pmv7_density_data_selection != _pmv7_density_mc_selection:
    raise ValueError("V7 data and MC event-display selections must be identical")
PASSING_MUON_V7_SCIFI_DENSITY_CATEGORIES = [
    str(category)
    for category in _pmv7_density_data_selection.get(
        "additional_failure_criteria", {}
    )
    if str(category).startswith("scifi_density_")
]
if not PASSING_MUON_V7_SCIFI_DENSITY_CATEGORIES:
    raise ValueError("V7 event displays define no scifi_density_* categories")
PASSING_MUON_V7_SCIFI_DENSITY_HIST = (
    f"{_pmv7_root}/histograms/"
    "loose_event_display_scifi_density_categories.root"
)
_pmv7_density_plot_root = (
    f"{_pmv7_plot_root}/event_display_category_distributions"
)
PASSING_MUON_V7_SCIFI_DENSITY_PLOTS = [
    f"{_pmv7_density_plot_root}/{category}.pdf"
    for category in PASSING_MUON_V7_SCIFI_DENSITY_CATEGORIES
]


rule make_passing_muon_scifi_density_hist_v7:
    input:
        mc_muon_up_metadata=PASSING_MUON_V7_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V7_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        event_display_config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        mc_event_display_config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_scifi_density_hist_v7.py"
        ),
        category_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_scifi_density.py"
        ),
        detector_engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V7_SCIFI_DENSITY_HIST,
    params:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/scifi_density_hist_v7.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {params.partition_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --event-display-config {input.event_display_config} \
          --mc-event-display-config {input.mc_event_display_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule plot_passing_muon_scifi_density_study_v7:
    input:
        root=PASSING_MUON_V7_SCIFI_DENSITY_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V7_CONFIG,
        event_display_config=PASSING_MUON_V7_EVENT_DISPLAY_CONFIG,
        mc_event_display_config=PASSING_MUON_MC_V7_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_scifi_density_study_v7.py"
        ),
        category_helper=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "passing_muon_v7_scifi_density.py"
        ),
        detector_engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=f"{_pmv7_density_plot_root}/{{category}}.pdf",
    wildcard_constraints:
        category="|".join(PASSING_MUON_V7_SCIFI_DENSITY_CATEGORIES)
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=3000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --event-display-config {input.event_display_config} \
          --mc-event-display-config {input.mc_event_display_config} \
          --category {wildcards.category} \
          --output-pdf {output.pdf}
        """


rule passing_muon_run8329_loose_event_display_scifi_density_distributions_v7:
    input:
        histogram=PASSING_MUON_V7_SCIFI_DENSITY_HIST,
        plots=PASSING_MUON_V7_SCIFI_DENSITY_PLOTS,
