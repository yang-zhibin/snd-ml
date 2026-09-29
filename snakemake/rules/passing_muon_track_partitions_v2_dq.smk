# Run-8329 v2: existing real-data DQ features versus DQ-intersected PassingMuonReco.

PASSING_MUON_V2_COMPARISON_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/passing_muon_run8329_comparison_v2.yaml"
PASSING_MUON_V2_PARTITION_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/partitions/passing_muon_track_partition_v2.yaml"
PASSING_MUON_V2_HIST_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/passing_muon_run8329_hist_v2.yaml"
PASSING_MUON_V2_EVAL_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/passing_muon_run8329_eval_v2.yaml"
with open(PASSING_MUON_V2_COMPARISON_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V2_COMPARISON = yaml.safe_load(stream)
with open(PASSING_MUON_V2_PARTITION_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V2_PARTITION = yaml.safe_load(stream)
with open(PASSING_MUON_V2_HIST_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V2_HISTS = yaml.safe_load(stream)
with open(PASSING_MUON_V2_EVAL_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V2_EVAL = yaml.safe_load(stream)

_pmv2_metadata_dir = f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['output']['metadata_dir']}"
PASSING_MUON_V2_INITIAL_REAL_METADATA = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['inclusive_metadata']}"
PASSING_MUON_V2_INITIAL_PASSING_METADATA = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['passing_metadata']}"
PASSING_MUON_V2_INITIAL_PAIRS = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['source_pairs']}"
PASSING_MUON_V2_SOURCE_VALIDATION = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['validation']}"
PASSING_MUON_V2_REAL_METADATA = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['final_real_metadata']}"
PASSING_MUON_V2_PASSING_METADATA = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['final_passing_metadata']}"
PASSING_MUON_V2_SOURCE_PAIRS = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['final_source_pairs']}"
PASSING_MUON_V2_METADATA_VALIDATION = f"{_pmv2_metadata_dir}/{PASSING_MUON_V2_COMPARISON['output']['final_validation']}"
PASSING_MUON_V2_PARTITION_METADATA = f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_PARTITION['output']['metadata_csv']}"
PASSING_MUON_V2_PARTITION_VALIDATION = f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_PARTITION['output']['validation_csv']}"
PASSING_MUON_V2_GROUPS = list(PASSING_MUON_V2_PARTITION["groups"])
PASSING_MUON_V2_PARTS = [f"{value:03d}" for value in range(1, int(PASSING_MUON_V2_PARTITION["n_partitions"]) + 1)]
PASSING_MUON_V2_FEATURES = list(PASSING_MUON_V2_HISTS["features"])
PASSING_MUON_V2_SELECTIONS = list(PASSING_MUON_V2_EVAL["selections"])

_pmv2_real = pd.read_csv(f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['inputs']['inclusive_metadata']}")
_pmv2_real = _pmv2_real.loc[pd.to_numeric(_pmv2_real.run) == int(PASSING_MUON_V2_COMPARISON["run"])].copy()
_pmv2_tracks = pd.read_csv(f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['inputs']['passing_metadata']}")
_pmv2_tracks = _pmv2_tracks.loc[pd.to_numeric(_pmv2_tracks.run) == int(PASSING_MUON_V2_COMPARISON["run"])].copy()
_pmv2_map = _pmv2_tracks[["source_file_index", "source_path"]].drop_duplicates().rename(columns={"source_path": "digi_path"})
_pmv2_real = _pmv2_map.merge(_pmv2_real, on="digi_path", validate="one_to_one").sort_values("source_file_index")
_pmv2_real_by_source = {f"{int(row.source_file_index):03d}": row for _, row in _pmv2_real.iterrows()}
_pmv2_passing_by_source = {}
for _, row in _pmv2_tracks.iterrows():
    source = f"{int(row.source_file_index):03d}"
    path = str(row.output_base_path).rstrip("/") + "/" + str(row.feature_path).lstrip("/")
    _pmv2_passing_by_source.setdefault(source, []).append(path)


def _pmv2_real_feature(source):
    root = PASSING_MUON_V2_COMPARISON["output"]["real_dq_feature_root"].rstrip("/")
    return f"{root}/realDataDQ_run008329_source{source}.root"


PASSING_MUON_V2_SOURCES = sorted(_pmv2_real_by_source)
PASSING_MUON_V2_MISSING_DQ_SOURCES = [
    f"{int(source):04d}"
    for source in PASSING_MUON_V2_COMPARISON["inputs"].get(
        "missing_dq_cutflow_source_indices", []
    )
]
_pmv2_filtered_root = PASSING_MUON_V2_COMPARISON["output"]["filtered_passing_root"].rstrip("/")
_pmv2_audit_dir = f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['output']['audit_dir']}"
PASSING_MUON_V2_FILTERED = [f"{_pmv2_filtered_root}/passingMuonDQ_run008329_source{s}.root" for s in PASSING_MUON_V2_SOURCES]
PASSING_MUON_V2_AUDITS = [f"{_pmv2_audit_dir}/source{s}.csv" for s in PASSING_MUON_V2_SOURCES]
PASSING_MUON_V2_REAL_FEATURES = [_pmv2_real_feature(s) for s in PASSING_MUON_V2_SOURCES]


def _pmv2_dq_cutflow(source):
    row = _pmv2_real_by_source[source]
    return str(row.output_base_path).rstrip("/") + "/" + str(row.nueAnalysisFilter_path).lstrip("/")


PASSING_MUON_V2_MISSING_DQ_CUTFLOWS = [
    _pmv2_dq_cutflow(f"{int(source):03d}")
    for source in PASSING_MUON_V2_MISSING_DQ_SOURCES
]
PASSING_MUON_V2_MISSING_DQ_MARKERS = [
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
    f"dq_cutflow_rerun/source{source}.done"
    for source in PASSING_MUON_V2_MISSING_DQ_SOURCES
]


rule process_run8329_nue_analysis_filter_v2:
    input:
        digi=lambda wc: str(_pmv2_real_by_source[f"{int(wc.source):03d}"].digi_path),
        geo=lambda wc: str(_pmv2_real_by_source[f"{int(wc.source):03d}"].geo_path),
    output:
        marker=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/dq_cutflow_rerun/source{{source}}.done",
    params:
        dq=lambda wc: _pmv2_dq_cutflow(f"{int(wc.source):03d}"),
    wildcard_constraints:
        source="|".join(PASSING_MUON_V2_MISSING_DQ_SOURCES)
    threads: 1
    resources:
        runtime=20 * 60, mem_mb=2000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        # Snakemake itself runs in a Miniconda Python environment.  Remove that
        # interpreter from PATH before loading sndsw, whose Python and standard
        # library must be used together.
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/run8329_nue_filter.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        sndEventFilter --input {input.digi} --geofile {input.geo} \
          --output "$tmp_dir/output.root" --pipeline \
          "$SNDSW_ROOT/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h"
        xrdcp -f "$tmp_dir/output.root" {params.dq}
        mkdir -p "$(dirname {output.marker})"
        printf '%s\n' "{params.dq}" > {output.marker}
        """


rule passing_muon_run8329_missing_dq_cutflows_v2:
    input:
        PASSING_MUON_V2_MISSING_DQ_MARKERS


def _pmv2_dq_rerun_dependency(wc):
    source4 = f"{int(wc.source):04d}"
    if source4 not in PASSING_MUON_V2_MISSING_DQ_SOURCES:
        return []
    return [
        f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
        f"dq_cutflow_rerun/source{source4}.done"
    ]


rule validate_run8329_dq_cutflow_v2:
    input:
        rerun=_pmv2_dq_rerun_dependency,
    output:
        marker=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
        "dq_cutflow_valid/source{source}.done",
    params:
        dq=lambda wc: _pmv2_dq_cutflow(wc.source),
    wildcard_constraints:
        source=r"\d{3}"
    threads: 1
    resources:
        runtime=5 * 60, mem_mb=500, disk_mb=100, nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        xrdfs root://eospublic.cern.ch stat {params.dq} >/dev/null
        mkdir -p "$(dirname {output.marker})"
        printf '%s\n' "{params.dq}" > {output.marker}
        """


rule process_run8329_real_dq_features_v2:
    input:
        digi=lambda wc: str(_pmv2_real_by_source[wc.source].digi_path),
        geo=lambda wc: str(_pmv2_real_by_source[wc.source].geo_path),
        dq_ready=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
        "dq_cutflow_valid/source{source}.done",
        script=f"{PERSONAL_WORK_SPACE}/convertData/digi_2_features.py",
        selection_module=f"{PERSONAL_WORK_SPACE}/convertData/sample_selection.py",
    output:
        root=PASSING_MUON_V2_COMPARISON["output"]["real_dq_feature_root"].rstrip("/")
        + "/realDataDQ_run008329_source{source}.root",
    params:
        dq=lambda wc: _pmv2_dq_cutflow(wc.source),
    wildcard_constraints:
        source=r"\d{3}"
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=2000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/run8329_real_dq_features.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        export PYTHONPATH="$SNDSW_ROOT:${{PYTHONPATH:-}}"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.script} \
          -p {params.dq} -d {input.digi} -g {input.geo} -o "$tmp_dir/output.root" \
          -t real_data --tree-name cbmsim --selection-mode external_cutflow
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule passing_muon_run8329_real_dq_features_v2:
    input:
        PASSING_MUON_V2_REAL_FEATURES


rule validate_run8329_passing_features_v2:
    input:
        metadata=f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['inputs']['passing_metadata']}",
    output:
        marker=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
        "passing_features_valid/source{source}.done",
    params:
        passing=lambda wc: _pmv2_passing_by_source[wc.source],
    wildcard_constraints:
        source=r"\d{3}"
    threads: 1
    resources:
        runtime=5 * 60, mem_mb=500, disk_mb=100, nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        for path in {params.passing:q}; do
          xrdfs root://eospublic.cern.ch stat "$path" >/dev/null
        done
        mkdir -p "$(dirname {output.marker})"
        printf '%s\n' "validated {wildcards.source}" > {output.marker}
        """


def _pmv2_partition_root(group, part):
    return f"{PASSING_MUON_V2_PARTITION['output']['root_dir'].rstrip('/')}/all_events/{group}/feature_all_events__{group}__part{part}.root"


PASSING_MUON_V2_PARTITION_ROOTS = [_pmv2_partition_root(g, p) for g in PASSING_MUON_V2_GROUPS for p in PASSING_MUON_V2_PARTS]
PASSING_MUON_V2_ROW_DIR = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/passing_muon_track_partition_v2_passing_muon_track_comparison_region_v2/rows"
PASSING_MUON_V2_ROWS = [f"{PASSING_MUON_V2_ROW_DIR}/all_events__{g}__part{p}.csv" for g in PASSING_MUON_V2_GROUPS for p in PASSING_MUON_V2_PARTS]
_pmv2_eval_root = PASSING_MUON_V2_EVAL["output"]["root_dir"].rstrip("/")
_pmv2_plot_root = f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_EVAL['output']['plot_dir']}"
PASSING_MUON_V2_EVAL_ROOTS = [
    f"{_pmv2_eval_root}/histograms/{selection}.root"
    for selection in PASSING_MUON_V2_SELECTIONS
]
PASSING_MUON_V2_EVAL_PLOTS = [
    f"{_pmv2_plot_root}/comparison/{selection}.pdf"
    for selection in PASSING_MUON_V2_SELECTIONS
]
PASSING_MUON_V2_TRACK_DIR = f"{_pmv2_plot_root}/track_inspection"
PASSING_MUON_V2_TRACK_HISTS = f"{PASSING_MUON_V2_TRACK_DIR}/passing_muon_tracks_v2_histograms.root"
PASSING_MUON_V2_TRACK_DONE = f"{PASSING_MUON_V2_TRACK_DIR}/passing_muon_tracks_v2.done"
PASSING_MUON_V2_TRACK_PLOTS = [f"{PASSING_MUON_V2_TRACK_DIR}/plots/{p}_{s}.pdf" for p in PASSING_MUON_TRACK_PLOT_PREFIXES for s in PASSING_MUON_TRACK_PLOT_SUFFIXES] + [f"{PASSING_MUON_V2_TRACK_DIR}/plots/reco_{m}_{s}.pdf" for m in ("simple", "hough") for s in ("ds_n_all_vs_scifi_n_all", "ds_best_chi2ndf_vs_scifi_best_chi2ndf")]


rule passing_muon_run8329_comparison_metadata_v2:
    input:
        config=PASSING_MUON_V2_COMPARISON_CONFIG,
        real=f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['inputs']['inclusive_metadata']}",
        passing=f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V2_COMPARISON['inputs']['passing_metadata']}",
        script=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/make_passing_muon_run8329_comparison_metadata.py",
    output:
        real=PASSING_MUON_V2_INITIAL_REAL_METADATA,
        passing=PASSING_MUON_V2_INITIAL_PASSING_METADATA,
        pairs=PASSING_MUON_V2_INITIAL_PAIRS,
        validation=PASSING_MUON_V2_SOURCE_VALIDATION,
    resources:
        runtime=30 * 60, mem_mb=2000, disk_mb=1000, nvidia_gpu=0
    shell:
        "python3 {input.script} --config {input.config} --repo-root {PERSONAL_WORK_SPACE} --output-dir {_pmv2_metadata_dir}"


rule filter_passing_muon_run8329_by_real_dq_v2:
    input:
        validation=PASSING_MUON_V2_SOURCE_VALIDATION,
        script=f"{PERSONAL_WORK_SPACE}/convertData/filter_passing_muon_by_real_data_dq.py",
        real=lambda wc: _pmv2_real_feature(wc.source),
        passing_ready=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_comparison_v2/"
        "passing_features_valid/source{source}.done",
    output:
        root=_pmv2_filtered_root + "/passingMuonDQ_run008329_source{source}.root",
        audit=_pmv2_audit_dir + "/source{source}.csv",
    params:
        passing=lambda wc: _pmv2_passing_by_source[wc.source],
        source_index=lambda wc: int(wc.source),
        expected=lambda wc: int(pd.to_numeric(_pmv2_tracks.loc[pd.to_numeric(_pmv2_tracks.source_file_index) == int(wc.source), "n_saved_event"]).sum()),
    wildcard_constraints:
        source=r"\d{3}"
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=5000, disk_mb=5000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/run8329_dq_intersection.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} --real-feature {input.real} --passing-features {params.passing} \
          --source-file-index {params.source_index} --run 8329 --expected-passing-events {params.expected} \
          --output-root "$tmp_dir/output.root" --published-root {output.root} --output-audit "$tmp_dir/audit.csv"
        mkdir -p "$(dirname {output.root})" "$(dirname {output.audit})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        cp "$tmp_dir/audit.csv" {output.audit}
        """


rule passing_muon_run8329_dq_intersection_v2:
    input:
        roots=PASSING_MUON_V2_FILTERED,
        audits=PASSING_MUON_V2_AUDITS,


rule finalize_passing_muon_run8329_dq_metadata_v2:
    input:
        real=PASSING_MUON_V2_INITIAL_REAL_METADATA,
        pairs=PASSING_MUON_V2_INITIAL_PAIRS,
        audits=PASSING_MUON_V2_AUDITS,
        config=PASSING_MUON_V2_COMPARISON_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/snakemake/metadata/finalize_passing_muon_run8329_dq_metadata.py",
    output:
        real=PASSING_MUON_V2_REAL_METADATA,
        passing=PASSING_MUON_V2_PASSING_METADATA,
        pairs=PASSING_MUON_V2_SOURCE_PAIRS,
        validation=PASSING_MUON_V2_METADATA_VALIDATION,
    resources:
        runtime=30 * 60, mem_mb=2000, disk_mb=1000, nvidia_gpu=0
    shell:
        "python3 {input.script} --real-metadata {input.real} --source-pairs {input.pairs} --audits {input.audits} --config {input.config} --output-real {output.real} --output-passing {output.passing} --output-pairs {output.pairs} --output-validation {output.validation}"


def _pmv2_group_metadata(wc):
    return PASSING_MUON_V2_REAL_METADATA if wc.group == "real_data_dq_run8329" else PASSING_MUON_V2_PASSING_METADATA


def _pmv2_group_features(wc):
    return PASSING_MUON_V2_REAL_FEATURES if wc.group == "real_data_dq_run8329" else PASSING_MUON_V2_FILTERED


rule build_passing_muon_track_partition_v2:
    input:
        metadata=_pmv2_group_metadata,
        features=_pmv2_group_features,
        config=PASSING_MUON_V2_PARTITION_CONFIG,
        validation=PASSING_MUON_V2_METADATA_VALIDATION,
        builder=f"{PERSONAL_WORK_SPACE}/convertData/build_paired_run8329_partition.py",
    output:
        root=PASSING_MUON_V2_PARTITION["output"]["root_dir"].rstrip("/") + "/all_events/{group}/feature_all_events__{group}__part{part}.root",
        row=PASSING_MUON_V2_ROW_DIR + "/all_events__{group}__part{part}.csv",
    wildcard_constraints:
        group="|".join(PASSING_MUON_V2_GROUPS), part=r"\d{3}"
    threads: 4
    resources:
        runtime=10 * 60 * 60, mem_mb=8000, disk_mb=20000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/run8329_partition_v2.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.builder} --metadata {input.metadata} --config {input.config} --group {wildcards.group} \
          --part-index {wildcards.part} --output-root "$tmp_dir/output.root" --published-root {output.root} \
          --output-row "$tmp_dir/output.csv" --threads {threads}
        mkdir -p "$(dirname {output.root})" "$(dirname {output.row})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        cp "$tmp_dir/output.csv" {output.row}
        """


rule validate_passing_muon_track_partitions_v2:
    input:
        rows=PASSING_MUON_V2_ROWS,
        roots=PASSING_MUON_V2_PARTITION_ROOTS,
        config=PASSING_MUON_V2_PARTITION_CONFIG,
        dq_validation=PASSING_MUON_V2_METADATA_VALIDATION,
        validator=f"{PERSONAL_WORK_SPACE}/convertData/validate_passing_muon_track_partitions_v2.py",
    output:
        metadata=PASSING_MUON_V2_PARTITION_METADATA,
        validation=PASSING_MUON_V2_PARTITION_VALIDATION,
    resources:
        runtime=12 * 60 * 60, mem_mb=6000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        set -euo pipefail
        python "{input.validator}" \
          --rows {input.rows} \
          --config "{input.config}" \
          --dq-validation "{input.dq_validation}" \
          --output-metadata "{output.metadata}" \
          --output-validation "{output.validation}" \
          --event-containment exact
        """


rule passing_muon_track_partitions_v2:
    input:
        roots=PASSING_MUON_V2_PARTITION_ROOTS,
        metadata=PASSING_MUON_V2_PARTITION_METADATA,
        validation=PASSING_MUON_V2_PARTITION_VALIDATION,


rule passing_muon_run8329_comparison_hist_v2:
    input:
        metadata=PASSING_MUON_V2_PARTITION_METADATA,
        validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        eval_config=PASSING_MUON_V2_EVAL_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_efficiency_hist.py",
    output:
        root=_pmv2_eval_root + "/histograms/{selection}.root",
    wildcard_constraints:
        selection="|".join(PASSING_MUON_V2_SELECTIONS)
    threads: 4
    resources:
        runtime=4 * 60 * 60, mem_mb=6000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} --metadata {input.metadata} --hist-config {input.hist_config} \
          --eval-config {input.eval_config} --selection {wildcards.selection} \
          --output {output.root} --threads {threads}
        """


rule passing_muon_run8329_comparison_plot_v2:
    input:
        root=_pmv2_eval_root + "/histograms/{selection}.root",
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        eval_config=PASSING_MUON_V2_EVAL_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_passing_muon_efficiency.py",
    output:
        pdf=_pmv2_plot_root + "/comparison/{selection}.pdf",
    wildcard_constraints:
        selection="|".join(PASSING_MUON_V2_SELECTIONS)
    resources:
        runtime=30 * 60, mem_mb=2000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} --input {input.root} --hist-config {input.hist_config} \
          --eval-config {input.eval_config} --output-pdf {output.pdf}
        """


rule passing_muon_run8329_track_hist_v2:
    input:
        metadata=PASSING_MUON_V2_PARTITION_METADATA,
        validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_reco_track_inspection_hist.py",
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        PASSING_MUON_V2_TRACK_HISTS
    threads: 4
    resources:
        runtime=4 * 60 * 60, mem_mb=6000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python {input.script} --partition-metadata {input.metadata} --region all_events \
          --particle-group passing_muon_dq_run8329 --output {output} --threads {threads}
        """


rule passing_muon_run8329_track_plots_v2:
    input:
        hists=PASSING_MUON_V2_TRACK_HISTS,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_reco_track_inspection.py",
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        plots=PASSING_MUON_V2_TRACK_PLOTS,
        done=PASSING_MUON_V2_TRACK_DONE,
    resources:
        runtime=60 * 60, mem_mb=3000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python {input.script} --input {input.hists} --output-dir {PASSING_MUON_V2_TRACK_DIR}/plots --done {output.done}
        """


rule passing_muon_run8329_comparison_v2:
    input:
        metadata_validation=PASSING_MUON_V2_METADATA_VALIDATION,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        roots=PASSING_MUON_V2_EVAL_ROOTS,
        plots=PASSING_MUON_V2_EVAL_PLOTS,
        track_plots=PASSING_MUON_V2_TRACK_PLOTS,
        track_done=PASSING_MUON_V2_TRACK_DONE,
