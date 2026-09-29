# V6: V5-compatible detector study with pooled MC in both coverage PDFs.

PASSING_MUON_V6_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v6.yaml"
)
with open(PASSING_MUON_V6_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V6 = yaml.safe_load(stream)

PASSING_MUON_V6_SELECTIONS = ["coverage_loose", "coverage_tight"]
_pmv6_root = PASSING_MUON_V6["output"]["root_dir"].rstrip("/")
_pmv6_plot_root = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V6['output']['plot_dir'].rstrip('/')}"
)
PASSING_MUON_V6_MC_UP_SOURCE_METADATA = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V6['mc_muon_up']['source_metadata']}"
)
PASSING_MUON_V6_MC_UP_METADATA = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V6['mc_muon_up']['metadata_csv']}"
)
PASSING_MUON_V6_MC_UP_VALIDATION = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V6['mc_muon_up']['validation_csv']}"
)


def _passing_muon_v6_mc_up_features():
    metadata = pd.read_csv(
        PASSING_MUON_V6_MC_UP_SOURCE_METADATA, keep_default_na=False
    )
    mc_config = PASSING_MUON_V6["mc_muon_up"]
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
            f"V6 expected {expected} MC-muon-up rows, found {len(rows)}"
        )
    return sorted(
        os.path.join(str(row.output_base_path), str(row.feature_path))
        if not os.path.isabs(str(row.feature_path))
        else str(row.feature_path)
        for row in rows.itertuples()
    )


PASSING_MUON_V6_MC_UP_FEATURES = _passing_muon_v6_mc_up_features()
PASSING_MUON_V6_HIST = (
    f"{_pmv6_root}/histograms/detector_selection_combined.root"
)
PASSING_MUON_V6_PLOTS = [
    f"{_pmv6_plot_root}/{selection}.pdf"
    for selection in PASSING_MUON_V6_SELECTIONS
]


rule prepare_mc_muon_up_reference_v6:
    input:
        source_metadata=PASSING_MUON_V6_MC_UP_SOURCE_METADATA,
        config=PASSING_MUON_V6_CONFIG,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "prepare_mc_muon_up_reference_v5.py"
        ),
        histogrammer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
    output:
        metadata=PASSING_MUON_V6_MC_UP_METADATA,
        validation=PASSING_MUON_V6_MC_UP_VALIDATION,
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


rule make_passing_muon_binary_region_hist_v6:
    input:
        mc_muon_up_metadata=PASSING_MUON_V6_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V6_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V6_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v6.py"
        ),
        engine=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V6_HIST,
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
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/detector_selection_hist_v6.XXXXXX")
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


rule plot_passing_muon_binary_region_study_v6:
    input:
        root=PASSING_MUON_V6_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V6_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_binary_region_study_v6.py"
        ),
        engine=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv6_plot_root + "/{selection}.pdf",
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


rule passing_muon_run8329_binary_region_study_hist_v6:
    input:
        PASSING_MUON_V6_HIST


rule passing_muon_run8329_binary_region_study_plots_v6:
    input:
        PASSING_MUON_V6_PLOTS


rule passing_muon_run8329_binary_region_study_v6:
    input:
        histogram=PASSING_MUON_V6_HIST,
        plots=PASSING_MUON_V6_PLOTS,


# V6 loose-selection event displays are a separate aggregate so producing the
# detector-study PDFs does not unexpectedly launch 80 SNDSW display pages.
PASSING_MUON_V6_EVENT_DISPLAY_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_v6.yaml"
)
with open(PASSING_MUON_V6_EVENT_DISPLAY_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V6_EVENT_DISPLAY = yaml.safe_load(stream)

_pmv6_samples = ("run8329", "mc_muon_up")
_pmv6_veto_categories = list(
    PASSING_MUON_V6_EVENT_DISPLAY["selection"]["failure_veto_categories"]
)
PASSING_MUON_V6_EVENT_CATEGORIES = [
    category
    for sample in _pmv6_samples
    for category in (
        f"{sample}__candidate",
        *(f"{sample}__failure__{veto}" for veto in _pmv6_veto_categories),
    )
]
_pmv6_event_root = (
    f"{PERSONAL_WORK_SPACE}/"
    f"{PASSING_MUON_V6_EVENT_DISPLAY['output']['directory'].rstrip('/')}"
)
PASSING_MUON_V6_EVENT_MANIFEST = (
    f"{_pmv6_event_root}/"
    f"{PASSING_MUON_V6_EVENT_DISPLAY['output']['selected_manifest']}"
)
PASSING_MUON_V6_EVENT_COUNTS = (
    f"{_pmv6_event_root}/"
    f"{PASSING_MUON_V6_EVENT_DISPLAY['output']['category_counts']}"
)
PASSING_MUON_V6_RENDERED_MANIFEST = (
    f"{_pmv6_event_root}/"
    f"{PASSING_MUON_V6_EVENT_DISPLAY['output']['rendered_manifest']}"
)
PASSING_MUON_V6_EVENT_PDFS = [
    f"{_pmv6_event_root}/pdf/{category}.pdf"
    for category in PASSING_MUON_V6_EVENT_CATEGORIES
]
PASSING_MUON_V6_EVENT_SUMMARIES = [
    f"{_pmv6_event_root}/summaries/{category}.csv"
    for category in PASSING_MUON_V6_EVENT_CATEGORIES
]


rule select_passing_muon_v6_loose_event_display_events:
    input:
        config=PASSING_MUON_V6_EVENT_DISPLAY_CONFIG,
        study_config=PASSING_MUON_V6_CONFIG,
        reference_histogram=PASSING_MUON_V6_HIST,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "select_passing_muon_v6_event_display_events.py"
        ),
        engine=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    params:
        # The already-validated V6 source metadata paths are read from the
        # event-display config. Keeping them as parameters prevents this
        # diagnostic target from rebuilding the full 299-file V2 production.
        real_metadata=PASSING_MUON_V2_REAL_METADATA,
        mc_metadata=PASSING_MUON_V6_MC_UP_SOURCE_METADATA,
    output:
        manifest=PASSING_MUON_V6_EVENT_MANIFEST,
        counts=PASSING_MUON_V6_EVENT_COUNTS,
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


rule plot_passing_muon_v6_loose_event_display:
    input:
        config=PASSING_MUON_V6_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_V6_EVENT_MANIFEST,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_v6_event_displays.py"
        ),
        common=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=f"{_pmv6_event_root}/pdf/{{category}}.pdf",
        summary=f"{_pmv6_event_root}/summaries/{{category}}.csv",
    wildcard_constraints:
        category="|".join(PASSING_MUON_V6_EVENT_CATEGORIES)
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


rule merge_passing_muon_v6_loose_event_display_summaries:
    input:
        summaries=PASSING_MUON_V6_EVENT_SUMMARIES,
        config=PASSING_MUON_V6_EVENT_DISPLAY_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "merge_passing_muon_v6_event_display_summaries.py"
        ),
    output:
        rendered=PASSING_MUON_V6_RENDERED_MANIFEST,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    params:
        summaries=" ".join(PASSING_MUON_V6_EVENT_SUMMARIES),
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --output {output.rendered} \
          {params.summaries}
        """


rule passing_muon_run8329_loose_event_displays_v6:
    input:
        manifest=PASSING_MUON_V6_EVENT_MANIFEST,
        counts=PASSING_MUON_V6_EVENT_COUNTS,
        pdfs=PASSING_MUON_V6_EVENT_PDFS,
        rendered=PASSING_MUON_V6_RENDERED_MANIFEST,
