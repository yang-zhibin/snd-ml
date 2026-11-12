import sys
sys.path.insert(0, str(Path(workflow.basedir).resolve()))
from output_paths import resolve_output

# Run-8329 v4: exhaustive detector-region slices and selected joint regions.

PASSING_MUON_V4_CONFIG = f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/passing_muon_region_study_v4.yaml"
PASSING_MUON_V4_FEATURE_HELPER = (
    f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
    "passing_muon_v4_features.py"
)
with open(PASSING_MUON_V4_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V4 = yaml.safe_load(stream)

PASSING_MUON_V4_AXIS_COMPARISONS = [
    f"{axis}_regions" for axis in PASSING_MUON_V4["axes"]
]
PASSING_MUON_V4_SELECTED_COMPARISONS = [
    f"region__{region}"
    for region in PASSING_MUON_V4["selected_joint_regions"]
]
PASSING_MUON_V4_COMPARISONS = (
    PASSING_MUON_V4_AXIS_COMPARISONS
    + ["joint_region_summary"]
    + PASSING_MUON_V4_SELECTED_COMPARISONS
)
_pmv4_root = PASSING_MUON_V4["output"]["root_dir"].rstrip("/")
_pmv4_plot_root = (
    resolve_output(PASSING_MUON_V4['output']['plot_dir'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_V4_HIST = f"{_pmv4_root}/histograms/regions.root"
PASSING_MUON_V4_PLOTS = [
    f"{_pmv4_plot_root}/{comparison}.pdf"
    for comparison in PASSING_MUON_V4_COMPARISONS
]


rule make_passing_muon_region_hist_v4:
    input:
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V4_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/make_passing_muon_region_hist_v4.py",
        feature_helper=PASSING_MUON_V4_FEATURE_HELPER,
    output:
        root=PASSING_MUON_V4_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    params:
        # V4 is an analysis of the already validated V2/V3 production.  Keep
        # these paths out of ``input`` so a plotting-code rerun cannot traverse
        # their producer rules and rebuild all run-8329 feature files.
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        missing_metadata=PASSING_MUON_V3_MISSING_METADATA,
        missing_validation=PASSING_MUON_V3_VALIDATION,
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        test -s {params.partition_metadata}
        test -s {params.partition_validation}
        test -s {params.missing_metadata}
        test -s {params.missing_validation}
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/region_study_hist_v4.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {params.partition_metadata} \
          --missing-metadata {params.missing_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule passing_muon_run8329_region_study_hist_v4:
    input:
        PASSING_MUON_V4_HIST


rule plot_passing_muon_region_study_v4:
    input:
        root=PASSING_MUON_V4_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V4_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/plot_passing_muon_region_study_v4.py",
        feature_helper=PASSING_MUON_V4_FEATURE_HELPER,
    output:
        pdf=_pmv4_plot_root + "/{comparison}.pdf",
    wildcard_constraints:
        comparison="|".join(PASSING_MUON_V4_COMPARISONS)
    threads: 1
    resources:
        runtime=45 * 60, mem_mb=3000, disk_mb=1500, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --comparison {wildcards.comparison} \
          --output-pdf {output.pdf}
        """


rule passing_muon_run8329_region_study_plots_v4:
    input:
        PASSING_MUON_V4_PLOTS


rule passing_muon_run8329_region_study_test_v4:
    input:
        config=PASSING_MUON_V4_CONFIG,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/test_passing_muon_region_study_v4.py",
        histogrammer=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/make_passing_muon_region_hist_v4.py",
        plotter=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/plot_passing_muon_region_study_v4.py",
        feature_helper=PASSING_MUON_V4_FEATURE_HELPER,
        common_builder=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/make_passing_muon_track_type_hist.py",
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=3000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script}
        """


rule passing_muon_run8329_region_study_v4:
    input:
        histogram=PASSING_MUON_V4_HIST,
        plots=PASSING_MUON_V4_PLOTS,


# V4 veto-time event displays are deliberately separate from the normal V4
# histogram/PDF aggregate. Producing the study must not unexpectedly launch
# eighty SNDSW event-display pages.
PASSING_MUON_V4_EVENT_DISPLAY_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_v4.yaml"
)
with open(PASSING_MUON_V4_EVENT_DISPLAY_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V4_EVENT_DISPLAY = yaml.safe_load(stream)

_pmv4_event_populations = ("passing", "missing")
_pmv4_event_veto_categories = list(
    PASSING_MUON_V4_EVENT_DISPLAY["selection"]["categories"]
)
PASSING_MUON_V4_EVENT_CATEGORIES = [
    f"{population}__{category}"
    for population in _pmv4_event_populations
    for category in _pmv4_event_veto_categories
]
_pmv4_event_root = (
    resolve_output(PASSING_MUON_V4_EVENT_DISPLAY['output']['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
)
PASSING_MUON_V4_EVENT_MANIFEST = (
    f"{_pmv4_event_root}/"
    f"{PASSING_MUON_V4_EVENT_DISPLAY['output']['selected_manifest']}"
)
PASSING_MUON_V4_EVENT_COUNTS = (
    f"{_pmv4_event_root}/"
    f"{PASSING_MUON_V4_EVENT_DISPLAY['output']['category_counts']}"
)
PASSING_MUON_V4_RENDERED_MANIFEST = (
    f"{_pmv4_event_root}/"
    f"{PASSING_MUON_V4_EVENT_DISPLAY['output']['rendered_manifest']}"
)
PASSING_MUON_V4_EVENT_PDFS = [
    f"{_pmv4_event_root}/pdf/{category}.pdf"
    for category in PASSING_MUON_V4_EVENT_CATEGORIES
]
PASSING_MUON_V4_EVENT_SUMMARIES = [
    f"{_pmv4_event_root}/summaries/{category}.csv"
    for category in PASSING_MUON_V4_EVENT_CATEGORIES
]


rule select_passing_muon_v4_veto_time_event_display_events:
    input:
        config=PASSING_MUON_V4_EVENT_DISPLAY_CONFIG,
        study_config=PASSING_MUON_V4_CONFIG,
        reference_histogram=PASSING_MUON_V4_HIST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_passing_muon_v4_event_display_events.py"
        ),
        region_engine=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_region_hist_v4.py"
        ),
        common_builder=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        manifest=PASSING_MUON_V4_EVENT_MANIFEST,
        counts=PASSING_MUON_V4_EVENT_COUNTS,
    params:
        # These already-validated metadata products are frozen inputs for the
        # diagnostic display workflow. Keep them out of ``input`` so Snakemake
        # cannot traverse and rebuild the 299-file production chain.
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        missing_metadata=PASSING_MUON_V3_MISSING_METADATA,
        missing_validation=PASSING_MUON_V3_VALIDATION,
        real_metadata=PASSING_MUON_V2_REAL_METADATA,
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=5000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        test -s {params.partition_metadata}
        test -s {params.partition_validation}
        test -s {params.missing_metadata}
        test -s {params.missing_validation}
        test -s {params.real_metadata}
        python {input.script} \
          --config {input.config} \
          --partition-metadata {params.partition_metadata} \
          --missing-metadata {params.missing_metadata} \
          --real-data-metadata {params.real_metadata} \
          --reference-histogram {input.reference_histogram} \
          --output-manifest {output.manifest} \
          --output-counts {output.counts} \
          --repo-root {PERSONAL_WORK_SPACE}
        """


rule plot_passing_muon_v4_veto_time_event_display:
    input:
        config=PASSING_MUON_V4_EVENT_DISPLAY_CONFIG,
        manifest=PASSING_MUON_V4_EVENT_MANIFEST,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_passing_muon_v4_event_displays.py"
        ),
        common=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
    output:
        pdf=f"{_pmv4_event_root}/pdf/{{category}}.pdf",
        summary=f"{_pmv4_event_root}/summaries/{{category}}.csv",
    wildcard_constraints:
        category="|".join(PASSING_MUON_V4_EVENT_CATEGORIES)
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


rule merge_passing_muon_v4_veto_time_event_display_summaries:
    input:
        summaries=PASSING_MUON_V4_EVENT_SUMMARIES,
        config=PASSING_MUON_V4_EVENT_DISPLAY_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "merge_passing_muon_v4_event_display_summaries.py"
        ),
    output:
        rendered=PASSING_MUON_V4_RENDERED_MANIFEST,
    params:
        summaries=" ".join(PASSING_MUON_V4_EVENT_SUMMARIES),
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --output {output.rendered} \
          {params.summaries}
        """


rule passing_muon_run8329_veto_time_event_displays_v4:
    input:
        manifest=PASSING_MUON_V4_EVENT_MANIFEST,
        counts=PASSING_MUON_V4_EVENT_COUNTS,
        pdfs=PASSING_MUON_V4_EVENT_PDFS,
        rendered=PASSING_MUON_V4_RENDERED_MANIFEST,


rule passing_muon_run8329_region_study_with_event_displays_v4:
    input:
        histogram=PASSING_MUON_V4_HIST,
        plots=PASSING_MUON_V4_PLOTS,
        event_manifest=PASSING_MUON_V4_EVENT_MANIFEST,
        event_counts=PASSING_MUON_V4_EVENT_COUNTS,
        event_pdfs=PASSING_MUON_V4_EVENT_PDFS,
        rendered_events=PASSING_MUON_V4_RENDERED_MANIFEST,
