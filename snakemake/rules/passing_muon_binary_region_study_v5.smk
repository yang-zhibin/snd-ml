# Run-8329 v5: detector-only through-going-muon selection study.

PASSING_MUON_V5_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v5.yaml"
with open(PASSING_MUON_V5_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V5 = yaml.safe_load(stream)

PASSING_MUON_V5_ENABLED_SELECTIONS = [
    *PASSING_MUON_V5["coverage_working_points"],
    *[
        name
        for name, definition in PASSING_MUON_V5.get(
            "selection_working_points", {}
        ).items()
        if definition.get("enabled", False)
    ],
]
PASSING_MUON_V5_MC_PRODUCTIONS = list(
    PASSING_MUON_V5["mc_muon_up"]["productions"]
)
PASSING_MUON_V5_MC_AGGREGATE = PASSING_MUON_V5["mc_muon_up"].get(
    "aggregate", {}
)
PASSING_MUON_V5_MC_AGGREGATE_COMPARISONS = [
    f"{selection}__mc__{PASSING_MUON_V5_MC_AGGREGATE['key']}"
    for selection in PASSING_MUON_V5_MC_AGGREGATE.get("selections", [])
]
PASSING_MUON_V5_COMPARISONS = [
    "diagnostics",
    "mc_muon_up_summary",
    *PASSING_MUON_V5_ENABLED_SELECTIONS,
    *[
        f"{selection}__mc__{production}"
        for selection in PASSING_MUON_V5_ENABLED_SELECTIONS
        for production in PASSING_MUON_V5_MC_PRODUCTIONS
    ],
    *PASSING_MUON_V5_MC_AGGREGATE_COMPARISONS,
]
_pmv5_root = PASSING_MUON_V5["output"]["root_dir"].rstrip("/")
_pmv5_plot_root = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V5['output']['plot_dir'].rstrip('/')}"
)
PASSING_MUON_V5_MC_UP_METADATA = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V5['mc_muon_up']['metadata_csv']}"
)
PASSING_MUON_V5_MC_UP_VALIDATION = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V5['mc_muon_up']['validation_csv']}"
)
PASSING_MUON_V5_MC_UP_SOURCE_METADATA = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V5['mc_muon_up']['source_metadata']}"
)


def _passing_muon_v5_mc_up_features():
    metadata = pd.read_csv(
        PASSING_MUON_V5_MC_UP_SOURCE_METADATA, keep_default_na=False
    )
    mc_config = PASSING_MUON_V5["mc_muon_up"]
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
            f"V5 expected {expected} MC-muon-up rows, found {len(rows)}"
        )
    return sorted(
        os.path.join(str(row.output_base_path), str(row.feature_path))
        if not os.path.isabs(str(row.feature_path))
        else str(row.feature_path)
        for row in rows.itertuples()
    )


PASSING_MUON_V5_MC_UP_FEATURES = _passing_muon_v5_mc_up_features()
PASSING_MUON_V5_HIST = f"{_pmv5_root}/histograms/detector_selection.root"
PASSING_MUON_V5_PLOTS = [
    f"{_pmv5_plot_root}/{comparison}.pdf"
    for comparison in PASSING_MUON_V5_COMPARISONS
]


rule prepare_mc_muon_up_reference_v5:
    input:
        source_metadata=PASSING_MUON_V5_MC_UP_SOURCE_METADATA,
        feature_files=PASSING_MUON_V5_MC_UP_FEATURES,
        config=PASSING_MUON_V5_CONFIG,
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
        metadata=PASSING_MUON_V5_MC_UP_METADATA,
        validation=PASSING_MUON_V5_MC_UP_VALIDATION,
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


rule make_passing_muon_binary_region_hist_v5:
    input:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        mc_muon_up_metadata=PASSING_MUON_V5_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V5_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V5_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py",
        common_builder=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_track_type_hist.py",
    output:
        root=PASSING_MUON_V5_HIST,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/detector_selection_hist_v5.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {input.partition_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule passing_muon_run8329_binary_region_study_hist_v5:
    input:
        PASSING_MUON_V5_HIST


rule plot_passing_muon_binary_region_study_v5:
    input:
        root=PASSING_MUON_V5_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V5_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_passing_muon_binary_region_study_v5.py",
        builder=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py",
        common_plotter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_passing_muon_region_study_v4.py",
    output:
        pdf=_pmv5_plot_root + "/{comparison}.pdf",
    wildcard_constraints:
        comparison="|".join(PASSING_MUON_V5_COMPARISONS)
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


rule passing_muon_run8329_binary_region_study_plots_v5:
    input:
        PASSING_MUON_V5_PLOTS


rule passing_muon_run8329_binary_region_study_test_v5:
    input:
        config=PASSING_MUON_V5_CONFIG,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        histogrammer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py",
        common_builder=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_passing_muon_track_type_hist.py",
        plotter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_passing_muon_binary_region_study_v5.py",
        common_plotter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_passing_muon_region_study_v4.py",
        mc_preparer=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "prepare_mc_muon_up_reference_v5.py"
        ),
        identity_cutflow=(
            f"{PERSONAL_WORK_SPACE}/convertData/make_identity_cutflow.py"
        ),
        test=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/test_passing_muon_binary_region_study_v5.py",
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=3000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.test}
        """


rule passing_muon_run8329_binary_region_study_v5:
    input:
        histogram=PASSING_MUON_V5_HIST,
        plots=PASSING_MUON_V5_PLOTS,
