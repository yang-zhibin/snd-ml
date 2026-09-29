PASSING_MUON_TRACK_RUN = 8329
PASSING_MUON_TRACK_PARTITION_COUNT = 20
PASSING_MUON_TRACK_EXPECTED_FILES = 2983
PASSING_MUON_TRACK_EXPECTED_EVENTS = 82145120
PASSING_MUON_TRACK_EXPECTED_LUMI = 0.5323492746290897

PASSING_MUON_TRACK_RUN_METADATA = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    "passing_muon_track_features_v1_run008329_metadata.csv"
)
PASSING_MUON_TRACK_PARTITION_REGION_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/regions/"
    "passing_muon_track_inspection_region_v1.yaml"
)
PASSING_MUON_TRACK_PARTITION_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/partitions/"
    "passing_muon_track_partition_v1.yaml"
)
PASSING_MUON_TRACK_PARTITION_VERSION = "passing_muon_track_partition_v1"
PASSING_MUON_TRACK_PARTITION_REGION_VERSION = (
    "passing_muon_track_inspection_region_v1"
)
PASSING_MUON_TRACK_PARTITION_ROOT = (
    "/eos/experiment/sndlhc/users/zhibin/snd-ml/"
    "passing_muon_track_partitions"
)
PASSING_MUON_TRACK_PARTITION_DIR = (
    f"{PASSING_MUON_TRACK_PARTITION_ROOT}/"
    f"{PASSING_MUON_TRACK_PARTITION_VERSION}/"
    f"{PASSING_MUON_TRACK_PARTITION_REGION_VERSION}/"
    "all_events/passing_muon"
)
PASSING_MUON_TRACK_PARTITION_ROW_DIR = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/"
    f"{PASSING_MUON_TRACK_PARTITION_VERSION}_"
    f"{PASSING_MUON_TRACK_PARTITION_REGION_VERSION}/rows"
)
PASSING_MUON_TRACK_PARTITION_METADATA = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/"
    f"{PASSING_MUON_TRACK_PARTITION_VERSION}_"
    f"{PASSING_MUON_TRACK_PARTITION_REGION_VERSION}.csv"
)
PASSING_MUON_TRACK_PARTITION_VALIDATION = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/"
    f"{PASSING_MUON_TRACK_PARTITION_VERSION}_"
    f"{PASSING_MUON_TRACK_PARTITION_REGION_VERSION}_validation.csv"
)


def _passing_muon_track_part_id(part_index):
    return f"{int(part_index):03d}"


def _passing_muon_track_partition_root(part_index):
    part = _passing_muon_track_part_id(part_index)
    return (
        f"{PASSING_MUON_TRACK_PARTITION_DIR}/"
        f"feature_all_events__passing_muon__part{part}.root"
    )


def _passing_muon_track_partition_row(part_index):
    part = _passing_muon_track_part_id(part_index)
    return (
        f"{PASSING_MUON_TRACK_PARTITION_ROW_DIR}/"
        f"all_events__passing_muon__part{part}.csv"
    )


PASSING_MUON_TRACK_PARTITION_INDICES = [
    _passing_muon_track_part_id(value)
    for value in range(1, PASSING_MUON_TRACK_PARTITION_COUNT + 1)
]
PASSING_MUON_TRACK_PARTITION_ROOTS = [
    _passing_muon_track_partition_root(value)
    for value in PASSING_MUON_TRACK_PARTITION_INDICES
]
PASSING_MUON_TRACK_PARTITION_ROWS = [
    _passing_muon_track_partition_row(value)
    for value in PASSING_MUON_TRACK_PARTITION_INDICES
]

PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
    "region_v2_inspection/passing_muon_track_partitions_v1"
)
PASSING_MUON_TRACK_PARTITION_HISTS = (
    f"{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/"
    "passing_muon_track_partitions_v1_histograms.root"
)
PASSING_MUON_TRACK_PARTITION_DONE = (
    f"{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/"
    "passing_muon_track_partitions_v1.done"
)
PASSING_MUON_TRACK_PARTITION_EQUIVALENCE = (
    f"{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/"
    "legacy_partition_histogram_equivalence.csv"
)
PASSING_MUON_TRACK_PARTITION_PLOTS = [
    f"{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/plots/{prefix}_{suffix}.pdf"
    for prefix in PASSING_MUON_TRACK_PLOT_PREFIXES
    for suffix in PASSING_MUON_TRACK_PLOT_SUFFIXES
] + [
    f"{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/plots/"
    f"reco_{method}_{suffix}.pdf"
    for method in ("simple", "hough")
    for suffix in (
        "ds_n_all_vs_scifi_n_all",
        "ds_best_chi2ndf_vs_scifi_best_chi2ndf",
    )
]


rule passing_muon_track_run008329_metadata:
    input:
        metadata=PASSING_MUON_TRACK_METADATA,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "filter_passing_muon_track_metadata.py"
        ),
    output:
        PASSING_MUON_TRACK_RUN_METADATA
    shell:
        """
        python3 {input.script} \
          --input {input.metadata} \
          --output {output} \
          --run {PASSING_MUON_TRACK_RUN} \
          --expected-files {PASSING_MUON_TRACK_EXPECTED_FILES} \
          --expected-events {PASSING_MUON_TRACK_EXPECTED_EVENTS} \
          --expected-lumi {PASSING_MUON_TRACK_EXPECTED_LUMI}
        """


rule build_passing_muon_track_partition:
    input:
        metadata=PASSING_MUON_TRACK_RUN_METADATA,
        region_config=PASSING_MUON_TRACK_PARTITION_REGION_CONFIG,
        partition_config=PASSING_MUON_TRACK_PARTITION_CONFIG,
        builder=f"{PERSONAL_WORK_SPACE}/convertData/build_region_partitions.py",
    output:
        root=(
            PASSING_MUON_TRACK_PARTITION_DIR
            + "/feature_all_events__passing_muon__part{part_index}.root"
        ),
        row=(
            PASSING_MUON_TRACK_PARTITION_ROW_DIR
            + "/all_events__passing_muon__part{part_index}.csv"
        ),
    wildcard_constraints:
        part_index=r"\d{3}"
    threads: 4
    resources:
        runtime=8 * 60 * 60,
        mem_mb=6000,
        disk_mb=12000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/passing_muon_track_partition.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_root="$tmp_dir/$(basename "{output.root}")"
        tmp_row="$tmp_dir/$(basename "{output.row}")"

        python "{input.builder}" \
          --metadata "{input.metadata}" \
          --region-config "{input.region_config}" \
          --partition-config "{input.partition_config}" \
          --output-dir "{PASSING_MUON_TRACK_PARTITION_ROOT}" \
          --output-metadata "$tmp_row" \
          --output-feature "$tmp_root" \
          --metadata-feature-output "{output.root}" \
          --only-regions all_events \
          --only-particles passing_muon \
          --only-part-index {wildcards.part_index} \
          --rdf-threads {threads} \
          --product feature

        mkdir -p "$(dirname "{output.root}")" "$(dirname "{output.row}")"
        xrdcp -f "$tmp_root" "{output.root}"
        tmp_row_publish="{output.row}.partial.${{RANDOM}}"
        cp "$tmp_row" "$tmp_row_publish"
        mv "$tmp_row_publish" "{output.row}"
        """


rule validate_passing_muon_track_partitions:
    input:
        rows=PASSING_MUON_TRACK_PARTITION_ROWS,
        roots=PASSING_MUON_TRACK_PARTITION_ROOTS,
        source_metadata=PASSING_MUON_TRACK_RUN_METADATA,
        partition_config=PASSING_MUON_TRACK_PARTITION_CONFIG,
        validator=(
            f"{PERSONAL_WORK_SPACE}/convertData/"
            "validate_passing_muon_track_partitions.py"
        ),
    output:
        metadata=PASSING_MUON_TRACK_PARTITION_METADATA,
        validation=PASSING_MUON_TRACK_PARTITION_VALIDATION,
    resources:
        runtime=2 * 60 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        python "{input.validator}" \
          --rows {input.rows} \
          --source-metadata "{input.source_metadata}" \
          --partition-config "{input.partition_config}" \
          --output-metadata "{output.metadata}" \
          --output-validation "{output.validation}" \
          --expected-files {PASSING_MUON_TRACK_EXPECTED_FILES} \
          --expected-events {PASSING_MUON_TRACK_EXPECTED_EVENTS} \
          --expected-lumi {PASSING_MUON_TRACK_EXPECTED_LUMI}
        """


rule passing_muon_track_partitions:
    input:
        roots=PASSING_MUON_TRACK_PARTITION_ROOTS,
        metadata=PASSING_MUON_TRACK_PARTITION_METADATA,
        validation=PASSING_MUON_TRACK_PARTITION_VALIDATION,


rule passing_muon_track_partition_hist:
    input:
        metadata=PASSING_MUON_TRACK_PARTITION_METADATA,
        validation=PASSING_MUON_TRACK_PARTITION_VALIDATION,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_reco_track_inspection_hist.py"
        ),
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        PASSING_MUON_TRACK_PARTITION_HISTS
    threads: 4
    resources:
        runtime=4 * 60 * 60,
        mem_mb=6000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python "{input.script}" \
          --partition-metadata "{input.metadata}" \
          --region all_events \
          --particle-group passing_muon \
          --output "{output}" \
          --threads {threads}
        """


rule passing_muon_track_partition_plots:
    input:
        hists=PASSING_MUON_TRACK_PARTITION_HISTS,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_reco_track_inspection.py"
        ),
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        plots=PASSING_MUON_TRACK_PARTITION_PLOTS,
        done=PASSING_MUON_TRACK_PARTITION_DONE,
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python "{input.script}" \
          --input "{input.hists}" \
          --output-dir "{PASSING_MUON_TRACK_PARTITION_INSPECTION_DIR}/plots" \
          --done "{output.done}"
        """


rule passing_muon_track_partition_inspection:
    input:
        PASSING_MUON_TRACK_PARTITION_PLOTS,
        PASSING_MUON_TRACK_PARTITION_DONE,
        PASSING_MUON_TRACK_PARTITION_METADATA,
        PASSING_MUON_TRACK_PARTITION_VALIDATION,
    resources:
        runtime=5 * 60,
        mem_mb=1000,
        disk_mb=100,
        nvidia_gpu=0


rule passing_muon_track_partition_equivalence:
    input:
        reference=PASSING_MUON_TRACK_HISTS,
        candidate=PASSING_MUON_TRACK_PARTITION_HISTS,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "compare_reco_track_histograms.py"
        ),
    output:
        PASSING_MUON_TRACK_PARTITION_EQUIVALENCE
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        python "{input.script}" \
          --reference "{input.reference}" \
          --candidate "{input.candidate}" \
          --output "{output}" \
          --tolerance 0
        """
