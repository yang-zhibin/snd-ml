REGION_V2_VETO_CONFIG = config.get(
    "region_v2_veto_config",
    (
        f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/"
        "region_v2_inspection/veto_inspection_v1.yaml"
    ),
)

with open(REGION_V2_VETO_CONFIG, "r", encoding="utf-8") as handle:
    REGION_V2_VETO_CFG = yaml.safe_load(handle)

REGION_V2_VETO_VERSION = REGION_V2_VETO_CFG["version"]
REGION_V2_VETO_DATASETS = [
    item["name"] for item in REGION_V2_VETO_CFG["datasets"]
]
REGION_V2_VETO_MANIFEST = _region_eval_output_path(
    REGION_V2_VETO_CFG["input"]["manifest"]
)
REGION_V2_VETO_OUTPUT_ROOT = _region_eval_output_path(
    REGION_V2_VETO_CFG["output"]["root_dir"]
)
REGION_V2_VETO_HIST_DIR = f"{REGION_V2_VETO_OUTPUT_ROOT}/hists"
REGION_V2_VETO_SUMMARY_ROW_DIR = f"{REGION_V2_VETO_OUTPUT_ROOT}/summary_rows"
REGION_V2_VETO_PLOT_DIR = f"{REGION_V2_VETO_OUTPUT_ROOT}/plots"
REGION_V2_VETO_COMBINED_ROOT = (
    f"{REGION_V2_VETO_OUTPUT_ROOT}/{REGION_V2_VETO_VERSION}_histograms.root"
)
REGION_V2_VETO_SUMMARY = (
    f"{REGION_V2_VETO_OUTPUT_ROOT}/{REGION_V2_VETO_VERSION}_summary.csv"
)
REGION_V2_VETO_DONE = (
    f"{REGION_V2_VETO_OUTPUT_ROOT}/{REGION_V2_VETO_VERSION}.done"
)

REGION_V2_VETO_PLOT_STEMS = []
if REGION_V2_VETO_CFG["plots"].get("include_inclusive_1d", True):
    REGION_V2_VETO_PLOT_STEMS.extend([
        "veto_count_all_unit_area",
        "veto_count_all_weighted",
        "veto_earliest_time_all_unit_area",
        "veto_earliest_time_all_weighted",
    ])
REGION_V2_VETO_PLOT_STEMS.extend([
    "veto_count_positive_unit_area",
    "veto_count_positive_weighted",
    "veto_earliest_time_valid_unit_area",
    "veto_earliest_time_valid_weighted",
    "veto_time_vs_count_unit_area",
    "veto_time_vs_count_weighted",
    "veto_time_vs_scifi_count_unit_area",
    "veto_time_vs_scifi_count_weighted",
    "veto_time_vs_scifi_density_unit_area",
    "veto_time_vs_scifi_density_weighted",
    "veto_count_vs_scifi_count_unit_area",
    "veto_count_vs_scifi_count_weighted",
    "veto_count_vs_scifi_density_unit_area",
    "veto_count_vs_scifi_density_weighted",
])
REGION_V2_VETO_FORMATS = [
    str(value) for value in REGION_V2_VETO_CFG["plots"]["formats"]
]
REGION_V2_VETO_PLOTS = [
    f"{REGION_V2_VETO_PLOT_DIR}/{stem}.{extension}"
    for stem in REGION_V2_VETO_PLOT_STEMS
    for extension in REGION_V2_VETO_FORMATS
]


def _region_v2_veto_manifest_rows():
    dataframe = pd.read_csv(REGION_V2_VETO_MANIFEST)
    if "dataset_name" not in dataframe.columns or "output_path" not in dataframe.columns:
        raise ValueError(
            f"Veto inspection manifest is missing dataset_name/output_path: "
            f"{REGION_V2_VETO_MANIFEST}"
        )
    rows = dataframe.set_index("dataset_name", drop=False)
    missing = set(REGION_V2_VETO_DATASETS) - set(rows.index.astype(str))
    if missing:
        raise ValueError(
            f"Veto inspection manifest is missing configured datasets: {sorted(missing)}"
        )
    return rows


REGION_V2_VETO_MANIFEST_ROWS = _region_v2_veto_manifest_rows()


def _region_v2_veto_input_root(wildcards):
    return str(REGION_V2_VETO_MANIFEST_ROWS.loc[wildcards.dataset, "output_path"])


rule region_v2_veto_inspection_hist:
    input:
        config=REGION_V2_VETO_CONFIG,
        manifest=REGION_V2_VETO_MANIFEST,
        root=_region_v2_veto_input_root,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "make_veto_inspection_hist.py"
        ),
    output:
        hist=f"{REGION_V2_VETO_HIST_DIR}/{{dataset}}.root",
        summary=f"{REGION_V2_VETO_SUMMARY_ROW_DIR}/{{dataset}}.csv",
    wildcard_constraints:
        dataset="|".join(REGION_V2_VETO_DATASETS)
    threads: 2
    resources:
        runtime=2 * 60 * 60,
        mem_mb=5000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        echo "building Region V2 veto inspection histogram for {wildcards.dataset}"
        set +u
        source "{env_script_lcg}"
        set -euo pipefail

        mkdir -p "$(dirname "{output.hist}")" "$(dirname "{output.summary}")"
        python "{input.script}" \
          --config "{input.config}" \
          --manifest "{input.manifest}" \
          --dataset "{wildcards.dataset}" \
          --output "{output.hist}" \
          --summary "{output.summary}" \
          --threads {threads} \
          --repo-root "{PERSONAL_WORK_SPACE}"
        """


rule region_v2_veto_inspection_plots:
    input:
        config=REGION_V2_VETO_CONFIG,
        script=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "plot_veto_inspection.py"
        ),
        hists=expand(
            f"{REGION_V2_VETO_HIST_DIR}/{{dataset}}.root",
            dataset=REGION_V2_VETO_DATASETS,
        ),
        summaries=expand(
            f"{REGION_V2_VETO_SUMMARY_ROW_DIR}/{{dataset}}.csv",
            dataset=REGION_V2_VETO_DATASETS,
        ),
    output:
        plots=REGION_V2_VETO_PLOTS,
        combined_root=REGION_V2_VETO_COMBINED_ROOT,
        summary=REGION_V2_VETO_SUMMARY,
        done=REGION_V2_VETO_DONE,
    params:
        hist_args=lambda wildcards, input: " ".join(
            f'--hist "{dataset}={path}"'
            for dataset, path in zip(REGION_V2_VETO_DATASETS, input.hists)
        ),
        summary_args=lambda wildcards, input: " ".join(
            f'--summary-row "{dataset}={path}"'
            for dataset, path in zip(REGION_V2_VETO_DATASETS, input.summaries)
        ),
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting Region V2 veto inspection"
        set +u
        source "{env_script_lcg}"
        set -euo pipefail

        mkdir -p "{REGION_V2_VETO_PLOT_DIR}"
        python "{input.script}" \
          --config "{input.config}" \
          {params.hist_args} \
          {params.summary_args} \
          --output-dir "{REGION_V2_VETO_PLOT_DIR}" \
          --combined-root "{output.combined_root}" \
          --summary "{output.summary}" \
          --done "{output.done}"
        """


rule region_v2_veto_inspection:
    input:
        REGION_V2_VETO_PLOTS,
        REGION_V2_VETO_COMBINED_ROOT,
        REGION_V2_VETO_SUMMARY,
        REGION_V2_VETO_DONE,
