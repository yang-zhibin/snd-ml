SPLIT_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/splits/split_v1.yaml"

with open(SPLIT_CONFIG, "r", encoding="utf-8") as handle:
    SPLIT_CONFIG_DATA = yaml.safe_load(handle)

SPLIT_VERSION = SPLIT_CONFIG_DATA["split_version"]


def _post_partition_ml_path(config_value):
    if os.path.isabs(config_value) or str(config_value).startswith("root://"):
        return str(config_value)
    return f"{PERSONAL_WORK_SPACE}/{config_value}"


REGION_PARTITION_SPLIT_METADATA_CSV = _post_partition_ml_path(
    SPLIT_CONFIG_DATA["output"]["metadata_csv"]
)
REGION_PARTITION_SPLIT_SUMMARY_CSV = _post_partition_ml_path(
    SPLIT_CONFIG_DATA["output"]["summary_csv"]
)
REGION_PARTITION_SPLIT_AUDIT_SUMMARY_CSV = REGION_PARTITION_SPLIT_SUMMARY_CSV.replace(
    "_summary.csv",
    "_summary_audit.csv",
)
REGION_PARTITION_SPLIT_SUMMARY_REPORT = REGION_PARTITION_SPLIT_SUMMARY_CSV.replace(
    "_summary.csv",
    "_summary_report.txt",
)


rule region_partition_splits:
    input:
        metadata=REGION_PARTITION_SPLIT_METADATA_CSV,
        summary=REGION_PARTITION_SPLIT_SUMMARY_CSV


rule summarize_region_partition_splits:
    input:
        summary=REGION_PARTITION_SPLIT_AUDIT_SUMMARY_CSV,
        report=REGION_PARTITION_SPLIT_SUMMARY_REPORT


rule split_region_partitions:
    input:
        partition_metadata=REGION_PARTITION_METADATA_CSV,
        split_config=SPLIT_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/convertData/split_region_partitions.py"
    output:
        metadata=REGION_PARTITION_SPLIT_METADATA_CSV,
        summary=REGION_PARTITION_SPLIT_SUMMARY_CSV
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "Splitting region partition metadata for {SPLIT_VERSION}"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python3 {input.script} \
            --input {input.partition_metadata} \
            --config {input.split_config} \
            --output {output.metadata} \
            --summary {output.summary}
        """


rule summarize_split_metadata:
    input:
        split_metadata=REGION_PARTITION_SPLIT_METADATA_CSV,
        script=f"{PERSONAL_WORK_SPACE}/convertData/summarize_split_metadata.py"
    output:
        summary=REGION_PARTITION_SPLIT_AUDIT_SUMMARY_CSV,
        report=REGION_PARTITION_SPLIT_SUMMARY_REPORT
    threads: 1
    resources:
        runtime=15 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "Summarizing split metadata for {SPLIT_VERSION}"
        set +u
        source /afs/cern.ch/user/z/zhibin/env_lcg.sh
        set -euo pipefail

        python3 {input.script} \
            --input {input.split_metadata} \
            --output {output.summary} \
            --report {output.report}
        """
