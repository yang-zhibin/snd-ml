import sys as region_v2_inspection_sys
region_v2_inspection_sys.path.insert(0, f"{PERSONAL_WORK_SPACE}/convertData")
from region_v2_inspection_common import load_config as region_v2_load_config
from region_v2_inspection_common import load_selected_rows as region_v2_load_rows

REGION_V2_INSPECTION_CONFIG = f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_v2_inspection/inspection_inputs_v1.yaml"
REGION_V2_INSPECTION_CFG = region_v2_load_config(REGION_V2_INSPECTION_CONFIG)
REGION_V2_INSPECTION_VERSION = REGION_V2_INSPECTION_CFG["version"]
REGION_V2_INSPECTION_DATASETS = list(REGION_V2_INSPECTION_CFG["datasets"])
REGION_V2_INSPECTION_ROOT_DIR = f'{REGION_V2_INSPECTION_CFG["output"]["root_dir"]}/{REGION_V2_INSPECTION_VERSION}'
REGION_V2_INSPECTION_META_DIR = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_v2_inspection"

def region_v2_inspection_metadata_input(wildcards):
    _, metadata_path, rows = region_v2_load_rows(REGION_V2_INSPECTION_CFG, wildcards.dataset)
    return str(metadata_path)

rule region_v2_inspection_input:
    input:
        config=REGION_V2_INSPECTION_CONFIG,
        metadata=region_v2_inspection_metadata_input,
    output:
        root=REGION_V2_INSPECTION_ROOT_DIR + "/{dataset}.root",
        manifest_row=REGION_V2_INSPECTION_META_DIR + "/rows/{dataset}.csv",
        source_audit=REGION_V2_INSPECTION_META_DIR + "/sources/{dataset}.csv",
    wildcard_constraints:
        dataset="|".join(REGION_V2_INSPECTION_DATASETS)
    params:
        builder=f"{PERSONAL_WORK_SPACE}/convertData/build_region_v2_inspection_input.py",
    threads: 4
    resources:
        mem_mb=8000,
        disk_mb=35000,
        runtime=1440,
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/region_v2_inspection.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        mkdir -p "$(dirname "{output.root}")" "$(dirname "{output.manifest_row}")" "$(dirname "{output.source_audit}")"
        python "{params.builder}" \
          --config "{input.config}" --dataset "{wildcards.dataset}" \
          --output "$tmp_dir/output.root" --published-path "{output.root}" \
          --manifest-row "{output.manifest_row}" --source-audit "{output.source_audit}" \
          --threads {threads}
        echo "[publish] Copying completed ROOT file to EOS staging path"
        partial="{output.root}.partial.${{RANDOM}}"
        cp "$tmp_dir/output.root" "$partial"
        echo "[publish] Atomically promoting EOS staging file to {output.root}"
        mv "$partial" "{output.root}"
        echo "[publish] Publication complete"
        """

rule validate_region_v2_inspection_input:
    input:
        config=REGION_V2_INSPECTION_CONFIG,
        root=REGION_V2_INSPECTION_ROOT_DIR + "/{dataset}.root",
        manifest_row=REGION_V2_INSPECTION_META_DIR + "/rows/{dataset}.csv",
        source_audit=REGION_V2_INSPECTION_META_DIR + "/sources/{dataset}.csv",
    output:
        REGION_V2_INSPECTION_META_DIR + "/validation_rows/{dataset}.csv"
    wildcard_constraints:
        dataset="|".join(REGION_V2_INSPECTION_DATASETS)
    params:
        validator=f"{PERSONAL_WORK_SPACE}/convertData/validate_region_v2_inspection_inputs.py",
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        python "{params.validator}" validate --config "{input.config}" \
          --dataset "{wildcards.dataset}" --input "{input.root}" \
          --manifest-row "{input.manifest_row}" --source-audit "{input.source_audit}" \
          --output "{output}"
        """

rule validate_region_v2_inspection_inputs:
    input:
        manifests=expand(REGION_V2_INSPECTION_META_DIR + "/rows/{dataset}.csv", dataset=REGION_V2_INSPECTION_DATASETS),
        audits=expand(REGION_V2_INSPECTION_META_DIR + "/sources/{dataset}.csv", dataset=REGION_V2_INSPECTION_DATASETS),
        validations=expand(REGION_V2_INSPECTION_META_DIR + "/validation_rows/{dataset}.csv", dataset=REGION_V2_INSPECTION_DATASETS),
    output:
        manifest=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_manifest.csv",
        audit=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_source_rows.csv",
        validation=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_validation.csv",
    params:
        validator=f"{PERSONAL_WORK_SPACE}/convertData/validate_region_v2_inspection_inputs.py",
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        python "{params.validator}" combine \
          --manifest-rows {input.manifests} --source-audits {input.audits} \
          --validation-rows {input.validations} --manifest "{output.manifest}" \
          --source-audit "{output.audit}" --validation "{output.validation}"
        """

rule region_v2_inspection_inputs:
    input:
        roots=expand(REGION_V2_INSPECTION_ROOT_DIR + "/{dataset}.root", dataset=REGION_V2_INSPECTION_DATASETS),
        manifest=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_manifest.csv",
        audit=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_source_rows.csv",
        validation=REGION_V2_INSPECTION_META_DIR + f"/{REGION_V2_INSPECTION_VERSION}_validation.csv",
