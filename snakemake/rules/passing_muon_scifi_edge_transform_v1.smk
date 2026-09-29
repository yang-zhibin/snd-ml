# Isolated Run-8329 SciFi edge-transform validation.  This workflow consumes
# frozen V7 identities and track products but never modifies them.
PASSING_MUON_SCIFI_EDGE_V1_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_scifi_edge_transform_v1.yaml"
)
with open(PASSING_MUON_SCIFI_EDGE_V1_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_SCIFI_EDGE_V1 = yaml.safe_load(stream)

_pm_edge_input = PASSING_MUON_SCIFI_EDGE_V1["input"]
_pm_edge_output = PASSING_MUON_SCIFI_EDGE_V1["output"]
_pm_edge_root = (
    f"{PERSONAL_WORK_SPACE}/{_pm_edge_output['directory'].rstrip('/')}"
)
PASSING_MUON_SCIFI_EDGE_V1_SELECTED = (
    f"{_pm_edge_root}/{_pm_edge_output['selected_events']}"
)
PASSING_MUON_SCIFI_EDGE_V1_REJECTED = (
    f"{_pm_edge_root}/{_pm_edge_output['rejected_events']}"
)
PASSING_MUON_SCIFI_EDGE_V1_PDF = (
    f"{_pm_edge_root}/{_pm_edge_output['comparison_pdf']}"
)
PASSING_MUON_SCIFI_EDGE_V1_PAGES = (
    f"{_pm_edge_root}/{_pm_edge_output['rendered_pages']}"
)
PASSING_MUON_SCIFI_EDGE_V1_AUDIT_CSV = (
    f"{_pm_edge_root}/{_pm_edge_output['audit_csv']}"
)
PASSING_MUON_SCIFI_EDGE_V1_AUDIT_JSON = (
    f"{_pm_edge_root}/{_pm_edge_output['audit_json']}"
)
PASSING_MUON_SCIFI_EDGE_V1_MARKER = (
    f"{_pm_edge_root}/{_pm_edge_output['validation_marker']}"
)


rule select_passing_muon_scifi_edge_transform_events_v1:
    input:
        config=PASSING_MUON_SCIFI_EDGE_V1_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "select_passing_muon_scifi_edge_transform_events.py"
        ),
        adapter=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_edge_transform_sndsw.py"
        ),
        transform=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_edge_transform.py"
        ),
    output:
        selected=PASSING_MUON_SCIFI_EDGE_V1_SELECTED,
        rejected=PASSING_MUON_SCIFI_EDGE_V1_REJECTED,
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/
        python {input.script} \
          --config {input.config} \
          --output-selected {output.selected} \
          --output-rejected {output.rejected}
        """


rule plot_passing_muon_scifi_edge_transform_events_v1:
    input:
        config=PASSING_MUON_SCIFI_EDGE_V1_CONFIG,
        manifest=PASSING_MUON_SCIFI_EDGE_V1_SELECTED,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_scifi_edge_transform.py"
        ),
        adapter=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_edge_transform_sndsw.py"
        ),
        transform=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_edge_transform.py"
        ),
        scifi_renderer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_sipm_event_display.py"
        ),
        sndsw_renderer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "sndsw_event_display_common.py"
        ),
        transformed_sndsw_renderer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "render_sndsw_edge_transform_comparison.py"
        ),
    output:
        pdf=PASSING_MUON_SCIFI_EDGE_V1_PDF,
        pages=PASSING_MUON_SCIFI_EDGE_V1_PAGES,
        audit_csv=PASSING_MUON_SCIFI_EDGE_V1_AUDIT_CSV,
        audit_json=PASSING_MUON_SCIFI_EDGE_V1_AUDIT_JSON,
    threads: 1
    resources:
        runtime=6 * 60 * 60, mem_mb=6000, disk_mb=8000, nvidia_gpu=0
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
          --output-pdf {output.pdf} \
          --output-pages {output.pages} \
          --output-audit-csv {output.audit_csv} \
          --output-audit-json {output.audit_json}
        """


rule validate_passing_muon_scifi_edge_transform_v1:
    input:
        config=PASSING_MUON_SCIFI_EDGE_V1_CONFIG,
        selected=PASSING_MUON_SCIFI_EDGE_V1_SELECTED,
        pdf=PASSING_MUON_SCIFI_EDGE_V1_PDF,
        pages=PASSING_MUON_SCIFI_EDGE_V1_PAGES,
        audit_csv=PASSING_MUON_SCIFI_EDGE_V1_AUDIT_CSV,
        audit_json=PASSING_MUON_SCIFI_EDGE_V1_AUDIT_JSON,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "validate_passing_muon_scifi_edge_transform.py"
        ),
    output:
        marker=PASSING_MUON_SCIFI_EDGE_V1_MARKER,
    threads: 1
    resources:
        runtime=10 * 60, mem_mb=1000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        python3 {input.script} \
          --config {input.config} \
          --selected {input.selected} \
          --pdf {input.pdf} \
          --rendered-pages {input.pages} \
          --audit-csv {input.audit_csv} \
          --audit-json {input.audit_json} \
          --output {output.marker}
        """


rule passing_muon_scifi_edge_transform_v1:
    input:
        PASSING_MUON_SCIFI_EDGE_V1_MARKER,
