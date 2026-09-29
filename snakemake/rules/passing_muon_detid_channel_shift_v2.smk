# One-event Run-8329 detector-ID channel-shift validation. This workflow is
# isolated from the rejected continuous-coordinate v1 prototype.
PASSING_MUON_DETID_SHIFT_V2_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detid_channel_shift_v2.yaml"
)
with open(PASSING_MUON_DETID_SHIFT_V2_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_DETID_SHIFT_V2 = yaml.safe_load(stream)

_detid_v2_input = PASSING_MUON_DETID_SHIFT_V2["input"]
_detid_v2_output = PASSING_MUON_DETID_SHIFT_V2["output"]
_detid_v2_root = f"{PERSONAL_WORK_SPACE}/{_detid_v2_output['directory'].rstrip('/')}"
_detid_v2_event_id = int(_detid_v2_input["event_id"])
_detid_v2_sequence = int(_detid_v2_input["event_sequence"])
PASSING_MUON_DETID_SHIFT_V2_MANIFEST = (
    f"{PERSONAL_WORK_SPACE}/{_detid_v2_input['manifest']}"
)
PASSING_MUON_DETID_SHIFT_V2_ROOT = (
    f"{_detid_v2_root}/"
    + _detid_v2_output["transformed_root"].format(eventId=_detid_v2_event_id)
)
PASSING_MUON_DETID_SHIFT_V2_CSV = f"{_detid_v2_root}/{_detid_v2_output['mapping_audit_csv']}"
PASSING_MUON_DETID_SHIFT_V2_JSON = f"{_detid_v2_root}/{_detid_v2_output['event_audit_json']}"
PASSING_MUON_DETID_SHIFT_V2_PDF = f"{_detid_v2_root}/{_detid_v2_output['comparison_pdf']}"
PASSING_MUON_DETID_SHIFT_V2_SCIFI_PDF = f"{_detid_v2_root}/{_detid_v2_output['scifi_channel_pdf']}"
PASSING_MUON_DETID_SHIFT_V2_SCIFI_AUDIT = f"{_detid_v2_root}/{_detid_v2_output['scifi_channel_audit']}"
PASSING_MUON_DETID_SHIFT_V2_MARKER = f"{_detid_v2_root}/{_detid_v2_output['validation_marker']}"


rule write_passing_muon_detid_channel_shift_v2:
    input:
        config=PASSING_MUON_DETID_SHIFT_V2_CONFIG,
        # The reviewed event identities are a frozen external input for v2;
        # never schedule the superseded v1 selector to regenerate them.
        manifest=ancient(PASSING_MUON_DETID_SHIFT_V2_MANIFEST),
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=PASSING_MUON_DETID_SHIFT_V2_ROOT,
        csv=PASSING_MUON_DETID_SHIFT_V2_CSV,
        json=PASSING_MUON_DETID_SHIFT_V2_JSON,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        python {input.writer} \
          --config {input.config} \
          --manifest {input.manifest} \
          --event-sequence {_detid_v2_sequence} \
          --output-root {output.root} \
          --output-audit-csv {output.csv} \
          --output-audit-json {output.json}
        """


rule display_passing_muon_detid_channel_shift_v2:
    input:
        root=PASSING_MUON_DETID_SHIFT_V2_ROOT,
        audit=PASSING_MUON_DETID_SHIFT_V2_JSON,
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V2_PDF,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        python {input.renderer} \
          --audit-json {input.audit} \
          --transformed-root {input.root} \
          --output-pdf {output.pdf}
        """


rule validate_passing_muon_detid_channel_shift_v2:
    input:
        root=PASSING_MUON_DETID_SHIFT_V2_ROOT,
        csv=PASSING_MUON_DETID_SHIFT_V2_CSV,
        json=PASSING_MUON_DETID_SHIFT_V2_JSON,
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=PASSING_MUON_DETID_SHIFT_V2_MARKER,
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=3000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        python {input.validator} \
          --audit-json {input.json} \
          --audit-csv {input.csv} \
          --output-root {input.root} \
          --marker {output.marker}
        """


rule display_passing_muon_detid_channel_shift_scifi_channels_v2:
    input:
        root=PASSING_MUON_DETID_SHIFT_V2_ROOT,
        audit=PASSING_MUON_DETID_SHIFT_V2_JSON,
        renderer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "render_detid_channel_shift_scifi_channels.py"
        ),
        v7_renderer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "scifi_sipm_event_display.py"
        ),
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V2_SCIFI_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V2_SCIFI_AUDIT,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        python {input.renderer} \
          --audit-json {input.audit} \
          --transformed-root {input.root} \
          --output-pdf {output.pdf} \
          --output-audit {output.audit}
        """


rule passing_muon_detid_channel_shift_v2:
    input:
        PASSING_MUON_DETID_SHIFT_V2_PDF,
        PASSING_MUON_DETID_SHIFT_V2_SCIFI_PDF,
        PASSING_MUON_DETID_SHIFT_V2_SCIFI_AUDIT,
        PASSING_MUON_DETID_SHIFT_V2_MARKER,
