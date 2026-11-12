from pathlib import Path
import sys
sys.path.insert(0, str(Path(workflow.basedir).resolve()))
from output_paths import resolve_output

# Three-way v5 comparison: original, accepted frozen-track v4 transform, and
# the reconstruction-free all-active-plane average-position linear fit.
import csv


PASSING_MUON_DETID_SHIFT_V5_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detid_channel_shift_v5.yaml"
)
with open(PASSING_MUON_DETID_SHIFT_V5_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_DETID_SHIFT_V5 = yaml.safe_load(stream)

_detid_v5_output = PASSING_MUON_DETID_SHIFT_V5["output"]
_detid_v5_root = resolve_output(_detid_v5_output['directory'].rstrip('/'), PERSONAL_WORK_SPACE)
_detid_v5_event_root = f"{_detid_v5_root}/events/event_{{event_id}}"
PASSING_MUON_DETID_SHIFT_V5_MANIFEST = f"{_detid_v5_root}/selected_events.csv"
PASSING_MUON_DETID_SHIFT_V5_REJECTED = f"{_detid_v5_root}/rejected_events.csv"
PASSING_MUON_DETID_SHIFT_V5_PDF = f"{_detid_v5_root}/three_way_comparison.pdf"
PASSING_MUON_DETID_SHIFT_V5_AUDIT = f"{_detid_v5_root}/three_way_audit.json"


checkpoint select_passing_muon_detid_channel_shift_v5:
    input:
        # Select the same ten track-valid events with the accepted v4 gates so
        # the two transformation methods are compared on an identical sample.
        config=PASSING_MUON_DETID_SHIFT_V4_CONFIG,
        selector=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "select_high_density_detid_shift_v3.py"
        ),
        track_helper=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
        policy=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/detid_channel_shift.py",
    output:
        manifest=PASSING_MUON_DETID_SHIFT_V5_MANIFEST,
        rejected=PASSING_MUON_DETID_SHIFT_V5_REJECTED,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=4000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.selector} --config {input.config} \
          --output-selected {output.manifest} --output-rejected {output.rejected}
        """


def _detid_v5_rows(wildcards):
    checkpoint_output = checkpoints.select_passing_muon_detid_channel_shift_v5.get(
        **wildcards
    ).output
    with open(checkpoint_output.manifest, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _detid_v5_manifest(wildcards):
    return checkpoints.select_passing_muon_detid_channel_shift_v5.get(
        **wildcards
    ).output.manifest


def _detid_v5_sequence(wildcards):
    matches = [
        row for row in _detid_v5_rows(wildcards)
        if str(row["eventId"]) == str(wildcards.event_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one v5 row for event {wildcards.event_id}")
    return int(matches[0]["event_sequence"])


def _detid_v5_products(wildcards, filename):
    return [
        f"{_detid_v5_root}/events/event_{row['eventId']}/{filename}"
        for row in _detid_v5_rows(wildcards)
    ]


rule write_passing_muon_detid_channel_shift_v5_track_event:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_CONFIG,
        manifest=_detid_v5_manifest,
        writer=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v5_event_root}/track_transform/transformed.root",
        csv=f"{_detid_v5_event_root}/track_transform/mapping_audit.csv",
        json=f"{_detid_v5_event_root}/track_transform/event_audit.json",
    params:
        sequence=_detid_v5_sequence,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.writer} --config {input.config} --manifest {input.manifest} \
          --event-sequence {params.sequence} --output-root {output.root} \
          --output-audit-csv {output.csv} --output-audit-json {output.json}
        """


rule write_passing_muon_detid_channel_shift_v5_simple_event:
    input:
        config=PASSING_MUON_DETID_SHIFT_V5_CONFIG,
        manifest=_detid_v5_manifest,
        writer=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v5_event_root}/simple_transform/transformed.root",
        csv=f"{_detid_v5_event_root}/simple_transform/mapping_audit.csv",
        json=f"{_detid_v5_event_root}/simple_transform/event_audit.json",
    params:
        sequence=_detid_v5_sequence,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.writer} --config {input.config} --manifest {input.manifest} \
          --event-sequence {params.sequence} --output-root {output.root} \
          --output-audit-csv {output.csv} --output-audit-json {output.json}
        """


rule validate_passing_muon_detid_channel_shift_v5_track_event:
    input:
        root=f"{_detid_v5_event_root}/track_transform/transformed.root",
        csv=f"{_detid_v5_event_root}/track_transform/mapping_audit.csv",
        json=f"{_detid_v5_event_root}/track_transform/event_audit.json",
        validator=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v5_event_root}/track_transform/validation.ok",
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=3000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.validator} --audit-json {input.json} --audit-csv {input.csv} \
          --output-root {input.root} --marker {output.marker}
        """


rule validate_passing_muon_detid_channel_shift_v5_simple_event:
    input:
        root=f"{_detid_v5_event_root}/simple_transform/transformed.root",
        csv=f"{_detid_v5_event_root}/simple_transform/mapping_audit.csv",
        json=f"{_detid_v5_event_root}/simple_transform/event_audit.json",
        validator=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v5_event_root}/simple_transform/validation.ok",
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=3000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.validator} --audit-json {input.json} --audit-csv {input.csv} \
          --output-root {input.root} --marker {output.marker}
        """


rule display_passing_muon_detid_channel_shift_v5_event:
    input:
        track_root=f"{_detid_v5_event_root}/track_transform/transformed.root",
        track_json=f"{_detid_v5_event_root}/track_transform/event_audit.json",
        simple_root=f"{_detid_v5_event_root}/simple_transform/transformed.root",
        simple_json=f"{_detid_v5_event_root}/simple_transform/event_audit.json",
        renderer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "render_detid_channel_shift_v5_comparison.py"
        ),
        display_helper=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=f"{_detid_v5_event_root}/comparison_three_way.pdf",
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.renderer} --track-audit-json {input.track_json} \
          --track-root {input.track_root} --simple-audit-json {input.simple_json} \
          --simple-root {input.simple_root} --output-pdf {output.pdf}
        """


rule compose_passing_muon_detid_channel_shift_v5:
    input:
        pdfs=lambda wildcards: _detid_v5_products(wildcards, "comparison_three_way.pdf"),
        track_markers=lambda wildcards: _detid_v5_products(
            wildcards, "track_transform/validation.ok"
        ),
        simple_markers=lambda wildcards: _detid_v5_products(
            wildcards, "simple_transform/validation.ok"
        ),
        composer=(
            f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/"
            "compose_detid_shift_three_way.py"
        ),
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V5_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V5_AUDIT,
    params:
        pdf_arguments=lambda wildcards, input: " ".join(
            f"--event-pdf {value}" for value in input.pdfs
        ),
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=5000, disk_mb=10000, nvidia_gpu=0
    shell:
        r"""
        /usr/bin/python3 {input.composer} {params.pdf_arguments} \
          --output-pdf {output.pdf} --output-audit {output.audit}
        """


rule passing_muon_detid_channel_shift_v5:
    input:
        PASSING_MUON_DETID_SHIFT_V5_PDF,
        PASSING_MUON_DETID_SHIFT_V5_AUDIT,
