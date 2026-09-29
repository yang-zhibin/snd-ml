# Slope-selected, single-axis detector-ID transformation. Kept separate from
# the validated v2 two-axis prototype so cached products cannot be confused.
PASSING_MUON_DETID_SHIFT_V3_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detid_channel_shift_v3.yaml"
)
with open(PASSING_MUON_DETID_SHIFT_V3_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_DETID_SHIFT_V3 = yaml.safe_load(stream)

_detid_v3_input = PASSING_MUON_DETID_SHIFT_V3["input"]
_detid_v3_output = PASSING_MUON_DETID_SHIFT_V3["output"]
_detid_v3_display = PASSING_MUON_DETID_SHIFT_V3.get("display", {})
_detid_v3_with_track = int(_detid_v3_display.get("with_track", 0))
_detid_v3_with_hough_track = int(_detid_v3_display.get("with_hough_track", 0))
_detid_v3_minimum_tracks = int(_detid_v3_display.get("minimum_tracks", 0))
_detid_v3_track_verbose = int(_detid_v3_display.get("track_verbose", 0))
_detid_v3_root = f"{PERSONAL_WORK_SPACE}/{_detid_v3_output['directory'].rstrip('/')}"
_detid_v3_event_id = int(_detid_v3_input["event_id"])
_detid_v3_sequence = int(_detid_v3_input["event_sequence"])
PASSING_MUON_DETID_SHIFT_V3_MANIFEST = f"{_detid_v3_root}/{_detid_v3_output['selected_events']}"
PASSING_MUON_DETID_SHIFT_V3_ROOT = (
    f"{_detid_v3_root}/" + _detid_v3_output["transformed_root"].format(eventId=_detid_v3_event_id)
)
PASSING_MUON_DETID_SHIFT_V3_CSV = f"{_detid_v3_root}/{_detid_v3_output['mapping_audit_csv']}"
PASSING_MUON_DETID_SHIFT_V3_JSON = f"{_detid_v3_root}/{_detid_v3_output['event_audit_json']}"
PASSING_MUON_DETID_SHIFT_V3_PDF = f"{_detid_v3_root}/{_detid_v3_output['comparison_pdf']}"
PASSING_MUON_DETID_SHIFT_V3_SCIFI_PDF = f"{_detid_v3_root}/{_detid_v3_output['scifi_channel_pdf']}"
PASSING_MUON_DETID_SHIFT_V3_SCIFI_AUDIT = f"{_detid_v3_root}/{_detid_v3_output['scifi_channel_audit']}"
PASSING_MUON_DETID_SHIFT_V3_MARKER = f"{_detid_v3_root}/{_detid_v3_output['validation_marker']}"


rule select_passing_muon_detid_channel_shift_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        # The reviewed v7 manifest is read through the frozen config path. Do
        # not schedule its producing rule and silently change the event pool.
        selector=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/select_detid_channel_shift_v3.py",
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
    output:
        manifest=PASSING_MUON_DETID_SHIFT_V3_MANIFEST,
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
        /usr/bin/python3 {input.selector} --config {input.config} --output {output.manifest}
        """


rule write_passing_muon_detid_channel_shift_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        manifest=PASSING_MUON_DETID_SHIFT_V3_MANIFEST,
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=PASSING_MUON_DETID_SHIFT_V3_ROOT,
        csv=PASSING_MUON_DETID_SHIFT_V3_CSV,
        json=PASSING_MUON_DETID_SHIFT_V3_JSON,
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
          --event-sequence {_detid_v3_sequence} --output-root {output.root} \
          --output-audit-csv {output.csv} --output-audit-json {output.json}
        """


rule display_passing_muon_detid_channel_shift_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        root=PASSING_MUON_DETID_SHIFT_V3_ROOT,
        audit=PASSING_MUON_DETID_SHIFT_V3_JSON,
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        display_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V3_PDF,
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    params:
        with_track=_detid_v3_with_track,
        with_hough_track=_detid_v3_with_hough_track,
        minimum_tracks=_detid_v3_minimum_tracks,
        track_verbose=_detid_v3_track_verbose,
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.renderer} --audit-json {input.audit} \
          --transformed-root {input.root} --output-pdf {output.pdf} \
          --with-track {params.with_track} --with-hough-track {params.with_hough_track} \
          --minimum-tracks {params.minimum_tracks} --track-verbose {params.track_verbose}
        """


rule display_passing_muon_detid_channel_shift_scifi_channels_v3:
    input:
        root=PASSING_MUON_DETID_SHIFT_V3_ROOT,
        audit=PASSING_MUON_DETID_SHIFT_V3_JSON,
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_scifi_channels.py",
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V3_SCIFI_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V3_SCIFI_AUDIT,
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
        /usr/bin/python3 {input.renderer} --audit-json {input.audit} \
          --transformed-root {input.root} --output-pdf {output.pdf} \
          --output-audit {output.audit}
        """


rule validate_passing_muon_detid_channel_shift_v3:
    input:
        root=PASSING_MUON_DETID_SHIFT_V3_ROOT,
        csv=PASSING_MUON_DETID_SHIFT_V3_CSV,
        json=PASSING_MUON_DETID_SHIFT_V3_JSON,
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=PASSING_MUON_DETID_SHIFT_V3_MARKER,
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


rule passing_muon_detid_channel_shift_v3:
    input:
        PASSING_MUON_DETID_SHIFT_V3_PDF,
        PASSING_MUON_DETID_SHIFT_V3_SCIFI_PDF,
        PASSING_MUON_DETID_SHIFT_V3_SCIFI_AUDIT,
        PASSING_MUON_DETID_SHIFT_V3_MARKER,


# Reusable, isolated per-event display products. Add reviewed event/category
# pairs to input.additional_display_events in the v3 YAML.
_detid_v3_extra_events = {
    str(event_id): category
    for event_id, category in _detid_v3_input.get("additional_display_events", {}).items()
}
_detid_v3_extra_root = f"{_detid_v3_root}/extra_events/run_008329_event_{{event_id}}"


rule select_passing_muon_detid_channel_shift_event_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        selector=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/select_detid_channel_shift_v3.py",
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
    output:
        manifest=f"{_detid_v3_extra_root}/selected_event.csv",
    params:
        category=lambda wildcards: _detid_v3_extra_events[str(wildcards.event_id)],
    wildcard_constraints:
        event_id="|".join(sorted(_detid_v3_extra_events)),
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
        /usr/bin/python3 {input.selector} --config {input.config} \
          --event-id {wildcards.event_id} --source-category {params.category} \
          --output {output.manifest}
        """


rule write_passing_muon_detid_channel_shift_event_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        manifest=f"{_detid_v3_extra_root}/selected_event.csv",
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v3_extra_root}/transformed.root",
        csv=f"{_detid_v3_extra_root}/mapping_audit.csv",
        json=f"{_detid_v3_extra_root}/event_audit.json",
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
          --event-sequence 1 --output-root {output.root} \
          --output-audit-csv {output.csv} --output-audit-json {output.json}
        """


rule display_passing_muon_detid_channel_shift_event_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        root=f"{_detid_v3_extra_root}/transformed.root",
        audit=f"{_detid_v3_extra_root}/event_audit.json",
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        display_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=f"{_detid_v3_extra_root}/comparison.pdf",
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    params:
        with_track=_detid_v3_with_track,
        with_hough_track=_detid_v3_with_hough_track,
        minimum_tracks=_detid_v3_minimum_tracks,
        track_verbose=_detid_v3_track_verbose,
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.renderer} --audit-json {input.audit} \
          --transformed-root {input.root} --output-pdf {output.pdf} \
          --with-track {params.with_track} --with-hough-track {params.with_hough_track} \
          --minimum-tracks {params.minimum_tracks} --track-verbose {params.track_verbose}
        """


rule display_passing_muon_detid_channel_shift_event_scifi_v3:
    input:
        root=f"{_detid_v3_extra_root}/transformed.root",
        audit=f"{_detid_v3_extra_root}/event_audit.json",
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_scifi_channels.py",
    output:
        pdf=f"{_detid_v3_extra_root}/scifi_channels.pdf",
        audit=f"{_detid_v3_extra_root}/scifi_channels_audit.json",
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
        /usr/bin/python3 {input.renderer} --audit-json {input.audit} \
          --transformed-root {input.root} --output-pdf {output.pdf} \
          --output-audit {output.audit}
        """


rule validate_passing_muon_detid_channel_shift_event_v3:
    input:
        root=f"{_detid_v3_extra_root}/transformed.root",
        csv=f"{_detid_v3_extra_root}/mapping_audit.csv",
        json=f"{_detid_v3_extra_root}/event_audit.json",
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v3_extra_root}/validation.ok",
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


rule passing_muon_detid_channel_shift_v3_more_events:
    input:
        expand(f"{_detid_v3_extra_root}/comparison.pdf", event_id=sorted(_detid_v3_extra_events)),
        expand(f"{_detid_v3_extra_root}/scifi_channels.pdf", event_id=sorted(_detid_v3_extra_events)),
        expand(f"{_detid_v3_extra_root}/scifi_channels_audit.json", event_id=sorted(_detid_v3_extra_events)),
        expand(f"{_detid_v3_extra_root}/validation.ok", event_id=sorted(_detid_v3_extra_events)),


# Dynamically selected high-density sample. The checkpoint freezes the ranked
# event/track manifest before per-event transformation jobs are expanded.
import csv

_detid_v3_density_root = f"{_detid_v3_root}/high_density_sample"
_detid_v3_density_event_root = f"{_detid_v3_density_root}/events/event_{{event_id}}"
PASSING_MUON_DETID_SHIFT_V3_DENSITY_MANIFEST = f"{_detid_v3_density_root}/selected_events.csv"
PASSING_MUON_DETID_SHIFT_V3_DENSITY_REJECTED = f"{_detid_v3_density_root}/rejected_events.csv"
PASSING_MUON_DETID_SHIFT_V3_DENSITY_PDF = f"{_detid_v3_density_root}/side_by_side_comparison.pdf"
PASSING_MUON_DETID_SHIFT_V3_DENSITY_AUDIT = f"{_detid_v3_density_root}/side_by_side_audit.json"


checkpoint select_passing_muon_detid_channel_shift_high_density_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        selector=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "select_high_density_detid_shift_v3.py"
        ),
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
        policy=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
    output:
        manifest=PASSING_MUON_DETID_SHIFT_V3_DENSITY_MANIFEST,
        rejected=PASSING_MUON_DETID_SHIFT_V3_DENSITY_REJECTED,
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


def _detid_v3_density_rows(wildcards):
    checkpoint_output = checkpoints.select_passing_muon_detid_channel_shift_high_density_v3.get(
        **wildcards
    ).output
    with open(checkpoint_output.manifest, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _detid_v3_density_manifest(wildcards):
    return checkpoints.select_passing_muon_detid_channel_shift_high_density_v3.get(
        **wildcards
    ).output.manifest


def _detid_v3_density_sequence(wildcards):
    matches = [
        row for row in _detid_v3_density_rows(wildcards)
        if str(row["eventId"]) == str(wildcards.event_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one high-density row for event {wildcards.event_id}")
    return int(matches[0]["event_sequence"])


def _detid_v3_density_products(wildcards, filename):
    return [
        f"{_detid_v3_density_root}/events/event_{row['eventId']}/{filename}"
        for row in _detid_v3_density_rows(wildcards)
    ]


rule write_passing_muon_detid_channel_shift_high_density_event_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        manifest=_detid_v3_density_manifest,
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v3_density_event_root}/transformed.root",
        csv=f"{_detid_v3_density_event_root}/mapping_audit.csv",
        json=f"{_detid_v3_density_event_root}/event_audit.json",
    params:
        sequence=_detid_v3_density_sequence,
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


rule display_passing_muon_detid_channel_shift_high_density_event_v3:
    input:
        config=PASSING_MUON_DETID_SHIFT_V3_CONFIG,
        root=f"{_detid_v3_density_event_root}/transformed.root",
        audit=f"{_detid_v3_density_event_root}/event_audit.json",
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        display_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=f"{_detid_v3_density_event_root}/comparison.pdf",
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=5000, disk_mb=3000, nvidia_gpu=0
    params:
        with_track=_detid_v3_with_track,
        with_hough_track=_detid_v3_with_hough_track,
        minimum_tracks=_detid_v3_minimum_tracks,
        track_verbose=_detid_v3_track_verbose,
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        /usr/bin/python3 {input.renderer} --audit-json {input.audit} \
          --transformed-root {input.root} --output-pdf {output.pdf} \
          --with-track {params.with_track} --with-hough-track {params.with_hough_track} \
          --minimum-tracks {params.minimum_tracks} --track-verbose {params.track_verbose}
        """


rule validate_passing_muon_detid_channel_shift_high_density_event_v3:
    input:
        root=f"{_detid_v3_density_event_root}/transformed.root",
        csv=f"{_detid_v3_density_event_root}/mapping_audit.csv",
        json=f"{_detid_v3_density_event_root}/event_audit.json",
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v3_density_event_root}/validation.ok",
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


rule compose_passing_muon_detid_channel_shift_high_density_v3:
    input:
        pdfs=lambda wildcards: _detid_v3_density_products(wildcards, "comparison.pdf"),
        markers=lambda wildcards: _detid_v3_density_products(wildcards, "validation.ok"),
        composer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "compose_detid_shift_side_by_side.py"
        ),
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V3_DENSITY_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V3_DENSITY_AUDIT,
    params:
        pdf_arguments=lambda wildcards, input: " ".join(
            f"--event-pdf {value}" for value in input.pdfs
        ),
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        /usr/bin/python3 {input.composer} {params.pdf_arguments} \
          --output-pdf {output.pdf} --output-audit {output.audit}
        """


rule passing_muon_detid_channel_shift_v3_high_density:
    input:
        PASSING_MUON_DETID_SHIFT_V3_DENSITY_PDF,
        PASSING_MUON_DETID_SHIFT_V3_DENSITY_AUDIT,
