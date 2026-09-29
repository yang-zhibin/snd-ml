# Proximity-constrained v4 high-density workflow. It is isolated from v3 so
# both direction policies remain reproducible and directly comparable.
import csv


PASSING_MUON_DETID_SHIFT_V4_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detid_channel_shift_v4.yaml"
)
with open(PASSING_MUON_DETID_SHIFT_V4_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_DETID_SHIFT_V4 = yaml.safe_load(stream)

_detid_v4_output = PASSING_MUON_DETID_SHIFT_V4["output"]
_detid_v4_display = PASSING_MUON_DETID_SHIFT_V4.get("display", {})
_detid_v4_root = f"{PERSONAL_WORK_SPACE}/{_detid_v4_output['directory'].rstrip('/')}"
_detid_v4_density_root = f"{_detid_v4_root}/high_density_sample"
_detid_v4_density_event_root = f"{_detid_v4_density_root}/events/event_{{event_id}}"
_detid_v4_with_track = int(_detid_v4_display.get("with_track", 0))
_detid_v4_with_hough_track = int(_detid_v4_display.get("with_hough_track", 0))
_detid_v4_minimum_tracks = int(_detid_v4_display.get("minimum_tracks", 0))
_detid_v4_track_verbose = int(_detid_v4_display.get("track_verbose", 0))

PASSING_MUON_DETID_SHIFT_V4_DENSITY_MANIFEST = f"{_detid_v4_density_root}/selected_events.csv"
PASSING_MUON_DETID_SHIFT_V4_DENSITY_REJECTED = f"{_detid_v4_density_root}/rejected_events.csv"
PASSING_MUON_DETID_SHIFT_V4_DENSITY_PDF = f"{_detid_v4_density_root}/side_by_side_comparison.pdf"
PASSING_MUON_DETID_SHIFT_V4_DENSITY_AUDIT = f"{_detid_v4_density_root}/side_by_side_audit.json"


checkpoint select_passing_muon_detid_channel_shift_high_density_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_CONFIG,
        selector=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "select_high_density_detid_shift_v3.py"
        ),
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
        policy=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
    output:
        manifest=PASSING_MUON_DETID_SHIFT_V4_DENSITY_MANIFEST,
        rejected=PASSING_MUON_DETID_SHIFT_V4_DENSITY_REJECTED,
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


def _detid_v4_density_rows(wildcards):
    checkpoint_output = checkpoints.select_passing_muon_detid_channel_shift_high_density_v4.get(
        **wildcards
    ).output
    with open(checkpoint_output.manifest, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _detid_v4_density_manifest(wildcards):
    return checkpoints.select_passing_muon_detid_channel_shift_high_density_v4.get(
        **wildcards
    ).output.manifest


def _detid_v4_density_sequence(wildcards):
    matches = [
        row for row in _detid_v4_density_rows(wildcards)
        if str(row["eventId"]) == str(wildcards.event_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one v4 high-density row for event {wildcards.event_id}")
    return int(matches[0]["event_sequence"])


def _detid_v4_density_products(wildcards, filename):
    return [
        f"{_detid_v4_density_root}/events/event_{row['eventId']}/{filename}"
        for row in _detid_v4_density_rows(wildcards)
    ]


rule write_passing_muon_detid_channel_shift_high_density_event_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_CONFIG,
        manifest=_detid_v4_density_manifest,
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v4_density_event_root}/transformed.root",
        csv=f"{_detid_v4_density_event_root}/mapping_audit.csv",
        json=f"{_detid_v4_density_event_root}/event_audit.json",
    params:
        sequence=_detid_v4_density_sequence,
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


rule display_passing_muon_detid_channel_shift_high_density_event_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_CONFIG,
        root=f"{_detid_v4_density_event_root}/transformed.root",
        audit=f"{_detid_v4_density_event_root}/event_audit.json",
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        display_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=f"{_detid_v4_density_event_root}/comparison.pdf",
    params:
        with_track=_detid_v4_with_track,
        with_hough_track=_detid_v4_with_hough_track,
        minimum_tracks=_detid_v4_minimum_tracks,
        track_verbose=_detid_v4_track_verbose,
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
          --with-track {params.with_track} --with-hough-track {params.with_hough_track} \
          --minimum-tracks {params.minimum_tracks} --track-verbose {params.track_verbose}
        """


rule validate_passing_muon_detid_channel_shift_high_density_event_v4:
    input:
        root=f"{_detid_v4_density_event_root}/transformed.root",
        csv=f"{_detid_v4_density_event_root}/mapping_audit.csv",
        json=f"{_detid_v4_density_event_root}/event_audit.json",
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v4_density_event_root}/validation.ok",
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


rule compose_passing_muon_detid_channel_shift_high_density_v4:
    input:
        pdfs=lambda wildcards: _detid_v4_density_products(wildcards, "comparison.pdf"),
        markers=lambda wildcards: _detid_v4_density_products(wildcards, "validation.ok"),
        composer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "compose_detid_shift_side_by_side.py"
        ),
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V4_DENSITY_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V4_DENSITY_AUDIT,
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


rule passing_muon_detid_channel_shift_v4_high_density:
    input:
        PASSING_MUON_DETID_SHIFT_V4_DENSITY_PDF,
        PASSING_MUON_DETID_SHIFT_V4_DENSITY_AUDIT,


# Larger density-ranked sample for studying display-time Hough-track failures.
# This has its own config and output tree; it does not replace the reviewed
# ten-event high-density sample above.
PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detid_channel_shift_v4_display100.yaml"
)
with open(PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_DETID_SHIFT_V4_DISPLAY100 = yaml.safe_load(stream)

_detid_v4_d100_output = PASSING_MUON_DETID_SHIFT_V4_DISPLAY100["output"]
_detid_v4_d100_display = PASSING_MUON_DETID_SHIFT_V4_DISPLAY100.get("display", {})
_detid_v4_d100_root = (
    f"{PERSONAL_WORK_SPACE}/{_detid_v4_d100_output['directory'].rstrip('/')}"
)
_detid_v4_d100_event_root = f"{_detid_v4_d100_root}/events/event_{{event_id}}"
_detid_v4_d100_with_track = int(_detid_v4_d100_display.get("with_track", 0))
_detid_v4_d100_with_hough_track = int(
    _detid_v4_d100_display.get("with_hough_track", 0)
)
_detid_v4_d100_minimum_tracks = int(
    _detid_v4_d100_display.get("minimum_tracks", 0)
)
_detid_v4_d100_track_verbose = int(_detid_v4_d100_display.get("track_verbose", 0))

PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_MANIFEST = f"{_detid_v4_d100_root}/selected_events.csv"
PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_REJECTED = f"{_detid_v4_d100_root}/rejected_events.csv"
PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_PDF = f"{_detid_v4_d100_root}/side_by_side_comparison.pdf"
PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_AUDIT = f"{_detid_v4_d100_root}/side_by_side_audit.json"


checkpoint select_passing_muon_detid_channel_shift_display100_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_CONFIG,
        selector=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "select_high_density_detid_shift_v3.py"
        ),
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
        policy=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
    output:
        manifest=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_MANIFEST,
        rejected=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_REJECTED,
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


def _detid_v4_d100_rows(wildcards):
    checkpoint_output = checkpoints.select_passing_muon_detid_channel_shift_display100_v4.get(
        **wildcards
    ).output
    with open(checkpoint_output.manifest, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _detid_v4_d100_manifest(wildcards):
    return checkpoints.select_passing_muon_detid_channel_shift_display100_v4.get(
        **wildcards
    ).output.manifest


def _detid_v4_d100_sequence(wildcards):
    matches = [
        row for row in _detid_v4_d100_rows(wildcards)
        if str(row["eventId"]) == str(wildcards.event_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one v4 display100 row for event {wildcards.event_id}")
    return int(matches[0]["event_sequence"])


def _detid_v4_d100_products(wildcards, filename):
    return [
        f"{_detid_v4_d100_root}/events/event_{row['eventId']}/{filename}"
        for row in _detid_v4_d100_rows(wildcards)
    ]


rule write_passing_muon_detid_channel_shift_display100_event_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_CONFIG,
        manifest=_detid_v4_d100_manifest,
        writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=f"{_detid_v4_d100_event_root}/transformed.root",
        csv=f"{_detid_v4_d100_event_root}/mapping_audit.csv",
        json=f"{_detid_v4_d100_event_root}/event_audit.json",
    params:
        sequence=_detid_v4_d100_sequence,
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


rule display_passing_muon_detid_channel_shift_display100_event_v4:
    input:
        config=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_CONFIG,
        root=f"{_detid_v4_d100_event_root}/transformed.root",
        audit=f"{_detid_v4_d100_event_root}/event_audit.json",
        renderer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/render_detid_channel_shift_comparison.py",
        display_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/sndsw_event_display_common.py",
    output:
        pdf=f"{_detid_v4_d100_event_root}/comparison.pdf",
    params:
        with_track=_detid_v4_d100_with_track,
        with_hough_track=_detid_v4_d100_with_hough_track,
        minimum_tracks=_detid_v4_d100_minimum_tracks,
        track_verbose=_detid_v4_d100_track_verbose,
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
          --with-track {params.with_track} --with-hough-track {params.with_hough_track} \
          --minimum-tracks {params.minimum_tracks} --track-verbose {params.track_verbose}
        """


rule validate_passing_muon_detid_channel_shift_display100_event_v4:
    input:
        root=f"{_detid_v4_d100_event_root}/transformed.root",
        csv=f"{_detid_v4_d100_event_root}/mapping_audit.csv",
        json=f"{_detid_v4_d100_event_root}/event_audit.json",
        validator=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_raw.py",
    output:
        marker=f"{_detid_v4_d100_event_root}/validation.ok",
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


rule compose_passing_muon_detid_channel_shift_display100_v4:
    input:
        pdfs=lambda wildcards: _detid_v4_d100_products(wildcards, "comparison.pdf"),
        markers=lambda wildcards: _detid_v4_d100_products(wildcards, "validation.ok"),
        composer=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "compose_detid_shift_side_by_side.py"
        ),
    output:
        pdf=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_PDF,
        audit=PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_AUDIT,
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


rule passing_muon_detid_channel_shift_v4_display100:
    input:
        PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_PDF,
        PASSING_MUON_DETID_SHIFT_V4_DISPLAY100_AUDIT,
