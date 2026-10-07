import shlex
import sys
import re


UNIFIED_REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
UNIFIED_CONFIG = (
    f"{UNIFIED_REPO_ROOT}/snakemake/metadata/configs/"
    "unified_shower_features_v1.yaml"
)
UNIFIED_METADATA_MODULE_DIR = f"{UNIFIED_REPO_ROOT}/snakemake/metadata"
if UNIFIED_METADATA_MODULE_DIR not in sys.path:
    sys.path.insert(0, UNIFIED_METADATA_MODULE_DIR)

from prepare_unified_shower_feature_manifests import (  # noqa: E402
    FAMILIES as UNIFIED_FAMILIES,
    build_manifests as _build_unified_manifests,
    load_config as _load_unified_config,
    write_manifest as _write_unified_manifest,
)


UNIFIED_SETTINGS = _load_unified_config(UNIFIED_CONFIG)
UNIFIED_MANIFESTS = _build_unified_manifests(UNIFIED_CONFIG)
UNIFIED_MANIFEST_ROOT = str(UNIFIED_SETTINGS["manifest_root"]).rstrip("/")
UNIFIED_FEATURE_ROOT = (
    "/eos/experiment/sndlhc/users/zhibin/snd-ml/"
    "unified_shower_features_v1/features"
)
for _family in UNIFIED_FAMILIES:
    _configured_root = str(UNIFIED_SETTINGS["feature_roots"][_family]).rstrip("/")
    _expected_root = f"{UNIFIED_FEATURE_ROOT}/{_family}"
    if _configured_root != _expected_root:
        raise ValueError(
            f"Unified feature-root mismatch for {_family}: "
            f"{_configured_root} != {_expected_root}"
        )
UNIFIED_RECORDS = {
    (family, str(record["source_uid"])): record
    for family, frame in UNIFIED_MANIFESTS.items()
    for record in frame.to_dict("records")
}
UNIFIED_FAMILY_PATTERN = "|".join(UNIFIED_FAMILIES)
UNIFIED_SHARD_SETTINGS = UNIFIED_SETTINGS["sharding"]
UNIFIED_SHARD_ROOT = str(UNIFIED_SHARD_SETTINGS["root"]).rstrip("/")
UNIFIED_SHARD_FAMILIES = tuple(
    str(value) for value in UNIFIED_SHARD_SETTINGS["enabled_families"]
)
UNIFIED_SHARD_THRESHOLD = int(
    UNIFIED_SHARD_SETTINGS["threshold_input_events"]
)
UNIFIED_SHARD_SIZES = {
    str(family): int(size)
    for family, size in UNIFIED_SHARD_SETTINGS[
        "selected_events_per_shard"
    ].items()
}
if UNIFIED_SHARD_THRESHOLD < 0:
    raise ValueError("Unified feature sharding sizes must be positive")
for _family in UNIFIED_SHARD_FAMILIES:
    if _family not in UNIFIED_FAMILIES:
        raise ValueError(f"Unknown unified sharding family {_family}")
    _mode = str(UNIFIED_SETTINGS["families"][_family]["selection_mode"])
    if _mode not in {"direct_input", "identity_cutflow"}:
        raise ValueError(
            f"Unified sharding requires identity/direct selection, got {_mode}"
        )
    if _family not in UNIFIED_SHARD_SIZES or UNIFIED_SHARD_SIZES[_family] <= 0:
        raise ValueError(f"Missing positive shard size for {_family}")


def _unified_shard_ranges_for_record(family, record):
    event_count = int(record["input_event_count"])
    if family not in UNIFIED_SHARD_FAMILIES or event_count <= UNIFIED_SHARD_THRESHOLD:
        return []
    shard_size = UNIFIED_SHARD_SIZES[family]
    return [
        (index, start, min(start + shard_size, event_count))
        for index, start in enumerate(range(0, event_count, shard_size))
    ]


UNIFIED_SHARD_RANGES = {
    (family, str(record["source_uid"])): _unified_shard_ranges_for_record(
        family, record
    )
    for family, frame in UNIFIED_MANIFESTS.items()
    for record in frame.to_dict("records")
    if _unified_shard_ranges_for_record(family, record)
}
UNIFIED_SHARDED_SOURCE_UID_PATTERN = "(?:" + "|".join(
    re.escape(source_uid)
    for _, source_uid in sorted(UNIFIED_SHARD_RANGES)
) + ")"


def _unified_manifest_path(family):
    return f"{UNIFIED_MANIFEST_ROOT}/{family}.csv"


def _unified_feature_path(family, source_uid):
    return (
        f"{UNIFIED_FEATURE_ROOT}/{family}/{source_uid}/"
        f"feature_scifi_cluster_v1_{source_uid}.root"
    )


def _unified_shard_path(family, source_uid, shard_index, suffix=""):
    shard = f"{int(shard_index):04d}"
    return (
        f"{UNIFIED_SHARD_ROOT}/{family}/{source_uid}/part_{shard}/"
        f"feature_scifi_cluster_v1_{source_uid}_part_{shard}.root{suffix}"
    )


def _unified_shard_range(wildcards):
    key = (str(wildcards.family), str(wildcards.source_uid))
    shard_index = int(str(wildcards.shard))
    ranges = UNIFIED_SHARD_RANGES.get(key, [])
    if shard_index < 0 or shard_index >= len(ranges):
        raise ValueError(f"Unknown unified feature shard: {key} part {shard_index}")
    return ranges[shard_index]


def _unified_shard_targets(wildcards, suffix=""):
    key = (str(wildcards.family), str(wildcards.source_uid))
    ranges = UNIFIED_SHARD_RANGES.get(key)
    if not ranges:
        raise ValueError(f"Source is not configured for sharding: {key}")
    return [
        _unified_shard_path(*key, shard_index, suffix)
        for shard_index, _, _ in ranges
    ]


def _unified_record(wildcards):
    key = (str(wildcards.family), str(wildcards.source_uid))
    if key not in UNIFIED_RECORDS:
        raise ValueError(f"Unknown unified feature source: {key}")
    record = UNIFIED_RECORDS[key]
    expected = _unified_feature_path(*key)
    if str(record["feature_scifi_cluster_v1_path"]) != expected:
        raise ValueError(
            f"Unified output mismatch for {key}: "
            f"{record['feature_scifi_cluster_v1_path']} != {expected}"
        )
    return record


def _unified_value(wildcards, column):
    return str(_unified_record(wildcards).get(column, ""))


def _unified_converter(wildcards):
    if str(wildcards.family) == "mc_muon_up":
        return f"{UNIFIED_REPO_ROOT}/convertData/digi_2_features_mc_muon.py"
    return f"{UNIFIED_REPO_ROOT}/convertData/digi_2_features.py"


def _unified_cli(wildcards):
    record = _unified_record(wildcards)
    arguments = []
    cutflow = str(record.get("cutflow_path", ""))
    if cutflow:
        arguments.extend(["-p", cutflow])
    arguments.extend([
        "-d", str(record["digi_path"]),
        "-g", str(record["geo_path"]),
        "-t", str(record["sample_type"]),
        "--tree-name", str(record["input_tree"]),
        "--selection-mode", str(record["selection_mode"]),
        "--source-domain-id", str(int(record["source_domain_id"])),
        "--source-uid", str(record["source_uid"]),
    ])
    pdg = record.get("source_pdg_code", "")
    energy = record.get("source_energy_gev", "")
    if str(pdg) != "" or str(energy) != "":
        if str(pdg) == "" or str(energy) == "":
            raise ValueError("Source PDG and energy must be supplied together")
        arguments.extend([
            "--source-pdg-code", str(int(float(pdg))),
            "--source-energy-gev", str(float(energy)),
        ])
    geometry_id = str(record.get("source_geometry_id", ""))
    geometry_checksum = str(record.get("source_geometry_checksum", ""))
    if geometry_id:
        arguments.extend(["--source-geometry-id", geometry_id])
    if geometry_checksum:
        arguments.extend(["--source-geometry-checksum", geometry_checksum])
    return " ".join(shlex.quote(value) for value in arguments)


def _unified_shard_cli(wildcards):
    _, start, stop = _unified_shard_range(wildcards)
    return (
        f"{_unified_cli(wildcards)} "
        f"--selected-entry-start {start} --selected-entry-stop {stop}"
    )


def _unified_shard_record_arguments(wildcards):
    return " ".join(
        f"--shard-record {shlex.quote(path)}"
        for path in _unified_shard_targets(wildcards, ".validated.json")
    )


def _unified_targets(family, canary=False):
    frame = UNIFIED_MANIFESTS[family]
    if canary:
        frame = frame.loc[frame["is_canary"].astype(int).eq(1)]
        if len(frame) != 1:
            raise ValueError(f"Expected one {family} canary row, found {len(frame)}")
    return [str(path) for path in frame["validation_record"].astype(str)]


def _unified_canary_record(family):
    frame = UNIFIED_MANIFESTS[str(family)]
    selected = frame.loc[frame["is_canary"].astype(int).eq(1)]
    if len(selected) != 1:
        raise ValueError(f"Expected one {family} canary row, found {len(selected)}")
    return selected.iloc[0].to_dict()


def _unified_local_smoke_value(wildcards, column):
    return str(_unified_canary_record(str(wildcards.family)).get(column, ""))


def _unified_local_smoke_cli(wildcards):
    record = _unified_canary_record(str(wildcards.family))
    class SmokeWildcards:
        family = str(wildcards.family)
        source_uid = str(record["source_uid"])
    limit = int(UNIFIED_SETTINGS["local_smoke"]["max_events"][str(wildcards.family)])
    return f"{_unified_cli(SmokeWildcards())} --max-events {limit}"


def _unified_local_smoke_path(family, suffix):
    root = str(UNIFIED_SETTINGS["local_smoke"]["output_root"]).rstrip("/")
    return f"{root}/{family}/feature_scifi_cluster_v1_local_smoke.root{suffix}"


localrules: unified_prepare_feature_manifest_v1


rule unified_prepare_feature_manifest_v1:
    input:
        config=UNIFIED_CONFIG,
        script=(
            f"{UNIFIED_REPO_ROOT}/snakemake/metadata/"
            "prepare_unified_shower_feature_manifests.py"
        ),
    output:
        f"{UNIFIED_MANIFEST_ROOT}/{{family}}.csv"
    wildcard_constraints:
        family=UNIFIED_FAMILY_PATTERN,
    threads: 1
    resources:
        runtime=5 * 60,
        mem_mb=1000,
        disk_mb=100,
        nvidia_gpu=0,
    run:
        _write_unified_manifest(UNIFIED_MANIFESTS[str(wildcards.family)], output[0])


rule unified_feature_manifests_v1:
    input:
        [_unified_manifest_path(family) for family in UNIFIED_FAMILIES]


ruleorder: unified_merge_shower_feature_shards_v1 > unified_shower_feature_v1 > process_features
ruleorder: unified_shower_feature_shard_v1 > process_features


rule unified_shower_feature_shard_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(str(wc.family)),
        digi=lambda wc: _unified_value(wc, "digi_path"),
        geo=lambda wc: _unified_value(wc, "geo_path"),
        converter=_unified_converter,
        feature_core=f"{UNIFIED_REPO_ROOT}/convertData/digi_2_features.py",
        selection=f"{UNIFIED_REPO_ROOT}/convertData/sample_selection.py",
        provenance=f"{UNIFIED_REPO_ROOT}/convertData/source_provenance.py",
        clustering=f"{UNIFIED_REPO_ROOT}/convertData/scifi_plane_clustering.py",
        validator=f"{UNIFIED_REPO_ROOT}/convertData/validate_unified_shower_feature_file.py",
        config=UNIFIED_CONFIG,
    output:
        feature=(
            f"{UNIFIED_SHARD_ROOT}/{{family}}/{{source_uid}}/part_{{shard}}/"
            "feature_scifi_cluster_v1_{source_uid}_part_{shard}.root"
        ),
        marker=(
            f"{UNIFIED_SHARD_ROOT}/{{family}}/{{source_uid}}/part_{{shard}}/"
            "feature_scifi_cluster_v1_{source_uid}_part_{shard}.root.validated.json"
        ),
    params:
        cli=_unified_shard_cli,
        cutflow=lambda wc: _unified_value(wc, "cutflow_path"),
        source_uid=lambda wc: _unified_value(wc, "source_uid"),
        source_domain_id=lambda wc: int(_unified_value(wc, "source_domain_id")),
        geometry_id=lambda wc: _unified_value(wc, "source_geometry_id"),
        geometry_checksum=lambda wc: _unified_value(wc, "source_geometry_checksum"),
        input_event_count=lambda wc: int(_unified_value(wc, "input_event_count")),
        sample_type=lambda wc: _unified_value(wc, "sample_type"),
        selection_mode=lambda wc: _unified_value(wc, "selection_mode"),
        input_tree=lambda wc: _unified_value(wc, "input_tree"),
        selected_start=lambda wc: _unified_shard_range(wc)[1],
        selected_stop=lambda wc: _unified_shard_range(wc)[2],
        manifest_version=str(UNIFIED_SETTINGS["version"]),
    wildcard_constraints:
        family="|".join(UNIFIED_SHARD_FAMILIES),
        source_uid=UNIFIED_SHARDED_SOURCE_UID_PATTERN,
        shard=r"[0-9]{4}",
    threads: 1
    resources:
        runtime=2 * 60 * 60,
        mem_mb=2500,
        disk_mb=3000,
        nvidia_gpu=0,
        eos_feature_read=1,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{UNIFIED_REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'

        if [[ -n "{params.cutflow}" && ! -f "{params.cutflow}" ]]; then
            echo "ERROR: frozen cutflow input does not exist: {params.cutflow}" >&2
            exit 1
        fi

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_shower_shard.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_marker="${{tmp_dir}}/validated.json"

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            {params.cli} -o "$tmp_feature"
        converter_commit=$(git -C {UNIFIED_REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_feature" \
            --manifest-version "{params.manifest_version}" \
            --expected-source-uid "{params.source_uid}" \
            --expected-source-domain-id "{params.source_domain_id}" \
            --expected-source-geometry-id "{params.geometry_id}" \
            --expected-source-geometry-checksum "{params.geometry_checksum}" \
            --expected-input-event-count "{params.input_event_count}" \
            --sample-type "{params.sample_type}" \
            --selection-mode "{params.selection_mode}" \
            --input-tree "{params.input_tree}" \
            --expected-max-events 0 \
            --expected-selected-entry-start "{params.selected_start}" \
            --expected-selected-entry-stop "{params.selected_stop}" \
            --input-digi-path "{input.digi}" \
            --input-cutflow-path "{params.cutflow}" \
            --published-feature-path "{output.feature}" \
            --converter-commit "$converter_commit" \
            --validation-record "$tmp_marker"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        test -r "{output.feature}"
        local_size=$(stat -c '%s' "$tmp_feature")
        published_size=$(stat -c '%s' "{output.feature}")
        if [[ "$local_size" != "$published_size" ]]; then
            echo "ERROR: published shard size mismatch" >&2
            exit 1
        fi
        xrdcp -f "$tmp_marker" "{output.marker}"
        test -r "{output.marker}"
        """


rule unified_merge_shower_feature_shards_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(str(wc.family)),
        digi=lambda wc: _unified_value(wc, "digi_path"),
        geo=lambda wc: _unified_value(wc, "geo_path"),
        shard_features=lambda wc: _unified_shard_targets(wc),
        shard_records=lambda wc: _unified_shard_targets(wc, ".validated.json"),
        merger=f"{UNIFIED_REPO_ROOT}/convertData/merge_unified_shower_feature_shards.py",
        validator=f"{UNIFIED_REPO_ROOT}/convertData/validate_unified_shower_feature_file.py",
        selection=f"{UNIFIED_REPO_ROOT}/convertData/sample_selection.py",
        config=UNIFIED_CONFIG,
    output:
        feature=(
            f"{UNIFIED_FEATURE_ROOT}/{{family}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root"
        ),
        marker=(
            f"{UNIFIED_FEATURE_ROOT}/{{family}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root.validated.json"
        ),
    params:
        shard_record_args=_unified_shard_record_arguments,
        cutflow=lambda wc: _unified_value(wc, "cutflow_path"),
        source_uid=lambda wc: _unified_value(wc, "source_uid"),
        source_domain_id=lambda wc: int(_unified_value(wc, "source_domain_id")),
        geometry_id=lambda wc: _unified_value(wc, "source_geometry_id"),
        geometry_checksum=lambda wc: _unified_value(wc, "source_geometry_checksum"),
        input_event_count=lambda wc: int(_unified_value(wc, "input_event_count")),
        sample_type=lambda wc: _unified_value(wc, "sample_type"),
        selection_mode=lambda wc: _unified_value(wc, "selection_mode"),
        input_tree=lambda wc: _unified_value(wc, "input_tree"),
        manifest_version=str(UNIFIED_SETTINGS["version"]),
    wildcard_constraints:
        family="|".join(UNIFIED_SHARD_FAMILIES),
        source_uid=UNIFIED_SHARDED_SOURCE_UID_PATTERN,
    threads: 1
    resources:
        runtime=4 * 60 * 60,
        mem_mb=4000,
        disk_mb=20000,
        nvidia_gpu=0,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{UNIFIED_REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_shower_merge.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_merge_record="${{tmp_dir}}/merge.json"
        tmp_marker="${{tmp_dir}}/validated.json"

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.merger} \
            {params.shard_record_args} \
            --output "$tmp_feature" --merge-record "$tmp_merge_record"
        converter_commit=$(git -C {UNIFIED_REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_feature" \
            --manifest-version "{params.manifest_version}" \
            --expected-source-uid "{params.source_uid}" \
            --expected-source-domain-id "{params.source_domain_id}" \
            --expected-source-geometry-id "{params.geometry_id}" \
            --expected-source-geometry-checksum "{params.geometry_checksum}" \
            --expected-input-event-count "{params.input_event_count}" \
            --sample-type "{params.sample_type}" \
            --selection-mode "{params.selection_mode}" \
            --input-tree "{params.input_tree}" \
            --expected-max-events 0 \
            --input-digi-path "{input.digi}" \
            --input-cutflow-path "{params.cutflow}" \
            --published-feature-path "{output.feature}" \
            --merge-record "$tmp_merge_record" \
            --converter-commit "$converter_commit" \
            --validation-record "$tmp_marker"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        test -r "{output.feature}"
        local_size=$(stat -c '%s' "$tmp_feature")
        published_size=$(stat -c '%s' "{output.feature}")
        if [[ "$local_size" != "$published_size" ]]; then
            echo "ERROR: published merged feature size mismatch" >&2
            exit 1
        fi
        xrdcp -f "$tmp_marker" "{output.marker}"
        test -r "{output.marker}"
        """


rule unified_shower_feature_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(str(wc.family)),
        digi=lambda wc: _unified_value(wc, "digi_path"),
        geo=lambda wc: _unified_value(wc, "geo_path"),
        converter=_unified_converter,
        feature_core=f"{UNIFIED_REPO_ROOT}/convertData/digi_2_features.py",
        selection=f"{UNIFIED_REPO_ROOT}/convertData/sample_selection.py",
        provenance=f"{UNIFIED_REPO_ROOT}/convertData/source_provenance.py",
        clustering=f"{UNIFIED_REPO_ROOT}/convertData/scifi_plane_clustering.py",
        validator=f"{UNIFIED_REPO_ROOT}/convertData/validate_unified_shower_feature_file.py",
        config=UNIFIED_CONFIG,
    output:
        feature=(
            f"{UNIFIED_FEATURE_ROOT}/{{family}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root"
        ),
        marker=(
            f"{UNIFIED_FEATURE_ROOT}/{{family}}/{{source_uid}}/"
            "feature_scifi_cluster_v1_{source_uid}.root.validated.json"
        ),
    params:
        cli=_unified_cli,
        # Frozen, pre-existing cutflows are runtime inputs. Keeping the path
        # outside ``input`` prevents the legacy catch-all cutflow producer from
        # claiming it against whichever metadata family the main workflow has
        # loaded. The shell preflight still requires it.
        cutflow=lambda wc: _unified_value(wc, "cutflow_path"),
        source_uid=lambda wc: _unified_value(wc, "source_uid"),
        source_domain_id=lambda wc: int(_unified_value(wc, "source_domain_id")),
        geometry_id=lambda wc: _unified_value(wc, "source_geometry_id"),
        geometry_checksum=lambda wc: _unified_value(wc, "source_geometry_checksum"),
        input_event_count=lambda wc: int(_unified_value(wc, "input_event_count")),
        sample_type=lambda wc: _unified_value(wc, "sample_type"),
        selection_mode=lambda wc: _unified_value(wc, "selection_mode"),
        input_tree=lambda wc: _unified_value(wc, "input_tree"),
        max_events=0,
        manifest_version=str(UNIFIED_SETTINGS["version"]),
    wildcard_constraints:
        family=UNIFIED_FAMILY_PATTERN,
        source_uid=r"(?:unified_src|tbfull_src)_[0-9a-f]{20}",
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2500,
        disk_mb=3000,
        nvidia_gpu=0,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{UNIFIED_REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'

        if [[ -n "{params.cutflow}" && ! -f "{params.cutflow}" ]]; then
            echo "ERROR: frozen cutflow input does not exist: {params.cutflow}" >&2
            exit 1
        fi

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_shower_feature.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_marker="${{tmp_dir}}/validated.json"

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            {params.cli} -o "$tmp_feature"
        converter_commit=$(git -C {UNIFIED_REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_feature" \
            --manifest-version "{params.manifest_version}" \
            --expected-source-uid "{params.source_uid}" \
            --expected-source-domain-id "{params.source_domain_id}" \
            --expected-source-geometry-id "{params.geometry_id}" \
            --expected-source-geometry-checksum "{params.geometry_checksum}" \
            --expected-input-event-count "{params.input_event_count}" \
            --sample-type "{params.sample_type}" \
            --selection-mode "{params.selection_mode}" \
            --input-tree "{params.input_tree}" \
            --expected-max-events "{params.max_events}" \
            --input-digi-path "{input.digi}" \
            --input-cutflow-path "{params.cutflow}" \
            --published-feature-path "{output.feature}" \
            --converter-commit "$converter_commit" \
            --validation-record "$tmp_marker"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        test -r "{output.feature}"
        local_size=$(stat -c '%s' "$tmp_feature")
        published_size=$(stat -c '%s' "{output.feature}")
        if [[ "$local_size" != "$published_size" ]]; then
            echo "ERROR: published feature size mismatch" >&2
            exit 1
        fi
        xrdcp -f "$tmp_marker" "{output.marker}"
        test -r "{output.marker}"
        """


rule unified_feature_local_smoke_v1:
    input:
        manifest=lambda wc: _unified_manifest_path(str(wc.family)),
        digi=lambda wc: _unified_local_smoke_value(wc, "digi_path"),
        geo=lambda wc: _unified_local_smoke_value(wc, "geo_path"),
        converter=lambda wc: _unified_converter(wc),
        feature_core=f"{UNIFIED_REPO_ROOT}/convertData/digi_2_features.py",
        selection=f"{UNIFIED_REPO_ROOT}/convertData/sample_selection.py",
        provenance=f"{UNIFIED_REPO_ROOT}/convertData/source_provenance.py",
        clustering=f"{UNIFIED_REPO_ROOT}/convertData/scifi_plane_clustering.py",
        validator=f"{UNIFIED_REPO_ROOT}/convertData/validate_unified_shower_feature_file.py",
        config=UNIFIED_CONFIG,
    output:
        feature=(
            str(UNIFIED_SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/{family}/feature_scifi_cluster_v1_local_smoke.root"
        ),
        marker=(
            str(UNIFIED_SETTINGS["local_smoke"]["output_root"]).rstrip("/")
            + "/{family}/feature_scifi_cluster_v1_local_smoke.root.validated.json"
        ),
    params:
        cli=_unified_local_smoke_cli,
        cutflow=lambda wc: _unified_local_smoke_value(wc, "cutflow_path"),
        source_uid=lambda wc: _unified_local_smoke_value(wc, "source_uid"),
        source_domain_id=lambda wc: int(_unified_local_smoke_value(wc, "source_domain_id")),
        input_event_count=lambda wc: int(
            _unified_local_smoke_value(wc, "input_event_count")
        ),
        sample_type=lambda wc: _unified_local_smoke_value(wc, "sample_type"),
        selection_mode=lambda wc: _unified_local_smoke_value(wc, "selection_mode"),
        input_tree=lambda wc: _unified_local_smoke_value(wc, "input_tree"),
        max_events=lambda wc: int(
            UNIFIED_SETTINGS["local_smoke"]["max_events"][str(wc.family)]
        ),
        manifest_version=str(UNIFIED_SETTINGS["version"]),
    wildcard_constraints:
        family="mc_neutrino|mc_muon_up",
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=2500,
        disk_mb=1000,
        nvidia_gpu=0,
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {UNIFIED_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:{UNIFIED_REPO_ROOT}/convertData:${{PYTHONPATH:-}}"
        set -euo pipefail
        hostname
        pwd
        command -v python3
        python3 --version
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/unified_local_smoke.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            {params.cli} -o "$tmp_dir/feature.root"
        converter_commit=$(git -C {UNIFIED_REPO_ROOT} rev-parse HEAD)
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_dir/feature.root" \
            --manifest-version "{params.manifest_version}" \
            --expected-source-uid "{params.source_uid}" \
            --expected-source-domain-id "{params.source_domain_id}" \
            --expected-input-event-count "{params.input_event_count}" \
            --sample-type "{params.sample_type}" \
            --selection-mode "{params.selection_mode}" \
            --input-tree "{params.input_tree}" \
            --expected-max-events "{params.max_events}" \
            --input-digi-path "{input.digi}" \
            --input-cutflow-path "{params.cutflow}" \
            --published-feature-path "{output.feature}" \
            --converter-commit "$converter_commit" \
            --validation-record "$tmp_dir/validated.json"
        mkdir -p "$(dirname "{output.feature}")"
        mv "$tmp_dir/feature.root" "{output.feature}"
        mv "$tmp_dir/validated.json" "{output.marker}"
        """


rule unified_features_mc_neutrino_local_smoke_v1:
    input:
        _unified_local_smoke_path("mc_neutrino", ".validated.json")


rule unified_features_mc_muon_up_local_smoke_v1:
    input:
        _unified_local_smoke_path("mc_muon_up", ".validated.json")


rule unified_features_mc_neutrino_canary_v1:
    input:
        lambda wc: _unified_targets("mc_neutrino", canary=True)


rule unified_features_mc_neutrino_v1:
    input:
        lambda wc: _unified_targets("mc_neutrino")


rule unified_features_mc_muon_up_canary_v1:
    input:
        lambda wc: _unified_targets("mc_muon_up", canary=True)


rule unified_features_mc_muon_up_v1:
    input:
        lambda wc: _unified_targets("mc_muon_up")


rule unified_features_real_data_2024_skim_canary_v1:
    input:
        lambda wc: _unified_targets("real_data_2024_skim", canary=True)


rule unified_features_real_data_2024_skim_v1:
    input:
        lambda wc: _unified_targets("real_data_2024_skim")


rule unified_features_testbeam_mc_2024_canary_v1:
    input:
        lambda wc: _unified_targets("testbeam_mc_2024", canary=True)


rule unified_features_testbeam_mc_2024_v1:
    input:
        lambda wc: _unified_targets("testbeam_mc_2024")


rule unified_features_testbeam_data_2024_canary_v1:
    input:
        lambda wc: _unified_targets("testbeam_data_2024", canary=True)


rule unified_features_testbeam_data_2024_v1:
    input:
        lambda wc: _unified_targets("testbeam_data_2024")
