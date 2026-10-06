# Full Run-8329 RealDataDQ detector-ID transform and matched-feature study.

PASSING_MUON_TRANSFORM_V8_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_transform_comparison_v8.yaml"
)
with open(PASSING_MUON_TRANSFORM_V8_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_TRANSFORM_V8 = yaml.safe_load(stream)

_pmtv8_root = PASSING_MUON_TRANSFORM_V8["output"]["root_dir"].rstrip("/")
_pmtv8_plot = (
    f"{PERSONAL_WORK_SPACE}/"
    f"{PASSING_MUON_TRANSFORM_V8['output']['plot_dir'].rstrip('/')}"
)
_pmtv8_metadata_path = PASSING_MUON_TRANSFORM_V8["inputs"]["real_dq_metadata"]
_pmtv8_metadata = pd.read_csv(_pmtv8_metadata_path, keep_default_na=False)
_pmtv8_metadata["source_file_index"] = _pmtv8_metadata["source_file_index"].astype(int)
PMTV8_SOURCES = [f"{value:03d}" for value in sorted(_pmtv8_metadata["source_file_index"])]
PMTV8_METHODS = ["simple", "reco_scifi"]
PMTV8_COMPARISONS = list(PASSING_MUON_TRANSFORM_V8["comparisons"])


def _pmtv8_row(wildcards):
    source = int(wildcards.source)
    rows = _pmtv8_metadata[_pmtv8_metadata["source_file_index"] == source]
    if len(rows) != 1:
        raise ValueError(f"Expected one Run-8329 metadata row for source {source}")
    return rows.iloc[0]


def _pmtv8_value(wildcards, field):
    return str(_pmtv8_row(wildcards)[field])


def _pmtv8_manifest(source):
    return f"{_pmtv8_root}/manifests/source{source}.csv.gz"


def _pmtv8_manifest_summary(source):
    return f"{_pmtv8_root}/manifests/source{source}.json"


def _pmtv8_digi(method, source):
    return f"{_pmtv8_root}/transformed_digi/{method}/run008329_source{source}.root"


def _pmtv8_audit(method, source):
    return f"{_pmtv8_root}/audits/{method}/source{source}.csv.gz"


def _pmtv8_summary(method, source):
    return f"{_pmtv8_root}/audits/{method}/source{source}.json"


def _pmtv8_feature(method, source):
    return f"{_pmtv8_root}/features/{method}/transformedFeatures_run008329_source{source}.root"


def _pmtv8_hist_shard(source):
    return f"{_pmtv8_root}/histograms/shards/source{source}.root"


PMTV8_HISTOGRAM = f"{_pmtv8_root}/{PASSING_MUON_TRANSFORM_V8['output']['histogram']}"
PMTV8_PDFS = {
    comparison: f"{_pmtv8_plot}/{comparison}.pdf"
    for comparison in PMTV8_COMPARISONS
}


rule prepare_run8329_transform_manifest_v8:
    input:
        config=PASSING_MUON_TRANSFORM_V8_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/prepare_run8329_transform_manifest.py",
        track_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/scifi_edge_transform_sndsw.py",
        transform_helper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
    output:
        manifest=_pmtv8_root + "/manifests/source{source}.csv.gz",
        summary=_pmtv8_root + "/manifests/source{source}.json",
    params:
        metadata=_pmtv8_metadata_path,
    wildcard_constraints:
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=12 * 60 * 60, mem_mb=5000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_manifest.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /usr/bin/python3 {input.script} --config {input.config} \
          --source-index {wildcards.source} --output-manifest "$tmp_dir/manifest.csv.gz" \
          --output-summary "$tmp_dir/summary.json"
        mkdir -p "$(dirname {output.manifest})"
        xrdcp -f "$tmp_dir/manifest.csv.gz" {output.manifest}
        xrdcp -f "$tmp_dir/summary.json" {output.summary}
        """


rule transform_run8329_detid_shard_v8:
    input:
        config=PASSING_MUON_TRANSFORM_V8_CONFIG,
        manifest=_pmtv8_root + "/manifests/source{source}.csv.gz",
        source=lambda wc: _pmtv8_value(wc, "source_path"),
        geometry=lambda wc: _pmtv8_value(wc, "geo_path"),
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_shard.py",
        event_writer=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/write_detid_channel_shift_raw.py",
        mapper=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift.py",
        adapter=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/detid_channel_shift_sndsw.py",
    output:
        root=_pmtv8_root + "/transformed_digi/{method}/run008329_source{source}.root",
        audit=_pmtv8_root + "/audits/{method}/source{source}.csv.gz",
        summary=_pmtv8_root + "/audits/{method}/source{source}.json",
    wildcard_constraints:
        method="simple|reco_scifi",
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=24 * 60 * 60, mem_mb=6000, disk_mb=12000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_digi.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /usr/bin/python3 {input.script} --config {input.config} \
          --method {wildcards.method} --source {input.source} --geometry {input.geometry} \
          --manifest {input.manifest} --output-root "$tmp_dir/transformed.root" \
          --output-audit "$tmp_dir/audit.csv.gz" --output-summary "$tmp_dir/summary.json"
        mkdir -p "$(dirname {output.root})" "$(dirname {output.audit})"
        xrdcp -f "$tmp_dir/transformed.root" {output.root}
        xrdcp -f "$tmp_dir/audit.csv.gz" {output.audit}
        xrdcp -f "$tmp_dir/summary.json" {output.summary}
        """


rule validate_run8329_detid_shard_v8:
    input:
        root=_pmtv8_root + "/transformed_digi/{method}/run008329_source{source}.root",
        audit=_pmtv8_root + "/audits/{method}/source{source}.csv.gz",
        summary=_pmtv8_root + "/audits/{method}/source{source}.json",
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_detid_channel_shift_shard.py",
    output:
        marker=_pmtv8_root + "/validation/digi/{method}/source{source}.ok",
    wildcard_constraints:
        method="simple|reco_scifi",
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=3000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_validate_digi.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /usr/bin/python3 {input.script} --root {input.root} --audit {input.audit} \
          --summary {input.summary} --marker "$tmp_dir/validated.ok"
        mkdir -p "$(dirname {output.marker})"
        xrdcp -f "$tmp_dir/validated.ok" {output.marker}
        """


rule extract_run8329_transform_features_v8:
    input:
        root=_pmtv8_root + "/transformed_digi/{method}/run008329_source{source}.root",
        validated=_pmtv8_root + "/validation/digi/{method}/source{source}.ok",
        geometry=lambda wc: _pmtv8_value(wc, "geo_path"),
        script=f"{PERSONAL_WORK_SPACE}/convertData/digi_2_features.py",
        selection=f"{PERSONAL_WORK_SPACE}/convertData/sample_selection.py",
        scifi_clustering=f"{PERSONAL_WORK_SPACE}/convertData/scifi_plane_clustering.py",
    output:
        feature=_pmtv8_root + "/features/{method}/transformedFeatures_run008329_source{source}.root",
    wildcard_constraints:
        method="simple|reco_scifi",
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=12 * 60 * 60, mem_mb=4000, disk_mb=8000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source /afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh
        conda deactivate 2>/dev/null || true
        source {env_script_sndsw_nue}
        set -euo pipefail
        export PYTHONPATH="$SNDSW_ROOT:${{PYTHONPATH:-}}"
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_features.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.script} \
          -d {input.root} -g {input.geometry} -o "$tmp_dir/features.root" \
          -t real_data --tree-name cbmsim --selection-mode direct_input
        mkdir -p "$(dirname {output.feature})"
        xrdcp -f "$tmp_dir/features.root" {output.feature}
        """


rule validate_run8329_transform_features_v8:
    input:
        feature=_pmtv8_root + "/features/{method}/transformedFeatures_run008329_source{source}.root",
        audit=_pmtv8_root + "/audits/{method}/source{source}.csv.gz",
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/validate_transform_feature_shard.py",
    output:
        marker=_pmtv8_root + "/validation/features/{method}/source{source}.ok",
    wildcard_constraints:
        method="simple|reco_scifi",
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=60 * 60, mem_mb=2500, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_validate_features.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} --feature {input.feature} --audit {input.audit} \
          --marker "$tmp_dir/validated.ok"
        mkdir -p "$(dirname {output.marker})"
        xrdcp -f "$tmp_dir/validated.ok" {output.marker}
        """


rule make_run8329_transform_comparison_hist_shard_v8:
    input:
        config=PASSING_MUON_TRANSFORM_V8_CONFIG,
        manifest=_pmtv8_root + "/manifests/source{source}.csv.gz",
        simple=_pmtv8_root + "/features/simple/transformedFeatures_run008329_source{source}.root",
        reco=_pmtv8_root + "/features/reco_scifi/transformedFeatures_run008329_source{source}.root",
        simple_ok=_pmtv8_root + "/validation/features/simple/source{source}.ok",
        reco_ok=_pmtv8_root + "/validation/features/reco_scifi/source{source}.ok",
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_transform_comparison_hist_shard.py",
    params:
        original=lambda wc: _pmtv8_value(wc, "feature_full_path"),
    output:
        root=_pmtv8_root + "/histograms/shards/source{source}.root",
    wildcard_constraints:
        source="[0-9]{3}"
    threads: 1
    resources:
        runtime=12 * 60 * 60, mem_mb=5000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} --config {input.config} --manifest {input.manifest} \
          --original {params.original} --simple {input.simple} \
          --reco-scifi {input.reco} --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/hist.root" {output.root}
        """


rule merge_run8329_transform_comparison_hist_v8:
    input:
        expand(_pmtv8_root + "/histograms/shards/source{source}.root", source=PMTV8_SOURCES)
    output:
        root=PMTV8_HISTOGRAM
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=8000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_base="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_base"
        tmp_dir=$(mktemp -d "$tmp_base/transform_merge.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        mkdir -p "$(dirname {output.root})"
        hadd -f "$tmp_dir/merged.root" {input}
        xrdcp -f "$tmp_dir/merged.root" {output.root}
        """


rule plot_run8329_transform_comparison_v8:
    input:
        root=PMTV8_HISTOGRAM,
        config=PASSING_MUON_TRANSFORM_V8_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_transform_feature_comparison.py",
    output:
        pdf=_pmtv8_plot + "/{comparison}.pdf",
    wildcard_constraints:
        comparison="scifi_track|no_veto|loose_v7"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=4000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} --input {input.root} --config {input.config} \
          --comparison {wildcards.comparison} --output {output.pdf}
        """


rule passing_muon_run8329_transform_manifests_v8:
    input:
        expand(_pmtv8_root + "/manifests/source{source}.csv.gz", source=PMTV8_SOURCES),
        expand(_pmtv8_root + "/manifests/source{source}.json", source=PMTV8_SOURCES),


rule passing_muon_run8329_transformed_digis_v8:
    input:
        expand(
            _pmtv8_root + "/validation/digi/{method}/source{source}.ok",
            method=PMTV8_METHODS, source=PMTV8_SOURCES,
        )


rule passing_muon_run8329_transformed_features_v8:
    input:
        expand(
            _pmtv8_root + "/validation/features/{method}/source{source}.ok",
            method=PMTV8_METHODS, source=PMTV8_SOURCES,
        )


rule passing_muon_run8329_transform_comparison_v8:
    input:
        histogram=PMTV8_HISTOGRAM,
        plots=list(PMTV8_PDFS.values()),
