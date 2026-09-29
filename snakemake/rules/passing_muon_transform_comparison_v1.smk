# V1 matched Muon-DIS transform comparison with the no-Veto signal region.

PASSING_MUON_TRANSFORM_V1_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_transform_comparison_v1.yaml"
)
with open(PASSING_MUON_TRANSFORM_V1_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_TRANSFORM_V1 = yaml.safe_load(stream)

_pmtv1_root = PASSING_MUON_TRANSFORM_V1["output"]["root_dir"].rstrip("/")
_pmtv1_plot = (
    f"{PERSONAL_WORK_SPACE}/"
    f"{PASSING_MUON_TRANSFORM_V1['output']['plot_dir'].rstrip('/')}"
)
_pmtv1_upstream = (
    PASSING_MUON_TRANSFORM_V1["inputs"]["transform_production_root"].rstrip("/")
)
_pmtv1_metadata = pd.read_csv(
    PASSING_MUON_TRANSFORM_V1["inputs"]["real_dq_metadata"],
    keep_default_na=False,
)
_pmtv1_metadata["source_file_index"] = _pmtv1_metadata["source_file_index"].astype(int)
PMTV1_SOURCES = [
    f"{value:03d}" for value in sorted(_pmtv1_metadata["source_file_index"])
]
PMTV1_METHODS = list(PASSING_MUON_TRANSFORM_V1["methods"])
PMTV1_HISTOGRAM = (
    f"{_pmtv1_root}/{PASSING_MUON_TRANSFORM_V1['output']['histogram']}"
)
PMTV1_PDFS = {
    method: f"{_pmtv1_plot}/{definition['output_pdf']}"
    for method, definition in PASSING_MUON_TRANSFORM_V1["methods"].items()
}


def _pmtv1_row(wildcards):
    rows = _pmtv1_metadata[
        _pmtv1_metadata["source_file_index"] == int(wildcards.source)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one Run-8329 row for source {wildcards.source}")
    return rows.iloc[0]


def _pmtv1_original(wildcards):
    return str(_pmtv1_row(wildcards)["feature_full_path"])


rule make_muon_dis_transform_comparison_hist_shard_v1:
    input:
        config=PASSING_MUON_TRANSFORM_V1_CONFIG,
        manifest=_pmtv1_upstream + "/manifests/source{source}.csv.gz",
        simple=(
            _pmtv1_upstream
            + "/features/simple/transformedFeatures_run008329_source{source}.root"
        ),
        reco=(
            _pmtv1_upstream
            + "/features/reco_scifi/transformedFeatures_run008329_source{source}.root"
        ),
        simple_ok=_pmtv1_upstream + "/validation/features/simple/source{source}.ok",
        reco_ok=_pmtv1_upstream + "/validation/features/reco_scifi/source{source}.ok",
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_muon_dis_transform_comparison_hist_shard.py"
        ),
        feature_helper=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        histogram_helper=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    params:
        original=_pmtv1_original,
    output:
        root=_pmtv1_root + "/histograms/shards/source{source}.root",
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
        tmp_dir=$(mktemp -d "$tmp_base/muon_dis_transform_hist.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --config {input.config} --manifest {input.manifest} \
          --original {params.original} --simple {input.simple} \
          --reco-scifi {input.reco} --output "$tmp_dir/hist.root"
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/hist.root" {output.root}
        """


rule merge_muon_dis_transform_comparison_hist_v1:
    input:
        expand(
            _pmtv1_root + "/histograms/shards/source{source}.root",
            source=PMTV1_SOURCES,
        )
    output:
        root=PMTV1_HISTOGRAM
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
        tmp_dir=$(mktemp -d "$tmp_base/muon_dis_transform_merge.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        hadd -f "$tmp_dir/merged.root" {input}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/merged.root" {output.root}
        """


rule plot_muon_dis_transform_comparison_v1:
    input:
        root=PMTV1_HISTOGRAM,
        config=PASSING_MUON_TRANSFORM_V1_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_muon_dis_transform_comparison.py"
        ),
        common_plotter=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmtv1_plot + "/{method}_transform_comparison.pdf",
    wildcard_constraints:
        method="reco_scifi|simple"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} --input {input.root} --config {input.config} \
          --method {wildcards.method} --output {output.pdf}
        """


rule passing_muon_run8329_transform_comparison_v1_hist:
    input:
        PMTV1_HISTOGRAM


rule passing_muon_run8329_transform_comparison_v1_plots:
    input:
        list(PMTV1_PDFS.values())


rule passing_muon_run8329_transform_comparison_v1:
    input:
        histogram=PMTV1_HISTOGRAM,
        plots=list(PMTV1_PDFS.values()),
