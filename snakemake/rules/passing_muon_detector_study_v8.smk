# V8 detector evaluation: V7 loose cuts, explicit Veto/no-Veto populations.

PASSING_MUON_V8_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "passing_muon_detector_study_v8.yaml"
)
with open(PASSING_MUON_V8_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V8 = yaml.safe_load(stream)

_pmv8_root = PASSING_MUON_V8["output"]["root_dir"].rstrip("/")
_pmv8_plot_root = (
    f"{PERSONAL_WORK_SPACE}/{PASSING_MUON_V8['output']['plot_dir'].rstrip('/')}"
)
PASSING_MUON_V8_HIST = (
    f"{_pmv8_root}/{PASSING_MUON_V8['output']['histogram']}"
)
PASSING_MUON_V8_PLOTS = {
    "passing_muon_selection": (
        f"{_pmv8_plot_root}/{PASSING_MUON_V8['output']['passing_muon_pdf']}"
    ),
    "signal_region": (
        f"{_pmv8_plot_root}/{PASSING_MUON_V8['output']['signal_region_pdf']}"
    ),
}


rule make_passing_muon_detector_study_v8:
    input:
        mc_muon_up_metadata=PASSING_MUON_V7_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V7_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V8_CONFIG,
        base_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_detector_study_v8.py"
        ),
        helper=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "passing_muon_v8.py"
        ),
        base_engine=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_builder=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_track_type_hist.py"
        ),
    output:
        root=PASSING_MUON_V8_HIST,
    params:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
    threads: 4
    resources:
        runtime=8 * 60 * 60, mem_mb=10000, disk_mb=6000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/passing_muon_v8.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {params.partition_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule plot_passing_muon_detector_study_v8:
    input:
        root=PASSING_MUON_V8_HIST,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        region_config=PASSING_MUON_V8_CONFIG,
        base_config=PASSING_MUON_V7_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_detector_study_v8.py"
        ),
        helper=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "passing_muon_v8.py"
        ),
        base_engine=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
        common_plotter=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_passing_muon_region_study_v4.py"
        ),
    output:
        pdf=_pmv8_plot_root + "/{mode}.pdf",
    wildcard_constraints:
        mode="passing_muon_selection|signal_region"
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=3000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --mode {wildcards.mode} \
          --output-pdf {output.pdf}
        """


rule passing_muon_run8329_detector_study_v8_hist:
    input:
        PASSING_MUON_V8_HIST


rule passing_muon_run8329_detector_study_v8_plots:
    input:
        list(PASSING_MUON_V8_PLOTS.values())


rule passing_muon_run8329_detector_study_v8:
    input:
        histogram=PASSING_MUON_V8_HIST,
        plots=list(PASSING_MUON_V8_PLOTS.values()),
