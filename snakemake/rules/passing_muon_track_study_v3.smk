import sys
sys.path.insert(0, str(Path(workflow.basedir).resolve()))
from output_paths import resolve_output

# Run-8329 v3: track-type categories and the exact NoPassingMuonReco complement.

PASSING_MUON_V3_CONFIG = f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/region_evaluation/passing_muon_track_study_v3.yaml"
with open(PASSING_MUON_V3_CONFIG, encoding="utf-8") as stream:
    PASSING_MUON_V3 = yaml.safe_load(stream)

PASSING_MUON_V3_WORKING_POINTS = list(PASSING_MUON_V3["working_points"])
PASSING_MUON_V3_TRACK_TYPES = list(PASSING_MUON_V3["track_types"])
PASSING_MUON_V3_ALGORITHM_COMPARISONS = [
    f"{algorithm}_track_combinations"
    for algorithm in PASSING_MUON_V3["algorithm_comparisons"]
]
PASSING_MUON_V3_COMPARISONS = (
    PASSING_MUON_V3_TRACK_TYPES
    + ["all_track_types"]
    + PASSING_MUON_V3_ALGORITHM_COMPARISONS
)
_pmv3_root = PASSING_MUON_V3["output"]["root_dir"].rstrip("/")
_pmv3_plot_root = resolve_output(PASSING_MUON_V3['output']['plot_dir'].rstrip('/'), PERSONAL_WORK_SPACE)
_pmv3_metadata_root = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/run8329_track_study_v3"

PASSING_MUON_V3_MISSING_ROOTS = [
    f"{_pmv3_root}/no_passing_muon_reco/part{part}.root"
    for part in PASSING_MUON_V2_PARTS
]
PASSING_MUON_V3_MISSING_AUDITS = [
    f"{_pmv3_metadata_root}/audits/part{part}.csv"
    for part in PASSING_MUON_V2_PARTS
]
PASSING_MUON_V3_MISSING_METADATA = (
    resolve_output(PASSING_MUON_V3['output']['metadata_csv'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V3_VALIDATION = (
    resolve_output(PASSING_MUON_V3['output']['validation_csv'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V3_MC_UP_SOURCE_METADATA = (
    resolve_output(PASSING_MUON_V3['mc_muon_up']['source_metadata'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V3_MC_UP_METADATA = (
    resolve_output(PASSING_MUON_V3['output']['mc_muon_up_metadata_csv'], PERSONAL_WORK_SPACE)
)
PASSING_MUON_V3_MC_UP_VALIDATION = (
    resolve_output(PASSING_MUON_V3['output']['mc_muon_up_validation_csv'], PERSONAL_WORK_SPACE)
)


def _passing_muon_v3_mc_up_feature():
    metadata = pd.read_csv(PASSING_MUON_V3_MC_UP_SOURCE_METADATA, keep_default_na=False)
    source = PASSING_MUON_V3["mc_muon_up"]["source"]
    rows = metadata.loc[
        metadata["subfolder"].astype(str).eq(str(source["subfolder"]))
        & metadata["partition"].astype(str).eq(str(source["partition"]))
        & metadata["feature_path"].astype(str).eq(str(source["feature_name"]))
    ]
    if len(rows) != 1:
        raise ValueError(
            "Configured v3 MC-muon-up source must match exactly one metadata row"
        )
    row = rows.iloc[0]
    feature = str(row["feature_path"])
    return feature if os.path.isabs(feature) else os.path.join(
        str(row["output_base_path"]), feature
    )


PASSING_MUON_V3_MC_UP_FEATURE = _passing_muon_v3_mc_up_feature()
PASSING_MUON_V3_HISTS = [
    f"{_pmv3_root}/histograms/{working_point}.root"
    for working_point in PASSING_MUON_V3_WORKING_POINTS
]
PASSING_MUON_V3_PLOTS = [
    f"{_pmv3_plot_root}/{working_point}/{comparison}.pdf"
    for working_point in PASSING_MUON_V3_WORKING_POINTS
    for comparison in PASSING_MUON_V3_COMPARISONS
]
PASSING_MUON_V3_MC_UP_REFERENCE_PDF = (
    f"{_pmv3_plot_root}/no_quality_cut/mc_muon_up_reference.pdf"
)


rule prepare_mc_muon_up_reference_v3:
    input:
        source_metadata=PASSING_MUON_V3_MC_UP_SOURCE_METADATA,
        config=PASSING_MUON_V3_CONFIG,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "prepare_mc_muon_up_reference_v3.py"
        ),
    output:
        metadata=PASSING_MUON_V3_MC_UP_METADATA,
        validation=PASSING_MUON_V3_MC_UP_VALIDATION,
    threads: 1
    resources:
        runtime=30 * 60, mem_mb=2000, disk_mb=1000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --source-metadata {input.source_metadata} \
          --study-config {input.config} \
          --hist-config {input.hist_config} \
          --output-metadata {output.metadata} \
          --output-validation {output.validation}
        """


rule build_no_passing_muon_partition_v3:
    input:
        real=lambda wc: _pmv2_partition_root("real_data_dq_run8329", wc.part),
        passing=lambda wc: _pmv2_partition_root("passing_muon_dq_run8329", wc.part),
        validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_no_passing_muon_partition.py",
    output:
        root=_pmv3_root + "/no_passing_muon_reco/part{part}.root",
        audit=_pmv3_metadata_root + "/audits/part{part}.csv",
    wildcard_constraints:
        part="|".join(PASSING_MUON_V2_PARTS)
    threads: 1
    resources:
        runtime=2 * 60 * 60, mem_mb=4000, disk_mb=5000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/no_passing_muon_v3.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --real-partition {input.real} \
          --passing-partition {input.passing} \
          --part-index {wildcards.part} \
          --run {PASSING_MUON_V3[run]} \
          --tree-name {PASSING_MUON_V3[tree_name]} \
          --output-root "$tmp_dir/output.root" \
          --published-root {output.root} \
          --output-audit "$tmp_dir/audit.csv" \
          --compression-level {PASSING_MUON_V3[output][compression_level]}
        mkdir -p "$(dirname {output.root})" "$(dirname {output.audit})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        cp "$tmp_dir/audit.csv" {output.audit}
        """


rule passing_muon_run8329_no_passing_reco_v3:
    input:
        roots=PASSING_MUON_V3_MISSING_ROOTS,
        audits=PASSING_MUON_V3_MISSING_AUDITS,


rule validate_passing_muon_track_study_v3:
    input:
        audits=PASSING_MUON_V3_MISSING_AUDITS,
        roots=PASSING_MUON_V3_MISSING_ROOTS,
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        # Track-category and plotting changes do not alter the already-built
        # exact RealDataDQ-minus-PassingMuonReco complement.
        config=ancient(PASSING_MUON_V3_CONFIG),
        script=f"{PERSONAL_WORK_SPACE}/convertData/validate_passing_muon_track_study_v3.py",
    output:
        metadata=PASSING_MUON_V3_MISSING_METADATA,
        validation=PASSING_MUON_V3_VALIDATION,
    threads: 1
    resources:
        runtime=4 * 60 * 60, mem_mb=5000, disk_mb=2000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --audits {input.audits} \
          --partition-metadata {input.partition_metadata} \
          --config {input.config} \
          --output-metadata {output.metadata} \
          --output-validation {output.validation}
        """


rule passing_muon_run8329_track_study_validation_v3:
    input:
        metadata=PASSING_MUON_V3_MISSING_METADATA,
        validation=PASSING_MUON_V3_VALIDATION,


rule make_passing_muon_track_type_hist_v3:
    input:
        partition_metadata=PASSING_MUON_V2_PARTITION_METADATA,
        partition_validation=PASSING_MUON_V2_PARTITION_VALIDATION,
        missing_metadata=PASSING_MUON_V3_MISSING_METADATA,
        missing_validation=PASSING_MUON_V3_VALIDATION,
        mc_muon_up_metadata=PASSING_MUON_V3_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V3_MC_UP_VALIDATION,
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        study_config=PASSING_MUON_V3_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/make_passing_muon_track_type_hist.py",
    output:
        root=_pmv3_root + "/histograms/{working_point}.root",
    wildcard_constraints:
        working_point="|".join(PASSING_MUON_V3_WORKING_POINTS)
    threads: 4
    resources:
        runtime=6 * 60 * 60, mem_mb=8000, disk_mb=5000, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/track_study_hist_v3.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python {input.script} \
          --partition-metadata {input.partition_metadata} \
          --missing-metadata {input.missing_metadata} \
          --mc-muon-up-metadata {input.mc_muon_up_metadata} \
          --hist-config {input.hist_config} \
          --study-config {input.study_config} \
          --working-point {wildcards.working_point} \
          --output "$tmp_dir/output.root" \
          --threads {threads}
        mkdir -p "$(dirname {output.root})"
        xrdcp -f "$tmp_dir/output.root" {output.root}
        """


rule passing_muon_run8329_track_study_hists_v3:
    input:
        PASSING_MUON_V3_HISTS


rule plot_passing_muon_mc_up_reference_v3:
    input:
        root=f"{_pmv3_root}/histograms/no_quality_cut.root",
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        study_config=PASSING_MUON_V3_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/plot_passing_muon_track_type.py",
    output:
        pdf=PASSING_MUON_V3_MC_UP_REFERENCE_PDF,
    threads: 1
    resources:
        runtime=45 * 60, mem_mb=3000, disk_mb=1500, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --study-config {input.study_config} \
          --comparison mc_muon_up_reference \
          --output-pdf {output.pdf}
        """


rule plot_passing_muon_track_type_v3:
    input:
        root=_pmv3_root + "/histograms/{working_point}.root",
        hist_config=PASSING_MUON_V2_HIST_CONFIG,
        study_config=PASSING_MUON_V3_CONFIG,
        script=f"{WORKFLOW_CONFIG_REPO}/evaluation_region_partitions/plot_passing_muon_track_type.py",
    output:
        pdf=_pmv3_plot_root + "/{working_point}/{comparison}.pdf",
    wildcard_constraints:
        working_point="|".join(PASSING_MUON_V3_WORKING_POINTS),
        comparison="|".join(PASSING_MUON_V3_COMPARISONS)
    threads: 1
    resources:
        runtime=45 * 60, mem_mb=2500, disk_mb=1500, nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        python {input.script} \
          --input {input.root} \
          --hist-config {input.hist_config} \
          --study-config {input.study_config} \
          --comparison {wildcards.comparison} \
          --output-pdf {output.pdf}
        """


rule passing_muon_run8329_track_study_plots_v3:
    input:
        plots=PASSING_MUON_V3_PLOTS,
        mc_muon_up_reference=PASSING_MUON_V3_MC_UP_REFERENCE_PDF,


rule passing_muon_run8329_track_study_v3:
    input:
        metadata=PASSING_MUON_V3_MISSING_METADATA,
        validation=PASSING_MUON_V3_VALIDATION,
        mc_muon_up_metadata=PASSING_MUON_V3_MC_UP_METADATA,
        mc_muon_up_validation=PASSING_MUON_V3_MC_UP_VALIDATION,
        histograms=PASSING_MUON_V3_HISTS,
        plots=PASSING_MUON_V3_PLOTS,
        mc_muon_up_reference=PASSING_MUON_V3_MC_UP_REFERENCE_PDF,
