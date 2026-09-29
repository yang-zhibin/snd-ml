DD3D_MUONDIS_FEATURE_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/ml_dataset/"
    "muondis_feature_sampling_v1.yaml"
)
with open(DD3D_MUONDIS_FEATURE_CONFIG, encoding="utf-8") as stream:
    DD3D_MUONDIS_FEATURE = yaml.safe_load(stream)

DD3D_MUONDIS_FEATURE_META = (
    f"{DD3D_DIAGNOSTICS_DIR}/muondis_feature_sampling"
)
DD3D_MUONDIS_FEATURE_ROOT = (
    f"{DD3D_ROOT}/diagnostics/muondis_feature_sampling"
)
DD3D_MUONDIS_FEATURE_TRAIN_CONFIG = DD3D_MUONDIS_FEATURE["training_config"]
DD3D_MUONDIS_HIST_CONFIG = DD3D_MUONDIS_FEATURE["hist_config"]
DD3D_MUONDIS_REGION_CONFIG = DD3D_MUONDIS_FEATURE["region_config"]
DD3D_MUONDIS_FEATURE_TREE = DD3D_MUONDIS_FEATURE["feature_tree"]


def _dd3d_muondis_feature_checkpoint(wildcards):
    return checkpoints.prepare_data_driven_3d_cnn_muondis_feature_sampling_v1.get(
        cv_fold=str(wildcards.cv_fold), epoch=str(wildcards.epoch)
    )


def _dd3d_muondis_feature_sources(wildcards):
    return _dd3d_muondis_feature_checkpoint(wildcards).output.sources


def _dd3d_muondis_feature_samples(wildcards):
    return _dd3d_muondis_feature_checkpoint(wildcards).output.samples


def _dd3d_muondis_feature_sampling_summary(wildcards):
    return _dd3d_muondis_feature_checkpoint(wildcards).output.summary


def _dd3d_muondis_feature_shards(wildcards):
    sources = pd.read_csv(
        _dd3d_muondis_feature_checkpoint(wildcards).output.sources,
        keep_default_na=False,
    )
    return [
        (
            f"{DD3D_MUONDIS_FEATURE_ROOT}/fold_{wildcards.cv_fold}/"
            f"epoch_{wildcards.epoch}/shards/{uid}.root"
        )
        for uid in sources["dataset_file_uid"].astype(str)
    ]


checkpoint prepare_data_driven_3d_cnn_muondis_feature_sampling_v1:
    input:
        validation=DD3D_VALIDATION,
        fold_views=DD3D_FOLD_VIEWS,
        training_config=DD3D_MUONDIS_FEATURE_TRAIN_CONFIG,
        workflow_config=DD3D_MUONDIS_FEATURE_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "prepare_muondis_feature_sampling.py"
        ),
        sampler=(
            f"{PERSONAL_WORK_SPACE}/train_3d/src/train_3d/data/"
            "data_driven_fold_dataset.py"
        ),
    output:
        sources=(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/source_manifest.csv"
        ),
        samples=(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/sampled_events.csv"
        ),
        summary=(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/sampling_summary.json"
        ),
    wildcard_constraints:
        cv_fold="[0-4]",
        epoch="[0-9]+"
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONPATH={PERSONAL_WORK_SPACE}/train_3d/src:${{PYTHONPATH:-}}
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --training-config {input.training_config} \
          --fold-views {input.fold_views} \
          --cv-fold {wildcards.cv_fold} --epoch {wildcards.epoch} \
          --output-sources {output.sources} \
          --output-samples {output.samples} \
          --output-summary {output.summary}
        """


rule build_data_driven_3d_cnn_muondis_feature_hist_shard_v1:
    input:
        sources=_dd3d_muondis_feature_sources,
        samples=_dd3d_muondis_feature_samples,
        dataset_config=DD3D_CONFIG,
        hist_config=DD3D_MUONDIS_HIST_CONFIG,
        region_config=DD3D_MUONDIS_REGION_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "build_muondis_feature_hist_shard.py"
        ),
        feature_helper=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_passing_muon_binary_region_hist_v5.py"
        ),
    output:
        DD3D_MUONDIS_FEATURE_ROOT
        + "/fold_{cv_fold}/epoch_{epoch}/shards/{dataset_file_uid}.root"
    wildcard_constraints:
        cv_fold="[0-4]",
        epoch="[0-9]+",
        dataset_file_uid=r"dsf_[0-9a-f]{20}"
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2500,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --source-manifest {input.sources} \
          --sample-manifest {input.samples} \
          --dataset-file-uid {wildcards.dataset_file_uid} \
          --dataset-config {input.dataset_config} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --feature-tree {DD3D_MUONDIS_FEATURE_TREE} \
          --threads {threads} --output {output}
        """


rule merge_data_driven_3d_cnn_muondis_feature_hists_v1:
    input:
        shards=_dd3d_muondis_feature_shards,
        sources=_dd3d_muondis_feature_sources,
        sampling_summary=_dd3d_muondis_feature_sampling_summary,
        hist_config=DD3D_MUONDIS_HIST_CONFIG,
        region_config=DD3D_MUONDIS_REGION_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "merge_muondis_feature_hist_shards.py"
        ),
    output:
        root=(
            DD3D_MUONDIS_FEATURE_ROOT
            + "/fold_{cv_fold}/epoch_{epoch}/merged_histograms.root"
        ),
        audit=(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/feature_integrals.csv"
        ),
        summary=(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/summary.json"
        ),
    wildcard_constraints:
        cv_fold="[0-4]",
        epoch="[0-9]+"
    threads: 1
    resources:
        runtime=2 * 60 * 60,
        mem_mb=8000,
        disk_mb=10000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --inputs {input.shards} \
          --source-manifest {input.sources} \
          --sampling-summary {input.sampling_summary} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output-root {output.root} \
          --output-audit {output.audit} \
          --output-summary {output.summary}
        """


rule plot_data_driven_3d_cnn_muondis_feature_sampling_v1:
    input:
        root=rules.merge_data_driven_3d_cnn_muondis_feature_hists_v1.output.root,
        summary=rules.merge_data_driven_3d_cnn_muondis_feature_hists_v1.output.summary,
        hist_config=DD3D_MUONDIS_HIST_CONFIG,
        region_config=DD3D_MUONDIS_REGION_CONFIG,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_data_driven_3d_cnn/"
            "plot_muondis_feature_sampling.py"
        ),
    output:
        DD3D_MUONDIS_FEATURE_META
        + "/fold_{cv_fold}/epoch_{epoch}/natural_vs_sampled.pdf"
    wildcard_constraints:
        cv_fold="[0-4]",
        epoch="[0-9]+"
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --input-root {input.root} --summary {input.summary} \
          --hist-config {input.hist_config} \
          --region-config {input.region_config} \
          --output-pdf {output}
        """


rule data_driven_3d_cnn_muondis_feature_sampling_v1:
    input:
        expand(
            DD3D_MUONDIS_FEATURE_META
            + "/fold_{cv_fold}/epoch_{epoch}/natural_vs_sampled.pdf",
            cv_fold=DD3D_MUONDIS_FEATURE["folds"],
            epoch=DD3D_MUONDIS_FEATURE["epochs"],
        )
