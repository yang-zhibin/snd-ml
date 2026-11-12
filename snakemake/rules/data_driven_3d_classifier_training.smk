DD3D_TRAINING_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/train_3d/configs/classifier_sparse_resnet_v1.yaml"
)
DD3D_SCIFI_META_ROOT = (
    f"{PERSONAL_WORK_SPACE}/train_3d/metadata/scifi_xyz_qdc_dettype_v1"
)
DD3D_SCIFI_META_OUTPUTS = [
    f"{DD3D_SCIFI_META_ROOT}/fold_{fold}/{name}"
    for fold in range(int(DD3D["cross_validation"]["n_folds"]))
    for name in ("normalization.json", "event_sizes.csv", "summary.json")
]


rule build_data_driven_3d_cnn_scifi_training_metadata_v1:
    input:
        dataset_validation=DD3D_VALIDATION,
        fold_views=DD3D_FOLD_VIEWS,
        config=DD3D_TRAINING_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/build_scifi_statistics.py",
        reader=(
            f"{PERSONAL_WORK_SPACE}/train_3d/src/train_3d/data/"
            "scifi_hit_event_dataset.py"
        ),
        statistics=(
            f"{PERSONAL_WORK_SPACE}/train_3d/src/train_3d/data/"
            "scifi_statistics.py"
        ),
    output:
        DD3D_SCIFI_META_OUTPUTS
    threads: 1
    resources:
        runtime=48 * 60 * 60,
        mem_mb=4000,
        disk_mb=10000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --output-root {DD3D_SCIFI_META_ROOT}
        """


rule data_driven_3d_cnn_scifi_training_metadata_v1:
    input:
        DD3D_SCIFI_META_OUTPUTS


# Compact v2 metadata preserves the v1 physics contract while sharding ROOT
# scans and replacing five enormous CSV dictionaries with memory-mapped arrays.
DD3D_SCIFI_META_V2_ROOT = (
    f"{PERSONAL_WORK_SPACE}/train_3d/metadata/"
    "scifi_xyz_qdc_dettype_metadata_v2"
)
DD3D_SCIFI_META_V2_SHARDS = 50
DD3D_SCIFI_META_V2_EVENT_GROUPS = 250
DD3D_SCIFI_META_V2_PLAN = f"{DD3D_SCIFI_META_V2_ROOT}/shard_plan.json"
DD3D_SCIFI_META_V2_STATS = f"{DD3D_SCIFI_META_V2_ROOT}/statistics_complete.json"
DD3D_SCIFI_META_V2_EVENT_PLAN = (
    f"{DD3D_SCIFI_META_V2_ROOT}/event_size_group_plan.json"
)
DD3D_SCIFI_META_V2_EVENT_CACHE = (
    f"{DD3D_SCIFI_META_V2_ROOT}/event_size_files"
)
DD3D_SCIFI_META_V2_MIGRATION = (
    f"{DD3D_SCIFI_META_V2_EVENT_CACHE}/legacy_migration_complete.json"
)
DD3D_SCIFI_META_V2_EVENT_GROUP_ROOT = (
    f"{DD3D_SCIFI_META_V2_ROOT}/event_size_groups"
)
DD3D_SCIFI_META_V2_COMPLETE = f"{DD3D_SCIFI_META_V2_ROOT}/complete.json"
DD3D_SCIFI_META_V2_VALIDATION = f"{DD3D_SCIFI_META_V2_ROOT}/validation.json"
DD3D_SCIFI_META_V2_STATS_SHARDS = [
    f"{DD3D_SCIFI_META_V2_ROOT}/statistics_shards/shard_{shard:03d}/complete.json"
    for shard in range(DD3D_SCIFI_META_V2_SHARDS)
]
DD3D_SCIFI_META_V2_SIZE_SHARDS = [
    f"{DD3D_SCIFI_META_V2_ROOT}/event_size_shards/shard_{shard:03d}/complete.json"
    for shard in range(DD3D_SCIFI_META_V2_SHARDS)
]
DD3D_SCIFI_META_V2_SIZE_GROUPS = [
    f"{DD3D_SCIFI_META_V2_EVENT_GROUP_ROOT}/group_{group:03d}/complete.json"
    for group in range(DD3D_SCIFI_META_V2_EVENT_GROUPS)
]
DD3D_SCIFI_META_V2_PRODUCTS = [
    f"{DD3D_SCIFI_META_V2_ROOT}/index.json",
    f"{DD3D_SCIFI_META_V2_ROOT}/source_entries.npy",
    f"{DD3D_SCIFI_META_V2_ROOT}/n_candidates.npy",
] + [
    f"{DD3D_SCIFI_META_V2_ROOT}/n_voxels_fold_{fold}.npy"
    for fold in range(int(DD3D["cross_validation"]["n_folds"]))
] + [
    f"{DD3D_SCIFI_META_V2_ROOT}/fold_{fold}/{name}"
    for fold in range(int(DD3D["cross_validation"]["n_folds"]))
    for name in ("normalization.json", "summary.json")
]


rule plan_data_driven_3d_cnn_scifi_training_metadata_v2:
    input:
        dataset_validation=DD3D_VALIDATION,
        fold_views=DD3D_FOLD_VIEWS,
        config=DD3D_TRAINING_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/plan_scifi_metadata_shards.py"
    output:
        DD3D_SCIFI_META_V2_PLAN
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} --num-shards {DD3D_SCIFI_META_V2_SHARDS} \
          --output {output}
        """


rule build_data_driven_3d_cnn_scifi_statistics_shard_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/build_scifi_metadata_shard.py"
    output:
        f"{DD3D_SCIFI_META_V2_ROOT}/statistics_shards/shard_{{shard}}/complete.json"
    wildcard_constraints:
        shard="[0-9]{3}"
    threads: 1
    resources:
        runtime=8 * 60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    params:
        step_size=5000,
        output_dir=lambda wc: (
            f"{DD3D_SCIFI_META_V2_ROOT}/statistics_shards/shard_{wc.shard}"
        )
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --shard-index $((10#{wildcards.shard})) --mode statistics \
          --step-size {params.step_size} \
          --output-dir {params.output_dir}
        """


rule reduce_data_driven_3d_cnn_scifi_statistics_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        shards=DD3D_SCIFI_META_V2_STATS_SHARDS,
        config=DD3D_TRAINING_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/reduce_scifi_statistics.py"
    output:
        complete=DD3D_SCIFI_META_V2_STATS,
        folds=[
            f"{DD3D_SCIFI_META_V2_ROOT}/fold_{fold}/{name}"
            for fold in range(int(DD3D["cross_validation"]["n_folds"]))
            for name in ("normalization.json", "summary.json")
        ]
    resources:
        runtime=15 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --shards-root {DD3D_SCIFI_META_V2_ROOT}/statistics_shards \
          --config {input.config} --output-root {DD3D_SCIFI_META_V2_ROOT}
        """


rule build_data_driven_3d_cnn_scifi_event_sizes_shard_v2_legacy:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        statistics=DD3D_SCIFI_META_V2_STATS,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/build_scifi_metadata_shard.py"
    output:
        f"{DD3D_SCIFI_META_V2_ROOT}/event_size_shards/shard_{{shard}}/complete.json"
    wildcard_constraints:
        shard="[0-9]{3}"
    resources:
        runtime=8 * 60 * 60,
        mem_mb=4000,
        disk_mb=5000,
        nvidia_gpu=0
    params:
        output_dir=lambda wc: (
            f"{DD3D_SCIFI_META_V2_ROOT}/event_size_shards/shard_{wc.shard}"
        )
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --shard-index $((10#{wildcards.shard})) --mode event-sizes \
          --statistics-root {DD3D_SCIFI_META_V2_ROOT} \
          --output-dir {params.output_dir}
        """


rule plan_data_driven_3d_cnn_scifi_event_size_groups_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        statistics=DD3D_SCIFI_META_V2_STATS,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/plan_scifi_event_size_groups.py"
    output:
        DD3D_SCIFI_META_V2_EVENT_PLAN
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --statistics-complete {input.statistics} \
          --num-groups {DD3D_SCIFI_META_V2_EVENT_GROUPS} --output {output}
        """


rule migrate_data_driven_3d_cnn_scifi_event_size_shards_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        group_plan=DD3D_SCIFI_META_V2_EVENT_PLAN,
        statistics=DD3D_SCIFI_META_V2_STATS,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/migrate_scifi_event_size_shards.py"
    output:
        DD3D_SCIFI_META_V2_MIGRATION
    resources:
        runtime=2 * 60 * 60,
        mem_mb=3000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --plan {input.plan} --group-plan {input.group_plan} \
          --legacy-shards-root {DD3D_SCIFI_META_V2_ROOT}/event_size_shards \
          --statistics-root {DD3D_SCIFI_META_V2_ROOT} \
          --cache-root {DD3D_SCIFI_META_V2_EVENT_CACHE} --output {output}
        """


rule build_data_driven_3d_cnn_scifi_event_size_group_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        group_plan=DD3D_SCIFI_META_V2_EVENT_PLAN,
        statistics=DD3D_SCIFI_META_V2_STATS,
        migration=DD3D_SCIFI_META_V2_MIGRATION,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/build_scifi_event_size_group.py"
    output:
        f"{DD3D_SCIFI_META_V2_EVENT_GROUP_ROOT}/group_{{group}}/complete.json"
    wildcard_constraints:
        group="[0-9]{3}"
    threads: 1
    resources:
        runtime=4 * 60 * 60,
        mem_mb=2000,
        disk_mb=2000,
        metadata_io=1,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        mkdir -p $(dirname {output})
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --plan {input.plan} --group-plan {input.group_plan} \
          --group-index $((10#{wildcards.group})) \
          --statistics-root {DD3D_SCIFI_META_V2_ROOT} \
          --cache-root {DD3D_SCIFI_META_V2_EVENT_CACHE} --output {output}
        """


rule reduce_data_driven_3d_cnn_scifi_event_sizes_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        group_plan=DD3D_SCIFI_META_V2_EVENT_PLAN,
        statistics=DD3D_SCIFI_META_V2_STATS,
        groups=DD3D_SCIFI_META_V2_SIZE_GROUPS,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/reduce_scifi_event_size_files.py"
    output:
        complete=DD3D_SCIFI_META_V2_COMPLETE,
        products=DD3D_SCIFI_META_V2_PRODUCTS[:8]
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --group-plan {input.group_plan} \
          --groups-root {DD3D_SCIFI_META_V2_EVENT_GROUP_ROOT} \
          --cache-root {DD3D_SCIFI_META_V2_EVENT_CACHE} \
          --statistics-root {DD3D_SCIFI_META_V2_ROOT} \
          --output-root {DD3D_SCIFI_META_V2_ROOT}
        """


rule validate_data_driven_3d_cnn_scifi_training_metadata_v2:
    input:
        plan=DD3D_SCIFI_META_V2_PLAN,
        complete=DD3D_SCIFI_META_V2_COMPLETE,
        products=DD3D_SCIFI_META_V2_PRODUCTS,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/validate_scifi_metadata.py"
    output:
        DD3D_SCIFI_META_V2_VALIDATION
    resources:
        runtime=2 * 60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --plan {input.plan} \
          --metadata-root {DD3D_SCIFI_META_V2_ROOT} --output {output}
        """


rule data_driven_3d_cnn_scifi_training_metadata_v2:
    input:
        DD3D_SCIFI_META_V2_VALIDATION


DD3D_NATIVE_SCIFI_META_ROOT = (
    f"{PERSONAL_WORK_SPACE}/train_3d/metadata/scifi_native_hits_metadata_v1"
)
DD3D_NATIVE_SCIFI_META_COMPLETE = f"{DD3D_NATIVE_SCIFI_META_ROOT}/complete.json"


rule build_data_driven_3d_native_scifi_training_metadata_v1:
    input:
        dataset_validation=DD3D_VALIDATION,
        fold_views=DD3D_FOLD_VIEWS,
        config=(
            f"{PERSONAL_WORK_SPACE}/train_3d/configs/"
            "classifier_native_scifi_gravnet_v1.yaml"
        ),
        script=f"{PERSONAL_WORK_SPACE}/train_3d/build_native_scifi_metadata.py",
    output:
        DD3D_NATIVE_SCIFI_META_COMPLETE
    threads: 1
    resources:
        runtime=72 * 60 * 60,
        mem_mb=4000,
        disk_mb=5000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} --output-root {DD3D_NATIVE_SCIFI_META_ROOT}
        """


rule data_driven_3d_native_scifi_training_metadata_v1:
    input:
        DD3D_NATIVE_SCIFI_META_COMPLETE


# One-epoch, online-W&B timing tests.  They intentionally use the A100-tested
# LCG overlay and a separate, checksummed W&B runtime rather than the training
# SIF, because the latter does not provide the PyROOT reader.
DD3D_WANDB_TEST_ROOT = (
    f"{PERSONAL_WORK_SPACE}/train_3d/test_outputs/"
    "three_model_wandb_benchmark_v1"
)
DD3D_WANDB_TEST_GROUP = "three_model_fold0_timing_v1"
DD3D_WANDB_TEST_CONFIGS = {
    "clustered_pyg_gravnet_v1": (
        f"{PERSONAL_WORK_SPACE}/train_3d/configs/"
        "classifier_clustered_pyg_gravnet_v1.yaml"
    ),
    "native_scifi_gravnet_v1": (
        f"{PERSONAL_WORK_SPACE}/train_3d/configs/"
        "classifier_native_scifi_gravnet_v1.yaml"
    ),
    "two_view_sparse_resnet_v1": (
        f"{PERSONAL_WORK_SPACE}/train_3d/configs/"
        "classifier_two_view_sparse_resnet_v1.yaml"
    ),
}
DD3D_WANDB_TEST_FAMILIES = {
    model: yaml.safe_load(open(path, "r", encoding="utf-8"))["model"]["family"]
    for model, path in DD3D_WANDB_TEST_CONFIGS.items()
}
DD3D_LCG_OVERLAY_URI = (
    "root://eosuser.cern.ch//eos/user/z/zhibin/snd-ml/overlays/"
    "snd-ml-lcg107cuda-overlay/v2/snd-ml-lcg107cuda-overlay-v2.tar.zst"
)
DD3D_LCG_OVERLAY_SHA256 = (
    "bcdcb3cfb07f7a0598b591eee51eb21b9f12b40902a9c18a20de955b6339e8f3"
)
DD3D_WANDB_RUNTIME = (
    f"{PERSONAL_WORK_SPACE}/containers/releases/"
    "snd-ml-wandb-runtime-v1.tar.zst"
)
DD3D_WANDB_RUNTIME_SHA = f"{DD3D_WANDB_RUNTIME}.sha256"


def dd3d_wandb_test_outputs(model):
    run_dir = f"{DD3D_WANDB_TEST_ROOT}/{model}/fold_0"
    return [
        f"{run_dir}/training_complete.json",
        f"{run_dir}/wandb_run.json",
        f"{run_dir}/training.time.txt",
    ]


rule test_classifier_training_wandb_fold0_v1:
    input:
        dataset=DD3D_VALIDATION,
        metadata=lambda wc: (
            DD3D_NATIVE_SCIFI_META_COMPLETE
            if DD3D_WANDB_TEST_FAMILIES[wc.model]
            in {"native_scifi_gravnet", "two_view_sparse_resnet"}
            else DD3D_SCIFI_META_V2_VALIDATION
        ),
        config=lambda wc: DD3D_WANDB_TEST_CONFIGS[wc.model],
        wandb_runtime=DD3D_WANDB_RUNTIME,
        wandb_runtime_sha=DD3D_WANDB_RUNTIME_SHA,
        activate=f"{PERSONAL_WORK_SPACE}/containers/lcg_overlay/activate.sh",
        stage_overlay=f"{PERSONAL_WORK_SPACE}/containers/lcg_overlay/stage_archive.sh",
        stage_python=(
            f"{PERSONAL_WORK_SPACE}/containers/lcg_overlay/"
            "stage_python_runtime.sh"
        ),
        script=f"{PERSONAL_WORK_SPACE}/train_3d/train_classifier.py",
    output:
        training=(
            DD3D_WANDB_TEST_ROOT + "/{model}/fold_0/training_complete.json"
        ),
        wandb=DD3D_WANDB_TEST_ROOT + "/{model}/fold_0/wandb_run.json",
        timing=DD3D_WANDB_TEST_ROOT + "/{model}/fold_0/training.time.txt",
    wildcard_constraints:
        model="|".join(DD3D_WANDB_TEST_CONFIGS)
    threads: 4
    resources:
        runtime=6 * 60 * 60,
        mem_mb=16000,
        disk_mb=12000,
        nvidia_gpu=1,
        minimum_gpu_capability="8.0"
    params:
        output_root=DD3D_WANDB_TEST_ROOT,
        run_dir=lambda wc: f"{DD3D_WANDB_TEST_ROOT}/{wc.model}/fold_0",
        wandb_dir=lambda wc: f"{DD3D_WANDB_TEST_ROOT}/{wc.model}/fold_0/wandb",
    shell:
        r"""
        set -euo pipefail
        scratch_root="${{_CONDOR_SCRATCH_DIR:-}}"
        if [[ -z "$scratch_root" ]]; then
          scratch_root=$(mktemp -d -p /tmp snd-ml-wandb-test.XXXXXX)
        fi
        overlay_root=$({input.stage_overlay} \
          {DD3D_LCG_OVERLAY_URI} {DD3D_LCG_OVERLAY_SHA256} \
          "$scratch_root/lcg-overlay")
        wandb_sha=$(awk '{{print $1}}' {input.wandb_runtime_sha})
        wandb_root=$({input.stage_python} {input.wandb_runtime} "$wandb_sha" \
          "$scratch_root/wandb-runtime")
        set +u
        source {input.activate} "$overlay_root"
        set -u
        export PYTHONPATH="$wandb_root/python:{PERSONAL_WORK_SPACE}/train_3d/src:$PYTHONPATH"
        export SND_ML_OVERLAY_SHA256={DD3D_LCG_OVERLAY_SHA256}
        export SND_ML_SOURCE_REVISION=live-worktree
        export SND_ML_METADATA_CACHE="$scratch_root/metadata-cache"
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export WANDB_MODE=online
        mkdir -p "$SND_ML_METADATA_CACHE" {params.run_dir} {params.wandb_dir}
        python3 -c 'import wandb; assert wandb.__version__ == "0.19.8"'
        resume_arg=()
        if [[ -s {params.run_dir}/checkpoints/last.ckpt && \
              -s {params.run_dir}/wandb_run_id.txt ]]; then
          resume_arg=(--resume)
        elif [[ -e {params.run_dir}/resolved_config.yaml ]]; then
          echo "Incomplete run has no resumable checkpoint: {params.run_dir}" >&2
          echo "Move that directory aside before starting a fresh test." >&2
          exit 1
        fi
        /usr/bin/time -v -o {output.timing} \
          python3 {input.script} --config {input.config} --cv-fold 0 \
            --output-dir {params.output_root} --max-epochs 1 --epoch-size 2000 \
            --limit-val-batches 50 --num-workers 0 --reader-backend pyroot \
            --wandb-mode online --wandb-group {DD3D_WANDB_TEST_GROUP} \
            --wandb-job-type benchmark --wandb-save-dir {params.wandb_dir} \
            --wandb-log-model false "${{resume_arg[@]}}"
        test -s {output.training}
        test -s {output.wandb}
        python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "finished" and p["url"].startswith("http")' {output.wandb}
        """


rule test_clustered_gravnet_training_wandb_fold0_v1:
    input:
        dd3d_wandb_test_outputs("clustered_pyg_gravnet_v1")


rule test_three_model_training_wandb_fold0_v1:
    input:
        [
            output
            for model in DD3D_WANDB_TEST_CONFIGS
            for output in dd3d_wandb_test_outputs(model)
        ]


DD3D_DEPLOY_CONFIG = (
    f"{WORKFLOW_CONFIG_REPO}/snakemake/metadata/configs/training/"
    "data_driven_3d_classifiers_v1.yaml"
)
with open(DD3D_DEPLOY_CONFIG, "r", encoding="utf-8") as stream:
    DD3D_DEPLOY = yaml.safe_load(stream)
DD3D_TRAIN_OUTPUT = DD3D_DEPLOY["output_root"]
DD3D_CONTAINER = os.environ.get(
    "SND_ML_CONTAINER_IMAGE", DD3D_DEPLOY["container_image"]
)
DD3D_CONTAINER_SHA = os.environ.get(
    "SND_ML_CONTAINER_SHA_FILE", DD3D_DEPLOY["container_sha256"]
)
DD3D_MODEL_CONFIGS = {
    yaml.safe_load(open(path, "r", encoding="utf-8"))["model_version"]: path
    for path in DD3D_DEPLOY["models"].values()
}
DD3D_MODEL_FAMILIES = {
    model: yaml.safe_load(open(path, "r", encoding="utf-8"))["model"]["family"]
    for model, path in DD3D_MODEL_CONFIGS.items()
}
DD3D_TRAIN_RES = DD3D_DEPLOY["resources"]["train"]
DD3D_EVAL_RES = DD3D_DEPLOY["resources"]["evaluate"]
DD3D_MERGE_RES = DD3D_DEPLOY["resources"]["merge"]


def dd3d_training_manifest(model, fold):
    return (
        f"{DD3D_TRAIN_OUTPUT}/{model}/fold_{fold}/"
        "training_complete.json"
    )


rule train_classifier_fold:
    input:
        dataset=DD3D_VALIDATION,
        metadata=lambda wc: (
            DD3D_NATIVE_SCIFI_META_COMPLETE
            if DD3D_MODEL_FAMILIES[wc.model]
            in {"native_scifi_gravnet", "two_view_sparse_resnet"}
            else DD3D_SCIFI_META_V2_VALIDATION
        ),
        config=lambda wc: DD3D_MODEL_CONFIGS[wc.model],
        image=DD3D_CONTAINER,
        image_sha=DD3D_CONTAINER_SHA,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/train_classifier.py",
    output:
        DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/training_complete.json"
    wildcard_constraints:
        model="|".join(DD3D_MODEL_CONFIGS),
        fold="[0-4]"
    threads: DD3D_TRAIN_RES["cpus"]
    resources:
        runtime=DD3D_TRAIN_RES["runtime_seconds"],
        mem_mb=DD3D_TRAIN_RES["memory_mb"],
        disk_mb=DD3D_TRAIN_RES["disk_mb"],
        nvidia_gpu=1,
        minimum_gpu_capability=str(DD3D_TRAIN_RES["minimum_gpu_capability"]),
        container_image=DD3D_CONTAINER
    params:
        run_dir=lambda wc: f"{DD3D_TRAIN_OUTPUT}/{wc.model}/fold_{wc.fold}"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH={PERSONAL_WORK_SPACE}/train_3d/src
        export SND_ML_CONTAINER_IMAGE={input.image}
        export SND_ML_CONTAINER_SHA256=$(awk '{{print $1}}' {input.image_sha})
        export PYTHONNOUSERSITE=1
        resume_arg=""
        if [[ -f {params.run_dir}/checkpoints/last.ckpt ]]; then
          resume_arg="--resume"
        fi
        /usr/bin/python3 {input.script} --config {input.config} --cv-fold {wildcards.fold} \
          --output-dir {DD3D_TRAIN_OUTPUT} --reader-backend uproot $resume_arg
        """


rule evaluate_classifier_shard:
    input:
        training=lambda wc: dd3d_training_manifest(wc.model, wc.fold),
        image=DD3D_CONTAINER,
        image_sha=DD3D_CONTAINER_SHA,
        script=f"{PERSONAL_WORK_SPACE}/train_3d/evaluate_classifier_shard.py",
    output:
        summary=(
            DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/"
            "shard_{shard}.json"
        ),
        predictions=(
            DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/"
            "shard_{shard}.predictions.jsonl"
        ),
    threads: DD3D_EVAL_RES["cpus"]
    resources:
        runtime=DD3D_EVAL_RES["runtime_seconds"],
        mem_mb=DD3D_EVAL_RES["memory_mb"],
        disk_mb=DD3D_EVAL_RES["disk_mb"],
        nvidia_gpu=0,
        container_image=DD3D_CONTAINER
    params:
        shard_count=int(DD3D_DEPLOY["evaluation_shards"])
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH={PERSONAL_WORK_SPACE}/train_3d/src
        PYTHONNOUSERSITE=1 /usr/bin/python3 {input.script} --training-manifest {input.training} \
          --shard-index {wildcards.shard} --num-shards {params.shard_count} \
          --num-threads {threads} --device cpu --output {output.summary} \
          --predictions {output.predictions}
        """


rule merge_classifier_evaluation:
    input:
        summaries=lambda wc: expand(
            DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/shard_{shard}.json",
            model=wc.model, fold=wc.fold,
            shard=range(int(DD3D_DEPLOY["evaluation_shards"])),
        ),
        predictions=lambda wc: expand(
            DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/"
            "shard_{shard}.predictions.jsonl",
            model=wc.model, fold=wc.fold,
            shard=range(int(DD3D_DEPLOY["evaluation_shards"])),
        ),
        script=f"{PERSONAL_WORK_SPACE}/train_3d/merge_classifier_evaluation.py",
    output:
        summary=DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/final.json",
        predictions=(
            DD3D_TRAIN_OUTPUT + "/{model}/fold_{fold}/evaluation/"
            "final.predictions.jsonl"
        ),
    threads: DD3D_MERGE_RES["cpus"]
    resources:
        runtime=DD3D_MERGE_RES["runtime_seconds"],
        mem_mb=DD3D_MERGE_RES["memory_mb"],
        disk_mb=DD3D_MERGE_RES["disk_mb"],
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONPATH={PERSONAL_WORK_SPACE}/train_3d/src
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 {input.script} --shards {input.summaries} \
          --output {output.summary} --predictions-output {output.predictions}
        """


rule data_driven_3d_classifiers_v1:
    input:
        [
            DD3D_TRAIN_OUTPUT + f"/{model}/fold_{fold}/"
            "evaluation/final.json"
            for model in DD3D_MODEL_CONFIGS
            for fold in DD3D_DEPLOY["folds"]
        ]
