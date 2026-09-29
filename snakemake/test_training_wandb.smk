"""Standalone LCG108/H100/W&B benchmarks for clustered GravNet.

The smoke and timing targets deliberately use separate output roots. Existing
validated metadata are immutable inputs, so these targets cannot rebuild the
full dataset DAG or wait for the native-hit dataset.
"""


REPO = "/afs/cern.ch/work/z/zhibin/snd-ml"
MODEL = "clustered_pyg_gravnet_v1"
MODEL_CONFIG = f"{REPO}/train_3d/configs/classifier_clustered_pyg_gravnet_v1.yaml"
OUTPUT_ROOT = f"{REPO}/train_3d/test_outputs/clustered_wandb_lcg108_v1"
DATASET_VALIDATION = (
    f"{REPO}/snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/"
    "validation_report.json"
)
LEGACY_METADATA = (
    f"{REPO}/train_3d/metadata/scifi_xyz_qdc_dettype_metadata_v2/validation.json"
)

OVERLAY_URI = (
    "root://eosuser.cern.ch//eos/user/z/zhibin/snd-ml/overlays/"
    "snd-ml-lcg108cuda-overlay/v1/snd-ml-lcg108cuda-overlay-v1.tar.zst"
)
OVERLAY_SHA256 = (
    "947f96166b6e8f1e23660bcf9d2f562af961f3d099eec222eb0ac766b09e494b"
)
WANDB_RUNTIME_URI = (
    "root://eosuser.cern.ch//eos/user/z/zhibin/snd-ml/overlays/"
    "snd-ml-lcg108cuda-overlay-wandb-runtime/v1/"
    "snd-ml-lcg108cuda-overlay-wandb-runtime-v1.tar.zst"
)
WANDB_RUNTIME_SHA256 = (
    "78abfdb0e2f46d28e13cadfdda78bbfc3f2752f0443e977111d7b45046db6813"
)

BENCHMARKS = {
    # Two validation batches exercise validation and checkpoint creation. The
    # resulting metrics are not a physics-quality validation measurement.
    "smoke": {
        "max_epochs": 1, "epoch_size": 200, "val_batches": 2,
        "validation_sampling": "natural", "validation_size": 0,
        "runtime": 2 * 60 * 60,
        "num_workers": 0, "max_events_per_batch": 16,
        "max_nodes_per_batch": 20000, "max_nodes_per_forward": 8192,
        "activation_checkpointing": "true", "precision": "32-true",
        "matmul_precision": "highest",
    },
    # This matches the earlier A100 sample size for a useful comparison.
    "timing": {
        "max_epochs": 1, "epoch_size": 2000, "val_batches": 50,
        "validation_sampling": "natural", "validation_size": 0,
        "runtime": 6 * 60 * 60,
        "num_workers": 0, "max_events_per_batch": 16,
        "max_nodes_per_batch": 20000, "max_nodes_per_forward": 8192,
        "activation_checkpointing": "true", "precision": "32-true",
        "matmul_precision": "highest",
    },
    # Highest-priority H100 utilization changes. This keeps the same sampled
    # event count and validation limit as timing, making the result comparable.
    "optimized": {
        "max_epochs": 1, "epoch_size": 2000, "val_batches": 50,
        "validation_sampling": "natural", "validation_size": 0,
        "runtime": 4 * 60 * 60,
        "num_workers": 4, "max_events_per_batch": 32,
        "max_nodes_per_batch": 100000, "max_nodes_per_forward": 65536,
        "activation_checkpointing": "false", "precision": "bf16-mixed",
        "matmul_precision": "highest",
    },
    # FP32 is required by torch-cluster kNN. The "high" matmul policy enables
    # Tensor Core acceleration internally without changing tensor dtypes.
    "optimized_tf32": {
        "max_epochs": 1, "epoch_size": 2000, "val_batches": 50,
        "validation_sampling": "natural", "validation_size": 0,
        "runtime": 4 * 60 * 60,
        "num_workers": 4, "max_events_per_batch": 32,
        "max_nodes_per_batch": 100000, "max_nodes_per_forward": 65536,
        "activation_checkpointing": "false", "precision": "32-true",
        "matmul_precision": "high",
    },
    # Five epochs test optimization stability. The complete fixed validation
    # set contains 200 usable held-out events from each of the seven classes;
    # it is selected before batching and therefore does not depend on batch
    # size or node-budget settings.
    "stability_tf32": {
        "max_epochs": 5, "epoch_size": 2000, "val_batches": 0,
        "validation_sampling": "stratified_fixed", "validation_size": 1400,
        "runtime": 3 * 60 * 60,
        "num_workers": 4, "max_events_per_batch": 32,
        "max_nodes_per_batch": 100000, "max_nodes_per_forward": 65536,
        "activation_checkpointing": "false", "precision": "32-true",
        "matmul_precision": "high",
    },
}


def benchmark_outputs(benchmark):
    run_dir = f"{OUTPUT_ROOT}/{benchmark}/{MODEL}/fold_0"
    return [
        f"{run_dir}/training_complete.json",
        f"{run_dir}/wandb_run.json",
        f"{run_dir}/training.time.txt",
    ]


rule train_clustered_fold0_wandb_lcg108:
    input:
        dataset=DATASET_VALIDATION,
        metadata=LEGACY_METADATA,
        config=MODEL_CONFIG,
        activate=f"{REPO}/containers/lcg_overlay/activate.sh",
        stage_overlay=f"{REPO}/containers/lcg_overlay/stage_archive.sh",
        stage_python=f"{REPO}/containers/lcg_overlay/stage_python_runtime.sh",
        script=f"{REPO}/train_3d/train_classifier.py",
    output:
        training=OUTPUT_ROOT + "/{benchmark}/" + MODEL + "/fold_0/training_complete.json",
        wandb=OUTPUT_ROOT + "/{benchmark}/" + MODEL + "/fold_0/wandb_run.json",
        timing=OUTPUT_ROOT + "/{benchmark}/" + MODEL + "/fold_0/training.time.txt",
    wildcard_constraints:
        benchmark="smoke|timing|optimized|optimized_tf32|stability_tf32"
    threads: 6
    resources:
        runtime=lambda wc: BENCHMARKS[wc.benchmark]["runtime"],
        mem_mb=16000,
        disk_mb=12000,
        nvidia_gpu=1,
        minimum_gpu_capability="9.0",
        gpu_device_name="NVIDIA H100 NVL"
    params:
        output_root=lambda wc: f"{OUTPUT_ROOT}/{wc.benchmark}",
        run_dir=lambda wc: f"{OUTPUT_ROOT}/{wc.benchmark}/{MODEL}/fold_0",
        wandb_dir=lambda wc: f"{OUTPUT_ROOT}/{wc.benchmark}/{MODEL}/fold_0/wandb",
        max_epochs=lambda wc: BENCHMARKS[wc.benchmark]["max_epochs"],
        epoch_size=lambda wc: BENCHMARKS[wc.benchmark]["epoch_size"],
        val_batches=lambda wc: BENCHMARKS[wc.benchmark]["val_batches"],
        validation_sampling=lambda wc: BENCHMARKS[wc.benchmark]["validation_sampling"],
        validation_size=lambda wc: BENCHMARKS[wc.benchmark]["validation_size"],
        num_workers=lambda wc: BENCHMARKS[wc.benchmark]["num_workers"],
        max_events=lambda wc: BENCHMARKS[wc.benchmark]["max_events_per_batch"],
        max_nodes=lambda wc: BENCHMARKS[wc.benchmark]["max_nodes_per_batch"],
        max_forward=lambda wc: BENCHMARKS[wc.benchmark]["max_nodes_per_forward"],
        checkpointing=lambda wc: BENCHMARKS[wc.benchmark]["activation_checkpointing"],
        precision=lambda wc: BENCHMARKS[wc.benchmark]["precision"],
        matmul_precision=lambda wc: BENCHMARKS[wc.benchmark]["matmul_precision"],
        wandb_group=lambda wc: f"clustered_fold0_lcg108_{wc.benchmark}_v1",
        job_type=lambda wc: f"benchmark_{wc.benchmark}",
    shell:
        r"""
        set -euo pipefail
        scratch_root="${{_CONDOR_SCRATCH_DIR:-}}"
        if [[ -z "$scratch_root" ]]; then
          scratch_root=$(mktemp -d -p /tmp snd-ml-lcg108-wandb.XXXXXX)
        fi
        overlay_root=$({input.stage_overlay} {OVERLAY_URI} {OVERLAY_SHA256} \
          "$scratch_root/lcg-overlay")
        wandb_root=$({input.stage_python} {WANDB_RUNTIME_URI} \
          {WANDB_RUNTIME_SHA256} "$scratch_root/wandb-runtime")
        set +u
        source {input.activate} "$overlay_root"
        set -u
        export PYTHONPATH="$wandb_root/python:{REPO}/train_3d/src:$PYTHONPATH"
        export SND_ML_OVERLAY_SHA256={OVERLAY_SHA256}
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
        validation_arg=()
        if [[ "{params.validation_sampling}" == "stratified_fixed" ]]; then
          validation_arg=(--validation-sampling stratified_fixed \
            --validation-size {params.validation_size} --validation-seed 42)
        else
          validation_arg=(--limit-val-batches {params.val_batches})
        fi
        /usr/bin/time -v -o {output.timing} \
          python3 {input.script} --config {input.config} --cv-fold 0 \
            --output-dir {params.output_root} --max-epochs {params.max_epochs} \
            --epoch-size {params.epoch_size} \
            --num-sanity-val-steps 0 --num-workers {params.num_workers} \
            --max-events-per-batch {params.max_events} \
            --max-nodes-per-batch {params.max_nodes} \
            --max-nodes-per-forward {params.max_forward} \
            --activation-checkpointing {params.checkpointing} \
            --precision {params.precision} \
            --float32-matmul-precision {params.matmul_precision} \
            --reader-backend pyroot \
            --wandb-mode online --wandb-group {params.wandb_group} \
            --wandb-job-type {params.job_type} \
            --wandb-save-dir {params.wandb_dir} --wandb-log-model false \
            "${{validation_arg[@]}}" "${{resume_arg[@]}}"
        test -s {output.training}
        test -s {output.wandb}
        python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "finished" and p["url"].startswith("http")' {output.wandb}
        """


rule clustered_wandb_smoke_lcg108:
    input:
        benchmark_outputs("smoke")


rule clustered_wandb_timing_lcg108:
    input:
        benchmark_outputs("timing")


rule clustered_wandb_optimized_lcg108:
    input:
        benchmark_outputs("optimized")


rule clustered_wandb_optimized_tf32_lcg108:
    input:
        benchmark_outputs("optimized_tf32")


rule clustered_wandb_stability_tf32_lcg108:
    input:
        benchmark_outputs("stability_tf32")


rule clustered_wandb_lcg108:
    input:
        benchmark_outputs("smoke") + benchmark_outputs("timing") \
        + benchmark_outputs("optimized_tf32")
