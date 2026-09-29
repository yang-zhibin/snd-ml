#!/usr/bin/env python3
"""Train one fold of the shared seven-class GravNet/sparse-ResNet pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import netrc
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "src"))

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # LCG currently provides the legacy import family.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

from train_3d.classification_module import ClassificationModule
from train_3d.data.classification_datamodule import (
    AdvanceSampledEpoch,
    ClassificationDataModule,
    ThroughputMonitor,
)
from train_3d.data.classification_labels import validate_scifi_classification_config


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_files(config, fold):
    """Resolve either compact-v2 or legacy preprocessing artifacts."""
    data = config["data"]
    metadata_root = data.get("scifi_metadata_root")
    if metadata_root:
        root = Path(str(metadata_root))
        native = config.get("model", {}).get("family") in {
            "native_scifi_gravnet", "two_view_sparse_resnet"
        }
        return {
            "normalization": root / f"fold_{fold}" / (
                "native_normalization.json" if native else "normalization.json"
            ),
            "representation_summary": root / f"fold_{fold}" / (
                "native_summary.json" if native else "summary.json"
            ),
            "event_size_index": root / "index.json",
            "metadata_complete": root / "complete.json",
            "metadata_validation": root / "validation.json",
        }
    return {
        "normalization": Path(str(data["normalization_path"]).format(cv_fold=fold)),
        "representation_summary": Path(
            str(data["representation_summary"]).format(cv_fold=fold)
        ),
        "event_size_manifest": Path(
            str(data["event_size_manifest"]).format(cv_fold=fold)
        ),
    }


def write_local_manifest(output_dir, config, best_checkpoint, last_checkpoint):
    """Write small, local lineage metadata without uploading dataset files."""
    files = {
        "resolved_config": output_dir / "resolved_config.yaml",
        "fold_views": Path(config["data"]["fold_views"]),
        "validation_report": Path(config["data"]["validation_report"]),
        **preprocessing_files(config, int(config["data"]["cv_fold"])),
    }
    payload = {
        "schema_version": 1,
        "model_version": config["model_version"],
        "cv_fold": int(config["data"]["cv_fold"]),
        "files": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                  for name, path in files.items()},
        "best_checkpoint": str(Path(best_checkpoint).resolve()),
        "best_checkpoint_sha256": sha256(best_checkpoint),
        "last_checkpoint": str(Path(last_checkpoint).resolve()),
        "last_checkpoint_sha256": sha256(last_checkpoint),
        "container": config["provenance"].get("container", {}),
        "lcg_overlay": config["provenance"].get("lcg_overlay", {}),
    }
    (output_dir / "training_complete.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git_state():
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except FileNotFoundError:
            return ""
    return {
        "commit": run("rev-parse", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
    }


def package_versions():
    names = [
        "torch", "torch-geometric", "torch-cluster", "pyg-lib", "MinkowskiEngine",
        "lightning", "wandb", "numpy", "pandas",
    ]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result.update({
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    try:
        import ROOT
        result["ROOT"] = str(ROOT.gROOT.GetVersion())
    except ImportError:
        result["ROOT"] = None
    return result


def validate_dataset(config, fold):
    report_path = Path(config["data"]["validation_report"])
    if not report_path.exists():
        raise FileNotFoundError(f"Dataset validation report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("valid") is not True:
        raise ValueError(f"Dataset validation did not pass: {report_path}")
    if report.get("dataset_version") != config["dataset_version"]:
        raise ValueError("Dataset validation version differs from training config")
    reported_classes = set(report.get("classes_present", config["class_names"]))
    expected_classes = set(config["class_names"])
    if reported_classes != expected_classes:
        raise ValueError(
            "Validated dataset classes differ from the seven-class contract: "
            f"reported={sorted(reported_classes)} expected={sorted(expected_classes)}"
        )
    excluded_present = reported_classes & set(config["excluded_classes"])
    if excluded_present:
        raise ValueError(f"Excluded classes are present: {sorted(excluded_present)}")
    n_folds = int(config["data"].get("n_folds", report.get("folds", 5)))
    if fold not in range(n_folds):
        raise ValueError(f"cv_fold must be in [0,{n_folds - 1}]")
    return report_path, report


def resolve_config(config, config_path, fold):
    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    resolved["data"]["cv_fold"] = int(fold)
    metadata_files = preprocessing_files(config, fold)
    for required in metadata_files.values():
        if not Path(required).exists():
            raise FileNotFoundError(
                f"Required SciFi preprocessing metadata is missing: {required}"
            )
    if "metadata_complete" in metadata_files:
        metadata_complete = json.loads(
            metadata_files["metadata_complete"].read_text(encoding="utf-8")
        )
        metadata_validation = json.loads(
            metadata_files["metadata_validation"].read_text(encoding="utf-8")
        )
        if metadata_complete.get("dataset_version") != config["dataset_version"]:
            raise ValueError("SciFi metadata dataset version differs from training config")
        metadata_version = config["representation"].get(
            "metadata_version", config["representation"]["version"]
        )
        if metadata_complete.get("representation_version") != metadata_version:
            raise ValueError("SciFi metadata representation version differs from config")
        if metadata_validation.get("valid") is not True:
            raise ValueError("SciFi metadata validation did not pass")
    resolved["provenance"] = {
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256(config_path),
        "fold_views_sha256": sha256(config["data"]["fold_views"]),
        "validation_report_sha256": sha256(config["data"]["validation_report"]),
        "normalization_sha256": sha256(metadata_files["normalization"]),
        "representation_summary_sha256": sha256(
            metadata_files["representation_summary"]
        ),
        "event_size_manifest_sha256": sha256(
            metadata_files["event_size_manifest"]
            if "event_size_manifest" in metadata_files
            else metadata_files["metadata_complete"]
        ),
        "metadata_index_sha256": (
            sha256(metadata_files["event_size_index"])
            if "event_size_index" in metadata_files else None
        ),
        "git": git_state(),
        "environment": package_versions(),
        "hostname": platform.node(),
        "resources": {
            "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "container": {
            "image": os.environ.get("SND_ML_CONTAINER_IMAGE"),
            "sha256": os.environ.get("SND_ML_CONTAINER_SHA256"),
        },
        "lcg_overlay": {
            "root": os.environ.get("SND_ML_OVERLAY_ROOT"),
            "sha256": os.environ.get("SND_ML_OVERLAY_SHA256"),
            "lcg_release": os.environ.get("SND_ML_LCG_RELEASE"),
            "lcg_platform": os.environ.get("SND_ML_LCG_PLATFORM"),
            "source_revision": os.environ.get("SND_ML_SOURCE_REVISION"),
        },
    }
    return resolved


def validate_resume_config(existing: dict, current: dict) -> None:
    checks = {
        "model_version": (existing.get("model_version"), current.get("model_version")),
        "dataset_version": (
            existing.get("dataset_version"), current.get("dataset_version")
        ),
        "class_names": (existing.get("class_names"), current.get("class_names")),
        "excluded_classes": (
            existing.get("excluded_classes"), current.get("excluded_classes")
        ),
        "model": (existing.get("model"), current.get("model")),
        "representation": (
            existing.get("representation"), current.get("representation")
        ),
        "objective": (existing.get("objective"), current.get("objective")),
        "fold": (existing.get("data", {}).get("cv_fold"), current.get("data", {}).get("cv_fold")),
        "seed": (existing.get("runtime", {}).get("seed"), current.get("runtime", {}).get("seed")),
        "validation_sampling": (
            existing.get("data", {}).get("validation_sampling"),
            current.get("data", {}).get("validation_sampling"),
        ),
        "validation_size": (
            existing.get("data", {}).get("validation_size"),
            current.get("data", {}).get("validation_size"),
        ),
        "validation_seed": (
            existing.get("data", {}).get("validation_seed"),
            current.get("data", {}).get("validation_seed"),
        ),
        "fold_views_sha256": (
            existing.get("provenance", {}).get("fold_views_sha256"),
            current.get("provenance", {}).get("fold_views_sha256"),
        ),
        "validation_report_sha256": (
            existing.get("provenance", {}).get("validation_report_sha256"),
            current.get("provenance", {}).get("validation_report_sha256"),
        ),
        "normalization_sha256": (
            existing.get("provenance", {}).get("normalization_sha256"),
            current.get("provenance", {}).get("normalization_sha256"),
        ),
        "representation_summary_sha256": (
            existing.get("provenance", {}).get("representation_summary_sha256"),
            current.get("provenance", {}).get("representation_summary_sha256"),
        ),
        "event_size_manifest_sha256": (
            existing.get("provenance", {}).get("event_size_manifest_sha256"),
            current.get("provenance", {}).get("event_size_manifest_sha256"),
        ),
        "metadata_index_sha256": (
            existing.get("provenance", {}).get("metadata_index_sha256"),
            current.get("provenance", {}).get("metadata_index_sha256"),
        ),
    }
    mismatches = {name: values for name, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Resume configuration differs from the saved run: {mismatches}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def require_wandb_credentials():
    """Fail before training when an online W&B run cannot authenticate."""
    if os.environ.get("WANDB_API_KEY", "").strip():
        return "environment"
    netrc_path = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    try:
        auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError) as exc:
        raise RuntimeError(
            "W&B online mode requires WANDB_API_KEY or a valid "
            "'machine api.wandb.ai' entry in ~/.netrc"
        ) from exc
    if not auth or not auth[2]:
        raise RuntimeError(
            "W&B online mode requires WANDB_API_KEY or a valid "
            "'machine api.wandb.ai' entry in ~/.netrc"
        )
    return "netrc"


def write_wandb_run_metadata(output_dir, run, status):
    payload = {
        "schema_version": 1,
        "status": status,
        "id": str(run.id),
        "name": str(run.name),
        "project": str(run.project),
        "entity": str(run.entity),
        "url": str(run.url),
    }
    (output_dir / "wandb_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_logger(config, fold, output_dir, resume):
    settings = config["wandb"]
    mode = str(settings.get("mode", "disabled"))
    if mode == "disabled":
        return False
    if mode not in {"offline", "online"}:
        raise ValueError("wandb.mode must be disabled, offline, or online")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            f"wandb.mode={mode!r}, but the wandb package is unavailable"
        ) from exc
    if mode == "online":
        require_wandb_credentials()
        # Do not permit a worker-level setting to silently turn an online
        # benchmark into an offline or disabled run.
        os.environ["WANDB_MODE"] = "online"
    run_id_path = output_dir / "wandb_run_id.txt"
    logger_args = {}
    if resume:
        if not run_id_path.exists():
            raise FileNotFoundError("Intentional resume requires wandb_run_id.txt")
        logger_args.update(id=run_id_path.read_text().strip(), resume="must")
    seed = int(config["runtime"]["seed"])
    logger = WandbLogger(
        project=settings["project"],
        entity=settings.get("entity") or None,
        save_dir=settings.get("save_dir", str(output_dir / "wandb")),
        offline=mode == "offline",
        name=f"{config['model_version']}__fold{fold}__seed{seed}",
        group=settings.get("group", config["model_version"]),
        job_type=settings.get("job_type", "train"),
        log_model=(settings.get("log_model", True) if mode == "online" else False),
        tags=[*settings.get("tags", []), f"fold_{fold}"],
        settings=wandb.Settings(
            mode=mode,
            init_timeout=int(settings.get("init_timeout_seconds", 60)),
        ),
        **logger_args,
    )
    # WandbLogger is the sole owner of initialization and identity.
    run = logger.experiment
    if mode == "online" and (not run.id or not str(run.url).startswith("http")):
        raise RuntimeError("W&B did not establish a valid online run")
    run_id_path.write_text(str(run.id) + "\n", encoding="utf-8")
    write_wandb_run_metadata(output_dir, run, "running")
    run.config.update(config, allow_val_change=False)
    for metric, summary in {
        "val/loss": "min", "val/macro_f1": "max",
        "val/balanced_accuracy": "max", "throughput/nodes_s": "max",
        "memory/gpu_peak_mb": "min",
        "memory/gpu_peak_reserved_mb": "min",
    }.items():
        run.define_metric(metric, summary=summary)
    return logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-fold", type=int)
    parser.add_argument("--output-dir", default="train_outputs/data_driven_7class")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--epoch-size", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument(
        "--validation-sampling", choices=("natural", "stratified_fixed")
    )
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--num-sanity-val-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-events-per-batch", type=int)
    parser.add_argument("--max-nodes-per-batch", type=int)
    parser.add_argument("--max-nodes-per-forward", type=int)
    parser.add_argument(
        "--activation-checkpointing", choices=("true", "false")
    )
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"))
    parser.add_argument(
        "--float32-matmul-precision", choices=("highest", "high", "medium")
    )
    parser.add_argument("--reader-backend", choices=("pyroot", "uproot"))
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--wandb-save-dir")
    parser.add_argument("--wandb-log-model", choices=("true", "false"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    validate_scifi_classification_config(config)
    fold = int(config["data"]["cv_fold"] if args.cv_fold is None else args.cv_fold)
    validate_dataset(config, fold)
    if args.wandb_mode:
        config["wandb"]["mode"] = args.wandb_mode
    if args.max_epochs is not None:
        config["trainer"]["max_epochs"] = int(args.max_epochs)
    if args.epoch_size is not None:
        if args.epoch_size <= 0:
            raise ValueError("--epoch-size must be positive")
        config["data"]["epoch_size"] = int(args.epoch_size)
    if args.limit_val_batches is not None:
        if args.limit_val_batches <= 0:
            raise ValueError("--limit-val-batches must be positive")
        config["trainer"]["limit_val_batches"] = int(args.limit_val_batches)
    if args.validation_sampling is not None:
        config["data"]["validation_sampling"] = args.validation_sampling
    if args.validation_size is not None:
        if args.validation_size <= 0:
            raise ValueError("--validation-size must be positive")
        config["data"]["validation_size"] = int(args.validation_size)
    if args.validation_seed is not None:
        if args.validation_seed < 0:
            raise ValueError("--validation-seed must be non-negative")
        config["data"]["validation_seed"] = int(args.validation_seed)
    if args.num_sanity_val_steps is not None:
        if args.num_sanity_val_steps < 0:
            raise ValueError("--num-sanity-val-steps must be non-negative")
        config["trainer"]["num_sanity_val_steps"] = int(
            args.num_sanity_val_steps
        )
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config["data"]["num_workers"] = int(args.num_workers)
        config["data"]["persistent_workers"] = args.num_workers > 0
    for argument, key in (
        (args.max_events_per_batch, "max_events_per_batch"),
        (args.max_nodes_per_batch, "max_nodes_per_batch"),
    ):
        if argument is not None:
            if argument <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
            config["data"][key] = int(argument)
    if args.max_nodes_per_forward is not None:
        if args.max_nodes_per_forward <= 0:
            raise ValueError("--max-nodes-per-forward must be positive")
        if config["model"].get("family") != "clustered_pyg_gravnet":
            raise ValueError(
                "--max-nodes-per-forward is only valid for clustered GravNet"
            )
        config["model"]["max_nodes_per_forward"] = int(
            args.max_nodes_per_forward
        )
    if args.activation_checkpointing is not None:
        if config["model"].get("family") != "clustered_pyg_gravnet":
            raise ValueError(
                "--activation-checkpointing is only valid for clustered GravNet"
            )
        config["model"]["activation_checkpointing"] = (
            args.activation_checkpointing == "true"
        )
    if args.precision is not None:
        config["trainer"]["precision"] = args.precision
    if args.float32_matmul_precision is not None:
        config["runtime"]["float32_matmul_precision"] = (
            args.float32_matmul_precision
        )
    if args.reader_backend:
        config["data"]["reader_backend"] = args.reader_backend
    if args.wandb_group:
        config["wandb"]["group"] = args.wandb_group
    if args.wandb_job_type:
        config["wandb"]["job_type"] = args.wandb_job_type
    if args.wandb_save_dir:
        config["wandb"]["save_dir"] = args.wandb_save_dir
    if args.wandb_log_model is not None:
        config["wandb"]["log_model"] = args.wandb_log_model == "true"
    # Validate the effective configuration as well as the input file so command
    # line benchmark overrides cannot bypass cross-field constraints.
    validate_scifi_classification_config(config)
    config = resolve_config(config, args.config, fold)
    seed = int(config["runtime"]["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision(
        str(config["runtime"].get("float32_matmul_precision", "highest"))
    )

    output_dir = Path(args.output_dir) / config["model_version"] / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    last_checkpoint = output_dir / "checkpoints" / "last.ckpt"
    run_id_path = output_dir / "wandb_run_id.txt"
    if not args.resume and (resolved_path.exists() or last_checkpoint.exists()):
        raise FileExistsError(
            f"Run state already exists in {output_dir}; use --resume for an "
            "intentional retry or choose a different output directory/seed"
        )
    if args.resume:
        required = [resolved_path, last_checkpoint]
        if config["wandb"].get("mode", "disabled") != "disabled":
            required.append(run_id_path)
        if any(not path.exists() for path in required):
            raise FileNotFoundError(
                "Resume requires resolved_config.yaml and checkpoints/last.ckpt; "
                "an enabled W&B run also requires wandb_run_id.txt"
            )
        validate_resume_config(load_config(resolved_path), config)
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint_settings = config["checkpoint"]
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor=checkpoint_settings.get("monitor", "val/macro_f1"),
        mode=checkpoint_settings.get("mode", "max"),
        save_top_k=int(checkpoint_settings.get("save_top_k", 1)),
        save_last=bool(checkpoint_settings.get("save_last", True)),
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(
        monitor=checkpoint_settings.get("monitor", "val/macro_f1"),
        mode=checkpoint_settings.get("mode", "max"),
        patience=int(config.get("early_stopping", {}).get("patience", 10)),
    )
    datamodule = ClassificationDataModule(config, fold)
    model = ClassificationModule(
        config["model"], config["optim"], config["class_names"], config["wandb"],
        objective_config=config["objective"],
    )
    logger = build_logger(config, fold, output_dir, args.resume)
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[checkpoint, early, AdvanceSampledEpoch(), ThroughputMonitor()],
        deterministic=True,
        default_root_dir=output_dir,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(last_checkpoint) if args.resume else None,
    )
    if not checkpoint.best_model_path or not last_checkpoint.exists():
        raise RuntimeError("Training finished without both best and last checkpoints")
    write_local_manifest(
        output_dir, config, checkpoint.best_model_path, last_checkpoint
    )
    if logger:
        run = logger.experiment
        run.finish(exit_code=0)
        write_wandb_run_metadata(output_dir, run, "finished")


if __name__ == "__main__":
    main()
