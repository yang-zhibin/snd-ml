from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import torch


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_3d.data.datamodule import Hits3DDataModule
from train_3d.lightning_module import Hits3DRegressor
from train_3d.utils.config import find_config_path, load_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split_paths(train_data_dir, split_version):
    train_data_dir = Path(train_data_dir)
    train_path = train_data_dir / f"{split_version}_train.npz"
    val_path = train_data_dir / f"{split_version}_val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation dataset: {val_path}")

    return train_path, val_path


def build_wandb_logger(wandb_config, split_version, model_version):
    init_kwargs = {
        "project": wandb_config["project"],
        "save_dir": wandb_config.get("save_dir", "wandb_logs"),
        "offline": wandb_config.get("offline", False),
        "name": f"{model_version}_split_{split_version}",
    }

    entity = wandb_config.get("entity")
    if entity is not None:
        entity = str(entity).strip()
        if entity and not entity.isdigit():
            init_kwargs["entity"] = entity

    return WandbLogger(**init_kwargs)


def resolve_trainer_hardware(config):
    trainer_cfg = dict(config["trainer"])
    accelerator = trainer_cfg.get("accelerator", "auto")
    devices = trainer_cfg.get("devices", 1)
    backend = str(config.get("model", {}).get("backend", "")).lower()

    if backend == "minkowski" and accelerator == "auto":
        trainer_cfg["accelerator"] = "cpu"
        trainer_cfg["devices"] = 1
        print(
            "MinkowskiEngine backend detected with trainer.accelerator='auto'. "
            "Defaulting to CPU to avoid CUDA mismatch with CPU-only MinkowskiEngine builds."
        )

    return trainer_cfg["accelerator"], trainer_cfg["devices"]


def main(args):
    config_path = find_config_path(THIS_DIR / "configs", args.model_version)
    config = load_config(config_path)

    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    train_path, val_path = resolve_split_paths(args.train_data_dir, args.split_version)

    datamodule = Hits3DDataModule(
        train_path=train_path,
        val_path=val_path,
        data_config=config["data"],
        voxel_config=config["voxel"],
    )

    model = Hits3DRegressor(
        model_config=config["model"],
        optim_config={
            **config["optim"],
            "target_transform": config["data"].get("target_transform", "none"),
        },
        eval_config=config.get("evaluation", {}),
    )

    checkpoint_config = config.get("checkpoint", {})
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get("monitor", "val_loss"),
        mode=checkpoint_config.get("mode", "min"),
        save_top_k=checkpoint_config.get("save_top_k", 1),
        filename=f"{args.model_version}-{args.split_version}" + "-{epoch:02d}-{val_mae:.4f}",
    )

    early_stopping_config = config.get("early_stopping", {})
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_config.get("monitor", checkpoint_config.get("monitor", "val_mae")),
        mode=early_stopping_config.get("mode", checkpoint_config.get("mode", "min")),
        patience=early_stopping_config.get("patience", 5),
        min_delta=early_stopping_config.get("min_delta", 0.0),
        verbose=True,
    )

    logger = build_wandb_logger(config["wandb"], args.split_version, args.model_version)
    logger.log_hyperparams(
        {
            "split_version": args.split_version,
            "model_version": args.model_version,
            "config_path": str(config_path),
            **config,
        }
    )

    accelerator, devices = resolve_trainer_hardware(config)

    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=config["trainer"].get("log_every_n_steps", 10),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True,
        default_root_dir=args.output_dir,
    )
    try:
        trainer.fit(model=model, datamodule=datamodule)
    except AssertionError as exc:
        if "compiled with CPU_ONLY flag" in str(exc):
            raise RuntimeError(
                "This MinkowskiEngine installation is CPU-only, but training tried to use CUDA. "
                "Set `trainer.accelerator: cpu` in the config, or reinstall MinkowskiEngine with CUDA support."
            ) from exc
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-data-dir", required=True, help="Directory containing <split>_train.npz and <split>_val.npz")
    parser.add_argument("--split-version", required=True, help="Split tag used in merged dataset filenames")
    parser.add_argument("--model-version", required=True, help="Model version used to select a config file by name")
    parser.add_argument("--output-dir", default="train_outputs", help="Directory for checkpoints and Lightning outputs")

    main(parser.parse_args())
