# `train_3d`

This folder is organized to keep data preparation, model code, training logic,
and experiment configs separated.

## Layout

```text
train_3d/
  train.py
  prepare_train_dataset.py
  README.md
  configs/
    model_baseline.yaml
  src/
    train_3d/
      __init__.py
      data/
        __init__.py
      models/
        __init__.py
      utils/
        __init__.py
```

## Purpose

- `prepare_train_dataset.py`
  Keeps the current merged train/validation dataset preparation script.
- `configs/`
  Stores experiment configuration files. The training entrypoint can select a
  config by matching the `model_version` in the filename.
- `src/train_3d/data/`
  Dataset loaders, voxelization helpers, and Lightning data modules.
- `src/train_3d/models/`
  Neural network definitions.
- `src/train_3d/utils/`
  Shared helpers such as config loading or path utilities.

## Training Entry Point

Use `train.py` to launch training. It expects merged dataset files named:

```text
<train_data_dir>/<split_version>_train.npz
<train_data_dir>/<split_version>_val.npz
```

Example:

```bash
python train.py \
  --train-data-dir /path/to/merged_npz \
  --split-version 0 \
  --model-version baseline
```

`model_version` is resolved by filename from `configs/`, so `baseline` matches
`configs/model_baseline.yaml`.

This training scaffold is configured for regression. Each event in the merged
`.npz` files is expected to contain `event["energy"]`, which is used as the
target value.

The current model path uses sparse voxel inputs with a MinkowskiEngine backend.
That means training expects:
- sparse voxel coordinates built from `all_3dHits`
- per-voxel feature vectors
- a Python environment with `MinkowskiEngine` installed

## Suggested next files

When we build the training pipeline, the next files should go here:

```text
train_3d/
  train.py
  configs/
    model_baseline.yaml
  src/train_3d/
    data/
      dataset.py
      datamodule.py
      voxelize.py
    models/
      simple_cnn.py
    utils/
      config.py
    lightning_module.py
```
