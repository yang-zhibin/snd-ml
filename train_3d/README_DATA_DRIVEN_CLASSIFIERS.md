# Seven-class SciFi classifiers

This implementation trains two models from the same natural five-fold dataset:

- `pyg_gravnet_v1`: continuous SciFi candidate nodes and upstream PyG
  `GravNetConv`;
- `sparse_resnet_v1`: the same candidates quantized to voxels and processed by
  a MinkowskiEngine residual network.

Both accept only `hit_x`, `hit_y`, `hit_z`, `hit_qdc`, and `hit_detType`.
The shared reader keeps `hit_detType == 1`; tau-neutrino events are rejected,
and the remaining classes are remapped to logits 0 through 6.

The accepted class/domain/density sampling, natural-validation, loss, and
checkpoint rules are frozen in
[`document/INITIAL_7CLASS_TRAINING_SAMPLING_POLICY.md`](../document/INITIAL_7CLASS_TRAINING_SAMPLING_POLICY.md).

## 1. Build fold metadata

After `data_driven_3d_cnn_dataset_v1` is valid, run:

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml/snakemake
snakemake data_driven_3d_cnn_scifi_training_metadata_v1 -j 1 \
  --rerun-triggers mtime
```

This makes, for each held-out fold, training-only normalization, an event-size
manifest for node-budget batches, and a representation summary. The scan reads
every physical event twice total, not twice per fold.

## 2. Validate the environment

The container source and smoke tests are in `containers/training/`. Build and
publish the OCI image in CI, run both CPU and assigned-GPU smoke tests, pull an
immutable digest to the `.sif` named in
`snakemake/metadata/configs/training/data_driven_3d_classifiers_v1.yaml`, and
write the `.sif` SHA-256 beside it.

Current LCG is enough for the PyG CPU unit test but not production: it lacks
MinkowskiEngine, W&B, and GPU kNN kernels. Its uproot 5.3.11 also hangs on the
tested ROOT vector encoding, so local metadata production defaults to PyROOT.
Container jobs explicitly select the pinned newer uproot reader.

## 3. Smoke a single fold

Before a long run, reduce `data.epoch_size` and `trainer.max_epochs` in a copied
config and use disabled or offline W&B:

```bash
python3 train_3d/train_classifier.py \
  --config train_3d/configs/classifier_pyg_gravnet_v1.yaml \
  --cv-fold 0 --max-epochs 1 --wandb-mode disabled
```

Then overfit 20--100 events. Do this separately for both architectures. A
failure to overfit is a correctness problem, not a hyperparameter result.

## 4. Production DAG

After the `.sif` and checksum exist:

```bash
cd snakemake
snakemake -n data_driven_3d_classifiers_v1 --rerun-triggers mtime
snakemake data_driven_3d_classifiers_v1 -j 20 \
  --profile .config/snakemake/htcondor --rerun-triggers mtime
```

One GPU job is created per model and fold. It performs sampled training and
natural held-out validation, writing best/last checkpoints and a checksummed
`training_complete.json`. CPU jobs then evaluate contiguous natural-validation
shards. The merge job sums loss and confusion counts before computing metrics;
it never averages shard F1 scores.

W&B defaults to `disabled`. Set a config to `offline` for the first tracking
test and `online` only after credentials are available in the worker user
environment. No credential or dataset file is placed in the image. The code
logs scalars and bounded diagnostic tables; local manifests retain checksums
for the full data lineage.

## 5. Remaining release gates

Source implementation is not equivalent to a validated image. Before starting
the five-fold production run, all of these must pass:

1. Docker/CI build and `pip check`;
2. PyROOT/new-uproot equality on representative production events;
3. both CPU forward/backward smoke tests in the image;
4. both GPU forward/backward smoke tests on an assigned CERN GPU;
5. one short fold-0 job per model with W&B disabled, then offline;
6. checkpoint interruption/resume using the same W&B run ID;
7. one sharded evaluation matching a single-process evaluation.
